// Checks Drama 3's text model against the official one on a dump of reference/drama3/dump.py, a layer at a time from
// the layer's own input in the dump, so that the first layer that departs is the one reported and no more than one
// layer's experts are loaded at once: 3.2 GB in F32, where the whole file holds 143 GB.
//
// The embeddings of the prompt's rows and of the frames drawn after it; then, for each layer, the prompt run at once
// and each frame's row after it alone, as the official generation runs them: the attention's or the Gated DeltaNet's
// output, the router's logits, the experts' output and the layer's output, and for a Gated DeltaNet layer its recurrent
// state and its convolution's inputs after the prompt; after the last layer, the final norm and the head's logits of
// the row before each frame. A row whose 8th and 9th experts' logits lie closer than twice the largest difference of
// its router's logits from the official ones may take the other of the two, which changes its experts' output and no
// other row's: such a row is left out of the experts' and the layer's comparisons, and counted. With experts-on-cpu
// after a GPU, each layer's routed experts are loaded into the CPU's memory and computed there, the rest on the GPU, as
// a model whose GPU does not hold them runs.
//
// usage: drama3-text-model-check <model.gguf> <dump dir> [gpu|cpu|device name] [experts-on-cpu]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "drama3/layout.h"
#include "drama3/placement.h"
#include "drama3/text-model.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "weight-types.h"

using namespace drama3;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/**
 * The rows of `official`, the official router's logits, [rows, experts] row-major, whose `used`-th and next largest
 * logits lie within twice the largest difference of `mine` from them in that row, where `mine` may choose the other.
 */
std::vector<bool> tied_rows(const std::vector<float> & mine, const std::vector<float> & official, int64_t rows, int64_t experts, int used) {
    std::vector<bool> tied((size_t) rows);
    std::vector<double> row((size_t) experts);
    for (int64_t r = 0; r < rows; r++) {
        double margin = 0;
        for (int64_t e = 0; e < experts; e++) {
            margin = std::max(margin, 2.0 * std::fabs((double) mine[(size_t) (r * experts + e)] - official[(size_t) (r * experts + e)]));
        }
        row.assign(official.begin() + r * experts, official.begin() + (r + 1) * experts);
        std::nth_element(row.begin(), row.begin() + used, row.end(), std::greater<double>());
        const double next = row[(size_t) used];
        const double last = *std::min_element(row.begin(), row.begin() + used);
        tied[(size_t) r] = last - next <= margin;
    }
    return tied;
}

/** The rows of `a`, rows of `width` values, that `drop` does not mark. */
std::vector<float> without(const std::vector<float> & a, int64_t width, const std::vector<bool> & drop) {
    std::vector<float> out;
    for (size_t r = 0; r < drop.size(); r++) {
        if (!drop[r]) out.insert(out.end(), a.begin() + (ptrdiff_t) (r * width), a.begin() + (ptrdiff_t) ((r + 1) * width));
    }
    return out;
}

/** Rows [first, first + n) of a row-major array of rows of `width` values. */
template <typename T>
std::vector<T> rows_of(const std::vector<T> & a, int64_t width, int64_t first, int64_t n) {
    return std::vector<T>(a.begin() + first * width, a.begin() + (first + n) * width);
}

/** The least SNR of a kind of stage over the layers or the rows, as keep_worst() keeps it, and where it was. */
struct Worst {
    Diff diff;
    int layer = -1;
    void keep(const Diff & d, int l) {
        if (keep_worst(diff, d)) layer = l;
    }
    void print(const std::string & what) const { print_diff(what + " (" + std::to_string(layer) + ")", diff); }
};

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name] [experts-on-cpu]\n", args[0].c_str());
        return 2;
    }
    const std::string path = args[1], dir = args[2];
    if (!takes_weights(path, model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    const bool split = args.size() > 4 && args[4] == "experts-on-cpu";
    if (args.size() > 4 && (!split || ggml_backend_is_cpu(backend))) {
        std::fprintf(stderr, "%s: experts-on-cpu follows a GPU, and nothing else follows the device\n", args[0].c_str());
        return 2;
    }
    ggml_backend_t cpu = split ? init_backend("cpu") : nullptr;
    std::printf("backend: %s%s\n", ggml_backend_name(backend), split ? ", the experts on the CPU" : "");

    const Npy prompt = read_npy(dir + "/prompt.npy");
    const Npy frames = read_npy(dir + "/frames.npy");
    const Npy input = read_npy(dir + "/backbone_input.npy");
    const Npy layers_out = read_npy(dir + "/backbone_layers.npy");
    const Npy mixers = read_npy(dir + "/backbone_mixer.npy");
    const Npy routers = read_npy(dir + "/backbone_router.npy");
    const Npy moes = read_npy(dir + "/backbone_moe.npy");
    const Npy hidden = read_npy(dir + "/backbone_hidden.npy");
    const Npy delta_states = read_npy(dir + "/delta_state.npy");
    const Npy conv_states = read_npy(dir + "/conv_state.npy");
    const Npy main_logits = read_npy(dir + "/main_logits.npy");
    const int64_t rows = prompt.shape[0], width = prompt.shape[1], total = input.shape[0], h = input.shape[1];
    const int64_t experts = routers.shape[2];
    // Measured on an Apple M5 on 2026-10-15 for the ja-ref dump, the lowest SNR of a stage over the layers: 120.2 dB on
    // the CPU in F32, with 6 of 17,040 rows' experts tied, and 63.0 dB on Metal, whose matrix kernel rounds its inputs
    // to half precision, with 1,101 tied. Key heads tiled over the value heads rather than repeated in a row give
    // -3.6 dB for the Gated DeltaNet's output on the CPU.
    const double bound_db = ggml_backend_is_cpu(backend) ? 100 : 50;
    bool ok = true;

    {
        const ModelFile model(path, backend, model_layout,
                              [](const std::string & name) { return name == "token_embd" || name == "ad.codebook_embd"; });
        TextModel text(model, backend, 0, 1);
        std::vector<int32_t> all = prompt.i32;
        all.insert(all.end(), frames.i32.begin(), frames.i32.begin() + (total - rows) * width);
        const Diff d = compare(text.embed(all, total), input.f32);
        print_diff("embeddings of " + std::to_string(total) + " rows", d);
        ok = ok && d.snr_db > bound_db;
    }

    Worst mixer, router, moe, layer, state, conv;
    int delta_index = 0, layer_count = 0;
    int64_t tied_count = 0;
    const int used = (int) ModelFile(path, model_layout).u32("drama3.text.num_experts_per_tok");
    std::vector<float> last_hidden;
    const auto t0 = std::chrono::steady_clock::now();
    for (int l = 0;; l++) {
        const std::string prefix = "blk." + std::to_string(l) + ".";
        const ModelFile model(path, model_layout, [&](const std::string & name) -> ggml_backend_t {
            if (name.rfind(prefix, 0) != 0 && name != "output_norm" && name != "output") return nullptr;
            return split && is_routed_experts(name) ? cpu : backend;
        });
        TextModel text(model, backend, l, l + 1, cpu);
        layer_count = text.layers();
        text.start(total);
        const std::vector<float> & x = l == 0 ? input.f32 : rows_of(layers_out.f32, total * h, l - 1, 1);
        TextModelStages stages, step;
        std::vector<float> out = text.run(rows_of(x, h, 0, rows), rows, &stages);
        if (!text.attention(l)) {
            // The official state is [value heads, key dim, value dim] with the value channel fastest.
            const std::vector<float> mine = text.delta_state(l);
            const int64_t heads = delta_states.shape[1], dk = delta_states.shape[2], dv = delta_states.shape[3];
            std::vector<float> theirs((size_t) (heads * dk * dv));
            const float * official = delta_states.f32.data() + delta_index * heads * dk * dv;
            for (int64_t hh = 0; hh < heads; hh++)
                for (int64_t i = 0; i < dk; i++)
                    for (int64_t j = 0; j < dv; j++) theirs[(size_t) (i + dk * (j + dv * hh))] = official[hh * dk * dv + i * dv + j];
            state.keep(compare(mine, theirs), l);
            const int64_t conv_size = conv_states.shape[1] * conv_states.shape[2];
            conv.keep(compare(text.conv_state(l), rows_of(conv_states.f32, conv_size, delta_index, 1)), l);
            delta_index++;
        }
        for (int64_t r = rows; r < total; r++) {
            const std::vector<float> y = text.run(rows_of(x, h, r, 1), 1, &step);
            out.insert(out.end(), y.begin(), y.end());
            for (auto [all, one] : {std::pair{&stages.mixer, &step.mixer}, {&stages.router, &step.router}, {&stages.moe, &step.moe},
                                    {&stages.layers, &step.layers}}) {
                (*all)[0].insert((*all)[0].end(), (*one)[0].begin(), (*one)[0].end());
            }
        }
        mixer.keep(compare(stages.mixer[0], rows_of(mixers.f32, total * h, l, 1)), l);
        const std::vector<float> official_router = rows_of(routers.f32, total * experts, l, 1);
        const Diff dr = compare(stages.router[0], official_router);
        router.keep(dr, l);
        const std::vector<bool> tied = tied_rows(stages.router[0], official_router, total, experts, used);
        const int64_t ties = std::count(tied.begin(), tied.end(), true);
        tied_count += ties;
        moe.keep(compare(without(stages.moe[0], h, tied), without(rows_of(moes.f32, total * h, l, 1), h, tied)), l);
        const Diff dl = compare(without(stages.layers[0], h, tied), without(rows_of(layers_out.f32, total * h, l, 1), h, tied));
        layer.keep(dl, l);
        std::printf("layer %2d (%s): %.1f dB, %lld tied rows, %.1f s so far\n", l, text.attention(l) ? "attention" : "Gated DeltaNet",
                    dl.snr_db, (long long) ties, seconds_since(t0));
        if (l + 1 == layer_count) {
            const Diff dh = compare(out, hidden.f32);
            print_diff("final norm", dh);
            Worst logits;
            for (int64_t s = 0; s < main_logits.shape[0]; s++) {
                logits.keep(compare(text.logits(rows_of(hidden.f32, h, rows - 1 + s, 1)), rows_of(main_logits.f32, main_logits.shape[1], s, 1)), (int) s);
            }
            logits.print("head's logits, least of " + std::to_string(main_logits.shape[0]) + " rows");
            ok = ok && dh.snr_db > bound_db && logits.diff.snr_db > bound_db;
            break;
        }
    }
    mixer.print("mixers");
    router.print("routers' logits");
    moe.print("experts");
    layer.print("layers");
    state.print("Gated DeltaNet states");
    conv.print("convolutions' inputs");
    std::printf("%-34s %lld of %lld\n", "rows whose experts tie, left out", (long long) tied_count, (long long) (total * layer_count));
    for (const Worst * w : {&mixer, &router, &moe, &layer, &state, &conv}) ok = ok && w->diff.snr_db > bound_db;
    if (cpu) ggml_backend_free(cpu);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
