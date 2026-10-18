// Checks Drama 3's audio decoder against the official one on a dump of reference/drama3/dump.py, with the text model's
// states of the dump as its input. The prompt: each layer's output for every row, with the frame after each row known.
// Each frame drawn: each layer's output and each head's logits, with the dump's frames drawn in place of the draws, and
// the share of the dump's codes that this decoder's logits give a place among the tokens the model card's settings draw
// from with the dump's settings, where a code outside them is one these logits could not have drawn. The last step drew the end of speech,
// whose frame the dump does not hold: there the shared layers mix in nothing, as the official layers do.
//
// usage: drama3-audio-decoder-check <model.gguf> <dump dir> [gpu|cpu|device name]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "drama3/audio-decoder.h"
#include "drama3/layout.h"
#include "drama3/sampler.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "npy.h"
#include "weight-types.h"

using namespace drama3;

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

/** The settings dump.py drew the dump's frames with, from its meta.json. */
Sampling dump_sampling(const std::string & dir) {
    std::ifstream f(std::filesystem::u8path(dir + "/meta.json"));
    if (!f) throw std::runtime_error("cannot open " + dir + "/meta.json");
    const JsonValue meta = parse_json(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    const JsonValue * s = meta.member("sampling");
    const auto number = [&](const char * name) {
        const JsonValue * v = s ? s->member(name) : nullptr;
        if (!v) throw std::runtime_error(dir + "/meta.json has no sampling." + name);
        return v->text;
    };
    Sampling out;
    out.temperature = std::stof(number("temperature"));
    out.top_k = std::stoi(number("top_k"));
    out.top_p = std::stof(number("top_p"));
    return out;
}

/** Rows [first, first + n) of a row-major array of rows of `width` values. */
template <typename T>
std::vector<T> rows_of(const std::vector<T> & a, int64_t width, int64_t first, int64_t n) {
    return std::vector<T>(a.begin() + first * width, a.begin() + (first + n) * width);
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <dump dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    if (!takes_weights(args[1], model_layout, {"F32"}, "its thresholds are those measured with them")) return 2;
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, model_layout, [](const std::string & name) { return name.rfind("ad.", 0) == 0; });
    AudioDecoder decoder(model, backend);
    const Sampling drawn_with = dump_sampling(dir);
    const int d = decoder.dim(), width = 1 + decoder.codebooks(), layers = decoder.layers();
    const int32_t eos = (int32_t) model.u32("drama3.eos_token_id");

    const Npy prompt = read_npy(dir + "/prompt.npy");
    const Npy frames = read_npy(dir + "/frames.npy");
    const Npy hidden = read_npy(dir + "/backbone_hidden.npy");
    const Npy prefill = read_npy(dir + "/tower_prefill.npy");
    const Npy steps = read_npy(dir + "/tower_layers.npy");
    const Npy heads = read_npy(dir + "/head_logits.npy");
    const int64_t rows = prompt.shape[0], drawn = frames.shape[0], n_steps = steps.shape[0];
    // Measured on an Apple M5 on 2026-10-15 for the ja-ref dump, the lowest SNR of a layer's output: on the CPU in F32
    // 116.0 dB for the prompt and 110.3 dB for the frames, and 107.7 dB for a head's logits; on Metal, whose matrix
    // kernel rounds its inputs to half precision, 60.9, 68.7 and 64.2 dB. RoPE that turns the halves of each head's
    // channels rather than adjacent pairs gives 2.1, 4.1 and -5.0 dB on the CPU.
    const double layer_db = ggml_backend_is_cpu(backend) ? 90 : 45, logits_db = layer_db;
    bool ok = true;

    decoder.start(rows + n_steps);
    std::vector<std::vector<float>> outputs;
    auto t0 = std::chrono::steady_clock::now();
    decoder.prefill(rows_of(hidden.f32, d, 0, rows - 1), prompt.i32, rows - 1, &outputs);
    std::printf("prompt of %lld rows: %.3f s\n", (long long) rows - 1, seconds_since(t0));
    Diff worst;
    int worst_layer = 0;
    for (int l = 0; l < layers; l++) {
        const Diff dl = compare(outputs[(size_t) l], rows_of(prefill.f32, (rows - 1) * d, l, 1));
        if (keep_worst(worst, dl)) worst_layer = l;
        ok = ok && outputs[(size_t) l].size() == (size_t) ((rows - 1) * d);
    }
    print_diff("prompt, least of " + std::to_string(layers) + " layers (" + std::to_string(worst_layer) + ")", worst);
    ok = ok && worst.snr_db > layer_db;

    Diff worst_step, worst_logits;
    int inside = 0, codes = 0;
    t0 = std::chrono::steady_clock::now();
    for (int64_t s = 0; s < n_steps; s++) {
        const std::vector<int32_t> frame = s == 0 ? rows_of(prompt.i32, width, rows - 1, 1) : rows_of(frames.i32, width, s - 1, 1);
        const bool ended = s >= drawn;
        const int32_t main = ended ? eos : frames.i32[(size_t) (s * width)];
        const ChooseCode choose = [&](int codebook, const std::vector<float> & logits) {
            const Diff dl = compare(logits, rows_of(heads.f32, (int64_t) logits.size(), s * (width - 2) + codebook - 1, 1));
            keep_worst(worst_logits, dl);
            if (ended) return 0;
            const int32_t code = frames.i32[(size_t) (s * width + 1 + codebook)];
            const Candidates c = candidates(logits, drawn_with);
            inside += std::find(c.tokens.begin(), c.tokens.end(), code) != c.tokens.end();
            codes++;
            return code;
        };
        decoder.step(rows_of(hidden.f32, d, rows - 1 + s, 1), frame, main, choose, &outputs);
        for (int l = 0; l < layers; l++) {
            const Diff dl = compare(outputs[(size_t) l], rows_of(steps.f32, d, s * layers + l, 1));
            keep_worst(worst_step, dl);
        }
    }
    std::printf("%lld frames: %.3f s (%.1f ms a frame)\n", (long long) n_steps, seconds_since(t0), 1000 * seconds_since(t0) / n_steps);
    print_diff("frames, least of every layer", worst_step);
    print_diff("frames, least of every head", worst_logits);
    std::printf("%-34s %d of %d\n", "dump's codes among the candidates", inside, codes);
    ok = ok && worst_step.snr_db > layer_db && worst_logits.snr_db > logits_db && inside == codes;
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
