// Checks the text stage of Irodori-TTS against the official implementation: the normalization and the
// tokenizer on the cases of reference/irodori-tts/text_cases.py, then the text encoder, layer by layer, on
// the tokens of every dump that has them (dump.py's, and text_cases.py's long texts). The text encoder is given
// the dump's tokens, so its error is its own. With a file that holds the caption's encoder, the same for captions:
// their strip and tokens on the cases of reference/irodori-tts/caption_cases.py, and the caption condition on the
// tokens of every dump that has one. Each takes the cases and dumps made with the model of the file alone.
//
// usage: irodori-text-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-dumps.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/text-encoder.h"
#include "irodori-tts/text-normalizer.h"
#include "irodori-tts/tokenizer.h"
#include "npy.h"
#include "weight-types.h"

using namespace irodori;

namespace {

std::string from_hex(const std::string & hex) {
    std::string s;
    for (size_t i = 0; i + 1 < hex.size(); i += 2) s += (char) std::stoi(hex.substr(i, 2), nullptr, 16);
    return s;
}

/**
 * The cases that fail `normalize` or the tokenizer, of all the cases in `path`: a text, what `normalize` makes of it and
 * its tokens, which a text that normalizes to nothing has none of.
 */
int check_cases(const Tokenizer & tokenizer, const std::string & path, std::string (*normalize)(const std::string &), int & total) {
    std::ifstream f(std::filesystem::u8path(path));
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string line;
    int failed = 0;
    while (std::getline(f, line)) {
        const size_t a = line.find('\t'), b = line.find('\t', a + 1);
        const std::string text = from_hex(line.substr(0, a)), want_text = from_hex(line.substr(a + 1, b - a - 1));
        std::vector<int32_t> want;
        for (size_t i = b + 1; i < line.size();) {
            size_t j = line.find(' ', i);
            if (j == std::string::npos) j = line.size();
            if (j > i) want.push_back(std::stoi(line.substr(i, j - i)));
            i = j + 1;
        }
        total++;
        const std::string got_text = normalize(text);
        const std::vector<int32_t> got = got_text.empty() ? std::vector<int32_t>{} : tokenizer.encode(got_text);
        if (got_text != want_text || got != want) {
            failed++;
            std::printf("MISMATCH: %s\n  normalized: %s\n  want:       %s\n  ids:", text.c_str(), got_text.c_str(),
                        want_text.c_str());
            for (int32_t id : got) std::printf(" %d", id);
            std::printf("\n  want ids:");
            for (int32_t id : want) std::printf(" %d", id);
            std::printf("\n");
        }
    }
    return failed;
}

/**
 * The folders under `root`, at any depth, that hold `file` and a meta.json naming the model of `repository`: a link to a
 * folder is followed, as in a worktree whose out/ links another checkout's dumps.
 */
std::vector<std::filesystem::path> folders_with(const std::filesystem::path & root, const std::string & file, const std::string & repository) {
    std::vector<std::filesystem::path> found;
    const auto follow = std::filesystem::directory_options::follow_directory_symlink;
    for (const auto & e : std::filesystem::recursive_directory_iterator(root, follow)) {
        if (!e.is_regular_file() || e.path().filename() != file) continue;
        const std::filesystem::path d = e.path().parent_path();
        if (std::filesystem::exists(d / "meta.json") && IrodoriDump(d).repository() == repository) found.push_back(d);
    }
    std::sort(found.begin(), found.end());
    return found;
}

/** The cases of the one folder under `root` that holds `file` for the model of `repository`. */
std::string cases_of(const std::filesystem::path & root, const std::string & file, const std::string & repository) {
    const std::vector<std::filesystem::path> found = folders_with(root, file, repository);
    if (found.size() != 1) {
        throw std::runtime_error(std::to_string(found.size()) + " folders under " + root.u8string() + " hold " + file + " for " + repository +
                                 "; make one with reference/irodori-tts/");
    }
    return (found[0] / file).u8string();
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], model_layout, {"F32", "F16", "Q8_0"},
                           "its bound is the one measured with them")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        const ModelFile model(args[1], backend, model_layout);
        const Tokenizer tokenizer(model);
        const TextEncoder encoder(model);
        const std::filesystem::path dir = std::filesystem::u8path(args[2]);
        const std::string repository = model.str("general.source.repo_url");

        int total = 0;
        const int failed = check_cases(tokenizer, cases_of(dir, "text-cases.tsv", repository), normalize_text, total);
        std::printf("normalization and tokenizer: %d of %d cases match\n", total - failed, total);
        bool ok = failed == 0;
        const bool caption = model.boolean("irodori-tts.caption_condition");
        if (caption) {
            int captions = 0;
            const int caption_failed = check_cases(tokenizer, cases_of(dir, "caption-cases.tsv", repository), strip_caption, captions);
            std::printf("caption strip and tokenizer: %d of %d cases match\n", captions - caption_failed, captions);
            ok = ok && caption_failed == 0;
        } else {
            std::printf("captions: skipped, the file lacks the caption's encoder\n");
        }

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        const std::vector<std::filesystem::path> dumps = folders_with(dir, "text_layers.npy", repository);
        double worst = INFINITY;
        for (const auto & d : dumps) {
            const Npy ids = read_npy((d / "input_ids.npy").u8string());
            const Npy layers = read_npy((d / "text_layers.npy").u8string());
            const Npy state = read_npy((d / "text_state.npy").u8string());
            Graph g;
            std::vector<ggml_tensor *> hidden;
            ggml_tensor * out = encoder.build(g, ids.i32, Condition::Text, &hidden);
            g.output(out);
            for (ggml_tensor * h : hidden) g.output(h);
            g.compute(backend, allocr);
            const size_t n = ids.i32.size(), width = (size_t) layers.shape[2];
            Diff worst_layer;
            int worst_index = 0;
            for (size_t l = 0; l < hidden.size(); l++) {
                const Diff dl = compare(Graph::read(hidden[l]).data(), &layers.f32[l * n * width], n * width);
                if (dl.snr_db < worst_layer.snr_db) {
                    worst_layer = dl;
                    worst_index = (int) l;
                }
            }
            const Diff ds = compare(Graph::read(out), state.f32);
            std::printf("%s (%zu tokens)\n", d.filename().u8string().c_str(), n);
            print_diff("  worst backbone layer " + std::to_string(worst_index), worst_layer);
            print_diff("  text condition", ds);
            worst = std::min(worst, std::min(ds.snr_db, worst_layer.snr_db));
        }
        const std::vector<std::filesystem::path> captions =
            caption ? folders_with(dir, "caption_state.npy", repository) : std::vector<std::filesystem::path>{};
        for (const auto & d : captions) {
            const Npy ids = read_npy((d / "caption_ids.npy").u8string());
            const Npy state = read_npy((d / "caption_state.npy").u8string());
            Graph g;
            ggml_tensor * out = encoder.build(g, ids.i32, Condition::Caption);
            g.output(out);
            g.compute(backend, allocr);
            const Diff ds = compare(Graph::read(out), state.f32);
            std::printf("%s (%zu caption tokens)\n", d.filename().u8string().c_str(), ids.i32.size());
            print_diff("  caption condition", ds);
            worst = std::min(worst, ds.snr_db);
        }
        ggml_gallocr_free(allocr);
        // Measured on an Apple M5: F32 weights give 118 dB on the CPU and 57 dB on Metal, whose matrix
        // kernels round to half precision; F16 weights 57 dB and Q8_0 weights 37 dB. A wrong operation
        // falls far below this bound.
        if (dumps.empty() || worst < 30) ok = false;
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
