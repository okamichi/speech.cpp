// Compares the T5Gemma2 backbone and both tokenizers with the official stage fixtures.
// Usage: irodori-text-stage-check OUT [cpu|gpu|DEVICE]
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-tts/t5gemma-encoder.h"
#include "irodori-tts/tokenizer.h"
#include "npy.h"

namespace {

/** The T5Gemma2 stage's tensors, whose dimensions the fixture's config gives. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    const std::string p = "irodori-tts.text.";
    const int64_t h = m.size(p + "hidden_size"), d = m.size(p + "head_dim");
    const int64_t q = d * m.size(p + "num_heads"), kv = d * m.size(p + "num_kv_heads");
    const int64_t vocab = m.width("text.embd", 1), inner = m.width("text.blk.0.ffn_act", 1);
    static const Storage f32 = float32_storage();
    std::vector<TensorSpec> result = {{"text.embd", {h, vocab}, f32}, {"text.eoi_embd", {h}, f32}, {"text.final_norm", {h}, f32}};
    add_numbered(result, "text.blk.", m.count(p + "num_layers"),
                 {{"attn_q", {h, q}, f32}, {"attn_k", {h, kv}, f32}, {"attn_v", {h, kv}, f32}, {"attn_out", {q, h}, f32},
                  {"q_norm", {d}, f32}, {"k_norm", {d}, f32}, {"attn_norm", {h}, f32}, {"attn_post_norm", {h}, f32},
                  {"ffn_norm", {h}, f32}, {"ffn_post_norm", {h}, f32}, {"ffn_act", {h, inner}, f32},
                  {"ffn_gate", {h, inner}, f32}, {"ffn_down", {inner, h}, f32}});
    return result;
}

std::string unhex(const std::string & s) {
    std::string result;
    for (size_t i = 0; i < s.size(); i += 2) result.push_back((char) std::stoi(s.substr(i, 2), nullptr, 16));
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 2) {
        std::fprintf(stderr, "usage: %s OUT [cpu|gpu|DEVICE]\n", args[0].c_str());
        return 2;
    }
    try {
        const auto root = std::filesystem::u8path(args[1]);
        ggml_backend_t backend = init_backend(args.size() > 2 ? args[2] : "cpu");
        const Layout layout = {"irodori-text-stage", 1, "regenerate with text_stage.py", tensors};
        const ModelFile model((root / "stage.gguf").u8string(), backend, layout);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        bool ok = true;
        for (const char * name : {"short", "window", "eoi"}) {
            const auto dir = root / name;
            const Npy ids = read_npy((dir / "input_ids.npy").u8string());
            const Npy reference = read_npy((dir / "text_layers.npy").u8string());
            const Npy final = read_npy((dir / "text_backbone.npy").u8string());
            Graph graph;
            std::vector<ggml_tensor *> layers;
            auto out = irodori::t5gemma_encode(model, graph, ids.i32, &layers);
            for (auto layer : layers) graph.output(layer);
            graph.output(out);
            graph.compute(backend, allocr);
            const size_t stride = ids.i32.size() * (size_t) reference.shape[2];
            if (layers.size() != (size_t) reference.shape[0]) throw std::runtime_error("stage layer counts differ");
            for (size_t l = 0; l < layers.size(); l++) {
                const Diff diff = compare(Graph::read(layers[l]).data(), &reference.f32[l * stride], stride);
                print_diff(std::string(name) + " layer " + std::to_string(l), diff);
                ok = ok && diff.snr_db >= 60;
            }
            const Diff diff = compare(Graph::read(out), final.f32);
            print_diff(std::string(name) + " output", diff);
            ok = ok && diff.snr_db >= 60;
        }
        for (const char * name : {"large", "anime"}) {
            const Layout token_layout = {"irodori-tokenizer-stage", 1, "regenerate with text_stage.py",
                                          [](const ModelFile &) { return std::vector<TensorSpec>{}; }};
            const ModelFile file((root / (std::string(name) + "-tokenizer.gguf")).u8string(), token_layout);
            const irodori::Tokenizer tokenizer(file);
            std::ifstream cases(std::filesystem::u8path((root / (std::string(name) + "-tokenizer.tsv")).u8string()));
            if (!cases) throw std::runtime_error("cannot open tokenizer cases");
            std::string line;
            int total = 0, failed = 0;
            while (std::getline(cases, line)) {
                const auto split = line.find('\t');
                const auto text = unhex(line.substr(0, split));
                std::istringstream input(line.substr(split + 1));
                std::vector<int32_t> expected;
                int32_t id;
                while (input >> id) expected.push_back(id);
                const auto actual = tokenizer.encode(text);
                total++;
                if (actual != expected) {
                    failed++;
                    std::printf("%s mismatch: %s\n", name, text.c_str());
                }
            }
            std::printf("%s tokenizer: %d/%d cases match\n", name, total - failed, total);
            ok = ok && total > 0 && failed == 0;
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
