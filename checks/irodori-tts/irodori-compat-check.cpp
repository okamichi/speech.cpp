// Compares caption truncation, conditions and inactive CFG with the pinned official runtime.
// Usage: irodori-compat-check MODEL.gguf OUT [cpu|gpu|DEVICE]
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>

#include "args.h"
#include "backend.h"
#include "ggml-cpu.h"
#include "compare.h"
#include "error.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/sampler.h"
#include "irodori-tts/text-encoder.h"
#include "irodori-tts/text-normalizer.h"
#include "irodori-tts/tokenizer.h"
#include "json-reader.h"
#include "npy.h"

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUT [cpu|gpu|DEVICE]\n", args[0].c_str());
        return 2;
    }
    try {
        const auto directory = std::filesystem::u8path(args[2]);
        std::ifstream input(directory / "compatibility.json");
        if (!input) throw std::runtime_error("cannot open compatibility.json; run compatibility_cases.py");
        const auto meta = parse_json(std::string(std::istreambuf_iterator<char>(input), {}));
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "cpu");
        const ModelFile model(args[1], backend, irodori::model_layout);
        const irodori::Tokenizer tokenizer(model);
        const irodori::TextEncoder encoder(model);
        const irodori::Dit dit(model, backend);
        const irodori::Sampler sampler(dit, model, backend);
        auto allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        bool ok = true;
        const size_t limit = model.u32("irodori-tts.caption.max_tokens");
        if (limit != std::stoul(meta.member("limit")->text)) throw std::runtime_error("caption limits differ");
        size_t index = 0;
        for (const auto & c : meta.member("captions")->items) {
            std::vector<int32_t> expected;
            for (const auto & id : c.member("ids")->items) expected.push_back(std::stoi(id.text));
            bool truncated = false;
            const auto text = irodori::strip_caption(c.member("text")->text);
            const auto actual = tokenizer.encode_caption(text, limit, true, truncated);
            ok = ok && actual == expected && truncated == c.member("truncated")->boolean;
            bool strict_refused = false;
            try { bool cut = false; tokenizer.encode_caption(text, limit, false, cut); }
            catch (const Error & e) { strict_refused = e.fault() == Fault::OutOfRange && e.input() == "instructions"; }
            ok = ok && strict_refused == truncated;
            std::printf("caption %zu: %zu tokens, truncated %d, strict refusal %d\n", index, actual.size(), truncated, strict_refused);
            if (!expected.empty()) {
                Graph graph;
                auto value = encoder.build(graph, expected, irodori::Condition::Caption);
                graph.output(value);
                graph.compute(backend, allocr);
                const auto want = read_npy((directory / ("caption-" + std::to_string(index) + ".npy")).u8string());
                const auto diff = compare(Graph::read(value), want.f32);
                print_diff("caption condition", diff);
                ok = ok && diff.snr_db >= (ggml_backend_is_cpu(backend) ? 60 : 35);
            }
            ++index;
        }
        for (const auto & c : meta.member("guidance")->items) {
            auto g = sampler.guidance(); g.text = 3; g.speaker = 8; g.caption = 7;
            auto got = sampler.check(g, 2, c.member("reference")->boolean, c.member("caption")->boolean, true);
            ok = ok && got.text == std::stof(c.member("text")->text) && got.speaker == std::stof(c.member("speaker")->text) &&
                 got.caption == std::stof(c.member("instructions")->text);
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        std::printf("official caption and CFG compatibility: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
