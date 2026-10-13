// Checks the talker and the code predictor against a reference dump.
//
// The prompt comes first, with the tokens of the dump's instruction where it has one, from speech.cpp's tokenizer.
// Teacher forcing then feeds the reference codes frame by frame and compares every stage's output, the code
// predictor's a step at a time, so the first stage that departs is the one reported. A free-running greedy decode then
// shows whether the same codes come out, the code predictor making each frame in one graph, whose logits are compared
// with the reference's while the decode has picked the reference's codes. It fails when the instruction's tokens or the
// prompt's rows differ, when a stage parts from the reference further than its arithmetic explains, or when the greedy
// decode picks another code than the reference where the reference's logit of its own pick exceeds that of the other by
// more than twice the largest error of that stack's logits under teacher forcing, which each of the two logits may have
// in opposite directions: a departure at a nearer tie is one the arithmetic explains, and the codes after it are no
// longer comparable.
//
// It takes an F32 file. The talker keeps its keys and values in F16, which parts its logits and hidden states from the
// official float32 by a relative 1.7e-3 at most and the code predictor's logits by 1.9e-3, on the CPU and on the Metal
// of an Apple M5, and the prompt by 1.2e-6 on the CPU and 7.8e-5 on Metal, for the 1.7B model with and without an
// instruction (2026-10-07).
//
// usage: qwen3-tts-talker-check <model.gguf> <reference dir> [gpu|cpu]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <string>

#include "args.h"
#include "backend.h"
#include "json-reader.h"
#include "json.h"
#include "npy.h"
#include "qwen2-tokenizer.h"
#include "qwen3-tts/layout.h"
#include "qwen3-tts/prompt.h"
#include "qwen3-tts/sampler.h"
#include "qwen3-tts/synthesizer.h"
#include "qwen3-tts/talker.h"

namespace {

/** The largest relative error of a prompt's row, and of a stage's output, that the arithmetic explains, with a margin. */
constexpr double kPromptTolerance = 1e-3;
constexpr double kStageTolerance = 1e-2;

struct Worst {
    double rel = 0, abs = 0;
    int argmax_miss = 0;
    int count = 0;

    void add(const float * got, const float * want, int n) {
        double num = 0, den = 0;
        for (int i = 0; i < n; i++) {
            num += (double) (got[i] - want[i]) * (got[i] - want[i]);
            den += (double) want[i] * want[i];
            abs = std::max(abs, (double) std::fabs(got[i] - want[i]));
        }
        rel = std::max(rel, std::sqrt(num / std::max(den, 1e-30)));
        if (std::max_element(got, got + n) - got != std::max_element(want, want + n) - want) argmax_miss++;
        count++;
    }
    void print(const char * what) const {
        std::printf("%-26s worst relative error %.2e, largest difference %.2e, argmax differs %d of %d\n", what, rel, abs, argmax_miss,
                    count);
    }
};

/** The dump's meta.json. */
JsonValue read_meta(const std::string & dir) {
    std::ifstream f(std::filesystem::u8path(dir + "/meta.json"));
    if (!f) throw std::runtime_error("cannot open " + dir + "/meta.json");
    return parse_json(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
}

/** A string member of the dump's meta.json, or "" when it has none. */
std::string meta_field(const JsonValue & meta, const std::string & key) {
    const JsonValue * v = meta.member(key);
    if (v && v->kind != JsonValue::Kind::String) throw std::runtime_error("meta.json's " + key + " is not a string");
    return v ? v->text : "";
}

/** The BCP 47 tag of a language as the official API names it, which the dump's meta.json keeps. */
std::string language_tag(const std::string & official) {
    static const std::map<std::string, std::string> tags = {
        {"auto", "auto"},    {"chinese", "zh"},  {"english", "en"}, {"french", "fr"},     {"german", "de"},  {"italian", "it"},
        {"japanese", "ja"}, {"korean", "ko"},   {"portuguese", "pt"}, {"russian", "ru"}, {"spanish", "es"}};
    const auto it = tags.find(official);
    if (it == tags.end()) throw std::runtime_error("the dump names the language " + official + ", which the check cannot tag");
    return it->second;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference dir> [gpu|cpu]\n", args[0].c_str());
        return 2;
    }
    const std::string dir = args[2];
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], backend, qwen3_tts_layout);
    Talker talker(model, backend);
    const PromptIds ids(model);
    const Generation generation(model);

    const Npy input_ids = read_npy(dir + "/input_ids.npy");
    const Npy ref_prefill = read_npy(dir + "/prefill_embeds.npy");
    const Npy ref_logits = read_npy(dir + "/talker_logits.npy");
    const Npy ref_hidden = read_npy(dir + "/talker_hidden.npy");
    const Npy ref_cp = read_npy(dir + "/cp_logits.npy");
    const Npy ref_codes = read_npy(dir + "/codes.npy");
    const JsonValue meta = read_meta(dir);
    const std::string speaker = meta_field(meta, "speaker"), language = language_tag(meta_field(meta, "language"));
    const int h = talker.hidden(), n_groups = talker.num_code_groups();
    const int n_frames = (int) ref_codes.shape[0];

    // The instruction's tokens from speech.cpp's tokenizer, against the official processor's.
    std::vector<int32_t> instruction;
    const std::string instruct = meta_field(meta, "instruct");
    if (!instruct.empty()) {
        instruction = instruction_ids(Qwen2Tokenizer(model, kTextTokenizer), ids, instruct);
        const Npy ref_instruction = read_npy(dir + "/instruct_ids.npy");
        if (instruction != ref_instruction.i32) {
            std::printf("FAIL: the instruction %s is %zu tokens, the official processor's %zu, or other tokens\n", json_string(instruct).c_str(),
                        instruction.size(), ref_instruction.i32.size());
            return 1;
        }
        std::printf("instruction %s: the official processor's %zu tokens\n", json_string(instruct).c_str(), instruction.size());
    }

    const Prompt prompt = build_prompt(talker, ids, input_ids.i32, speaker, language, instruction);
    Worst w_prompt;
    if (prompt.n != ref_prefill.shape[0]) {
        std::printf("FAIL: the prompt has %d rows, the reference %lld\n", prompt.n, (long long) ref_prefill.shape[0]);
        return 1;
    }
    for (int r = 0; r < prompt.n; r++) w_prompt.add(&prompt.embeds[r * h], &ref_prefill.f32[r * h], h);
    w_prompt.print("prompt embeddings");

    // Teacher forcing.
    Worst w_logits, w_hidden, w_cp;
    talker.prefill(prompt.embeds, prompt.n, (int64_t) prompt.n + n_frames);
    w_logits.add(talker.logits().data(), &ref_logits.f32[0], talker.vocab());
    w_hidden.add(talker.hidden_state().data(), &ref_hidden.f32[0], h);
    for (int f = 0; f < n_frames; f++) {
        const int32_t * codes = &ref_codes.i32[f * n_groups];
        const float * cp_ref = &ref_cp.f32[(size_t) f * (n_groups - 1) * talker.cp_vocab()];
        w_cp.add(talker.cp_begin(codes[0]).data(), cp_ref, talker.cp_vocab());
        for (int g = 1; g < n_groups - 1; g++) {
            w_cp.add(talker.cp_next(g, codes[g]).data(), cp_ref + (size_t) g * talker.cp_vocab(), talker.cp_vocab());
        }
        talker.step(codes, prompt.frame_extra);
        w_logits.add(talker.logits().data(), &ref_logits.f32[(size_t) (f + 1) * talker.vocab()], talker.vocab());
        w_hidden.add(talker.hidden_state().data(), &ref_hidden.f32[(size_t) (f + 1) * h], h);
    }
    w_logits.print("talker logits");
    w_hidden.print("talker hidden state");
    w_cp.print("code predictor logits");

    // Free-running greedy decode with the official talker settings, as dump.py ran it.
    // The graph's error in all, and at each step, so that the first step that departs shows.
    Worst w_frame;
    std::vector<Worst> w_frame_steps((size_t) n_groups - 1);
    bool one_graph = true;
    SamplingParams talker_p = generation.talker;
    talker_p.greedy = true;
    SamplingParams cp_p = generation.code_predictor;
    cp_p.greedy = true;
    std::vector<bool> none;
    std::mt19937_64 rng(0);
    std::vector<int32_t> history, codes(n_groups);
    int frames = 0;
    // The first code the decode picks other than the reference, and how far the reference's logits prefer their own.
    int diff_frame = -1, diff_group = -1;
    double diff_margin = 0;
    const auto compare = [&](int f, int g, int32_t got, const float * ref_row) {
        if (diff_frame >= 0) return;
        const int32_t want = f < n_frames ? ref_codes.i32[f * n_groups + g] : ids.codec_eos;
        if (got == want) return;
        diff_frame = f;
        diff_group = g;
        diff_margin = (double) ref_row[want] - ref_row[got];
    };
    const int max_frames = 2 * n_frames + 10;
    talker.prefill(prompt.embeds, prompt.n, (int64_t) prompt.n + max_frames);
    for (int f = 0; f < max_frames; f++) {
        codes[0] = sample(talker.logits(), talker_p, history, generation.banned(talker.vocab(), ids.codec_eos, f), rng);
        if (f <= n_frames) {
            // The reference's logits as the greedy pick sees them, with the repetition penalty on the codes before.
            std::vector<float> row(&ref_logits.f32[(size_t) f * talker.vocab()], &ref_logits.f32[(size_t) (f + 1) * talker.vocab()]);
            std::vector<bool> seen(row.size(), false);
            for (int32_t t : history) {
                if (seen[t]) continue;
                seen[t] = true;
                row[t] = row[t] < 0 ? row[t] * talker_p.repetition_penalty : row[t] / talker_p.repetition_penalty;
            }
            compare(f, 0, codes[0], row.data());
        }
        if (codes[0] == ids.codec_eos) break;
        history.push_back(codes[0]);
        one_graph &= talker.cp_frame(codes[0], cp_p, {}, &codes[1]);
        for (int g = 1; g < n_groups; g++) {
            const float * ref_row = &ref_cp.f32[((size_t) f * (n_groups - 1) + g - 1) * talker.cp_vocab()];
            if (!one_graph) {
                // The code predictor's repetition penalty sees the codes of the frame it has made, as the synthesis gives it.
                const std::vector<int32_t> made(codes.begin() + 1, codes.begin() + g);
                codes[g] = sample(g == 1 ? talker.cp_begin(codes[0]) : talker.cp_next(g - 1, codes[g - 1]), cp_p, made, none, rng);
            } else if (f < n_frames && diff_frame < 0) {
                // The graph's logits of a code whose frame and frames before follow the reference read what the reference's did.
                const std::vector<float> got = talker.cp_frame_logits(g - 1);
                w_frame.add(got.data(), ref_row, talker.cp_vocab());
                w_frame_steps[(size_t) g - 1].add(got.data(), ref_row, talker.cp_vocab());
            }
            if (f < n_frames) compare(f, g, codes[g], ref_row);
        }
        talker.step(codes.data(), prompt.frame_extra);
        frames++;
    }
    if (one_graph) {
        w_frame.print("code predictor in a graph");
        for (size_t step = 0; step < w_frame_steps.size(); step++) {
            w_frame_steps[step].print(("  step " + std::to_string(step) + ", code " + std::to_string(step + 1)).c_str());
        }
    }
    else std::printf("the device cannot compute the code predictor's frame in one graph, which the decode made a step at a time\n");
    std::printf("code predictor graphs: %.2f MiB on the device\n", talker.cp_graph_bytes() / 1048576.0);
    ggml_backend_free(backend);
    bool ok = w_prompt.rel <= kPromptTolerance && w_logits.rel <= kStageTolerance && w_hidden.rel <= kStageTolerance &&
              w_cp.rel <= kStageTolerance && w_frame.rel <= kStageTolerance;
    if (diff_frame < 0) {
        std::printf("greedy decode: the reference's %d frames\n", n_frames);
    } else {
        const double explained = 2 * (diff_group == 0 ? w_logits.abs : w_cp.abs);
        std::printf("greedy decode: %d frames (reference %d), the first code that differs at frame %d, group %d, where the reference "
                    "prefers its own by %.2e of the logits, against %.2e that the arithmetic explains\n",
                    frames, n_frames, diff_frame, diff_group, diff_margin, explained);
        ok &= diff_margin <= explained;
    }
    std::printf("%s\n", ok ? "ok" : "FAIL: a stage parts from the reference further than its arithmetic explains");
    return ok ? 0 : 1;
}
