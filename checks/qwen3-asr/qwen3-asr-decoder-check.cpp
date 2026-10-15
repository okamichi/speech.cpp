// Checks Qwen3-ASR's prompt, decoder, decoding and parse against transformers' on each dump of
// reference/qwen3-asr/dump.py and each of its requests (auto, forced, auto-prompt, forced-prompt), after the parse of
// the cases of reference/qwen3-asr/parse_cases.py against qwen-asr's, its language and its text, in the order data
// flows: the prompt's ids; the decoder's input, the token embeddings with the dump's projector output spliced in; the
// logits of the prompt's last four rows from the dump's input; the decoder teacher-forced on the dump's ids; the
// tokenizer's decoding of the dump's ids and the parse of the dump's raw text, its language and its text; the greedy
// decoding from the dump's projector output with every later stage ours; and the language and the text from the dump's
// audio with every stage ours, which it also times. A language qwen-asr parses is compared by the index of
// general.languages whose name qwen3-asr.language_names holds, and one it does not hold with none. The dumps are those
// of the model, in <reference out dir>/<its general.name>/.
//
// Where a greedy choice of ours differs from the dump's, the check takes it for the arithmetic's when the dump's margin
// between the two tokens is within the sum of our errors on their two logits, and for a defect otherwise. The decoder
// attends as it chooses on the device, or as the last argument forces it, so that both attentions are checked on a GPU.
//
// usage: qwen3-asr-decoder-check <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen3-asr/layout.h"
#include "qwen3-asr/recognizer.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace qwen3_asr;

namespace {

using Clock = std::chrono::steady_clock;

const char * const kVariants[] = {"auto", "forced", "auto-prompt", "forced-prompt"};

JsonValue read_meta(const std::filesystem::path & dir) {
    return parse_json(dump_text(dir / "meta.json"));
}

/** The index of general.languages that qwen3-asr.language_names names `name`, or none for "" and a name it does not hold. */
std::optional<size_t> language_named(const ModelFile & m, const std::string & name) {
    const std::vector<std::string> names = m.str_array("qwen3-asr.language_names");
    const auto it = std::find(names.begin(), names.end(), name);
    if (it == names.end()) return std::nullopt;
    return (size_t) (it - names.begin());
}

/** The request of a variant's meta.json: its context, and its forced language as an index of the model's languages. */
RecognitionRequest request_of(const JsonValue & meta, const ModelFile & m) {
    RecognitionRequest r;
    r.context = meta.member("prompt")->text;
    const JsonValue * language = meta.member("language");
    if (language && language->kind == JsonValue::Kind::String) {
        r.language = language_named(m, language->text);
        if (!r.language) throw std::runtime_error("the dump forces " + language->text + ", which the model does not name");
    }
    return r;
}

/** A language as an index of the model's languages, or none, written as its tag or "none". */
std::string tag_of(const std::optional<size_t> & language, const ModelFile & m) {
    return language ? m.str_array("general.languages").at(*language) : "none";
}

/** The languages of a recognition as tags joined with commas, "" for none. */
std::string tags_of(const std::vector<size_t> & languages, const ModelFile & m) {
    std::string out;
    for (size_t language : languages) out += (out.empty() ? "" : ",") + tag_of(language, m);
    return out;
}

/**
 * How our greedy choices compare with the dump's at the steps it compares: each step's top logits of the dump against
 * ours at the same ids, and every step where our choice differs, with whether the error explains it.
 */
struct Steps {
    size_t steps = 0, differ = 0, unexplained = 0;
    double worst_snr = INFINITY, worst_abs = 0;
    std::optional<size_t> first_differ;

    /**
     * Compares our logits at step `i` with the dump's top ids and logits; returns whether our choice is the dump's id
     * `want`.
     */
    bool add(size_t i, const std::vector<float> & logits, const Npy & top_ids, const Npy & top_logits, int32_t want) {
        const size_t k = (size_t) top_ids.shape[1];
        std::vector<float> ours(k), theirs(top_logits.f32.begin() + (std::ptrdiff_t) (i * k), top_logits.f32.begin() + (std::ptrdiff_t) ((i + 1) * k));
        for (size_t j = 0; j < k; j++) ours[j] = logits[(size_t) top_ids.i32[i * k + j]];
        const Diff d = compare(ours, theirs);
        worst_snr = std::min(worst_snr, d.snr_db);
        worst_abs = std::max(worst_abs, d.max_abs);
        steps++;
        const int32_t got = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (got == want) return true;
        differ++;
        if (!first_differ) first_differ = i;
        // The dump's margin between its choice and ours, against our errors on those two logits.
        size_t at_want = k, at_got = k;
        for (size_t j = 0; j < k; j++) {
            if (top_ids.i32[i * k + j] == want) at_want = j;
            if (top_ids.i32[i * k + j] == got) at_got = j;
        }
        const bool explained = at_want < k && at_got < k &&
                               theirs[at_want] - theirs[at_got] <= std::fabs(ours[at_want] - theirs[at_want]) + std::fabs(ours[at_got] - theirs[at_got]);
        if (!explained) unexplained++;
        std::printf("    step %zu: ours %d where the dump's is %d, margin %.4f, %s\n", i, got, want,
                    at_got < k ? theirs[at_want] - theirs[at_got] : NAN, explained ? "within our error on the two logits" : "UNEXPLAINED");
        return false;
    }

    void print(const char * what) const {
        std::printf("  %-32s %zu of %zu steps ours, top logits' worst SNR %.1f dB, max |diff| %.2e\n", what, steps - differ, steps, worst_snr,
                    worst_abs);
    }
};

/**
 * Decodes greedily from the decoder's prefill, comparing each step with the dump's until the first that differs;
 * returns the ids it wrote.
 */
std::vector<int32_t> follow(Decoder & decoder, const Npy & ids, const Npy & top_ids, const Npy & top_logits, Steps & steps) {
    bool same = true;
    const auto compare_step = [&](size_t i) {
        if (same && i < (size_t) ids.shape[0]) same = steps.add(i, decoder.logits(), top_ids, top_logits, ids.i32[i]);
    };
    // The decoding calls back after each token but the last, before it feeds the token; the logits of the last stay.
    const Generation g = *decoder.generate([&](size_t n) {
        compare_step(n - 1);
        return true;
    });
    compare_step(g.ids.size() - 1);
    return g.ids;
}

/**
 * Decodes greedily from the decoder's prefill with a draft of `draft`, compares each step with the dump's as `greedy`, a
 * decoding without one, compared, until the first that differs, and checks the steps against the dump as that
 * decoding's: no choice of ours that the error does not explain, and top logits no more than 12 dB further from the
 * dump's than its. A step of several rows rounds otherwise than a step of one: on an RTX 2080 with Vulkan, whose
 * products of several columns quantize their activations to 8 bits, the drafted steps of Qwen3-ASR 0.6B in Q8_0 lay up
 * to 6.9 dB further from the dump, on an Apple M5's CPU 2.1 dB and on its GPU 0.1 dB; with the rewind one row short,
 * they lay 21.8 dB further on the M5's GPU (2026-10-15). Prints whether it wrote the ids written without a draft,
 * `written`.
 */
bool drafted(Decoder & decoder, const char * what, const std::vector<int32_t> & draft, const Recognizer & recognizer, bool forced,
             const Npy & ids, const Npy & top_ids, const Npy & top_logits, const Steps & greedy, const std::vector<int32_t> & written) {
    Draft d(draft, recognizer.asr_text(), forced);
    std::vector<std::vector<float>> seen;
    const std::vector<int32_t> got = decoder.generate([](size_t) { return true; }, &d, &seen)->ids;
    Steps steps;
    for (size_t i = 0; i < got.size() && i < (size_t) ids.shape[0]; i++) {
        if (!steps.add(i, seen.at(i), top_ids, top_logits, ids.i32[i])) break;
    }
    const bool close = steps.worst_snr >= greedy.worst_snr - 12;
    std::printf("  %-32s %zu of %zu steps the dump's, top logits' worst SNR %.1f dB against %.1f without one, %s; %d of %d drafted tokens "
                "kept in %d steps\n",
                what, steps.steps - steps.differ, steps.steps, steps.worst_snr, greedy.worst_snr, got == written ? "the ids without one" : "other ids",
                d.kept, d.checked, d.checks);
    if (!close) std::printf("    the drafted steps lie more than 12 dB further from the dump than the steps without a draft\n");
    return close && steps.unexplained == 0;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3 || (args.size() > 4 && args[4] != "flash" && args[4] != "products")) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32", "F16", "Q8_0"},
                           "its bounds are those measured with them")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::optional<Qwen3Attention> attention;
        if (args.size() > 4) attention = args[4] == "flash" ? Qwen3Attention::Flash : Qwen3Attention::Products;
        bool ok = true;
        {
            Recognizer recognizer(args[1], backend, attention);
            std::printf("backend: %s, %s\n", ggml_backend_name(backend), qwen3_attention_name(recognizer.decoder().attention()));
            const ModelFile & m = recognizer.model();
            Decoder & decoder = recognizer.decoder();
            // The stages from the dump's input run with an F32 cache, which leaves the arithmetic of the weights alone;
            // the greedy decodings run as the recognizer does, with its F16 cache.
            Decoder exact(m, backend, GGML_TYPE_F32, Qwen3Readback::Logits, recognizer.decoder().attention());
            // The recognizer's decoder reads back only its picks; the greedy decodings follow the logits of one with its
            // cache, and the recognizer's must pick the first of each step's largest logits.
            Decoder followed(m, backend, kCacheType, Qwen3Readback::Logits, recognizer.decoder().attention());
            // Measured on an Apple M5 on 2026-10-06, the prompt's last logits from the dumps' input against transformers'
            // float32, over the ten inputs and four requests of the 0.6B and the 1.7B model: with F32 weights 97.4 and
            // 96.1 dB on the CPU, and 43.3 and 46.6 dB on Metal, whose matrix kernel rounds its inputs to half precision,
            // as it does with F16 weights, which give the same there and 37.3 dB with the 0.6B model on the CPU, whose
            // dot product sums in half precision; with Q8_0 weights 19.3 and 18.2 dB on the CPU, which quantizes the
            // activations to 8 bits too, and 23.3 and 21.7 dB on Metal. The greedy choices where these errors exceed the
            // dump's margin between two tokens differ, which the steps below tell from a defect.
            const ggml_type type = m.tensor("dec.blk.0.attn_q")->type;
            const double threshold_db = type == GGML_TYPE_Q8_0 ? 15 : type == GGML_TYPE_F32 && ggml_backend_is_cpu(backend) ? 90 : 35;
            std::printf("prefill logits threshold: %.0f dB\n", threshold_db);
            // The parse of texts that hold what the dumps do not: repetitions, whitespace, outputs without <asr_text>, and
            // the languages before it.
            {
                std::ifstream f(std::filesystem::u8path(args[2]) / "parse-cases.jsonl", std::ios::binary);
                if (!f) throw std::runtime_error("no parse-cases.jsonl of reference/qwen3-asr/parse_cases.py is in " + args[2]);
                int cases = 0, same = 0, others = 0;
                for (std::string line; std::getline(f, line);) {
                    const JsonValue c = parse_json(line);
                    const JsonValue * forced = c.member("forced"), * language = c.member("language");
                    if (!forced || !language) throw std::runtime_error("parse-cases.jsonl holds no languages; write it again with parse_cases.py");
                    const std::optional<size_t> forced_language =
                        forced->kind == JsonValue::Kind::String ? language_named(m, forced->text) : std::nullopt;
                    const Parsed got = recognizer.transcript().parse(c.member("raw")->text, forced_language);
                    // A name qwen-asr parses that the model's languages do not hold is none for us, which the parse must
                    // also give as the name it found.
                    const std::optional<size_t> want = language_named(m, language->text);
                    const bool other = !language->text.empty() && !want;
                    const bool equal = got.text == c.member("text")->text && got.language == want && got.other_language.empty() != other;
                    cases++;
                    same += equal;
                    others += other;
                    if (!equal) {
                        std::printf("  parse of %s: %s, %s where qwen-asr gives %s, %s\n", to_json(*c.member("raw")).c_str(), tag_of(got.language, m).c_str(),
                                    got.text.c_str(), to_json(*language).c_str(), c.member("text")->text.c_str());
                    }
                }
                std::printf("parse: %d of %d cases qwen-asr's language and text, %d of them a name the model's languages do not hold\n", same,
                            cases, others);
                ok = ok && cases > 0 && same == cases;
            }
            // A second of silence first, so that the times below leave out the compilation of a GPU's kernels.
            recognizer.recognize(std::vector<float>((size_t) recognizer.sample_rate(), 0.0f), {}, [](double) { return true; });
            int texts = 0, same_texts = 0;
            for (const auto & d : reference_dumps(args[2], m)) {
                const Npy embeds_dump = read_npy((d / "audio_embeds.npy").u8string());
                const Npy audio = read_npy((d / "audio.npy").u8string());
                std::printf("%s (%.2f s, %lld audio tokens)\n", d.filename().u8string().c_str(), (double) audio.f32.size() / recognizer.sample_rate(),
                            (long long) embeds_dump.shape[0]);
                for (const char * variant : kVariants) {
                    const std::filesystem::path v = d / variant;
                    const JsonValue meta = read_meta(v);
                    const RecognitionRequest request = request_of(meta, m);
                    const std::optional<size_t> language = language_named(m, meta.member("parsed_language")->text);
                    if (!meta.member("parsed_language")->text.empty() && !language) {
                        throw std::runtime_error(v.u8string() + ": qwen-asr parsed " + meta.member("parsed_language")->text + ", which the model does not name");
                    }
                    const std::vector<size_t> languages = language ? std::vector<size_t>{*language} : std::vector<size_t>{};
                    const auto npy = [&](const char * name) { return read_npy((v / (std::string(name) + ".npy")).u8string()); };
                    const Npy prompt_ids = npy("prompt_ids"), embeds = npy("embeds"), prefill_logits = npy("prefill_logits");
                    const Npy ids = npy("ids"), top_ids = npy("step_top_ids"), top_logits = npy("step_top_logits");
                    const std::string raw = dump_text(v / "raw.txt"), text = dump_text(v / "text.txt");
                    std::printf(" %s (%lld ids written)\n", variant, (long long) ids.shape[0]);

                    const PromptIds prompt = recognizer.prompt().ids(request.context, request.language, embeds_dump.shape[0]);
                    const bool same_prompt = prompt.ids == prompt_ids.i32;
                    std::printf("  %-32s %s (%zu ids)\n", "prompt ids", same_prompt ? "the dump's" : "DIFFER", prompt.ids.size());
                    ok = ok && same_prompt;
                    if (!same_prompt) continue;
                    const std::vector<float> ours = decoder.embeddings(prompt, embeds_dump.f32);
                    print_diff("  decoder input", compare(ours, embeds.f32));

                    // The logits of the prompt's last rows from the dump's input, a prefill ending at each.
                    const int64_t n = (int64_t) prompt.ids.size(), rows = prefill_logits.shape[0], vocab = prefill_logits.shape[1];
                    for (int64_t r = 0; r < rows; r++) {
                        exact.prefill(embeds.f32, n - rows + 1 + r, [](int64_t) { return true; });
                        const Diff diff = compare(exact.logits().data(), &prefill_logits.f32[(size_t) (r * vocab)], (size_t) vocab);
                        print_diff("  prefill logits, row " + std::to_string(n - rows + r), diff);
                        ok = ok && diff.snr_db > threshold_db;
                    }

                    // Teacher forcing on the dump's ids from the dump's input.
                    Steps forced;
                    exact.prefill(embeds.f32, n, [](int64_t) { return true; });
                    for (int64_t i = 0; i < ids.shape[0]; i++) {
                        forced.add((size_t) i, exact.logits(), top_ids, top_logits, ids.i32[i]);
                        if (i + 1 < ids.shape[0]) exact.step(ids.i32[i]);
                    }
                    forced.print("teacher-forced");
                    ok = ok && forced.unexplained == 0;

                    const std::string decoded = recognizer.tokenizer().decode(ids.i32);
                    const Parsed parsed = recognizer.transcript().parse(raw, request.language);
                    const bool same_parse = parsed.text == text && parsed.language == language && parsed.other_language.empty();
                    std::printf("  %-32s %s\n", "decoding of the dump's ids", decoded == raw ? "the dump's raw text" : "DIFFERS");
                    std::printf("  %-32s %s (%s)\n", "parse of the dump's raw text", same_parse ? "the dump's language and text" : "DIFFERS",
                                tag_of(parsed.language, m).c_str());
                    ok = ok && decoded == raw && same_parse;

                    // Greedy from the dump's projector output, the prompt and the embeddings ours.
                    Steps greedy;
                    followed.prefill(ours, n, [](int64_t) { return true; });
                    const std::vector<int32_t> written = follow(followed, ids, top_ids, top_logits, greedy);
                    decoder.prefill(ours, n, [](int64_t) { return true; });
                    const bool same_picks = decoder.generate([](size_t) { return true; })->ids == written;
                    std::printf("  %-32s %s\n", "picks on the device", same_picks ? "the first of the largest logits" : "DIFFER");
                    ok = ok && same_picks;
                    const std::string greedy_text = recognizer.transcript().parse(recognizer.tokenizer().decode(written), request.language).text;
                    greedy.print("greedy from the projector");
                    std::printf("  %-32s %s\n", "its ids and text", written == ids.i32 ? "the dump's" : greedy_text == text ? "other ids, the dump's text" : "DIFFER");
                    ok = ok && greedy.unexplained == 0 && (greedy.differ > 0 || written == ids.i32);

                    // The same decoding with drafts: its own text, whose every token a step should keep, and that text with
                    // the second half of its tokens reversed, whose steps must give back the rows they do not keep.
                    std::vector<int32_t> right = recognizer.tokenizer().encode(greedy_text), wrong = right;
                    std::reverse(wrong.begin() + (std::ptrdiff_t) (wrong.size() / 2), wrong.end());
                    for (const auto & [what, draft] : {std::pair<const char *, const std::vector<int32_t> &>{"drafted, its own text", right},
                                                       {"drafted, half reversed", wrong}}) {
                        followed.prefill(ours, n, [](int64_t) { return true; });
                        ok = drafted(followed, what, draft, recognizer, request.language.has_value(), ids, top_ids, top_logits, greedy, written) && ok;
                    }

                    // The text from the audio, every stage ours.
                    std::vector<PartReport> parts;
                    const auto start = Clock::now();
                    const Recognition found = recognizer.recognize(audio.f32, request, [](double) { return true; }, &parts);
                    const PartReport & stats = parts.at(0);
                    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
                    const bool same_text = found.text == text;
                    texts++;
                    same_texts += same_text;
                    std::printf("  %-32s %s in %.3f s%s\n", "text from the audio", same_text ? "the dump's" : "DIFFERS", seconds,
                                found.limited ? ", at the model's limit" : "");
                    // Where the text differs, the steps below tell the arithmetic from a defect at the first id that differs,
                    // the language's among them.
                    const bool same_languages = found.languages == languages;
                    std::printf("  %-32s %s (%s)\n", "language from the audio", same_languages ? "the dump's" : "DIFFERS",
                                found.languages.empty() ? "none" : tags_of(found.languages, m).c_str());
                    ok = ok && (same_languages || !same_text);
                    std::printf("    frontend %.3f s, encoder %.3f s, prefill of %lld rows %.3f s, %zu tokens in %.3f s (%.1f per second)\n",
                                stats.frontend, stats.encoder, (long long) stats.prompt_rows, stats.prefill, stats.ids.size(), stats.decode,
                                stats.ids.size() / stats.decode);
                    if (!same_text) {
                        std::printf("    got  %s\n    want %s\n", found.text.c_str(), text.c_str());
                        // The same stages one by one, to find the step where the decoding departs and whether the error
                        // explains it.
                        const std::vector<float> encoded = *recognizer.encoder().encode(recognizer.frontend().features(Frontend::normalize(audio.f32)));
                        followed.prefill(decoder.embeddings(prompt, encoded), n, [](int64_t) { return true; });
                        Steps from_audio;
                        follow(followed, ids, top_ids, top_logits, from_audio);
                        from_audio.print("greedy from the audio");
                        ok = ok && from_audio.unexplained == 0;
                    }
                }
            }
            std::printf("texts from the audio: %d of %d the dump's\n", same_texts, texts);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
