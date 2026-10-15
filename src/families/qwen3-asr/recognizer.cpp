#include "recognizer.h"

#include <chrono>
#include <cstddef>
#include <utility>

#include "error.h"
#include "layout.h"
#include "log.h"

namespace qwen3_asr {

namespace {

/** Adds the seconds from its construction to its destruction to `total`, when `total` is set. */
struct Timer {
    double * total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~Timer() {
        if (total) *total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
};

}  // namespace

Recognizer::Recognizer(const std::string & path, ggml_backend_t backend, std::optional<Qwen3Attention> attention)
    : model_(path, backend, layout),
      frontend_(model_),
      encoder_(model_, backend),
      decoder_(model_, backend, kCacheType, Qwen3Readback::Pick, attention),
      tokenizer_(model_),
      prompt_(model_, tokenizer_),
      asr_text_(tokenizer_.encode(model_.str("qwen3-asr.prompt.asr_text")).back()),
      transcript_(model_),
      splitter_(model_) {}

Recognition Recognizer::recognize(const std::vector<float> & samples, const RecognitionRequest & request,
                                  const std::function<bool(double)> & progress, std::vector<PartReport> * parts) {
    const std::vector<float> audio = Frontend::normalize(samples);
    const std::vector<int64_t> bounds = splitter_.bounds(audio);
    const size_t n_parts = bounds.size() - 1;
    // Every part's prompt is made, and checked against the decoder's positions, before any work.
    std::vector<PromptIds> prompts;
    for (size_t k = 0; k < n_parts; k++) {
        const int64_t audio_tokens = encoder_.tokens(frontend_.frames((size_t) (bounds[k + 1] - bounds[k])));
        prompts.push_back(prompt_.ids(request.context, request.language, audio_tokens));
        const int64_t n = (int64_t) prompts.back().ids.size();
        if (decoder_.positions(n) > decoder_.max_positions()) {
            throw Error(Fault::OutOfRange,
                        "the prompt takes " + std::to_string(n - audio_tokens) + " tokens, which with the audio's " + std::to_string(audio_tokens) +
                            " and the " + std::to_string(decoder_.max_new_tokens()) + " the model may write leave more than its " +
                            std::to_string(decoder_.max_positions()) + " positions; shorten it",
                        "prompt");
        }
    }
    std::optional<Draft> draft;
    if (!request.draft.empty()) draft.emplace(naming("draft", [&] { return tokenizer_.encode(request.draft); }), asr_text_, false);
    Recognition out;
    for (size_t k = 0; k < n_parts; k++) {
        const std::vector<float> part(audio.begin() + (std::ptrdiff_t) bounds[k], audio.begin() + (std::ptrdiff_t) bounds[k + 1]);
        PartReport report;
        // A forced language ends the prompt with the token after which the text begins.
        if (draft) draft->next_part(request.language.has_value());
        const std::optional<Parsed> parsed = recognize_part(part, prompts[k], request.language,
                                                            [&](double done) { return progress((k + done) / n_parts); },
                                                            draft ? &*draft : nullptr, report);
        if (!parsed) return {};
        // qwen-asr joins the texts of the parts without a separator.
        out.text += parsed->text;
        if (parsed->language && (out.languages.empty() || out.languages.back() != *parsed->language)) out.languages.push_back(*parsed->language);
        if (!parsed->other_language.empty()) {
            log_message(LogLevel::Warn, "Qwen3-ASR wrote the language \"" + parsed->other_language +
                                            "\", which is none of the languages its file names; the result leaves it out");
        }
        out.limited = out.limited || report.limited;
        if (parts) parts->push_back(std::move(report));
    }
    return out;
}

std::optional<Parsed> Recognizer::recognize_part(const std::vector<float> & part, const PromptIds & prompt, std::optional<size_t> forced,
                                                 const std::function<bool(double)> & progress, Draft * draft, PartReport & report) {
    const int64_t n = (int64_t) prompt.ids.size();
    std::vector<float> embeds;
    {
        std::vector<float> features;
        {
            Timer t{&report.frontend};
            features = frontend_.features(part);
        }
        const size_t windows = encoder_.windows((int64_t) features.size() / frontend_.mels()).size();
        std::optional<std::vector<float>> encoded;
        {
            Timer t{&report.encoder};
            encoded = encoder_.encode(features, [&](size_t done) { return progress((double) done / windows / 3); });
        }
        if (!encoded) return std::nullopt;
        Timer t{&report.prefill};
        embeds = decoder_.embeddings(prompt, *encoded);
    }
    report.prompt_rows = n;
    {
        Timer t{&report.prefill};
        if (!decoder_.prefill(embeds, n, [&](int64_t rows) { return progress((1 + (double) rows / n) / 3); })) return std::nullopt;
    }
    embeds.clear();
    embeds.shrink_to_fit();
    std::optional<Generation> generation;
    {
        Timer t{&report.decode};
        generation = decoder_.generate([&](size_t tokens) { return progress((2 + (double) tokens / decoder_.max_new_tokens()) / 3); }, draft);
    }
    if (!generation) return std::nullopt;
    report.ids = generation->ids;
    if (draft) {
        report.draft_checks = draft->checks;
        report.draft_checked = draft->checked;
        report.draft_kept = draft->kept;
    }
    report.limited = generation->limited;
    return transcript_.parse(tokenizer_.decode(generation->ids), forced);
}

}  // namespace qwen3_asr
