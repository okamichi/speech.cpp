#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "decoder.h"
#include "encoder.h"
#include "frontend.h"
#include "model-file.h"
#include "prompt.h"
#include "split.h"
#include "tokenizer.h"
#include "transcript.h"

namespace qwen3_asr {

/** What a recognition asks of the model besides its audio: the context and the language it forces, or none. */
struct RecognitionRequest {
    std::string context;
    /** An index of general.languages, or none to let the model write the language. */
    std::optional<size_t> language;
    /** The text the recognition is likely to write, whose tokens the decoding checks a few at a time, or none. */
    std::string draft;
};

/**
 * What a recognition found: its text; the languages of its parts, as indices of general.languages, merged as qwen-asr's
 * merge_languages() merges them: in the order of the audio, one for each run of parts of the same language, without the
 * parts of none; and whether the decoding reached the most tokens the model writes.
 */
struct Recognition {
    std::string text;
    std::vector<size_t> languages;
    bool limited = false;
};

/**
 * What the recognition of one part of the audio did, for the checks: where its time went in seconds, the rows of its
 * prompt, and the ids it wrote, the end token included, with whether they reached the most the model writes.
 */
struct PartReport {
    double frontend = 0, encoder = 0, prefill = 0, decode = 0;
    int64_t prompt_rows = 0;
    std::vector<int32_t> ids;
    bool limited = false;
    /** With a draft, the steps that checked its tokens, the tokens they checked and those kept, in the parts so far. */
    int draft_checks = 0, draft_checked = 0, draft_kept = 0;
};

/**
 * The type of the decoder's key/value cache, as the Qwen3-TTS talker's: a recognition of 1200 s and the 4096 tokens
 * after it take 2.2 GB in F16 rather than 4.4 in F32. On the CPU with F32 weights it puts the prompt's last logits
 * 42 to 56 dB from transformers' float32 where an F32 cache gives 98 to 111 dB (Apple M5, 2026-10-06), and every text
 * of the checks' dumps comes out the same.
 */
constexpr ggml_type kCacheType = GGML_TYPE_F16;

/**
 * Qwen3-ASR as qwen-asr's transcribe() runs it with the windowed encoder of transformers 5.18: the
 * audio normalized and, when it is too long for the model, split into parts; for each part its features, the encoder
 * and the projector, the prompt with the request's context and forced language, the decoder's prefill and greedy
 * decoding, and the text parsed from what it wrote; and the parts' texts joined. It reads one GGUF file and computes
 * on one backend, holding one part at a time.
 */
class Recognizer {
public:
    /** The model of the file `path` on `backend`, its decoder attending as `attention` says or as it chooses. */
    Recognizer(const std::string & path, ggml_backend_t backend, std::optional<Qwen3Attention> attention = std::nullopt);

    /**
     * Recognizes mono samples at sample_rate(). `progress` hears how far the work has come, from 0 to 1: of each part
     * an equal share, of which the encoder's windows, the prefill's blocks and the tokens written take a third each;
     * false stops the work, and what it returns is then not a recognition. A request whose prompt, beside a part's
     * audio and the most tokens the model writes, does not fit the decoder's positions is refused before any work, out
     * of range naming the option prompt. A part whose language is a name the file does not hold has none, which a
     * warning in the log reports. `parts`, when given, gets the report of each part.
     */
    Recognition recognize(const std::vector<float> & samples, const RecognitionRequest & request, const std::function<bool(double)> & progress,
                          std::vector<PartReport> * parts = nullptr);

    int sample_rate() const { return frontend_.sample_rate(); }
    const std::vector<std::string> & languages() const { return prompt_.languages(); }

    const ModelFile & model() const { return model_; }
    const Frontend & frontend() const { return frontend_; }
    Encoder & encoder() { return encoder_; }
    Decoder & decoder() { return decoder_; }
    const Tokenizer & tokenizer() const { return tokenizer_; }
    /** The token after which the model writes its text, where a draft begins. */
    int32_t asr_text() const { return asr_text_; }
    const Prompt & prompt() const { return prompt_; }
    const Transcript & transcript() const { return transcript_; }

    const Splitter & splitter() const { return splitter_; }

private:
    /**
     * Recognizes one part of normalized audio with its prompt and the language it forces, decoding with `draft` where
     * given, `progress` hearing the part's share from 0 to 1, and returns the parse of its output, or none when
     * `progress` stopped it.
     */
    std::optional<Parsed> recognize_part(const std::vector<float> & part, const PromptIds & prompt, std::optional<size_t> forced,
                                         const std::function<bool(double)> & progress, Draft * draft, PartReport & report);

    ModelFile model_;
    Frontend frontend_;
    Encoder encoder_;
    Decoder decoder_;
    Tokenizer tokenizer_;
    Prompt prompt_;
    /** The token after which the model writes its text, where a draft begins. */
    int32_t asr_text_;
    Transcript transcript_;
    Splitter splitter_;
};

}  // namespace qwen3_asr
