#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "model-file.h"
#include "prompt.h"
#include "qwen3-decoder.h"

namespace qwen3_asr {

/** The ids a greedy decoding wrote, the end token included when one ended it, and whether it reached the limit. */
struct Generation {
    std::vector<int32_t> ids;
    bool limited = false;
};

/**
 * The tokens of a text a decoding is likely to write, and where it is in them. Its text begins after the token that
 * ends the language the model writes, or at once where the prompt names the language; from there the decoding is at the
 * draft's next token while it writes the draft's tokens, and after one it does not, at the token after the draft's
 * nearest two that it last wrote, or nowhere. A recognition of several parts goes on in the draft where the part before
 * left it.
 */
class Draft {
public:
    /** The draft `ids`, whose text begins after `marker`, or at once with `started`. */
    Draft(std::vector<int32_t> ids, int32_t marker, bool started) : ids_(std::move(ids)), marker_(marker), started_(started) {}
    /** Starts the next part, whose text begins after the marker, or at once with `started`. */
    void next_part(bool started) { started_ = started; }
    /** Takes the ids written so far, the last of them new. */
    void follow(const std::vector<int32_t> & written);
    /** The draft's tokens from where the decoding is, `count` at most. */
    std::vector<int32_t> ahead(size_t count) const;

    /** What the drafts of a decoding did: the steps that checked them, the tokens they checked and those kept. */
    int checks = 0, checked = 0, kept = 0;

private:
    static constexpr size_t kNowhere = SIZE_MAX;
    std::vector<int32_t> ids_;
    int32_t marker_;
    bool started_;
    size_t at_ = 0;
};

/**
 * Qwen3-ASR's decoder as transformers' Qwen3ASRForConditionalGeneration runs it under generate() with the checkpoint's
 * generation_config.json: the token embeddings of a prompt with the projector's output in the rows of the audio's
 * tokens, the shared Qwen3 stack (dec.*) with its output tied to the token embeddings, and greedy decoding, the first
 * of equal logits taken as torch.argmax() takes it, until one of qwen3-asr.generation.eos_ids or
 * qwen3-asr.generation.max_new_tokens tokens.
 */
/**
 * The draft's tokens one step checks at most. Qwen3-ASR 0.6B in Q8_0 on an Apple M5, live transcription's readings of
 * a 6.4 s Japanese and a 10.0 s German recording and the recognition of each whole, with the readings before as drafts:
 * 2, 4 and 7 tokens took the whole's decoding from 120 to 52, 140 to 41 and 141 to 35 ms (Japanese, 18 tokens) and
 * from 241 to 99, 233 to 68 and 249 to 70 ms (German, 34), and the readings' from 2.4 to 1.3, 2.8 to 1.1 and 2.7 to
 * 1.0 s and from 7.1 to 3.7, 7.1 to 2.4 and 7.9 to 2.2 s (2026-10-15); on the GPU a step of 5 rows costs little more
 * than a step of one, and more than 8 rows run another kernel.
 */
constexpr size_t kDraftTokens = 4;

class Decoder {
public:
    /**
     * The decoder of the model file `m`, which outlives it, computing on `backend` with a cache of `cache_type`, attending
     * as `attention` says or, without it, as qwen3_attention() chooses, and reading back each step's pick, as a
     * recognition does, or its logits, as a check that compares them does (Qwen3Readback::Pick or Logits).
     */
    Decoder(const ModelFile & m, ggml_backend_t backend, ggml_type cache_type, Qwen3Readback readback,
            std::optional<Qwen3Attention> attention = std::nullopt);
    ~Decoder();
    Decoder(const Decoder &) = delete;
    Decoder & operator=(const Decoder &) = delete;

    /**
     * The decoder's input for `prompt`: the token embeddings of its ids with `audio`, the projector's output
     * [audio tokens, hidden] row-major, in the rows of the audio's tokens; [ids, hidden] row-major.
     */
    std::vector<float> embeddings(const PromptIds & prompt, const std::vector<float> & audio);

    /**
     * Starts a sequence with the `n` rows of `embeds`, [n, hidden] row-major, which it runs through the stack a block of
     * kQwen3BlockRows rows at a time, leaving the logits of the last row. `keep_going` hears the rows done after each
     * block, and false stops the prefill there, which then returns false.
     */
    bool prefill(const std::vector<float> & embeds, int64_t n, const std::function<bool(int64_t rows)> & keep_going);

    /** Feeds the token `id` after the sequence, leaving the logits of the next. */
    void step(int32_t id);

    /** The logits of the next token, [vocabulary], where the decoder reads logits. */
    const std::vector<float> & logits() const { return stack_.logits(); }

    /**
     * Decodes greedily from the prefill's logits. `keep_going` hears the number of tokens written after each but the
     * last, and false stops the decoding there, which then returns none. With a `draft`, a step feeds the token written
     * and the draft's next kDraftTokens tokens, keeps those its picks agree with and the pick after them, and rewinds
     * the rest. `seen`, where given, gets the logits each token was picked from, of a decoder that reads logits.
     */
    std::optional<Generation> generate(const std::function<bool(size_t tokens)> & keep_going, Draft * draft = nullptr,
                                       std::vector<std::vector<float>> * seen = nullptr);

    /** The positions a sequence with a prompt of `n` rows takes: the prompt and the most tokens the model writes. */
    int64_t positions(int64_t n) const { return n + max_new_tokens_; }
    int64_t max_positions() const { return stack_.max_positions(); }
    int64_t max_new_tokens() const { return max_new_tokens_; }
    int hidden() const { return stack_.shape().hidden; }
    Qwen3Attention attention() const { return stack_.attention(); }

private:
    const ModelFile & m_;
    ggml_backend_t backend_;
    Qwen3Decoder stack_;
    int64_t max_new_tokens_;
    std::vector<int32_t> eos_;
    /** The allocator of the embedding graphs; the stack has its own. */
    ggml_gallocr_t allocr_ = nullptr;
};

}  // namespace qwen3_asr
