#pragma once

#include <cstdint>
#include <vector>

#include "model-file.h"
#include "sampling.h"

namespace drama3 {

/** The settings of one draw, as fish-speech's sample_top_k_top_p() takes them. */
struct Sampling {
    float temperature = 1.0f;
    /** At least 1. */
    int top_k = 1;
    float top_p = 1.0f;
};

/** The settings a request samples with when it sets none, the model card's, which the file holds. */
Sampling file_sampling(const ModelFile & m);

/**
 * The tokens sample_top_k_top_p() draws from and their weights: the logits divided by the temperature, or by 1e-5 where
 * it is smaller, by falling logit, and of them the first, and each other within the first top_k whose probability,
 * accumulated in that order over the softmax of every logit, is at most top_p, so that the token that passes top_p is
 * dropped, where transformers' top-p keeps it. The weights are the probabilities, which the draw normalizes over the
 * tokens kept. sample_top_k_top_p() computes them in float32, whose sum of every probability passes 1 by a few units of
 * rounding, so that a top_p of 1 drops the least likely tokens there and not here; a token whose share is that close to
 * top_p may be kept by one and dropped by the other. A logit that is not a finite number throws an Error of the device,
 * and a temperature that takes the largest logit beyond a float temperature_overflow().
 */
Candidates candidates(const std::vector<float> & logits, const Sampling & s);

/**
 * How Drama 3 draws the main token and the codes of a frame: with a request's settings, and with the settings of the
 * file's redraw for a main token that repeats one of the last main tokens, as the official audio decoder draws them.
 */
class FrameSampler {
public:
    FrameSampler(const ModelFile & m, const Sampling & request);

    /**
     * The main token of the frame after a row, an id of the text model's vocabulary, from `logits` of the file's
     * output.token_ids, the end id and the semantic ids: drawn with the request's settings by `first`, and, where that
     * draw is a semantic id among the last repeat.window ids of `history`, the main tokens of the rows so far, prompt
     * rows included, drawn again with the redraw's settings by `again`. Both numbers come from uniform(), and the
     * official sampler draws both for every frame.
     */
    int32_t main(const std::vector<float> & logits, const std::vector<int32_t> & history, double first, double again) const;

    /** A code of a frame's residual codebooks from its head's `logits`, drawn with the request's settings by `u`. */
    int32_t code(const std::vector<float> & logits, double u) const;

    bool semantic(int32_t id) const { return id >= semantic_begin_ && id <= semantic_end_; }

private:
    Sampling request_, repeat_;
    int window_ = 0;
    int32_t semantic_begin_ = 0, semantic_end_ = 0;
    std::vector<int32_t> token_ids_;
};

}  // namespace drama3
