#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "error.h"
#include "model-file.h"

/** Sampling settings of one stack; the defaults leave the distribution as it is. */
struct SamplingParams {
    bool greedy = false;
    float temperature = 1.0f;
    /** 0 keeps every token. */
    int top_k = 0;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;

    bool operator==(const SamplingParams & o) const {
        return greedy == o.greedy && temperature == o.temperature && top_k == o.top_k && top_p == o.top_p &&
               repetition_penalty == o.repetition_penalty;
    }
};

/**
 * How the official generate() samples a model, read from its file: each stack's settings, the frames made before the
 * end of speech may come and at most, and the ids at the end of the talker's vocabulary that are never sampled, the
 * end of speech excepted.
 */
struct Generation {
    SamplingParams talker, code_predictor;
    int min_frames = 0, max_frames = 0;
    int suppressed_tokens = 0;

    explicit Generation(const ModelFile & m);

    /**
     * The talker's ids that the frame after `frames` frames may not take: the suppressed ones, and the end of speech
     * before min_frames.
     */
    std::vector<bool> banned(int vocab, int32_t end_of_speech, int frames) const;
};

/** The tokens a draw picks from, by falling logit, and a weight of each in proportion to its probability. */
struct Candidates {
    std::vector<int32_t> tokens;
    std::vector<double> weights;
};

/**
 * The tokens transformers' generate() draws from and their weights: the repetition penalty on the tokens in `history`,
 * the tokens in `banned` removed, then temperature, top-k and top-p, as its RepetitionPenalty, SuppressTokens,
 * MinNewTokensLength, Temperature, TopK and TopP processors do in that order. Top-k keeps exactly `top_k` tokens where
 * transformers keeps every token tied with the k-th. A penalty or a temperature that pushes the largest logit beyond a
 * float throws an Error of its kind naming "repetition_penalty" or "temperature".
 */
Candidates candidates(const std::vector<float> & logits, const SamplingParams & p, const std::vector<int32_t> & history,
                      const std::vector<bool> & banned);

/**
 * A number in [0, 1] from the next 64 bits of `rng`, rounded to a double; the largest draws round to 1. It is defined
 * here rather than by a distribution of the standard library, whose results differ between libc++, libstdc++ and
 * Microsoft's, so that a seed draws the same numbers on every platform.
 */
double uniform(std::mt19937_64 & rng);

/**
 * The index of the first weight whose cumulative share of the sum passes `u`, the last where none does (u of 1):
 * the shares are the weights divided by their sum, accumulated in order in double.
 */
int pick(const std::vector<double> & weights, double u);

/**
 * Picks a token the way transformers' generate() does: the argmax of the logits with the repetition penalty on the
 * tokens in `history` and the tokens in `banned` removed, or a draw from candidates(), which pick()s with `u` from the
 * candidates' weights; a greedy pick leaves `u` unused. A penalty that pushes the largest logit beyond a float throws an
 * Error naming "repetition_penalty" in either case.
 */
int32_t sample(const std::vector<float> & logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, double u);

/** sample() with the next uniform() of `rng` where `p` draws, and no number of it where `p` is greedy. */
int32_t sample(const std::vector<float> & logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, std::mt19937_64 & rng);

/** The Error of a temperature that pushes the largest logit beyond the range of a float, naming "temperature". */
Error temperature_overflow(float temperature);
