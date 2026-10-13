#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "sampler.h"

/**
 * sample()'s picks built into a graph, so that a step that reads the id a pick gives finds it on the device. A greedy
 * pick is the argmax of the logits. A draw takes the candidates of candidates() in float: the top-k of the logits by
 * falling logit, which the temperature leaves in that order since it is positive, their weights the softmax of the k
 * logits under the temperature, cumulated in order, and with top-p the smallest prefix whose cumulative weight reaches
 * it, at least one. It picks as pick() does, with the number uniform() gave rounded to a float: the first kept candidate
 * whose cumulative weight passes the number times the weight kept, and the last kept where none does. candidates()
 * computes the weights in double, so a draw within rounding of the boundary between two candidates may pick the other.
 */
class GraphSampler {
public:
    /**
     * Whether picks in a graph follow `p`: a repetition penalty of 1, and a temperature whose reciprocal, the scale of the
     * softmax, is a float. A penalty would need the logits of the ids picked before changed on the device, and the
     * largest of them read back to refuse one that leaves the range of a float; Qwen3-TTS's code predictor, the one
     * stack that picks in a graph, takes the penalty of the model file, which no request sets and both official
     * checkpoints give as 1.
     */
    static bool follows(const SamplingParams & p);

    /** `picks` picks under `p` from logits of `vocab` tokens, built into `g`. A `p` it does not follow throws. */
    GraphSampler(Graph & g, const SamplingParams & p, int vocab, int picks);

    /** The settings it picks under. */
    const SamplingParams & params() const { return p_; }

    /** Builds pick `i` into `g` from `logits`, [vocab]: the id it picks, I32 [1]. */
    ggml_tensor * pick(Graph & g, int i, ggml_tensor * logits);

    /** Gives the picks of `g` the numbers `u` they draw with, one each from uniform(), and none for greedy picks. */
    void draw(Graph & g, const std::vector<double> & u) const;

    /**
     * The ids the picks of the computed graph gave. Where the temperature takes a draw's largest logit beyond a float, it
     * throws temperature_overflow(), as candidates() does.
     */
    std::vector<int32_t> read() const;

private:
    SamplingParams p_;
    int vocab_ = 0, k_ = 0, picks_ = 0;
    /** The numbers of the draws, F32 [picks]. */
    ggml_tensor * draws_ = nullptr;
    /**
     * What a compute reads back in one copy, F32 [picks, 1] greedy and [picks, 2] drawing: the ids picked, and the
     * largest logit of each draw, which read() divides by the temperature as candidates() does.
     */
    ggml_tensor * out_ = nullptr;
};
