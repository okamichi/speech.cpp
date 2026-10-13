#include "graph-sampler.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

bool GraphSampler::follows(const SamplingParams & p) {
    return p.repetition_penalty == 1.0f && (p.greedy || std::isfinite(1.0f / p.temperature));
}

GraphSampler::GraphSampler(Graph & g, const SamplingParams & p, int vocab, int picks)
    : p_(p), vocab_(vocab), k_(p.top_k > 0 ? std::min(p.top_k, vocab) : vocab), picks_(picks) {
    if (!follows(p)) throw std::logic_error("a graph cannot pick under a repetition penalty or a temperature without a float reciprocal");
    // A draw from one candidate reads no number, and an input that no node reads has no memory to upload to.
    if (!p.greedy && k_ > 1) draws_ = g.input(std::vector<float>((size_t) picks, 0.0f), picks);
    out_ = ggml_new_tensor_2d(g.ctx(), GGML_TYPE_F32, picks, p.greedy ? 1 : 2);
    g.output(out_);
}

ggml_tensor * GraphSampler::pick(Graph & g, int i, ggml_tensor * logits) {
    ggml_context * ctx = g.ctx();
    const auto slot = [&](int row) { return ggml_view_1d(ctx, out_, 1, (size_t) (row * picks_ + i) * out_->nb[0]); };
    ggml_tensor * id;
    if (p_.greedy) {
        id = ggml_argmax(ctx, ggml_reshape_2d(ctx, logits, vocab_, 1));
    } else {
        ggml_tensor * order = ggml_argsort_top_k(ctx, ggml_reshape_2d(ctx, logits, vocab_, 1), k_);
        ggml_tensor * values = ggml_get_rows(ctx, ggml_reshape_2d(ctx, logits, 1, vocab_), order);
        // The softmax reads the logits less the largest, which no temperature takes to +inf: an infinite scaled logit
        // makes ggml's softmax assert that its sum is positive. read() refuses a temperature that takes the largest
        // logit itself beyond a float, as candidates() does.
        ggml_tensor * largest = ggml_view_1d(ctx, values, 1, 0);
        g.copy(largest, slot(1));
        ggml_tensor * below_largest = ggml_sub(ctx, ggml_reshape_1d(ctx, values, k_), largest);
        ggml_tensor * weights = ggml_soft_max_ext(ctx, below_largest, nullptr, 1.0f / p_.temperature, 0.0f);
        if (k_ == 1) {
            id = order;
        } else {
            // The first kept candidate whose cumulative weight passes the threshold is the count of those before the last
            // kept whose cumulative weight does not, as the cumulative weights do not fall: the last kept less those
            // before it that pass. Counted with sums of steps, it takes no comparison of indices, which ggml lacks.
            ggml_tensor * cumulative = ggml_cumsum(ctx, weights);
            ggml_tensor * before_last = ggml_view_1d(ctx, cumulative, k_ - 1, 0);
            ggml_tensor * u = ggml_view_1d(ctx, draws_, 1, (size_t) i * draws_->nb[0]);
            ggml_tensor * index;
            if (p_.top_p < 1.0f) {
                // The candidates before the first whose cumulative weight reaches top_p, which is the last kept.
                ggml_tensor * below = ggml_step(ctx, ggml_scale_bias(ctx, before_last, -1.0f, p_.top_p));
                ggml_tensor * last = ggml_sum(ctx, below);
                ggml_tensor * kept = ggml_get_rows(ctx, ggml_reshape_2d(ctx, cumulative, 1, k_), ggml_cast(ctx, last, GGML_TYPE_I32));
                ggml_tensor * threshold = ggml_mul(ctx, ggml_reshape_1d(ctx, kept, 1), u);
                ggml_tensor * passing = ggml_mul(ctx, ggml_step(ctx, ggml_sub(ctx, before_last, threshold)), below);
                index = ggml_sub(ctx, last, ggml_sum(ctx, passing));
            } else {
                ggml_tensor * threshold = ggml_mul(ctx, ggml_view_1d(ctx, cumulative, 1, (size_t) (k_ - 1) * cumulative->nb[0]), u);
                ggml_tensor * passing = ggml_step(ctx, ggml_sub(ctx, before_last, threshold));
                index = ggml_scale_bias(ctx, ggml_sum(ctx, passing), -1.0f, (float) (k_ - 1));
            }
            id = ggml_reshape_1d(ctx, ggml_get_rows(ctx, ggml_view_2d(ctx, order, 1, k_, order->nb[0], 0), ggml_cast(ctx, index, GGML_TYPE_I32)), 1);
        }
    }
    g.copy(id, slot(0));
    return id;
}

void GraphSampler::draw(Graph & g, const std::vector<double> & u) const {
    if (p_.greedy) {
        if (!u.empty()) throw std::logic_error("greedy picks were given numbers to draw with");
        return;
    }
    if ((int) u.size() != picks_) throw std::logic_error("picks in a graph were given another count of numbers to draw with");
    if (draws_) g.set(draws_, std::vector<float>(u.begin(), u.end()));
}

std::vector<int32_t> GraphSampler::read() const {
    const std::vector<float> out = Graph::read(out_);
    std::vector<int32_t> ids((size_t) picks_);
    for (int i = 0; i < picks_; i++) {
        if (!p_.greedy && !std::isfinite(out[(size_t) (picks_ + i)] / p_.temperature)) throw temperature_overflow(p_.temperature);
        ids[(size_t) i] = (int32_t) out[(size_t) i];
    }
    return ids;
}
