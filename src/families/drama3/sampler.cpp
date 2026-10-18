#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "error.h"

namespace drama3 {

namespace {

/** The temperature below which sample_top_k_top_p() takes 1e-5, so that a temperature of 0 divides by no zero. */
constexpr float kMinTemperature = 1e-5f;

Sampling read_sampling(const ModelFile & m, const std::string & prefix) {
    Sampling s;
    s.temperature = m.f32(prefix + "temperature");
    s.top_k = (int) m.u32(prefix + "top_k");
    s.top_p = m.f32(prefix + "top_p");
    return s;
}

}  // namespace

Sampling file_sampling(const ModelFile & m) {
    return read_sampling(m, "drama3.sampling.");
}

Candidates candidates(const std::vector<float> & logits, const Sampling & s) {
    const float t = std::max(s.temperature, kMinTemperature);
    const int n = (int) logits.size();
    std::vector<float> scaled((size_t) n);
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(logits[(size_t) i])) {
            throw Error(Fault::Device, "the device computed logits that are not finite numbers; run the model on another device or with another weight type");
        }
        scaled[(size_t) i] = logits[(size_t) i] / t;
    }
    const int k = std::min(std::max(s.top_k, 1), n);
    std::vector<int32_t> order((size_t) n);
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int32_t a, int32_t b) {
        return scaled[(size_t) a] > scaled[(size_t) b] || (scaled[(size_t) a] == scaled[(size_t) b] && a < b);
    });
    const float top = scaled[(size_t) order[0]];
    if (!std::isfinite(top)) throw temperature_overflow(s.temperature);
    double sum = 0;
    for (float v : scaled) sum += std::exp((double) v - top);
    Candidates c;
    double cumulative = 0;
    for (int r = 0; r < k; r++) {
        const double p = std::exp((double) scaled[(size_t) order[(size_t) r]] - top);
        cumulative += p / sum;
        if (r > 0 && cumulative > s.top_p) break;
        c.tokens.push_back(order[(size_t) r]);
        c.weights.push_back(p);
    }
    return c;
}

FrameSampler::FrameSampler(const ModelFile & m, const Sampling & request)
    : request_(request),
      repeat_(read_sampling(m, "drama3.repeat.")),
      window_((int) m.u32("drama3.repeat.window")),
      semantic_begin_((int32_t) m.u32("drama3.semantic_start_token_id")),
      semantic_end_((int32_t) m.u32("drama3.semantic_end_token_id")),
      token_ids_(m.i32_array("drama3.output.token_ids")) {}

int32_t FrameSampler::main(const std::vector<float> & logits, const std::vector<int32_t> & history, double first, double again) const {
    const Candidates drawn = candidates(logits, request_);
    const int32_t id = token_ids_[(size_t) drawn.tokens[(size_t) pick(drawn.weights, first)]];
    const Candidates redrawn = candidates(logits, repeat_);
    const int32_t other = token_ids_[(size_t) redrawn.tokens[(size_t) pick(redrawn.weights, again)]];
    // A semantic id alone is drawn again: the window begins with prompt rows, whose text holds the end id of the user's
    // turn, and an end of speech is never drawn again.
    const auto recent = history.end() - std::min<ptrdiff_t>(window_, (ptrdiff_t) history.size());
    return semantic(id) && std::find(recent, history.end(), id) != history.end() ? other : id;
}

int32_t FrameSampler::code(const std::vector<float> & logits, double u) const {
    const Candidates c = candidates(logits, request_);
    return c.tokens[(size_t) pick(c.weights, u)];
}

}  // namespace drama3
