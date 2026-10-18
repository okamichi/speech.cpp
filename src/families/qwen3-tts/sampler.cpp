#include "sampler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "error.h"
#include "json.h"

namespace {

SamplingParams read_params(const ModelFile & m, const std::string & prefix) {
    SamplingParams p;
    p.greedy = !m.boolean(prefix + "do_sample");
    p.temperature = m.f32(prefix + "temperature");
    p.top_k = (int) m.u32(prefix + "top_k");
    p.top_p = m.f32(prefix + "top_p");
    p.repetition_penalty = m.f32(prefix + "repetition_penalty");
    return p;
}

/**
 * The logits with the repetition penalty on the tokens in `history` and the tokens in `banned` set to -inf, which a
 * greedy pick and a draw both choose from. A penalty that takes the largest logit beyond a float throws an Error naming
 * "repetition_penalty": a pick among infinite logits takes the first of them rather than the largest.
 */
std::vector<float> restricted(std::vector<float> logits, const SamplingParams & p, const std::vector<int32_t> & history,
                              const std::vector<bool> & banned) {
    const int n = (int) logits.size();
    if (p.repetition_penalty != 1.0f) {
        std::vector<bool> seen(n, false);
        for (int32_t t : history) {
            if (t < 0 || t >= n || seen[t]) continue;
            seen[t] = true;
            logits[t] = logits[t] < 0 ? logits[t] * p.repetition_penalty : logits[t] / p.repetition_penalty;
        }
    }
    for (int i = 0; i < n && i < (int) banned.size(); i++) {
        if (banned[i]) logits[i] = -INFINITY;
    }
    const float top = *std::max_element(logits.begin(), logits.end());
    if (top == INFINITY && p.repetition_penalty != 1.0f) {
        throw Error(Fault::OutOfRange,
                    "the repetition penalty " + json_number(p.repetition_penalty) + " pushes a logit beyond the range of a float; give a penalty nearer 1",
                    "repetition_penalty");
    }
    if (!std::isfinite(top)) throw std::runtime_error("no token has a finite logit to pick from");
    return logits;
}

}  // namespace

Generation::Generation(const ModelFile & m)
    : talker(read_params(m, "qwen3-tts.generation.talker.")),
      code_predictor(read_params(m, "qwen3-tts.generation.code_predictor.")),
      min_frames((int) m.u32("qwen3-tts.generation.min_frames")),
      max_frames((int) m.u32("qwen3-tts.generation.max_frames")),
      suppressed_tokens((int) m.u32("qwen3-tts.talker.suppressed_tokens")) {}

std::vector<bool> Generation::banned(int vocab, int32_t end_of_speech, int frames) const {
    std::vector<bool> b(vocab, false);
    for (int i = vocab - suppressed_tokens; i < vocab; i++) b[i] = i != end_of_speech;
    if (frames < min_frames) b[end_of_speech] = true;
    return b;
}

Candidates candidates(const std::vector<float> & raw, const SamplingParams & p, const std::vector<int32_t> & history,
                      const std::vector<bool> & banned) {
    std::vector<float> logits = restricted(raw, p, history, banned);
    const int n = (int) logits.size();
    for (float & l : logits) l /= p.temperature;
    Candidates c;
    c.tokens.resize(n);
    std::iota(c.tokens.begin(), c.tokens.end(), 0);
    const int k = p.top_k > 0 ? std::min(p.top_k, n) : n;
    std::partial_sort(c.tokens.begin(), c.tokens.begin() + k, c.tokens.end(),
                      [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    c.tokens.resize(k);

    const float max_logit = logits[c.tokens[0]];
    if (!std::isfinite(max_logit)) throw temperature_overflow(p.temperature);
    c.weights.resize(k);
    double sum = 0;
    for (int i = 0; i < k; i++) sum += c.weights[i] = std::exp((double) logits[c.tokens[i]] - max_logit);
    for (double & q : c.weights) q /= sum;
    if (p.top_p < 1.0f) {
        // Keep the smallest set whose probability reaches top_p, always at least one token. The weights are left
        // as they are rather than normalized again, since the draw normalizes them.
        double cumulative = 0;
        int keep = 0;
        while (keep < k) {
            cumulative += c.weights[keep++];
            if (cumulative >= p.top_p) break;
        }
        c.weights.resize(keep);
        c.tokens.resize(keep);
    }
    return c;
}

int32_t sample(const std::vector<float> & logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, double u) {
    if (p.greedy) {
        const std::vector<float> r = restricted(logits, p, history, banned);
        return (int32_t) (std::max_element(r.begin(), r.end()) - r.begin());
    }
    const Candidates c = candidates(logits, p, history, banned);
    return c.tokens[pick(c.weights, u)];
}

int32_t sample(const std::vector<float> & logits, const SamplingParams & p, const std::vector<int32_t> & history,
               const std::vector<bool> & banned, std::mt19937_64 & rng) {
    return sample(logits, p, history, banned, p.greedy ? 0.0 : uniform(rng));
}
