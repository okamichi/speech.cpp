#include "rnnt-greedy.h"

#include <algorithm>
#include <memory>
#include <map>

#include "decoder-step.h"

namespace fastconformer {

namespace {

/**
 * The frames of the first run after a token, and the longest run. Each graph costs a round trip to the device, and each
 * frame of a run past the token it finds an evaluation of the joint for nothing. In reazonspeech-nemo-v2's greedy
 * decoding of the 311 s input of the checks (3,891 frames, 931 tokens, 890 of them within 7 frames of the one before),
 * first runs of 8 frames take 1,008 graphs and 9,604 evaluations of the joint; first runs of 4 take 1,178 and 6,888,
 * of 16 take 975 and 16,908, and of 1 take 2,398 and 6,513 (2026-10-07). The longest run changes the graphs of
 * silence alone.
 */
constexpr int64_t kFirstRun = 8;
constexpr int64_t kLongestRun = 64;

}  // namespace

RnntGreedyDecoder::RnntGreedyDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      max_symbols_(m.u32("fastconformer.decoder.rnnt.max_symbols")) {}

Decoding RnntGreedyDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const {
    const int hidden = joint_.hidden(), outputs = joint_.outputs();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    std::map<int64_t, std::unique_ptr<DecoderStep>> runs;

    Decoding d;
    PredictionState state = prediction_.initial_state();
    int32_t label = blank_;
    int64_t t = 0, run = kFirstRun, last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    std::vector<float> logits;
    while (t < frames) {
        if (progress && !progress((double) t / (double) frames)) break;
        // The prediction for the last label and the joint at frames t to t + n - 1 with it, [outputs, n].
        const int64_t n = std::min(run, frames - t);
        auto & step = runs[n];
        if (!step) step = std::make_unique<DecoderStep>(prediction_, joint_, backend, n);
        const std::vector<float> f(projected.begin() + t * hidden, projected.begin() + (t + n) * hidden);
        step->compute(label, state, f);
        d.graphs++;
        step->logits(logits);

        int64_t found = 0;
        int token = blank_;
        for (; found < n; found++) {
            token = first_argmax(&logits[(size_t) (found * outputs)], outputs);
            if (token != blank_) break;
        }
        if (token == blank_) {
            t += n;
            run = std::min(2 * run, kLongestRun);
            continue;
        }
        t += found;
        d.ids.push_back(token);
        d.frames.push_back(t);
        state = step->state();
        label = token;
        tokens_on_frame = t == last_token_frame ? tokens_on_frame + 1 : 1;
        last_token_frame = t;
        if (tokens_on_frame >= max_symbols_) t++;
        run = kFirstRun;
    }
    return d;
}

}  // namespace fastconformer
