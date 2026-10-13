#include "tdt.h"

#include <memory>
#include <stdexcept>

#include "error.h"
#include "decoder-step.h"

namespace fastconformer {

namespace {

using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

Allocator new_allocator(ggml_backend_t backend) {
    Allocator a(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), &ggml_gallocr_free);
    if (!a) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    return a;
}

}  // namespace

TdtDecoder::TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      durations_(m.i32_array("fastconformer.decoder.tdt.durations")),
      max_symbols_(m.u32("fastconformer.decoder.tdt.max_symbols")) {}

Decoding TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const {
    const int hidden = joint_.hidden();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    auto frame = [&](int64_t t) {
        return std::vector<float>(projected.begin() + t * hidden, projected.begin() + (t + 1) * hidden);
    };
    // The step's graph and the joint's alone keep an allocator each, so that neither is planned again when the
    // other runs.
    DecoderStep step(prediction_, joint_, backend, 1);
    const Allocator joint_allocator = new_allocator(backend);

    Decoding d;
    PredictionState state = prediction_.initial_state();
    int32_t label = blank_;
    int64_t t = 0, last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    while (t < frames) {
        if (progress && !progress((double) t / (double) frames)) break;
        // The prediction network takes the last label, and the joint is evaluated at the current frame with its
        // output, in one graph.
        step.compute(label, state, frame(t));
        d.graphs++;
        state = step.state();
        const std::vector<float> prediction = step.prediction();
        std::vector<float> out;
        step.logits(out);

        int64_t label_frame;
        int32_t duration;
        for (;;) {
            label = first_argmax(out.data(), blank_ + 1);
            duration = durations_[(size_t) first_argmax(out.data() + blank_ + 1, (int) durations_.size())];
            if (label == blank_ && duration == 0) duration = 1;
            label_frame = t;
            t += duration;
            if (label != blank_ || t >= frames) break;
            Graph j(64);
            ggml_tensor * next = joint_.build(j.ctx(), j.input(frame(t), hidden), j.input(prediction, hidden));
            j.output(next);
            j.compute(backend, joint_allocator.get());
            d.graphs++;
            out = Graph::read(next);
        }
        if (label == blank_) break;
        d.ids.push_back(label);
        d.frames.push_back(label_frame);
        d.durations.push_back(duration);
        tokens_on_frame = label_frame == last_token_frame ? tokens_on_frame + 1 : 1;
        last_token_frame = label_frame;
        if (t < frames && t == label_frame && tokens_on_frame >= max_symbols_) t++;
    }
    return d;
}

}  // namespace fastconformer
