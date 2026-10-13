#include "transducer.h"

#include <string>

namespace fastconformer {

namespace {

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

}  // namespace

bool decoding_tensor(const std::string & name) {
    for (const char * prefix : {"pred.", "joint.pred.", "joint.out."}) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

bool quantized_decoding(const ModelFile & m) {
    std::vector<std::string> names = {"joint.pred.weight", "joint.out.weight"};
    for (uint32_t l = 0; l < m.u32("fastconformer.decoder.prediction_layers"); l++) {
        for (const char * part : {".ih.weight", ".hh.weight"}) names.push_back("pred.lstm." + std::to_string(l) + part);
    }
    for (const std::string & name : names) {
        if (!ggml_is_quantized(m.stored_type(name))) return false;
    }
    return true;
}

PredictionNetwork::PredictionNetwork(const ModelFile & m)
    : m_(m), layers_((int) m.u32("fastconformer.decoder.prediction_layers")), hidden_((int) m.tensor("pred.embed.weight")->ne[0]) {}

PredictionState PredictionNetwork::initial_state() const {
    const size_t n = (size_t) hidden_ * (size_t) layers_;
    return {std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)};
}

/**
 * torch.nn.LSTM's step: gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output;
 * c' = sigmoid(f) c + sigmoid(i) tanh(g) and h' = sigmoid(o) tanh(c'). The converter sums b_ih and b_hh.
 */
PredictionNetwork::Step PredictionNetwork::build(Graph & g, int32_t label, const PredictionState & state) const {
    return build(g, g.input(std::vector<int32_t>{label}, 1), g.input(state.h, hidden_, layers_), g.input(state.c, hidden_, layers_));
}

PredictionNetwork::Step PredictionNetwork::build(Graph & g, ggml_tensor * label, ggml_tensor * h_in, ggml_tensor * c_in) const {
    ggml_context * ctx = g.ctx();
    const size_t width = (size_t) hidden_ * sizeof(float);
    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), label);
    ggml_tensor * h_out = nullptr;
    ggml_tensor * c_out = nullptr;
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.lstm." + std::to_string(l);
        ggml_tensor * h = ggml_view_1d(ctx, h_in, hidden_, (size_t) l * width);
        ggml_tensor * c = ggml_view_1d(ctx, c_in, hidden_, (size_t) l * width);
        ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, mul_mat(ctx, m_.tensor(p + ".ih.weight"), x), mul_mat(ctx, m_.tensor(p + ".hh.weight"), h)),
                                       m_.tensor(p + ".bias"));
        auto gate = [&](int i) { return ggml_view_1d(ctx, gates, hidden_, (size_t) i * width); };
        ggml_tensor * next_c = ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c), ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
        ggml_tensor * next_h = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, next_c));
        h_out = h_out ? ggml_concat(ctx, h_out, next_h, 0) : next_h;
        c_out = c_out ? ggml_concat(ctx, c_out, next_c, 0) : next_c;
        x = next_h;
    }
    return {x, h_out, c_out};
}

PredictionState PredictionNetwork::read_state(const Step & step) {
    return {Graph::read(step.h), Graph::read(step.c)};
}

Joint::Joint(const ModelFile & m)
    : m_(m), hidden_((int) m.tensor("joint.out.weight")->ne[0]), outputs_((int) m.tensor("joint.out.weight")->ne[1]) {}

ggml_tensor * Joint::project_encoder(ggml_context * ctx, ggml_tensor * encoded) const {
    return linear(ctx, m_, encoded, "joint.enc");
}

ggml_tensor * Joint::project_prediction(ggml_context * ctx, ggml_tensor * prediction) const {
    return linear(ctx, m_, prediction, "joint.pred");
}

ggml_tensor * Joint::build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const {
    return linear(ctx, m_, ggml_relu(ctx, ggml_add(ctx, f, g)), "joint.out");
}

}  // namespace fastconformer
