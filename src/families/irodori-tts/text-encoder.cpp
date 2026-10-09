#include "text-encoder.h"
#include "t5gemma-encoder.h"

#include <cmath>
#include <cstdlib>
#include <string>

namespace irodori {

/*
 * Activations are channel-first ([channels, tokens]), so a linear layer is one matrix product. ModernBERT
 * rotates the two halves of each head (GGML_ROPE_TYPE_NEOX), with a larger base in its global layers.
 */

TextEncoder::TextEncoder(const ModelFile & m) : m_(m) {
    hidden_ = (int) m.u32("irodori-tts.text.hidden_size");
    heads_ = (int) m.u32("irodori-tts.text.num_heads");
    layers_ = (int) m.u32("irodori-tts.text.num_layers");
    window_ = (int) m.u32("irodori-tts.text.window");
    dim_ = (int) m.u32("irodori-tts.text.dim");
    max_tokens_ = (int) m.u32("irodori-tts.text.max_tokens");
    eps_ = m.f32("irodori-tts.text.norm_eps");
    norm_eps_ = m.f32("irodori-tts.norm_eps");
    theta_global_ = m.f32("irodori-tts.text.rope_theta_global");
    theta_local_ = m.f32("irodori-tts.text.rope_theta_local");
    global_ = m.i32_array("irodori-tts.text.layer_global");
}

ggml_tensor * TextEncoder::build(Graph & g, const std::vector<int32_t> & ids, Condition condition, std::vector<ggml_tensor *> * layers) const {
    ggml_context * ctx = g.ctx();
    const int64_t n = (int64_t) ids.size();
    const int head_dim = hidden_ / heads_;
    auto layer_norm = [&](ggml_tensor * x, const std::string & w) { return ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(w)); };

    ggml_tensor * x;
    if (m_.str("irodori-tts.text.encoder") == "t5gemma2") {
        x = t5gemma_encode(m_, g, ids, layers);
    } else {
        std::vector<int32_t> positions(n);
        for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) i;
        ggml_tensor * pos = g.input(positions, n);
        std::vector<float> window(n * n);
        for (int64_t q = 0; q < n; q++)
            for (int64_t k = 0; k < n; k++) window[q * n + k] = std::llabs(q - k) <= window_ ? 0.0f : -INFINITY;
        ggml_tensor * local_mask = g.input(window, n, n);

        x = ggml_get_rows(ctx, m_.tensor("text.embd"), g.input(ids, n));
        x = layer_norm(x, "text.embd_norm");
        if (layers) layers->push_back(x);
        for (int l = 0; l < layers_; l++) {
            const std::string b = "text.blk." + std::to_string(l) + ".";
            const bool global = global_.at(l) != 0;
            ggml_tensor * h = l == 0 ? x : layer_norm(x, b + "attn_norm");
            ggml_tensor * q = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_q"), h), head_dim, heads_, n);
            ggml_tensor * k = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_k"), h), head_dim, heads_, n);
            ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_v"), h), head_dim, heads_, n);
            const float theta = global ? theta_global_ : theta_local_;
            q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
            ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
            ggml_tensor * kq = mul_mat(ctx, kh, qh);
            kq = ggml_soft_max_ext(ctx, kq, global ? nullptr : local_mask, 1.0f / std::sqrt((float) head_dim), 0.0f);
            ggml_tensor * o = mul_mat(ctx, vt, kq);
            o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), hidden_, n);
            x = ggml_add(ctx, x, mul_mat(ctx, m_.tensor(b + "attn_out"), o));

            h = layer_norm(x, b + "ffn_norm");
            ggml_tensor * act = ggml_gelu_erf(ctx, mul_mat(ctx, m_.tensor(b + "ffn_act"), h));
            h = ggml_mul(ctx, act, mul_mat(ctx, m_.tensor(b + "ffn_gate"), h));
            x = ggml_add(ctx, x, mul_mat(ctx, m_.tensor(b + "ffn_down"), h));
            if (layers && l + 1 < layers_) layers->push_back(x);
        }
        x = layer_norm(x, "text.final_norm");
        if (layers) layers->push_back(x);

    }

    auto linear = [&](ggml_tensor * in, const std::string & name) {
        return ggml_add(ctx, mul_mat(ctx, m_.tensor(name + ".weight"), in), m_.tensor(name + ".bias"));
    };
    const std::string c = condition == Condition::Text ? "text." : "caption.";
    ggml_tensor * p = linear(x, c + "proj");
    ggml_tensor * r = ggml_mul(ctx, ggml_rms_norm(ctx, x, norm_eps_), m_.tensor(c + "proj.res_norm"));
    r = ggml_silu(ctx, linear(r, c + "proj.res_up"));
    p = ggml_add(ctx, p, linear(r, c + "proj.res_down"));
    return ggml_mul(ctx, ggml_rms_norm(ctx, p, norm_eps_), m_.tensor(c + "norm"));
}

}  // namespace irodori
