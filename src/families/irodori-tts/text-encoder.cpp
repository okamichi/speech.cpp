#include "text-encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace irodori {

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

/*
 * Activations are channel-first ([channels, tokens]), so a linear layer is one matrix product. Both backbones rotate
 * the two halves of each head (GGML_ROPE_TYPE_NEOX), with another base in their global layers.
 */

TextEncoder::TextEncoder(const ModelFile & m) : m_(m) {
    gemma_ = m.one_of("irodori-tts.text.backbone", {"modernbert", "t5gemma2"}) == "t5gemma2";
    hidden_ = (int) m.u32("irodori-tts.text.hidden_size");
    heads_ = (int) m.u32("irodori-tts.text.num_heads");
    layers_ = (int) m.u32("irodori-tts.text.num_layers");
    dim_ = (int) m.u32("irodori-tts.text.dim");
    max_tokens_ = (int) m.u32("irodori-tts.text.max_tokens");
    eps_ = m.f32("irodori-tts.text.norm_eps");
    norm_eps_ = m.f32("irodori-tts.norm_eps");
    theta_global_ = m.f32("irodori-tts.text.rope_theta_global");
    theta_local_ = m.f32("irodori-tts.text.rope_theta_local");
    global_ = m.i32_array("irodori-tts.text.layer_global");
    if (gemma_) {
        kv_heads_ = (int) m.u32("irodori-tts.text.num_kv_heads");
        head_dim_ = (int) m.u32("irodori-tts.text.head_dim");
        before_ = (int) m.u32("irodori-tts.text.window_before");
        after_ = (int) m.u32("irodori-tts.text.window_after");
        eoi_ = (int32_t) m.u32("irodori-tts.text.eoi_id");
        scale_global_ = 1.0f / m.f32("irodori-tts.text.rope_scale_global");
        query_scale_ = 1.0f / std::sqrt(m.f32("irodori-tts.text.query_pre_attn_scalar"));
    } else {
        window_ = (int) m.u32("irodori-tts.text.window");
        head_dim_ = hidden_ / heads_;
    }
}

ggml_tensor * TextEncoder::build(Graph & g, const std::vector<int32_t> & ids, Condition condition, std::vector<ggml_tensor *> * layers) const {
    ggml_context * ctx = g.ctx();
    ggml_tensor * x = gemma_ ? t5gemma2(g, ids, layers) : modernbert(g, ids, layers);
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

ggml_tensor * TextEncoder::modernbert(Graph & g, const std::vector<int32_t> & ids, std::vector<ggml_tensor *> * layers) const {
    ggml_context * ctx = g.ctx();
    const int64_t n = (int64_t) ids.size();
    auto layer_norm = [&](ggml_tensor * x, const std::string & w) { return ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(w)); };

    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, n);
    std::vector<float> window(n * n);
    for (int64_t q = 0; q < n; q++)
        for (int64_t k = 0; k < n; k++) window[q * n + k] = std::llabs(q - k) <= window_ ? 0.0f : -INFINITY;
    ggml_tensor * local_mask = g.input(window, n, n);

    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("text.embd"), g.input(ids, n));
    x = layer_norm(x, "text.embd_norm");
    if (layers) layers->push_back(x);
    for (int l = 0; l < layers_; l++) {
        const std::string b = "text.blk." + std::to_string(l) + ".";
        const bool global = global_.at(l) != 0;
        ggml_tensor * h = l == 0 ? x : layer_norm(x, b + "attn_norm");
        ggml_tensor * q = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_q"), h), head_dim_, heads_, n);
        ggml_tensor * k = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_k"), h), head_dim_, heads_, n);
        ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_v"), h), head_dim_, heads_, n);
        const float theta = global ? theta_global_ : theta_local_;
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kq = mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, global ? nullptr : local_mask, 1.0f / std::sqrt((float) head_dim_), 0.0f);
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
    return x;
}

/*
 * T5Gemma 2's encoder is Gemma 3 attending both ways: embeddings scaled by the square root of their width, a norm before
 * and after the attention and the feed-forward layer of each block, a norm of each head's queries and keys, and a key
 * and value head that every query head shares. Its RMS norms multiply by 1 + weight, which the converter stores as the
 * weight.
 */
ggml_tensor * TextEncoder::t5gemma2(Graph & g, const std::vector<int32_t> & ids, std::vector<ggml_tensor *> * layers) const {
    ggml_context * ctx = g.ctx();
    const int64_t n = (int64_t) ids.size();
    auto rms_norm = [&](ggml_tensor * x, const std::string & w) { return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps_), m_.tensor(w)); };

    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, n);
    std::vector<float> window(n * n);
    for (int64_t q = 0; q < n; q++)
        for (int64_t k = 0; k < n; k++) window[q * n + k] = q - k <= before_ && k - q <= after_ ? 0.0f : -INFINITY;
    ggml_tensor * sliding_mask = g.input(window, n, n);

    ggml_tensor * x = ggml_get_rows(ctx, m_.tensor("text.embd"), g.input(ids, n));
    x = ggml_scale(ctx, x, std::sqrt((float) hidden_));
    // The token that ends an image takes an embedding of its own, unscaled.
    std::vector<float> eoi(n);
    for (int64_t i = 0; i < n; i++) eoi[i] = ids[i] == eoi_ ? 1.0f : 0.0f;
    if (std::find(eoi.begin(), eoi.end(), 1.0f) != eoi.end()) {
        ggml_tensor * is_eoi = g.input(eoi, 1, n);
        ggml_tensor * kept = ggml_mul(ctx, x, ggml_scale_bias(ctx, is_eoi, -1.0f, 1.0f));
        x = ggml_add(ctx, kept, ggml_mul(ctx, ggml_repeat(ctx, ggml_reshape_2d(ctx, m_.tensor("text.embd_eoi"), hidden_, 1), x), is_eoi));
    }
    if (layers) layers->push_back(x);
    // gelu_pytorch_tanh, from operations every backend computes in float32: the CPU's ggml_gelu() reads a table of
    // half-precision values.
    auto gelu_tanh = [&](ggml_tensor * a) {
        ggml_tensor * cube = ggml_mul(ctx, ggml_mul(ctx, a, a), a);
        ggml_tensor * inner = ggml_scale(ctx, ggml_add(ctx, a, ggml_scale(ctx, cube, 0.044715f)), (float) std::sqrt(2.0 / kPi));
        return ggml_mul(ctx, a, ggml_scale_bias(ctx, ggml_tanh(ctx, inner), 0.5f, 0.5f));
    };
    for (int l = 0; l < layers_; l++) {
        const std::string b = "text.blk." + std::to_string(l) + ".";
        const bool global = global_.at(l) != 0;
        ggml_tensor * h = rms_norm(x, b + "attn_norm");
        ggml_tensor * q = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_q"), h), head_dim_, heads_, n);
        ggml_tensor * k = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_k"), h), head_dim_, kv_heads_, n);
        ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m_.tensor(b + "attn_v"), h), head_dim_, kv_heads_, n);
        q = ggml_mul(ctx, ggml_rms_norm(ctx, q, eps_), m_.tensor(b + "attn_q_norm"));
        k = ggml_mul(ctx, ggml_rms_norm(ctx, k, eps_), m_.tensor(b + "attn_k_norm"));
        const float theta = global ? theta_global_ : theta_local_, scale = global ? scale_global_ : 1.0f;
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, theta, scale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim_, GGML_ROPE_TYPE_NEOX, 0, theta, scale, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kq = mul_mat(ctx, kh, qh);
        kq = ggml_soft_max_ext(ctx, kq, global ? nullptr : sliding_mask, query_scale_, 0.0f);
        ggml_tensor * o = mul_mat(ctx, vt, kq);
        o = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), head_dim_ * heads_, n);
        x = ggml_add(ctx, x, rms_norm(mul_mat(ctx, m_.tensor(b + "attn_out"), o), b + "attn_post_norm"));

        h = rms_norm(x, b + "ffn_norm");
        h = ggml_mul(ctx, gelu_tanh(mul_mat(ctx, m_.tensor(b + "ffn_gate"), h)), mul_mat(ctx, m_.tensor(b + "ffn_up"), h));
        x = ggml_add(ctx, x, rms_norm(mul_mat(ctx, m_.tensor(b + "ffn_down"), h), b + "ffn_post_norm"));
        if (layers && l + 1 < layers_) layers->push_back(x);
    }
    x = rms_norm(x, "text.final_norm");
    if (layers) layers->push_back(x);
    return x;
}

}  // namespace irodori
