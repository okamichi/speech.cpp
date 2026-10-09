#include "t5gemma-encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace irodori {

ggml_tensor * t5gemma_encode(const ModelFile & m, Graph & g, const std::vector<int32_t> & ids,
                             std::vector<ggml_tensor *> * layers) {
    ggml_context * ctx = g.ctx();
    const std::string p = "irodori-tts.text.";
    const int heads = m.size(p + "num_heads");
    const int kv_heads = m.size(p + "num_kv_heads"), head_dim = m.size(p + "head_dim");
    const int count = m.count(p + "num_layers"), window = (int) m.u32(p + "window");
    const float eps = m.f32(p + "norm_eps"), scale = m.f32(p + "attention_scale");
    const auto global = m.i32_array(p + "layer_global");
    const int64_t n = (int64_t) ids.size();
    auto norm = [&](ggml_tensor * x, const std::string & name) {
        // The converter folds Gemma's 1 + weight into each RMS normalization weight.
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), m.tensor(name));
    };
    std::vector<int32_t> positions(n);
    for (int64_t i = 0; i < n; i++) positions[i] = (int32_t) i;
    ggml_tensor * pos = g.input(positions, n);
    std::vector<float> mask(n * n);
    // T5Gemma2's bidirectional window includes one more position on the right for an even size.
    const int64_t left = (window + 1LL) / 2, right = window / 2 + 1LL;
    for (int64_t q = 0; q < n; q++)
        for (int64_t k = 0; k < n; k++) mask[q * n + k] = q - k < left && k - q < right ? 0.0f : -INFINITY;
    ggml_tensor * local_mask = g.input(mask, n, n);
    ggml_tensor * x = ggml_scale(ctx, ggml_get_rows(ctx, m.tensor("text.embd"), g.input(ids, n)), m.f32(p + "embedding_scale"));
    const int32_t eoi = (int32_t) m.u32(p + "eoi_id");
    if (std::find(ids.begin(), ids.end(), eoi) != ids.end()) {
        std::vector<float> ordinary(n), special(n);
        for (int64_t i = 0; i < n; i++) {
            ordinary[i] = ids[i] == eoi ? 0.0f : 1.0f;
            special[i] = ids[i] == eoi ? 1.0f : 0.0f;
        }
        x = ggml_add(ctx, ggml_mul(ctx, x, g.input(ordinary, 1, n)),
                     ggml_mul(ctx, ggml_repeat(ctx, m.tensor("text.eoi_embd"), x), g.input(special, 1, n)));
    }
    if (layers) layers->push_back(x);
    for (int l = 0; l < count; l++) {
        const std::string b = "text.blk." + std::to_string(l) + ".";
        ggml_tensor * h = norm(x, b + "attn_norm");
        ggml_tensor * q = norm(ggml_reshape_3d(ctx, mul_mat(ctx, m.tensor(b + "attn_q"), h), head_dim, heads, n), b + "q_norm");
        ggml_tensor * k = norm(ggml_reshape_3d(ctx, mul_mat(ctx, m.tensor(b + "attn_k"), h), head_dim, kv_heads, n), b + "k_norm");
        ggml_tensor * v = ggml_reshape_3d(ctx, mul_mat(ctx, m.tensor(b + "attn_v"), h), head_dim, kv_heads, n);
        const bool full = global.at(l) != 0;
        const float theta = m.f32(p + (full ? "rope_theta_global" : "rope_theta_local"));
        const float freq_scale = full ? m.f32(p + "rope_scale_global") : 1.0f;
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0, theta, freq_scale, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * scores = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), full ? nullptr : local_mask, scale, 0.0f);
        ggml_tensor * out = mul_mat(ctx, vt, scores);
        out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), head_dim * heads, n);
        out = mul_mat(ctx, m.tensor(b + "attn_out"), out);
        out = norm(out, b + "attn_post_norm");
        x = ggml_add(ctx, x, out);
        h = norm(x, b + "ffn_norm");
        h = ggml_mul(ctx, ggml_gelu(ctx, mul_mat(ctx, m.tensor(b + "ffn_act"), h)), mul_mat(ctx, m.tensor(b + "ffn_gate"), h));
        x = ggml_add(ctx, x, norm(mul_mat(ctx, m.tensor(b + "ffn_down"), h), b + "ffn_post_norm"));
        if (layers && l + 1 < count) layers->push_back(x);
    }
    x = norm(x, "text.final_norm");
    if (layers) layers->push_back(x);
    return x;
}

}  // namespace irodori
