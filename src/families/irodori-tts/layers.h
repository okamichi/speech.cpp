#pragma once

#include <cmath>
#include <optional>
#include <string>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/** How a transformer of Irodori-TTS attends. */
enum class Attention {
    /** ggml_flash_attn_ext, which holds no scores. */
    Flash,
    /** Two matrix products and a softmax. */
    Products,
};

/**
 * The attention of heads of `head_dim` on `backend`: flash attention on Metal where it computes it, and the two products
 * on Vulkan, the CPU and any other backend. Metal's flash attention keeps the keys and values in float32, where its
 * matrix kernel rounds them to half precision, and the DiT's worst step of rf-hai-speaker-kv-min1 lies 48 dB from the
 * official instead of 31. With v4.1-Small-MF and v4.1-Small in F16 on an Apple M5 (2026-10-13), it brought the first
 * audio of 16 to 19 s of speech 5 to 7% earlier and spoke 20 sentences of 2 to 19 s 1 to 3% faster, while 0.8 to 1.2 s
 * with v4.1-Small's guidance took 15 to 30 ms longer. Vulkan's matrix product does not round the keys and values so:
 * on an RTX 2080 flash attention was no faster and about 4 dB further from the official, and on the M5's CPU it was no
 * faster at 27 to 685 frames and no more precise.
 */
Attention attention_on(ggml_backend_t backend, int64_t head_dim);

/** The attention's name, as a log says it. */
const char * attention_name(Attention attention);

/**
 * `attention` or, without it, attention_on()'s choice for heads of `head_dim` on `backend`, which `what` logs. Flash
 * attention that the backend cannot compute throws an Error of the device.
 */
Attention choose_attention(ggml_backend_t backend, int64_t head_dim, std::optional<Attention> attention, const std::string & what);

/** The building blocks Irodori-TTS's own modules share, on channel-first activations. */
struct Layers {
    ggml_context * ctx;
    const ModelFile & m;
    float eps;
    Attention attends = Attention::Products;

    /** A linear layer with the bias `name.bias`. */
    ggml_tensor * linear(ggml_tensor * x, const std::string & name) const {
        return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
    }

    /** RMSNorm with the weight `name`. */
    ggml_tensor * rms(ggml_tensor * x, const std::string & name) const {
        return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), m.tensor(name));
    }

    /** SwiGLU: ffn_down(silu(ffn_gate x) * ffn_up x). */
    ggml_tensor * swiglu(ggml_tensor * x, const std::string & prefix) const {
        ggml_tensor * gate = ggml_silu(ctx, mul_mat(ctx, m.tensor(prefix + "ffn_gate"), x));
        return mul_mat(ctx, m.tensor(prefix + "ffn_down"), ggml_mul(ctx, gate, mul_mat(ctx, m.tensor(prefix + "ffn_up"), x)));
    }

    /**
     * Scaled dot-product attention of q [head_dim, heads, n_q, batch] over k and v [head_dim, heads, n_kv,
     * batch], with an optional additive mask [n_kv, n_q, 1, batch] in half precision; the result is [head_dim * heads,
     * n_q, batch].
     */
    ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask) const {
        const int64_t head_dim = q->ne[0], heads = q->ne[1], n_q = q->ne[2], batch = q->ne[3];
        const float scale = 1.0f / std::sqrt((float) head_dim);
        if (attends == Attention::Flash) {
            ggml_tensor * o = ggml_flash_attn_ext(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), ggml_permute(ctx, k, 0, 2, 1, 3),
                                                  ggml_permute(ctx, v, 0, 2, 1, 3), mask, scale, 0.0f, 0.0f);
            ggml_prec_set_acc(o, GGML_PREC_F32);
            return ggml_reshape_3d(ctx, o, head_dim * heads, n_q, batch);
        }
        ggml_tensor * qh = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
        ggml_tensor * kh = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
        ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
        ggml_tensor * kq = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qh), mask, scale, 0.0f);
        ggml_tensor * o = mul_mat(ctx, vt, kq);
        return ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3)), head_dim * heads, n_q, batch);
    }

    /** x * (1 + scale) + shift, in that order of operations. */
    ggml_tensor * modulate(ggml_tensor * x, ggml_tensor * shift, ggml_tensor * scale) const {
        return ggml_add(ctx, ggml_mul(ctx, x, ggml_scale_bias(ctx, scale, 1.0f, 1.0f)), shift);
    }
};

}  // namespace irodori
