#include "layers.h"

#include <cstring>
#include <memory>

#include "error.h"
#include "log.h"

namespace irodori {

namespace {

/** Whether `backend` computes flash attention of float32 heads of `head_dim` with a half-precision mask. */
bool computes_flash_attention(ggml_backend_t backend, int64_t head_dim) {
    ggml_init_params params = {ggml_tensor_overhead() * 8, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, 2, 1, 2);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, 3, 1, 2);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, 3, 1, 2);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, 3, 2, 1, 2);
    ggml_tensor * attention = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f, 0.0f, 0.0f);
    ggml_prec_set_acc(attention, GGML_PREC_F32);
    return ggml_backend_supports_op(backend, attention);
}

}  // namespace

Attention attention_on(ggml_backend_t backend, int64_t head_dim) {
    const bool metal = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend))), "MTL") == 0;
    return metal && computes_flash_attention(backend, head_dim) ? Attention::Flash : Attention::Products;
}

const char * attention_name(Attention attention) {
    return attention == Attention::Flash ? "flash attention" : "two matrix products";
}

Attention choose_attention(ggml_backend_t backend, int64_t head_dim, std::optional<Attention> attention, const std::string & what) {
    const Attention chosen = attention.value_or(attention_on(backend, head_dim));
    if (chosen == Attention::Flash && !computes_flash_attention(backend, head_dim)) {
        throw Error(Fault::Device, std::string("the device ") + ggml_backend_name(backend) + " cannot compute flash attention over heads of " +
                                       std::to_string(head_dim));
    }
    log_message(LogLevel::Info, what + " attends with " + attention_name(chosen) + " on " + ggml_backend_name(backend));
    return chosen;
}

}  // namespace irodori
