#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/** A condition that the text backbone makes of tokens, by its projector and norm. */
enum class Condition { Text, Caption };

/**
 * The text and the caption conditions: a pretrained backbone on the tokens, which the two share, then the condition's
 * own residual projector and norm that map the backbone's channels to the condition's space. The backbone is
 * ModernBERT-ja in v4.1-Small and T5Gemma 2's text encoder, a bidirectional Gemma 3, in v4-Large.
 *
 * The official runtime pads the tokens to 256 (512 for a caption) and masks the padding out of every attention; a
 * padded position never reaches a real one, so this runs on the real tokens alone.
 */
class TextEncoder {
public:
    explicit TextEncoder(const ModelFile & m);

    /**
     * The condition of `ids`, [dim, n]. When `layers` is given it receives the backbone's hidden states: the embeddings,
     * then the output of each layer, the last after the final norm.
     */
    ggml_tensor * build(Graph & g, const std::vector<int32_t> & ids, Condition condition = Condition::Text,
                        std::vector<ggml_tensor *> * layers = nullptr) const;

    int dim() const { return dim_; }
    int max_tokens() const { return max_tokens_; }

private:
    ggml_tensor * modernbert(Graph & g, const std::vector<int32_t> & ids, std::vector<ggml_tensor *> * layers) const;
    ggml_tensor * t5gemma2(Graph & g, const std::vector<int32_t> & ids, std::vector<ggml_tensor *> * layers) const;

    const ModelFile & m_;
    bool gemma_;
    int hidden_, heads_, layers_, dim_, max_tokens_;
    float eps_, norm_eps_, theta_global_, theta_local_;
    std::vector<int32_t> global_;
    /** ModernBERT's local layers attend to the tokens at most `window_` away on either side. */
    int window_ = 0;
    /** T5Gemma 2's sliding layers attend to the tokens at most `before_` before and `after_` after. */
    int before_ = 0, after_ = 0;
    int kv_heads_ = 0, head_dim_ = 0;
    int32_t eoi_ = -1;
    float scale_global_ = 1.0f, query_scale_ = 1.0f;
};

}  // namespace irodori
