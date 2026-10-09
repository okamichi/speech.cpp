#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/** A condition that the text backbone makes of tokens, by its projector and norm. */
enum class Condition { Text, Caption };

/**
 * The text and the caption conditions: ModernBERT-ja or T5Gemma2 on the tokens, which the two share, then the condition's own
 * residual projector and norm that map its hidden channels to the condition's space.
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
    const ModelFile & m_;
    int hidden_, heads_, layers_, window_, dim_, max_tokens_;
    float eps_, norm_eps_, theta_global_, theta_local_;
    std::vector<int32_t> global_;
};

}  // namespace irodori
