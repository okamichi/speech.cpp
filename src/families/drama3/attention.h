#pragma once

#include <cstdint>
#include <vector>

#include "ggml-backend.h"
#include "graph.h"

namespace drama3 {

/**
 * The keys and values of the attention layers of a stack, for one sequence: the keys of each layer, [head_dim, rows,
 * kv heads], and its values transposed, [rows, head_dim, kv heads], so that each is a matrix product's first operand.
 */
class KeysValues {
public:
    KeysValues(ggml_backend_t backend, int layers, int head_dim, int kv_heads, int64_t rows);
    ~KeysValues();
    KeysValues(const KeysValues &) = delete;
    KeysValues & operator=(const KeysValues &) = delete;

    int64_t capacity() const { return capacity_; }

    /**
     * Writes `k` and `v`, [head_dim, kv heads, n], to layer `l` at rows [first, first + n) and builds the attention of
     * `q`, [head_dim, heads, n], over the rows [0, first + n) under `mask` (causal_mask()), scaled by 1/sqrt(head_dim):
     * [head_dim * heads, n]. Head i reads kv head i / (heads / kv heads), as repeat_interleave() gives them.
     */
    ggml_tensor * attend(Graph & g, int l, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int64_t first, ggml_tensor * mask) const;

private:
    ggml_context * ctx_ = nullptr;
    ggml_backend_buffer_t buffer_ = nullptr;
    std::vector<ggml_tensor *> k_, v_;
    int64_t capacity_ = 0;
};

/**
 * The mask of `n` rows after `first` that attend to every row up to their own, [first + n, n], or none for one row,
 * which attends to every row there is.
 */
ggml_tensor * causal_mask(Graph & g, int64_t first, int64_t n);

/** The positions of `n` rows after `first`, I32 [n], as RoPE takes them. */
ggml_tensor * positions(Graph & g, int64_t first, int64_t n);

}  // namespace drama3
