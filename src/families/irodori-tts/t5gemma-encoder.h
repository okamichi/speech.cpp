#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace irodori {

/** The bidirectional T5Gemma2 text backbone, with its embeddings and layer outputs when requested. */
ggml_tensor * t5gemma_encode(const ModelFile & model, Graph & graph, const std::vector<int32_t> & ids,
                             std::vector<ggml_tensor *> * layers);

}  // namespace irodori
