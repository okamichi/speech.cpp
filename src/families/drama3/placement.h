#pragma once

#include <cstdint>
#include <string>

#include "model-file.h"

namespace drama3 {

/** Whether the tensor `name` holds a layer's routed experts, which the placement may leave in the CPU's memory. */
bool is_routed_experts(const std::string & name);

/** The layer of a layer's tensor `name`, "blk.<layer>." followed by the rest; -1 for a tensor of no layer. */
int layer_of(const std::string & name);

/**
 * The memory a device needs beside the weights it holds to serve every request the model file `m` takes: the keys and
 * values of the text model's attention layers and of the audio decoder for the most rows a chunk's prompt and speech
 * reach, in float32 as the caches hold them, and what the graphs and the backend allocate while they compute.
 */
uint64_t working_bytes(const ModelFile & m);

/**
 * The layers, from the first, whose routed experts a device with `free` bytes free holds beside every other tensor of
 * the model file `m` and working_bytes(): every layer where the whole model fits, fewer where it does not. A device
 * that cannot hold the rest of the model with the experts of no layer throws an Error of its memory naming `device`.
 */
int expert_layers_on_device(const ModelFile & m, uint64_t free, const std::string & device);

}  // namespace drama3
