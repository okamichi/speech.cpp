#include "placement.h"

#include <algorithm>
#include <cstdlib>
#include <vector>

#include "error.h"

namespace drama3 {

namespace {

/**
 * What the graphs and the backend allocate while a model computes, beyond its weights and its caches: the buffers of
 * the text model's, the audio decoder's and the codec's graphs at their largest, a block of 512 prompt rows attending
 * to the most rows and a window of 128 frames of the codec, the Gated DeltaNet's states and their copy at the end of a
 * voice's head of the prompt, and the backend's own. For a prompt of 8,018 rows and 327 frames of speech with the Q4_K
 * file on an Apple M5 the three buffers took 271, 275 and 497 MiB and the states 64 MiB, as their copy does; on an RTX
 * 2080 with Vulkan the backend's own took about 0.3 GB more (2026-10-15).
 */
constexpr uint64_t kComputeBytes = (uint64_t) 1536 << 20;

uint64_t gib_tenths(uint64_t bytes) {
    return (bytes * 10 + (1u << 29)) >> 30;
}

std::string gigabytes(uint64_t bytes) {
    const uint64_t tenths = gib_tenths(bytes);
    return std::to_string(tenths / 10) + "." + std::to_string(tenths % 10) + " GiB";
}

}  // namespace

bool is_routed_experts(const std::string & name) {
    const size_t at = name.rfind("_exps");
    return name.rfind("blk.", 0) == 0 && at != std::string::npos && at + 5 == name.size();
}

int layer_of(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    char * end = nullptr;
    const long layer = std::strtol(name.c_str() + 4, &end, 10);
    return end && *end == '.' ? (int) layer : -1;
}

uint64_t working_bytes(const ModelFile & m) {
    const std::string x = "drama3.text.", a = "drama3.audio_decoder.";
    const uint64_t rows = (uint64_t) m.size("drama3.generation.max_prompt_rows") + (uint64_t) m.size("drama3.generation.max_frames");
    const std::vector<std::string> types = m.str_array(x + "layer_types");
    const uint64_t attention_layers = (uint64_t) std::count(types.begin(), types.end(), "full_attention");
    const uint64_t decoder_layers = (uint64_t) m.count(a + "shared_enrichment_layers") + (uint64_t) m.count(a + "num_codebooks") - 1;
    // A key and a value of float32 for each kv head of each row.
    const uint64_t text = 2 * sizeof(float) * (uint64_t) m.size(x + "head_dim") * (uint64_t) m.size(x + "num_key_value_heads");
    const uint64_t decoder = 2 * sizeof(float) * (uint64_t) m.size(a + "head_dim") * (uint64_t) m.size(a + "n_local_heads");
    return rows * (attention_layers * text + decoder_layers * decoder) + kComputeBytes;
}

int expert_layers_on_device(const ModelFile & m, uint64_t free, const std::string & device) {
    const int layers = m.count("drama3.text.num_hidden_layers");
    const uint64_t rest = m.weight_bytes([](const std::string & name) { return !is_routed_experts(name); }) + working_bytes(m);
    if (rest > free) {
        throw Error(Fault::OutOfMemory, device + " has " + gigabytes(free) + " free, and Drama 3 needs " + gigabytes(rest) +
                                            " there beside its experts; load it on the CPU or on a GPU with more memory free");
    }
    uint64_t left = free - rest;
    for (int l = 0; l < layers; l++) {
        const uint64_t experts = m.weight_bytes([l](const std::string & name) { return is_routed_experts(name) && layer_of(name) == l; });
        if (experts > left) return l;
        left -= experts;
    }
    return layers;
}

}  // namespace drama3
