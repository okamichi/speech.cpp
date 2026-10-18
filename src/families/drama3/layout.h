#pragma once

#include "model-file.h"

namespace drama3 {

/**
 * Layout 1 of a Fish Audio Drama 3 model file, which reference/drama3/convert.py writes: the text model, a Qwen3.5-MoE
 * backbone of Gated DeltaNet and gated attention layers with 256 experts and a shared expert each, its output head for
 * the ids a frame's main token is drawn from, the audio decoder that draws a frame's other codes, the tokenizer, and
 * the codec, a causal DAC at 44.1 kHz with transformers around its quantizer; every key of drama3.* and speech.* that
 * the family reads, and the tensors the keys call for.
 */
extern const Layout model_layout;

/** The residual units of each codec block, of dilations 1, 3 and 9, which the official codec fixes in its code. */
constexpr int codec_residual_units = 3;

}  // namespace drama3
