#pragma once

#include "model-file.h"

namespace irodori {

/**
 * Layout 3 of an Irodori-TTS model file, which reference/irodori-tts/convert.py writes: the Unigram or BPE tokenizer, ModernBERT-ja or T5Gemma2
 * and the projectors of the text and the caption, the speaker encoder, the duration predictor with its null speaker,
 * the DiT with the caption's keys and values and the DACVAE codec, every key of irodori-tts.* and speech.* that the
 * family reads, and the tensors the keys call for. A file of layout 1 reads as one of layout 2 without the null speaker
 * and the caption's encoder, which speaks with a reference and no caption. Layout 2 reads as ModernBERT-ja with Unigram.
 */
extern const Layout model_layout;

/** The general.architecture and the speech.layout of a voice file, which write_voice_file() writes. */
constexpr const char * kVoiceArchitecture = "irodori-tts-voice";
constexpr uint32_t kVoiceLayout = 2;

/** The release whose reader first takes voice files of kVoiceLayout, which a voice file names as speech.requires. */
constexpr const char * kVoiceLayoutRequires = "0.8.0";

/**
 * Layout 2 of a voice file for the model file `model`, which outlives it: the codec latent of one or more reference
 * recordings, encoded one by one and joined, F32 [latent_dim, frames], the hash of the codec that encoded it, each
 * recording's length and rate, whether their loudness was brought to a target and which, and the kind of device that
 * encoded them; or a speaker-inversion embedding, F32 [speaker.dim, tokens], and the general.source.url of the model it
 * was made for. A voice file of a codec or a model other than the model's is refused as the caller's mistake, before
 * its tensor is checked. A file of layout 1 reads as one of a recording brought to -16 LUFS.
 */
Layout voice_layout(const ModelFile & model);

}  // namespace irodori
