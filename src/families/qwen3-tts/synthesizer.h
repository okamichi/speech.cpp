#pragma once

#include <climits>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "codec.h"
#include "model-file.h"
#include "prompt.h"
#include "qwen2-tokenizer.h"
#include "sampler.h"
#include "talker.h"

struct SynthesisRequest {
    std::string text;
    std::string speaker;
    /** A BCP 47 tag of one of the model's languages (`ja`, `ja-JP`), or "auto". */
    std::string language = "auto";
    /** An instruction of how to speak, for a model that takes_instructions(), or empty for none. */
    std::string instructions;
    /** The frames to make at most; the model's own limit applies as well. */
    int max_frames = INT_MAX;
    uint64_t seed = 0;
    /** How the talker and the code predictor sample, where the request sets it rather than the model file. */
    std::optional<SamplingParams> talker, code_predictor;
};

/** How a synthesis ended: the frames it made, and whether the talker ended the speech or a limit stopped it. */
struct SynthesisOutcome {
    int frames = 0;
    bool ended = false;
};

/** Where the time of one synthesis went, in seconds: the talker's prefill of the prompt, and its steps apart. */
struct SynthesisStats {
    double prompt = 0, prefill = 0, talker = 0, code_predictor = 0, codec = 0;
};

/** The prefix of the keys of a model file that hold the text's tokenizer. */
constexpr const char * kTextTokenizer = "qwen3-tts.tokenizer";

/**
 * The longest text in tokens that a model file speaks: what leaves the talker's positions room for a prompt and the
 * most frames.
 */
int text_token_limit(const ModelFile & m);

/** Called with each piece of 24 kHz audio as it is decoded; returning false stops the synthesis. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/**
 * The frames of the chunk of audio a synthesis passes after `sent` chunks: 1, 1, 2 and then 4.
 *
 * The first chunk is the first frame alone, so that audio starts once one frame is made. With 4 frames in every chunk
 * after it, as speech.cpp 0.7.1 sent them, the second chunk of the 1.7B model arrived 0.096 s after the first on an
 * RTX 2080 (Vulkan) and 0.136 s after it on an Apple M5 (Metal), where a frame takes 24 ms and 34 ms to make, so a
 * player that starts on the first chunk's 0.08 s ran dry for 16 to 59 ms in each of speech-bench's 20 Japanese
 * sentences (2026-10-07). Chunks of 1 and 2 frames keep it fed while the speech gets ahead of it, and 4 frames from
 * then on keep the codec's calls few: each takes about 16 ms on the M5's Metal whether it decodes 1 frame or 4 (a
 * rough measure on a shared GPU), and 4 frames bound how long a cancel and the worker's silence last, as before. The
 * sizes are fixed rather than following the measured speed, since the codec's samples depend, by rounding, on how the
 * frames are grouped into calls, and the same request with the same seed gives the same samples.
 */
int chunk_frames(int sent);

/** Text in, audio out: the tokenizer, the talker, the code predictor and the codec decoder of one model file together. */
class Synthesizer {
public:
    Synthesizer(const std::string & path, ggml_backend_t backend);

    /**
     * Speaks `r.text`, passing the audio to the sink in chunks of chunk_frames() frames, until the talker ends the
     * speech, `r.max_frames` or the model's limit of frames is reached, or the sink stops it. A text longer than
     * max_text_tokens() throws, and so does an instruction whose tokens leave the text fewer than it has.
     */
    SynthesisOutcome synthesize(const SynthesisRequest & r, const AudioSink & sink, SynthesisStats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return model_; }
    const PromptIds & ids() const { return ids_; }
    /** How the model file says the official generate() samples. */
    const Generation & generation() const { return generation_; }
    /** The frames a request makes at most. */
    int max_frames() const { return generation_.max_frames; }
    /** The samples of one frame at sample_rate(). */
    int samples_per_frame() const { return codec_.samples_per_frame(); }
    int max_text_tokens() const { return text_token_limit(model_); }

private:
    ModelFile model_;
    Talker talker_;
    CodecDecoder codec_;
    Qwen2Tokenizer tokenizer_;
    PromptIds ids_;
    Generation generation_;
};
