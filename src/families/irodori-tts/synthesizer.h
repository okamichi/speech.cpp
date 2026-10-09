#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "codec.h"
#include "dit.h"
#include "duration.h"
#include "reference.h"
#include "sampler.h"
#include "speaker-encoder.h"
#include "text-encoder.h"
#include "tokenizer.h"

namespace irodori {

/**
 * A voice: the latent of its reference and the speaker condition the model makes of it, or a speaker-inversion
 * embedding as the speaker condition without a latent. A voice without a reference has neither and speaks as the
 * runtime's no_ref, with the null speaker of a file that holds it.
 */
struct Voice {
    std::vector<float> latent;
    int frames = 0;
    std::vector<float> speaker;
    int speaker_tokens = 0;
};

/**
 * The official runtime's cut where a sampled latent goes flat (find_flattening_point()), with its settings from the
 * model file or the request: the first frame from which `window` frames, zeros past the end, have a standard deviation
 * under `std_threshold` and a mean within `mean_threshold` of zero, or the number of frames when none does. The
 * thresholds are float32, as the runtime compares its float32 tensors with them. A request that keeps the tail is not
 * cut.
 */
struct TailCut {
    bool keep = false;
    int window = 0;
    float std_threshold = 0, mean_threshold = 0;

    TailCut() = default;
    explicit TailCut(const ModelFile & m);
    int flattening_point(const std::vector<float> & latent, int frames, int latent_dim) const;
    /**
     * Throws, naming the option, when `asked` keeps the tail and changes a setting of the cut from this one's, which
     * then has no effect.
     */
    void check(const TailCut & asked) const;
};

struct Request {
    std::string text;
    /** The caption that describes the voice and the way of speaking (VoiceDesign); one that strips to nothing is none. */
    std::string caption;
    bool compatible = false;
    uint64_t seed = 0;
    /** The sampler's steps; 0 takes the model's default (4 for MeanFlow, 40 for RF). */
    int steps = 0;
    LengthOptions length;
    /** The RF sampler's guidance and schedule; none takes the model file's. A MeanFlow model takes none. */
    std::optional<Guidance> guidance;
    /** The cut at the tail; none takes the model file's settings. */
    std::optional<TailCut> tail;
    /** The sampler's draw of noise, row-major [frames, latent_dim], instead of one from the seed. */
    std::vector<float> noise;
    /** With `noise`, the draw that the guidance's speaker noise is made of, of the voice's speaker condition's size. */
    std::vector<float> speaker_noise;
    /**
     * Told the fraction of the sampler's steps done before each step and once the last is done, while no audio
     * reaches the sink; once it answers false, synthesize() returns without audio.
     */
    std::function<bool(double done)> progress;
};

/** Where the time of one synthesis went, in seconds, and how long the speech is. */
struct Stats {
    double text = 0, sampling = 0, first_audio = 0, codec = 0;
    int tokens = 0, frames = 0;
    size_t samples = 0;
    bool instructions_truncated = false;
};

/**
 * Text in, 48 kHz audio out, as the official runtime makes it: the normalized text's tokens, the text
 * condition and the predicted length, the sampler from seeded noise, the tail cut where the latent goes
 * flat, and the codec decoding window by window so that the first audio comes before the rest is decoded.
 */
class Synthesizer {
public:
    Synthesizer(const std::string & model_path, ggml_backend_t backend);
    ~Synthesizer();

    /**
     * A voice from a reference WAVE file or a voice file of this model's codec or, for an embedding, of this model
     * (voice-file.h). A voice file of another codec or model throws.
     */
    Voice load_voice(const std::string & path);
    /** A voice from the latent of its reference, row-major [frames, latent_dim]. */
    Voice voice_from_latent(std::vector<float> latent);

    /** Speaks `r.text` in `voice`, passing the audio to `sink` window by window. Returns the samples. */
    size_t synthesize(const Request & r, const Voice & voice, const AudioSink & sink, Stats * stats = nullptr);

    int sample_rate() const { return codec_.sample_rate(); }
    const ModelFile & model() const { return *model_; }
    const Codec & codec() const { return codec_; }
    /** The guidance and the cut at the tail of a request that asks for none of its own. */
    const Guidance & default_guidance() const { return sampler_.guidance(); }
    const TailCut & default_tail() const { return tail_; }
    /** Whether the file holds the null speaker, which a voice without a reference speaks with. */
    bool has_null_speaker() const { return model_->boolean("irodori-tts.duration.null_speaker"); }
    /** Whether the file holds the caption's encoder, which a request with a caption needs. */
    bool has_caption() const { return model_->boolean("irodori-tts.caption_condition"); }

    /**
     * How the decoder sizes its windows, a short first one of 12 frames, which brings the first audio early, and the
     * later ones as it measures its speed, for a codec's frames (synthesizer.cpp says why).
     */
    static WindowRule window_rule(int hop, int sample_rate);

private:
    ggml_backend_t backend_;
    std::unique_ptr<ModelFile> model_;
    Codec codec_;
    Tokenizer tokenizer_;
    TextEncoder text_;
    SpeakerEncoder speaker_;
    DurationPredictor duration_;
    Dit dit_;
    Sampler sampler_;
    ReferenceRules reference_;
    TailCut tail_;
    ggml_gallocr_t allocr_ = nullptr;
};

}  // namespace irodori
