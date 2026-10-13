#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "graph.h"
#include "model-file.h"

namespace irodori {

/** The residual units of each encoder and decoder block, of dilations 1, 3 and 9, as the official module builds them. */
constexpr int kCodecResidualUnits = 3;

/** Called with each piece of audio as it is decoded; returning false stops the decoding. */
using AudioSink = std::function<bool(const float * samples, size_t n)>;

/** What the decoder knows when it sizes a window: what it has passed on, and how fast it decodes. */
struct DecodeProgress {
    /** Frames passed on so far, 0 for the first window. */
    int64_t frames_done = 0;
    /** Frames still to decode and pass on. */
    int64_t frames_left = 0;
    /** Seconds of audio passed on so far. */
    double audio_sent = 0;
    /** Seconds since the first window's audio was passed on. */
    double since_first = 0;
    /** The running estimate of the decoder's seconds per frame it decodes, a window's margins included. */
    double seconds_per_frame = 0;
};

/** The frames of the next window, the first included, which must be at least 1 and at most the frames left. */
using WindowChoice = std::function<int(const DecodeProgress & progress)>;

/** The running estimate of the seconds per decoded frame after a window measured at `measured`; the first is its own. */
inline double next_estimate(double previous, double measured) {
    return previous > 0 ? 0.5 * (previous + measured) : measured;
}

/**
 * How a streaming decoder sizes its windows: `first` frames first, then from `floor` to `ceiling` frames, so that the
 * next one arrives while the listener still has `margin` seconds of audio, a frame being `frame_seconds` of audio and
 * each window decoding `context` frames on either side of its own.
 */
struct WindowRule {
    int first = 0, floor = 0, ceiling = 0, context = 0;
    double frame_seconds = 0, margin = 0;
};

/**
 * The frames of the next window under `rule`. The first window is `first` frames, or the frames left when fewer. After
 * it, the frames left when they are no more than the floor; otherwise the
 * largest size from the floor to the ceiling whose decoding, estimated at `seconds_per_frame` for its frames and its
 * margins, ends before the listener's audio falls below the margin, the listener having the audio sent less the time
 * since the first window was sent, as if playback began with the first window. When not even the floor ends in time,
 * the floor where its decoding still takes less than the audio it gives, which gains on the listener, and the ceiling
 * where decoding is slower than real time, since smaller windows would only decode more margins and fall further
 * behind; the player has to buffer then. It reads no clock, and the result is never past the ceiling or the frames
 * left.
 */
int next_window(const WindowRule & rule, const DecodeProgress & progress);

/**
 * Semantic-DACVAE-Japanese-32dim, the codec of Irodori-TTS: 48 kHz audio and a 32-dimensional latent at
 * 25 frames a second. Every convolution is non-causal with symmetric padding.
 */
class Codec {
public:
    /** The codec of the model file `m`, which outlives it. */
    Codec(const ModelFile & m, ggml_backend_t backend);
    ~Codec();
    Codec(const Codec &) = delete;
    Codec & operator=(const Codec &) = delete;

    int sample_rate() const { return sample_rate_; }
    int hop() const { return hop_; }
    int latent_dim() const { return latent_dim_; }
    /** The SHA-256 of the official codec's tensors, which binds a voice file to the codec that encoded it. */
    const std::string & sha256() const { return sha256_; }

    /**
     * The latent, row-major [frames, latent_dim], of mono audio at sample_rate() as the official runtime
     * encodes a reference after normalizing its loudness: padded at the end by reflection to whole frames,
     * then encoded in windows of `window` frames, each with enough frames around it that the frames it keeps
     * are those of encoding the whole at once.
     */
    std::vector<float> encode(const std::vector<float> & audio, int window = 100);

    /** The encoder's output [latent_dim, samples / hop] for `samples` that are a whole number of frames. */
    ggml_tensor * build_encoder(Graph & g, const std::vector<float> & samples) const;

    /**
     * The audio [1, frames * hop] of a latent, row-major [frames, latent_dim], of at least two frames.
     * `stages`, when given, receives the output of the input projection, of the first convolution and of
     * each upsampling block.
     */
    ggml_tensor * build_decoder(Graph & g, const std::vector<float> & latent, std::vector<ggml_tensor *> * stages = nullptr) const;

    /**
     * Decodes the first `samples` samples of a latent, row-major [frames, latent_dim], in windows of the sizes `choose`
     * gives as it measures the decoding, the first included, each with enough frames around it that its samples are
     * those of decoding the whole latent at once, and passes each window's samples to `sink`.
     */
    void decode(const std::vector<float> & latent, int64_t samples, const WindowChoice & choose, const AudioSink & sink);

    /** The frames on each side of a window that the encoder's receptive field reaches. */
    static constexpr int kEncoderMargin = 8;
    /** The frames on each side of a window that the decoder's receptive field reaches, which its layers give. */
    int decoder_margin() const { return decoder_margin_; }

private:
    ggml_backend_t backend_;
    const ModelFile & m_;
    ggml_gallocr_t allocr_ = nullptr;
    int sample_rate_ = 0, hop_ = 0, latent_dim_ = 0, decoder_margin_ = 0;
    std::string sha256_;
    std::vector<int32_t> encoder_rates_, decoder_rates_;
};

}  // namespace irodori
