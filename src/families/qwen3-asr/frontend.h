#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "fft.h"
#include "model-file.h"

namespace qwen3_asr {

/**
 * What qwen-asr's normalize_audio_input() and transformers' Qwen3ASRFeatureExtractor do to audio at the model's rate
 * (qwen_asr/inference/utils.py and transformers/models/qwen3_asr/feature_extraction_qwen3_asr.py): the whole audio's
 * samples divided by their peak when it exceeds 1 (normalize()), before qwen-asr splits audio too long for the model;
 * then, for each utterance, one shorter than qwen3-asr.audio.min_samples padded with zeros to it, and Whisper's log-mel
 * (features()): a centred STFT with reflected padding and a periodic Hann window, the power of every frame but the last
 * through a Slaney mel filterbank, its log10 floored at log_floor, floored again at the utterance's largest value less
 * dynamic_range, and shifted by log_offset and divided by log_divisor.
 *
 * It runs on the host in double precision rather than in a ggml graph, as FastConformer's frontend does: it costs a
 * few milliseconds a second of audio, and Metal's matrix kernel rounds its inputs to half precision, which the log
 * of the quiet frames' power would magnify.
 */
class Frontend {
public:
    explicit Frontend(const ModelFile & m);

    /** Mono samples at sample_rate() divided by their peak when it exceeds 1, and clipped to [-1, 1]. */
    static std::vector<float> normalize(std::vector<float> samples);

    /**
     * The features of an utterance of samples that normalize() gave, [frames, mels] row-major (ggml [mels, frames]).
     */
    std::vector<float> features(const std::vector<float> & samples) const;

    /** The frames features() gives for that many samples. */
    int64_t frames(size_t samples) const;

    int sample_rate() const { return sample_rate_; }
    int mels() const { return mels_; }

private:
    int sample_rate_, n_fft_, hop_, mels_, min_samples_;
    float log_floor_, dynamic_range_, log_offset_, log_divisor_;
    std::vector<double> window_;
    /** The mel filterbank, [mels, n_fft / 2 + 1] row-major. */
    std::vector<double> filterbank_;
    /** The bins [first, second) where each mel filter is not zero. */
    std::vector<std::pair<int, int>> bands_;
    RealFft fft_;
};

}  // namespace qwen3_asr
