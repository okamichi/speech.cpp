#pragma once

#include <complex>
#include <utility>
#include <vector>

#include "fft.h"
#include "model-file.h"

namespace fastconformer {

/**
 * NeMo's AudioToMelSpectrogramPreprocessor as it runs in evaluation, without dither (FilterbankFeatures.forward()
 * and normalize_batch() in nemo/collections/asr/parts/preprocessing/features.py): pre-emphasis, a centred STFT
 * with zero padding, the power through the checkpoint's mel filterbank, the log with its additive guard, and the
 * normalization of each mel bin over the utterance.
 *
 * It runs on the host in double precision rather than in a ggml graph: it costs a few milliseconds, and
 * Metal's matrix kernel rounds its inputs to half precision, which the log of the quiet frames' power would
 * magnify.
 */
class Frontend {
public:
    explicit Frontend(const ModelFile & m);

    /** The features of mono samples at sample_rate(), [frames, mels] row-major (ggml [mels, frames]). */
    std::vector<float> features(const std::vector<float> & samples) const;

    /** The frames features() gives for that many samples. */
    int64_t frames(size_t samples) const;

    int sample_rate() const { return sample_rate_; }
    int hop_length() const { return hop_; }
    int mels() const { return mels_; }

private:
    int sample_rate_, n_fft_, hop_, mels_;
    float preemphasis_, log_guard_, std_guard_;
    /** The window zero-padded to n_fft and centred, as torch.stft() pads it. */
    std::vector<double> window_;
    /** The mel filterbank, [mels, n_fft / 2 + 1] row-major. */
    std::vector<double> filterbank_;
    /** The bins [first, second) where each mel filter is not zero. */
    std::vector<std::pair<int, int>> bands_;
    RealFft fft_;
};

}  // namespace fastconformer
