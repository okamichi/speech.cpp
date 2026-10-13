#include "frontend.h"

#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>

#include "error.h"
namespace fastconformer {

namespace {

/** A tensor that the layout stores in F32, as doubles. */
std::vector<double> read_tensor(const ModelFile & m, const std::string & name) {
    ggml_tensor * t = m.tensor(name);
    std::vector<float> f(ggml_nelements(t));
    ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    return std::vector<double>(f.begin(), f.end());
}

}  // namespace

Frontend::Frontend(const ModelFile & m)
    : sample_rate_((int) m.u32("speech.sample_rate")),
      n_fft_((int) m.u32("fastconformer.frontend.n_fft")),
      hop_((int) m.u32("fastconformer.frontend.hop_length")),
      mels_((int) m.u32("fastconformer.frontend.n_mels")),
      preemphasis_(m.f32("fastconformer.frontend.preemphasis")),
      log_guard_(m.f32("fastconformer.frontend.log_guard")),
      std_guard_(m.f32("fastconformer.frontend.std_guard")),
      fft_(n_fft_) {
    const std::vector<double> w = read_tensor(m, "frontend.window");
    const int64_t window = (int64_t) w.size();
    window_.assign(n_fft_, 0.0);
    // torch.stft() centres a window shorter than n_fft in it.
    const int64_t left = (n_fft_ - window) / 2;
    for (int64_t i = 0; i < window; i++) window_[left + i] = w[i];
    filterbank_ = read_tensor(m, "frontend.filterbank");
    const int bins = n_fft_ / 2 + 1;
    for (int b = 0; b < mels_; b++) {
        int first = 0, last = bins;
        while (first < bins && filterbank_[(size_t) b * bins + first] == 0) first++;
        while (last > first && filterbank_[(size_t) b * bins + last - 1] == 0) last--;
        bands_.push_back({first, last});
    }
}

int64_t Frontend::frames(size_t samples) const {
    // FilterbankFeatures.get_seq_len(): the STFT gives one frame more, which the official code masks out.
    return (int64_t) samples / hop_;
}

std::vector<float> Frontend::features(const std::vector<float> & samples) const {
    const int64_t frames = this->frames(samples.size());
    // normalize_batch() divides by frames - 1.
    if (frames < 2) {
        throw Error(Fault::OutOfRange, "the audio has " + std::to_string(samples.size()) + " samples at " + std::to_string(sample_rate_) +
                                           " Hz; recognition needs at least " + std::to_string(2 * hop_),
                    "audio");
    }
    // Pre-emphasis in float32, as FilterbankFeatures.forward() computes it.
    std::vector<float> x(samples.size());
    x[0] = samples[0];
    for (size_t i = 1; i < samples.size(); i++) x[i] = samples[i] - preemphasis_ * samples[i - 1];

    const int bins = n_fft_ / 2 + 1;
    const int64_t n = (int64_t) x.size();
    std::vector<double> mel((size_t) (frames * mels_));
    std::vector<double> frame(n_fft_);
    std::vector<std::complex<double>> spectrum(bins);
    std::vector<double> power(bins);
    for (int64_t f = 0; f < frames; f++) {
        // center=True with pad_mode="constant": the signal has n_fft / 2 zeros before it.
        const int64_t start = f * hop_ - n_fft_ / 2;
        for (int k = 0; k < n_fft_; k++) {
            const int64_t i = start + k;
            frame[k] = i >= 0 && i < n ? x[i] * window_[k] : 0.0;
        }
        fft_.transform(frame.data(), spectrum.data());
        for (int k = 0; k < bins; k++) power[k] = std::norm(spectrum[k]);
        for (int m = 0; m < mels_; m++) {
            double sum = 0;
            for (int k = bands_[m].first; k < bands_[m].second; k++) sum += filterbank_[(size_t) m * bins + k] * power[k];
            mel[(size_t) (f * mels_ + m)] = std::log(sum + (double) log_guard_);
        }
    }

    // normalize_batch(normalize_type="per_feature"): the unbiased standard deviation, plus a guard.
    std::vector<float> out(mel.size());
    for (int m = 0; m < mels_; m++) {
        double mean = 0;
        for (int64_t f = 0; f < frames; f++) mean += mel[(size_t) (f * mels_ + m)];
        mean /= (double) frames;
        double var = 0;
        for (int64_t f = 0; f < frames; f++) {
            const double d = mel[(size_t) (f * mels_ + m)] - mean;
            var += d * d;
        }
        const double std = std::sqrt(var / (double) (frames - 1)) + (double) std_guard_;
        for (int64_t f = 0; f < frames; f++) {
            out[(size_t) (f * mels_ + m)] = (float) ((mel[(size_t) (f * mels_ + m)] - mean) / std);
        }
    }
    return out;
}

}  // namespace fastconformer
