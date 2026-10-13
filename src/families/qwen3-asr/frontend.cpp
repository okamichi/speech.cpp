#include "frontend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>

namespace qwen3_asr {

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
      n_fft_((int) m.u32("qwen3-asr.frontend.n_fft")),
      hop_((int) m.u32("qwen3-asr.frontend.hop_length")),
      mels_((int) m.u32("qwen3-asr.frontend.n_mels")),
      min_samples_((int) m.u32("qwen3-asr.audio.min_samples")),
      log_floor_(m.f32("qwen3-asr.frontend.log_floor")),
      dynamic_range_(m.f32("qwen3-asr.frontend.dynamic_range")),
      log_offset_(m.f32("qwen3-asr.frontend.log_offset")),
      log_divisor_(m.f32("qwen3-asr.frontend.log_divisor")),
      window_(read_tensor(m, "frontend.window")),
      filterbank_(read_tensor(m, "frontend.filterbank")),
      fft_(n_fft_) {
    const int bins = n_fft_ / 2 + 1;
    for (int b = 0; b < mels_; b++) {
        int first = 0, last = bins;
        while (first < bins && filterbank_[(size_t) b * bins + first] == 0) first++;
        while (last > first && filterbank_[(size_t) b * bins + last - 1] == 0) last--;
        bands_.push_back({first, last});
    }
}

int64_t Frontend::frames(size_t samples) const {
    // The STFT gives one frame per hop and one more, which the extractor drops.
    return (int64_t) std::max(samples, (size_t) min_samples_) / hop_;
}

std::vector<float> Frontend::normalize(std::vector<float> samples) {
    // float_range_normalize() divides in float32 by a peak above 1, then clips to [-1, 1].
    float peak = 0;
    for (float s : samples) peak = std::max(peak, std::fabs(s));
    for (float & s : samples) s = std::clamp(peak > 1.0f ? s / peak : s, -1.0f, 1.0f);
    return samples;
}

std::vector<float> Frontend::features(const std::vector<float> & samples) const {
    std::vector<float> x(std::max(samples.size(), (size_t) min_samples_), 0.0f);
    std::copy(samples.begin(), samples.end(), x.begin());

    const int64_t frames = this->frames(samples.size()), n = (int64_t) x.size();
    const int bins = n_fft_ / 2 + 1;
    std::vector<double> mel((size_t) (frames * mels_));
    std::vector<double> frame(n_fft_);
    std::vector<std::complex<double>> spectrum(bins);
    std::vector<double> power(bins);
    double largest = -INFINITY;
    for (int64_t f = 0; f < frames; f++) {
        // center=True with pad_mode="reflect": sample -i is sample i, and sample n - 1 + i is sample n - 1 - i.
        const int64_t start = f * hop_ - n_fft_ / 2;
        for (int k = 0; k < n_fft_; k++) {
            int64_t i = start + k;
            if (i < 0) i = -i;
            if (i >= n) i = 2 * (n - 1) - i;
            frame[k] = x[i] * window_[k];
        }
        fft_.transform(frame.data(), spectrum.data());
        for (int k = 0; k < bins; k++) power[k] = std::norm(spectrum[k]);
        for (int m = 0; m < mels_; m++) {
            double sum = 0;
            for (int k = bands_[m].first; k < bands_[m].second; k++) sum += filterbank_[(size_t) m * bins + k] * power[k];
            const double value = std::log10(std::max(sum, (double) log_floor_));
            mel[(size_t) (f * mels_ + m)] = value;
            largest = std::max(largest, value);
        }
    }
    std::vector<float> out(mel.size());
    const double floor = largest - dynamic_range_;
    for (size_t i = 0; i < mel.size(); i++) out[i] = (float) ((std::max(mel[i], floor) + log_offset_) / log_divisor_);
    return out;
}

}  // namespace qwen3_asr
