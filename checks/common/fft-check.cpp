// Checks the real FFT against a direct DFT: the sizes of the speech frontends, NeMo's 512 and Whisper's 400, and
// sizes whose halves take each kind of stage, on varied values, an impulse and a constant.
//
// usage: fft-check

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include "fft.h"

namespace {

/** A sum whose rounding errors are carried along (Neumaier's), so that its error does not grow with the terms. */
class Sum {
public:
    void add(double v) {
        const double t = sum_ + v;
        carry_ += std::fabs(sum_) >= std::fabs(v) ? (sum_ - t) + v : (v - t) + sum_;
        sum_ = t;
    }
    double value() const { return sum_ + carry_; }

private:
    double sum_ = 0, carry_ = 0;
};

/** X[k] for k ≤ n / 2, each term's angle from (i k) mod n, which keeps it below 2π. */
std::vector<std::complex<double>> direct(const std::vector<double> & x) {
    const size_t n = x.size();
    std::vector<std::complex<double>> out;
    for (size_t k = 0; k <= n / 2; k++) {
        Sum re, im;
        for (size_t i = 0; i < n; i++) {
            const double angle = -2 * 3.14159265358979323846 * (double) (i * k % n) / (double) n;
            re.add(x[i] * std::cos(angle));
            im.add(x[i] * std::sin(angle));
        }
        out.push_back({re.value(), im.value()});
    }
    return out;
}

}  // namespace

int main() {
    bool ok = true;
    // Halves of 1, 3, 5, 7, 8 = 4 × 2, 21 = 3 × 7, 25 = 5 × 5, 200 = 4 × 2 × 5 × 5 and 256 = 4⁴.
    for (int n : {2, 6, 10, 14, 16, 42, 50, 400, 512}) {
        const RealFft fft(n);
        int log2_size = 0;
        for (int m = n; m > 1; m /= 2) log2_size++;
        for (int pattern = 0; pattern < 3; pattern++) {
            std::vector<double> input(n);
            for (int i = 0; i < n; i++) {
                if (pattern == 0) input[i] = std::sin(i * 1.37) + 0.5 * std::cos(i * i * 0.79);
                if (pattern == 1) input[i] = i == n / 3 ? 1.0 : 0.0;
                if (pattern == 2) input[i] = 0.75;
            }
            std::vector<std::complex<double>> output(n / 2 + 1);
            fft.transform(input.data(), output.data());
            const std::vector<std::complex<double>> expected = direct(input);
            double error = 0, norm = 0;
            for (size_t k = 0; k < expected.size(); k++) {
                error += std::norm(output[k] - expected[k]);
                norm += std::norm(expected[k]);
            }
            const double relative = std::sqrt(error / norm) / 2.220446049250313e-16;
            std::printf("size %d pattern %d: error %.2f eps of the norm\n", n, pattern, relative);
            // A radix-2 FFT with accurate twiddles errs by at most about 6 ε log2(n) in the norm (Higham, Accuracy and
            // Stability of Numerical Algorithms, 2002, chapter 24); the stages of 4 and 5 and the separation of the
            // halves round no more often than radix 2's log2(n) stages would. Measured on an Apple M5 on 2026-10-13:
            // 3.1 ε at most, against 54 ε allowed for 512 values. A wrong twiddle, order or separation errs by a
            // sizable fraction of the norm.
            ok = ok && relative < 6 * log2_size;
        }
    }
    return ok ? 0 : 1;
}
