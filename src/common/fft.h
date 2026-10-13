#pragma once

#include <complex>
#include <cstddef>
#include <vector>

/**
 * The discrete Fourier transform of real values, X[k] = Σ x[n] exp(-2πi k n / size) for k ≤ size / 2, in double
 * precision. The values at even and odd n are the real and imaginary parts of size / 2 complex values, whose transform
 * is Cooley and Tukey's decimation in time, iterated stage by stage over the factors of size / 2: fours, then twos, then
 * the other primes. A last pass separates the transforms of the even and the odd values and joins them. It takes any
 * even size, and is fast for sizes whose prime factors are small, such as NeMo's 512 = 2⁹ and Whisper's
 * 400 = 2⁴ × 5².
 */
class RealFft {
public:
    /** Throws std::logic_error unless size is even and positive. */
    explicit RealFft(int size);

    int size() const { return size_; }

    /** Transforms the size() values of `in` into the size() / 2 + 1 values of `out`; the two do not overlap. */
    void transform(const double * in, std::complex<double> * out) const;

private:
    struct Stage {
        /** The size of the transforms the stage joins and how many it joins. */
        size_t length, radix;
        /** Where the stage's twiddles start in twiddles_. */
        size_t twiddles;
        /** exp(-2πi t / radix) for t < radix, for a radix the stage has no butterfly of its own for, or empty. */
        std::vector<std::complex<double>> roots;
    };

    int size_;
    std::vector<Stage> stages_;
    /** Where the complex value of in[2 m] and in[2 m + 1] goes before the first stage: m with its digits reversed. */
    std::vector<size_t> order_;
    /** For each stage, exp(-2πi s j / (length radix)) for 1 ≤ s < radix at j (radix - 1) + s - 1, j < length. */
    std::vector<std::complex<double>> twiddles_;
    /** exp(-2πi k / size) for k ≤ size / 4, which joins the transforms of the even and the odd values. */
    std::vector<std::complex<double>> halves_;
};
