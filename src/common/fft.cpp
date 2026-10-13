#include "fft.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace {

using Complex = std::complex<double>;

constexpr double kPi = 3.14159265358979323846;

/**
 * The product written out: std::complex's operator* checks for infinities and NaNs, which g++ compiles to a call
 * of __muldc3 for every product unless -ffast-math or -fcx-limited-range is given.
 */
inline Complex mul(Complex a, Complex b) {
    return {a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real()};
}

/** -i a. */
inline Complex rotate(Complex a) {
    return {a.imag(), -a.real()};
}

}  // namespace

RealFft::RealFft(int size) : size_(size) {
    if (size < 2 || size % 2 != 0) throw std::logic_error("a real Fourier transform of " + std::to_string(size) + " values");
    const size_t half = (size_t) size / 2;
    std::vector<size_t> radices;
    size_t n = half;
    for (; n % 4 == 0; n /= 4) radices.push_back(4);
    for (size_t p = 2; n > 1;) {
        if (n % p == 0) {
            radices.push_back(p);
            n /= p;
        } else {
            p = p * p > n ? n : p + 1;
        }
    }
    size_t length = 1;
    for (size_t p : radices) {
        Stage stage{length, p, twiddles_.size(), {}};
        for (size_t j = 0; j < length; j++) {
            for (size_t s = 1; s < p; s++) twiddles_.push_back(std::polar(1.0, -2 * kPi * (double) (s * j) / (double) (length * p)));
        }
        if (p != 2 && p != 4 && p != 5) {
            for (size_t t = 0; t < p; t++) stage.roots.push_back(std::polar(1.0, -2 * kPi * (double) t / (double) p));
        }
        stages_.push_back(std::move(stage));
        length *= p;
    }
    // The last stage joins the transforms of the values whose index has each remainder modulo its radix, the stage
    // before it those of the remainders modulo its own radix of what is left, and so on.
    order_.resize(half);
    for (size_t m = 0; m < half; m++) {
        size_t rest = m, span = half, at = 0;
        for (auto stage = stages_.rbegin(); stage != stages_.rend(); ++stage) {
            span /= stage->radix;
            at += rest % stage->radix * span;
            rest /= stage->radix;
        }
        order_[m] = at;
    }
    for (size_t k = 0; k <= half / 2; k++) halves_.push_back(std::polar(1.0, -2 * kPi * (double) k / size));
}

void RealFft::transform(const double * in, std::complex<double> * out) const {
    const size_t half = (size_t) size_ / 2;
    for (size_t m = 0; m < half; m++) out[order_[m]] = {in[2 * m], in[2 * m + 1]};

    // Each stage joins `radix` transforms of `length` values, at x[s length + j] for s < radix, into one of
    // length × radix values, X[j + q length] = Σ_s exp(-2πi s q / radix) exp(-2πi s j / (length radix)) x[s length + j].
    std::vector<Complex> a;
    for (const Stage & stage : stages_) {
        const size_t l = stage.length, p = stage.radix;
        const Complex * twiddles = twiddles_.data() + stage.twiddles;
        if (!stage.roots.empty()) a.resize(p);
        for (Complex * x = out; x < out + half; x += l * p) {
            for (size_t j = 0; j < l; j++) {
                const Complex * w = twiddles + j * (p - 1);
                if (p == 2) {
                    const Complex a0 = x[j], a1 = mul(x[l + j], w[0]);
                    x[j] = a0 + a1;
                    x[l + j] = a0 - a1;
                } else if (p == 4) {
                    const Complex a0 = x[j], a1 = mul(x[l + j], w[0]), a2 = mul(x[2 * l + j], w[1]), a3 = mul(x[3 * l + j], w[2]);
                    const Complex b0 = a0 + a2, b1 = a0 - a2, b2 = a1 + a3, b3 = rotate(a1 - a3);
                    x[j] = b0 + b2;
                    x[l + j] = b1 + b3;
                    x[2 * l + j] = b0 - b2;
                    x[3 * l + j] = b1 - b3;
                } else if (p == 5) {
                    // cos(2π/5), cos(4π/5), sin(2π/5) and sin(4π/5).
                    constexpr double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;
                    constexpr double s1 = 0.95105651629515357212, s2 = 0.58778525229247312917;
                    const Complex a0 = x[j], a1 = mul(x[l + j], w[0]), a2 = mul(x[2 * l + j], w[1]);
                    const Complex a3 = mul(x[3 * l + j], w[2]), a4 = mul(x[4 * l + j], w[3]);
                    const Complex b1 = a1 + a4, b2 = a2 + a3, d1 = a1 - a4, d2 = a2 - a3;
                    const Complex r1 = a0 + c1 * b1 + c2 * b2, r2 = a0 + c2 * b1 + c1 * b2;
                    const Complex i1 = rotate(s1 * d1 + s2 * d2), i2 = rotate(s2 * d1 - s1 * d2);
                    x[j] = a0 + b1 + b2;
                    x[l + j] = r1 + i1;
                    x[2 * l + j] = r2 + i2;
                    x[3 * l + j] = r2 - i2;
                    x[4 * l + j] = r1 - i1;
                } else {
                    a[0] = x[j];
                    for (size_t s = 1; s < p; s++) a[s] = mul(x[s * l + j], w[s - 1]);
                    for (size_t q = 0; q < p; q++) {
                        Complex sum = a[0];
                        for (size_t s = 1, t = q; s < p; s++, t = t + q < p ? t + q : t + q - p) sum += mul(a[s], stage.roots[t]);
                        x[q * l + j] = sum;
                    }
                }
            }
        }
    }

    // With Z the transform of z[m] = in[2 m] + i in[2 m + 1], the even values' transform is E[k] = (Z[k] +
    // conj(Z[half - k])) / 2 and the odd values' O[k] = (Z[k] - conj(Z[half - k])) / 2i, X[k] = E[k] +
    // exp(-2πi k / size) O[k] and X[half - k] = conj(E[k] - exp(-2πi k / size) O[k]).
    const Complex z0 = out[0];
    out[0] = z0.real() + z0.imag();
    out[half] = z0.real() - z0.imag();
    for (size_t k = 1; k <= half / 2; k++) {
        const Complex z = out[k], y = std::conj(out[half - k]);
        const Complex e = 0.5 * (z + y), o = mul(halves_[k], 0.5 * rotate(z - y));
        out[k] = e + o;
        out[half - k] = std::conj(e - o);
    }
}
