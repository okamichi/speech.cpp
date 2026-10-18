#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

/** How far a result is from its reference: the largest difference, and the reference's power over the error's. */
struct Diff {
    double max_abs = 0;
    double snr_db = INFINITY;
    size_t n = 0;
};

template <typename T>
Diff compare(const T * got, const T * want, size_t n) {
    Diff d;
    d.n = n;
    double signal = 0, noise = 0;
    for (size_t i = 0; i < n; i++) {
        const double e = (double) got[i] - want[i];
        d.max_abs = std::max(d.max_abs, std::fabs(e));
        signal += (double) want[i] * want[i];
        noise += e * e;
    }
    d.snr_db = noise == 0 ? INFINITY : 10 * std::log10(signal / noise);
    return d;
}

template <typename T>
Diff compare(const std::vector<T> & got, const std::vector<T> & want) {
    return compare(got.data(), want.data(), std::min(got.size(), want.size()));
}

/**
 * Keeps in `worst` the comparison of lower SNR and says whether it took `d`. A comparison whose SNR is not a number,
 * which NaNs in a result give, is the worst, and stays so once kept.
 */
inline bool keep_worst(Diff & worst, const Diff & d) {
    if (std::isnan(worst.snr_db) || d.snr_db >= worst.snr_db) return false;
    worst = d;
    return true;
}

inline void print_diff(const std::string & what, const Diff & d) {
    std::printf("%-34s SNR %6.1f dB, max |diff| %.2e\n", what.c_str(), d.snr_db, d.max_abs);
}
