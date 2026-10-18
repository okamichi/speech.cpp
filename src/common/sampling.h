#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "error.h"

/** The tokens a draw picks from, by falling logit, and a weight of each in proportion to its probability. */
struct Candidates {
    std::vector<int32_t> tokens;
    std::vector<double> weights;
};

/**
 * A number in [0, 1] from the next 64 bits of `rng`, rounded to a double; the largest draws round to 1. It is defined
 * here rather than by a distribution of the standard library, whose results differ between libc++, libstdc++ and
 * Microsoft's, so that a seed draws the same numbers on every platform.
 */
double uniform(std::mt19937_64 & rng);

/**
 * The index of the first weight whose cumulative share of the sum passes `u`, the last where none does (u of 1):
 * the shares are the weights divided by their sum, accumulated in order in double.
 */
int pick(const std::vector<double> & weights, double u);

/** The Error of a temperature that pushes the largest logit beyond the range of a float, naming "temperature". */
Error temperature_overflow(float temperature);
