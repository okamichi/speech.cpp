#include "sampling.h"

#include "json.h"

double uniform(std::mt19937_64 & rng) {
    return (double) rng() * 0x1.0p-64;
}

int pick(const std::vector<double> & weights, double u) {
    double sum = 0;
    for (double w : weights) sum += w;
    const int last = (int) weights.size() - 1;
    double cumulative = 0;
    for (int i = 0; i < last; i++) {
        cumulative += weights[i] / sum;
        if (cumulative > u) return i;
    }
    return last;
}

Error temperature_overflow(float temperature) {
    return Error(Fault::OutOfRange,
                 "the temperature " + json_number(temperature) + " pushes the logits beyond the range of a float; give a temperature nearer 1",
                 "temperature");
}
