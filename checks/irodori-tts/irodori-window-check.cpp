// Checks the rule by which Irodori-TTS's decoder sizes its windows (next_window()): each way it decides, the first
// window of a speech shorter than it among them, and simulated machines that decode a frame in a set time: one as fast
// as an Apple M5, one half as fast, one that slows down after the first window, and one slower than real time. Each
// speaks a sentence of 9 frames, one of 145 (5.8 s) and one of 685 (27.4 s) as Codec::decode() would, its estimate of
// the time per frame made as decode() makes it, and the listener plays from the first window on. It fails unless the
// first window is the rule's or the frames left and every later one lies from the floor to the ceiling or is the frames
// left; the fast
// machine keeps 0.7.1's 48 frames with the margin in hand; the slow one never runs dry, where 48-frame windows do; the
// one that slows to where 24-frame windows are slower than real time and 48-frame ones are not keeps the ceiling and
// never runs dry; and the one slower than real time takes the ceiling.
//
// usage: irodori-window-check

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "irodori-tts/synthesizer.h"

using namespace irodori;

namespace {

/** A machine: the seconds it takes to decode a frame, a window's margins included, at a time from the start. */
struct Machine {
    const char * name;
    std::function<double(double now)> seconds_per_frame;
};

/** The windows of one sentence, the least audio the listener had when a window arrived, and the time it had none. */
struct Played {
    std::vector<int> windows;
    double least = INFINITY, dry = 0;
};

Played play(const WindowRule & rule, const Machine & machine, int64_t frames, bool adaptive) {
    Played out;
    double now = 0, first_sent = 0, sent = 0, estimate = 0;
    for (int64_t a = 0; a < frames;) {
        const DecodeProgress progress{a, frames - a, sent, a == 0 ? 0 : now - first_sent, estimate};
        const int window = adaptive || a == 0 ? next_window(rule, progress) : (int) std::min<int64_t>(rule.ceiling, frames - a);
        out.windows.push_back(window);
        const int64_t from = std::max<int64_t>(0, a - rule.context), to = std::min(frames, a + window + rule.context);
        const double per_frame = machine.seconds_per_frame(now);
        now += per_frame * (double) (to - from);
        estimate = next_estimate(estimate, per_frame);
        if (a == 0) {
            first_sent = now;
        } else {
            const double left = sent - (now - first_sent);
            out.least = std::min(out.least, left);
            if (left < 0) out.dry -= left;
        }
        sent += window * rule.frame_seconds;
        a += window;
    }
    return out;
}

std::string list(const std::vector<int> & windows) {
    std::string out;
    for (int w : windows) out += (out.empty() ? "" : " ") + std::to_string(w);
    return out;
}

}  // namespace

int main() {
    // The synthesizer's rule for Irodori-TTS's codec: frames of 1920 samples at 48 kHz, and a decoder whose receptive
    // field reaches 8 frames.
    const WindowRule rule = Synthesizer::window_rule(1920, 48000, 8);
    // An Apple M5 decoded the 68 frames of a 48-frame window in 0.288 s through 0.7.1's worker (2026-10-07).
    const double m5 = 0.288 / 68;
    const Machine machines[] = {
        {"as fast as an Apple M5", [&](double) { return m5; }},
        {"half as fast", [&](double) { return 2 * m5; }},
        // From its third window on, which starts after 0.38 s, a sixth as fast: faster than real time with 48 frames and
        // slower with 24.
        {"as fast, then a sixth as fast", [&](double now) { return now < 0.3 ? m5 : 6 * m5; }},
        {"slower than real time", [&](double) { return 0.06; }},
    };
    bool ok = true;
    // Each way the rule decides, on a rule of its own: 12 frames first, then 24 to 48 frames of 0.04 s, margins of 10,
    // 0.1 s in hand.
    const WindowRule own{12, 24, 48, 10, 0.04, 0.1};
    const struct {
        const char * what;
        DecodeProgress progress;
        int want;
    } cases[] = {
        {"the first window", {0, 100, 0, 0, 0}, 12},
        {"the first window of a speech shorter than it", {0, 9, 0, 0, 0}, 9},
        {"time in hand for the ceiling", {12, 100, 2.4, 0.4, 0.005}, 48},
        {"time for 33 frames and some", {12, 100, 0.48, 0.0, 0.0071}, 33},
        {"no time for the floor, decoding faster than real time at it", {12, 100, 0.48, 0.45, 0.005}, 24},
        {"no time for the floor, decoding slower than real time at it", {12, 100, 0.48, 0.45, 0.03}, 48},
        {"fewer frames left than the floor", {60, 10, 0.48, 0.0, 0.005}, 10},
        {"fewer frames left than fit", {60, 30, 2.4, 0.4, 0.005}, 30},
    };
    for (const auto & c : cases) {
        const int got = next_window(own, c.progress);
        std::printf("%s: %d frames (want %d)\n", c.what, got, c.want);
        ok = ok && got == c.want;
    }
    for (int64_t frames : {9, 145, 685}) {
        for (const Machine & m : machines) {
            const Played adaptive = play(rule, m, frames, true), fixed = play(rule, m, frames, false);
            std::printf("%lld frames, %s:\n  adaptive %s; least in hand %.3f s, dry %.3f s\n  48 a window %s; least in hand %.3f s, dry %.3f s\n",
                        (long long) frames, m.name, list(adaptive.windows).c_str(), adaptive.least, adaptive.dry, list(fixed.windows).c_str(),
                        fixed.least, fixed.dry);
            int64_t left = frames;
            for (size_t i = 0; i < adaptive.windows.size(); i++) {
                const int w = adaptive.windows[i];
                ok = ok && (i == 0 ? w == std::min<int64_t>(rule.first, frames) : (w >= rule.floor && w <= rule.ceiling) || w == left);
                left -= w;
            }
            // A speech within the first window has no later ones to size.
            if (frames <= rule.first) continue;
            const std::string name = m.name;
            if (name == "as fast as an Apple M5") ok = ok && adaptive.windows == fixed.windows && adaptive.least >= rule.margin;
            if (name == "half as fast") ok = ok && adaptive.dry == 0 && fixed.dry > 0;
            if (name == "as fast, then a sixth as fast") ok = ok && adaptive.dry == 0 && adaptive.windows == fixed.windows;
            if (name == "slower than real time") {
                for (size_t i = 1; i + 1 < adaptive.windows.size(); i++) ok = ok && adaptive.windows[i] == rule.ceiling;
            }
        }
    }
    std::printf(ok ? "the rule holds on every machine\n" : "FAIL: the windows break the rule on a machine above\n");
    return ok ? 0 : 1;
}
