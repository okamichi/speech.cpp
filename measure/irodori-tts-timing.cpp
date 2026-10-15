// Times Irodori-TTS's stages on each text given, in the voice of a reference WAVE file, of a voice file or of none,
// from seed 1 with the model's steps: the voice's encoding once, then for each text the median over the runs of the
// whole synthesis, its time to first audio, the text and the condition (the text encoder, the caption's where there is
// one, and the duration predictor), the DiT's sampling, the codec's decoding, and its first window, the time from the
// end of the sampling to the first audio, and the RTFx, the seconds of audio over the median synthesis's. One
// synthesis of the first text goes first, untimed, so that the times leave out the compilation of a GPU's kernels. It
// prints one JSON object per line.
//
// usage: irodori-tts-timing <model.gguf> <gpu|cpu|device name> <runs> <reference.wav|voice file|none> <text>...

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "irodori-tts/synthesizer.h"
#include "json.h"

using namespace irodori;

namespace {

using Clock = std::chrono::steady_clock;

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 6) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <runs> <reference.wav|voice file|none> <text>...\n", args[0].c_str());
        return 2;
    }
    try {
        const int runs = std::stoi(args[3]);
        if (runs < 1) throw std::runtime_error("runs must be positive");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(args[2]), ggml_backend_free);
        Synthesizer synth(args[1], backend.get());
        const auto start = Clock::now();
        const Voice voice = args[4] == "none" ? Voice{} : synth.load_voice(args[4]);
        std::printf("{\"voice\":%s,\"device\":%s,\"attention\":%s,\"encoding\":%.3f}\n", json_string(args[4]).c_str(),
                    json_string(ggml_backend_name(backend.get())).c_str(), json_string(attention_name(synth.attention())).c_str(),
                    std::chrono::duration<double>(Clock::now() - start).count());
        Request request;
        request.seed = 1;
        request.text = args[5];
        const auto ignore = [](const float *, size_t) { return true; };
        synth.synthesize(request, voice, ignore);
        for (size_t a = 5; a < args.size(); a++) {
            request.text = args[a];
            std::vector<double> total, first_audio, text, sampling, codec, first_window;
            Stats stats;
            for (int r = 0; r < runs; r++) {
                stats = Stats{};
                const auto begin = Clock::now();
                synth.synthesize(request, voice, ignore, &stats);
                total.push_back(std::chrono::duration<double>(Clock::now() - begin).count());
                first_audio.push_back(stats.first_audio);
                text.push_back(stats.text);
                sampling.push_back(stats.sampling);
                codec.push_back(stats.codec);
                first_window.push_back(stats.first_audio - stats.text - stats.sampling);
            }
            const double seconds = (double) stats.samples / synth.sample_rate();
            std::printf("{\"text\":%s,\"tokens\":%d,\"frames\":%d,\"seconds\":%.2f,\"total\":%.3f,\"time_to_first_audio\":%.3f,\"text_and_condition\":%.4f,"
                        "\"sampling\":%.3f,\"codec\":%.3f,\"first_window\":%.4f,\"rtfx\":%.2f}\n",
                        json_string(request.text).c_str(), stats.tokens, stats.frames, seconds, median(total), median(first_audio),
                        median(text), median(sampling), median(codec), median(first_window), seconds / median(total));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
