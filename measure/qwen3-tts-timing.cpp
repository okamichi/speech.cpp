// Times Qwen3-TTS's stages on each text given, spoken in the model's first voice with the language left to the model
// and the model file's sampling from seed 1: for each, the median over the runs of the whole synthesis, its time to
// first audio, the prompt (the text's tokens and the prompt's rows), the talker's prefill, the talker's steps, the code
// predictor and the codec, and of the talker's step and the code predictor's frame each per frame, and the RTFx, the
// seconds of audio over the median synthesis's. One synthesis of the first text goes first, untimed, so that the times
// leave out the compilation of a GPU's kernels. It prints one JSON object per line.
//
// usage: qwen3-tts-timing <model.gguf> <gpu|cpu|device name> <runs> <text>...

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "json.h"
#include "qwen3-tts/synthesizer.h"

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
    if (args.size() < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <runs> <text>...\n", args[0].c_str());
        return 2;
    }
    try {
        const int runs = std::stoi(args[3]);
        if (runs < 1) throw std::runtime_error("runs must be positive");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(init_backend(args[2]), ggml_backend_free);
        Synthesizer synth(args[1], backend.get());
        SynthesisRequest request;
        request.speaker = synth.ids().voices.at(0);
        request.seed = 1;
        request.text = args[4];
        synth.synthesize(request, [](const float *, size_t) { return true; });
        for (size_t a = 4; a < args.size(); a++) {
            request.text = args[a];
            std::vector<double> total, first_audio, prompt, prefill, talker, talker_frame, code_predictor, code_predictor_frame, codec;
            int frames = 0;
            size_t samples = 0;
            for (int r = 0; r < runs; r++) {
                SynthesisStats stats;
                samples = 0;
                const auto start = Clock::now();
                Clock::time_point first;
                frames = synth.synthesize(request, [&](const float *, size_t n) {
                    if (samples == 0) first = Clock::now();
                    samples += n;
                    return true;
                }, &stats).frames;
                total.push_back(std::chrono::duration<double>(Clock::now() - start).count());
                first_audio.push_back(std::chrono::duration<double>(first - start).count());
                prompt.push_back(stats.prompt);
                prefill.push_back(stats.prefill);
                talker.push_back(stats.talker);
                code_predictor.push_back(stats.code_predictor);
                codec.push_back(stats.codec);
                // The talker steps once after each frame, the last one's step giving the logits that end the speech.
                talker_frame.push_back(stats.talker / frames);
                code_predictor_frame.push_back(stats.code_predictor / frames);
            }
            const double seconds = (double) samples / synth.sample_rate();
            std::printf("{\"text\":%s,\"voice\":%s,\"device\":%s,\"frames\":%d,\"seconds\":%.2f,\"total\":%.3f,\"time_to_first_audio\":%.3f,"
                        "\"prompt\":%.4f,\"prefill\":%.4f,\"talker\":%.3f,\"talker_per_frame\":%.4f,\"code_predictor\":%.3f,"
                        "\"code_predictor_per_frame\":%.4f,\"codec\":%.3f,\"rtfx\":%.2f}\n",
                        json_string(request.text).c_str(), json_string(request.speaker).c_str(),
                        json_string(ggml_backend_name(backend.get())).c_str(), frames, seconds, median(total), median(first_audio),
                        median(prompt), median(prefill), median(talker), median(talker_frame), median(code_predictor),
                        median(code_predictor_frame), median(codec), seconds / median(total));
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
