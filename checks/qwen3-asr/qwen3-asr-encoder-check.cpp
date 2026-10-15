// Checks the Qwen3-ASR encoder and projector against transformers' on each dump of reference/qwen3-asr/dump.py, from
// the dump's own features, so that the encoder's error is its own: the windows, then for each window its graph's
// convolutions, conv_out with the positions, every layer, ln_post and the projector, compared joined over the windows
// and the encoder's output per window, and the projector's output once more as encode() computes it. The dumps are
// those of the model, in <reference out dir>/<its general.name>/.
//
// usage: qwen3-asr-encoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "qwen3-asr/encoder.h"
#include "qwen3-asr/layout.h"
#include "reference-dumps.h"
#include "weight-types.h"

using namespace qwen3_asr;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32", "F16", "Q8_0"},
                           "its thresholds are those measured with them")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout, [](const std::string & name) { return name.rfind("dec.", 0) != 0; });
            Encoder encoder(model, backend);
            // Measured on an Apple M5 on 2026-10-06, the projector output's lowest SNR over the nine inputs of the 0.6B and
            // the 1.7B model: with F32 weights 85.6 and 94.4 dB on the CPU, and 44.4 and 50.2 dB on Metal, whose matrix
            // kernel rounds its inputs to half precision; with F16 weights 27.0 and 40.4 dB on the CPU, which rounds the
            // activations to half precision too, and 49.8 and 50.7 dB on Metal; with Q8_0 weights 20.2 and 24.3 dB on the
            // CPU, which quantizes the activations to 8 bits too, and 20.4 and 17.1 dB on Metal. The lowest are windows of
            // near-silence, en_us-2880067776280655708's first and en_us-10197164397713068203, whose last layers and
            // ln_post magnify any rounding 10 to 30 dB: there torch's own float32 is 86.1 dB (0.6B) and 92.6 dB (1.7B) from a
            // float64 run of the official encoder, where this port's float32 is 100.8 and 107.4 dB from it. With F32
            // weights on the CPU, a feed-forward with tanh's GELU instead of erf's gives 43.1 dB, and attention over the
            // whole utterance instead of its windows 7 to 11 dB.
            const ggml_type type = model.tensor("enc.blk.0.attn_q.weight")->type;
            const double threshold_db = type == GGML_TYPE_Q8_0 ? 12 : type == GGML_TYPE_F32 && ggml_backend_is_cpu(backend) ? 80 : 20;
            std::printf("projector output threshold: %.0f dB\n", threshold_db);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], model)) {
                const auto npy = [&](const char * name) { return read_npy((d / (std::string(name) + ".npy")).u8string()); };
                const Npy features = npy("features"), windows = npy("windows"), input = npy("encoder_input");
                const Npy layers = npy("encoder_layers"), output = npy("encoder_output"), embeds = npy("audio_embeds");
                const Npy convolutions[] = {npy("conv1"), npy("conv2"), npy("conv3")};
                const int64_t frames = features.shape[0];
                const std::vector<EncoderWindow> ours = encoder.windows(frames);
                std::printf("%s (%lld frames, %lld tokens, %zu windows)\n", d.filename().u8string().c_str(), (long long) frames,
                            (long long) encoder.tokens(frames), ours.size());
                bool same_windows = (int64_t) ours.size() + 1 == windows.shape[0] && encoder.tokens(frames) == embeds.shape[0];
                for (size_t i = 0; same_windows && i < ours.size(); i++) {
                    same_windows = ours[i].first_token == windows.i32[i] && ours[i].first_token + ours[i].tokens == windows.i32[i + 1];
                }
                if (!same_windows) {
                    std::printf("  the windows or the tokens differ from the dump's\n");
                    ok = false;
                    continue;
                }

                std::vector<std::vector<float>> got_convolutions(3), got_layers(layers.shape[0]);
                std::vector<float> got_input, got_output, got_embeds;
                for (size_t i = 0; i < ours.size(); i++) {
                    Graph g;
                    EncoderStages stages;
                    ggml_tensor * out = encoder.build(g, features.f32, {ours[i]}, &stages);
                    g.output(out);
                    g.output(stages.input);
                    g.output(stages.output);
                    for (ggml_tensor * t : stages.convolutions) g.output(t);
                    for (ggml_tensor * t : stages.layers) g.output(t);
                    g.compute(backend, allocr);
                    const auto append = [](std::vector<float> & to, const ggml_tensor * t) {
                        const std::vector<float> v = Graph::read(t);
                        to.insert(to.end(), v.begin(), v.end());
                    };
                    for (int c = 0; c < 3; c++) append(got_convolutions[c], stages.convolutions[c]);
                    for (size_t l = 0; l < stages.layers.size(); l++) append(got_layers[l], stages.layers[l]);
                    append(got_input, stages.input);
                    append(got_embeds, out);
                    const std::vector<float> window_output = Graph::read(stages.output);
                    got_output.insert(got_output.end(), window_output.begin(), window_output.end());
                    const size_t width = (size_t) encoder.d_model();
                    print_diff("  window " + std::to_string(i) + " encoder output",
                               compare(window_output.data(), &output.f32[(size_t) ours[i].first_token * width], window_output.size()));
                }
                for (int c = 0; c < 3; c++) {
                    print_diff("  convolution " + std::to_string(c + 1), compare(got_convolutions[c], convolutions[c].f32));
                    ok = ok && got_convolutions[c].size() == convolutions[c].f32.size();
                }
                print_diff("  encoder input", compare(got_input, input.f32));
                const size_t n = got_input.size();
                for (size_t l = 0; l < got_layers.size(); l++) {
                    print_diff("  layer " + std::to_string(l), compare(got_layers[l].data(), &layers.f32[l * n], n));
                }
                print_diff("  encoder output", compare(got_output, output.f32));
                const Diff projector = compare(got_embeds, embeds.f32);
                print_diff("  projector output", projector);

                const auto start = std::chrono::steady_clock::now();
                const std::vector<float> encoded = *encoder.encode(features.f32);
                const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
                const Diff whole = compare(encoded, embeds.f32);
                print_diff("  encode()", whole);
                std::printf("  encode() took %.3f s\n", seconds);
                ok = ok && n == input.f32.size() && got_embeds.size() == embeds.f32.size() && encoded.size() == embeds.f32.size() &&
                     projector.snr_db > threshold_db && whole.snr_db > threshold_db;
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
