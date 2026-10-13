// Checks the FastConformer encoder against NeMo's ConformerEncoder on each dump of
// reference/fastconformer/dump.py: the subsampling, every conformer layer and the output, all from the dump's
// own features, so that the encoder's error is its own. The dumps are those of the model, in
// <reference out dir>/<its general.name>/.
//
// usage: fastconformer-encoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "reference-dumps.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "npy.h"

using namespace fastconformer;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout);
            const Encoder encoder(model, backend);
            // Measured on an Apple M5, the encoder output's lowest SNR: parakeet-tdt_ctc-0.6b-ja (three utterances,
            // 2026-10-05) 113.8 dB on the CPU with F32 weights, 54.6 dB with F16 and 63.1 dB on Metal with either;
            // parakeet-tdt-0.6b-v3 (twelve, 2026-10-06) 105.9, 31.4 and 52.0 dB; reazonspeech-nemo-v2 (eight and two
            // of 65 and 311 s, 2026-10-06) 95.9, 29.7 and 37.7 dB. Metal's matrix kernel rounds its inputs to half
            // precision and sums in float32, and ggml's CPU dot product with F16 weights also sums in half precision on
            // ARM (vfmaq_f16). Half precision loses most where v3's layers carry values of 250 to 500, and in the
            // near-silent frames between the utterances of reazonspeech's long inputs, where the error grows from
            // 70 dB at the first layer to 42 dB at the thirteenth in a few frames; the text stays NeMo's. In float32 on
            // the CPU a wrong operation shows at once: local attention whose band misses one frame on one side gives
            // 27 to 50 dB, so that case takes 90 dB, and the half-precision ones 25 dB.
            const bool float32 = ggml_backend_is_cpu(backend) && model.tensor("blk.0.ff1_up.weight")->type == GGML_TYPE_F32;
            const double threshold_db = float32 ? 90 : 25;
            std::printf("encoder output threshold: %.0f dB\n", threshold_db);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], model)) {
                const Npy features = read_npy((d / "features.npy").u8string());
                const Npy subsampled = read_npy((d / "subsampled.npy").u8string());
                const Npy layers = read_npy((d / "layers.npy").u8string());
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const int64_t frames = features.shape[0];
                Graph g;
                EncoderStages stages;
                ggml_tensor * out = encoder.build(g, features.f32, frames, &stages);
                g.output(out);
                g.output(stages.subsampled);
                for (ggml_tensor * t : stages.layers) g.output(t);
                g.compute(backend, allocr);

                const int64_t t = encoder.subsampled_frames(frames);
                const size_t width = (size_t) encoder.d_model(), n = (size_t) t * width;
                std::printf("%s (%lld frames, %lld after subsampling)\n", d.filename().u8string().c_str(), (long long) frames,
                            (long long) t);
                ok = ok && t == subsampled.shape[0] && t == encoded.shape[0] && (size_t) layers.shape[0] == stages.layers.size();
                print_diff("  subsampling", compare(Graph::read(stages.subsampled), subsampled.f32));
                for (size_t l = 0; l < stages.layers.size(); l++) {
                    const Diff dl = compare(Graph::read(stages.layers[l]).data(), &layers.f32[l * n], n);
                    print_diff("  layer " + std::to_string(l), dl);
                }
                const Diff de = compare(Graph::read(out), encoded.f32);
                print_diff("  encoder output", de);
                ok = ok && de.snr_db > threshold_db;
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
