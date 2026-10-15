// Checks the speaker encoder and the duration predictor of Irodori-TTS against the official implementation
// on every dump of reference/irodori-tts/dump.py made with the file's model, each stage from the dump's own inputs:
// the reference latent for the speaker encoder, and the text, speaker and caption conditions for the duration
// predictor, the null speaker for a dump without a reference, the embedding's first token for one of an embedding, and
// the null caption for one without instructions. The length is checked against the frames the official runtime
// synthesized, with the dump's seconds, duration scale and speed: a dump with fixed seconds has no prediction, and only
// its length is checked.
//
// usage: irodori-condition-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-dumps.h"
#include "irodori-tts/duration.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/speaker-encoder.h"
#include "npy.h"
#include "weight-types.h"

using namespace irodori;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], model_layout, {"F32", "F16", "Q8_0"},
                           "its bounds are those measured with them")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        const ModelFile model(args[1], backend, model_layout);
        const SpeakerEncoder speaker(model);
        const DurationPredictor duration(model);
        const bool null_speaker = model.boolean("irodori-tts.duration.null_speaker");
        const bool caption = model.boolean("irodori-tts.caption_condition");
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        const std::vector<IrodoriDump> dumps = irodori_dumps(args[2], model.str("general.source.repo_url"), "dit_t.npy");
        bool ok = !dumps.empty();
        for (const IrodoriDump & d : dumps) {
            std::printf("%s\n", d.dir().filename().u8string().c_str());
            if (!d.reference() && !null_speaker) {
                std::printf("  skipped: the file lacks the null speaker that a request without a reference speaks with\n");
                continue;
            }
            if (d.caption() && !caption) {
                std::printf("  skipped: the file lacks the caption's encoder that instructions need\n");
                continue;
            }
            std::vector<float> summary;
            if (d.embedding()) {
                // An embedding is the speaker condition itself; its first token is the speaker's summary.
                const Npy state = read_npy(d.file("speaker_state.npy").u8string());
                summary.assign(state.f32.begin(), state.f32.begin() + state.shape[1]);
            } else if (d.reference()) {
                const Npy latent = read_npy(d.file("ref_latent.npy").u8string());
                const Npy encoded_ref = read_npy(d.file("speaker_encoded.npy").u8string());
                const Npy state_ref = read_npy(d.file("speaker_state.npy").u8string());
                Graph g;
                ggml_tensor * encoded = nullptr;
                ggml_tensor * state = speaker.build(g, latent.f32, &encoded);
                g.output(encoded);
                g.output(state);
                g.compute(backend, allocr);
                print_diff("  speaker encoder", compare(Graph::read(encoded), encoded_ref.f32));
                const Diff ds = compare(Graph::read(state), state_ref.f32);
                print_diff("  speaker condition", ds);
                // Measured on an Apple M5: 111 dB on the CPU in F32, 47 dB on Metal, 50 dB with F16 weights
                // and 25 dB with Q8_0. A wrong operation falls far below this bound.
                ok = ok && ds.snr_db > 20;
                summary.assign(state_ref.f32.begin(), state_ref.f32.begin() + state_ref.shape[1]);
            }
            const LengthOptions options = d.length();
            const int want = (int) d.number("latent_frames", -1);
            if (options.fixed()) {
                const int frames = duration.length(options, 0).frames;
                std::printf("  length: %g s at speed %g, %d frames (official %d)\n", options.seconds, options.speed, frames, want);
                ok = ok && frames == want;
                continue;
            }
            const Npy text = read_npy(d.file("text_state.npy").u8string());
            const Npy log_frames = read_npy(d.file("duration_log_frames.npy").u8string());
            Graph g;
            ggml_tensor * text_state = g.input(text.f32, text.shape[1], text.shape[0]);
            ggml_tensor * speaker_summary = d.reference() ? g.input(summary, (int64_t) summary.size()) : model.tensor("duration.null_speaker");
            ggml_tensor * caption_state = nullptr;
            if (d.caption()) {
                const Npy c = read_npy(d.file("caption_state.npy").u8string());
                caption_state = g.input(c.f32, c.shape[1], c.shape[0]);
            }
            ggml_tensor * sum = duration.build(g, text_state, speaker_summary, caption_state);
            g.output(sum);
            g.compute(backend, allocr);
            const float predicted = Graph::read(sum)[0];
            const int frames = duration.length(options, predicted).frames;
            std::printf("  duration%s%s: log(1 + frames) %.6f (official %.6f), scale %g at speed %g, %d frames (official %d)\n",
                        d.reference() ? "" : " with the null speaker", d.caption() ? " with the caption" : "", std::log1p(predicted), log_frames.f32[0], options.duration_scale,
                        options.speed, frames, want);
            // F32 and F16 give the official frames; Q8_0 weights move a 27 s text by one frame (40 ms).
            ok = ok && std::abs(frames - want) <= 1;
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
