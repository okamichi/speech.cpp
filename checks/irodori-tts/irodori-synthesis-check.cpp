// Checks the whole synthesis of Irodori-TTS against the official runtime: for every dump of
// reference/irodori-tts/dump.py made with the same model, the dump's text spoken in the voice of the dump's
// reference latent, of its embedding, or without a reference, from the dump's noise and with its request's length,
// steps, guidance and cut at the tail, against the audio the official synthesize() returned, the cut at the tail
// included. The DiT attends as it chooses on the device, or as the last argument forces it, so that both attentions are
// checked on a GPU.
//
// usage: irodori-synthesis-check <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-dumps.h"
#include "irodori-tts/synthesizer.h"
#include "npy.h"

using namespace irodori;

namespace {

/**
 * The least SNR against the official that a dump's audio reaches. Measured on an Apple M5 on 2026-10-07: 72 to 116 dB on
 * the CPU in F32. On Metal the sampler carries the half-precision rounding of Metal's matrix kernel into the latent, and
 * the audio lies 22 to 63 dB from the official (11 to 29 dB with Q8_0 weights): the same speech, not the same waveform.
 * The guidance's branch without the caption carries it further: rf-weather-caption lies 11.8 dB from the official on
 * Metal and 88 dB on the CPU, while each of its DiT steps lies 44 dB or more from the official's. A wrong stage gives a
 * few dB.
 */
double least_snr_db(const std::string & dump) {
    // rf-weather-cfg's speaker scale of 7, guiding from t 0.9 down to 0.3, carries the rounding furthest: its audio lies
    // 8.7 dB from the official on Metal and 72 dB on the CPU, and the same request at speaker scales of 3 and 1 lies 24
    // and 28 dB from it on Metal. Each of its DiT steps, run on Metal from the dump's own inputs, lies 42 dB or more from
    // the official's, the least at t 0.32, the last guided step, and ReazonSpeech hears the same sentence in both audios.
    if (dump == "rf-weather-cfg") return 5;
    return 10;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]\n", args[0].c_str());
        return 2;
    }
    try {
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        Synthesizer synth(args[1], backend, forced_attention(args, 4));
        std::printf("DiT: %s\n", attention_name(synth.attention()));
        const bool meanflow = synth.model().str("irodori-tts.flow") == "meanflow";
        const std::vector<IrodoriDump> dumps = irodori_dumps(args[2], synth.model().str("general.source.repo_url"), "audio.npy");
        bool ok = !dumps.empty();
        for (const IrodoriDump & d : dumps) {
            const std::string name = d.dir().filename().u8string();
            if (!d.reference() && !synth.has_null_speaker()) {
                std::printf("%s: skipped, the file lacks the null speaker that a request without a reference speaks with\n", name.c_str());
                continue;
            }
            if (d.caption() && !synth.has_caption()) {
                std::printf("%s: skipped, the file lacks the caption's encoder that instructions need\n", name.c_str());
                continue;
            }
            Voice voice;
            if (d.embedding()) {
                const Npy state = read_npy(d.file("speaker_state.npy").u8string());
                voice.speaker = state.f32;
                voice.speaker_tokens = (int) state.shape[0];
            } else if (d.reference()) {
                voice = synth.voice_from_latent(read_npy(d.file("ref_latent.npy").u8string()).f32);
            }
            const Npy official = read_npy(d.file("audio.npy").u8string());
            Request r;
            r.text = d.text();
            r.caption = d.instructions();
            r.steps = (int) d.number("steps", 0);
            r.length = d.length();
            if (!meanflow) r.guidance = d.guidance(synth.default_guidance());
            r.tail = d.tail(synth.default_tail());
            r.noise = read_npy(d.file("noise.npy").u8string()).f32;
            if (d.has("speaker_noise.npy")) r.speaker_noise = read_npy(d.file("speaker_noise.npy").u8string()).f32;
            std::vector<float> audio;
            Stats stats;
            try {
                synth.synthesize(r, voice, [&](const float * s, size_t n) {
                    audio.insert(audio.end(), s, s + n);
                    return true;
                }, &stats);
            } catch (const std::exception & e) {
                // Q8_0 weights can move the predicted length by a frame, and the dump's noise then does not fit.
                std::printf("%s: %s\n", name.c_str(), e.what());
                ok = false;
                continue;
            }
            const Diff da = compare(audio, official.f32);
            std::printf("%s: %zu samples (official %zu), %d frames\n", name.c_str(), audio.size(), official.f32.size(),
                        stats.frames);
            print_diff("  audio against the official", da);
            std::printf("  first audio %.3f s: text and duration %.3f s, sampling %.3f s, codec %.3f s in all\n", stats.first_audio,
                        stats.text, stats.sampling, stats.codec);
            ok = ok && audio.size() == official.f32.size() && da.snr_db > least_snr_db(name);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
