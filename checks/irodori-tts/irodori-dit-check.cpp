// Checks the DiT and the sampler of Irodori-TTS against the official implementation on the dumps of
// reference/irodori-tts/dump.py made with the same model (MF or RF), each with the guidance and schedule of its
// request: the schedule, the timestep condition and every block of the first step, the speaker noise of a request
// that asks for it, then every step's velocity (after the guidance and the rescaling for RF) from the dump's own
// latent before it, then the whole sampler from the dump's noise. The DiT attends as it chooses on the device, or as the
// last argument forces it, so that both attentions are checked on a GPU.
//
// usage: irodori-dit-check <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "irodori-dumps.h"
#include "irodori-tts/layout.h"
#include "irodori-tts/sampler.h"
#include "npy.h"
#include "weight-types.h"

using namespace irodori;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name] [flash|products]\n", args[0].c_str());
        return 2;
    }
    try {
        // With F16 weights on the CPU of an Apple M5 (2026-10-13), RF's rf-hai-rescale samples a latent 15.6 dB from
        // the official's and 16.2 dB from the one of the default guidance, so the check cannot tell the request's
        // guidance from the weights' rounding; MF's dumps pass with F16.
        if (!takes_weights(args[1], model_layout, {"F32"},
                           "the sampler's steps carry the rounding of half-precision or quantized weights as far as a "
                           "request's guidance moves the latent")) {
            return 2;
        }
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        const ModelFile model(args[1], backend, model_layout);
        const Dit dit(model, backend, forced_attention(args, 4));
        std::printf("DiT: %s\n", attention_name(dit.attention()));
        Sampler sampler(dit, model, backend);
        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        const std::vector<IrodoriDump> dumps = irodori_dumps(args[2], model.str("general.source.repo_url"), "dit_velocity.npy");
        bool ok = !dumps.empty();
        for (const IrodoriDump & d : dumps) {
            auto npy = [&](const char * f) { return read_npy(d.file(f).u8string()); };
            const Npy text = npy("text_state.npy"), noise = npy("noise.npy");
            const Npy times = npy("dit_t.npy"), velocity = npy("dit_velocity.npy"), xs = npy("dit_x.npy");
            const Npy cond = npy("dit_cond.npy"), blocks = npy("dit_step0_blocks.npy");
            // A request without a reference attends to no speaker.
            const Npy speaker = d.reference() ? npy("speaker_state.npy") : Npy{};
            if (d.caption() && !model.boolean("irodori-tts.caption_condition")) {
                std::printf("%s: skipped, the file lacks the caption's encoder that instructions need\n", d.dir().filename().u8string().c_str());
                continue;
            }
            // A request without a caption attends to none.
            const Npy caption = d.caption() ? npy("caption_state.npy") : Npy{};
            Conditions c{text.f32, (int) text.shape[0], speaker.f32, d.reference() ? (int) speaker.shape[0] : 0, {},
                         caption.f32, d.caption() ? (int) caption.shape[0] : 0};
            const int frames = (int) noise.shape[0], steps = (int) times.shape[0];
            const size_t n = (size_t) frames * dit.latent_dim();
            std::printf("%s (%d frames, %d steps)\n", d.dir().filename().u8string().c_str(), frames, steps);
            const Guidance guidance =
                sampler.check(dit.meanflow() ? sampler.guidance() : d.guidance(sampler.guidance()), steps, d.reference(), d.caption());
            // The dump's times and the 0 the last step ends at.
            std::vector<float> dumped = times.f32;
            dumped.push_back(0.0f);

            const std::vector<float> schedule = sampler.schedule(steps, guidance.sway);
            double schedule_error = 0;
            for (int i = 0; i < steps; i++) schedule_error = std::max(schedule_error, (double) std::fabs(schedule[i] - times.f32[i]));
            std::printf("  schedule: largest difference %.1e\n", schedule_error);
            // torch's float32 cosine and the C library's differ by a rounding at most, which moves a time by its last bit.
            ok = ok && schedule_error < 1e-6;

            if (d.has("speaker_uncond.npy")) {
                const Npy uncond = npy("speaker_uncond.npy");
                const std::vector<float> made = speaker_noise(npy("speaker_noise.npy").f32, speaker.f32);
                const Diff dn = compare(made, uncond.f32);
                print_diff("  speaker noise", dn);
                // The standard deviation in double precision rounds as torch's; one rounding of the float32 product apart.
                ok = ok && dn.snr_db > 120;
                c.speaker_noise = uncond.f32;
            }

            std::vector<float> x0 = noise.f32;
            if (guidance.truncation) {
                for (float & v : x0) v = v * *guidance.truncation;
            }
            const float t0 = times.f32[0], t1 = schedule[1];
            Graph g;
            std::vector<ggml_tensor *> hidden;
            ggml_tensor * cond_out = nullptr;
            g.output(dit.build(g, x0, frames, t0, t0 - t1, c, sampler.branches(0, t0, guidance), sampler.speaker_scale(guidance, dumped, 0), &hidden,
                               &cond_out));
            g.output(cond_out);
            for (ggml_tensor * h : hidden) g.output(h);
            g.compute(backend, allocr);
            print_diff("  timestep condition, step 0", compare(Graph::read(cond_out).data(), cond.f32.data(), (size_t) cond.shape[1]));
            // The runtime's first call of a step is the branch with every condition alone for the joint and the
            // alternating guidance, and the whole batch otherwise, which is the first branches of the port's.
            Diff worst_block;
            const size_t block_size = blocks.f32.size() / hidden.size();
            for (size_t l = 0; l < hidden.size(); l++) {
                const Diff db = compare(Graph::read(hidden[l]).data(), &blocks.f32[l * block_size], block_size);
                if (db.snr_db < worst_block.snr_db) worst_block = db;
            }
            print_diff("  worst block of step 0", worst_block);

            Diff worst_step;
            int worst_index = 0;
            for (int i = 0; i < steps; i++) {
                const std::vector<float> x(i == 0 ? x0 : std::vector<float>(xs.f32.begin() + (i - 1) * n, xs.f32.begin() + i * n));
                const std::vector<float> v = sampler.velocity(c, x, frames, dumped, i, guidance);
                const Diff dv = compare(v.data(), &velocity.f32[i * n], n);
                if (dv.snr_db < worst_step.snr_db) {
                    worst_step = dv;
                    worst_index = i;
                }
            }
            print_diff("  worst step, step " + std::to_string(worst_index), worst_step);

            const auto start = std::chrono::steady_clock::now();
            const std::vector<float> final_x = sampler.sample(c, noise.f32, frames, steps, guidance);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const Diff df = compare(final_x.data(), &xs.f32[(steps - 1) * n], n);
            print_diff("  sampled latent", df);
            std::printf("  sampling took %.3f s\n", seconds);
            // Measured on an Apple M5 for the MF dumps: 86 to 122 dB on the CPU in F32 and 37 to 70 dB on Metal,
            // whose matrix kernel rounds its inputs to half precision, which four large steps amplify; 21 to
            // 41 dB with Q8_0 weights. RF's dumps lie 89 to 123 dB from the official on the CPU and 33 to 68 dB
            // on Metal (2026-10-13), the least with a caption or the strongest guidance. The speaker's keys
            // scaled in every layer make the attention sensitive to the keys' rounding: rf-hai-speaker-kv-min1's
            // worst step lies 31 dB from the official on Metal with the two products, whose kernel rounds the keys
            // to half precision, as far as on the CPU with the keys alone rounded, and 48 to 51 dB with flash attention,
            // which keeps them in float32. A wrong operation gives a few dB.
            ok = ok && df.snr_db > 15;
            if (!dit.meanflow() && d.sets_guidance()) {
                // The request's guidance must move the latent further from the default's than the port is from the
                // runtime, or a port that ignored it would pass.
                Conditions plain = c;
                plain.speaker_noise.clear();
                const std::vector<float> default_x =
                    sampler.sample(plain, noise.f32, frames, steps, sampler.check(sampler.guidance(), steps, d.reference(), d.caption()));
                const Diff dd = compare(default_x.data(), &xs.f32[(steps - 1) * n], n);
                // Printed as the distance it is, not as an SNR: here a low SNR is what the check wants, and checks/run.py
                // shows a check's least SNR as its result.
                std::printf("  the default guidance's latent lies %.1f dB from the official's, %.1f dB further than the request's\n",
                            dd.snr_db, df.snr_db - dd.snr_db);
                ok = ok && dd.snr_db + 6 < df.snr_db;
            }
        }
        ggml_gallocr_free(allocr);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
