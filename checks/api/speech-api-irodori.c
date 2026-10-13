/*
 * The checks of speech-api-check that Irodori-TTS alone takes: the length's rules and the longest speech, the steps and
 * the progress, the options of the official runtime's request, the voice none, instructions, and voice files of several
 * references, of another loudness and of an embedding.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"

/** Whether the file lists the voice none, which a layout 2 file has and a layout 1 file lacks. */
static int has_voice_none(const speech_model_info * info) {
    for (size_t i = 0; i < speech_model_info_voice_count(info); i++) {
        if (!strcmp(speech_model_info_voice_name(info, i), "none")) return 1;
    }
    return 0;
}

/** Irodori-TTS: the length's rules, the steps, the progress of the sampler, and a text past the longest. */
static int check_irodori_tts(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    Audio audio = {NULL, 0, 0};
    speech_request * r = new_request(model, "三つ目です。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 1);
    if (speak(r, &audio, NULL, NULL) != SPEECH_OK) return fail("a request of 1 s");
    printf("seconds 1 gave %.3f s of audio\n", (double) audio.n / rate);
    int ok = audio.n > 0 && audio.n <= (size_t) rate;

    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 2);
    speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 1.2);
    audio.n = 0;
    ok &= expect(speech_synthesize(r, collect, &audio), SPEECH_ERROR_INVALID_ARGUMENT, "seconds", "seconds with a duration scale") &&
          speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 1) == SPEECH_OK &&
          expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "the same request run again with a scale of 1") && audio.n > 0;
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 30);
    speech_request_set_float(r, SPEECH_OPT_SPEED, 0.5);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "seconds", "30 s at a speed of 0.5");
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_float(r, SPEECH_OPT_DURATION_SCALE, 50);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "duration_scale", "a predicted length scaled by 50");
    // A speed of 4 takes a prediction under 2 s below the shortest length of 0.5 s, and only the voice none fixes the
    // prediction: in it both v4.1 models predict 0.94 s for はい。, where a voice made of calm, slow speech
    // (voice-calm-adult-man.wav) predicts 3.28 s, in speech.cpp and in the official runtime alike (2026-10-11).
    if (has_voice_none(info)) {
        r = new_request(model, "はい。", "none", 5);
        speech_request_set_float(r, SPEECH_OPT_SPEED, 4);
        ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "speed", "a short predicted length at a speed of 4");
    } else {
        printf("a short predicted length at a speed of 4 is checked in the voice none, which this file does not have\n");
    }

    Progress progress;
    memset(&progress, 0, sizeof progress);
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_int(r, SPEECH_OPT_STEPS, 2);
    speech_request_set_progress(r, record_progress, &progress);
    audio.n = 0;
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "two steps") && progress.n == 3 && progress_rises(&progress, 1, "two steps");
    memset(&progress, 0, sizeof progress);
    progress.stop_at = 0.5;
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_progress(r, record_progress, &progress);
    audio.n = 0;
    speech_stop stop;
    ok &= expect(speak(r, &audio, &stop, NULL), SPEECH_CANCELLED, NULL, "a progress callback that stops at 0.5") && audio.n == 0 &&
          stop == SPEECH_STOP_CANCELLED;
    free(audio.samples);

    size_t tokens = 0;
    char text[8192] = "";
    for (int i = 0; i < 120; i++) strcat(text, "あいうえお、");
    if (speech_model_info_text_tokens(info, text, &tokens) != SPEECH_OK) return fail("speech_model_info_text_tokens");
    printf("a text of %zu tokens, the model takes %zu\n", tokens, speech_model_info_max_text_tokens(info));
    Audio none = {NULL, 0, 0};
    ok &= tokens > speech_model_info_max_text_tokens(info) &&
          expect(speak(new_request(model, text, voice, 6), &none, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "text", "a text past the longest");
    if (!ok) fprintf(stderr, "FAIL: Irodori-TTS's rules of a request are not followed\n");
    return ok ? 0 : 1;
}

/** Whether `a` and `b` hold the same samples, bit for bit. */
static int same_audio(const Audio * a, const Audio * b) {
    return a->n == b->n && memcmp(a->samples, b->samples, a->n * sizeof(float)) == 0;
}

/**
 * Irodori-TTS's options of the official runtime's request: the cut at the tail and, for an RF model, the guidance and
 * the schedule. A value its float32 cannot hold is refused when it is set, a request that sets each of them speaks, and
 * what only the whole request shows is refused naming the option.
 */
static int check_irodori_options(speech_model * model, const speech_model_info * info, const char * voice) {
    const int rate = speech_model_info_sample_rate(info);
    const int rf = speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_TEXT);
    Audio set = {NULL, 0, 0}, cut = {NULL, 0, 0}, kept = {NULL, 0, 0};
    int ok = 1;

    // The family computes with these as float32: a value that would turn into infinity, or into 0 where the option is
    // above 0, is out_of_range, where it once ran into audio of infinities.
    const struct {
        speech_option option;
        int above_zero;
    } narrowed[] = {
        {SPEECH_OPT_CFG_SCALE_TEXT, 0},      {SPEECH_OPT_CFG_SCALE_SPEAKER, 0}, {SPEECH_OPT_CFG_MIN_T, 0},
        {SPEECH_OPT_CFG_MAX_T, 0},           {SPEECH_OPT_TRUNCATION_FACTOR, 1}, {SPEECH_OPT_RESCALE_K, 1},
        {SPEECH_OPT_RESCALE_SIGMA, 1},       {SPEECH_OPT_SWAY_COEFF, 0},        {SPEECH_OPT_TAIL_STD_THRESHOLD, 1},
        {SPEECH_OPT_TAIL_MEAN_THRESHOLD, 1}, {SPEECH_OPT_SPEAKER_KV_SCALE, 1},  {SPEECH_OPT_SPEAKER_KV_MIN_T, 0},
        {SPEECH_OPT_CFG_SCALE_INSTRUCTIONS, 0},
    };
    speech_request * held = new_request(model, NULL, voice, -1);
    for (size_t i = 0; i < sizeof narrowed / sizeof narrowed[0]; i++) {
        const speech_option o = narrowed[i].option;
        if (!speech_model_info_takes(info, o)) continue;
        const char * name = speech_option_name(o);
        char what[128];
        snprintf(what, sizeof what, "%s at 1e300", name);
        ok &= expect(speech_request_set_float(held, o, 1e300), SPEECH_ERROR_OUT_OF_RANGE, name, what);
        snprintf(what, sizeof what, "%s at -1e300", name);
        ok &= expect(speech_request_set_float(held, o, -1e300), SPEECH_ERROR_OUT_OF_RANGE, name, what);
        if (narrowed[i].above_zero) {
            snprintf(what, sizeof what, "%s at 1e-300", name);
            ok &= expect(speech_request_set_float(held, o, 1e-300), SPEECH_ERROR_OUT_OF_RANGE, name, what);
        }
    }
    speech_request_free(held);

    // The latent of v4.1 never turns as flat as the default thresholds ask, even in the silence after "はい。" in 3 s,
    // which keep_tail therefore leaves as it is; thresholds of 0.6 and 0.2 cut it in that silence, and the cut audio
    // is the start of the whole.
    Audio whole = {NULL, 0, 0};
    speech_request * r = new_request(model, "はい。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 3);
    ok &= expect(speak(r, &whole, NULL, NULL), SPEECH_OK, NULL, "はい。 in 3 s");
    r = new_request(model, "はい。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 3);
    speech_request_set_bool(r, SPEECH_OPT_KEEP_TAIL, 1);
    ok &= expect(speak(r, &kept, NULL, NULL), SPEECH_OK, NULL, "はい。 in 3 s with keep_tail");
    r = new_request(model, "はい。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 3);
    speech_request_set_float(r, SPEECH_OPT_TAIL_STD_THRESHOLD, 0.6);
    speech_request_set_float(r, SPEECH_OPT_TAIL_MEAN_THRESHOLD, 0.2);
    ok &= expect(speak(r, &cut, NULL, NULL), SPEECH_OK, NULL, "はい。 in 3 s with thresholds of 0.6 and 0.2");
    printf("はい。 in 3 s: %.3f s, %.3f s with keep_tail, %.3f s cut by thresholds of 0.6 and 0.2\n", (double) whole.n / rate,
           (double) kept.n / rate, (double) cut.n / rate);
    ok &= whole.n == (size_t) 3 * rate && same_audio(&whole, &kept) && cut.n < whole.n &&
          memcmp(cut.samples, whole.samples, cut.n * sizeof(float)) == 0;
    free(whole.samples);
    r = new_request(model, "はい。", voice, 5);
    speech_request_set_bool(r, SPEECH_OPT_KEEP_TAIL, 1);
    speech_request_set_int(r, SPEECH_OPT_TAIL_WINDOW_SIZE, 8);
    ok &= expect(speak(r, &kept, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "tail_window_size", "keep_tail with a window");
    // A cut at the tail that leaves fewer frames than the decoder's first window of 12 (7 for v4.1-Small-MF and 10 for
    // v4.1-Small in this voice) is spoken whole, in one window.
    Audio shortest = {NULL, 0, 0};
    r = new_request(model, "はい。", voice, 5);
    speech_request_set_float(r, SPEECH_OPT_SECONDS, 0.5);
    speech_request_set_int(r, SPEECH_OPT_TAIL_WINDOW_SIZE, 1);
    speech_request_set_float(r, SPEECH_OPT_TAIL_STD_THRESHOLD, 1);
    speech_request_set_float(r, SPEECH_OPT_TAIL_MEAN_THRESHOLD, 0.1);
    ok &= expect(speak(r, &shortest, NULL, NULL), SPEECH_OK, NULL, "はい。 cut at the tail within the first window") && shortest.n > 0;
    printf("はい。 cut at the tail within the first window: %.3f s\n", (double) shortest.n / rate);
    free(shortest.samples);

    if (rf) {
        struct {
            const char * what;
            const char * mode;
            double text, speaker, min_t, max_t, rescale_k, sway;
            const char * uncond;
            speech_status status;
            const char * option;
        } refused[] = {
            {"the joint guidance with two scales", "joint", 3, 5, -1, -1, 0, 0, NULL, SPEECH_ERROR_INVALID_ARGUMENT, "cfg_guidance_mode"},
            {"rescale_k alone", NULL, -1, -1, -1, -1, 1.5, 0, NULL, SPEECH_ERROR_INVALID_ARGUMENT, "rescale_sigma"},
            {"cfg_min_t above cfg_max_t", NULL, -1, -1, 0.9, 0.5, 0, 0, NULL, SPEECH_ERROR_INVALID_ARGUMENT, "cfg_min_t"},
            {"a guidance mode without a scale", "alternating", 0, 0, -1, -1, 0, 0, NULL, SPEECH_ERROR_INVALID_ARGUMENT, "cfg_guidance_mode"},
            {"speaker noise without a speaker scale", NULL, -1, 0, -1, -1, 0, 0, "noise", SPEECH_ERROR_INVALID_ARGUMENT, "speaker_uncond_mode"},
            {"a sway that stops the schedule", NULL, -1, -1, -1, -1, 0, -3, NULL, SPEECH_ERROR_OUT_OF_RANGE, "sway_coeff"},
        };
        r = new_request(model, "はい。", voice, 5);
        speech_request_set_float(r, SPEECH_OPT_SPEAKER_KV_MIN_T, 0.5);
        ok &= expect(speak(r, &set, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "speaker_kv_min_t", "speaker_kv_min_t without a scale");
        r = new_request(model, "はい。", voice, 5);
        speech_request_set_int(r, SPEECH_OPT_SPEAKER_KV_MAX_LAYERS, 3);
        ok &= expect(speak(r, &set, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "speaker_kv_max_layers", "speaker_kv_max_layers without a scale");
        for (size_t i = 0; i < sizeof refused / sizeof refused[0]; i++) {
            r = new_request(model, "はい。", voice, 5);
            speech_request_set_int(r, SPEECH_OPT_STEPS, 8);
            if (refused[i].mode) speech_request_set_string(r, SPEECH_OPT_CFG_GUIDANCE_MODE, refused[i].mode);
            if (refused[i].text >= 0) speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_TEXT, refused[i].text);
            if (refused[i].speaker >= 0) speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_SPEAKER, refused[i].speaker);
            if (refused[i].min_t >= 0) speech_request_set_float(r, SPEECH_OPT_CFG_MIN_T, refused[i].min_t);
            if (refused[i].max_t >= 0) speech_request_set_float(r, SPEECH_OPT_CFG_MAX_T, refused[i].max_t);
            if (refused[i].rescale_k > 0) speech_request_set_float(r, SPEECH_OPT_RESCALE_K, refused[i].rescale_k);
            if (refused[i].sway != 0) speech_request_set_float(r, SPEECH_OPT_SWAY_COEFF, refused[i].sway);
            if (refused[i].uncond) speech_request_set_string(r, SPEECH_OPT_SPEAKER_UNCOND_MODE, refused[i].uncond);
            Audio none = {NULL, 0, 0};
            ok &= expect(speak(r, &none, NULL, NULL), refused[i].status, refused[i].option, refused[i].what) && none.n == 0;
        }
        r = new_request(model, SENTENCE, voice, 5);
        speech_request_set_int(r, SPEECH_OPT_STEPS, 8);
        speech_request_set_string(r, SPEECH_OPT_CFG_GUIDANCE_MODE, "alternating");
        speech_request_set_string(r, SPEECH_OPT_SPEAKER_UNCOND_MODE, "noise");
        speech_request_set_float(r, SPEECH_OPT_TRUNCATION_FACTOR, 0.9);
        speech_request_set_float(r, SPEECH_OPT_RESCALE_K, 1.5);
        speech_request_set_float(r, SPEECH_OPT_RESCALE_SIGMA, 1.0);
        speech_request_set_float(r, SPEECH_OPT_SWAY_COEFF, -0.5);
        speech_request_set_float(r, SPEECH_OPT_SPEAKER_KV_SCALE, 1.5);
        speech_request_set_float(r, SPEECH_OPT_SPEAKER_KV_MIN_T, 0.6);
        speech_request_set_int(r, SPEECH_OPT_SPEAKER_KV_MAX_LAYERS, 6);
        set.n = 0;
        ok &= expect(speak(r, &set, NULL, NULL), SPEECH_OK, NULL, "a request of every guidance option") && set.n > 0;
        if (set.n > 0) printf("a request of every guidance option spoke %.2f s\n", (double) set.n / rate);

        // The largest text scale the range takes, which a float32 holds but whose guidance overflows it, gives speech that
        // is not finite on every device (1e38 does on Metal alone, where the CPU's latent stays finite and the codec's
        // tanh saturates); it is refused naming no option, with nothing passed on.
        double lowest = 0, largest = 0;
        int exclusive = 0;
        speech_model_info_option_range(info, SPEECH_OPT_CFG_SCALE_TEXT, &lowest, &largest, &exclusive);
        r = new_request(model, "はい。", voice, 5);
        speech_request_set_int(r, SPEECH_OPT_STEPS, 8);
        speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_TEXT, largest);
        Audio overflow = {NULL, 0, 0};
        char what[96];
        snprintf(what, sizeof what, "a text scale of %g", largest);
        ok &= expect(speak(r, &overflow, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, NULL, what) && overflow.n == 0;
        free(overflow.samples);
    }
    free(set.samples);
    free(cut.samples);
    free(kept.samples);
    if (!ok) fprintf(stderr, "FAIL: Irodori-TTS's options of the runtime's request are not followed\n");
    return ok ? 0 : 1;
}

/**
 * Irodori-TTS without a reference: a file with the null speaker lists the voice none, which speaks and refuses what
 * has no speaker to act on; a file without it refuses none naming the null speaker it lacks.
 */
static int check_irodori_without_reference(speech_model * model, const speech_model_info * info) {
    int ok = 1;
    if (!has_voice_none(info)) {
        speech_request * r = new_request(model, "はい。", NULL, 3);
        ok &= expect(speech_request_set_string(r, SPEECH_OPT_VOICE, "none"), SPEECH_ERROR_OUT_OF_RANGE, "voice", "the voice none without the null speaker") &&
              strstr(speech_last_error(), "null speaker") != NULL;
        speech_request_free(r);
        if (ok) printf("a file without the null speaker refuses the voice none: %s\n", speech_last_error());
        else fprintf(stderr, "FAIL: a file without the null speaker does not refuse the voice none by what it lacks\n");
        return ok ? 0 : 1;
    }
    const int rate = speech_model_info_sample_rate(info);
    Audio audio = {NULL, 0, 0};
    ok &= expect(speak(new_request(model, SENTENCE, "none", 3), &audio, NULL, NULL), SPEECH_OK, NULL, "the voice none") && audio.n > 0;
    if (ok) printf("the voice none spoke %.2f s without a reference\n", (double) audio.n / rate);
    if (speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_SPEAKER)) {
        speech_request * r = new_request(model, "はい。", "none", 3);
        speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_SPEAKER, 3);
        ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "cfg_scale_speaker", "a speaker scale without a reference");
        r = new_request(model, "はい。", "none", 3);
        speech_request_set_string(r, SPEECH_OPT_SPEAKER_UNCOND_MODE, "noise");
        ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "speaker_uncond_mode", "speaker noise without a reference");
        r = new_request(model, "はい。", "none", 3);
        speech_request_set_float(r, SPEECH_OPT_SPEAKER_KV_SCALE, 1.5);
        ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "speaker_kv_scale", "the speaker's scaling without a reference");
        r = new_request(model, "はい。", "none", 3);
        speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_SPEAKER, 0);
        audio.n = 0;
        ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "a speaker scale of 0 without a reference") && audio.n > 0;
    }
    free(audio.samples);
    if (!ok) fprintf(stderr, "FAIL: the voice none does not speak, or does not refuse what it leaves without effect\n");
    return ok ? 0 : 1;
}

/**
 * Irodori-TTS's instructions, the runtime's caption: a file without the caption's encoder takes "" and refuses any
 * other naming what it lacks; one with it speaks a caption, gives the audio of none for one that strips to nothing,
 * refuses one past its longest and RF's caption scale without one, and stops a caption's request at its progress and
 * from another thread.
 */
static int check_irodori_instructions(speech_model * model, const speech_model_info * info, const char * voice) {
    const char * caption = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。";
    int ok = 1;
    speech_request * r = new_request(model, "はい。", voice, 3);
    if (!speech_model_info_takes(info, SPEECH_OPT_INSTRUCTIONS)) {
        ok &= expect(speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, ""), SPEECH_OK, NULL, "empty instructions") &&
              expect(speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, caption), SPEECH_ERROR_UNSUPPORTED, "instructions",
                     "instructions without the caption's encoder") &&
              strstr(speech_last_error(), "caption") != NULL;
        speech_request_free(r);
        if (ok) printf("a file without the caption's encoder refuses instructions: %s\n", speech_last_error());
        else fprintf(stderr, "FAIL: a file without the caption's encoder does not refuse instructions by what it lacks\n");
        return ok ? 0 : 1;
    }
    speech_request_free(r);
    const int rate = speech_model_info_sample_rate(info);
    Audio plain = {NULL, 0, 0}, blank = {NULL, 0, 0}, described = {NULL, 0, 0};
    ok &= expect(speak(new_request(model, SENTENCE, voice, 4), &plain, NULL, NULL), SPEECH_OK, NULL, "a request without instructions");
    r = new_request(model, SENTENCE, voice, 4);
    speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, "\xe3\x80\x80 \n");
    ok &= expect(speak(r, &blank, NULL, NULL), SPEECH_OK, NULL, "instructions of spaces alone") && same_audio(&plain, &blank);
    r = new_request(model, SENTENCE, voice, 4);
    speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, caption);
    ok &= expect(speak(r, &described, NULL, NULL), SPEECH_OK, NULL, "instructions") && described.n > 0 && !same_audio(&plain, &described);
    if (ok) {
        printf("instructions of spaces alone give the %zu samples of none, and a caption %.2f s of other audio\n", plain.n,
               (double) described.n / rate);
    }
    // 600 copies of a two-token word, past the 512 tokens of v4.1's captions.
    char * long_caption = (char *) malloc(600 * 6 + 1);
    long_caption[0] = '\0';
    for (int i = 0; i < 600; i++) strcat(long_caption, "声、");
    r = new_request(model, "はい。", voice, 4);
    speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, long_caption);
    ok &= expect(speak(r, &blank, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "instructions", "instructions past the longest");
    free(long_caption);
    if (speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_INSTRUCTIONS)) {
        r = new_request(model, "はい。", voice, 4);
        speech_request_set_float(r, SPEECH_OPT_CFG_SCALE_INSTRUCTIONS, 5);
        ok &= expect(speak(r, &blank, NULL, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "cfg_scale_instructions", "a caption scale without instructions");
    }
    Progress progress;
    memset(&progress, 0, sizeof progress);
    progress.stop_at = 0.5;
    r = new_request(model, SENTENCE, voice, 5);
    speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, caption);
    speech_request_set_progress(r, record_progress, &progress);
    Audio none = {NULL, 0, 0};
    speech_stop stop;
    ok &= expect(speak(r, &none, &stop, NULL), SPEECH_CANCELLED, NULL, "a caption's request stopped by its progress at 0.5") && none.n == 0 &&
          stop == SPEECH_STOP_CANCELLED;
    r = new_request(model, SENTENCE, voice, 6);
    speech_request_set_string(r, SPEECH_OPT_INSTRUCTIONS, caption);
    ok &= cancelled_from_another_thread(r, "a caption's request cancelled from another thread");
    free(plain.samples);
    free(blank.samples);
    free(described.samples);
    if (!ok) fprintf(stderr, "FAIL: Irodori-TTS's instructions are not followed as the runtime follows its caption\n");
    return ok ? 0 : 1;
}

/** The bytes of the file at `path`, which the caller frees, and their number in `n`; NULL when it cannot be read. */
static unsigned char * read_file(const char * path, size_t * n) {
    FILE * f = open_utf8(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *n = (size_t) ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char * bytes = (unsigned char *) malloc(*n ? *n : 1);
    if (fread(bytes, 1, *n, f) != *n) {
        free(bytes);
        bytes = NULL;
    }
    fclose(f);
    return bytes;
}

/** Whether the files at `a` and `b` hold the same bytes. */
static int same_file(const char * a, const char * b) {
    size_t na = 0, nb = 0;
    unsigned char * x = read_file(a, &na), * y = read_file(b, &nb);
    const int same = x && y && na == nb && memcmp(x, y, na) == 0;
    free(x);
    free(y);
    return same;
}

/**
 * Voice files of several references and of another loudness: one reference at the model's loudness gives the bytes of
 * speech_voice_make(), several and another or the kept loudness give other files, and what cannot be made is refused
 * naming its input.
 */
int check_voice_params(const char * model_path, const char * reference, const char * made, const speech_load_params * params) {
    char path[4096];
    int ok = 1;
    speech_voice_params * v = NULL;
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    snprintf(path, sizeof path, "%s.from.gguf", made);
    ok &= expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "references", "a voice without a reference");
    ok &= expect(speech_voice_params_set_loudness(v, NAN), SPEECH_ERROR_INVALID_ARGUMENT, "loudness", "a loudness of NaN");
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a reference") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_OK, NULL, "a voice of one reference") && same_file(path, made);
    if (ok) printf("a voice of one reference at the model's loudness is the file speech_voice_make() writes\n");
    char two[4096], quiet[4096], kept[4096];
    snprintf(two, sizeof two, "%s.two.gguf", made);
    snprintf(quiet, sizeof quiet, "%s.quiet.gguf", made);
    snprintf(kept, sizeof kept, "%s.kept.gguf", made);
    ok &= expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a second reference") &&
          expect(speech_voice_make_from(model_path, v, two, params), SPEECH_OK, NULL, "a voice of two references") && !same_file(two, made);
    speech_voice_params_free(v);
    v = NULL;
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    speech_voice_params_add_reference(v, reference);
    ok &= expect(speech_voice_params_set_loudness(v, -23), SPEECH_OK, NULL, "a loudness of -23 LUFS") &&
          expect(speech_voice_make_from(model_path, v, quiet, params), SPEECH_OK, NULL, "a voice at -23 LUFS") && !same_file(quiet, made);
    ok &= expect(speech_voice_params_keep_loudness(v), SPEECH_OK, NULL, "the loudness kept") &&
          expect(speech_voice_make_from(model_path, v, kept, params), SPEECH_OK, NULL, "a voice of the loudness kept") && !same_file(kept, made) &&
          !same_file(kept, quiet);
    // Thirteen copies of a reference of about 10 s are past the 120 s a voice takes.
    for (int i = 0; i < 12; i++) speech_voice_params_add_reference(v, reference);
    ok &= expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_OUT_OF_RANGE, "references", "references past 120 s together");
    speech_voice_params_free(v);
    remove(path);
    // An embedding of eight tokens of a model's 768 values, made up, and one of another width.
    float values[8 * 768];
    for (int i = 0; i < 8 * 768; i++) values[i] = 0.5f * sinf(0.01f * (float) i);
    snprintf(path, sizeof path, "%s.embedding.gguf", made);
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    ok &= expect(speech_voice_params_set_embedding(v, values, 8, 512), SPEECH_OK, NULL, "an embedding of 512 values a token") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "embedding", "an embedding of another width");
    values[5] = NAN;
    ok &= expect(speech_voice_params_set_embedding(v, values, 8, 768), SPEECH_ERROR_INVALID_ARGUMENT, "embedding", "an embedding with NaN");
    values[5] = 0;
    ok &= expect(speech_voice_params_set_embedding(v, values, 8, 768), SPEECH_OK, NULL, "an embedding") &&
          expect(speech_voice_params_add_reference(v, reference), SPEECH_OK, NULL, "a reference beside an embedding") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_ERROR_INVALID_ARGUMENT, "embedding", "an embedding with a reference");
    speech_voice_params_free(v);
    if (speech_voice_params_new(&v) != SPEECH_OK) return fail("speech_voice_params_new");
    ok &= expect(speech_voice_params_set_embedding(v, values, 8, 768), SPEECH_OK, NULL, "an embedding") &&
          expect(speech_voice_make_from(model_path, v, path, params), SPEECH_OK, NULL, "a voice of an embedding");
    speech_voice_params_free(v);
    if (ok) printf("voices of two references, at -23 LUFS, of the loudness kept and of an embedding were made: %s, %s, %s, %s\n", two, quiet, kept,
                   path);
    else fprintf(stderr, "FAIL: a voice of several references or another loudness is not made as speech_voice_make() makes one\n");
    return ok ? 0 : 1;
}

/**
 * Irodori-TTS's longest speech, in the voice none, whose lengths of these texts are known: a text whose predicted
 * length passes the longest is refused naming the text before any progress or audio, where the runtime squeezes it
 * into the longest; a speed that brings it within is followed; and a shorter text is spoken at its predicted length.
 */
static int check_irodori_longest(speech_model * model, const speech_model_info * info) {
    if (!has_voice_none(info)) {
        printf("the longest speech is checked in the voice none, which this file does not have\n");
        return 0;
    }
    // In the voice none both v4.1 models predict 28.8 s for the first six sentences and 30.7 s for all eight
    // (2026-10-08). Bounded to 30 s, the eight lost two whole sentences; spoken one at a time, they take 38.3 s.
    static const char * const sentences[] = {
        "今日は朝から雨が降っていました。",
        "駅までの道は水たまりだらけで、靴がすっかり濡れてしまいました。",
        "電車はいつもより混んでいて、窓の外はずっと灰色でした。",
        "会社に着くと、同僚が温かいお茶を入れてくれました。",
        "午後には雨が上がり、雲の間から青い空が見えてきました。",
        "帰り道、公園の木々が夕日に照らされて輝いていました。",
        "小さな子どもたちが水たまりで楽しそうに遊んでいました。",
        "家に帰ってから、久しぶりにゆっくりと本を読みました。",
    };
    char six[1024] = "", eight[1024] = "";
    for (int i = 0; i < 8; i++) {
        if (i < 6) strcat(six, sentences[i]);
        strcat(eight, sentences[i]);
    }
    const int rate = speech_model_info_sample_rate(info);
    const size_t longest = (size_t) 30 * rate;
    Audio audio = {NULL, 0, 0};
    Progress progress;
    memset(&progress, 0, sizeof progress);
    speech_request * r = new_request(model, eight, "none", 1);
    speech_request_set_progress(r, record_progress, &progress);
    int ok = expect(speak(r, &audio, NULL, NULL), SPEECH_ERROR_OUT_OF_RANGE, "text", "eight sentences predicted past 30 s") && audio.n == 0 &&
             progress.n == 0;
    audio.n = 0;
    r = new_request(model, eight, "none", 1);
    speech_request_set_float(r, SPEECH_OPT_SPEED, 1.5);
    ok &= expect(speak(r, &audio, NULL, NULL), SPEECH_OK, NULL, "the same at a speed of 1.5") && audio.n > 0 && audio.n <= longest;
    printf("eight sentences at a speed of 1.5: %.2f s\n", (double) audio.n / rate);
    audio.n = 0;
    speech_stop stop = SPEECH_STOP_CANCELLED;
    ok &= expect(speak(new_request(model, six, "none", 1), &audio, &stop, NULL), SPEECH_OK, NULL, "six sentences predicted within 30 s") &&
          stop == SPEECH_STOP_COMPLETE && audio.n > 25 * (size_t) rate && audio.n <= longest;
    printf("six sentences: %.2f s\n", (double) audio.n / rate);
    free(audio.samples);
    if (!ok) fprintf(stderr, "FAIL: a text past Irodori-TTS's longest speech is not refused, or one within it is not spoken whole\n");
    return ok ? 0 : 1;
}

int check_irodori(speech_model * model, const speech_model_info * info, const char * voice) {
    return check_irodori_tts(model, info, voice) || check_irodori_options(model, info, voice) || check_irodori_without_reference(model, info) ||
           check_irodori_longest(model, info) || check_irodori_instructions(model, info, voice);
}
