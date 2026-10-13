/*
 * What the parts of speech-api-check share: messages, the checks of the library before a model is loaded, the
 * quantizing it refuses, the information read without loading against a loaded model's, the refusal of every option's values, the reading
 * of a dump's files, and the requests of a synthesis, their audio and their cancellation from another thread.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "speech-api-check.h"

int fail(const char * what) {
    const char * option = speech_last_error_option();
    fprintf(stderr, "FAIL: %s (%s): %s\n", what, option ? option : "no option", speech_last_error());
    return 1;
}

int expect(speech_status got, speech_status want, const char * option, const char * what) {
    const char * named = got == SPEECH_OK || got == SPEECH_CANCELLED ? NULL : speech_last_error_option();
    const int same_option = option ? named && !strcmp(option, named) : named == NULL;
    const char * got_name = speech_status_name(got);
    if (got != want || (got < 0 && !same_option)) {
        fprintf(stderr, "FAIL: %s: %s (%s) where %s (%s) was expected: %s\n", what, got_name ? got_name : "?", named ? named : "no option",
                speech_status_name(want), option ? option : "no option", got < 0 ? speech_last_error() : "");
        return 0;
    }
    if (got < 0) printf("%s: %s (%s): %s\n", what, got_name, named ? named : "no option", speech_last_error());
    return 1;
}

/** The versions, the names of the statuses, the stop reasons and the options, and the devices. */
int check_library(void) {
    if (speech_api_version_major() != SPEECH_API_VERSION_MAJOR || speech_api_version_minor() != SPEECH_API_VERSION_MINOR) {
        fprintf(stderr, "FAIL: the library has API %d.%d, the header %d.%d\n", speech_api_version_major(), speech_api_version_minor(),
                SPEECH_API_VERSION_MAJOR, SPEECH_API_VERSION_MINOR);
        return 1;
    }
    if (strcmp(speech_version(), SPEECH_EXPECTED_VERSION) != 0) {
        fprintf(stderr, "FAIL: the library is speech.cpp %s, the check was built for %s\n", speech_version(), SPEECH_EXPECTED_VERSION);
        return 1;
    }
    printf("speech.cpp %s, C API %d.%d\n", speech_version(), speech_api_version_major(), speech_api_version_minor());
    static const char * statuses[] = {"internal", "io", "out_of_memory", "device", "model_file", "out_of_range", "unsupported",
                                      "invalid_argument", "ok", "cancelled"};
    for (int s = SPEECH_ERROR_INTERNAL; s <= SPEECH_CANCELLED; s++) {
        const char * name = speech_status_name((speech_status) s);
        if (!name || strcmp(name, statuses[s - SPEECH_ERROR_INTERNAL]) != 0) {
            fprintf(stderr, "FAIL: the status %d is named %s\n", s, name ? name : "NULL");
            return 1;
        }
    }
    static const char * stops[] = {"complete", "max_seconds", "model_limit", "cancelled"};
    for (int s = SPEECH_STOP_COMPLETE; s <= SPEECH_STOP_CANCELLED; s++) {
        if (!speech_stop_name((speech_stop) s) || strcmp(speech_stop_name((speech_stop) s), stops[s]) != 0) {
            fprintf(stderr, "FAIL: the stop reason %d is not named %s\n", s, stops[s]);
            return 1;
        }
    }
    if (speech_status_name((speech_status) 2) || speech_status_name((speech_status) -9) || speech_stop_name((speech_stop) 4)) {
        fprintf(stderr, "FAIL: a status or stop reason the library does not know has a name\n");
        return 1;
    }
    static const char * options[] = {"voice", "language", "seed", "speed", "seconds", "duration_scale", "steps", "max_seconds",
                                     "timestamps", "prompt", "decoding", "do_sample", "top_k", "top_p", "temperature",
                                     "repetition_penalty", "code_predictor_do_sample", "code_predictor_top_k",
                                     "code_predictor_top_p", "code_predictor_temperature", "instructions", "cfg_scale_text",
                                     "cfg_scale_speaker", "cfg_guidance_mode", "cfg_min_t", "cfg_max_t", "truncation_factor",
                                     "rescale_k", "rescale_sigma", "speaker_uncond_mode", "sway_coeff", "keep_tail",
                                     "tail_window_size", "tail_std_threshold", "tail_mean_threshold", "speaker_kv_scale",
                                     "speaker_kv_min_t", "speaker_kv_max_layers", "cfg_scale_instructions", "threshold",
                                     "min_speech_duration_ms", "min_silence_duration_ms", "speech_pad_ms", "max_speech_duration_s",
                                      "irodori_compatible", "decode_on_cpu"};
    if (speech_option_count() != sizeof options / sizeof options[0]) {
        fprintf(stderr, "FAIL: the library knows %zu options\n", speech_option_count());
        return 1;
    }
    for (size_t i = 0; i < speech_option_count(); i++) {
        speech_option o = SPEECH_OPT_TIMESTAMPS;
        if (strcmp(speech_option_name((speech_option) i), options[i]) != 0 || speech_option_from_name(options[i], &o) != SPEECH_OK ||
            o != (speech_option) i) {
            fprintf(stderr, "FAIL: the option %zu is not named %s both ways\n", i, options[i]);
            return 1;
        }
    }
    speech_option unknown;
    if (!expect(speech_option_from_name("durationScale", &unknown), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the option durationScale")) return 1;
    if (speech_option_name((speech_option) speech_option_count()) != NULL) {
        fprintf(stderr, "FAIL: an option past the last has a name\n");
        return 1;
    }

    const size_t n = speech_device_count();
    for (size_t i = 0; i < n; i++) {
        speech_device_kind kind;
        uint64_t total = 0, free_bytes = 0;
        if (speech_device_get_kind(i, &kind) != SPEECH_OK || speech_device_memory(i, &total, &free_bytes) != SPEECH_OK) {
            return fail("a device's kind or memory");
        }
        printf("device %s (%s), %s, %llu of %llu bytes free\n", speech_device_name(i), speech_device_description(i),
               kind == SPEECH_DEVICE_CPU ? "cpu" : kind == SPEECH_DEVICE_GPU ? "gpu" : "igpu", (unsigned long long) free_bytes,
               (unsigned long long) total);
    }
    speech_device_kind kind;
    if (n == 0 || speech_device_name(n) != NULL || speech_device_description(n) != NULL ||
        !expect(speech_device_get_kind(n, &kind), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the kind of a device past the last")) {
        fprintf(stderr, "FAIL: the devices are not listed as speech.h declares\n");
        return 1;
    }
    return 0;
}

/** Load parameters and loads that are refused, before any model is loaded. */
int check_load_refusals(const char * model_path) {
    speech_load_params * params = NULL;
    speech_model * model = NULL;
    speech_model_info * info = NULL;
    if (speech_load_params_new(&params) != SPEECH_OK) return fail("speech_load_params_new");
    int ok = expect(speech_load_params_set_threads(params, 0), SPEECH_ERROR_INVALID_ARGUMENT, "threads", "0 threads");
    ok &= expect(speech_load_params_set_device(params, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "device", "a NULL device");
    ok &= expect(speech_load_params_set_device(params, ""), SPEECH_ERROR_INVALID_ARGUMENT, "device", "an empty device");
    ok &= expect(speech_load_params_set_device(params, "no-such-device"), SPEECH_OK, NULL, "a device's name, before it is looked for");
    ok &= expect(speech_model_load(model_path, params, &model), SPEECH_ERROR_DEVICE, "device", "a load on a device that is not there");
    ok &= model == NULL;
    ok &= expect(speech_model_load(NULL, NULL, &model), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "a load without a path");
    ok &= expect(speech_model_info_open("no-such-folder/no-such-model.gguf", &info), SPEECH_ERROR_IO, NULL, "a model file that is not there");
    speech_load_params_free(params);
    return ok ? 0 : 1;
}

int check_quantize_refusals(const char * model_path) {
    speech_model_info * info = NULL;
    if (speech_model_info_open(model_path, &info) != SPEECH_OK) return fail("speech_model_info_open");
    const int f32 = !strcmp(speech_model_info_weight_type(info), "F32");
    // Silero VAD's layout keeps every tensor in F32, of which no file of another type is made.
    const int detection = speech_model_info_task(info) == SPEECH_TASK_DETECTION;
    speech_model_info_free(info);
    // The folder does not exist, so that no refusal can leave a file behind and a file of F32 weights, which the library
    // quantizes, is refused only when it comes to create its output.
    const char * out = "no-such-folder/quantized.gguf";
    int ok = expect(speech_quantize(NULL, "q8_0", out), SPEECH_ERROR_INVALID_ARGUMENT, "model_path", "quantizing without a model");
    ok &= expect(speech_quantize(model_path, NULL, out), SPEECH_ERROR_INVALID_ARGUMENT, "type", "quantizing without a type");
    ok &= expect(speech_quantize(model_path, "q8_0", NULL), SPEECH_ERROR_INVALID_ARGUMENT, "out_path", "quantizing without a path to write");
    ok &= expect(speech_quantize(model_path, "q3_k", out), SPEECH_ERROR_INVALID_ARGUMENT, "type", "quantizing to Q3_K, a type it does not write");
    ok &= expect(speech_quantize(model_path, "F32", out), SPEECH_ERROR_INVALID_ARGUMENT, "type", "quantizing to F32, the type it writes from");
    ok &= expect(speech_quantize("no-such-folder/no-such-model.gguf", "q8_0", out), SPEECH_ERROR_IO, "model_path",
                 "quantizing a model file that is not there");
    ok &= expect(speech_quantize(model_path, "Q4_K", model_path), SPEECH_ERROR_INVALID_ARGUMENT, "out_path", "quantizing a model file over itself");
    if (detection) {
        ok &= expect(speech_quantize(model_path, "q8_0", out), SPEECH_ERROR_INVALID_ARGUMENT, "type", "quantizing a model whose layout keeps F32");
    } else if (f32) {
        ok &= expect(speech_quantize(model_path, "q4_k", out), SPEECH_ERROR_IO, "out_path", "quantizing F32 weights into a folder that is not there");
    } else {
        ok &= expect(speech_quantize(model_path, "q4_k", out), SPEECH_ERROR_INVALID_ARGUMENT, "model_path",
                     "quantizing weights that are not F32");
    }
    return ok ? 0 : 1;
}

/** The JSON value of the metadata entry `key`, or NULL when the file has none. */
static const char * meta_value_of(const speech_model_info * info, const char * key) {
    for (size_t i = 0; i < speech_model_info_meta_count(info); i++) {
        if (!strcmp(speech_model_info_meta_key(info, i), key)) return speech_model_info_meta_value(info, i);
    }
    return NULL;
}

/**
 * Whether `value` is the string of the metadata entry `key`, or NULL where the file has no such entry. The values
 * compared hold no character that JSON escapes.
 */
static int is_meta_string(const speech_model_info * info, const char * key, const char * value) {
    const char * meta = meta_value_of(info, key);
    char quoted[1024];
    if (!value || !meta) return !value && !meta;
    snprintf(quoted, sizeof quoted, "\"%s\"", value);
    return !strcmp(quoted, meta);
}

/** The identity's accessors against the general keys they are read from. Returns 1 when each gives its key. */
static int check_identity(const speech_model_info * info) {
    const char * repository = NULL, * revision = NULL;
    int ok = expect(speech_model_info_source(info, NULL, &revision), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "the source without a repository");
    ok &= expect(speech_model_info_source(info, &repository, &revision), SPEECH_OK, NULL, "the source");
    if (!ok) return 0;
    char url[1024];
    snprintf(url, sizeof url, "%s/tree/%s", repository, revision);
    static const char * const file_types[][2] = {{"0", "F32"}, {"1", "F16"}, {"7", "Q8_0"}, {"18", "Q6_K"}, {"16", "Q5_K"}, {"14", "Q4_K"}};
    const char * file_type = meta_value_of(info, "general.file_type"), * weight_type = speech_model_info_weight_type(info);
    int typed = 0;
    for (size_t i = 0; i < sizeof file_types / sizeof file_types[0]; i++) {
        typed |= file_type && !strcmp(file_type, file_types[i][0]) && !strcmp(weight_type, file_types[i][1]);
    }
    const char * finetune = speech_model_info_finetune(info), * version = speech_model_info_version(info);
    ok = is_meta_string(info, "general.name", speech_model_info_name(info)) &&
         is_meta_string(info, "general.organization", speech_model_info_organization(info)) &&
         is_meta_string(info, "general.basename", speech_model_info_basename(info)) &&
         is_meta_string(info, "general.size_label", speech_model_info_size_label(info)) && is_meta_string(info, "general.finetune", finetune) &&
         is_meta_string(info, "general.version", version) && is_meta_string(info, "general.license", speech_model_info_license(info)) &&
         is_meta_string(info, "general.source.repo_url", repository) && is_meta_string(info, "general.source.url", url) && typed;
    if (!ok) {
        fprintf(stderr, "FAIL: the identity's accessors do not give the general keys of the file\n");
        return 0;
    }
    printf("identity: %s of %s, %s %s%s%s%s%s, %s, %s at %s, %s weights, as the general keys give them\n", speech_model_info_name(info),
           speech_model_info_organization(info), speech_model_info_basename(info), speech_model_info_size_label(info), finetune ? " " : "",
           finetune ? finetune : "", version ? " " : "", version ? version : "", speech_model_info_license(info), repository, revision, weight_type);
    return 1;
}

int check_info_matches(const char * path, const speech_model * model) {
    speech_model_info * file = NULL, * loaded = NULL;
    const double start = now_seconds();
    if (speech_model_info_open(path, &file) != SPEECH_OK) return fail("speech_model_info_open");
    printf("information read without loading in %.3f s:\n%s\n", now_seconds() - start, speech_model_info_json(file));
    if (speech_model_get_info(model, &loaded) != SPEECH_OK) return fail("speech_model_get_info");
    const char * file_json = speech_model_info_json(file), * loaded_json = speech_model_info_json(loaded);
    const char * device = speech_model_info_device(loaded);
    char expected[1 << 16];
    const size_t n = strlen(file_json);
    int ok = device != NULL && speech_model_info_device(file) == NULL && speech_model_info_threads(file) == 0 && n > 2 &&
             n + 256 < sizeof expected;
    if (ok) {
        snprintf(expected, sizeof expected, "%.*s,\"device\":\"%s\",\"threads\":%d}", (int) (n - 1), file_json, device,
                 speech_model_info_threads(loaded));
        ok = strcmp(expected, loaded_json) == 0;
    }
    if (!is_meta_string(file, "general.architecture", speech_model_info_architecture(file)) ||
        speech_model_info_meta_key(file, speech_model_info_meta_count(file)) != NULL) {
        fprintf(stderr, "FAIL: the metadata does not give general.architecture as %s\n", speech_model_info_architecture(file));
        ok = 0;
    }
    if (!ok) {
        fprintf(stderr, "FAIL: the loaded model's information is not the file's with its device and threads:\n%s\n", loaded_json);
    } else {
        printf("the loaded model's information is the file's with \"device\":\"%s\" and \"threads\":%d\n", device,
               speech_model_info_threads(loaded));
    }
    const int identified = check_identity(file);
    speech_model_info_free(file);
    speech_model_info_free(loaded);
    return ok && identified ? 0 : 1;
}

/** The neutral value of an option, set through `r`, or SPEECH_OK when the option has none. */
static speech_status set_neutral(speech_request * r, speech_option option, int * has) {
    *has = 1;
    switch (option) {
        case SPEECH_OPT_SPEED:
        case SPEECH_OPT_DURATION_SCALE: return speech_request_set_float(r, option, 1.0);
        case SPEECH_OPT_LANGUAGE: return speech_request_set_string(r, option, "auto");
        case SPEECH_OPT_TIMESTAMPS: return speech_request_set_bool(r, option, 0);
        case SPEECH_OPT_PROMPT:
        case SPEECH_OPT_INSTRUCTIONS: return speech_request_set_string(r, option, "");
        default: *has = 0; return SPEECH_OK;
    }
}

/** The default and the choices of `option` as the information gives them, against what it says it takes. */
static int check_declaration(const speech_model_info * info, speech_option option) {
    const char * name = speech_option_name(option);
    const speech_type type = speech_option_type(option);
    const char * text = NULL;
    int64_t integer = 0;
    double number = 0;
    int boolean = 0;
    speech_status s = SPEECH_OK;
    switch (type) {
        case SPEECH_TYPE_STRING: s = speech_model_info_option_default_string(info, option, &text); break;
        case SPEECH_TYPE_INT: s = speech_model_info_option_default_int(info, option, &integer); break;
        case SPEECH_TYPE_FLOAT: s = speech_model_info_option_default_float(info, option, &number); break;
        case SPEECH_TYPE_BOOL: s = speech_model_info_option_default_bool(info, option, &boolean); break;
    }
    const speech_status want = !speech_model_info_takes(info, option) ? SPEECH_ERROR_UNSUPPORTED
                               : speech_model_info_option_has_default(info, option) ? SPEECH_OK
                                                                                    : SPEECH_ERROR_INVALID_ARGUMENT;
    if (s != want || (s < 0 && (!speech_last_error_option() || strcmp(speech_last_error_option(), name) != 0))) {
        fprintf(stderr, "FAIL: the default of %s is %s where %s was expected\n", name, speech_status_name(s), speech_status_name(want));
        return 1;
    }
    // A string option other than voice and language has the choices its family declares, among them its default, or
    // none, as prompt, which takes any text.
    size_t choices = speech_model_info_option_choice_count(info, option);
    size_t want_choices = !speech_model_info_takes(info, option) ? 0
                          : option == SPEECH_OPT_VOICE           ? speech_model_info_voice_count(info)
                          : option == SPEECH_OPT_LANGUAGE        ? speech_model_info_language_count(info)
                          : option == SPEECH_OPT_PROMPT          ? 0
                          : type == SPEECH_TYPE_STRING           ? choices
                                                                 : 0;
    if (choices != want_choices || (choices > 0 && speech_model_info_option_choice(info, option, choices) != NULL)) {
        fprintf(stderr, "FAIL: %s has %zu choices where %zu were expected\n", name, choices, want_choices);
        return 1;
    }
    int default_chosen = s != SPEECH_OK || type != SPEECH_TYPE_STRING || choices == 0 || option == SPEECH_OPT_LANGUAGE;
    for (size_t k = 0; k < choices && !default_chosen; k++) default_chosen = !strcmp(speech_model_info_option_choice(info, option, k), text);
    if (!default_chosen) {
        fprintf(stderr, "FAIL: the default of %s, %s, is not among its choices\n", name, text);
        return 1;
    }
    if (s == SPEECH_OK) {
        printf("%s: default ", name);
        if (type == SPEECH_TYPE_STRING) printf("%s", text);
        else if (type == SPEECH_TYPE_INT) printf("%lld", (long long) integer);
        else if (type == SPEECH_TYPE_FLOAT) printf("%g", number);
        else printf("%s", boolean ? "true" : "false");
        printf(", %zu choices, steers %d\n", choices, speech_model_info_option_steers(info, option));
    }
    return 0;
}

int check_option_refusals(speech_model * model) {
    speech_model_info * info = NULL;
    speech_request * r = NULL;
    if (speech_model_get_info(model, &info) != SPEECH_OK) return fail("speech_model_get_info");
    if (speech_request_new(model, &r) != SPEECH_OK) return fail("speech_request_new");
    int ok = 1;
    char what[256];
    for (size_t i = 0; i < speech_option_count(); i++) {
        const speech_option o = (speech_option) i;
        const char * name = speech_option_name(o);
        const speech_type type = speech_option_type(o);
        snprintf(what, sizeof what, "%s through a setter of another type", name);
        ok &= expect(type == SPEECH_TYPE_STRING ? speech_request_set_int(r, o, 1) : speech_request_set_string(r, o, "1"),
                     SPEECH_ERROR_INVALID_ARGUMENT, name, what);
        if (type == SPEECH_TYPE_FLOAT) {
            snprintf(what, sizeof what, "%s at NaN", name);
            ok &= expect(speech_request_set_float(r, o, NAN), SPEECH_ERROR_INVALID_ARGUMENT, name, what);
        }
        int has_neutral = 0;
        const speech_status neutral = set_neutral(r, o, &has_neutral);
        if (has_neutral) {
            snprintf(what, sizeof what, "%s at its neutral value", name);
            ok &= expect(neutral, SPEECH_OK, NULL, what);
            if (neutral == SPEECH_OK) printf("%s: accepted at its neutral value\n", what);
        }
        if (check_declaration(info, o) != 0) return 1;
        if (!speech_model_info_takes(info, o)) {
            snprintf(what, sizeof what, "%s, which the model does not take", name);
            speech_status s = SPEECH_OK;
            switch (type) {
                case SPEECH_TYPE_STRING: s = speech_request_set_string(r, o, "x"); break;
                case SPEECH_TYPE_INT: s = speech_request_set_int(r, o, 1); break;
                case SPEECH_TYPE_FLOAT: s = speech_request_set_float(r, o, 2.0); break;
                case SPEECH_TYPE_BOOL: s = speech_request_set_bool(r, o, 1); break;
            }
            ok &= expect(s, SPEECH_ERROR_UNSUPPORTED, name, what);
            continue;
        }
        if (type == SPEECH_TYPE_INT || type == SPEECH_TYPE_FLOAT) {
            double minimum = 0, maximum = 0;
            int exclusive = 0;
            if (speech_model_info_option_range(info, o, &minimum, &maximum, &exclusive) != SPEECH_OK) return fail("speech_model_info_option_range");
            if (isfinite(minimum)) {
                // A number below a minimum as large as a float's, -3.4e38, which 1 less does not change in a double.
                const double below = exclusive ? minimum : type == SPEECH_TYPE_INT ? minimum - 1 : minimum - fabs(minimum) - 1;
                snprintf(what, sizeof what, "%s below its range", name);
                ok &= expect(type == SPEECH_TYPE_INT ? speech_request_set_int(r, o, (int64_t) below) : speech_request_set_float(r, o, below),
                             SPEECH_ERROR_OUT_OF_RANGE, name, what);
            }
            if (isfinite(maximum)) {
                snprintf(what, sizeof what, "%s above its range", name);
                ok &= expect(type == SPEECH_TYPE_INT ? speech_request_set_int(r, o, (int64_t) maximum + 1)
                                                     : speech_request_set_float(r, o, maximum * 2 + 1),
                             SPEECH_ERROR_OUT_OF_RANGE, name, what);
            }
        } else if (o != SPEECH_OPT_VOICE && o != SPEECH_OPT_LANGUAGE && speech_model_info_option_choice_count(info, o) > 0) {
            // The choices a family declares are compared with case.
            for (size_t k = 0; k < speech_model_info_option_choice_count(info, o); k++) {
                const char * choice = speech_model_info_option_choice(info, o, k);
                char upper[256];
                snprintf(upper, sizeof upper, "%s", choice);
                for (char * c = upper; *c; c++) *c = (char) (*c >= 'a' && *c <= 'z' ? *c - 32 : *c);
                snprintf(what, sizeof what, "%s %s, one of its choices", name, choice);
                ok &= expect(speech_request_set_string(r, o, choice), SPEECH_OK, NULL, what);
                snprintf(what, sizeof what, "%s %s, one of its choices in other case", name, upper);
                ok &= expect(speech_request_set_string(r, o, upper), SPEECH_ERROR_OUT_OF_RANGE, name, what);
            }
            snprintf(what, sizeof what, "%s not among its choices", name);
            ok &= expect(speech_request_set_string(r, o, "no-such-choice"), SPEECH_ERROR_OUT_OF_RANGE, name, what);
        } else if (o == SPEECH_OPT_VOICE || o == SPEECH_OPT_LANGUAGE) {
            snprintf(what, sizeof what, "%s not among its choices", name);
            ok &= expect(speech_request_set_string(r, o, o == SPEECH_OPT_LANGUAGE ? "zz" : "no-such-voice"), SPEECH_ERROR_OUT_OF_RANGE, name,
                         what);
            if (o == SPEECH_OPT_VOICE && speech_model_info_voice_count(info) > 0) {
                char upper[256];
                snprintf(upper, sizeof upper, "%s", speech_model_info_voice_name(info, 0));
                for (char * c = upper; *c; c++) *c = (char) (*c >= 'a' && *c <= 'z' ? *c - 32 : *c);
                snprintf(what, sizeof what, "the voice %s, a voice's name in other case", upper);
                ok &= expect(speech_request_set_string(r, o, upper), SPEECH_ERROR_OUT_OF_RANGE, name, what);
            }
            if (o == SPEECH_OPT_LANGUAGE) {
                char region[64];
                snprintf(region, sizeof region, "%s-XX", speech_model_info_language(info, 0));
                snprintf(what, sizeof what, "the language %s, a region of one of the model's", region);
                ok &= expect(speech_request_set_string(r, o, region), SPEECH_OK, NULL, what);
                // A language of three letters, which has no code of two, is named by all three; its first two name
                // another language or none.
                for (size_t l = 0; l < speech_model_info_language_count(info); l++) {
                    const char * language = speech_model_info_language(info, l);
                    if (strlen(language) != 3) continue;
                    char upper[8], two[3] = {language[0], language[1], '\0'};
                    snprintf(region, sizeof region, "%s-PH", language);
                    snprintf(upper, sizeof upper, "%c%c%c", language[0] - 32, language[1] - 32, language[2] - 32);
                    snprintf(what, sizeof what, "the language %s", region);
                    ok &= expect(speech_request_set_string(r, o, region), SPEECH_OK, NULL, what);
                    snprintf(what, sizeof what, "the language %s", upper);
                    ok &= expect(speech_request_set_string(r, o, upper), SPEECH_OK, NULL, what);
                    int two_is_one = 0;
                    for (size_t k = 0; k < speech_model_info_language_count(info); k++) two_is_one |= !strcmp(speech_model_info_language(info, k), two);
                    snprintf(what, sizeof what, "the language %s, the first two letters of %s", two, language);
                    ok &= expect(speech_request_set_string(r, o, two), two_is_one ? SPEECH_OK : SPEECH_ERROR_OUT_OF_RANGE, two_is_one ? NULL : name, what);
                }
            }
        }
    }
    ok &= expect(speech_request_set_int(r, (speech_option) 99, 1), SPEECH_ERROR_INVALID_ARGUMENT, NULL, "an option the library does not know");
    ok &= expect(speech_request_set_string(r, SPEECH_OPT_VOICE, NULL), SPEECH_ERROR_INVALID_ARGUMENT, "voice", "a NULL string");
    speech_request_free(r);
    speech_model_info_free(info);
    if (!ok) fprintf(stderr, "FAIL: an option's value is not refused or accepted as speech.h declares\n");
    return ok ? 0 : 1;
}

char * read_whole(const char * path, size_t * size) {
    FILE * f = open_utf8(path, "rb");
    if (!f) return NULL;
    size_t capacity = 1 << 16, n = 0;
    char * data = (char *) malloc(capacity);
    for (;;) {
        n += fread(data + n, 1, capacity - n - 1, f);
        if (n < capacity - 1) break;
        capacity *= 2;
        data = (char *) realloc(data, capacity);
    }
    fclose(f);
    data[n] = '\0';
    *size = n;
    return data;
}

float * read_audio(const char * dump, size_t * n) {
    char path[4096];
    snprintf(path, sizeof path, "%s/audio.npy", dump);
    size_t size = 0;
    char * data = read_whole(path, &size);
    if (!data) return NULL;
    const size_t header = size >= 10 ? (size_t) (unsigned char) data[8] | (size_t) (unsigned char) data[9] << 8 : 0;
    if (size < 10 || memcmp(data, "\x93NUMPY\x01", 7) != 0 || 10 + header > size || !strstr(data + 10, "'<f4'") ||
        !strstr(data + 10, "'fortran_order': False") || !strstr(data + 10, ",)")) {
        free(data);
        return NULL;
    }
    *n = (size - 10 - header) / sizeof(float);
    float * samples = (float *) malloc(*n * sizeof(float));
    memcpy(samples, data + 10 + header, *n * sizeof(float));
    free(data);
    return samples;
}

int record_progress(double done, void * user_data) {
    Progress * p = (Progress *) user_data;
    if (p->n == 0) p->first = done;
    else if (done < p->last) p->fell = 1;
    p->last = done;
    p->n++;
    return p->stop_at > 0 && done >= p->stop_at;
}

int progress_rises(const Progress * p, double last, const char * what) {
    if (p->n == 0 || p->first < 0 || p->last != last || p->fell) {
        fprintf(stderr, "FAIL: %s: the progress does not rise to %g: %d reports from %g to %g%s\n", what, last, p->n, p->first, p->last,
                p->fell ? ", falling on the way" : "");
        return 0;
    }
    printf("%s: progress in %d reports from %g to %g\n", what, p->n, p->first, p->last);
    return 1;
}

int collect(const float * samples, size_t n, void * user_data) {
    Audio * a = (Audio *) user_data;
    if (a->n + n > a->capacity) {
        a->capacity = (a->n + n) * 2;
        a->samples = (float *) realloc(a->samples, a->capacity * sizeof(float));
        if (!a->samples) {
            fprintf(stderr, "out of memory\n");
            exit(1);
        }
    }
    memcpy(a->samples + a->n, samples, n * sizeof(float));
    a->n += n;
    return 0;
}

/** A request of `text` in `voice` (NULL for none) with `seed` (negative for none). */
speech_request * new_request(speech_model * model, const char * text, const char * voice, int64_t seed) {
    speech_request * r = NULL;
    if (speech_request_new(model, &r) != SPEECH_OK || (text && speech_request_set_text(r, text) != SPEECH_OK) ||
        (voice && speech_request_set_string(r, SPEECH_OPT_VOICE, voice) != SPEECH_OK) ||
        (seed >= 0 && speech_request_set_int(r, SPEECH_OPT_SEED, seed) != SPEECH_OK)) {
        fail("a request");
        exit(1);
    }
    return r;
}

speech_status speak(speech_request * r, Audio * audio, speech_stop * stop, int64_t * seed) {
    const size_t before = audio->n;
    speech_status s = speech_synthesize(r, collect, audio);
    const speech_result * result = speech_request_result(r);
    if (stop) *stop = result ? speech_result_stop(result) : SPEECH_STOP_COMPLETE;
    if (seed) *seed = result ? speech_result_seed(result) : -1;
    const size_t passed = audio->n - before;
    if (result && speech_result_samples(result) != passed) {
        fprintf(stderr, "FAIL: a result counts %llu samples where the callback was given %zu\n",
                (unsigned long long) speech_result_samples(result), passed);
        s = SPEECH_ERROR_INTERNAL;
    }
    speech_request_free(r);
    return s;
}

/** A request that another thread cancels once it has reported progress or passed audio. */
typedef struct {
    speech_request * request;
    Monitor monitor;
    int started, cancelled, after_cancel;
} Cancelling;

static void started(Cancelling * c) {
    monitor_lock(&c->monitor);
    c->started = 1;
    if (c->cancelled) c->after_cancel++;
    monitor_signal(&c->monitor);
    monitor_unlock(&c->monitor);
}

static int on_cancelling_audio(const float * samples, size_t n, void * user_data) {
    (void) samples;
    (void) n;
    started((Cancelling *) user_data);
    return 0;
}

static int on_cancelling_progress(double done, void * user_data) {
    (void) done;
    started((Cancelling *) user_data);
    return 0;
}

static void canceller(void * arg) {
    Cancelling * c = (Cancelling *) arg;
    monitor_lock(&c->monitor);
    while (!c->started) monitor_wait(&c->monitor);
    speech_request_cancel(c->request);
    c->cancelled = 1;
    monitor_unlock(&c->monitor);
}

/**
 * Runs `request`, which it frees, while another thread cancels it once it has reported progress or passed audio;
 * whether it stopped as cancelled with at most one call of a callback after the cancel.
 */
int cancelled_from_another_thread(speech_request * request, const char * what) {
    Cancelling c;
    memset(&c, 0, sizeof c);
    monitor_init(&c.monitor);
    c.request = request;
    speech_request_set_progress(request, on_cancelling_progress, &c);
    Thread thread;
    thread_start(&thread, canceller, &c);
    const speech_status status = speech_synthesize(request, on_cancelling_audio, &c);
    thread_join(&thread);
    const speech_result * result = speech_request_result(request);
    const int ok = expect(status, SPEECH_CANCELLED, NULL, what) && result && speech_result_stop(result) == SPEECH_STOP_CANCELLED &&
                   c.after_cancel <= 1;
    printf("%s: %llu samples, %d call(s) of a callback after the cancel\n", what, result ? (unsigned long long) speech_result_samples(result) : 0ULL,
           c.after_cancel);
    speech_request_free(request);
    monitor_free(&c.monitor);
    return ok;
}
