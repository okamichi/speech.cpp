#include <climits>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "engine.h"
#include "error.h"
#include "json.h"
#include "qwen3-tts/codec.h"
#include "qwen3-tts/synthesizer.h"

namespace {

/** The frames whose audio fits in `seconds`, counted in whole samples so that 0.24 s at 24 kHz is three frames. */
int frames_within(double seconds, int sample_rate, int samples_per_frame) {
    return (int) ((int64_t) std::floor(seconds * sample_rate) / samples_per_frame);
}

/**
 * A float of the model file as the shortest decimal that reads back as it (0.9 for the float nearest 0.9), so that the
 * information shows the value the checkpoint gives. Converting that decimal back to float gives the same float for every
 * float from 2^-20 to 2^20, as a run through all of them showed on 2026-10-07, so a request at the default the
 * information shows samples with the file's own float.
 */
double decimal(float v) {
    return std::strtod(json_number(v).c_str(), nullptr);
}

/** The options that set one stack's sampling; the code predictor's repetition penalty is none, and stays the file's. */
struct SamplingOptions {
    speech_option do_sample, top_k, top_p, temperature;
    std::optional<speech_option> repetition_penalty;
};

const SamplingOptions kTalkerOptions = {SPEECH_OPT_DO_SAMPLE, SPEECH_OPT_TOP_K, SPEECH_OPT_TOP_P, SPEECH_OPT_TEMPERATURE,
                                        SPEECH_OPT_REPETITION_PENALTY};
const SamplingOptions kCodePredictorOptions = {SPEECH_OPT_CODE_PREDICTOR_DO_SAMPLE, SPEECH_OPT_CODE_PREDICTOR_TOP_K,
                                               SPEECH_OPT_CODE_PREDICTOR_TOP_P, SPEECH_OPT_CODE_PREDICTOR_TEMPERATURE, std::nullopt};

/**
 * A temperature or a repetition penalty as the float the sampler divides by, as torch turns a Python float into the
 * float32 of the logits it divides. One that the float rounds to 0 or to infinity throws.
 */
float divisor(const RequestValues & values, speech_option option) {
    const double v = values.number(option);
    const float f = (float) v;
    if (v > 0 && (f == 0 || std::isinf(f))) {
        throw Error(Fault::OutOfRange, std::string("the option ") + speech_option_name(option) + " is " + json_number(v) +
                                           ", beyond what the sampler's float holds; give a value nearer 1",
                    speech_option_name(option));
    }
    return f;
}

/**
 * One stack's sampling as the request sets it, from the file's where the request sets none. top_k, top_p and the
 * temperature of a stack that does not draw are refused, as transformers' generate() leaves them unused.
 */
SamplingParams sampling(const RequestValues & values, const SamplingOptions & o, SamplingParams p) {
    p.greedy = !values.boolean(o.do_sample);
    for (speech_option unused : {o.top_k, o.top_p, o.temperature}) {
        if (p.greedy && values.given(unused)) {
            throw Error(Fault::InvalidArgument, std::string("the option ") + speech_option_name(unused) + " applies only when " +
                                                    speech_option_name(o.do_sample) + " is true; leave it out or set " +
                                                    speech_option_name(o.do_sample),
                        speech_option_name(unused));
        }
    }
    p.top_k = (int) values.integer(o.top_k);
    p.top_p = (float) values.number(o.top_p);
    p.temperature = divisor(values, o.temperature);
    if (o.repetition_penalty) p.repetition_penalty = divisor(values, *o.repetition_penalty);
    return p;
}

/** The declarations of one stack's options, its defaults the file's. */
std::vector<OptionSpec> sampling_specs(const SamplingOptions & o, const SamplingParams & file) {
    std::vector<OptionSpec> specs = {
        {o.do_sample, false, true, !file.greedy},
        {o.top_k, false, true, (int64_t) file.top_k, 0, (double) INT_MAX},
        {o.top_p, false, true, decimal(file.top_p), 0, 1},
        {o.temperature, false, true, decimal(file.temperature), 0, INFINITY, true},
    };
    if (o.repetition_penalty) specs.push_back({*o.repetition_penalty, false, true, decimal(file.repetition_penalty), 0, INFINITY, true});
    return specs;
}

class Qwen3TtsEngine : public Engine {
public:
    Qwen3TtsEngine(const std::string & path, ggml_backend_t backend) : synth_(path, backend) {}

    speech_stop speak(const std::string & text, const RequestValues & values, Run & run) override {
        SynthesisRequest r;
        r.text = text;
        r.speaker = values.string(SPEECH_OPT_VOICE);
        r.language = values.string(SPEECH_OPT_LANGUAGE);
        r.seed = (uint64_t) values.integer(SPEECH_OPT_SEED);
        if (values.has(SPEECH_OPT_INSTRUCTIONS)) r.instructions = values.string(SPEECH_OPT_INSTRUCTIONS);
        r.talker = sampling(values, kTalkerOptions, synth_.generation().talker);
        r.code_predictor = sampling(values, kCodePredictorOptions, synth_.generation().code_predictor);
        const bool capped = values.has(SPEECH_OPT_MAX_SECONDS);
        if (capped) r.max_frames = frames_within(values.number(SPEECH_OPT_MAX_SECONDS), synth_.sample_rate(), synth_.samples_per_frame());
        // Qwen3-TTS passes its audio in chunks of a few frames as it makes them (chunk_frames()), so it stops at the sink.
        const SynthesisOutcome outcome = synth_.synthesize(r, [&](const float * samples, size_t n) { return run.audio(samples, n); });
        if (outcome.ended) return SPEECH_STOP_COMPLETE;
        return capped && outcome.frames == r.max_frames ? SPEECH_STOP_MAX_SECONDS : SPEECH_STOP_MODEL_LIMIT;
    }

    void warm_up() override {
        // The first synthesis compiles the GPU kernels, which on Vulkan takes seconds for every new shape, so a short and
        // a longer text run through the prompt, the talker, the code predictor and the codec at the sizes speech uses.
        for (const auto & [text, frames] : std::vector<std::pair<std::string, int>>{
                 {"あ", 4}, {"明日の東京は晴れで、最高気温は二十四度の予報です。", 40}}) {
            SynthesisRequest warmup;
            warmup.text = text;
            warmup.speaker = synth_.ids().voices[0];
            warmup.max_frames = frames;
            synth_.synthesize(warmup, [](const float *, size_t) { return true; });
        }
    }

private:
    Synthesizer synth_;
};

/** The tokens of a text as the synthesis counts them against its limit. */
struct TokenCounter {
    explicit TokenCounter(const ModelFile & m) : tokenizer(m, kTextTokenizer) {}
    size_t count(const std::string & text) const {
        return naming("text", [&] { return tokenizer.encode(text); }).size();
    }
    Qwen2Tokenizer tokenizer;
};

}  // namespace

/**
 * The table of the options Qwen3-TTS takes. The official implementation has no control of the rate or the length, and
 * changing the audio's rate afterwards would change its pitch or add a time stretch's artifacts, so it takes neither:
 * its speech ends where the talker ends it, at the request's max_seconds, or at the model's limit of frames. The talker
 * and the code predictor sample as generate_custom_voice() lets a caller set, with its defaults from the file and the
 * ranges transformers' processors take, and a model that takes_instructions() follows an instruction as it does.
 */
FamilyInfo describe_qwen3_tts(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.incremental = true;
    const std::vector<std::string> names = m.str_array("speech.voices"), languages = m.str_array("speech.voice_languages"),
                                   genders = m.str_array("speech.voice_genders"), descriptions = m.str_array("speech.voice_descriptions");
    for (size_t i = 0; i < names.size(); i++) info.voices.push_back({names[i], languages[i], genders[i], descriptions[i]});
    info.max_text_tokens = (size_t) text_token_limit(m);
    const double longest = (double) m.u32("qwen3-tts.generation.max_frames") * codec_samples_per_frame(m) / m.u32("speech.sample_rate");
    info.options = {
        {SPEECH_OPT_VOICE, true, true},
        {SPEECH_OPT_LANGUAGE, false, true, std::string("auto")},
        {SPEECH_OPT_SEED, false, true, std::nullopt, 0, (double) kMaxSeed},
        {SPEECH_OPT_MAX_SECONDS, false, true, std::nullopt, 0, longest, true},
    };
    const Generation generation(m);
    for (const OptionSpec & spec : sampling_specs(kTalkerOptions, generation.talker)) info.options.push_back(spec);
    for (const OptionSpec & spec : sampling_specs(kCodePredictorOptions, generation.code_predictor)) info.options.push_back(spec);
    if (takes_instructions(m)) info.options.push_back({SPEECH_OPT_INSTRUCTIONS, false, true, std::string()});
    const auto counter = std::make_shared<Lazy<TokenCounter>>(file);
    info.count_tokens = [counter](const std::string & text) { return counter->get().count(text); };
    return info;
}

std::unique_ptr<Engine> load_qwen3_tts(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<Qwen3TtsEngine>(path, backend);
}
