#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "language.h"
#include "qwen3-asr/recognizer.h"

namespace {

class Qwen3AsrEngine : public Engine {
public:
    Qwen3AsrEngine(const std::string & path, ggml_backend_t backend) : recognizer_(path, backend) {}

    Recognized transcribe(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        qwen3_asr::RecognitionRequest request;
        request.language = forced_language(values.string(SPEECH_OPT_LANGUAGE));
        request.context = values.string(SPEECH_OPT_PROMPT);
        const qwen3_asr::Recognition found = recognizer_.recognize(samples, request, [&](double done) { return run.progress(done); });
        if (run.stopped()) return {};
        run.progress(1);
        Recognized out;
        out.text = found.text;
        out.stop = found.limited ? SPEECH_STOP_MODEL_LIMIT : SPEECH_STOP_COMPLETE;
        for (size_t language : found.languages) out.languages.push_back(recognizer_.languages()[language]);
        return out;
    }

    void warm_up() override {
        // The first recognition builds Metal's pipelines and compiles Vulkan's shaders, which take seconds.
        recognizer_.recognize(std::vector<float>((size_t) recognizer_.sample_rate(), 0.0f), {}, [](double) { return true; });
    }

private:
    /** The model's language that a request's tag names, which the C API has checked, or none for auto. */
    std::optional<size_t> forced_language(const std::string & tag) const {
        if (language_is_auto(tag)) return std::nullopt;
        const std::vector<std::string> & languages = recognizer_.languages();
        for (size_t i = 0; i < languages.size(); i++) {
            if (bcp47_matches(tag, languages[i])) return i;
        }
        throw std::logic_error("the language " + tag + " passed the C API's check without naming a language of the model");
    }

    qwen3_asr::Recognizer recognizer_;
};

}  // namespace

/**
 * The table of the options Qwen3-ASR takes. The language steers: a language forced as qwen-asr forces it, by the prefill
 * "language <Name><asr_text>", or auto, with which the model writes the language it hears before the text; the result
 * gives either as its tag. The prompt
 * is qwen-asr's context, which goes into the system turn of the prompt. The model gives no times, so a request takes
 * timestamps only as false, the neutral value.
 */
FamilyInfo describe_qwen3_asr(const std::shared_ptr<const ModelFile> &) {
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_LANGUAGE, false, true, std::string("auto")},
        {SPEECH_OPT_PROMPT, false, true, std::string()},
    };
    return info;
}

std::unique_ptr<Engine> load_qwen3_asr(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<Qwen3AsrEngine>(path, backend);
}
