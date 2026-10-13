#include "regions.h"

#include <algorithm>
#include <cmath>

namespace {

/**
 * The longest region, in seconds, that transcription by regions gives a recognizer. On 106 minutes of FLEURS and Common
 * Voice ja joined into recordings (measure/region_cap_compare.py, Apple M5, Metal, 2026-10-08), 10 s gave the lowest
 * CER on average over reazonspeech-nemo-v2, parakeet-tdt_ctc-0.6b-ja and Qwen3-ASR 0.6B: longer regions let the
 * FastConformer models drop more sentences, and shorter ones cost Qwen3-ASR the context it reads them with.
 */
constexpr double kLongestRegion = 10;

}  // namespace

std::string server_vad_member(const std::string & option) {
    for (const VadMember & m : kServerVad) {
        if (option == speech_option_name(m.option)) return m.name;
    }
    return "";
}

std::pair<size_t, size_t> region_samples(const Region & region, int rate, size_t n) {
    const auto at = [&](double seconds) { return std::min(n, (size_t) std::max(0.0, std::round(seconds * rate))); };
    return {at(region.start), at(region.end)};
}

std::vector<RequestOption> region_options(const std::vector<RequestOption> & given, const std::vector<RequestOption> & defaults) {
    std::vector<RequestOption> out = {
        {SPEECH_OPT_THRESHOLD, 0.5},
        {SPEECH_OPT_SPEECH_PAD_MS, (int64_t) 300},
        {SPEECH_OPT_MIN_SILENCE_DURATION_MS, (int64_t) 500},
        {SPEECH_OPT_MAX_SPEECH_DURATION_S, kLongestRegion},
    };
    for (const RequestOption & o : with_defaults(defaults, given)) {
        const auto same = std::find_if(out.begin(), out.end(), [&](const RequestOption & d) { return d.option == o.option; });
        if (same != out.end()) *same = o;
        else out.push_back(o);
    }
    return out;
}

Request detection_request(speech_model * detector, const std::vector<float> & samples, int rate, const std::vector<RequestOption> & options) {
    Request request = new_request(detector);
    check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
    apply_options(request.get(), options);
    return request;
}

std::optional<std::vector<Region>> detect_regions(speech_request * detection, Cancellation & cancellation) {
    if (check(cancellation.run(detection, [&] { return speech_detect(detection); })) == SPEECH_CANCELLED) return std::nullopt;
    const speech_result * result = speech_request_result(detection);
    std::vector<Region> regions(speech_result_segment_count(result));
    for (size_t i = 0; i < regions.size(); i++) check(speech_result_segment(result, i, &regions[i].start, &regions[i].end, nullptr));
    return regions;
}

void check_recognition(speech_model * recognizer, const std::vector<RequestOption> & options) {
    const ModelInfo info = model_info(recognizer);
    const speech_task task = speech_model_info_task(info.get());
    if (task != SPEECH_TASK_RECOGNITION) {
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      std::string(speech_model_info_name(info.get())) + " is a model of speech " + task_name(task) +
                          "; transcription by regions recognizes them with a speech recognition model");
    }
    apply_options(new_request(recognizer).get(), options);
}

std::optional<Transcript> transcribe_regions(speech_model * recognizer, const std::vector<float> & samples, int rate,
                                             const std::vector<Region> & regions, const std::vector<RequestOption> & options,
                                             bool timestamps, Cancellation & cancellation) {
    check_recognition(recognizer, options);
    Transcript whole;
    for (const Region & region : regions) {
        const auto [first, last] = region_samples(region, rate, samples.size());
        const Request request = new_request(recognizer);
        check(speech_request_set_audio(request.get(), samples.data() + first, last - first, rate));
        apply_options(request.get(), options);
        if (check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); })) == SPEECH_CANCELLED) return std::nullopt;
        append_part(whole, transcript_of(speech_request_result(request.get()), timestamps), (double) first / rate);
    }
    return whole;
}
