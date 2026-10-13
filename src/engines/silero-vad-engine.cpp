#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "engine.h"
#include "silero-vad/detector.h"
#include "silero-vad/regions.h"

namespace {

/** The rule of a request's options, with the constants of the model file. */
silero_vad::RegionRule rule_of(const silero_vad::Detector & detector, const RequestValues & values) {
    silero_vad::RegionRule rule = silero_vad::default_rule(detector.model());
    rule.threshold = values.number(SPEECH_OPT_THRESHOLD);
    rule.min_speech_duration_ms = values.integer(SPEECH_OPT_MIN_SPEECH_DURATION_MS);
    rule.min_silence_duration_ms = values.integer(SPEECH_OPT_MIN_SILENCE_DURATION_MS);
    rule.speech_pad_ms = values.integer(SPEECH_OPT_SPEECH_PAD_MS);
    if (values.has(SPEECH_OPT_MAX_SPEECH_DURATION_S)) rule.max_speech_duration_s = values.number(SPEECH_OPT_MAX_SPEECH_DURATION_S);
    return rule;
}

/** A region in samples as the C API gives it, in seconds at the model's rate. */
TimedText in_seconds(const silero_vad::Region & r, int sample_rate) {
    return {(double) r.start / sample_rate, (double) r.end / sample_rate, ""};
}

/**
 * The chunks of the audio computed as their samples arrive and walked by the region rule at once, which settles the
 * regions after each push from the samples heard so far.
 */
class SileroVadStream : public DetectionStream {
public:
    SileroVadStream(silero_vad::Detector & detector, const silero_vad::RegionRule & rule)
        : chunks_(detector), rule_(rule, detector.sample_rate(), detector.network().chunk()), sample_rate_(detector.sample_rate()) {}

    void push(const float * samples, size_t n) override {
        probs_.clear();
        chunks_.push(samples, n, probs_);
        for (const float p : probs_) rule_.add(p);
        rule_.settle(chunks_.samples());
        collect();
    }

    void end() override {
        probs_.clear();
        chunks_.end(probs_);
        for (const float p : probs_) rule_.add(p);
        rule_.end(chunks_.samples());
        collect();
    }

    const std::vector<TimedText> & regions() const override { return regions_; }

    std::optional<Begun> open() const override {
        const std::optional<silero_vad::Begun> begun = rule_.open();
        if (!begun) return std::nullopt;
        return Begun{(double) begun->start / sample_rate_, begun->kept};
    }

private:
    void collect() {
        for (size_t i = regions_.size(); i < rule_.regions().size(); i++) regions_.push_back(in_seconds(rule_.regions()[i], sample_rate_));
    }

    silero_vad::ChunkStream chunks_;
    silero_vad::RegionStream rule_;
    int sample_rate_;
    std::vector<float> probs_;
    std::vector<TimedText> regions_;
};

class SileroVadEngine : public Engine {
public:
    SileroVadEngine(const std::string & path, ggml_backend_t backend) : detector_(path, backend) {}

    /** The blocks of chunks report their progress as they finish; the regions follow from the probabilities at once. */
    std::vector<TimedText> detect(const std::vector<float> & samples, const RequestValues & values, Run & run) override {
        const silero_vad::RegionRule rule = rule_of(detector_, values);
        const std::vector<float> probs = detector_.probabilities(samples, [&](double done) { return run.progress(done); });
        if (run.stopped()) return {};
        std::vector<TimedText> out;
        for (const silero_vad::Region & r : silero_vad::speech_regions(probs, (int64_t) samples.size(), detector_.sample_rate(), detector_.network().chunk(), rule)) {
            out.push_back(in_seconds(r, detector_.sample_rate()));
        }
        return out;
    }

    std::unique_ptr<DetectionStream> start_detection(const RequestValues & values) override {
        return std::make_unique<SileroVadStream>(detector_, rule_of(detector_, values));
    }

    void warm_up() override {
        // The first graph builds Metal's pipelines and compiles Vulkan's shaders.
        detector_.probabilities(std::vector<float>((size_t) detector_.sample_rate(), 0.0f));
    }

private:
    silero_vad::Detector detector_;
};

}  // namespace

/**
 * The table of the options Silero VAD takes: those of get_speech_timestamps() that a caller tunes, under its names and
 * with its defaults, which the file gives. The model takes no language and no timestamps, whose neutral values every
 * model accepts. max_speech_duration_s has no default, as the official's infinity is no value a request sets.
 */
FamilyInfo describe_silero_vad(const std::shared_ptr<const ModelFile> & file) {
    const ModelFile & m = *file;
    FamilyInfo info;
    info.options = {
        {SPEECH_OPT_THRESHOLD, false, true, m.f64("silero-vad.threshold"), 0, 1},
        {SPEECH_OPT_MIN_SPEECH_DURATION_MS, false, true, (int64_t) m.u32("silero-vad.min_speech_duration_ms"), 0},
        {SPEECH_OPT_MIN_SILENCE_DURATION_MS, false, true, (int64_t) m.u32("silero-vad.min_silence_duration_ms"), 0},
        {SPEECH_OPT_SPEECH_PAD_MS, false, true, (int64_t) m.u32("silero-vad.speech_pad_ms"), 0},
        {SPEECH_OPT_MAX_SPEECH_DURATION_S, false, true, std::nullopt, 0, INFINITY, true},
    };
    return info;
}

std::unique_ptr<Engine> load_silero_vad(const std::string & path, ggml_backend_t backend, int) {
    return std::make_unique<SileroVadEngine>(path, backend);
}
