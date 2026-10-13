#pragma once

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "irodori-tts/sampler.h"
#include "irodori-tts/synthesizer.h"
#include "json-reader.h"

/**
 * A dump of reference/irodori-tts/dump.py and the request it was made with, from its meta.json: the length's settings,
 * whether it spoke with a reference or without one, as the voice none does, and, under "options", the request options
 * it set by the C API's names. An option it did not set is the model's default, and the dumps made before the options
 * existed set none.
 */
class IrodoriDump {
public:
    explicit IrodoriDump(std::filesystem::path dir) : dir_(std::move(dir)) {
        std::ifstream f(dir_ / "meta.json");
        if (!f) throw std::runtime_error("cannot open " + (dir_ / "meta.json").u8string());
        meta_ = parse_json(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
        const JsonValue * options = meta_.member("options");
        if (options) options_ = options->members;
        for (const auto & option : options_) {
            if (std::find(std::begin(kKnown), std::end(kKnown), option.first) == std::end(kKnown)) {
                throw std::runtime_error((dir_ / "meta.json").u8string() + " sets the option " + option.first + ", which the checks do not know");
            }
        }
    }

    const std::filesystem::path & dir() const { return dir_; }
    std::filesystem::path file(const std::string & name) const { return dir_ / name; }
    bool has(const std::string & name) const { return std::filesystem::exists(dir_ / name); }

    /** The model's repository, as general.source.repo_url names it. */
    std::string repository() const { return "https://huggingface.co/" + string(*member(*member(meta_, "model"), "repository")); }
    std::string text() const { return string(*member(meta_, "text")); }
    /** Whether the request spoke with a reference; meta.json's "reference" is null for one without. */
    bool reference() const { return member(meta_, "reference")->kind != JsonValue::Kind::Null; }
    /** Whether the reference was a speaker-inversion embedding, which the dump's speaker_state is. */
    bool embedding() const {
        const JsonValue * v = meta_.member("embedding");
        return v && v->boolean;
    }
    /** The request's instructions, the runtime's caption, or "" for a request without. */
    std::string instructions() const {
        for (const auto & [name, v] : options_) {
            if (name == "instructions") return v.text;
        }
        return "";
    }
    /** Whether the request's references were brought to a loudness of their own (ref_normalize_db), or kept. */
    bool has_loudness() const { return meta_.member("ref_normalize_db") != nullptr; }
    /** That loudness in LUFS, or none for references kept as recorded. */
    std::optional<double> loudness() const {
        const JsonValue * v = member(meta_, "ref_normalize_db");
        if (v->kind == JsonValue::Kind::Null) return std::nullopt;
        return std::stod(v->text);
    }
    /** Whether the runtime spoke with a caption, which dump.py saved the condition of. */
    bool caption() const { return has("caption_state.npy"); }
    /** A number of meta.json, or `absent` when it has none. */
    double number(const std::string & key, double absent) const {
        const JsonValue * v = meta_.member(key);
        return v && v->kind == JsonValue::Kind::Number ? std::stod(v->text) : absent;
    }

    /** The request's length's settings. */
    irodori::LengthOptions length() const {
        irodori::LengthOptions o;
        o.seconds = number("seconds", 0);
        o.duration_scale = number("duration_scale", 1);
        o.speed = number("speed", 1);
        return o;
    }

    /** Whether the request set an option of RF's guidance or schedule. */
    bool sets_guidance() const {
        for (const auto & option : options_) {
            if (option.first.rfind("tail_", 0) != 0 && option.first != "keep_tail" && option.first != "instructions") return true;
        }
        return false;
    }

    /** The RF request's guidance: `defaults`, the model file's, with what the request set. */
    irodori::Guidance guidance(irodori::Guidance g) const {
        for (const auto & [name, v] : options_) {
            if (name == "cfg_scale_text") g.text = (float) std::stod(v.text);
            else if (name == "cfg_scale_speaker") g.speaker = (float) std::stod(v.text);
            else if (name == "cfg_guidance_mode") g.mode = mode(v.text);
            else if (name == "cfg_min_t") g.min_t = std::stod(v.text);
            else if (name == "cfg_max_t") g.max_t = std::stod(v.text);
            else if (name == "truncation_factor") g.truncation = (float) std::stod(v.text);
            else if (name == "rescale_k") g.rescale_k = std::stod(v.text);
            else if (name == "rescale_sigma") g.rescale_sigma = std::stod(v.text);
            else if (name == "speaker_uncond_mode") g.speaker_noise = v.text == irodori::kSpeakerNoise;
            else if (name == "sway_coeff") g.sway = (float) std::stod(v.text);
            else if (name == "speaker_kv_scale") g.speaker_kv_scale = (float) std::stod(v.text);
            else if (name == "speaker_kv_min_t") g.speaker_kv_min_t = (float) std::stod(v.text);
            else if (name == "speaker_kv_max_layers") g.speaker_kv_layers = std::stoi(v.text);
            else if (name == "cfg_scale_instructions") g.caption = (float) std::stod(v.text);
        }
        return g;
    }

    /** The request's cut at the tail: `defaults`, the model file's, with what the request set. */
    irodori::TailCut tail(irodori::TailCut t) const {
        for (const auto & [name, v] : options_) {
            if (name == "keep_tail") t.keep = v.boolean;
            else if (name == "tail_window_size") t.window = std::stoi(v.text);
            else if (name == "tail_std_threshold") t.std_threshold = (float) std::stod(v.text);
            else if (name == "tail_mean_threshold") t.mean_threshold = (float) std::stod(v.text);
        }
        return t;
    }

private:
    /** The options dump.py writes. */
    static constexpr const char * kKnown[] = {"cfg_scale_text", "cfg_scale_speaker", "cfg_guidance_mode", "cfg_min_t", "cfg_max_t",
                                              "truncation_factor", "rescale_k", "rescale_sigma", "speaker_uncond_mode", "sway_coeff",
                                              "keep_tail", "tail_window_size", "tail_std_threshold", "tail_mean_threshold",
                                              "speaker_kv_scale", "speaker_kv_min_t", "speaker_kv_max_layers", "instructions",
                                              "cfg_scale_instructions"};

    static irodori::GuidanceMode mode(const std::string & name) {
        for (size_t i = 0; i < std::size(irodori::kGuidanceModes); i++) {
            if (name == irodori::kGuidanceModes[i]) return (irodori::GuidanceMode) i;
        }
        throw std::runtime_error("the guidance mode " + name + " is not one this check knows");
    }

    const JsonValue * member(const JsonValue & object, const std::string & key) const {
        const JsonValue * v = object.member(key);
        if (!v) throw std::runtime_error(key + " is missing from " + (dir_ / "meta.json").u8string());
        return v;
    }

    std::string string(const JsonValue & v) const {
        if (v.kind != JsonValue::Kind::String) throw std::runtime_error("a string of " + (dir_ / "meta.json").u8string() + " is not one");
        return v.text;
    }

    std::filesystem::path dir_;
    JsonValue meta_;
    std::vector<std::pair<std::string, JsonValue>> options_;
};

/**
 * The attention that the check's argument `index` forces, "flash" or "products", or none where the argument is absent,
 * so that both attentions are checked on a GPU.
 */
inline std::optional<irodori::Attention> forced_attention(const std::vector<std::string> & args, size_t index) {
    if (args.size() <= index) return std::nullopt;
    if (args[index] == "flash") return irodori::Attention::Flash;
    if (args[index] == "products") return irodori::Attention::Products;
    throw std::runtime_error("the attention is flash or products, not " + args[index]);
}

/** The dumps under `root` made with the model of `repository` that hold `file`, sorted by their folders' names. */
inline std::vector<IrodoriDump> irodori_dumps(const std::string & root, const std::string & repository, const std::string & file) {
    std::vector<std::filesystem::path> dirs;
    for (const auto & e : std::filesystem::directory_iterator(std::filesystem::u8path(root))) {
        if (std::filesystem::exists(e.path() / file) && std::filesystem::exists(e.path() / "meta.json")) dirs.push_back(e.path());
    }
    std::sort(dirs.begin(), dirs.end());
    std::vector<IrodoriDump> dumps;
    for (const auto & d : dirs) {
        IrodoriDump dump(d);
        if (dump.repository() == repository) dumps.push_back(std::move(dump));
    }
    return dumps;
}
