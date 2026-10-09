#include "irodori-request.h"

#include <cmath>
#include <map>

#include "openai-error.h"
#include "request-options.h"

namespace openai {

double irodori_request(JsonValue & json, const std::string & model_name) {
    if (const auto * model = json.member("model"); model && model->kind == JsonValue::Kind::String && model->text == "irodori-tts") {
        for (auto & member : json.members) if (member.first == "model") member.second.text = model_name;
    }
    const auto * extra = json.member("irodori");
    if (!extra || extra->kind == JsonValue::Kind::Null) return 0;
    if (extra->kind != JsonValue::Kind::Object) throw ApiError{400, "irodori must be an object.", "irodori", "invalid_type"};
    const auto members = extra->members;
    const std::map<std::string, std::string> names = {
        {"caption", "instructions"}, {"seed", "seed"}, {"num_steps", "steps"},
        {"cfg_scale_text", "cfg_scale_text"}, {"cfg_scale_caption", "cfg_scale_instructions"},
        {"cfg_scale_speaker", "cfg_scale_speaker"}, {"max_seconds", "max_seconds"}, {"chunking_enabled", "chunking_enabled"}
    };
    double limit = 120;
    for (const auto & [key, value] : members) {
        const std::string input = "irodori." + key;
        const auto mapped = names.find(key);
        if (mapped == names.end()) throw ApiError{400, "Unknown Irodori option: " + key + ".", input, "unknown_parameter"};
        if (value.kind == JsonValue::Kind::Null) continue;
        if (const auto * top = json.member(mapped->second); top && top->kind != JsonValue::Kind::Null) {
            throw ApiError{400, "Give " + mapped->second + " either at the top level or in irodori, not both.", input, "invalid_value"};
        }
        if (key == "chunking_enabled") {
            if (value.kind != JsonValue::Kind::Bool) throw ApiError{400, "chunking_enabled must be a boolean.", input, "invalid_type"};
            continue;
        }
        JsonValue translated = value;
        try {
            if (key == "seed") {
                const int64_t seed = std::get<int64_t>(option_from_json(SPEECH_OPT_SEED, value));
                if (seed < 0) throw std::invalid_argument("seed must be from 0 to 9223372036854775807");
                translated.text = std::to_string(seed % (int64_t(1) << 53));
            } else if (key == "max_seconds") {
                limit = std::get<double>(option_from_json(SPEECH_OPT_MAX_SECONDS, value));
                if (!std::isfinite(limit) || limit <= 0 || limit > 120) {
                    throw std::invalid_argument("max_seconds must be above 0 and at most 120");
                }
                continue;
            }
        } catch (const std::invalid_argument & e) {
            throw ApiError{400, e.what(), input, "invalid_value"};
        }
        // A null top-level option is absent, so remove it before inserting its translated value.
        for (auto it = json.members.begin(); it != json.members.end();) {
            if (it->first == mapped->second) it = json.members.erase(it);
            else ++it;
        }
        json.members.push_back({mapped->second, std::move(translated)});
    }
    for (auto it = json.members.begin(); it != json.members.end();) {
        if (it->first == "irodori") it = json.members.erase(it);
        else ++it;
    }
    if (const auto * option = json.member("irodori_compatible"); option && option->kind != JsonValue::Kind::Null) {
        throw ApiError{400, "irodori already selects compatibility processing; omit irodori_compatible.", "irodori_compatible", "invalid_value"};
    }
    return limit;
}

}  // namespace openai
