#include "request-options.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

#include "failure.h"
#include "json.h"
#include "library.h"
#include "transcript.h"

namespace {

/** What an option of `type` takes, for a message: "takes an integer". */
std::string takes(speech_type type) {
    switch (type) {
        case SPEECH_TYPE_STRING: return "takes a string";
        case SPEECH_TYPE_INT: return "takes an integer";
        case SPEECH_TYPE_FLOAT: return "takes a number";
        case SPEECH_TYPE_BOOL: return "takes true or false";
    }
    return "";
}

/** An integer of decimal digits with an optional minus sign, or false when `text` is not one or is beyond int64_t. */
bool whole_integer(const std::string & text, int64_t & out) {
    const size_t first = !text.empty() && text[0] == '-' ? 1 : 0;
    if (text.size() == first || text.find_first_not_of("0123456789", first) != std::string::npos) return false;
    errno = 0;
    out = std::strtoll(text.c_str(), nullptr, 10);
    return errno != ERANGE;
}

/** A finite decimal number written whole, or false. */
bool whole_number(const std::string & text, double & out) {
    if (text.empty() || text.find_first_not_of("0123456789+-.eE") != std::string::npos) return false;
    char * end = nullptr;
    out = std::strtod(text.c_str(), &end);
    return *end == '\0' && std::isfinite(out);
}

}  // namespace

std::vector<speech_option> vocabulary() {
    std::vector<speech_option> out;
    for (size_t i = 0; i < speech_option_count(); i++) out.push_back((speech_option) i);
    return out;
}

bool option_named(const std::string & name, speech_option & option) {
    for (speech_option o : vocabulary()) {
        if (name == speech_option_name(o)) {
            option = o;
            return true;
        }
    }
    return false;
}

std::string option_names() {
    const std::vector<speech_option> all = vocabulary();
    std::string out;
    for (size_t i = 0; i < all.size(); i++) out += (i == 0 ? "" : i + 1 == all.size() ? " and " : ", ") + std::string(speech_option_name(all[i]));
    return out;
}

std::string option_flag(speech_option option) {
    std::string flag = std::string("--") + speech_option_name(option);
    for (char & c : flag) {
        if (c == '_') c = '-';
    }
    return flag;
}

const char * option_metavar(speech_option option) {
    switch (speech_option_type(option)) {
        case SPEECH_TYPE_STRING: return "TEXT";
        case SPEECH_TYPE_INT: return "INTEGER";
        case SPEECH_TYPE_FLOAT: return "NUMBER";
        case SPEECH_TYPE_BOOL: return "";
    }
    return "";
}

OptionValue option_from_text(speech_option option, const std::string & text) {
    const speech_type type = speech_option_type(option);
    switch (type) {
        case SPEECH_TYPE_STRING: return text;
        case SPEECH_TYPE_INT: {
            int64_t v = 0;
            if (!whole_integer(text, v)) throw std::invalid_argument(option_flag(option) + " " + takes(type) + ", not \"" + text + "\"");
            return v;
        }
        case SPEECH_TYPE_FLOAT: {
            double v = 0;
            if (!whole_number(text, v)) throw std::invalid_argument(option_flag(option) + " " + takes(type) + ", not \"" + text + "\"");
            return v;
        }
        case SPEECH_TYPE_BOOL:
            if (text == "true" || text == "false") return text == "true";
            break;
    }
    throw std::invalid_argument(option_flag(option) + " is true alone and takes true or false after =, not \"" + text + "\"");
}

OptionValue option_from_json(speech_option option, const JsonValue & value) {
    const speech_type type = speech_option_type(option);
    const std::string name = speech_option_name(option);
    switch (type) {
        case SPEECH_TYPE_STRING:
            if (value.kind == JsonValue::Kind::String) return value.text;
            break;
        case SPEECH_TYPE_INT: {
            int64_t v = 0;
            if (value.is_integer() && whole_integer(value.text, v)) return v;
            if (value.is_integer()) {
                throw std::invalid_argument("\"" + name + "\" is " + value.text + ", beyond the integers speech.cpp reads; give a value the model takes");
            }
            if (value.kind == JsonValue::Kind::Number) {
                throw std::invalid_argument("\"" + name + "\" is " + value.text + "; " + name + " takes an integer, written without a fraction or an exponent");
            }
            break;
        }
        case SPEECH_TYPE_FLOAT: {
            double v = 0;
            if (value.kind == JsonValue::Kind::Number && whole_number(value.text, v)) return v;
            if (value.kind == JsonValue::Kind::Number) throw std::invalid_argument("\"" + name + "\" is " + value.text + ", beyond a finite number");
            break;
        }
        case SPEECH_TYPE_BOOL:
            if (value.kind == JsonValue::Kind::Bool) return value.boolean;
            break;
    }
    throw std::invalid_argument("\"" + name + "\" is " + json_excerpt(value) + "; " + name + " " + takes(type));
}

void apply_options(speech_request * request, const std::vector<RequestOption> & options) {
    for (const RequestOption & o : options) {
        switch (o.value.index()) {
            case 0: check(speech_request_set_string(request, o.option, std::get<std::string>(o.value).c_str())); break;
            case 1: check(speech_request_set_int(request, o.option, std::get<int64_t>(o.value))); break;
            case 2: check(speech_request_set_float(request, o.option, std::get<double>(o.value))); break;
            default: check(speech_request_set_bool(request, o.option, std::get<bool>(o.value) ? 1 : 0)); break;
        }
    }
}

std::string options_text(const std::vector<RequestOption> & options) {
    std::string out;
    for (const RequestOption & o : options) {
        if (!out.empty()) out += " ";
        out += option_flag(o.option);
        switch (o.value.index()) {
            case 0: out += " " + json_string(std::get<std::string>(o.value)); break;
            case 1: out += " " + std::to_string(std::get<int64_t>(o.value)); break;
            case 2: out += " " + json_number(std::get<double>(o.value)); break;
            default: out += std::get<bool>(o.value) ? "=true" : "=false"; break;
        }
    }
    return out;
}

std::string options_json(const std::vector<RequestOption> & options) {
    std::string out = "{";
    for (const RequestOption & o : options) {
        if (out.size() > 1) out += ",";
        out += json_string(speech_option_name(o.option)) + ":";
        switch (o.value.index()) {
            case 0: out += json_string(std::get<std::string>(o.value)); break;
            case 1: out += std::to_string(std::get<int64_t>(o.value)); break;
            case 2: out += json_number(std::get<double>(o.value)); break;
            default: out += std::get<bool>(o.value) ? "true" : "false"; break;
        }
    }
    return out + "}";
}

std::vector<RequestOption> defaults_for(speech_model * model, const std::vector<RequestOption> & defaults) {
    const ModelInfo info = model_info(model);
    std::vector<RequestOption> taken;
    for (const RequestOption & o : defaults) {
        if (speech_model_info_takes(info.get(), o.option)) taken.push_back(o);
    }
    apply_options(new_request(model).get(), taken);
    return taken;
}

std::vector<RequestOption> with_defaults(const std::vector<RequestOption> & defaults, const std::vector<RequestOption> & options) {
    std::vector<RequestOption> out;
    for (const RequestOption & d : defaults) {
        const bool set = std::any_of(options.begin(), options.end(), [&](const RequestOption & o) { return o.option == d.option; });
        if (!set) out.push_back(d);
    }
    out.insert(out.end(), options.begin(), options.end());
    return out;
}

bool timestamps_in_effect(const speech_model_info * info, const std::vector<RequestOption> & options) {
    for (const RequestOption & o : options) {
        if (o.option == SPEECH_OPT_TIMESTAMPS) return std::get<bool>(o.value);
    }
    if (!speech_model_info_takes(info, SPEECH_OPT_TIMESTAMPS) || !speech_model_info_option_has_default(info, SPEECH_OPT_TIMESTAMPS)) return false;
    int value = 0;
    check(speech_model_info_option_default_bool(info, SPEECH_OPT_TIMESTAMPS, &value));
    return value != 0;
}

std::string recognition_members(const speech_result * result, bool timestamps) {
    return recognition_members(transcript_of(result, timestamps), timestamps);
}
