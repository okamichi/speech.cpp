#include "openai-api.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>

#include "httplib.h"

#include "base64.h"
#include "error.h"
#include "json-reader.h"
#include "irodori-request.h"
#include "json.h"
#include "regions.h"
#include "wav.h"

namespace openai {

namespace {

ApiError unknown_member(const std::string & name, const std::string & takes) {
    return {400, "Unrecognized request argument supplied: " + name + ". speech.cpp takes " + takes + ".", name, "unknown_parameter"};
}

/**
 * The detection options of a form's chunking_strategy, or nothing without one: "auto", or the members of server_vad as
 * OpenAI's SDKs send an object in a form, chunking_strategy[type] "server_vad" and chunking_strategy[threshold] and the
 * others, which take the defaults of transcription by regions where they are left out.
 */
std::optional<std::vector<RequestOption>> chunking_strategy(const std::function<const FormPart *(const std::string &)> & field) {
    const FormPart * strategy = field("chunking_strategy"), * type = field("chunking_strategy[type]");
    std::vector<RequestOption> given;
    std::string members;
    for (const VadMember & m : kServerVad) {
        const std::string name = std::string("chunking_strategy[") + m.name + "]";
        members += (members.empty() ? "" : ", ") + name;
        const FormPart * part = field(name);
        if (!part) continue;
        try {
            given.push_back({m.option, option_from_text(m.option, part->content)});
        } catch (const std::invalid_argument &) {
            throw ApiError{400, name + " is " + json_string(part->content) + "; it takes " +
                                    (speech_option_type(m.option) == SPEECH_TYPE_INT ? "an integer" : "a number") + ".",
                           name, "invalid_type"};
        }
    }
    const std::string how = "give chunking_strategy \"auto\", or chunking_strategy[type] \"server_vad\" with any of " + members;
    if (strategy && (type || !given.empty())) {
        throw ApiError{400, "chunking_strategy is given both as a value and as members; " + how + ".", "chunking_strategy", "invalid_value"};
    }
    if (strategy) {
        if (strategy->content != "auto") {
            throw ApiError{400, "The chunking_strategy " + json_string(strategy->content) + " is not supported; " + how + ".", "chunking_strategy",
                           "unsupported_value"};
        }
        return region_options({});
    }
    if (type) {
        if (type->content != "server_vad") {
            throw ApiError{400, "The chunking_strategy type " + json_string(type->content) + " is not supported; " + how + ".",
                           "chunking_strategy[type]", "unsupported_value"};
        }
        return region_options(given);
    }
    if (!given.empty()) throw ApiError{400, "The members of chunking_strategy need its type; " + how + ".", "chunking_strategy[type]", "missing_required_parameter"};
    return std::nullopt;
}

ApiError not_served(const std::string & asked, const std::string & model_name) {
    return {404, "The model " + json_string(asked) + " is not served here; this server serves " + json_string(model_name) +
                     ". Name it in \"model\" or leave \"model\" out.",
            "model", "model_not_found"};
}

}  // namespace

void send_error(httplib::Response & res, const ApiError & e) {
    res.status = e.status;
    res.set_content(error_json(e), "application/json");
}

SpeechRequest read_speech_request(const std::string & body, const std::string & model_name, bool compatible) {
    JsonValue json;
    try {
        json = parse_json(body);
    } catch (const std::invalid_argument & e) {
        throw ApiError{400, std::string("The body cannot be read as JSON: ") + e.what() + ". Send one JSON object.", "", ""};
    }
    if (json.kind != JsonValue::Kind::Object) throw ApiError{400, "The body is not a JSON object. Send one JSON object.", "", ""};
    const std::vector<std::string> own = {"model", "input", "response_format", "stream_format"};
    SpeechRequest r;
    if (compatible) r.max_seconds = irodori_request(json, model_name);
    if (r.max_seconds) r.options.push_back({SPEECH_OPT_IRODORI_COMPATIBLE, true});
    std::vector<std::pair<std::string, const JsonValue *>> text;
    for (const auto & [name, value] : json.members) {
        if (value.kind == JsonValue::Kind::Null) continue;
        speech_option option;
        if (std::find(own.begin(), own.end(), name) != own.end()) {
            if (value.kind != JsonValue::Kind::String) {
                throw ApiError{400, "\"" + name + "\" is " + json_excerpt(value) + "; give it as a string.", name, "invalid_type"};
            }
            text.push_back({name, &value});
        } else if (option_named(name, option)) {
            try {
                r.options.push_back({option, option_from_json(option, value)});
            } catch (const std::invalid_argument & e) {
                throw ApiError{400, std::string(e.what()) + ".", name, "invalid_type"};
            }
        } else {
            throw unknown_member(name, "model, input, response_format, stream_format and the options " + option_names());
        }
    }
    const auto member = [&](const std::string & name) -> const std::string * {
        for (const auto & [key, value] : text) {
            if (key == name) return &value->text;
        }
        return nullptr;
    };
    if (const std::string * model = member("model"); model && *model != model_name) throw not_served(*model, model_name);
    const std::string * input = member("input");
    if (!input) throw ApiError{400, "The request has no \"input\"; it is required.", "input", "missing_required_parameter"};
    r.input = *input;
    if (const std::string * format = member("response_format")) r.format = *format;
    if (r.format != "wav" && r.format != "pcm") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"wav\" or \"pcm\".",
                       "response_format", "unsupported_value"};
    }
    const std::string stream_format = member("stream_format") ? *member("stream_format") : "audio";
    if (stream_format != "audio" && stream_format != "sse") {
        throw ApiError{400, "The stream_format " + json_string(stream_format) + " is not supported; give \"audio\" or \"sse\".", "stream_format",
                       "unsupported_value"};
    }
    r.sse = stream_format == "sse";
    if (r.sse && r.format != "pcm") {
        throw ApiError{400, "stream_format \"sse\" carries \"pcm\" audio; give \"response_format\": \"pcm\" with it.", "response_format",
                       "unsupported_value"};
    }
    return r;
}

TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts, const std::string & model_name,
                                                bool gives_times) {
    if (!multipart) {
        throw ApiError{400, "The body is not multipart/form-data. Send the audio as the form's \"file\", as OpenAI's create transcription takes it.",
                       "", ""};
    }
    std::vector<std::string> known = {"file", "model", "language", "prompt", "response_format", "decoding", "chunking_strategy", "chunking_strategy[type]"};
    for (const VadMember & m : kServerVad) known.push_back(std::string("chunking_strategy[") + m.name + "]");
    std::string listed;
    for (const std::string & name : known) listed += name + ", ";
    std::map<std::string, const FormPart *> given;
    std::vector<std::string> granularities;
    for (const FormPart & part : parts) {
        if (part.name == "timestamp_granularities[]") {
            granularities.push_back(part.content);
            continue;
        }
        if (std::find(known.begin(), known.end(), part.name) == known.end()) throw unknown_member(part.name, listed + "and timestamp_granularities[]");
        if (given.count(part.name)) throw ApiError{400, "\"" + part.name + "\" is given twice; give it once.", part.name, "invalid_value"};
        given[part.name] = &part;
    }
    const auto field = [&](const std::string & name) -> const FormPart * {
        const auto found = given.find(name);
        return found == given.end() ? nullptr : found->second;
    };
    const FormPart * file = field("file"), * model = field("model"), * language = field("language"), * prompt = field("prompt"),
                   * format = field("response_format"), * decoding = field("decoding");
    if (model && model->content != model_name) throw not_served(model->content, model_name);
    if (!file) throw ApiError{400, "The request has no \"file\"; it is required.", "file", "missing_required_parameter"};
    if (!file->file) throw ApiError{400, "\"file\" is a field; send it as a file, with a filename.", "file", "invalid_type"};
    TranscriptionRequest r;
    if (format) r.format = format->content;
    if (r.format != "json" && r.format != "text" && r.format != "verbose_json") {
        throw ApiError{400, "The response_format " + json_string(r.format) + " is not supported; speech.cpp answers with \"json\", \"text\" or "
                       "\"verbose_json\", since it gives neither subtitles nor speakers.",
                       "response_format", "unsupported_value"};
    }
    for (const std::string & g : granularities) {
        if (g != "segment") {
            throw ApiError{400, "The timestamp granularity " + json_string(g) + " is not supported; speech.cpp gives \"segment\".",
                           "timestamp_granularities[]", "unsupported_value"};
        }
        if (r.format != "verbose_json") {
            throw ApiError{400, "timestamp_granularities[] needs \"response_format\": \"verbose_json\", which carries the segments.",
                           "timestamp_granularities[]", "invalid_value"};
        }
    }
    r.regions = chunking_strategy(field);
    if (language) r.options.push_back({SPEECH_OPT_LANGUAGE, language->content});
    if (prompt) r.options.push_back({SPEECH_OPT_PROMPT, prompt->content});
    if (decoding) r.options.push_back({SPEECH_OPT_DECODING, decoding->content});
    // OpenAI's verbose_json requires the language, the duration and the text alone, and its timestamp granularity is
    // segment unless the request names one: a model that gives no times answers it without segments, and refuses a
    // granularity the request names.
    r.timestamps = r.format == "verbose_json" && (gives_times || !granularities.empty());
    if (r.timestamps) r.options.push_back({SPEECH_OPT_TIMESTAMPS, true});
    const std::string & bytes = file->content;
    if (bytes.size() < 12 || bytes.compare(0, 4, "RIFF") != 0 || bytes.compare(8, 4, "WAVE") != 0) {
        throw ApiError{400, "The file " + json_string(file->filename) + " is not a WAV file. speech.cpp reads WAV alone (16-, 24- or 32-bit PCM or "
                       "32-bit float); convert the audio first, for example with ffmpeg -i in.mp3 out.wav.",
                       "file", "unsupported_value"};
    }
    try {
        const Wav wav = parse_wav(bytes, file->filename);
        r.samples = wav.mono();
        r.sample_rate = wav.sample_rate;
    } catch (const Error & e) {
        throw ApiError{400, std::string("The file cannot be read: ") + e.what() + ".", "file", "invalid_value"};
    }
    return r;
}

std::string sse_delta(const std::string & pcm) {
    return "data: {\"type\":\"speech.audio.delta\",\"audio\":\"" + base64((const uint8_t *) pcm.data(), pcm.size()) + "\"}\n\n";
}

std::string sse_done(int64_t seed, uint64_t samples, const char * stop) {
    return "data: {\"type\":\"speech.audio.done\",\"seed\":" + std::to_string(seed) + ",\"samples\":" + std::to_string(samples) +
           ",\"stop\":\"" + stop + "\"}\n\n";
}

std::string sse_error(const ApiError & e) {
    return "data: {\"type\":\"error\",\"error\":" + error_object(e) + "}\n\n";
}

std::string transcription_json(const std::string & text) {
    return "{\"text\":" + json_string(text) + "}";
}

std::string transcription_verbose_json(const Transcript & transcript, double duration, bool timestamps) {
    std::string language;
    for (size_t i = 0; i < transcript.languages.size(); i++) language += (i ? "," : "") + transcript.languages[i];
    std::string out = "{\"task\":\"transcribe\"" + (language.empty() ? "" : ",\"language\":" + json_string(language)) +
                      ",\"duration\":" + json_number(duration) + ",\"text\":" + json_string(transcript.text);
    if (!timestamps) return out + "}";
    out += ",\"segments\":[";
    for (size_t i = 0; i < transcript.segments.size(); i++) {
        const Timed & s = transcript.segments[i];
        out += std::string(i ? "," : "") + "{\"id\":" + std::to_string(i) + ",\"start\":" + json_number(s.start) + ",\"end\":" + json_number(s.end) +
               ",\"text\":" + json_string(s.text) + "}";
    }
    return out + "]}";
}

}  // namespace openai
