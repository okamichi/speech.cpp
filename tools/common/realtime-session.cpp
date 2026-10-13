#include "realtime-session.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>

#include "base64.h"
#include "json.h"
#include "regions.h"

namespace realtime {

using openai::ApiError;

namespace {

/** The client events a session takes. */
const char * const kTaken = "session.update, input_audio_buffer.append, input_audio_buffer.commit and input_audio_buffer.clear";

ApiError refusal(const std::string & code, const std::string & param, const std::string & message) {
    return {400, message, param, code};
}

/** The value at `param` as an object, or an error that says it is not one. */
const JsonValue & object_at(const JsonValue & value, const std::string & param) {
    if (value.kind != JsonValue::Kind::Object) throw refusal("invalid_type", param, param + " is " + json_excerpt(value) + "; give an object.");
    return value;
}

/** The value at `param` as a string, or an error that says it is not one. */
const std::string & string_at(const JsonValue & value, const std::string & param) {
    if (value.kind != JsonValue::Kind::String) throw refusal("invalid_type", param, param + " is " + json_excerpt(value) + "; give a string.");
    return value.text;
}

/** Refuses a member of `object`, at `param` ("" for the event itself), that is not one of `taken`. */
void only(const JsonValue & object, const std::string & param, std::initializer_list<const char *> taken) {
    for (const auto & member : object.members) {
        const std::string & name = member.first;
        if (std::find_if(taken.begin(), taken.end(), [&](const char * t) { return name == t; }) != taken.end()) continue;
        std::string listed;
        for (const char * t : taken) listed += std::string(listed.empty() ? "" : ", ") + t;
        const std::string at = param.empty() ? name : param + "." + name;
        throw refusal("unknown_parameter", at, "Unknown parameter: '" + at + "'. speech.cpp's transcription session takes " + listed + " there.");
    }
}

/** A member that is present and not null. */
const JsonValue * given(const JsonValue & object, const char * name) {
    const JsonValue * value = object.member(name);
    return value && value->kind != JsonValue::Kind::Null ? value : nullptr;
}

const char * const kTurns = "session.audio.input.turn_detection";

/** Whether two lists of options set the same values in the same order. */
bool same(const std::vector<RequestOption> & a, const std::vector<RequestOption> & b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
                      [](const RequestOption & x, const RequestOption & y) { return x.option == y.option && x.value == y.value; });
}

/**
 * The detection options of a turn_detection that is not null, server_vad's members given; its other members and
 * semantic_vad are refused, since a transcription session creates no response for them to steer.
 */
std::vector<RequestOption> server_vad(const JsonValue & turns) {
    object_at(turns, kTurns);
    const std::string at = kTurns;
    const JsonValue * type = turns.member("type");
    if (!type) throw refusal("missing_required_parameter", at + ".type", "turn_detection needs its type; give \"server_vad\", or null for commits alone.");
    if (string_at(*type, at + ".type") == "semantic_vad") {
        throw refusal("unsupported_value", at + ".type",
                      "semantic_vad is not supported; speech.cpp detects turns with server_vad, by the regions where its detection model finds speech.");
    }
    if (type->text != "server_vad") throw refusal("invalid_value", at + ".type", "turn_detection's type is " + json_string(type->text) + "; give \"server_vad\".");
    for (const char * own : {"create_response", "interrupt_response", "idle_timeout_ms"}) {
        if (given(turns, own)) {
            throw refusal("unsupported_parameter", at + "." + own,
                          at + "." + own + " is not supported: a transcription session creates no response. Leave it out.");
        }
    }
    only(turns, at, {"type", "threshold", "prefix_padding_ms", "silence_duration_ms", "create_response", "interrupt_response", "idle_timeout_ms"});
    std::vector<RequestOption> out;
    for (const VadMember & m : kServerVad) {
        const JsonValue * value = given(turns, m.name);
        if (!value) continue;
        try {
            out.push_back({m.option, option_from_json(m.option, *value)});
        } catch (const std::invalid_argument &) {
            throw refusal("invalid_type", at + "." + m.name, at + "." + m.name + " is " + json_excerpt(*value) + "; it takes " +
                                                               (speech_option_type(m.option) == SPEECH_TYPE_INT ? "an integer." : "a number."));
        }
    }
    return out;
}

/** 16-bit little-endian PCM as the library takes samples, each scaled as a 16-bit WAVE file is read. */
std::vector<float> samples_of(const std::string & pcm) {
    std::vector<float> out(pcm.size() / 2);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = (float) (int16_t) ((uint16_t) (unsigned char) pcm[2 * i] | (uint16_t) (unsigned char) pcm[2 * i + 1] << 8) / 32768.0f;
    }
    return out;
}

}  // namespace

std::vector<RequestOption> Session::Config::options() const {
    std::vector<RequestOption> out;
    if (language) out.push_back({SPEECH_OPT_LANGUAGE, *language});
    if (prompt) out.push_back({SPEECH_OPT_PROMPT, *prompt});
    return out;
}

std::optional<utterances::Asked> Session::Config::asked() const {
    if (!transcribe) return std::nullopt;
    return utterances::Asked{model.value_or(""), options()};
}

std::unique_ptr<utterances::Reservation> Session::Reader::reserve(const utterances::Asked & asked) {
    return models_.reserve(asked.model);
}

std::optional<Transcript> Session::Reader::transcribe(utterances::Reservation & reservation, const utterances::Asked & asked,
                                                      std::vector<float> samples, int, Cancellation & cancellation, uint64_t utterance,
                                                      bool provisional) {
    // A reading of an utterance that is not yet announced, which the detection may still drop, takes no item.
    return models_.transcribe(reservation, std::move(samples), asked.options, cancellation, provisional ? "" : events_.item(utterance), provisional);
}

Session::Session(Models & models, std::function<void(const std::string &)> send, size_t limit, const std::string & model)
    : models_(models), events_(std::move(send)), reader_(models_, events_), limit_(limit),
      assembly_(reader_, kRate, [this](const utterances::Event & e) { events_.utterance(e); }) {
    if (!model.empty()) config_.model = model;
    assembly_.ask(config_.asked());
    watcher_ = std::thread([this] { watch(); });
}

Session::~Session() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
    }
    closing_changed_.notify_all();
    watcher_.join();
}

void Session::watch() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!closing_changed_.wait_for(lock, std::chrono::milliseconds(200), [&] { return closing_; })) tick();
}

void Session::open() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.emit("session.created", ",\"session\":" + session_json());
}

std::string Session::session_json() {
    std::string transcription = "null";
    if (config_.transcribe) {
        const std::string model = config_.model.value_or(models_.held());
        transcription = "{" + (model.empty() ? "" : "\"model\":" + json_string(model) + ",") +
                        "\"language\":" + (config_.language ? json_string(*config_.language) : "null") +
                        (config_.prompt ? ",\"prompt\":" + json_string(*config_.prompt) : "") + "}";
    }
    std::string turns = "null";
    if (config_.turns) {
        turns = "{\"type\":\"server_vad\"";
        for (const RequestOption & o : region_options(*config_.turns, models_.detection_defaults())) {
            const std::string member = server_vad_member(speech_option_name(o.option));
            if (member.empty()) continue;
            const OptionValue & v = o.value;
            turns += ",\"" + member + "\":" + (v.index() == 1 ? std::to_string(std::get<int64_t>(v)) : json_number(std::get<double>(v)));
        }
        turns += "}";
    }
    return "{\"type\":\"transcription\",\"id\":\"" + events_.session_id() + "\",\"object\":\"realtime.transcription_session\",\"audio\":{\"input\":{\"format\":"
           "{\"type\":\"audio/pcm\",\"rate\":" + std::to_string(kRate) + "},\"transcription\":" + transcription +
           ",\"noise_reduction\":null,\"turn_detection\":" + turns + "}},\"include\":null}";
}

void Session::tick() {
    if (!config_.turns) return;
    const utterances::Detection * detection = assembly_.detection();
    if (detection && models_.holds(*detection)) return;
    try {
        if (detection) assembly_.detect(nullptr);
        assembly_.detect(models_.detect(region_options(*config_.turns, models_.detection_defaults())));
        told_no_detection_ = false;
    } catch (...) {
        // While the page loads another detection model, turn detection waits for it; without one, or with one that
        // refuses the options, the session says so once and goes on with commits alone until one is held.
        ApiError e = error_of(std::current_exception());
        if (e.code != "model_loading" && !told_no_detection_) {
            if (e.param.empty()) e.param = kTurns;
            events_.error(e, "");
            told_no_detection_ = true;
        }
    }
}

void Session::receive(const std::string & message, bool text) {
    std::lock_guard<std::mutex> lock(mutex_);
    // The audio of an append goes to the detection model held now, not one the host has let go since the last tick.
    tick();
    std::string event_id;
    try {
        if (!text) {
            throw refusal("invalid_value", "", "The message is binary. speech.cpp takes each client event as JSON in a text message, the audio "
                                               "in input_audio_buffer.append as base64.");
        }
        JsonValue event;
        try {
            event = parse_json(message);
        } catch (const std::invalid_argument & e) {
            throw refusal("invalid_json", "", std::string("The message cannot be read as JSON: ") + e.what() + ". Send one JSON object per event.");
        }
        if (event.kind != JsonValue::Kind::Object) throw refusal("invalid_json", "", "The message is not a JSON object. Send one JSON object per event.");
        if (const JsonValue * id = given(event, "event_id")) event_id = string_at(*id, "event_id");
        const JsonValue * type = event.member("type");
        if (!type) throw refusal("missing_required_parameter", "type", "The event has no \"type\"; speech.cpp takes " + std::string(kTaken) + ".");
        const std::string & name = string_at(*type, "type");
        if (name == "session.update") {
            update(event);
        } else if (name == "input_audio_buffer.append") {
            append(event);
        } else if (name == "input_audio_buffer.commit") {
            only(event, "", {"type", "event_id"});
            commit();
        } else if (name == "input_audio_buffer.clear") {
            only(event, "", {"type", "event_id"});
            clear();
        } else {
            throw refusal("unsupported_value", "type", "The event " + json_string(name) + " is not supported; speech.cpp's transcription session takes " +
                                                           kTaken + ".");
        }
    } catch (const ApiError & e) {
        events_.error(e, event_id);
    } catch (const std::exception &) {
        events_.error(error_of(std::current_exception()), event_id);
    }
}

void Session::update(const JsonValue & event) {
    only(event, "", {"type", "event_id", "session"});
    const JsonValue * member = given(event, "session");
    if (!member) throw refusal("missing_required_parameter", "session", "session.update needs \"session\", the configuration to change.");
    const JsonValue & session = object_at(*member, "session");
    only(session, "session", {"type", "audio", "include"});
    const JsonValue * type = session.member("type");
    if (!type) throw refusal("missing_required_parameter", "session.type", "session.type is required; give \"transcription\".");
    if (string_at(*type, "session.type") == "realtime") {
        throw refusal("unsupported_value", "session.type", "A conversation session (session.type \"realtime\") is not supported; speech.cpp's "
                                                           "Realtime sessions transcribe. Give session.type \"transcription\".");
    }
    if (type->text != "transcription") {
        throw refusal("invalid_value", "session.type", "session.type is " + json_string(type->text) + "; give \"transcription\".");
    }
    if (const JsonValue * include = given(session, "include")) {
        if (include->kind != JsonValue::Kind::Array) throw refusal("invalid_type", "session.include", "session.include is not an array.");
        for (const JsonValue & item : include->items) {
            const std::string & what = string_at(item, "session.include");
            throw refusal(what == "item.input_audio_transcription.logprobs" ? "unsupported_value" : "invalid_value", "session.include",
                          "session.include holds " + json_string(what) + ", which speech.cpp does not give; leave it empty.");
        }
    }
    Config next = config_;
    const JsonValue * audio = given(session, "audio");
    const JsonValue * input = audio ? given(object_at(*audio, "session.audio"), "input") : nullptr;
    if (audio) only(*audio, "session.audio", {"input"});
    if (input) {
        object_at(*input, "session.audio.input");
        only(*input, "session.audio.input", {"format", "transcription", "noise_reduction", "turn_detection"});
        if (const JsonValue * format = given(*input, "format")) {
            object_at(*format, "session.audio.input.format");
            only(*format, "session.audio.input.format", {"type", "rate"});
            const JsonValue * kind = format->member("type");
            if (!kind) throw refusal("missing_required_parameter", "session.audio.input.format.type", "The format needs its type; give \"audio/pcm\".");
            if (string_at(*kind, "session.audio.input.format.type") != "audio/pcm") {
                throw refusal(kind->text == "audio/pcmu" || kind->text == "audio/pcma" ? "unsupported_value" : "invalid_value",
                              "session.audio.input.format.type",
                              "The audio format " + json_string(kind->text) + " is not supported; speech.cpp takes \"audio/pcm\", 16-bit PCM at 24000 Hz.");
            }
            if (const JsonValue * rate = given(*format, "rate"); rate && !(rate->is_integer() && rate->text == std::to_string(kRate))) {
                throw refusal("invalid_value", "session.audio.input.format.rate",
                              "audio/pcm is " + std::to_string(kRate) + " Hz, not " + json_excerpt(*rate) + "; give rate 24000 or leave it out.");
            }
        }
        if (const JsonValue * transcription = input->member("transcription")) {
            next.transcribe = transcription->kind != JsonValue::Kind::Null;
            if (next.transcribe) {
                object_at(*transcription, kTranscription);
                const std::string at = kTranscription;
                for (const char * own : {"keywords", "delay"}) {
                    if (given(*transcription, own)) {
                        throw refusal("unsupported_parameter", at + "." + own,
                                      at + "." + own + " is not supported; speech.cpp takes the model, the language or languages, and the prompt.");
                    }
                }
                only(*transcription, at, {"model", "language", "languages", "prompt", "keywords", "delay"});
                // A member given sets its value, null taking it away, and one left out keeps it, as OpenAI's
                // session.update changes only the members it holds.
                const auto text = [&](const char * name, std::optional<std::string> & into) {
                    const JsonValue * value = transcription->member(name);
                    if (value) into = value->kind == JsonValue::Kind::Null ? std::nullopt : std::optional<std::string>(string_at(*value, at + "." + name));
                };
                text("model", next.model);
                text("language", next.language);
                text("prompt", next.prompt);
                if (const JsonValue * languages = transcription->member("languages")) {
                    if (transcription->member("language")) throw refusal("invalid_value", at + ".languages", "Give the language or the languages, not both.");
                    next.language.reset();
                    if (languages->kind != JsonValue::Kind::Null) {
                        if (languages->kind != JsonValue::Kind::Array || languages->items.empty()) {
                            throw refusal("invalid_type", at + ".languages", at + ".languages is " + json_excerpt(*languages) + "; give an array of one tag.");
                        }
                        if (languages->items.size() > 1) {
                            throw refusal("unsupported_value", at + ".languages",
                                          "A recognition takes one language or leaves it to the model; give one language, or none for the model to tell.");
                        }
                        next.language = string_at(languages->items[0], at + ".languages");
                    }
                }
            }
        }
        if (given(*input, "noise_reduction")) {
            throw refusal("unsupported_parameter", "session.audio.input.noise_reduction",
                          "Noise reduction is not supported; give session.audio.input.noise_reduction null or leave it out.");
        }
        if (const JsonValue * turns = input->member("turn_detection")) {
            next.turns = turns->kind == JsonValue::Kind::Null ? std::nullopt : std::optional<std::vector<RequestOption>>(server_vad(*turns));
        }
    }
    if (next.transcribe) {
        try {
            models_.check(next.model.value_or(""), next.options());
        } catch (ApiError & e) {
            if (e.param.empty()) e.param = std::string(kTranscription) + ".model";
            throw;
        }
    }
    // A detection starts on the options given, which the library checks, before anything changes; the same options keep
    // the detection under way and the utterance it is cutting.
    std::unique_ptr<utterances::Detection> detection;
    const bool restart = next.turns && (!config_.turns || !same(*next.turns, *config_.turns) || !assembly_.detection());
    if (restart) {
        try {
            detection = models_.detect(region_options(*next.turns, models_.detection_defaults()));
        } catch (ApiError & e) {
            e.param = kTurns;
            throw;
        } catch (const Failure & failure) {
            ApiError e = openai::library_error(failure);
            e.param = std::string(kTurns) + "." + server_vad_member(failure.option());
            throw e;
        }
    }
    const bool asked_otherwise = next.transcribe != config_.transcribe || next.model != config_.model || next.language != config_.language ||
                                 next.prompt != config_.prompt;
    const bool manual = config_.turns && !next.turns;
    config_ = next;
    if (asked_otherwise) assembly_.ask(config_.asked());
    if (restart) {
        assembly_.detect(std::move(detection));
        told_no_detection_ = false;
    }
    if (manual) assembly_.manual();
    events_.emit("session.updated", ",\"session\":" + session_json());
}

void Session::append(const JsonValue & event) {
    only(event, "", {"type", "event_id", "audio"});
    const JsonValue * audio = event.member("audio");
    if (!audio) throw refusal("missing_required_parameter", "audio", "input_audio_buffer.append needs \"audio\", base64 of 16-bit PCM at 24000 Hz.");
    std::string pcm;
    try {
        pcm = base64_decode(string_at(*audio, "audio"));
    } catch (const std::invalid_argument & e) {
        throw refusal("invalid_value", "audio", std::string("The audio is not base64: ") + e.what() + ".");
    }
    if (pcm.size() % 2) {
        throw refusal("invalid_value", "audio", "The audio has " + std::to_string(pcm.size()) + " bytes, and 16-bit PCM has two for each sample.");
    }
    const size_t held = assembly_.held() * 2;
    if (held + pcm.size() > limit_) {
        char message[320];
        std::snprintf(message, sizeof message,
                      "The session holds %.1f s of audio not yet transcribed, and this append would take it past the %.1f s this server "
                      "holds; commit shorter utterances, wait for their transcriptions, or clear the buffer.",
                      (double) held / 2 / kRate, (double) limit_ / 2 / kRate);
        throw refusal("input_audio_buffer_full", "audio", message);
    }
    const std::vector<float> samples = samples_of(pcm);
    assembly_.push(samples.data(), samples.size());
}

void Session::commit() {
    if (!assembly_.commit()) {
        throw refusal("input_audio_buffer_commit_empty", "", "The input audio buffer is empty; append audio before committing it.");
    }
}

void Session::clear() {
    assembly_.clear();
    events_.emit("input_audio_buffer.cleared", "");
}

}  // namespace realtime
