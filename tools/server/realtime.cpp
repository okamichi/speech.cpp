#include "realtime.h"

#include <chrono>
#include <cstdio>
#include <memory>

#include "jobs.h"
#include "json.h"
#include "realtime-session.h"

namespace server {

namespace {

using openai::ApiError;

std::string name_of(const Served & served) {
    return speech_model_info_name(served.info().get());
}

/**
 * A recognition's place on the recognition model held when it was accepted: the model, kept until the recognition has
 * run, and its turn, which is passed or given up as the place goes, the model let go after it.
 */
struct Held : utterances::Reservation {
    explicit Held(std::shared_ptr<Served> model) : on(std::move(model)), turn(on->turns()) {}
    std::shared_ptr<Served> on;
    Turn turn;
    Clock::time_point accepted = Clock::now();
};

/**
 * The models a session works with: the recognition model the server holds when a recognition is accepted, and the
 * detection model it holds, which a session's detection keeps until the session lets it go.
 */
class HeldModels : public realtime::Models {
public:
    explicit HeldModels(ServedModels & models) : models_(models) {}

    std::string held() override {
        const auto served = models_.of(SPEECH_TASK_RECOGNITION);
        return served ? name_of(*served) : "";
    }

    void check(const std::string & model, const std::vector<RequestOption> & options) override {
        const auto on = served(model);
        apply_options(new_request(on->get()).get(), on->with_defaults(options));
    }

    std::unique_ptr<utterances::Reservation> reserve(const std::string & model) override {
        return std::make_unique<Held>(served(model));
    }

    std::optional<Transcript> transcribe(utterances::Reservation & reservation, std::vector<float> samples,
                                         const std::vector<RequestOption> & options, Cancellation & cancellation, const std::string & item,
                                         bool provisional) override {
        Held & held = static_cast<Held &>(reservation);
        const Request request = new_request(held.on->get());
        const size_t n = samples.size();
        ::check(speech_request_set_audio(request.get(), samples.data(), n, realtime::kRate));
        // The request holds a copy of its own, so that a reading, whose samples are a copy of the buffer, adds nothing to
        // the audio the session holds while it waits and runs.
        samples = std::vector<float>();
        apply_options(request.get(), held.on->with_defaults(options));
        held.turn.wait();
        const Clock::time_point started = Clock::now();
        // The readings of an utterance under way, several a second, are logged only where one stops or fails.
        const std::string name = provisional ? "a reading" : item;
        speech_status status = SPEECH_CANCELLED;
        try {
            status = ::check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); }));
        } catch (const Failure & e) {
            log(name, n, held.accepted, started, ", failed: " + e.code() + ": " + e.what());
            throw;
        }
        if (status == SPEECH_CANCELLED) {
            log(name, n, held.accepted, started, provisional ? ", stopped: its utterance ended" : ", stopped: the session ended");
            return std::nullopt;
        }
        Transcript t = transcript_of(speech_request_result(request.get()), false);
        if (!provisional) log(name, n, held.accepted, started, ", " + std::to_string(t.text.size()) + " bytes of text");
        return t;
    }

    std::unique_ptr<utterances::Detection> detect(const std::vector<RequestOption> & options) override {
        const auto on = models_.of(SPEECH_TASK_DETECTION);
        if (!on) {
            const std::string coming = models_.replacing(SPEECH_TASK_DETECTION);
            if (!coming.empty()) {
                throw ApiError{503, "The speech detection model, which server_vad needs, is being loaded (" + coming + "); turn detection begins once "
                                    "it is in place.", "", "model_loading"};
            }
            throw ApiError{400, "server_vad finds the turns where a speech detection model finds that someone speaks, and this server holds none. "
                                "Give speech serve one beside the recognition model, as in speech serve " + held() + " silero-vad, or give "
                                "turn_detection null and commit each utterance.", "", "unsupported_value"};
        }
        return std::make_unique<utterances::Detection>(on->get(), on, options, realtime::kRate);
    }

    std::vector<RequestOption> detection_defaults() override {
        const auto on = models_.of(SPEECH_TASK_DETECTION);
        return on ? on->defaults() : std::vector<RequestOption>();
    }

    bool holds(const utterances::Detection & detection) override {
        return models_.of(SPEECH_TASK_DETECTION).get() == detection.model();
    }

private:
    static void log(const std::string & item, size_t samples, Clock::time_point arrived, Clock::time_point started, const std::string & outcome) {
        std::fprintf(stderr, "realtime %s: %.2f s of audio: waited %.3f s%s in %.3f s\n", item.c_str(), (double) samples / realtime::kRate,
                     seconds_between(arrived, started), outcome.c_str(), seconds_between(started, Clock::now()));
    }

    /** The recognition model held, named `model` unless that is "". */
    std::shared_ptr<Served> served(const std::string & model) const {
        const auto on = models_.of(SPEECH_TASK_RECOGNITION);
        if (!on) {
            const std::string coming = models_.replacing(SPEECH_TASK_RECOGNITION);
            if (!coming.empty()) {
                throw ApiError{503, "The speech recognition model is being loaded (" + coming + "); commit the audio again once it is in place.", "",
                               "model_loading"};
            }
            throw ApiError{404, "This server holds no speech recognition model to transcribe with; give speech serve one.", "", "model_not_found"};
        }
        if (!model.empty() && model != name_of(*on)) {
            throw ApiError{404, "The model " + json_string(model) + " is not served here; this server transcribes with " + json_string(name_of(*on)) +
                                    ". Name it, or leave the model out.",
                           "", "model_not_found"};
        }
        return on;
    }

    ServedModels & models_;
};

}  // namespace

std::optional<ApiError> Realtime::refusal(const httplib::Request & req) const {
    if (access_.loopback()) {
        if (auto refused = access_.host_refusal(req)) return refused;
    }
    std::string model;
    for (const auto & [name, value] : req.params) {
        if (name != "model") {
            return ApiError{400, "Unknown parameter in the address: " + name + ". speech.cpp's /v1/realtime takes model alone, in a query.", name,
                            "unknown_parameter"};
        }
        model = value;
    }
    const auto on = models_.of(SPEECH_TASK_RECOGNITION);
    if (!on) {
        const std::string coming = models_.replacing(SPEECH_TASK_RECOGNITION);
        if (!coming.empty()) {
            return ApiError{503, "The speech recognition model is being loaded (" + coming + "); connect again once it is in place.", "",
                            "model_loading"};
        }
        return ApiError{404, "This server holds no speech recognition model, which /v1/realtime transcribes with. Give speech serve one.", "", ""};
    }
    if (!model.empty() && model != name_of(*on)) {
        return ApiError{404, "The model " + json_string(model) + " is not served here; this server transcribes with " + json_string(name_of(*on)) +
                                 ". Name it in \"model\" or leave \"model\" out.",
                        "model", "model_not_found"};
    }
    return std::nullopt;
}

void Realtime::route(httplib::Server & http) {
    http.WebSocket("/v1/realtime", [this](const httplib::Request & req, httplib::ws::WebSocket & ws) {
        HeldModels held(models_);
        realtime::Session session(held, [&ws](const std::string & event) { ws.send(event); }, limit_, req.get_param_value("model"));
        std::fprintf(stderr, "%s realtime session opened\n", req.remote_addr.c_str());
        session.open();
        std::string message;
        for (;;) {
            const httplib::ws::ReadResult got = ws.read(message);
            if (got == httplib::ws::Fail) break;
            if (got == httplib::ws::Timeout) continue;
            session.receive(message, got == httplib::ws::Text);
        }
        std::fprintf(stderr, "%s realtime session closed\n", req.remote_addr.c_str());
    });
}

}  // namespace server
