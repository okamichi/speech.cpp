#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#include "commands.h"
#include "fetch.h"
#include "jobs.h"
#include "json.h"
#include "openai-api.h"

#include "access.h"
#include "audio-speech.h"
#include "page.h"
#include "realtime.h"
#include "served-models.h"

// speech serve: serves a model of each task over HTTP with OpenAI's audio API, so that a program that speaks HTTP (a web
// app, Python with requests, curl) can use speech.cpp without starting the worker: the synthesis model answers
// POST /v1/audio/speech and the recognition model POST /v1/audio/transcriptions, which with chunking_strategy recognizes
// the regions where the detection model finds that someone speaks; GET /v1/models gives the models with their
// information, and GET /health says the server is up, which it is once the models given are loaded. A WebSocket at
// /v1/realtime carries OpenAI's Realtime transcription session (realtime.h). A failure of
// the library becomes OpenAI's error object by its category alone (openai::library_error()). A model serves one request
// at a time in the order they arrive, and a client that goes away while it waits or while its request runs cancels it.
// On a loopback address the server also serves its page (page.h), which replaces the model of a task by one of the
// catalog.

using namespace server;
using openai::ApiError;
using openai::send_error;

namespace {

/** What a task's model serves. */
const char * use_of(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "/v1/audio/speech";
        case SPEECH_TASK_RECOGNITION: return "/v1/audio/transcriptions";
        case SPEECH_TASK_DETECTION: return "chunking_strategy of /v1/audio/transcriptions and server_vad of /v1/realtime";
    }
    throw std::logic_error("a task speech.h does not have");
}

/** The task of the other of the two endpoints that take a model. */
speech_task other_task(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return SPEECH_TASK_RECOGNITION;
        case SPEECH_TASK_RECOGNITION: return SPEECH_TASK_SYNTHESIS;
        case SPEECH_TASK_DETECTION: break;
    }
    throw std::logic_error("no endpoint takes a model of speech detection");
}

/** OpenAI's model object, with the release of speech.cpp and the model's information. */
std::string model_json(const Served & served) {
    const auto info = served.info();
    return "{\"id\":" + json_string(speech_model_info_name(info.get())) + ",\"object\":\"model\",\"created\":" +
           std::to_string(served.created()) + ",\"owned_by\":\"speech.cpp\",\"version\":" + json_string(speech_version()) +
           ",\"speech\":" + speech_model_info_json(info.get()) + "}";
}

class Server {
public:
    Server(ServedModels & models, const Access & access, const Realtime & realtime) : models_(models), access_(access), realtime_(realtime) {}

    void route(httplib::Server & http) {
        http.set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) { return admit(req, res); });
        http.set_error_handler([](const httplib::Request & req, httplib::Response & res) {
            if (res.body.empty()) {
                send_error(res, {res.status, res.status == 404 ? "Invalid URL (" + req.method + " " + req.path + ")."
                                                               : std::string(httplib::status_message(res.status)) + ".", "", ""});
            }
        });
        http.set_logger([](const httplib::Request & req, const httplib::Response & res) {
            std::fprintf(stderr, "%s %s %s %d\n", req.remote_addr.c_str(), req.method.c_str(), req.path.c_str(), res.status);
        });
        http.Get("/health", [this](const httplib::Request &, httplib::Response & res) { res.set_content(models_.health(), "application/json"); });
        if (models_.compatible()) route_voices(http, models_);
        http.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            std::string data;
            for (const auto & served : models_.all()) data += (data.empty() ? "" : ",") + model_json(*served);
            res.set_content("{\"object\":\"list\",\"data\":[" + data + "]}", "application/json");
        });
        http.Get("/v1/models/(.+)", [this](const httplib::Request & req, httplib::Response & res) {
            for (const auto & served : models_.all()) {
                if (req.matches[1] == speech_model_info_name(served->info().get())) {
                    res.set_content(model_json(*served), "application/json");
                    return;
                }
            }
            send_error(res, {404, "The model " + json_string(req.matches[1]) + " does not exist.", "model", "model_not_found"});
        });
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_SYNTHESIS, req, res)) answer_speech(std::move(served), req, res, models_.compatible());
        });
        http.Post("/v1/audio/transcriptions", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_RECOGNITION, req, res)) transcription(std::move(served), req, res);
        });
    }

private:
    ServedModels & models_;
    const Access & access_;
    const Realtime & realtime_;

    /**
     * The model of the task the endpoint asks for; otherwise answers a 404, as for a path the server does not have,
     * or a 503 while the page loads one in its place.
     */
    std::shared_ptr<Served> held(speech_task wanted, const httplib::Request & req, httplib::Response & res) const {
        if (auto served = models_.of(wanted)) return served;
        const std::string task = std::string("speech ") + task_name(wanted);
        if (const std::string coming = models_.replacing(wanted); !coming.empty()) {
            res.set_header("Retry-After", "5");
            send_error(res, {503, "The " + task + " model is being loaded (" + coming + "); send the request again once it is in place.", "",
                             "model_loading"});
            return nullptr;
        }
        std::string message = "This server holds no " + task + " model, which " + req.path + " is for";
        const speech_task other = other_task(wanted);
        if (const auto served = models_.of(other)) {
            message += "; it holds " + std::string(speech_model_info_name(served->info().get())) + ", a speech " + task_name(other) +
                       " model, which " + use_of(other) + " is for";
        }
        send_error(res, {404, message + ". Give speech serve a " + task + " model" + (access_.loopback() ? ", or pick one on its page." : "."), "", ""});
        return nullptr;
    }

    /**
     * Refuses a request from an origin that may not call the server, adds the CORS headers for an origin --cors-origin
     * allows, and answers a preflight.
     */
    httplib::Server::HandlerResponse admit(const httplib::Request & req, httplib::Response & res) const {
        if (const auto refusal = access_.origin_refusal(req)) {
            send_error(res, *refusal);
            return httplib::Server::HandlerResponse::Handled;
        }
        // A WebSocket is answered by its handler only once it is upgraded, so a refusal of one is an HTTP answer here.
        if (req.path == "/v1/realtime") {
            if (const auto refusal = realtime_.refusal(req)) {
                send_error(res, *refusal);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        const std::string origin = req.get_header_value("Origin");
        const bool allowed = access_.cross_origin_allowed(origin);
        if (allowed) {
            res.set_header("Access-Control-Allow-Origin", access_.any_origin() ? "*" : origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Speech-Seed, X-Speech-Stop, X-Speech-Instructions-Truncated");
            if (!access_.any_origin()) res.set_header("Vary", "Origin");
        }
        if (req.method != "OPTIONS") return httplib::Server::HandlerResponse::Unhandled;
        if (allowed) {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            const std::string headers = req.get_header_value("Access-Control-Request-Headers");
            res.set_header("Access-Control-Allow-Headers", headers.empty() ? "Content-Type, Authorization" : headers);
            res.set_header("Access-Control-Max-Age", "86400");
        }
        res.status = 204;
        return httplib::Server::HandlerResponse::Handled;
    }

    /**
     * Recognizes the form's WAV file, or with chunking_strategy each region of it where the detection model finds that
     * someone speaks, and answers with its text once it is done, as json, text or verbose_json, which also carries the
     * language the model heard, and with why the recognition ended in X-Speech-Stop.
     */
    void transcription(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
        const auto info = served->info();
        openai::TranscriptionRequest asked;
        try {
            std::vector<openai::FormPart> parts;
            for (const auto & [field_name, field] : req.form.fields) parts.push_back({field_name, field.content, "", false});
            for (const auto & [file_name, file] : req.form.files) parts.push_back({file_name, file.content, file.filename, true});
            asked = openai::read_transcription_request(req.is_multipart_form_data(), parts, speech_model_info_name(info.get()),
                                                       speech_model_info_takes(info.get(), SPEECH_OPT_TIMESTAMPS));
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        auto job = std::make_shared<TranscriptionJob>();
        job->sample_rate = asked.sample_rate;
        job->duration = (double) asked.samples.size() / asked.sample_rate;
        job->served = served;
        job->timestamps = asked.timestamps;
        if (asked.regions) {
            job->detector = detector(speech_model_info_name(info.get()), res);
            if (!job->detector) return;
            try {
                job->detection = detection_request(job->detector->get(), asked.samples, asked.sample_rate, *asked.regions);
            } catch (const Failure & e) {
                ApiError error = openai::library_error(e);
                const std::string member = server_vad_member(e.option());
                if (!member.empty()) error.param = "chunking_strategy[" + member + "]";
                send_error(res, error);
                return;
            }
        }
        try {
            job->request = new_request(served->get());
            if (!asked.regions) check(speech_request_set_audio(job->request.get(), asked.samples.data(), asked.samples.size(), asked.sample_rate));
            apply_options(job->request.get(), asked.options);
        } catch (const Failure & e) {
            send_error(res, openai::library_error(e));
            return;
        }
        if (asked.regions) {
            job->request.reset();
            job->recognizer = served->get();
            job->samples = std::move(asked.samples);
            job->options = std::move(asked.options);
        }
        const uint64_t ticket = served->turns().take();
        std::thread(transcribe, job, std::ref(served->turns()), ticket).detach();
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait_for(*job, lock, [] { return false; }, [&] { return req.is_connection_closed(); })) return;
        if (job->failure) {
            send_error(res, openai::library_error(*job->failure));
            return;
        }
        if (job->status != SPEECH_OK) return;
        res.set_header("X-Speech-Stop", speech_stop_name(job->transcript.stop));
        if (asked.format == "text") {
            res.set_content(job->transcript.text, "text/plain; charset=utf-8");
        } else if (asked.format == "verbose_json") {
            res.set_content(openai::transcription_verbose_json(job->transcript, job->duration, asked.timestamps), "application/json");
        } else {
            res.set_content(openai::transcription_json(job->transcript.text), "application/json");
        }
    }

    /**
     * The detection model that chunking_strategy needs; otherwise answers a 400 that says to give speech serve one, or a
     * 503 while the page loads one.
     */
    std::shared_ptr<Served> detector(const std::string & recognizer, httplib::Response & res) const {
        if (auto served = models_.of(SPEECH_TASK_DETECTION)) return served;
        if (const std::string coming = models_.replacing(SPEECH_TASK_DETECTION); !coming.empty()) {
            res.set_header("Retry-After", "5");
            send_error(res, {503, "The speech detection model, which chunking_strategy needs, is being loaded (" + coming +
                                      "); send the request again once it is in place.", "", "model_loading"});
            return nullptr;
        }
        send_error(res, {400, "chunking_strategy recognizes the regions where a speech detection model finds that someone speaks, and this server "
                              "holds none. Give speech serve one beside the recognition model, as in speech serve " + recognizer + " silero-vad, or "
                              "leave chunking_strategy out.",
                         "chunking_strategy", "unsupported_parameter"});
        return nullptr;
    }
};

/** Says on stderr which model a task has, as the server starts or the page loads one. */
void report(const Served & served) {
    const auto info = served.info();
    std::fprintf(stderr, "speech serve: %s (%s) on %s, %d Hz, for %s\n", speech_model_info_name(info.get()), speech_model_info_architecture(info.get()),
                 speech_model_info_device(info.get()), speech_model_info_sample_rate(info.get()),
                 use_of(speech_model_info_task(info.get())));
}

int run_serve(const CommandLine & line, FILE *) {
    const std::string host = line.value("--host").value_or("127.0.0.1");
    const int port = line.integer("--port").value_or(8080);
    if (port < 0 || port > 65535) throw UsageError("--port takes a port from 0 to 65535, not " + std::to_string(port));
    const bool open = line.has("--open");
    if (open && host != "127.0.0.1" && host != "::1" && host != "localhost") {
        throw UsageError("--open opens the page, which the server has only on 127.0.0.1, ::1 or localhost, not on " + host +
                         "; leave --host out");
    }
    if (line.args.empty() && !open) {
        throw UsageError(no_model_message("serve MODEL [options]", ModelKind::Any) + "\nOr give --open, and pick the models on the page.");
    }
    const Loading loading = line.loading(true);
    ServedModels models(loading, line.value("--voices-dir").value_or(""), line.has("--irodori-compatible"));
    for (const std::string & argument : line.args) {
        models.load_given(model_file(argument), is_model_path(argument) ? "" : choice_name(find_model(argument)));
    }
    if (models.compatible() && !models.of(SPEECH_TASK_SYNTHESIS)) {
        throw UsageError("--irodori-compatible needs an Irodori synthesis model on the command line");
    }
    if (!loading.voices.empty()) {
        const auto synthesis = models.of(SPEECH_TASK_SYNTHESIS);
        if (!synthesis) throw UsageError("--add-voice adds voices to the synthesis model, and none is given");
        for (const auto & [name, file] : loading.voices) synthesis->add_voice(name, file);
    }
    for (const auto & served : models.all()) report(*served);

    httplib::Server http;
    // Nagle's algorithm would hold a small chunk of a stream until the client acknowledges the previous one.
    http.set_tcp_nodelay(true);
    // A file to recognize, or a recording to make a voice of, may take OpenAI's limit for an upload, 25 MB, and a
    // Realtime session may hold as much audio not yet transcribed, 546 s of its 16-bit PCM at 24000 Hz.
    constexpr size_t kUpload = 25 << 20;
    http.set_payload_max_length(kUpload);
    const int bound = port == 0 ? http.bind_to_any_port(host) : (http.bind_to_port(host, port) ? port : -1);
    if (bound < 0) {
        throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot listen on " + host + ":" + std::to_string(port) + "; choose another --port or --host");
    }
    const Access access(host, bound, line.values("--cors-origin"));
    Realtime realtime(models, access, kUpload);
    Server server(models, access, realtime);
    server.route(http);
    realtime.route(http);
    Page page(models, access, report);
    page.route(http);
    const std::string address = host.find(':') == std::string::npos ? host : "[" + host + "]";
    std::fprintf(stderr, "speech serve: speech.cpp %s, listening on http://%s:%d\n", speech_version(), address.c_str(), bound);
    if (access.loopback()) {
        std::fprintf(stderr, "speech serve: the page, which fetches and loads models, is at %s\n", access.page_url().c_str());
        if (open) open_in_browser(access.page_url());
    }
    if (!http.listen_after_bind()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "the server stopped listening");
    return 0;
}

}  // namespace

Command serve_command() {
    Command c;
    c.name = "serve";
    c.usage = "serve [MODEL [MODEL [MODEL]]] [options]";
    c.summary = "serve models over HTTP with OpenAI's audio API, and a page to try them";
    c.description =
        "Loads each MODEL, a synthesis model, a recognition model and a detection model at most, warmed up unless\n"
        "--no-warmup, adds the voices of --add-voice to the synthesis model, and serves them over HTTP: POST\n"
        "/v1/audio/speech, POST /v1/audio/transcriptions, which with chunking_strategy recognizes the regions where the\n"
        "detection model finds speech, GET /v1/models, GET /health, and OpenAI's Realtime transcription over a WebSocket at\n"
        "/v1/realtime, which transcribes each buffer the client commits, or with server_vad each region of the detection\n"
        "model, and the text agreed so far while it is said. It has no authentication and no TLS, and refuses a\n"
        "request that a web page at another origin than --cors-origin's sends. On 127.0.0.1, ::1 or localhost it also\n"
        "serves a page on which to pick models of the catalog, fetch them and try them, at the address it prints;\n"
        "--open opens it, and starts the server without a model if none is given.";
    c.flags = {
        {"--host", "ADDRESS", false, "the address to listen on, 127.0.0.1 unless given"},
        {"--port", "N", false, "the port, 8080 unless given; 0 for any free one"},
        {"--open", "", false, "open the page in the browser once the server listens"},
        {"--cors-origin", "ORIGIN|*", true, "an origin a web page may call the server from, or * for any"},
        {"--irodori-compatible", "", false, "accept the Irodori bridge API; requires an RF model and --voices-dir"},
        {"--voices-dir", "DIR", false, "save uploaded WAV voices and restore them when loading a synthesis model"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists; a detection model "
                    "runs on the CPU"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.model = ModelKind::Any;
    c.max_args = 3;
    c.run = run_serve;
    return c;
}
