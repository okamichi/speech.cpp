#include "page.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
extern char ** environ;
#endif

#include "cache.h"
#include "fetch.h"
#include "json-reader.h"
#include "json.h"
#include "openai-api.h"
#include "page-files.h"

namespace fs = std::filesystem;
using openai::ApiError;
using openai::send_error;

namespace server {

namespace {

/** What the page and its endpoints say while the server listens elsewhere than on a loopback address. */
constexpr const char * kOff =
    "The page of speech serve, which fetches and loads models, is on only while the server listens on 127.0.0.1, ::1 or "
    "localhost, so that no other machine reaches it. Run speech serve --open on the machine you use, or reach a server on "
    "another machine through SSH with the same port at both ends: ssh -L 8080:127.0.0.1:8080 HOST, run speech serve there "
    "without --host, and open the address it prints.";

const char * content_type(const std::string & name) {
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".html") == 0) return "text/html; charset=utf-8";
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".css") == 0) return "text/css; charset=utf-8";
    return "text/javascript; charset=utf-8";
}

const PageFile * page_file(const std::string & name) {
    for (size_t i = 0; i < speech_page_file_count; i++) {
        if (name == speech_page_files[i].name) return &speech_page_files[i];
    }
    return nullptr;
}

void send_file(const PageFile & file, httplib::Response & res) {
    res.set_content(reinterpret_cast<const char *>(file.bytes), file.size, content_type(file.name));
    res.set_header("X-Content-Type-Options", "nosniff");
    // Each release builds its page in, so a browser asks again rather than keep an earlier release's.
    res.set_header("Cache-Control", "no-cache");
}

speech_task task_of(const CatalogModel & model) {
    for (speech_task task : {SPEECH_TASK_SYNTHESIS, SPEECH_TASK_RECOGNITION, SPEECH_TASK_DETECTION}) {
        if (model.task == task_name(task)) return task;
    }
    throw std::logic_error("the catalog gives " + model.name + " the task " + model.task + ", which speech.h does not have");
}

std::string error_event(const ApiError & e) {
    return "{\"type\":\"error\",\"error\":" + openai::error_object(e) + "}";
}

/** A replacement begun for a load, given up unless the load got as far as replacing the model. */
class Replacing {
public:
    Replacing(ServedModels & models, speech_task task) : models_(models), task_(task) {}
    ~Replacing() {
        if (!loading_) models_.cancel_replacing(task_);
    }
    Replacing(const Replacing &) = delete;
    Replacing & operator=(const Replacing &) = delete;
    /** The load replaces the model from here on, and ends the replacement itself. */
    void loading() { loading_ = true; }

private:
    ServedModels & models_;
    speech_task task_;
    bool loading_ = false;
};

}  // namespace

std::string served_json(const Served & served) {
    return "{\"name\":" + (served.catalog_name().empty() ? std::string("null") : json_string(served.catalog_name())) +
           ",\"path\":" + json_string(served.path()) + ",\"defaults\":" + options_json(served.defaults()) +
           ",\"model\":" + speech_model_info_json(served.info().get()) + "}";
}

void Page::route(httplib::Server & http) {
    http.Get("/", [this](const httplib::Request & req, httplib::Response & res) {
        if (!access_.loopback()) {
            res.status = 404;
            res.set_content(std::string("<!doctype html><meta charset=\"utf-8\"><title>speech.cpp</title><p>") + kOff + "</p>\n",
                            "text/html; charset=utf-8");
            return;
        }
        if (!admits(req, res)) return;
        send_file(*page_file("index.html"), res);
        res.set_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
                                                  "img-src 'self' data:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; "
                                                  "form-action 'none'");
        res.set_header("Referrer-Policy", "no-referrer");
    });
    http.Get("/page/([a-z0-9-]+\\.(?:js|css))", [this](const httplib::Request & req, httplib::Response & res) {
        const PageFile * file = page_file(req.matches[1]);
        if (!admits(req, res)) return;
        if (!file) {
            send_error(res, {404, "The page has no file " + json_string(req.matches[1]) + ".", "", ""});
            return;
        }
        send_file(*file, res);
    });
    http.Get("/speech/models", [this](const httplib::Request & req, httplib::Response & res) {
        if (admits(req, res)) models(res);
    });
    http.Post("/speech/load", [this](const httplib::Request & req, httplib::Response & res) {
        if (admits(req, res)) load(req, res);
    });
    http.Post("/speech/voices", [this](const httplib::Request & req, httplib::Response & res) {
        if (admits(req, res)) add_voice(req, res);
    });
}

bool Page::admits(const httplib::Request & req, httplib::Response & res) const {
    if (!access_.loopback()) {
        send_error(res, {404, kOff, "", "page_off"});
        return false;
    }
    const std::optional<ApiError> refusal = access_.host_refusal(req);
    if (!refusal) return true;
    send_error(res, *refusal);
    return false;
}

void Page::models(httplib::Response & res) const {
    const auto task = [this](speech_task t) {
        const auto served = models_.of(t);
        const std::string coming = models_.replacing(t);
        return "{\"held\":" + (served ? served_json(*served) : "null") + ",\"replacing\":" + (coming.empty() ? "null" : json_string(coming)) + "}";
    };
    try {
        res.set_content("{\"catalog\":" + models_json() + ",\"synthesis\":" + task(SPEECH_TASK_SYNTHESIS) + ",\"recognition\":" +
                            task(SPEECH_TASK_RECOGNITION) + ",\"detection\":" + task(SPEECH_TASK_DETECTION) + "}",
                        "application/json");
    } catch (const Failure & e) {
        send_error(res, openai::library_error(e));
    }
}

void Page::load(const httplib::Request & req, httplib::Response & res) {
    std::string name;
    try {
        const JsonValue body = parse_json(req.body);
        const JsonValue * model = body.member("model");
        if (body.kind != JsonValue::Kind::Object || body.members.size() != 1 || !model || model->kind != JsonValue::Kind::String) throw std::invalid_argument("");
        name = model->text;
    } catch (const std::invalid_argument &) {
        send_error(res, {400, "Send {\"model\": NAME}, the name of a model of the catalog.", "model", "invalid_value"});
        return;
    }
    if (is_model_path(name)) {
        send_error(res, {400, "The page loads the models of the catalog alone, by the names speech models lists; give speech serve a model file "
                              "on its command line.", "model", "invalid_value"});
        return;
    }
    CatalogChoice choice;
    try {
        choice = find_model(name);
    } catch (const UsageError & e) {
        send_error(res, {404, e.what(), "model", "model_not_found"});
        return;
    }
    const speech_task task = task_of(*choice.model);
    if (!models_.begin_replacing(task, choice_name(choice))) {
        send_error(res, {409, "The speech " + std::string(task_name(task)) + " model is being replaced by " + models_.replacing(task) +
                              "; wait until it is in place.", "model", "model_loading"});
        return;
    }
    auto replacing = std::make_shared<Replacing>(models_, task);
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider("text/event-stream", [this, choice, task, replacing](size_t, httplib::DataSink & sink) {
        const auto send = [&sink](const std::string & event) {
            const std::string data = "data: " + event + "\n\n";
            return sink.is_writable() && sink.write(data.data(), data.size());
        };
        // A page that goes away stops the fetch, whose part the next load resumes; a load once begun ends all the same.
        // fetch_model() tells of no progress while it waits for another process's lock, nor when that process has put
        // the file in place, so whether the page is still there is asked again before the model is replaced.
        bool listening = true;
        fs::path path;
        try {
            path = fetch_model(
                choice,
                [&](uint64_t done, uint64_t total) {
                    listening = listening && send("{\"type\":\"fetch\",\"done\":" + std::to_string(done) + ",\"total\":" + std::to_string(total) + "}");
                    return listening;
                },
                [&](const std::string & note) { listening = listening && send("{\"type\":\"note\",\"message\":" + json_string(note) + "}"); });
        } catch (const Failure & e) {
            send(error_event(openai::library_error(e)));
            sink.done();
            return true;
        }
        if (!listening || !send("{\"type\":\"load\"}")) {
            sink.done();
            return true;
        }
        replacing->loading();
        try {
            const auto served = models_.replace(task, path.u8string());
            report_(*served);
            send("{\"type\":\"loaded\",\"task\":\"" + std::string(task_name(task)) + "\",\"held\":" + served_json(*served) + "}");
        } catch (const Failure & e) {
            send(error_event(openai::library_error(e)));
        } catch (const UsageError & e) {
            send(error_event({400, e.what(), "model", "unsupported_value"}));
        } catch (const std::exception & e) {
            send(error_event({500, e.what(), "model", "io"}));
        }
        sink.done();
        return true;
    });
}

void Page::add_voice(const httplib::Request & req, httplib::Response & res) {
    if (!models_.of(SPEECH_TASK_SYNTHESIS)) {
        send_error(res, {409, "No speech synthesis model is in place to add a voice to; pick one first.", "", "model_not_found"});
        return;
    }
    const auto name = req.form.fields.find("name");
    const auto file = req.form.files.find("file");
    if (!req.is_multipart_form_data() || name == req.form.fields.end() || file == req.form.files.end()) {
        send_error(res, {400, "Send a multipart form with the voice's \"name\" and its recording as the WAVE file \"file\".", "file",
                         "missing_required_parameter"});
        return;
    }
    try {
        const auto uploaded = models_.upload_voice(name->second.content, file->second.content);
        res.set_content("{\"held\":" + served_json(*uploaded.first) + "}", "application/json");
    } catch (const openai::ApiError & e) {
        send_error(res, e);
    } catch (const Failure & e) {
        send_error(res, openai::library_error(e));
    } catch (const std::exception & e) {
        send_error(res, {500, e.what(), "file", "io"});
    }
}

void open_in_browser(const std::string & url) {
#ifdef _WIN32
    const std::wstring wide(url.begin(), url.end());
    const auto opened = (INT_PTR) ShellExecuteW(nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    if (opened <= 32) std::fprintf(stderr, "speech serve: cannot open a browser (error %lld); open the address above in one\n", (long long) opened);
#else
#ifdef __APPLE__
    const char * opener = "open";
#else
    const char * opener = "xdg-open";
#endif
    // The opener's output would reach the server's stdout, which carries nothing.
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    // The browser it starts holds no file of the server open, as curl does not (fetch.cpp).
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
#ifdef __APPLE__
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
#else
    posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
    char * argv[] = {const_cast<char *>(opener), const_cast<char *>(url.c_str()), nullptr};
    pid_t pid = 0;
    const int error = posix_spawnp(&pid, opener, &actions, &attributes, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    if (error != 0) {
        std::fprintf(stderr, "speech serve: cannot start %s (%s); open the address above in a browser\n", opener, std::strerror(error));
        return;
    }
    std::thread([pid] {
        int status = 0;
        waitpid(pid, &status, 0);
    }).detach();
#endif
}

}  // namespace server
