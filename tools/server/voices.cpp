#include "voices.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <set>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <io.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

#include "ggml.h"
#include "httplib.h"
#include "json.h"
#include "log.h"
#include "openai-error.h"
#include "openai-api.h"
#include "wav.h"
#include "error.h"
#include "served-models.h"

namespace server {
namespace {
namespace fs = std::filesystem;

std::string lower(std::string s) {
    for (char & c : s) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
    return s;
}

void valid_name(const std::string & name) {
    const std::set<std::string> reserved = {"none", "no_ref", "no-ref", "null", "text-only"};
    if (name.empty() || reserved.count(lower(name)) || name.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") != std::string::npos) {
        throw openai::ApiError{400, "Give a voice_id of ASCII letters, digits, underscores or hyphens, other than a no-reference name.", "voice_id", "invalid_value"};
    }
}

[[noreturn]] void io_error(const fs::path & path, const std::string & reason) {
    throw Failure("io", "file", "Cannot save or restore " + path.u8string() + ": " + reason + "; check the voice directory and its permissions.");
}

void sync_directory(const fs::path & directory) {
#ifndef _WIN32
    const int fd = ::open(directory.c_str(), O_RDONLY);
    if (fd < 0) io_error(directory, std::strerror(errno));
    const int result = fsync(fd), saved = errno;
    const int closed = close(fd);
    if (result || closed) io_error(directory, std::strerror(result ? saved : errno));
#else
    (void) directory;
#endif
}

/** A uniquely created file whose unfinished name is never scanned as a voice. */
class Recording {
public:
    Recording(const fs::path & directory, const std::string & bytes) {
        std::random_device random;
        path = directory / fs::u8path(".speech-voice-" + std::to_string(random()) + "-" + std::to_string(random()) + ".tmp");
        FILE * file = ggml_fopen(path.u8string().c_str(), "wbx");
        if (!file) io_error(path, std::strerror(errno));
        const bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
        const bool flushed = std::fflush(file) == 0;
#ifdef _WIN32
        const bool synced = _commit(_fileno(file)) == 0;
#else
        const bool synced = fsync(fileno(file)) == 0;
#endif
        const bool closed = std::fclose(file) == 0;
        if (!written || !flushed || !synced || !closed) {
            std::error_code ec;
            fs::remove(path, ec);
            io_error(path, "writing or synchronizing the recording failed" + (ec ? "; cleanup failed: " + ec.message() : ""));
        }
    }
    ~Recording() {
        std::error_code ec;
        fs::remove(path, ec);
        if (ec) log("Cannot remove " + path.u8string() + ": " + ec.message());
    }
    fs::path path;
};

}  // namespace

void require_irodori_compatible(const speech_model_info * info) {
    if (std::string(speech_model_info_architecture(info)) != "irodori-tts" ||
        !speech_model_info_voice_files(info) || !speech_model_info_takes(info, SPEECH_OPT_INSTRUCTIONS) ||
        !speech_model_info_takes(info, SPEECH_OPT_IRODORI_COMPATIBLE) || !speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_TEXT) ||
        !speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_INSTRUCTIONS) || !speech_model_info_takes(info, SPEECH_OPT_CFG_SCALE_SPEAKER)) {
        throw UsageError("--irodori-compatible needs an Irodori RF synthesis model with caption and reference conditioning");
    }
    for (size_t i = 0; i < speech_model_info_voice_count(info); ++i) {
        if (std::string(speech_model_info_voice_name(info, i)) == "none") return;
    }
    throw UsageError("--irodori-compatible needs a model with the built-in voice none; use a model containing the null speaker");
}

VoiceStore::VoiceStore(const std::string & directory) {
    if (directory.empty()) return;
    directory_ = fs::absolute(fs::u8path(directory));
    std::error_code ec;
    fs::create_directories(directory_, ec);
    if (ec) io_error(directory_, ec.message());
    const Recording probe(directory_, "");
    sync_directory(directory_);
}

void VoiceStore::restore(Served & served) const {
    if (!persistent()) return;
    std::vector<fs::path> files;
    std::error_code ec;
    fs::directory_iterator iterator(directory_, ec);
    if (ec) io_error(directory_, ec.message());
    for (const auto & entry : iterator) {
        if (lower(entry.path().extension().u8string()) != ".wav") continue;
        if (!entry.is_regular_file() || entry.is_symlink()) io_error(entry.path(), "a voice must be a regular WAV file, not a link");
        files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    std::set<std::string> names;
    for (const auto & path : files) {
        const std::string name = path.stem().u8string();
        try {
            valid_name(name);
            if (!names.insert(name).second) io_error(path, "duplicate voice ID " + name);
            served.add_voice(name, path.u8string());
        } catch (const openai::ApiError & e) {
            io_error(path, e.message);
        } catch (const Failure & e) {
            io_error(path, e.what());
        }
    }
}

std::string VoiceStore::add(Served & served, const std::string & name, const std::string & bytes) const {
    if (persistent()) valid_name(name);
    if (bytes.empty()) throw openai::ApiError{400, "The recording is empty; upload a WAV file.", "file", "invalid_value"};
    try {
        parse_wav(bytes, "file");
    } catch (const Error & e) {
        throw openai::ApiError{400, e.what(), "file", "invalid_value"};
    }
    const auto info = served.info();
    for (size_t i = 0; i < speech_model_info_voice_count(info.get()); ++i) {
        if (name == speech_model_info_voice_name(info.get(), i)) {
            if (!persistent()) throw Failure("invalid_argument", "name", "A voice with this name already exists; choose another name.");
            throw openai::ApiError{409, "A voice named " + name + " already exists; choose another voice_id.", "voice_id", "voice_exists"};
        }
    }
    const fs::path directory = persistent() ? directory_ : fs::temp_directory_path();
    const fs::path destination = directory / fs::u8path(name + ".wav");
    if (persistent()) {
        for (const auto & entry : fs::directory_iterator(directory)) {
            if (lower(entry.path().extension().u8string()) == ".wav" && entry.path().stem().u8string() == name) {
                throw openai::ApiError{409, "A recording for " + name + " already exists; choose another voice_id.", "voice_id", "voice_exists"};
            }
        }
    }
    const std::string result = "{\"id\":" + json_string(name) + ",\"object\":\"voice_file\",\"filename\":" + json_string(name + ".wav") +
                               ",\"bytes\":" + std::to_string(bytes.size()) + ",\"created_at\":" + std::to_string(std::time(nullptr)) + "}";
    const Recording staged(directory, bytes);
    if (!persistent()) {
        served.add_voice(name, staged.path.u8string());
        return result;
    }
    std::error_code ec;
#ifdef _WIN32
    if (!MoveFileExW(staged.path.c_str(), destination.c_str(), MOVEFILE_WRITE_THROUGH)) {
        const auto code = GetLastError();
        ec = code == ERROR_ALREADY_EXISTS || code == ERROR_FILE_EXISTS ? std::make_error_code(std::errc::file_exists)
                                                                       : std::error_code((int) code, std::system_category());
    }
#else
    fs::create_hard_link(staged.path, destination, ec);
#endif
    if (ec == std::errc::file_exists) throw openai::ApiError{409, "The voice file already exists; choose another voice_id.", "voice_id", "voice_exists"};
    if (ec) io_error(destination, ec.message());
    try {
        sync_directory(directory);
        served.add_voice(name, destination.u8string());
    } catch (...) {
        const auto failure = std::current_exception();
        fs::remove(destination, ec);
        if (ec) io_error(destination, "registration failed and cleanup failed: " + ec.message());
        sync_directory(directory);
        std::rethrow_exception(failure);
    }
    return result;
}

void route_voices(httplib::Server & http, ServedModels & models) {
    http.Get("/v1/audio/voices", [&models](const httplib::Request &, httplib::Response & res) {
        const auto served = models.of(SPEECH_TASK_SYNTHESIS);
        if (!served) {
            const bool loading = !models.replacing(SPEECH_TASK_SYNTHESIS).empty();
            if (loading) res.set_header("Retry-After", "5");
            openai::send_error(res, {loading ? 503 : 404, "No synthesis model is ready; load one first.", "", loading ? "model_loading" : "model_not_found"});
            return;
        }
        const auto info = served->info();
        std::string data;
        for (size_t i = 0; i < speech_model_info_voice_count(info.get()); ++i) {
            const std::string name = speech_model_info_voice_name(info.get(), i);
            data += (data.empty() ? "" : ",") + std::string("{\"id\":") + json_string(name) +
                    ",\"object\":\"voice\",\"no_ref\":" + (name == "none" ? "true" : "false") + "}";
        }
        res.set_content("{\"object\":\"list\",\"data\":[" + data + "]}", "application/json");
    });
    http.Post("/v1/audio/voices", [&models](const httplib::Request & req, httplib::Response & res) {
        try {
            if (!req.is_multipart_form_data() || req.form.files.count("file") != 1 || req.form.files.size() != 1 ||
                req.form.fields.count("voice_id") > 1 || req.form.fields.size() != req.form.fields.count("voice_id")) {
                throw openai::ApiError{400, "Send one WAV file as file and at most one voice_id field.", "file", "invalid_value"};
            }
            const auto & file = req.form.files.find("file")->second;
            const auto id = req.form.fields.find("voice_id");
            std::string name = id == req.form.fields.end() || id->second.content.empty()
                                   ? fs::u8path(file.filename).stem().u8string() : id->second.content;
            const size_t first = name.find_first_not_of(" \t\r\n"), last = name.find_last_not_of(" \t\r\n");
            name = first == std::string::npos ? "" : name.substr(first, last - first + 1);
            valid_name(name);
            auto uploaded = models.upload_voice(name, file.content);
            res.status = 201;
            res.set_content(std::move(uploaded.second), "application/json");
        } catch (const openai::ApiError & e) {
            openai::send_error(res, e);
        } catch (const Failure & e) {
            openai::send_error(res, openai::library_error(e));
        } catch (const std::exception & e) {
            openai::send_error(res, {500, e.what(), "file", "io"});
        }
    });
}

}  // namespace server
