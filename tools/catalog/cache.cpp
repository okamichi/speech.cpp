#include "cache.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <system_error>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "failure.h"
#include "json.h"
#include "speech.h"

namespace fs = std::filesystem;

namespace {

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

/**
 * An environment variable as a path, or an empty path when it is unset or empty. On Windows the value is read as
 * UTF-16, since getenv() gives it in the ANSI code page, which cannot hold most user names outside ASCII.
 */
fs::path environment(const char * name) {
#ifdef _WIN32
    const std::wstring wide(name, name + std::char_traits<char>::length(name));
    const wchar_t * value = _wgetenv(wide.c_str());
#else
    const char * value = std::getenv(name);
#endif
    return value ? fs::path(value) : fs::path();
}

/** The folder of a model's repository, which holds a folder for each content of a file. */
fs::path repository_dir(const CatalogModel & m) {
    std::string repository = m.repository;
    repository.replace(repository.find('/'), 1, "--");
    return model_dir() / fs::u8path(repository);
}

/** A name of 64 lower-case hexadecimal digits, as a SHA-256 is written. */
bool sha256_name(const std::string & name) {
    return name.size() == 64 && name.find_first_not_of("0123456789abcdef") == std::string::npos;
}

uint64_t size_or_zero(const fs::path & path) {
    std::error_code e;
    const uintmax_t size = fs::file_size(path, e);
    return e ? 0 : (uint64_t) size;
}

/** The files or the folders in `dir`, none when it is not there. */
std::vector<fs::path> children(const fs::path & dir, bool files) {
    std::vector<fs::path> out;
    std::error_code e;
    if (!fs::is_directory(dir, e)) return out;
    for (fs::directory_iterator it(dir, e), end; !e && it != end; it.increment(e)) {
        if (files ? it->is_regular_file(e) : it->is_directory(e)) out.push_back(it->path());
    }
    if (e) throw io_failure("cannot list " + dir.u8string() + ": " + e.message());
    return out;
}


/** Removes `dir` and its parent when each is empty; another process may have put a file into either since. */
void remove_empty_folders(fs::path dir) {
    const fs::path root = model_dir();
    for (int i = 0; i < 2 && dir != root; i++, dir = dir.parent_path()) {
        std::error_code e;
        if (!fs::is_empty(dir, e) || e) return;
        fs::remove(dir, e);
    }
}

}  // namespace

fs::path cache_dir() {
#if defined(_WIN32)
    const fs::path base = environment("LOCALAPPDATA");
    if (base.empty()) throw io_failure("LOCALAPPDATA is not set, so speech.cpp has no cache folder; set SPEECH_MODEL_DIR to a folder for the models");
    return base / "speech.cpp";
#else
    const fs::path home = environment("HOME");
    if (home.empty()) throw io_failure("HOME is not set, so speech.cpp has no cache folder; set SPEECH_MODEL_DIR to a folder for the models");
#if defined(__APPLE__)
    return home / "Library" / "Caches" / "speech.cpp";
#else
    // The XDG Base Directory specification ignores a value that is not an absolute path.
    const fs::path xdg = environment("XDG_CACHE_HOME");
    return (xdg.is_absolute() ? xdg : home / ".cache") / "speech.cpp";
#endif
#endif
}

fs::path model_dir() {
    const fs::path chosen = environment("SPEECH_MODEL_DIR");
    if (!chosen.empty()) {
        if (!chosen.is_absolute()) throw io_failure("SPEECH_MODEL_DIR is " + chosen.u8string() + ", not an absolute path; give the folder's whole path");
        return chosen;
    }
    return cache_dir() / "models";
}

fs::path model_path(const CatalogChoice & choice) {
    return repository_dir(*choice.model) / fs::u8path(choice.file->sha256) / fs::u8path(choice.file->file);
}

fs::path partial_path(const fs::path & path) {
    fs::path out = path;
    out += ".part";
    return out;
}

Presence presence(const CatalogChoice & choice) {
    const fs::path path = model_path(choice);
    std::error_code e;
    if (fs::is_regular_file(path, e)) return {true, 0};
    return {false, size_or_zero(partial_path(path))};
}

std::vector<OldFile> old_files() {
    const fs::path root = model_dir();
    std::set<fs::path> current;
    for (const CatalogModel & m : catalog()) {
        for (const CatalogFile & f : m.files) {
            const fs::path path = model_path({&m, &f});
            current.insert(path);
            current.insert(partial_path(path));
        }
    }
    // Only the folders a fetch makes are looked into, <owner>--<name>/<sha256>, so that a SPEECH_MODEL_DIR that holds
    // other files too never offers them for removal.
    std::vector<OldFile> out;
    for (const fs::path & repository : children(root, false)) {
        if (repository.filename().u8string().find("--") == std::string::npos) continue;
        for (const fs::path & content : children(repository, false)) {
            if (!sha256_name(content.filename().u8string())) continue;
            for (const fs::path & file : children(content, true)) {
                if (file.extension() != ".lock" && !current.count(file)) out.push_back({file, size_or_zero(file)});
            }
        }
    }
    return out;
}

FileLock::FileLock(const fs::path & file, const std::function<void()> & waiting) {
    path_ = file;
    path_ += ".lock";
    bool told = false;
#ifdef _WIN32
    int deleting = 0;
#endif
    for (;;) {
#ifdef _WIN32
        // FILE_SHARE_DELETE lets the process that holds the lock remove it while others wait on it; they find it
        // delete-pending once they hold it, and one being deleted cannot be opened until its last handle closes.
        HANDLE h = CreateFileW(path_.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            // A lock file being deleted cannot be opened for the moment between its holder's DeleteFileW() and
            // CloseHandle(), which looks like a folder that is not writable, so the second is told after 5 s.
            const DWORD error = GetLastError();
            if (error == ERROR_ACCESS_DENIED && ++deleting < 100) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            throw io_failure("cannot open " + path_.u8string() + " (Windows error " + std::to_string(error) + "); check that its folder is writable");
        }
        OVERLAPPED at = {};
        if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &at)) {
            if (!told) waiting();
            told = true;
            at = {};
            if (!LockFileEx(h, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &at)) {
                CloseHandle(h);
                throw io_failure("cannot lock " + path_.u8string() + " (Windows error " + std::to_string(GetLastError()) + ")");
            }
        }
        FILE_STANDARD_INFO info = {};
        if (!GetFileInformationByHandleEx(h, FileStandardInfo, &info, sizeof info)) {
            CloseHandle(h);
            throw io_failure("cannot read the state of " + path_.u8string() + " (Windows error " + std::to_string(GetLastError()) + ")");
        }
        if (!info.DeletePending) {
            handle_ = h;
            return;
        }
        CloseHandle(h);
#else
        const int fd = ::open(path_.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
        if (fd < 0) throw io_failure("cannot open " + path_.u8string() + ": " + std::generic_category().message(errno) + "; check that its folder is writable");
        if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
            if (!told) waiting();
            told = true;
            if (flock(fd, LOCK_EX) != 0) {
                const int error = errno;
                ::close(fd);
                throw io_failure("cannot lock " + path_.u8string() + ": " + std::generic_category().message(error));
            }
        }
        struct stat held, named;
        if (fstat(fd, &held) == 0 && stat(path_.c_str(), &named) == 0 && held.st_dev == named.st_dev && held.st_ino == named.st_ino) {
            fd_ = fd;
            return;
        }
        ::close(fd);
#endif
    }
}

FileLock::~FileLock() {
#ifdef _WIN32
    // The file goes once its last handle closes, this one.
    DeleteFileW(path_.c_str());
    CloseHandle((HANDLE) handle_);
#else
    ::unlink(path_.c_str());
    ::close(fd_);
#endif
}

uint64_t remove_from_folder(const fs::path & path) {
    // A partial fetch is guarded by the lock of the file it becomes.
    const fs::path file = path.extension() == ".part" ? fs::path(path).replace_extension() : path;
    uint64_t removed = 0;
    {
        const FileLock lock(file, [&] { std::fprintf(stderr, "waiting for another process that fetches or removes %s\n", file.u8string().c_str()); });
        for (const fs::path & p : {file, partial_path(file)}) {
            std::error_code e;
            const uint64_t size = size_or_zero(p);
            if (fs::remove(p, e)) removed += size;
            if (e) throw io_failure("cannot remove " + p.u8string() + ": " + e.message() + "; stop the program that uses it and run this again");
        }
    }
    remove_empty_folders(file.parent_path());
    return removed;
}

std::string models_json() {
    std::string out = "{\"version\":" + json_string(speech_version()) + ",\"directory\":" + json_string(model_dir().u8string()) + ",\"models\":[";
    const auto strings = [](const std::vector<std::string> & items) {
        std::string s = "[";
        for (size_t i = 0; i < items.size(); i++) s += (i ? "," : "") + json_string(items[i]);
        return s + "]";
    };
    for (size_t i = 0; i < catalog().size(); i++) {
        const CatalogModel & m = catalog()[i];
        out += std::string(i ? "," : "") + "{\"name\":" + json_string(m.name) + ",\"repository\":" + json_string(m.repository) +
               ",\"revision\":" + json_string(m.revision) + ",\"task\":" + json_string(m.task) + ",\"languages\":" + strings(m.languages) +
               ",\"voice_files\":" + (m.voice_files ? "true" : "false") + ",\"start\":" + strings(m.start) + ",\"type\":" + json_string(m.type) +
               ",\"files\":[";
        for (size_t k = 0; k < m.files.size(); k++) {
            const CatalogFile & f = m.files[k];
            const CatalogChoice choice{&m, &f};
            const Presence p = presence(choice);
            out += std::string(k ? "," : "") + "{\"type\":" + json_string(f.type) + ",\"file\":" + json_string(f.file) +
                   ",\"size\":" + std::to_string(f.size) + ",\"sha256\":" + json_string(f.sha256) + ",\"url\":" + json_string(file_url(choice)) +
                   ",\"path\":" + json_string(model_path(choice).u8string()) + ",\"fetched\":" + (p.whole ? "true" : "false") +
                   ",\"partial\":" + std::to_string(p.partial) + "}";
        }
        out += "]}";
    }
    out += "],\"old\":[";
    const std::vector<OldFile> old = old_files();
    for (size_t i = 0; i < old.size(); i++) {
        out += std::string(i ? "," : "") + "{\"path\":" + json_string(old[i].path.u8string()) + ",\"size\":" + std::to_string(old[i].size) + "}";
    }
    return out + "]}";
}
