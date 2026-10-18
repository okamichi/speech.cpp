#include "fetch.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <system_error>
#include <thread>
#include <vector>

#include "cache.h"
#include "curl.h"
#include "failure.h"
#include "ggml.h"
#include "sha256.h"
#include "speech.h"
#include "terminal.h"

namespace fs = std::filesystem;

namespace {

using Clock = std::chrono::steady_clock;

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

std::string megabytes(uint64_t bytes) {
    char s[32];
    std::snprintf(s, sizeof s, "%.0f MB", bytes / 1e6);
    return s;
}

/** A size in gigabytes, or in megabytes below 10 MB, which two decimals of a gigabyte would show as 0.00 GB. */
std::string gigabytes(uint64_t bytes) {
    char s[32];
    if (bytes < 10000000) std::snprintf(s, sizeof s, "%.1f MB", bytes / 1e6);
    else std::snprintf(s, sizeof s, "%.2f GB", bytes / 1e9);
    return s;
}

/** Hashes the first `size` bytes of `file`, the part a fetch resumes after. */
void hash_part(const fs::path & part, uint64_t size, Sha256 & hash) {
    const std::unique_ptr<FILE, int (*)(FILE *)> f(ggml_fopen(part.u8string().c_str(), "rb"), std::fclose);
    if (!f) throw io_failure("cannot read " + part.u8string());
    std::vector<char> buffer(1 << 20);
    for (uint64_t left = size; left > 0;) {
        const size_t n = std::fread(buffer.data(), 1, (size_t) std::min<uint64_t>(left, buffer.size()), f.get());
        if (n == 0) throw io_failure("cannot read " + part.u8string());
        hash.update(buffer.data(), n);
        left -= n;
    }
}

/** Removes a part whose bytes cannot become the file, and says so. */
Failure discarded(const fs::path & part, const std::string & why) {
    std::error_code ignored;
    fs::remove(part, ignored);
    return io_failure(why + "; the fetched bytes are removed, and the same command fetches the file again from its start");
}

/**
 * The progress of a fetch on stderr: on a terminal one line, a bar with the share, the sizes, the rate and the time
 * left, rewritten as it moves and erased once the file is whole; elsewhere a line for each tenth.
 */
class StderrProgress {
public:
    ~StderrProgress() {
        // A failure's message starts on a line of its own.
        if (open_) std::fprintf(stderr, "\n");
    }

    bool operator()(uint64_t done, uint64_t total) {
        const Clock::time_point now = Clock::now();
        if (!first_) first_ = {done, now};
        const double seconds = std::chrono::duration<double>(now - first_->second).count();
        const double rate = seconds > 0 ? (done - first_->first) / seconds / 1e6 : 0;
        const int tenth = (int) (done * 10 / total);
        char line[96];
        std::snprintf(line, sizeof line, "%s of %s, %d%%, %.1f MB/s", megabytes(done).c_str(), megabytes(total).c_str(), (int) (done * 100 / total), rate);
        if (terminal_) {
            open_ = done != total;
            if (!open_) {
                std::fputs("\r\033[K", stderr);
                return true;
            }
            char shown[96];
            std::snprintf(shown, sizeof shown, "%3d%%  %s of %s  %.1f MB/s", (int) (done * 100 / total), megabytes(done).c_str(), megabytes(total).c_str(), rate);
            std::string text = shown;
            if (rate > 0) text += "  " + std::to_string((long long) ((total - done) / (rate * 1e6)) + 1) + " s left";
            // The line stays narrower than the terminal, since a line that wraps is no longer the one "\r" returns to: the
            // bar takes what the numbers leave, up to 30 columns, and goes where fewer than 10 are left.
            const int room = terminal::columns(stderr) - 1 - 2;
            const int width = std::min(30, room - 2 - (int) text.size());
            const std::string bar = width >= 10 ? terminal::bar(done, total, width) + "  " : "";
            std::fprintf(stderr, "\r\033[K  %s%s", bar.c_str(), terminal::paint(terminal::Color::Gray, terminal::head(text, width >= 10 ? (int) text.size() : room)).c_str());
            std::fflush(stderr);
        } else if (tenth > tenth_) {
            std::fprintf(stderr, "  %s\n", line);
        }
        tenth_ = tenth;
        return true;
    }

private:
    const bool terminal_ = terminal::draws();
    bool open_ = false;
    int tenth_ = 0;
    std::optional<std::pair<uint64_t, Clock::time_point>> first_;
};

}  // namespace

fs::path fetch_model(const CatalogChoice & choice, const FetchProgress & progress, const FetchNote & note) {
    const fs::path path = model_path(choice);
    std::error_code e;
    // A file is renamed into place only once it is whole and checked.
    if (fs::is_regular_file(path, e)) return path;
    fs::create_directories(path.parent_path(), e);
    if (e) throw io_failure("cannot make the folder " + path.parent_path().u8string() + ": " + e.message());
    const FileLock lock(path, [&] { note("waiting for another process that fetches or removes " + path.u8string()); });
    if (fs::is_regular_file(path, e)) return path;

    const CatalogFile & file = *choice.file;
    const fs::path part = partial_path(path);
    const uint64_t offset = fs::exists(part, e) ? (uint64_t) fs::file_size(part, e) : 0;
    if (e) throw io_failure("cannot read the size of " + part.u8string() + ": " + e.message());
    if (offset > file.size) throw discarded(part, part.u8string() + " holds " + std::to_string(offset) + " bytes, more than the file's " + std::to_string(file.size));
    Sha256 hash;
    if (offset > 0) {
        note("resuming at " + megabytes(offset) + " of " + megabytes(file.size));
        hash_part(part, offset, hash);
    }

    std::atomic<uint64_t> done{offset};
    if (offset < file.size) {
        const std::unique_ptr<FILE, int (*)(FILE *)> out(ggml_fopen(part.u8string().c_str(), "ab"), std::fclose);
        if (!out) throw io_failure("cannot write " + part.u8string() + "; check that its folder is writable");
        std::vector<std::string> args = {"--fail", "--location", "--silent", "--show-error", "--speed-limit", "1024", "--speed-time", "60"};
        if (offset > 0) args.insert(args.end(), {"--continue-at", std::to_string(offset)});
        args.push_back(file_url(choice));
        Curl curl(args, "speech fetches models with the system's curl, which is not there; install curl, or fetch the model file yourself and "
                        "give its path");

        bool too_long = false, unwritten = false;
        std::string curl_error;
        std::thread writer([&] {
            std::vector<char> buffer(1 << 20);
            while (const size_t n = read_pipe(curl.out(), buffer.data(), buffer.size())) {
                too_long = done + n > file.size;
                unwritten = !too_long && std::fwrite(buffer.data(), 1, n, out.get()) != n;
                if (too_long || unwritten) break;
                hash.update(buffer.data(), n);
                done += n;
            }
            // curl stops at its next write once the pipe has no reader.
            close_pipe(curl.out());
        });
        std::thread reader([&] { curl_error = read_to_end(curl.err()); });
        int code = 0;
        bool stopped = false;
        while (!curl.wait(250, code)) {
            if (!stopped && !progress(done, file.size)) {
                curl.stop();
                stopped = true;
            }
        }
        writer.join();
        reader.join();
        if (too_long) throw discarded(part, "the server sent more than the file's " + std::to_string(file.size) + " bytes");
        if (unwritten || std::fflush(out.get()) != 0) throw io_failure("cannot write " + part.u8string() + "; check the space left on its disk");
        if (stopped) throw io_failure("the fetch of " + choice_name(choice) + " was stopped; the same command resumes it");
        if (code != 0) {
            while (!curl_error.empty() && (curl_error.back() == '\n' || curl_error.back() == '\r')) curl_error.pop_back();
            throw io_failure("fetching " + file_url(choice) + " failed with curl's exit code " + std::to_string(code) +
                             (curl_error.empty() ? "" : " (" + curl_error + ")") + "; the same command resumes it");
        }
    }
    progress(done, file.size);
    if (done != file.size) {
        throw io_failure("curl ended after " + std::to_string(done) + " of the file's " + std::to_string(file.size) + " bytes; the same command resumes it");
    }
    const std::string sha256 = hash.hex();
    if (sha256 != file.sha256) {
        throw discarded(part, "the SHA-256 of the fetched " + file.file + " is " + sha256 + ", not the catalog's " + file.sha256);
    }
    fs::rename(part, path, e);
    if (e) throw io_failure("cannot rename " + part.u8string() + " to " + path.filename().u8string() + ": " + e.message());
    return path;
}

fs::path fetch_on_stderr(const CatalogChoice & choice) {
    const fs::path path = model_path(choice);
    const std::string from = ", from https://huggingface.co/" + choice.model->repository + " into " + path.parent_path().u8string();
    std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Fetch, "Fetching " + choice_name(choice) + ", " + choice.file->file + " (" +
                                                                            gigabytes(choice.file->size) + ")" + terminal::paint(terminal::Color::Gray, from))
                                     .c_str());
    const auto t0 = Clock::now();
    StderrProgress progress;
    fetch_model(choice, std::ref(progress), [](const std::string & note) { std::fprintf(stderr, "  %s\n", note.c_str()); });
    char took[64];
    std::snprintf(took, sizeof took, " in %.0f s and checked its SHA-256", std::chrono::duration<double>(Clock::now() - t0).count());
    std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, "Fetched " + choice_name(choice) + took).c_str());
    return path;
}

std::string model_file(const std::string & argument) {
    if (is_model_path(argument)) return argument;
    const CatalogChoice choice = find_model(argument);
    const fs::path path = model_path(choice);
    std::error_code e;
    return (fs::is_regular_file(path, e) ? path : fetch_on_stderr(choice)).u8string();
}
