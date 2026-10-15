#include "release.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#include "cache.h"
#include "curl.h"
#include "failure.h"
#include "ggml.h"
#include "json-reader.h"
#include "json.h"
#include "speech.h"

namespace fs = std::filesystem;

namespace {

const std::string kLatest = "https://github.com/nyosegawa/speech.cpp/releases/latest";
constexpr int64_t kDay = 24 * 60 * 60;

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

/** The three numbers of a version X.Y.Z, or none for any other text. */
std::optional<std::array<long, 3>> version_numbers(const std::string & s) {
    std::array<long, 3> numbers{};
    size_t at = 0;
    for (size_t i = 0; i < numbers.size(); i++) {
        const size_t end = i + 1 < numbers.size() ? s.find('.', at) : s.size();
        if (end == std::string::npos) return std::nullopt;
        const std::string part = s.substr(at, end - at);
        if (part.empty() || part.size() > 9 || part.find_first_not_of("0123456789") != std::string::npos) return std::nullopt;
        numbers[i] = std::stol(part);
        at = end + 1;
    }
    return numbers;
}

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

/** What release.json holds: when github.com was last asked, and the latest release it named, or "". */
struct Kept {
    int64_t checked = 0;
    std::string latest;
};

/** release.json as kept, or none when it is missing or holds anything else. */
std::optional<Kept> read_kept() {
    try {
        const fs::path path = cache_dir() / "release.json";
        std::unique_ptr<FILE, int (*)(FILE *)> f(ggml_fopen(path.u8string().c_str(), "rb"), std::fclose);
        if (!f) return std::nullopt;
        std::string text;
        char buffer[512];
        while (const size_t n = std::fread(buffer, 1, sizeof buffer, f.get())) text.append(buffer, n);
        const JsonValue value = parse_json(text);
        const JsonValue * checked = value.member("checked");
        const JsonValue * latest = value.member("latest");
        if (!checked || !checked->is_integer() || !latest || latest->kind != JsonValue::Kind::String) return std::nullopt;
        if (!latest->text.empty() && !version_numbers(latest->text)) return std::nullopt;
        return Kept{std::stoll(checked->text), latest->text};
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

/** Writes release.json beside itself and renames it into place, so that a process never reads half of it. */
void keep(const Kept & kept) {
    const fs::path dir = cache_dir();
    std::error_code e;
    fs::create_directories(dir, e);
    if (e) throw io_failure("cannot make the folder " + dir.u8string() + ": " + e.message());
    const fs::path path = dir / "release.json";
#ifdef _WIN32
    const int pid = _getpid();
#else
    const int pid = (int) getpid();
#endif
    fs::path temporary = path;
    temporary += "." + std::to_string(pid) + ".tmp";
    const std::string text = "{\"checked\":" + std::to_string(kept.checked) + ",\"latest\":" + json_string(kept.latest) + "}\n";
    FILE * f = ggml_fopen(temporary.u8string().c_str(), "wb");
    if (!f) throw io_failure("cannot write " + temporary.u8string());
    const bool written = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (std::fclose(f) != 0 || !written) {
        fs::remove(temporary, e);
        throw io_failure("cannot write " + temporary.u8string());
    }
    fs::rename(temporary, path, e);
    if (e) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw io_failure("cannot rename " + temporary.u8string() + " to " + path.u8string() + ": " + e.message());
    }
}

void keep_quietly(const Kept & kept) {
    try {
        keep(kept);
    } catch (const std::exception &) {
    }
}

/** curl's arguments for the address the latest release redirects to, as install.sh asks it. */
std::vector<std::string> question(int seconds) {
    return {"--fail", "--silent", "--show-error", "--location", "--head", "--output", null_device(), "--write-out", "%{url_effective}",
            "--max-time", std::to_string(seconds), kLatest};
}

/** What curl gave: its exit code, its stdout and its stderr. */
struct Answer {
    int code = 0;
    std::string out, err;
};

/**
 * Runs curl to its end, reading its stdout and stderr on threads of their own while it runs: a configuration such as a
 * .curlrc with `trace = "-"` makes curl write more than a pipe holds, and curl would then wait for a reader forever.
 * Once `stop` is true, curl is ended.
 */
Answer ask(const std::vector<std::string> & args, const std::string & missing, const std::atomic<bool> & stop) {
    Curl curl(args, missing);
    Answer answer;
    std::thread out([&] { answer.out = read_to_end(curl.out()); });
    std::thread err([&] { answer.err = read_to_end(curl.err()); });
    bool stopped = false;
    while (!curl.wait(20, answer.code)) {
        if (stop && !stopped) {
            curl.stop();
            stopped = true;
        }
    }
    out.join();
    err.join();
    return answer;
}

/**
 * The version curl's answer names. --write-out writes the address the redirect ends at to stdout, among what a
 * configuration such as `trace = "-"` writes there before and after it, so the version is read after the last tag.
 */
std::string answered(Answer answer) {
    if (answer.code != 0) {
        std::string & err = answer.err;
        while (!err.empty() && (err.back() == '\n' || err.back() == '\r')) err.pop_back();
        // curl's own message is its last line; a trace before it is of no use in a message.
        const size_t line = err.find_last_of('\n');
        if (line != std::string::npos) err = err.substr(line + 1);
        throw io_failure("cannot reach github.com to find the latest release: curl's exit code " + std::to_string(answer.code) +
                         (err.empty() ? "" : " (" + err + ")") + "; check the network and run this again");
    }
    const std::string tag = "/releases/tag/v";
    const size_t at = answer.out.rfind(tag);
    const std::string rest = at == std::string::npos ? "" : answer.out.substr(at + tag.size());
    const std::string version = rest.substr(0, rest.find_first_not_of("0123456789."));
    if (!version_numbers(version)) {
        throw io_failure("github.com sends " + kLatest + " to " + answer.out.substr(0, 200) + ", which is no release's page; run this again later");
    }
    return version;
}

}  // namespace

bool is_version(const std::string & s) {
    return version_numbers(s).has_value();
}

bool later_version(const std::string & a, const std::string & b) {
    const auto x = version_numbers(a), y = version_numbers(b);
    return x && y && *x > *y;
}

std::string latest_release() {
    const std::optional<Kept> kept = read_kept();
    Kept next{now_seconds(), kept ? kept->latest : ""};
    try {
        const std::atomic<bool> never{false};
        next.latest = answered(ask(question(10),
                                   "speech asks github.com for the latest release with the system's curl, which is not there; install curl and run "
                                   "this again",
                                   never));
    } catch (const std::exception &) {
        keep_quietly(next);
        throw;
    }
    keep_quietly(next);
    return next.latest;
}

ReleaseCheck::ReleaseCheck() {
    try {
        const std::optional<Kept> kept = read_kept();
        if (kept) latest_ = kept->latest;
        const int64_t now = now_seconds();
        // A check kept with a time ahead of the clock, which was set back since, is due.
        if (kept && kept->checked <= now && now - kept->checked < kDay) return;
        thread_ = std::thread([this] { run(); });
    } catch (const std::exception &) {
    }
}

ReleaseCheck::~ReleaseCheck() {
    finish();
}

std::string ReleaseCheck::finish() {
    stopping_ = true;
    if (thread_.joinable()) thread_.join();
    return latest();
}

std::string ReleaseCheck::latest() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return latest_;
}

void ReleaseCheck::run() {
    Kept next{now_seconds(), latest()};
    try {
        next.latest = answered(ask(question(5), "there is no curl", stopping_));
        const std::lock_guard<std::mutex> lock(mutex_);
        latest_ = next.latest;
    } catch (const std::exception &) {
    }
    keep_quietly(next);
}
