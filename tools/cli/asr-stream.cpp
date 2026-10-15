#include "asr-stream.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "microphone.h"
#include "realtime-events.h"
#include "regions.h"
#include "utf8.h"
#include "utterances.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/**
 * The samples a stream holds before they are recognized, its buffer and the utterances waiting (Assembly::held()), as
 * speech serve bounds the audio of a Realtime session: 25 MiB of 16-bit PCM, 13.6 min at 16 kHz and 4.5 min at 48 kHz.
 */
constexpr size_t kHeld = (25 << 20) / 2;

/**
 * The failure of a stream whose audio held has passed kHeld: the detection's options, which keep that much uncommitted
 * where no recognition waits to free any, or else a recognition that has fallen behind the audio.
 */
Failure past_the_bound(const utterances::Assembly & assembly, int rate) {
    char message[320];
    if (assembly.committed() == 0) {
        std::snprintf(message, sizeof message,
                      "the detection's options keep %.0f s of audio before an utterance is committed, more than the %.0f s speech asr holds; "
                      "lower --speech-pad-ms or --max-speech-duration-s",
                      (double) assembly.held() / rate, (double) kHeld / rate);
    } else {
        std::snprintf(message, sizeof message,
                      "the recognition has fallen %.0f s of audio behind, the most speech asr holds; recognize on a faster device or with a faster "
                      "model",
                      (double) assembly.held() / rate);
    }
    return Failure(speech_status_name(SPEECH_ERROR_OUT_OF_MEMORY), "", message);
}

std::atomic<bool> interrupted{false};

/** The first Ctrl-C ends the audio, after which the utterances heard are written; a second ends the program. */
void on_interrupt(int) {
    interrupted = true;
    std::signal(SIGINT, SIG_DFL);
}

/** Recognizes on the command line's model, and times each utterance's recognition for -v. */
class ModelRecognizer : public utterances::Recognizer {
public:
    explicit ModelRecognizer(speech_model * model) : model_(model) {}

    /** The command line's model serves the assembly alone, whose recognitions run one at a time in their order. */
    std::unique_ptr<utterances::Reservation> reserve(const utterances::Asked &) override {
        return std::make_unique<utterances::Reservation>();
    }

    std::optional<Transcript> transcribe(utterances::Reservation &, const utterances::Asked & asked, std::vector<float> samples, int rate,
                                         Cancellation & cancellation, uint64_t utterance, bool provisional, const std::string & draft) override {
        const auto t0 = Clock::now();
        const Request request = new_request(model_);
        check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
        apply_options(request.get(), asked.options);
        apply_draft(request.get(), model_, draft);
        if (check(cancellation.run(request.get(), [&] { return speech_transcribe(request.get()); })) == SPEECH_CANCELLED) return std::nullopt;
        std::lock_guard<std::mutex> lock(mutex_);
        if (provisional) readings_++;
        else took_[utterance] = seconds_since(t0);
        return transcript_of(speech_request_result(request.get()), false);
    }

    double took(uint64_t utterance) {
        std::lock_guard<std::mutex> lock(mutex_);
        return took_[utterance];
    }

    size_t readings() {
        std::lock_guard<std::mutex> lock(mutex_);
        return readings_;
    }

private:
    speech_model * model_;
    std::mutex mutex_;
    std::map<uint64_t, double> took_;
    size_t readings_ = 0;
};

/** The columns a terminal draws a code point in: none for a combining mark, two for a wide East Asian character. */
int columns_of(uint32_t cp) {
    if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x200B && cp <= 0x200F) || (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE00 && cp <= 0xFE0F) ||
        (cp >= 0xFE20 && cp <= 0xFE2F)) {
        return 0;
    }
    const bool wide = (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0x303E) || (cp >= 0x3041 && cp <= 0x33FF) ||
                      (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xA000 && cp <= 0xA4CF) ||
                      (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) ||
                      (cp >= 0xFF00 && cp <= 0xFF60) || (cp >= 0xFFE0 && cp <= 0xFFE6) || (cp >= 0x1F300 && cp <= 0x1F64F) ||
                      (cp >= 0x1F900 && cp <= 0x1F9FF) || (cp >= 0x20000 && cp <= 0x3FFFD);
    return wide ? 2 : 1;
}

/**
 * Writes each utterance's text as a line, and on a terminal the agreed text of the utterance under way on the current
 * line, its end where it is longer than the line, until the utterance's text replaces it.
 */
class TextOutput {
public:
    explicit TextOutput(FILE * out) : out_(out) {
#ifdef _WIN32
        // A console draws "\r" and the erasing of a line only with its virtual terminal processing on.
        const HANDLE console = (HANDLE) _get_osfhandle(_fileno(out));
        DWORD mode = 0;
        terminal_ = GetConsoleMode(console, &mode) && SetConsoleMode(console, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
        terminal_ = isatty(fileno(out));
#endif
    }

    void delta(const std::string & text) {
        if (!terminal_) return;
        shown_ += text;
        std::fputs(("\r\033[K" + tail(shown_, columns() - 1)).c_str(), out_);
        std::fflush(out_);
    }

    void line(const std::string & text) {
        std::fputs(((terminal_ && !shown_.empty() ? "\r\033[K" : "") + text + "\n").c_str(), out_);
        std::fflush(out_);
        shown_.clear();
    }

private:
    int columns() const {
#ifdef _WIN32
        CONSOLE_SCREEN_BUFFER_INFO info;
        if (GetConsoleScreenBufferInfo((HANDLE) _get_osfhandle(_fileno(out_)), &info)) return info.srWindow.Right - info.srWindow.Left + 1;
#else
        winsize size{};
        if (ioctl(fileno(out_), TIOCGWINSZ, &size) == 0 && size.ws_col > 0) return size.ws_col;
#endif
        return 80;
    }

    /** The end of `text` that fits in `width` columns, after an ellipsis where it does not fit whole. */
    static std::string tail(const std::string & text, int width) {
        const std::vector<uint32_t> cps = decode_utf8(text).value_or(std::vector<uint32_t>());
        int whole = 0;
        for (const uint32_t cp : cps) whole += columns_of(cp);
        if (whole <= width) return text;
        int used = 0;
        size_t first = cps.size();
        while (first > 0 && used + columns_of(cps[first - 1]) <= width - 1) used += columns_of(cps[--first]);
        return "\xE2\x80\xA6" + encode_utf8(std::vector<uint32_t>(cps.begin() + (long) first, cps.end()));
    }

    FILE * out_;
    bool terminal_ = false;
    std::string shown_;
};

/**
 * stdin read on a thread of its own, what has arrived without waiting for a whole buffer, at most `ahead` bytes before
 * what the caller has taken, so that the caller notices a failure of the recognition while stdin stays open and silent.
 * The thread is let go rather than joined, since it may wait in read() for as long as the writer keeps stdin open; what it
 * shares with the caller outlives both.
 */
class StdinReader {
public:
    explicit StdinReader(size_t ahead) : shared_(std::make_shared<Shared>()) {
        shared_->ahead = ahead;
        std::thread([shared = shared_] { read_all(*shared); }).detach();
    }

    /** What has arrived since the last call, waiting up to `wait` while nothing has. */
    std::string take(std::chrono::milliseconds wait) {
        std::unique_lock<std::mutex> lock(shared_->mutex);
        shared_->changed.wait_for(lock, wait, [&] { return !shared_->bytes.empty() || shared_->ended; });
        std::string out;
        out.swap(shared_->bytes);
        shared_->changed.notify_all();
        return out;
    }

    /** Whether stdin has ended and every byte of it was taken; a failure to read it throws. */
    bool ended() {
        std::lock_guard<std::mutex> lock(shared_->mutex);
        if (!shared_->error.empty()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "audio", shared_->error);
        return shared_->ended && shared_->bytes.empty();
    }

private:
    struct Shared {
        std::mutex mutex;
        std::condition_variable changed;
        std::string bytes, error;
        bool ended = false;
        size_t ahead = 0;
    };

    static void read_all(Shared & s) {
        std::vector<char> buffer(8192);
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(s.mutex);
                s.changed.wait(lock, [&] { return s.bytes.size() < s.ahead; });
            }
#ifdef _WIN32
            const int got = _read(0, buffer.data(), (unsigned) buffer.size());
#else
            const ssize_t got = read(0, buffer.data(), buffer.size());
#endif
            if (got < 0 && errno == EINTR) continue;
            std::lock_guard<std::mutex> lock(s.mutex);
            if (got < 0) s.error = std::string("cannot read stdin: ") + std::strerror(errno);
            if (got > 0) s.bytes.append(buffer.data(), (size_t) got);
            s.ended = got <= 0;
            s.changed.notify_all();
            if (s.ended) return;
        }
    }

    std::shared_ptr<Shared> shared_;
};

}  // namespace

int transcribe_stream(const CommandLine & line, FILE * out, speech_model * recognizer, const std::vector<RequestOption> & options,
                      speech_model * detector, const std::vector<RequestOption> & detection, bool json) {
    // The recognition's options are refused before any audio, as a file's are, where they would otherwise wait for the
    // first utterance, which silence never brings.
    check_recognition(recognizer, options);
    const bool live = line.has("--live");
    std::unique_ptr<Microphone> microphone;
    if (live) microphone = std::make_unique<Microphone>();
    const int rate = live ? microphone->rate() : line.integer("--rate").value_or(16000);
    const bool verbose = line.has("-v");

    ModelRecognizer recognition(recognizer);
    TextOutput text(out);
    realtime::Events events([out](const std::string & event) {
        std::fprintf(out, "%s\n", event.c_str());
        std::fflush(out);
    });
    std::mutex mutex;
    std::exception_ptr failure;
    bool model_limit = false;
    size_t completed = 0;
    std::map<uint64_t, std::pair<double, double>> times;
    const auto failed = [&] {
        std::lock_guard<std::mutex> lock(mutex);
        return failure != nullptr;
    };
    utterances::Assembly assembly(recognition, rate, [&](const utterances::Event & e) {
        using Kind = utterances::Event::Kind;
        if (json) events.utterance(e);
        else if (e.kind == Kind::Delta) text.delta(e.delta);
        else if (e.kind == Kind::Completed) text.line(e.transcript.text);
        std::lock_guard<std::mutex> lock(mutex);
        if (e.kind == Kind::SpeechStarted) times[e.utterance].first = e.at;
        if (e.kind == Kind::SpeechStopped) times[e.utterance].second = e.at;
        if (e.kind == Kind::Failed && !failure) failure = e.failure;
        if (e.kind != Kind::Completed) return;
        completed++;
        if (e.transcript.stop == SPEECH_STOP_MODEL_LIMIT) model_limit = true;
        if (verbose) {
            std::fprintf(stderr, "%.2f to %.2f s: recognized in %.3f s%s\n", times[e.utterance].first, times[e.utterance].second,
                         recognition.took(e.utterance), e.transcript.stop == SPEECH_STOP_MODEL_LIMIT ? ", stopped at the most the model writes" : "");
        }
        times.erase(e.utterance);
    });
    assembly.ask(utterances::Asked{"", options});
    assembly.detect(std::make_unique<utterances::Detection>(detector, nullptr, detection, rate));

    if (live) {
        std::fprintf(stderr, "listening to %s at %d Hz; Ctrl-C ends\n", microphone->name().c_str(), rate);
        interrupted = false;
        std::signal(SIGINT, on_interrupt);
        // A microphone always hears some noise, so audio of nothing but zeros for seconds is one the system keeps from
        // the program, as macOS does without the terminal's permission to use the microphone.
        double silent = 0;
        bool warned = false;
        while (!interrupted && !failed()) {
            const std::vector<float> samples = microphone->take(std::chrono::milliseconds(50));
            bool zeros = true;
            for (const float s : samples) zeros = zeros && s == 0;
            silent = zeros ? silent + (double) samples.size() / rate : 0;
            if (silent >= 3 && !warned) {
                std::fprintf(stderr, "the microphone has given nothing but zeros for 3 s: check that it is not muted, and on macOS that the terminal "
                                     "may use it (System Settings, Privacy & Security, Microphone)\n");
                warned = true;
            }
            assembly.push(samples.data(), samples.size());
            // The microphone cannot wait, and audio is not dropped, so audio held past the bound ends the run.
            if (assembly.held() > kHeld) throw past_the_bound(assembly, rate);
        }
        microphone.reset();
        std::signal(SIGINT, SIG_DFL);
    } else {
        std::fprintf(stderr, "reading 16-bit PCM at %d Hz from stdin\n", rate);
        StdinReader reader(1 << 20);
        std::string pending;
        std::vector<float> samples;
        while (!failed()) {
            // stdin waits while the audio held is at its bound and recognitions to come will free some of it, so that a fast
            // writer is held back to the recognition's pace; audio the options keep uncommitted is freed only by more audio.
            // held() is read again once committed() is 0: a recognition that ends between the two reads frees audio, and
            // only this thread commits more.
            if (assembly.held() >= kHeld) {
                if (assembly.committed() == 0 && assembly.held() >= kHeld) throw past_the_bound(assembly, rate);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                continue;
            }
            const std::string got = reader.take(std::chrono::milliseconds(50));
            if (got.empty()) {
                if (reader.ended()) break;
                continue;
            }
            pending += got;
            samples.resize(pending.size() / 2);
            for (size_t i = 0; i < samples.size(); i++) {
                samples[i] = (float) (int16_t) ((uint16_t) (unsigned char) pending[2 * i] | (uint16_t) (unsigned char) pending[2 * i + 1] << 8) / 32768.0f;
            }
            pending.erase(0, 2 * samples.size());
            assembly.push(samples.data(), samples.size());
        }
        if (!pending.empty() && !failed()) {
            throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "audio",
                          "stdin ended within a sample: 16-bit PCM has two bytes for each, and an odd number arrived");
        }
    }
    if (!failed()) assembly.end();
    if (failed()) std::rethrow_exception(failure);
    std::fprintf(stderr, "%zu utterance%s in %.2f s of audio, read %zu times while they went on; the detection took %.3f s\n", completed,
                 completed == 1 ? "" : "s", assembly.heard(), recognition.readings(), assembly.detecting());
    if (model_limit) {
        std::fprintf(stderr, "a recognition reached the most the model writes and was stopped there; its text is what was written\n");
        return 3;
    }
    return 0;
}
