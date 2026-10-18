#include "audio-speech.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "jobs.h"
#include "openai-api.h"
#include "sentences.h"

namespace server {

namespace {

using openai::ApiError;
using openai::send_error;

/** The audio of a request as the worker sends it: 16-bit little-endian samples, clamped to [-1, 1] and rounded. */
void append_pcm(std::string & out, const float * s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const int16_t v = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
        out += (char) (v & 0xFF);
        out += (char) ((uint16_t) v >> 8);
    }
}

/** One speech, which hands its audio to the HTTP handler as it is made. */
struct SpeechJob : Job {
    /** The model's sample rate, which the log gives the length in. */
    int sample_rate = 0;
    /** The text and the options, the seed among them, which the handler has had the library check. */
    std::string text;
    std::vector<RequestOption> options;
    /** The PCM made and not yet sent. */
    std::string pending;
    /** A callback has been called, so the library has accepted a request and its work has begun. */
    bool accepted = false;
    uint64_t samples = 0;
    int64_t seed = -1;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    size_t requests = 0;
    uint64_t limit_samples = 0;
    bool instructions_truncated = false;
    Clock::time_point first_audio;
};

int on_audio(const float * s, size_t n, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    if (job.samples == 0) job.first_audio = Clock::now();
    append_pcm(job.pending, s, n);
    job.samples += n;
    job.changed.notify_all();
    return 0;
}

/** The library reports progress once it has checked a request, so a request that reports any is accepted. */
int on_progress(double, void * user_data) {
    SpeechJob & job = *static_cast<SpeechJob *>(user_data);
    std::lock_guard<std::mutex> lock(job.mutex);
    job.accepted = true;
    job.changed.notify_all();
    return 0;
}

void synthesize(const std::shared_ptr<SpeechJob> & job, Turns & turns, uint64_t ticket, const std::string & name) {
    run_in_turn(*job, turns, ticket, name, [&] {
        const std::optional<Spoken> spoken =
            speak_text(job->served->get(), job->text, job->options, job->limit_samples ? Split::Compatible : Split::Sentences, on_audio, on_progress, job.get(), job->cancellation, job->limit_samples);
        if (!spoken) return SPEECH_CANCELLED;
        std::lock_guard<std::mutex> lock(job->mutex);
        job->seed = spoken->seed;
        job->stop = spoken->stop;
        job->requests = spoken->requests;
        job->instructions_truncated = spoken->instructions_truncated;
        return SPEECH_OK;
    }, [&](Clock::time_point started) {
        if (!job->samples) return std::string();
        char line[256];
        std::snprintf(line, sizeof line, ", time to first audio %.3f s, %.2f s of audio", seconds_between(started, job->first_audio),
                      (double) job->samples / job->sample_rate);
        return std::string(line) + (job->requests > 1 ? " from " + std::to_string(job->requests) + " requests" : "") +
               (job->instructions_truncated ? ", instructions cut to the model's token limit" : "");
    });
}

std::string wav_header(size_t data_bytes, int sample_rate) {
    std::string h;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) h += (char) ((v >> (8 * i)) & 0xFF); };
    auto u16 = [&](uint16_t v) { h += (char) (v & 0xFF); h += (char) (v >> 8); };
    h += "RIFF";
    u32((uint32_t) (36 + data_bytes));
    h += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(1);
    u32((uint32_t) sample_rate);
    u32((uint32_t) sample_rate * 2);
    u16(2);
    u16(16);
    h += "data";
    u32((uint32_t) data_bytes);
    return h;
}

/** A seed from 0 to 2^53 - 1, the range the library draws from, for a request that sets none. */
int64_t draw_seed() {
    std::random_device device;
    return (int64_t) (((uint64_t) device() << 32 | device()) & ((1ull << 53) - 1));
}

/**
 * Sends the audio as it is made. A failure after the stream has begun ends a pcm stream without its last chunk,
 * which the client reads as a broken transfer, and an SSE stream with an error event.
 */
bool stream(SpeechJob & job, bool sse, httplib::DataSink & sink) {
    std::unique_lock<std::mutex> lock(job.mutex);
    for (;;) {
        if (!job.pending.empty()) {
            std::string pcm;
            pcm.swap(job.pending);
            lock.unlock();
            const std::string out = sse ? openai::sse_delta(pcm) : pcm;
            // A write to a socket the client has closed succeeds once more before it fails, so a client that went
            // away is noticed a chunk earlier by looking first.
            if (!sink.is_writable() || !sink.write(out.data(), out.size())) {
                job.abandon();
                return false;
            }
            lock.lock();
            continue;
        }
        if (job.finished) break;
        if (job.changed.wait_for(lock, POLL) == std::cv_status::timeout && !job.finished && job.pending.empty()) {
            lock.unlock();
            if (!sink.is_writable()) {
                job.abandon();
                return false;
            }
            lock.lock();
        }
    }
    const speech_status status = job.status;
    const std::optional<Failure> failure = job.failure;
    const int64_t seed = job.seed;
    const uint64_t samples = job.samples;
    const speech_stop stop = job.stop;
    lock.unlock();
    if (status != SPEECH_OK && !sse) return false;
    if (sse) {
        std::string out;
        if (status == SPEECH_OK) out = openai::sse_done(seed, samples, speech_stop_name(stop));
        else if (failure) out = openai::sse_error(openai::library_error(*failure));
        else return false;
        if (!sink.write(out.data(), out.size())) return false;
    }
    sink.done();
    return true;
}

}  // namespace

void answer_speech(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res, bool compatible) {
    const auto info = served->info();
    const int sample_rate = speech_model_info_sample_rate(info.get());
    openai::SpeechRequest asked;
    try {
        asked = openai::read_speech_request(req.body, speech_model_info_name(info.get()), compatible);
    } catch (const ApiError & e) {
        send_error(res, e);
        return;
    }
    asked.options = served->with_defaults(asked.options);
    auto job = std::make_shared<SpeechJob>();
    job->sample_rate = sample_rate;
    if (asked.max_seconds) {
        job->limit_samples = (uint64_t) std::floor(asked.max_seconds * sample_rate);
        if (!job->limit_samples) {
            send_error(res, {400, "max_seconds must allow at least one audio sample.", "irodori.max_seconds", "invalid_value"});
            return;
        }
    }
    std::optional<int64_t> seed;
    std::string voice;
    for (const RequestOption & o : asked.options) {
        if (o.option == SPEECH_OPT_SEED) seed = std::get<int64_t>(o.value);
        if (o.option == SPEECH_OPT_VOICE) voice = std::get<std::string>(o.value);
    }
    // A pcm stream's headers leave before its result, so the server draws the seed the library would draw.
    if (!seed && speech_model_info_takes(info.get(), SPEECH_OPT_SEED)) {
        seed = draw_seed();
        asked.options.push_back({SPEECH_OPT_SEED, *seed});
    }
    job->served = served;
    try {
        // The library checks each value as it is set, so a request it refuses is answered before it waits for its turn.
        const Request checked = new_request(served->get());
        check(speech_request_set_text(checked.get(), asked.input.c_str()));
        apply_options(checked.get(), asked.options);
    } catch (const Failure & e) {
        send_error(res, openai::library_error(e));
        return;
    }
    job->text = std::move(asked.input);
    job->options = asked.options;
    char name[256];
    std::snprintf(name, sizeof name, "voice %s, seed %lld", voice.c_str(), (long long) seed.value_or(-1));
    job->method = req.method;
    job->path = req.path;
    job->client = req.remote_addr;
    const uint64_t ticket = served->turns().take();
    std::thread(synthesize, job, std::ref(served->turns()), ticket, std::string(name)).detach();
    const LogLine log_line(*job);

    // A wav is sent whole, so it waits for the end; a stream waits until the library has begun a request's work, so
    // that a request it refuses is answered with an error status rather than a stream that breaks off.
    const bool whole = asked.format == "wav";
    {
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait_for(*job, lock, [&] { return !whole && job->accepted; }, [&] { return req.is_connection_closed(); })) return;
        if (job->finished && job->status == SPEECH_CANCELLED) return;
        if (job->finished && job->failure && (whole || !job->accepted)) {
            send_error(res, openai::library_error(*job->failure));
            return;
        }
    }
    res.set_header("X-Sample-Rate", std::to_string(sample_rate));
    if (seed) res.set_header("X-Speech-Seed", std::to_string(*seed));
    if (whole) {
        res.set_header("X-Speech-Stop", speech_stop_name(job->stop));
        res.set_header("X-Speech-Instructions-Truncated", job->instructions_truncated ? "true" : "false");
        res.set_content(wav_header(job->pending.size(), sample_rate) + job->pending, "audio/wav");
        return;
    }
    const bool sse = asked.sse;
    if (sse) res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        sse ? "text/event-stream" : "audio/pcm", [job, sse](size_t, httplib::DataSink & sink) { return stream(*job, sse, sink); },
        [job](bool success) {
            if (!success) job->abandon();
        });
}

}  // namespace server
