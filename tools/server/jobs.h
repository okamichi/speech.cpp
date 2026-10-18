#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "log.h"
#include "regions.h"

// How the server runs requests on a model it holds: each set up by its HTTP handler as it arrives, so that the library
// refuses a value at once, then run in the order the requests arrived, each on a thread of its own that hands what it
// makes to the handler, and stopped when its client goes away.

namespace server {

class Served;

using Clock = std::chrono::steady_clock;

/**
 * Lets requests take the model in the order they arrive. The library serializes concurrent calls by itself, but in
 * no particular order, and a request must know when it is its turn to tell a client that went away while waiting
 * from one whose request is under way.
 */
class Turns {
public:
    uint64_t take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return next_++;
    }
    void wait(uint64_t ticket) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] { return serving_ == ticket; });
    }
    void pass() {
        std::lock_guard<std::mutex> lock(mutex_);
        advance();
    }
    /** Gives up a ticket that will not run, which the turns then skip, now or once the tickets before it have passed. */
    void give_up(uint64_t ticket) {
        std::lock_guard<std::mutex> lock(mutex_);
        given_up_.insert(ticket);
        if (ticket == serving_) {
            given_up_.erase(ticket);
            advance();
        }
    }

private:
    void advance() {
        serving_++;
        while (given_up_.erase(serving_)) serving_++;
        changed_.notify_all();
    }

    std::mutex mutex_;
    std::condition_variable changed_;
    uint64_t next_ = 0, serving_ = 0;
    std::set<uint64_t> given_up_;
};

/**
 * A turn on a model, taken when the request is accepted, so that requests run in the order they arrived whatever thread
 * runs them and when: wait() waits for it, and letting it go passes it once waited for, or gives it up otherwise.
 */
class Turn {
public:
    explicit Turn(Turns & turns) : turns_(turns), ticket_(turns.take()) {}
    ~Turn() {
        if (waited_) turns_.pass();
        else turns_.give_up(ticket_);
    }
    Turn(const Turn &) = delete;
    Turn & operator=(const Turn &) = delete;

    void wait() {
        turns_.wait(ticket_);
        waited_ = true;
    }

private:
    Turns & turns_;
    uint64_t ticket_;
    bool waited_ = false;
};

/** What every request's run shares: the library's request, its state, which the HTTP handler waits on, and its client's departure. */
struct Job {
    /** The HTTP request's method, path and client, which its log line gives. */
    std::string method, path, client;
    /** The model the request runs on, which the server keeps until the request has ended and been freed. */
    std::shared_ptr<Served> served;
    /**
     * The request as its handler set it up, where the run makes one alone, which the run spends and a client that goes
     * away cancels.
     */
    Request request{nullptr, speech_request_free};
    /**
     * Stops the library requests the run makes: its request's, those of the regions of a transcription, or those of the
     * sentences of a speech.
     */
    Cancellation cancellation;
    std::mutex mutex;
    std::condition_variable changed;
    bool finished = false;
    bool abandoned = false;
    /** What the run did, for the request's log line, once it has finished. */
    std::string what;
    /** The handler has returned before the run finished, so the run writes the request's log line. */
    bool handler_left = false;
    speech_status status = SPEECH_OK;
    /** The library's failure of a run that failed. */
    std::optional<Failure> failure;
    Clock::time_point arrived = Clock::now(), gone;

    /** Stops the request for a client that went away, whether it waits or runs. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!abandoned) gone = Clock::now();
        abandoned = true;
        cancellation.cancel();
    }
};

/**
 * One request's recognition, whose result the HTTP handler reads from the request once it is done; or with
 * chunking_strategy, a recognition of each region the detection request finds, joined.
 */
struct TranscriptionJob : Job {
    /** The length of the audio in seconds and its rate, for the log and verbose_json. */
    double duration = 0;
    int sample_rate = 0;
    /** With chunking_strategy: the detection model, kept as `served` is, and its request set up with the audio. */
    std::shared_ptr<Served> detector;
    Request detection{nullptr, speech_request_free};
    /** With chunking_strategy: the model of `served`, the audio, the recognition options and whether they set timestamps. */
    speech_model * recognizer = nullptr;
    std::vector<float> samples;
    std::vector<RequestOption> options;
    bool timestamps = false;
    size_t regions = 0;
    /** The recognition once it is done. */
    Transcript transcript;
};

/** The interval at which a waiting handler looks whether its client is still there. */
constexpr auto POLL = std::chrono::milliseconds(50);

/**
 * Waits, holding `lock` on the job's mutex, until the job has finished or `until` says so; returns false for a client
 * that went away meanwhile, as `gone` tells, whose request it then cancels.
 */
template <typename Until, typename Gone>
bool wait_for(Job & job, std::unique_lock<std::mutex> & lock, Until until, Gone gone) {
    while (!job.finished && !until()) {
        if (job.changed.wait_for(lock, POLL) == std::cv_status::timeout && gone()) {
            lock.unlock();
            job.abandon();
            return false;
        }
    }
    return true;
}

inline double seconds_between(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double>(t1 - t0).count();
}

/**
 * Hands the request's log line over as its handler returns: to cpp-httplib's logger, with what the run did, where the
 * run has finished, and otherwise to the run, which writes the line as it ends.
 */
class LogLine {
public:
    explicit LogLine(Job & job) : job_(job) {}
    ~LogLine() {
        std::lock_guard<std::mutex> lock(job_.mutex);
        if (job_.finished) {
            note_request(job_.what);
        } else {
            job_.handler_left = true;
            note_request("", true);
        }
    }
    LogLine(const LogLine &) = delete;
    LogLine & operator=(const LogLine &) = delete;

private:
    Job & job_;
};

/**
 * Runs a request on the model in its turn, on a thread of its own, unless its client went away while it waited, and
 * says what it did for the request's log line (LogLine): `name` begins it, `run` runs the request through the job's
 * cancellation and returns its status or throws the library's Failure, and `outcome`, called with the job locked, says
 * what the run made.
 */
template <typename Run, typename Outcome>
void run_in_turn(Job & job, Turns & turns, uint64_t ticket, const std::string & name, Run run, Outcome outcome) {
    turns.wait(ticket);
    const Clock::time_point started = Clock::now();
    bool ran = false;
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        ran = !job.abandoned;
    }
    speech_status status = SPEECH_CANCELLED;
    std::optional<Failure> failure;
    if (ran) {
        try {
            status = run();
        } catch (const Failure & e) {
            status = SPEECH_ERROR_INTERNAL;
            failure = e;
        }
    }
    {
        std::lock_guard<std::mutex> lock(job.mutex);
        job.finished = true;
        job.status = status;
        job.failure = failure;
        job.changed.notify_all();
        const Clock::time_point now = Clock::now();
        char line[256];
        std::snprintf(line, sizeof line, ": waited %.3f s", seconds_between(job.arrived, started));
        std::string what = name + line;
        if (!ran) {
            what += ", not started: the client went away while it waited";
        } else {
            what += outcome(started);
            std::snprintf(line, sizeof line, " in %.3f s", seconds_between(started, now));
            what += line;
            if (failure) what += ", failed: " + failure->code() + ": " + failure->what();
            if (job.abandoned) {
                std::snprintf(line, sizeof line, ", stopped %.3f s after the client went away", seconds_between(job.gone, now));
                what += line;
            }
        }
        job.what = what;
        // A stream's status, sent before the run, is 200, which a failure breaks off.
        const std::string ended = !ran || job.abandoned ? "gone" : failure ? "failed" : status == SPEECH_OK ? "200" : speech_status_name(status);
        if (job.handler_left) log_request(job.method, job.path, ended, what, job.client);
    }
    turns.pass();
}

/** Runs a transcription, by the regions of its detection request where it has one, and keeps its recognition in the job. */
inline void transcribe(const std::shared_ptr<TranscriptionJob> & job, Turns & turns, uint64_t ticket) {
    char name[256];
    std::snprintf(name, sizeof name, "%.2f s of audio at %d Hz", job->duration, job->sample_rate);
    run_in_turn(*job, turns, ticket, name, [&] {
        if (!job->detection) {
            speech_request * r = job->request.get();
            const speech_status s = check(job->cancellation.run(r, [&] { return speech_transcribe(r); }));
            if (s == SPEECH_OK) job->transcript = transcript_of(speech_request_result(r), job->timestamps);
            return s;
        }
        const std::optional<std::vector<Region>> regions = detect_regions(job->detection.get(), job->cancellation);
        if (!regions) return SPEECH_CANCELLED;
        job->regions = regions->size();
        std::optional<Transcript> t =
            transcribe_regions(job->recognizer, job->samples, job->sample_rate, *regions, job->options, job->timestamps, job->cancellation);
        if (!t) return SPEECH_CANCELLED;
        job->transcript = std::move(*t);
        return SPEECH_OK;
    }, [&](Clock::time_point) {
        std::string out = job->detection ? ", " + std::to_string(job->regions) + " regions" : "";
        return job->status == SPEECH_OK ? out + ", " + std::to_string(job->transcript.text.size()) + " bytes of text" : out;
    });
}

}  // namespace server
