#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"
#include "transcript.h"

// Utterance assembly: audio that arrives a piece at a time, at a rate given when the assembly starts, cut into
// utterances, each recognized alone, as transcription by regions recognizes a whole recording. With a detection, the
// utterances are the regions where someone speaks, which the detection gives sample for sample as speech_detect() gives
// them for the whole audio; without one, the caller commits each utterance, all the audio appended since the last
// commit. While an utterance goes on, its beginning is read again and again, and the beginning that two readings in a
// row agree on goes out as deltas, which only ever add text; once it is committed, it is recognized whole, and that
// recognition, which may differ from the deltas, completes it. The events go out in order: an utterance's speech started,
// its deltas, its speech stopped and its commit, then its completion, with the next utterance's events begun meanwhile.

namespace utterances {

/** What a recognition is asked: the name of the model, or "" for the one the host holds, and the request options. */
struct Asked {
    std::string model;
    std::vector<RequestOption> options;
};

/** An event of the assembly, about one utterance. */
struct Event {
    enum class Kind { SpeechStarted, SpeechStopped, Committed, Delta, Completed, Failed };
    Kind kind = Kind::Committed;
    /** The utterance, numbered from 0 in the order the utterances began. */
    uint64_t utterance = 0;
    /** SpeechStarted: the utterance's start; SpeechStopped: its end; in seconds from the start of the audio. */
    double at = 0;
    /** Committed: whether its completion or failure follows, which it does unless recognitions are not asked. */
    bool recognized = false;
    /** Delta: the text that follows what the utterance's deltas gave before. */
    std::string delta;
    /** Completed: the recognition of the whole utterance, and the utterance's length in seconds. */
    Transcript transcript;
    double duration = 0;
    /** Failed: what the recognizer threw. */
    std::exception_ptr failure;
};

/**
 * A recognition's place among the host's requests, taken when the recognition is accepted, so that it runs before any
 * request accepted later, and kept until it has run or is dropped.
 */
class Reservation {
public:
    virtual ~Reservation() = default;
};

/** The recognizer of an assembly, which its host gives. */
class Recognizer {
public:
    virtual ~Recognizer() = default;
    /** Takes the place of a recognition asked as `asked`; it throws what the host refuses, as transcribe() does. */
    virtual std::unique_ptr<Reservation> reserve(const Asked & asked) = 0;
    /**
     * Recognizes `samples` at `rate` as `asked`, in the place `reservation` took, or returns nothing once `cancellation`
     * stops it, and throws what it refuses. The samples are its own, to let go once the library's request has copied
     * them. `utterance` is the utterance's number, and `provisional` says whether the samples are its beginning, read
     * while it goes on, or all of it. `draft` is the text the utterance's last reading gave, or "", which a model that
     * takes the option draft checks as the text it is likely to write.
     */
    virtual std::optional<Transcript> transcribe(Reservation & reservation, const Asked & asked, std::vector<float> samples, int rate,
                                                 Cancellation & cancellation, uint64_t utterance, bool provisional,
                                                 const std::string & draft) = 0;
};

/**
 * A detection of audio given a piece at a time (speech_detection), on a detection model that `keep` keeps until the
 * detection is freed, with the request options it was started with.
 */
class Detection {
public:
    /**
     * Starts a detection of audio at `rate` with `options`, which the library checks; its refusal throws a Failure, and so
     * does a rate below 8000 Hz, whose resampler holds back more audio than the assembly keeps for a region to come.
     */
    Detection(speech_model * model, std::shared_ptr<const void> keep, std::vector<RequestOption> options, int rate);

    int rate() const { return rate_; }
    const std::vector<RequestOption> & options() const { return options_; }
    /** What keeps the model, which tells the model a host holds now from the one the detection runs on. */
    const void * model() const { return keep_.get(); }
    /**
     * How far before the audio heard a region not yet begun may start, its padding included, so that the audio before
     * that, outside every region, is no longer needed.
     */
    double reach_back() const { return reach_back_; }

    void push(const float * samples, size_t n);
    void end();
    size_t region_count() const;
    /** The region at `index`, in seconds from the start of the detection's audio. */
    std::pair<double, double> region(size_t index) const;
    /**
     * The start of the region begun and not given yet, and whether it is certain to be given, as
     * speech_detection_speaking() gives them, or nothing.
     */
    std::optional<std::pair<double, bool>> speaking() const;
    /** Starts again, on the audio that follows, with the same model and options. */
    void restart();

private:
    void start();

    speech_model * model_;
    std::shared_ptr<const void> keep_;
    std::vector<RequestOption> options_;
    int rate_;
    double reach_back_ = 0;
    std::unique_ptr<speech_detection, decltype(&speech_detection_free)> detection_{nullptr, speech_detection_free};
};

/** One thread at a time calls an assembly's functions; its recognitions run on a thread of its own. */
class Assembly {
public:
    /**
     * An assembly of audio at `rate`, whose utterances the caller commits until it is given a detection. `emit` takes
     * each event, one at a time, on the caller's thread or the assembly's, and must not call the assembly.
     */
    Assembly(Recognizer & recognizer, int rate, std::function<void(const Event &)> emit);
    /** Stops the recognition under way and drops those that wait; no event follows. */
    ~Assembly();
    Assembly(const Assembly &) = delete;
    Assembly & operator=(const Assembly &) = delete;

    /** How the utterances committed from now on, and the one under way, are recognized; nothing, for none to be. */
    void ask(std::optional<Asked> asked);
    /**
     * Cuts the audio that follows by the regions of `detection`, of the assembly's rate, or waits without cutting while it
     * is null. A detection under way is ended first, and the regions it gives at its end are committed.
     */
    void detect(std::unique_ptr<Detection> detection);
    /** Leaves the commits to the caller from now on, ending a detection under way as detect() does. */
    void manual();
    /** The detection under way, or nullptr. */
    const Detection * detection() const { return detection_.get(); }

    /** Takes the next `n` samples. */
    void push(const float * samples, size_t n);
    /**
     * Commits the buffer as an utterance, the one under way, and with a detection starts it again on the audio that
     * follows; false when the buffer is empty. The buffer is the audio taken since the last commit or clear, but for
     * what a detection has found to lie outside every region still to come.
     */
    bool commit();
    /** Drops the buffer and the utterance under way, and starts a detection again. */
    void clear();
    /** Ends the audio, commits what a detection gives at its end, and returns once every utterance is recognized. */
    void end();

    /**
     * The samples the assembly holds: the buffer and the utterances committed and not yet recognized. A reading holds
     * none of its own once it has handed its copy to the library's request.
     */
    size_t held() const;
    /** The samples of held() that are utterances committed and not yet recognized, which their recognitions free. */
    size_t committed() const;
    /** The seconds of audio taken, and the seconds the detections took to take them. */
    double heard() const;
    double detecting() const { return detecting_; }

private:
    /** An utterance under way. */
    struct Utterance {
        uint64_t number = 0;
        /** Its first sample, counted from the start of the audio. */
        uint64_t start = 0;
        /** Whether its speech started has gone out, before which its deltas wait. */
        bool announced = false;
        /** The sample up to which the last reading read. */
        uint64_t read_to = 0;
        /** The last reading's text, what it and the reading before agree on, and what the deltas have given. */
        std::string last_reading;
        std::string agreed;
        std::string sent;
    };

    /** A committed utterance that waits for its recognition, with its place or what refused it one. */
    struct Final {
        uint64_t number = 0;
        /** What its deltas gave while it was said, which the rest of its final text follows as one more delta. */
        std::string sent;
        /** The text of its last reading, which its recognition checks as a draft. */
        std::string draft;
        std::vector<float> samples;
        /** The samples' count, which counts toward held() until the recognition has ended. */
        size_t length = 0;
        Asked asked;
        std::unique_ptr<Reservation> reservation;
        std::exception_ptr refused;
    };

    void work();
    // These run with the mutex held.
    void take_regions();
    void announce(Utterance & u);
    void send_agreed(Utterance & u);
    void commit_samples(uint64_t number, uint64_t first, uint64_t last);
    /** Moves the buffer's start to `sample`, dropping the audio before it. */
    void keep_from(uint64_t sample);
    void drop_current();
    void end_detection();
    void restart_detection();
    void begin_manual_utterance();
    bool reading_due() const;
    void send(const Event & e);

    Recognizer & recognizer_;
    const int rate_;
    std::function<void(const Event &)> emit_;
    /** The detection, used by the caller alone, and whether the caller commits while there is none. */
    std::unique_ptr<Detection> detection_;
    bool manual_ = true;
    double detecting_ = 0;

    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::optional<Asked> asked_;
    /** The buffer, whose first sample is the sample `buffer_` of the whole; the samples taken are `heard_`. */
    std::vector<float> audio_;
    uint64_t heard_ = 0, buffer_ = 0;
    /** The samples of the utterances committed and not yet recognized. */
    size_t committed_ = 0;
    /** Where the detection's audio began, how much of it it took, and how many of its regions are committed. */
    uint64_t origin_ = 0, pushed_ = 0;
    size_t regions_ = 0;
    uint64_t next_ = 0;
    std::optional<Utterance> current_;
    std::deque<Final> finals_;
    /** Whether the worker recognizes, an utterance or the beginning of one. */
    bool busy_ = false;
    std::shared_ptr<Cancellation> reading_, final_;
    bool closing_ = false;
    std::thread worker_;
};

}  // namespace utterances
