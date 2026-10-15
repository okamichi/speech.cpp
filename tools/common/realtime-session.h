#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "cancellation.h"
#include "json-reader.h"
#include "openai-error.h"
#include "realtime-events.h"
#include "request-options.h"
#include "transcript.h"
#include "utterances.h"

// A session of OpenAI's Realtime transcription, as its API reference gives the GA events at github.com/openai/
// openai-openapi commit 234829e (2026-10-07): the client's session.update, input_audio_buffer.append, .commit and
// .clear, and the server's session.created, session.updated, input_audio_buffer.speech_started, .speech_stopped,
// .committed and .cleared, conversation.item.input_audio_transcription.delta, .completed and .failed, and error. The
// session reads and checks each client event, keeps its configuration, and gives the audio to an assembly of utterances
// (utterances.h), which cuts it by the regions of the detection model with turn_detection server_vad, or at the client's
// commits without it, and recognizes each utterance, and its beginning while it goes on for the deltas; it writes the
// server's events as JSON text, and whoever carries the messages (the server's WebSocket) hands it the client's and
// sends its own. A member or a value speech.cpp does not take is answered with an error event rather than ignored.

namespace realtime {

/** The models a session works with, as its host holds them. */
class Models {
public:
    virtual ~Models() = default;
    /** The name of the recognition model held now, or "" while there is none. */
    virtual std::string held() = 0;
    /**
     * Checks `options` against the recognition model held, named `model` unless that is "": a model of another name or
     * none throws an ApiError, and an option the model does not take the library's Failure.
     */
    virtual void check(const std::string & model, const std::vector<RequestOption> & options) = 0;
    /**
     * Takes a place on the recognition model held, named `model` unless that is "", for a recognition accepted now: the
     * model, which it keeps until the recognition has run, and its turn among the requests on that model, given up if it
     * never runs. It throws as check() does.
     */
    virtual std::unique_ptr<utterances::Reservation> reserve(const std::string & model) = 0;
    /**
     * Recognizes `samples` at kRate in `reservation`'s turn on its model, with `options`, and returns the recognition, or
     * nothing once `cancellation` stops it; an option the model does not take throws the library's Failure. It lets the
     * samples go once the library's request has copied them. `item` names the item for a log, `provisional` says
     * whether the samples are the beginning of an utterance under way, read for its deltas, and `draft` is the text the
     * utterance's last reading gave, or "", which it sets as the option draft where the model takes it.
     */
    virtual std::optional<Transcript> transcribe(utterances::Reservation & reservation, std::vector<float> samples,
                                                 const std::vector<RequestOption> & options, Cancellation & cancellation,
                                                 const std::string & item, bool provisional, const std::string & draft) = 0;
    /**
     * A detection of audio at kRate with `options` on the detection model held, which it keeps: none held, or one being
     * loaded, throws an ApiError, and an option the model refuses the library's Failure.
     */
    virtual std::unique_ptr<utterances::Detection> detect(const std::vector<RequestOption> & options) = 0;
    /** The defaults the detection model held was given on the command line, which turn detection takes before OpenAI's. */
    virtual std::vector<RequestOption> detection_defaults() = 0;
    /** Whether `detection` runs on the detection model held now, which it no longer does once the host lets that go. */
    virtual bool holds(const utterances::Detection & detection) = 0;
};

class Session {
public:
    /**
     * `send` takes each server event as one JSON text, one at a time, from the thread that calls the session or its own.
     * `limit` bounds the bytes of 16-bit PCM the session holds before they are transcribed (utterances::Assembly::held()),
     * and `model` names the recognition model the session starts with, or is "" for the one held.
     */
    Session(Models & models, std::function<void(const std::string &)> send, size_t limit, const std::string & model);
    /** Stops the session's watch over the detection model, then its recognitions. */
    ~Session();
    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;

    /** Sends session.created, the configuration the session starts with. */
    void open();
    /** Handles a client event: one JSON object in a text message; a binary message is refused. */
    void receive(const std::string & message, bool text);

private:
    /** The configuration a session.update changes. */
    struct Config {
        /** The transcription model the client named, or nothing for the one held. */
        std::optional<std::string> model;
        /** Whether a committed buffer is transcribed: audio.input.transcription is not null. */
        bool transcribe = true;
        std::optional<std::string> language;
        std::optional<std::string> prompt;
        /** With turn_detection server_vad, the detection options of the members given; nothing for null. */
        std::optional<std::vector<RequestOption>> turns;

        std::vector<RequestOption> options() const;
        std::optional<utterances::Asked> asked() const;
    };

    /** The assembly's recognizer: the host's recognition model, its items named as the events name them. */
    class Reader : public utterances::Recognizer {
    public:
        Reader(Models & models, Events & events) : models_(models), events_(events) {}
        std::unique_ptr<utterances::Reservation> reserve(const utterances::Asked & asked) override;
        std::optional<Transcript> transcribe(utterances::Reservation & reservation, const utterances::Asked & asked, std::vector<float> samples,
                                             int rate, Cancellation & cancellation, uint64_t utterance, bool provisional,
                                             const std::string & draft) override;

    private:
        Models & models_;
        Events & events_;
    };

    void update(const JsonValue & event);
    void append(const JsonValue & event);
    void commit();
    void clear();
    std::string session_json();
    /**
     * With server_vad, lets the detection model go once the host no longer holds it, the regions its detection gives at
     * its end committed, and goes on with the one held next.
     */
    void tick();
    /**
     * Ticks every 0.2 s on a thread of its own, so that a session lets a detection model go soon after the host replaces
     * it while the client sends nothing, and the carrier reads its messages whole, with its own timeout; each client event
     * ticks first too.
     */
    void watch();

    Models & models_;
    Events events_;
    Reader reader_;
    Config config_;
    const size_t limit_;
    /** Whether the session has said that turn detection waits for a detection model the host does not hold. */
    bool told_no_detection_ = false;
    /** Declared after what its recognitions use, so that they end before it goes. */
    utterances::Assembly assembly_;

    /** Taken by receive() and tick(), so that one thread at a time works the assembly. */
    std::mutex mutex_;
    std::condition_variable closing_changed_;
    bool closing_ = false;
    std::thread watcher_;
};

}  // namespace realtime
