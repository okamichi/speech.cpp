#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "failure.h"
#include "openai-error.h"
#include "request-options.h"
#include "transcript.h"

namespace httplib {
struct Response;
}

// OpenAI's audio API as speech.cpp reads and writes it: the create speech request, the create transcription request,
// its errors as HTTP answers, and the events of an SSE stream. The shapes follow
// OpenAI's API reference, components CreateSpeechRequest, CreateTranscriptionRequest, CreateTranscriptionResponseJson,
// CreateTranscriptionResponseVerboseJson, TranscriptionSegment, Error, SpeechAudioDeltaEvent and SpeechAudioDoneEvent
// and the path /audio/transcriptions of github.com/openai/openai-openapi at commit 31af4fc (2026-10-05), and its
// chunking_strategy as TranscriptionChunkingStrategy and VadConfig give it at commit 234829e (2026-10-07), whose object
// openai-python 3.26.0 sends in a form as one field per member, chunking_strategy[type] and so on.

namespace openai {

/** Answers with the error's status and body. */
void send_error(httplib::Response & res, const ApiError & e);

/** A create speech request as read from its JSON body. */
struct SpeechRequest {
    std::string input;
    std::string format = "wav";
    bool sse = false;
    double max_seconds = 0;
    /** The options of the vocabulary that the body sets, by their names. */
    std::vector<RequestOption> options;
};

/**
 * Reads a create speech request: "model", "input", "response_format", "stream_format" and every option of the
 * vocabulary by its name, OpenAI's "instructions" among them. A member it does not have is refused rather than ignored,
 * and a member set to null counts as left out. The model's name, which "model" may give, is the one /v1/models
 * lists. Anything it cannot read throws an ApiError; the values of the options are checked by the library's setters.
 */
SpeechRequest read_speech_request(const std::string & body, const std::string & model_name, bool compatible = false);

/** One part of a multipart/form-data body: a field, or a file when it has a filename. */
struct FormPart {
    std::string name, content, filename;
    bool file = false;
};

/** A create transcription request as read from its form, with the file's audio decoded. */
struct TranscriptionRequest {
    std::vector<float> samples;
    int sample_rate = 0;
    /** "json", "text" or "verbose_json". */
    std::string format = "json";
    /** The language, the prompt, the decoding, and timestamps for verbose_json. */
    std::vector<RequestOption> options;
    /** Whether the options set timestamps, so that the answer carries the segments. */
    bool timestamps = false;
    /**
     * With chunking_strategy, the members of server_vad it gives as detection options, which region_options() completes
     * with the detection model's defaults; without it, nothing.
     */
    std::optional<std::vector<RequestOption>> regions;
};

/**
 * Reads a create transcription request from the parts of its form: "file", a WAV file, its channels averaged;
 * "model"; "language"; "prompt", the option prompt; "response_format"; "decoding", the option decoding, speech.cpp's
 * own; "chunking_strategy", "auto" or the members of server_vad; and "timestamp_granularities[]", "segment" alone, with
 * verbose_json. verbose_json sets the option timestamps for a model that `gives_times`, and for any model when the
 * request names a granularity, which a model without times then refuses. A member it does not have is refused, and so
 * is one given twice but timestamp_granularities[]. Anything it cannot read throws an ApiError.
 */
TranscriptionRequest read_transcription_request(bool multipart, const std::vector<FormPart> & parts, const std::string & model_name,
                                                bool gives_times);

/** A speech.audio.delta event with a chunk of 16-bit PCM. */
std::string sse_delta(const std::string & pcm);

/**
 * The speech.audio.done event. OpenAI's carries the usage in tokens, which speech.cpp does not count, so it carries
 * the seed, the number of samples and the stop reason instead.
 */
std::string sse_done(int64_t seed, uint64_t samples, const char * stop);

/** An error after an SSE stream has begun, {"type": "error", "error": {...}}, which OpenAI's speech stream does not define. */
std::string sse_error(const ApiError & e);

/** A transcription in OpenAI's json form; OpenAI's also carries the usage, which speech.cpp does not count. */
std::string transcription_json(const std::string & text);

/**
 * A transcription in OpenAI's verbose_json form: the task; the language, the tag of the language the model heard, or
 * the tags joined with commas where it heard several in the parts of long audio or in its regions, as qwen-asr joins
 * their names; the
 * audio's duration in seconds; the text; and with `timestamps` the segments. A value the recognizer does not give is
 * left out rather than made up: the language where the result has none, and the members of OpenAI's segment the
 * recognizers have no value for (seek, tokens, temperature, avg_logprob, compression_ratio, no_speech_prob).
 */
std::string transcription_verbose_json(const Transcript & transcript, double duration, bool timestamps);

}  // namespace openai
