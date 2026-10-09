#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "cancellation.h"
#include "library.h"
#include "request-options.h"

// Speaking a text a sentence at a time, for a model whose request speaks only a short time: Irodori-TTS speaks at most
// 30 s in a request, which a few sentences pass, and refuses a text whose speech it predicts to last longer. Such a
// model speaks each sentence of a text as a request of its own, with the same options and seed, and the audio of the
// requests is passed on as it is made, each join after a sentence holding the pause of one inside a request. A model
// whose request speaks longer reads a text in one request, with its prosody across the sentences.

/** Where a text is cut: after its sentences, after its clauses, or at its spaces. */
enum class Cut { Sentences, Clauses, Words };

/**
 * The pieces of `text` cut after each end of a sentence (。｡！？． always; . ! ? when a space or the end follows;
 * a line break), or after each comma (、､，；, and , ; when a space or the end follows) with the closing marks that
 * follow them, or at each run of spaces, each without the spaces around it; pieces of nothing but spaces are left
 * out.
 */
std::vector<std::string> cut_text(const std::string & text, Cut cut);

/**
 * The longest speech one synthesis request of the model makes, in seconds: the upper bound of its option seconds, or
 * of max_seconds; infinity for a model that takes neither.
 */
double longest_request_seconds(const speech_model_info * info);

/** Whether the model speaks a text of several sentences a sentence at a time. */
bool speaks_by_sentence(const speech_model_info * info);

/** What a synthesis of a text made. */
struct Spoken {
    /** The first request's stop that was not complete, which ended the synthesis, or complete. */
    speech_stop stop = SPEECH_STOP_COMPLETE;
    /** The seed of every request: the options' seed, or the one the first request drew. */
    int64_t seed = -1;
    /** The samples passed on, the pauses at the joins included. */
    uint64_t samples = 0;
    /** The library requests that spoke. */
    size_t requests = 0;
    /** Whether any request truncated its instructions at the model's token limit. */
    bool instructions_truncated = false;
};

/** Where a model that speaks by sentence begins to cut a text into requests. */
enum class Split {
    /** Each sentence of a text of several is a request of its own: speech tts and speech serve. */
    Sentences,
    /** The text is one request, cut only when the library refuses it as too long: the worker, whose caller chose the text. */
    Refused,
    /** Sentence splitting with Unicode bisection when no punctuation or space fits the model. */
    Compatible,
};

/**
 * Speaks `text` on `model` with `options`, passing the audio to `on_audio` with `user_data` as it is made, request
 * after request, and reporting each request's progress to `on_progress` where it is set. A model that speaks by
 * sentence speaks each sentence of a text of several as a request of its own where `split` says Sentences; a text of
 * one sentence, a text where it says Refused, and any text on another model, is one request, unchanged. On a model that
 * speaks by sentence, a text the library refuses as too long, before any audio, is cut after its sentences, or after
 * its commas where it is one sentence, or at its spaces where it has none, and each piece is spoken the same way in turn; a piece
 * with no place to cut that the library refuses fails, naming the piece. The options' seed, or the first request's,
 * applies to every request. A request that stops other than complete ends the synthesis. It returns nothing once
 * cancelled, by `cancellation` or by a callback returning nonzero; a failure of the library throws. Compatible also
 * bisects a refused piece at Unicode boundaries. A nonzero limit_samples caps the whole response, including pauses,
 * and reports max_seconds rather than cancellation.
 */
std::optional<Spoken> speak_text(speech_model * model, const std::string & text, const std::vector<RequestOption> & options, Split split,
                                 speech_audio_callback on_audio, speech_progress_callback on_progress, void * user_data,
                                 Cancellation & cancellation, uint64_t limit_samples = 0);
