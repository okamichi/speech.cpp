#pragma once

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "json-reader.h"
#include "speech.h"

// The C API's vocabulary of request options as every subcommand reads it: a flag of its name in kebab-case on the
// command line, a member of its name in the worker's messages and the server's requests, each value read by the
// option's type and set through the setter of that type, which checks it against the model.

/** A value of an option, of the option's type. */
using OptionValue = std::variant<std::string, int64_t, double, bool>;

/** An option and its value, as a command line or a message gives it. */
struct RequestOption {
    speech_option option;
    OptionValue value;
};

/** Every option the library knows, in the order of the vocabulary. */
std::vector<speech_option> vocabulary();

/** The option named `name` in snake_case, or false for a name the vocabulary does not have. */
bool option_named(const std::string & name, speech_option & option);

/** The names of the vocabulary in its order, joined for a message: "voice, language, ... and timestamps". */
std::string option_names();

/** The option's flag on the command line, its name in kebab-case: "--duration-scale". */
std::string option_flag(speech_option option);

/** The option's type as the command line's help writes its value: "TEXT", "INTEGER" or "NUMBER", "" for a boolean. */
const char * option_metavar(speech_option option);

/**
 * The value of an option read from the text a command line gives it, whole: an integer of decimal digits with an
 * optional minus sign, a finite decimal number, or "true" or "false". Anything else throws std::invalid_argument naming
 * what the option takes.
 */
OptionValue option_from_text(speech_option option, const std::string & text);

/**
 * The value of an option read from a JSON member: a string, an integer written without a fraction or an exponent, any
 * finite number, or true or false, by the option's type. Another JSON type throws std::invalid_argument naming what
 * the option takes.
 */
OptionValue option_from_json(speech_option option, const JsonValue & value);

/** Sets each option on `request` through the setter of its type, in order; the library's refusal throws a Failure. */
void apply_options(speech_request * request, const std::vector<RequestOption> & options);

/**
 * Sets `draft` as the option draft of `request`, a request on `model`, where the draft is not empty and the model takes
 * the option; a model that does not take it would refuse the request.
 */
void apply_draft(speech_request * request, const speech_model * model, const std::string & draft);

/** Options as a command line writes them, for a log: "--steps 8 --keep-tail=true". */
std::string options_text(const std::vector<RequestOption> & options);

/** Options as a JSON object of their names and values, as a request's members give them: {"steps":8,"language":"ja"}. */
std::string options_json(const std::vector<RequestOption> & options);

/**
 * The options among `defaults` that the model takes, each checked against it through a request's setters, so that a
 * value it refuses fails where the program starts rather than at its first request. The setters check each value
 * alone; values that contradict each other are refused by the requests that run with them.
 */
std::vector<RequestOption> defaults_for(speech_model * model, const std::vector<RequestOption> & defaults);

/**
 * A request's options after the defaults its program was started with: each default whose option `options` does not
 * set, then `options`, so that the request's own value wins and the model's default holds for the rest.
 */
std::vector<RequestOption> with_defaults(const std::vector<RequestOption> & defaults, const std::vector<RequestOption> & options);

/**
 * Whether a recognition with `options` carries its segments and tokens: the timestamps option's value where the
 * options set it, else the model's default, and false for a model that does not take it.
 */
bool timestamps_in_effect(const speech_model_info * info, const std::vector<RequestOption> & options);

/**
 * The members of a recognition's result in the form of the worker's messages, each after a comma: "text", "stop" (why
 * the recognition ended, "complete" or "model_limit"), "languages", the tags of the languages the model heard, where the
 * result has any, and with `timestamps` "segments" and "tokens", each a list of {"start", "end", "text"} with the times
 * in seconds.
 */
std::string recognition_members(const speech_result * result, bool timestamps);
