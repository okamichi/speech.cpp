# speech serve and speech worker take request options as the defaults of the models they are started with

## Context

A request option's default is the model's, from its file, and a model file holds the official runtime's default:
Irodori-TTS v4.1-Small-MF samples in 4 steps. A program that wants another value for every request, such as 8 steps,
had to send it in each request to `speech serve` or `speech worker`; an OpenAI client sends it through `extra_body` in
every call. `speech tts`, `speech asr` and `speech vad` already take every option of the C API's vocabulary as a flag.

`speech serve` holds up to three models, one of each task, and an option can be taken by more than one task:
`language` by a synthesis and a recognition model, which may want different languages, and which Irodori-TTS, speaking
Japanese alone, refuses for any other. Its page loads other models in place of those given, and sets the options of
each request it sends from a form that leaves out what the user did not change.

Other servers settle this otherwise (as of 2026-10-11): llama.cpp's llama-server takes sampling flags as the defaults
of its requests and reports them at `/props`, and in its router mode reads a file of sections for every model and for
each one; NeMo-Speech.cpp names each engine setting after its task (`--tts.steps`, `--asr.vad.masker.onset`); audio.cpp
takes the defaults from a configuration file of models; whisper.cpp's server takes a fixed set of flags for its one
model. None gives a flag a meaning by where it stands on the command line.

## Decision

- **Both take every request option as a flag, as `speech tts` does**, and a flag's value is the default of every
  request that does not set the option, in place of the model's own. A request's own value wins.
- **`speech serve` gives a flag to every model given that takes the option, and the flag after `tts.`, `asr.` or
  `vad.` to that task's model alone**, winning over the flag without the prefix. The prefixes are the names of the
  subcommands that run each task. A flag no model given takes, or one for a task without a model, stops the server;
  a value one model refuses stops it with the prefixed flags that would give the value to the others alone.
- **Each value is checked against its model through a request's setters when the program starts**, so a value out of
  range stops it before it serves. Values that contradict each other are refused by the requests that run with them.
- **The defaults are listed beside the model information**: as `defaults` in the worker's `ready` and `info`, and in
  each model of `/v1/models`, whose page shows them in the options' fields. The model information itself keeps the
  model's own defaults.
- **A model the page loads keeps its own defaults.** The flags belong to the models the command line named.
- **The detection model's defaults come before OpenAI's server_vad defaults** in `chunking_strategy` and Realtime turn
  detection, which fill only what neither the request nor the command line gives.

The alternatives were turned down:

- A flag that belongs to the model it follows on the command line, as ffmpeg's options belong to the input they precede.
  It names one model without a prefix, but gives a flag's place a meaning no other server gives it, and a flag written
  before the first model is an error.
- A flag's place mattering only when several models take the option. Adding a model to the command line would change
  what an earlier flag means, and a flag after the wrong model would need a second rule to say where it goes.
- Flags with a task's prefix alone. Every option would need its task named even where one model takes it, and the
  server's flags would read otherwise than `speech tts`'s.
- A configuration file of models and their defaults. It suits several models of one task, which the server does not
  hold, and a server of a model or two would need a file where a flag does.
- Rewriting the defaults in the model information. The information would differ from `speech info --json`, which a
  caller compares it with to learn what a model takes and what it defaults to.

## Consequences

A program sets a model's behavior once where it starts the server or the worker, and an OpenAI client needs no
`extra_body`. Adding a model to `speech serve` can stop a command line whose flag the new model refuses; the message
names the prefixed flags that set it for one model.
