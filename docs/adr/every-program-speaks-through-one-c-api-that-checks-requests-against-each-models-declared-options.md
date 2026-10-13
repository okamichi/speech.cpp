# Every program speaks through one C API that checks requests against each model's declared options

## Context

speech.cpp serves any program that speaks text or recognizes speech: its own command line, worker and server, ASIST,
other tools, and bindings in other languages. A C++ interface with `std::function` and `std::string` has no stable ABI
across compilers and standard libraries, and a binding in Python or another language needs a C layer anyway.

The families differ in what a request may ask: Qwen3-TTS has named speakers, sampling settings and a cap on the length,
Irodori-TTS takes voices from files, a speed, a length, steps and guidance, and the recognizers a language, a prompt or
a decoding. A caller has to learn what a model takes without trying it and before loading it, or it keeps tables of its
own that drift from the model's. A struct that gains a field changes the ABI, and a binding through ctypes with an older
layout reads the wrong memory instead of failing.

Elsewhere: OpenAI refuses a parameter a model does not take with its name and accepts the neutral value, and once
ignored `speed` on gpt-4o-mini-tts for five weeks of user reports. CrispASR answers an unsupported setter with -2, which
its own Python binding treats as success. audio.cpp declares each family's options with their defaults and bounds, reads
that one declaration to validate and to answer questions, and keeps a major and a minor ABI version for bindings that
bind their imports up front.

## Decision

- **One C API for every program.** `include/speech.h` is plain C: opaque handles, UTF-8 strings, statuses, and every
  object and string the library returns owned by it. Every program and binding speaks through it, the worker, the
  server and the command line included, so the API they run every day is the one others get. The library is built
  statically into `speech` and as the shared `libspeech`, which exports the `speech_*` functions alone.
- **A model does one task**, synthesis, recognition or detection
  ([the record of detection](detecting-speech-is-a-task-of-its-own-run-by-speech-detect-with-silero-vads-options-and-f32-files.md)), and the same handles serve every task: a model is loaded, described and freed
  alike, and a request is made, set, cancelled and freed alike, `speech_synthesize()`, `speech_transcribe()` or
  `speech_detect()` running it. A detection of audio given a piece at a time is an object made from a request's
  options ([its record](a-detection-of-audio-given-a-piece-at-a-time-gives-speech-detects-regions-each-once-it-is-certain.md)).
  The call of another task is `unsupported`. The families are one table in `src/api/speech.cpp`, with each
  family's task, layout, information, engine and maker of voice files.
- **One vocabulary, declared per family.** Every request option is a value of `speech_option` with a fixed snake_case
  name, which the worker, the server and the command line use as it is. Each family declares in one table which options
  it takes, with their types, defaults, ranges or choices, reading the model's own defaults and bounds from its file.
  The setters, the model information and its JSON, which the worker's `ready`, the server's `/v1/models` and
  `speech info` carry as it is, all come from that table, so there is no second copy to drift, and a misspelled option
  is a compile error in C or an unknown name, never an answer of "not supported".
- **Neutral values pass, anything else unsupported is an error.** Speed 1, duration scale 1, language `auto`, no
  timestamps, an empty prompt and empty instructions are accepted by every model, so a caller that knows nothing of a
  model can still send a request; any other value of an option a model does not take is `unsupported`, with the option
  named.
- **Requests are objects whose setters check each value.** A request is made for one model; a setter refuses a value the
  model does not take as it is set, and the request checks what only the whole request shows before any work. An added
  option never changes the ABI, a binding mirrors no struct, and the mistake fails at the call that made it. A request
  runs once and is cancelled by itself, from any thread, and a model serves one request at a time. Its result (the stop
  reason, the seed, the text and the rest) belongs to it, and progress reaches a callback of its own while its work
  passes no audio.
- **The information is read without loading**, from the file's metadata (`speech_model_info_open()`), and the same
  information comes from a loaded model with its device, threads and added voices.
- **Errors carry a category and the input at fault**, so that a server derives its status and a program its message
  from the category alone, and no input reaches an abort of ggml: the device list leaves out accelerators that cannot
  run a model by themselves, such as BLAS, and the reader checks every key's type.
- **Loading takes the same parameters for every family.** The device is `auto` by default and reported, and a device
  asked for by name or as `gpu` that is not there is an error, never replaced. Threads are a preference about how the
  work runs, which applies to what the model computes on the CPU, and the information reports the number in effect. The
  warm-up is a switch, off in the library and on in the worker and the server, because a one-shot run gains nothing
  from computing twice. Voices are added after loading.
- **The process is left as it was found.** Logs go to a callback the host sets, and until it sets one only warnings
  and errors go to stderr. The library sets nothing in the environment, and ggml's Metal backend only its own
  `AGX_RELAX_CDM_CTXSTORE_TIMEOUT` while it lists its devices; ggml's fixes are in its source
  ([the record of the fork](ggml-comes-from-a-fork-that-carries-tested-fixes-for-the-defects-of-its-upstream-tag-and-measured-speed-ups.md)).
- **The API has a major and a minor version**, and the shared library's SOVERSION is the major: an added function,
  option or enum value raises the minor, so that a binding that binds its imports up front can tell an older library
  that lacks a function, and a change an existing caller notices raises the major.

The alternatives were turned down:

- A C++ interface as the only one. It has no stable ABI, and every other language would need a C layer.
- A C API beside a worker that calls the families directly. The worker would not exercise the API, and the two ways in
  would drift apart.
- An HTTP mode in the worker as the way in for other programs. It would tie every caller to a process and a protocol.
- A handle per task (`speech_synthesizer`, `speech_recognizer`) with its own load, free, getters and cancel. Loading,
  devices, cancellation, the one-request rule and most getters are the same for both, and two handles would double the
  API for a difference one getter states.
- Load parameters that one family ignores, such as a codec or voices given to a recognizer. The caller would not learn
  that what it passed means nothing.
- A text the caller frees with a function of its own. Every string the library returns stays the library's, living as
  long as the object it was read from.
- Structs with a leading size, read by offset, as Piper and the Linux kernel do. A binding still mirrors each version's
  layout, and a wrong value is still found only when the request runs.
- Ignoring an option a model cannot follow. The caller gets audio other than what it asked for and no sign of it.
- Options and capability questions as free strings, as audio.cpp has them. A misspelled question is answered "no".
- A request type per family, as whisper.cpp's `parakeet.h` and sherpa-onnx's sub-structs per model have it. Most
  options mean the same in every family, and the API would grow with each one.
- One API version that additions do not raise. A binding that binds up front cannot tell a missing function.

## Consequences

ASIST and other callers read voices, languages, ranges and limits from a model's information instead of tables of their
own. docs/c-api.md lists the options each family takes, the JSON form of the information and the names of the error
categories. A family is added with its line in the table, its engine and its table of options; the setters, the
information and every entry point take its options without a change of their own.
