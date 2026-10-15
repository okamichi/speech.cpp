# One speech executable holds every subcommand

## Context

speech.cpp's programs for users are many: speaking a text and recognizing recordings on the command line, making voice
files, reading a model's information, listing devices, naming, fetching and removing models, quantizing a model file,
the worker and the HTTP server. Each needs the library and ggml, tens of megabytes with Vulkan's shaders, and each
reads options of the same vocabulary. Whatever ggml, a system framework or a GPU driver prints goes to the process's
stdout, which also carries a program's output: a WAVE file, JSON, the worker's protocol.

## Decision

- **One executable, `speech`, with a subcommand per program**: `tts`, `asr`, `vad`, `voice`, `info`, `devices`,
  `models`, `pull`, `rm`, `quantize`, `update`, `serve` and `worker`. It has the library and ggml linked in statically, so that a release is
  `speech` beside the shared library `libspeech` and the header.
- **One parser reads every command line.** Every option of the C API's vocabulary is a flag of its own name in
  kebab-case, so that an option the library adds is a flag without the parser or a subcommand changing; a boolean flag
  is true alone and takes `=true` or `=false`, since an option true by default could not otherwise be turned off.
  `--add-voice NAME=FILE` adds a voice and `--voice` selects one, as the option does everywhere.
- **cpp-httplib is compiled into `speech`** and runs only for `serve`, and **miniaudio** only for the microphone of
  `asr --live`; the library and `libspeech` stay free of both. miniaudio is one header that captures through each
  system's own audio API and, but on macOS, loads the system's audio library only when the microphone opens, so that
  `speech` builds without an audio library's development files and starts on a machine without one.
- **stdout carries a subcommand's output alone.** `speech` keeps the stdout it was started with for its output and
  points descriptor 1 at stderr before anything else runs, so that whatever other code prints to stdout reaches stderr
  with the logs.

The alternatives were turned down:

- An executable per program. Each would carry the same library and ggml in every release and read its command line its
  own way.
- `speech serve` in an executable of its own, to keep cpp-httplib out of the one the worker runs in. It would bring back
  a second executable with its copy of the library and a second parser.
- The microphone through SDL2, as whisper.cpp's `whisper-stream` takes it, or PortAudio, as sherpa-onnx does. Each is a
  library a user installs, or a build adds, beside `speech`.
- No microphone, with PCM piped into `speech asr -` from another program. The command line would not transcribe speech
  as it is said without a second program, which differs on every system.
- Recognition as a mode of `speech tts`. The two share `--device` and `--language` and nothing else: one writes a WAVE
  file from text, the other text from WAVE files.
- A `--no-<flag>` for each boolean option. It doubles the flags, and the vocabulary's names would no longer be the flags
  alone.

## Consequences

Every program that starts speech.cpp starts `speech <subcommand>`; a release is one archive per platform. docs/cli.md
gives the subcommands and their flags.
