---
name: add-model
description: Port a speech model to speech.cpp, or update or extend one it has. Covers trying a candidate in speech-bench before taking it up, pinning the official implementation, dumping its tensors, converting to one GGUF file, the family's C++ port, stage-by-stage checks against the official code, request options, the entry points, the Hugging Face files and catalog, and measuring in speech-bench. Use when asked to add, port, support, convert or update a model or model family, or to expose an official option or feature of one (新しいモデルを追加、移植、対応、モデルの更新、公式のオプションに対応、変換し直し). Not for a defect in one existing stage, a documentation change (docs), or cutting a release (release).
---

# Adding or updating a model

A model in speech.cpp is trusted because every stage is compared with the official implementation and because a
request that asks nothing new gives what the previous release gave. Everything below protects those two promises.
Read AGENTS.md first; this skill adds the order of work and the checks that are easy to forget.

## 1. Before writing code

- **Try the model in speech-bench first** (github.com/nyosegawa/speech-bench): in its official implementation, through
  an adapter (section 7), and in the runtimes that already run it. Take it up only if its accuracy on the bench's sets,
  its speed and its memory justify a port; the measurement stays in speech-bench, where later ports compare with it.
- **List the official request's every field** (the runtime's request object, `transcribe()`'s or `generate()`'s
  arguments, the server's form) and decide each one: a request option, a property of a voice file, or not offered,
  with the reason. Write the decision in an ADR before porting; a field found later changes the C API.
- **List what the checkpoint holds.** Dropping weights at conversion (a second head, a null speaker, a caption encoder)
  means a new layout later, an upgrade of every file users have, and a new upload. Drop nothing without an ADR saying
  why.
- Read the family's existing ADRs in docs/adr/ whose names cover the behaviour you change.

## 2. Reference, dumps and conversion

`reference/<family>/` holds the official code pinned by uv, `pins.py` pinning each checkpoint by revision, size and
SHA-256, `dump.py`, and `convert.py`. Read references/reference.md for what a dump must contain and how conversion
names keys, files and layouts.

- Dump every stage, with the noise fixed, for each new option and each setting a request can take. Use only audio
  that may be published (FLEURS, Common Voice); never a recording of a person who has not agreed to it.
- `convert.py` writes F32, which the checks run on; `speech quantize` makes every other type, each tensor as the
  family's `layout.cpp` table stores it. Decide in that table which tensors a type may change.

## 3. The port

- `src/families/<family>/` reads one GGUF file through its `layout.cpp`, which says every key, every tensor and the
  types each tensor may take. A layout change raises `speech.layout`, adds the release that reads it, and brings the
  previous layout up in the family's one upgrade function, so that no user downloads a file again.
- The engine (`src/engines/<family>-engine.cpp`) declares the options the model takes, with defaults and ranges read
  from the file. An option without effect, or a value the model cannot compute, is refused with the option named, never
  ignored.

## 4. Checks

Read references/checks.md before setting or loosening any bound. Each check prints its usage when run without
arguments, and each `dump.py` and `convert.py` says its own at the top.

- Each stage from the dump's own inputs, on the CPU in F32 to a tight bound, on Metal, and on Vulkan (windows-check).
  Tokens and text compare exactly; tensors compare by SNR.
- **Unchanged behaviour:** a request that sets none of the new options gives the previous release's float samples bit
  for bit (`checks/compare/same_audio.py`), or the difference is measured and written down. The same seed gives the same
  output. A computation may follow the machine's speed (window or chunk sizes) only where every size gives the same
  samples, shown bit for bit on every backend.
- A check that fails without your change, wherever one can be written.

## 5. Entry points

- C API: new options go at the end of the vocabulary, marked with the minor version that adds them; the minor version
  rises once per release. Options a family does not take are refused by the table, not by code.
- Worker, server and command line take the option by its name; run every smoke script the change touches, with a model
  of each family (AGENTS.md's Workflow names them).

## 6. Synthesis or recognition

Read references/synthesis.md for voices, length, sampling, streaming, interruption and what the official output adds
that speech.cpp does not; references/recognition.md for decoding, prompts, languages, long audio and results.

## 7. Files, catalog and measurement

Read references/distribution.md. In short: one `sakasegawa/<Upstream>-GGUF` repository per upstream model with a card
and the `speech info --json` of each file beside it, uploaded only when the maintainer approves; then
`python3 tools/catalog/update_catalog.py`; then speech-bench's catalog and a measurement against the official
implementation and the other runtimes on the same day, on a quiet machine, with peak memory and VRAM beside the speed.
When speech-bench has no adapter for the model's official implementation, add one: a uv project pinned as the
reference is, running on MPS on macOS and on CUDA on Windows.

## 8. Finishing

- An ADR for every choice the code cannot show; the docs pages that describe the model (docs skill).
- One pull request per coherent unit (pull-request skill).

## Working in parallel

Several families can move at once, each in a worktree of its own. Commit as each part works, so that a stopped session
loses nothing. Run the long CPU F32 checks one family at a time, and speed measurements only when nothing else runs.
Agents sharing a machine serialize their GPU work with a lock held for one run (`/usr/bin/lockf -k <lock file>
<command>`) and build with a bounded `-j`. Wait for a background job by its notification or a file it writes, never with
a `pgrep -f` loop, which matches the waiting shell's own command line and never ends.
