---
name: release
description: Release a new version of speech.cpp. Covers the model files on Hugging Face and their cards, the catalog compiled into speech, re-measuring the numbers the docs give (RTFx, time to first audio, latency, load time, peak memory, VRAM), the pull request that raises VERSION, the README's screenshots of the page, the tag that builds and publishes the archives, and checking what was published. Use when asked to release, publish, cut or tag a version, bump VERSION, upload converted model files, or write a model card (リリース、バージョンを上げる、タグ、公開、HF に上げる、モデルカード). Not for an ordinary pull request (pull-request), porting a model (add-model) or making one faster (speed-up).
---

# Releasing speech.cpp

A release is the archives CI builds from a tag, and the model files the catalog in those archives names. The catalog
is compiled into `speech`, so the files on Hugging Face are settled before the version is raised. Releases and tags are
never deleted or moved: callers such as ASIST pin the archives by SHA-256.

## 1. Before the release

- Everything the release carries is on main, its CI green, and checked on Vulkan as well (windows-check).
- The C API's minor version was raised once for this release by the first change that added to it; nothing raises it
  again. A change an existing caller notices raises the major version, which is the shared library's SOVERSION.

## 2. Model files (only when a file changes)

A file changes when its family's layout changes, a type is added, or a checkpoint is converted again.

1. Make the files: `reference/<family>/convert.py` writes F32 and `speech quantize F32.gguf OUT --type <type>` the
   other types; `checks/compare/quantize_compare.py` shows that an unchanged file comes out byte for byte. Check each
   with the family's stage checks and `speech-api-check`. Before a type is published for the first time,
   `checks/compare/quantize_releases.py` with the previous release's `speech` shows that each file's `speech.requires`
   names a release that reads it.
2. Write each file's information beside it: `speech info --json <file> > <file>.json`.
3. Write the card, as references/model-card.md says.
4. **Only with the maintainer's approval**, upload the file, its JSON and the card to the model's repository in one
   commit, deleting a file the new one replaces under another name (a layout that changes the size label changes the
   name): `hf upload sakasegawa/<Upstream>-GGUF <folder> . --delete "<old file>*"`. The old files stay in the
   repository's history, at the revisions earlier catalogs pin. Check that the repository lists each file with the
   SHA-256 you made.

The repositories: one for each upstream repository, named after it with `-GGUF`
(`sakasegawa/Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF` for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`), holding one file per
model and type, the codec inside it, under the name its converter gives it, the licenses of what it holds, and the
`.json` beside each file. Lower-bit files
go up only with their measured accuracy in the card, and never as the type a name fetches.

## 3. The catalog

`python3 tools/catalog/update_catalog.py` pins every model at its repository's current revision. Review the diff: each
changed revision must be one you uploaded on purpose. Build, and check that `speech pull` fetches each changed file and
the model loads. The catalog's update is a pull request of its own, or part of the version's.

## 4. The numbers

The numbers of README.md, README.ja.md and the model pages are measured again for the release, with speech-bench
against the build that will be released: `SPEECH_BENCH_SPEECH_CPP=<build dir>` for a local build, `ci:<run id>` for a CI
run's archive (the Windows machine takes this one). A model whose speed or memory changed since the last release has its
page's tables measured again for every type they list, not only the headline.

### What is measured

| Figure | How |
|---|---|
| RTFx | seconds of audio over seconds of wall time, the median over the inputs |
| time to first audio | from a synthesis request to its first chunk: p50, p90 and p99 |
| latency | from a recognition request to its result: p50, p90 and p99 |
| the first request | the first request after loading, apart from the rest, which it would skew |
| load time | cold, from a copy of the file that is not in the OS's file cache, and warm, loaded again |
| peak memory | the process's peak: "peak memory footprint" of `/usr/bin/time -l` on macOS |
| VRAM | the GPU's memory in use, its peak above the start, on Windows (windows-check) |

### How

- A cold load needs a copy that the OS has not cached, made without administrator rights: on macOS an APFS clone
  (`cp -c`), on Windows an unbuffered copy (`robocopy /J`), a new one for each load. Show first that it is cold: its
  first load takes clearly longer than loading it again.
- Alternate the builds or runtimes compared (A B A B), three rounds or more, all on the same day with nothing else
  running, and report medians.
- Compare with the official implementations, through speech-bench's adapters (MPS on macOS, CUDA on Windows), and
  with the other runtimes that run the model.
- Write the environment with the numbers: the machine, its chip or GPU and memory, the OS and driver versions, the
  release or commit, the model's file and type, and the date. Each number in the pages carries its model, type and
  device.
- A number that moved goes into the pages in the version's pull request or before it.

## 5. The version

- The release's number is written in `VERSION` and nowhere else in the code: CMake reads it, and `speech --version`,
  `speech_version()` and the worker's `ready` report it.
- Versions follow Semantic Versioning. While they are 0.x, a change a caller must adapt to (the worker protocol, the C
  API, the GGUF layout, a voice file's form, the command line's arguments) raises the minor version, and any other
  release raises the patch.
- A pull request of its own raises `VERSION` just before the tag, with every doc that names the version.
- Merge it when CI passes (pull-request skill).

The README's screenshots of the page show no version, so the version's pull request leaves them as they are. The pull
request that changes how the page looks takes them again (docs skill), with its build:

  ```sh
  node skills/release/scripts/page-screenshot.mjs build/speech docs/images/page qwen3-tts-0.6b qwen3-asr-0.6b silero-vad \
      "Hello. It will be sunny in Tokyo tomorrow, with a high of twenty-four degrees." ryan 6368195046531003
  ```

  It starts `speech serve` with the three models on port 8080 and a headless Chrome in English, speaks the text with
  the seed (`page-speak.png`), hands the speech to Transcribe and transcribes it (`page-transcribe.png`), and transcribes
  a microphone that says the text a sentence at a time (`page-live.png`), taken while the first sentence is final and as
  much of the next as possible is grey. Look at each image, the transcripts the script prints included, before committing
  them; a model that samples can say the text badly with a seed that once read well, and another seed is the fix.

## 6. Tag and publish

```sh
git fetch origin && git tag v<version> origin/main && git push origin v<version>
```

The tag runs CI's build on every platform and the release job, which checks the tag against `VERSION`, sums the
archives into `SHA256SUMS` and publishes the GitHub release with them. Watch it with `gh run watch`. The archives,
`speech-<VERSION>-<platform>.zip`, are the ones CI builds and checks on every run; a change to what they hold changes
the packaging steps of `.github/workflows/build.yml` and docs/install.md's Release archives together.

## 7. Check what was published

- `gh release view v<version>` lists the four archives and `SHA256SUMS`.
- The installer installs it: `curl -fsSL …/install.sh | sh` in a temporary `HOME` on macOS, and `install.ps1` on
  Windows with `LOCALAPPDATA` pointed at a temporary folder and `-NoModifyPath`; `speech --version` names the release.
- `speech pull <name>` and one synthesis and one recognition by name work from the installed `speech`.

## 8. After it

Programs that pin speech.cpp take the release in their own repositories: speech-bench's runtime pin, and ASIST's
accepted series and the model pins it reads from `speech models --json`. Note them for their maintainers; do not change
other repositories from this one.
