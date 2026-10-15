---
name: windows-check
description: Check a speech.cpp change on a Windows x64 machine with a Vulkan GPU. Covers taking the build and the check executables from CI, getting models and small dumps there, running stage checks, bit-for-bit comparisons with the previous release, ggml's test-backend-ops, timing two builds and their VRAM, the smoke scripts and the installer, and leaving the machine as it was. Use when a change must be verified on Windows or Vulkan, before a release, or when asked to test on Windows (Windows で確かめる、Vulkan で確認、RTX で計測、install.ps1 を試す). Not for building on macOS or Linux, or for measuring speed across runtimes (speech-bench).
---

# Checking on Windows with Vulkan

CI builds and links on Windows but has no GPU, so a change that touches computation is checked on a Windows machine
with a Vulkan GPU before it is released. The commands below run in PowerShell or cmd on that machine, which a session
reaches through whatever remote shell it has.

## Ground rules

- Work in one temporary folder (`%TEMP%\speech-<task>`), and remove it when done. Large data (several builds, models,
  dumps) goes in a folder of the same kind on a drive with room, removed the same way. Change nothing else on the
  machine: no installs, no PATH, no settings, no other program's data. The one exception is the Vulkan SDK of the
  version CI pins (`VULKAN_VERSION` in `.github/workflows/build.yml`), installed under `C:\VulkanSDK`; its installer
  needs an administrator, so the user runs it.
- Large files are fetched by the Windows machine itself (GitHub, Hugging Face), not copied over a remote shell, which
  may be slow. Send only small files: scripts and short dumps.

## 1. The build

- The pull request's archive: `gh run download <run> -R nyosegawa/speech.cpp -n speech-windows-x64-vulkan -D build`,
  then unpack it. `<run>` is the pull request's latest CI run (`gh pr checks <n>` links it).
- The check executables are not in that archive. Run the workflow on the branch with them kept:
  `gh workflow run build.yml --ref <branch> -f checks=true`, then download the artifact
  `speech-checks-windows-x64-vulkan` from that run.
- The previous release, for comparisons: its `speech-<version>-windows-x64-vulkan.zip` from GitHub releases.

## 2. Models and dumps

- Models: `speech pull <name>` with `SPEECH_MODEL_DIR` set to a folder inside the temporary one, or a copy already on
  the machine whose SHA-256 you checked.
- Dumps: make a short case on the Mac (`reference/<family>/dump.py` with a sentence of a second or two), check it there
  with the same check, pack it and send it; a full dump is hundreds of megabytes.

## 3. What to run

| What | How | Passes when |
|---|---|---|
| Stage checks | `python checks\run.py <family> --build <checks folder> --devices gpu --models <model> --reference <folder that holds <family>\out>` | every line passes; the summary names the log of each that fails |
| ggml's fixes | `test-backend-ops.exe` from the checks artifact, built from the submodule's commit: `-o <op> -b Vulkan0` with the op of the fix's cases (`-o MUL_MAT -p large_b` for the q8_1 sums); on a newer upstream tag, the same from a build of the commit of cases alone (ggml-fork skill) | every case passes with the fixes; a case that passes on the tag alone marks a fix to drop once the rest of its "dropped when" holds (ggml-fork skill) |
| ggml's speed-ups | the same `test-backend-ops.exe`, `-o <op> -b Vulkan0` with the op of the speed-up's cases (`SNAKE_FUSE` for Snake); then the model the speed-up is for, timed as section 4 says | every case passes; the commit message gets the medians, the model, the type and the date |
| Window or chunk independence | the family's codec check with its forced patterns | every pattern gives the whole decode's samples |
| Unchanged behaviour | `python same_audio.py <old speech.dll> <model> <new speech.dll> <model> <prompts.json> --seeds 1,2,3` with `--voice FILE`, or `--model-voice NAME` for a model with built-in voices | the same audio, bit for bit |
| Entry points | `worker_smoke.py`, `server_smoke.py`, `server_realtime_smoke.py`, `speech_cli_smoke.py` with a model of each family | ok |
| Installer | `$env:LOCALAPPDATA` set to a folder in the temporary one, then `install.ps1 -NoModifyPath`: a first install, a second run, `-Version` of the previous release and back | each step says what it did; the user PATH is unchanged |

A fast GPU may never take the path a slow machine takes (smaller windows, for example); force it with the check's
patterns rather than concluding from the fast path alone.

Run a queue of checks as one script (PowerShell or Python) that runs them in order and writes each result to a file,
started so that the remote shell returns. Never chain waits on `pgrep -f` or a similar match of a command line: the
waiting shell's own command line holds the pattern, so it matches itself and never ends.

## 4. Measuring

- Time the two builds alternately (A B A B, three rounds or more) and report medians. Record any other load on the
  machine while they run: work on the CPU, such as FastConformer's decoding there, varies with it more than work on the
  GPU does.
- VRAM: under WDDM, `nvidia-smi` does not give a process's own use of the GPU's memory. Read the GPU's memory in use
  every 100 ms while the program runs (`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100`),
  with nothing else using the GPU, and take its peak above the reading before the program started.

## 5. Windows specifics

- Paths and text are UTF-8 in speech.cpp; PowerShell 5.1 pipes may garble Japanese. Write results to files from the
  program itself, or pass a script with `powershell -EncodedCommand` (UTF-16LE, base64).
- Python on Windows decodes its pipes in the ANSI code page. A Python script that the check feeds UTF-8 over a pipe
  reads `sys.stdin.buffer` and decodes UTF-8 itself.
- `--host 0.0.0.0` may raise the firewall's prompt; prefer loopback.
- Processes started from a remote shell may keep it waiting; start long ones so that they return, and stop what you
  started.

## 6. Record it

Write what ran, on what GPU and driver, and what it gave, as a comment on the pull request. Then remove the temporary
folder.
