---
name: speed-up
description: Make a speech.cpp model faster or lighter in memory, and show it with numbers. Covers finding where the time goes, prototyping on an experiment branch, measuring before and after (RTFx, time to first audio, latency, load time, peak memory, VRAM), checking that the output stays the same or changes only by rounding, checking on Vulkan, and the tracking issue. Use when asked to speed up, profile or optimize a model, a stage or a request, to cut its memory, or to measure a gain (速くする、高速化、メモリーを減らす、プロファイル、レイテンシを下げる、RTFx、speed up, latency, profile). A speed-up inside a ggml kernel also follows ggml-fork. Not for re-measuring the numbers of a release (release) or porting a model (add-model).
---

# Making a model faster or smaller

A speed-up goes in only when a measurement shows a gain beyond the noise and the output stays what the checks allow.
Everything below protects those two. Read AGENTS.md first; a change inside a ggml kernel also follows the ggml-fork
skill.

## 1. Find where the time goes

- Time each stage before choosing what to change: a stage that takes 5% of a request cannot give more than 5%.
  `measure/` times some families (`qwen3-asr-timing` and `fastconformer-timing`, built with the checks, and
  `irodori-caption-timing.py` through the worker); each prints its usage.
- Where no tool covers the stage, add timers on an experiment branch and never on main. A timing tool worth running
  again goes into `measure/` in a pull request of its own, written as the ones there are.
- Measure memory as well: the peak of the process, and what each stage allocates on the device.

## 2. Prototype, then reimplement

- Try ideas on an experiment branch in a worktree of its own, behind environment switches, so that one build compares
  them. That branch is a measurement aid and never merges.
- An idea that gains is written again in a pull request from `origin/main`, without switches, as AGENTS.md's code rules
  ask, and measured again on that build: the reimplementation is not the prototype.

## 3. Measure

- The same machine on the same day, with nothing else running. Build the old and the new into separate folders and
  alternate them (A B A B …), at least 3 rounds; report the median of each build, and p50, p90 and p99 where the
  requests differ in length.
- The first request after loading compiles kernels and fills caches; measure it apart from the rest.
- Use the names the docs use, each with its condition (the model, the type, the device and backend, the inputs, the
  date):

  | Figure | What it is |
  |---|---|
  | RTFx | seconds of audio over seconds of wall time; higher is faster |
  | time to first audio | from a synthesis request to its first chunk |
  | latency | from a recognition or detection request to its result |
  | load time | from loading the model to its first request being accepted |
  | peak memory | the process's peak: "peak memory footprint" of `/usr/bin/time -l` on macOS |
  | VRAM | the GPU's memory in use, its peak above the start, on Windows (windows-check) |

- Several agents on one machine serialize their GPU work with a lock held for one run, not for a whole loop
  (`/usr/bin/lockf -k <lock file> <command>`), build with a bounded `-j`, and wait for a background job by its
  notification or by a file it writes. Never wait with a `pgrep -f` loop: the waiting shell's own command line holds
  the pattern, so it matches itself and never ends.

## 4. Check the output

- **Output that should not change** is compared bit for bit, on the Mac's CPU and Metal and on Vulkan on Windows:
  `checks/compare/same_audio.py <old lib> <model> <new lib> <model> <prompts.json>` with `--voice FILE`, or
  `--model-voice NAME` for a model with built-in voices, and several seeds. A codec decoded in windows runs its check's
  forced window patterns on every backend. A recognition gives the same text, tokens and times from both builds.
- **Output that changes by rounding** (another kernel, a fused operation, flash attention, another order of sums) passes
  every stage check it touches within its bound, on the CPU in F32, Metal and Vulkan, and is judged as heard: the CER or
  WER of the same sentences and seeds before and after, a synthesis transcribed by a recognition model. A bound is
  loosened only with the evidence add-model's [checks reference](../add-model/references/checks.md) asks for.

## 5. Check on Vulkan

Gains and defects differ by backend: a speed-up can show only on Metal, and a graph of a new shape can meet a Vulkan
defect that no earlier shape met. Run the stage checks, the bit-for-bit comparison and the A/B timings with VRAM on the
Windows machine (windows-check). Where a change gains on one backend and loses on another, choose by backend when the
model loads, and record the choice (adr skill).

## 6. Decide

- A change whose gain stays within the noise, the spread between rounds of one build, does not go in, however clean its
  code. It goes into the tracking issue's ideas not taken, with its numbers.
- A change that trades speed for memory, or the reverse, states both; when the trade is not clear, the user decides.
- The pull request's body and the commit's "Checked:" give the numbers before and after, with their conditions. A
  choice the code cannot show gets a decision record (adr skill); a measured value in a comment carries its condition
  and date.

## 7. The tracking issue

Keep one GitHub issue for an effort (one model's speed, a family's memory), opened before the first change: the
baseline with its conditions, each idea measured with its numbers, which went in and through which pull request, and
which did not and why. Update it as each measurement lands, so that nobody measures an idea again without knowing.
