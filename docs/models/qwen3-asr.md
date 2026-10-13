# Qwen3-ASR

This page describes speech.cpp's port of Qwen's [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), which writes
the text of speech in 30 languages, finds the language itself, and can be told the names and terms the audio holds.

| Name | Upstream | File |
|---|---|---|
| `qwen3-asr-0.6b` | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `Qwen3-ASR-0.6B-Q8_0.gguf`, 0.84 GB |
| `qwen3-asr-1.7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | `Qwen3-ASR-1.7B-Q8_0.gguf`, 2.18 GB |

```sh
speech asr qwen3-asr-1.7b utterance.wav                          # the language left to the model
speech asr qwen3-asr-1.7b --language ja utterance.wav            # Japanese, forced
speech asr qwen3-asr-1.7b --prompt "Claude Code、CI、渋谷" meeting.wav
speech worker qwen3-asr-1.7b
speech serve qwen3-asr-0.6b                                      # POST /v1/audio/transcriptions, with "prompt"
```

## What it does

- It writes the text the official qwen-asr package writes, with greedy decoding, and says which language it heard.
- It takes up to 1200 s at once, and longer audio in parts (below).

Not supported: timestamps, which need a second model, Qwen3-ForcedAligner-0.6B; and qwen-asr's streaming.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the 30 languages, which then steers the model, or `auto`, which lets it find the language |
| `prompt` | `""` | what the model is told of the audio before it hears it: the names and terms it may hold. The model uses it as background, not as instructions to follow |

The languages are `ar` `cs` `da` `de` `el` `en` `es` `fa` `fi` `fil` `fr` `hi` `hu` `id` `it` `ja` `ko` `mk` `ms` `nl`
`pl` `pt` `ro` `ru` `sv` `th` `tr` `vi` `yue` `zh`, and 22 Chinese dialects under `zh` or `auto`.

The result gives the language the model heard, or the forced one: the worker's and `speech asr --format json`'s
`languages`, and the server's `verbose_json` `language`. It gives none for audio without speech.

## Long audio

Audio over 1200 s is cut as qwen-asr cuts it, at the quietest point near each 1200 s, and each part is recognized alone.
The texts are joined, and the languages listed in the order of the audio, so a recording heard in Japanese and then in
English gives `ja` and `en`. The memory is that of the longest part.

A part stops at the 4096 tokens the model writes at most, and the request then says `model_limit`, with the text of
every part. A recording of over a few minutes may reach them: on 20 minutes of read Japanese, the 0.6B model repeats a
few sentences until the limit, as the official implementation does. To get the whole text of a long recording, cut it
at its pauses into pieces of a few minutes.

## Accuracy

Every stage is checked against transformers' tensors on 10 inputs from FLEURS' test split (ja_jp, en_us, cmn_hans_cn
and de_de, and two near-silent cuts), each with four requests, on an Apple M5:

| Weights | Text from the audio, 0.6B and 1.7B |
|---|---|
| F32 on the CPU, F32 or F16 on Metal | the official text on all 80 requests |
| Q8_0 on the CPU | the official text on 37 and 35 of 40 |
| Q8_0 on Metal | the official text on 37 and 40 of 40 |

Where a Q8_0 text differs, the official choice between two tokens was within the arithmetic's error of a tie.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER in Q8_0 is 11.89% for the
0.6B model and 9.49% for the 1.7B (7.05% and 4.67% accepting other spellings of the same words).

## Weight types

Each model comes in four types; `NAME:TYPE` fetches one (`speech asr qwen3-asr-1.7b:q6_k`). speech-bench on 2026-10-12,
on an Apple M5 with Metal: the CER on Common Voice 8.0's 4,483 Japanese test utterances, and after the slash the CER
accepting other spellings of the same words. The memory is the peak of one request.

| Type | 0.6B file | Memory | CER | 1.7B file | Memory | CER |
|---|---|---|---|---|---|---|
| Q8_0 | 0.84 GB | 1.00 GB | 11.89% / 7.05% | 2.18 GB | 2.34 GB | 9.49% / 4.67% |
| Q6_K | 0.68 GB | 0.84 GB | 12.01% / 7.02% | 1.68 GB | 1.85 GB | 9.52% / 4.72% |
| Q5_K | 0.59 GB | 0.75 GB | 12.49% / 7.57% | 1.41 GB | 1.58 GB | 9.85% / 4.96% |
| Q4_K | 0.51 GB | 0.67 GB | 14.23% / 8.66% | 1.16 GB | 1.32 GB | 11.21% / 5.96% |

In the 0.6B model's Q6_K, Q5_K and Q4_K files, the encoder and the projector stay Q8_0.

## Speed

speech-bench on 2026-10-08: Common Voice's Japanese test set, Q8_0, one utterance at a time, the language left to the
model.

| Model | Median, M5 / RTX 2080 | p90, M5 / RTX 2080 |
|---|---|---|
| 0.6B | 0.132 s / 0.087 s | 0.231 s / 0.158 s |
| 1.7B | 0.309 s / 0.155 s | 0.544 s / 0.286 s |

Beside llama.cpp b11246 with ggml-org's Q8_0 files on the same day, the medians are the same on the M5, where
llama.cpp's p90 is 8% lower, and speech.cpp's are 10% to 17% lower on the RTX 2080. 1338 s of audio take 120 s with
the 0.6B model, nearly all of it the first part's 4096 tokens, with a peak memory footprint of 2.03 GB.
