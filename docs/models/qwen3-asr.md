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
| `draft` | `""` | the text the recognition is likely to write, such as an earlier recognition of the beginning of the same audio. The model checks several of its tokens at once and keeps those it would write, so the text is the same, sooner where the draft is right; where two tokens score almost the same, it can choose the other. Live transcription gives each reading the reading before |

The languages are `ar` `cs` `da` `de` `el` `en` `es` `fa` `fi` `fil` `fr` `hi` `hu` `id` `it` `ja` `ko` `mk` `ms` `nl`
`pl` `pt` `ro` `ru` `sv` `th` `tr` `vi` `yue` `zh`, and 22 Chinese dialects under `zh` or `auto`.

The result gives the language the model heard, or the forced one: the worker's and `speech asr --format json`'s
`languages`, and the server's `verbose_json` `language`. It gives none for audio without speech.

## Long audio

Audio over 1200 s is cut as qwen-asr cuts it, at the quietest point near each 1200 s, and each part is recognized alone.
The texts are joined, and the languages listed in the order of the audio, so a recording heard in Japanese and then in
English gives `ja` and `en`. The peak memory is that of the longest part.

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
0.6B model and 9.49% for the 1.7B (7.05% and 4.67% accepting other spellings of the same words), on the M5 and within
0.01 points on the RTX 2080.

## Weight types

Each model comes in four types; `NAME:TYPE` fetches one (`speech asr qwen3-asr-1.7b:q6_k`). speech-bench, two
rounds, on an Apple M5 with Metal and an RTX 2080 with Vulkan on 2026-10-16: the CER on the first 500 utterances of Common Voice
8.0's Japanese test set, on which the CER is higher than on the whole set, and after the slash the CER accepting other
spellings of the same words. The CER is the M5's, and the RTX 2080's is within 0.1 points of it. The peak memory is the
process's on the M5, whose GPU shares it, and the VRAM the video memory in use on the RTX 2080.

| Model | Type | File | CER | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| 0.6B | Q8_0 | 0.84 GB | 14.17% / 6.60% | 1.01 GB | 1.13 GB | 34 / 50 |
| 0.6B | Q6_K | 0.68 GB | 14.41% / 6.55% | 0.85 GB | 0.97 GB | 38 / 48 |
| 0.6B | Q5_K | 0.59 GB | 14.89% / 7.04% | 0.76 GB | 0.88 GB | 41 / 57 |
| 0.6B | Q4_K | 0.51 GB | 17.13% / 8.65% | 0.68 GB | 0.79 GB | 45 / 57 |
| 1.7B | Q8_0 | 2.18 GB | 12.43% / 4.35% | 2.35 GB | 2.47 GB | 14 / 26 |
| 1.7B | Q6_K | 1.68 GB | 12.17% / 4.28% | 1.86 GB | 1.98 GB | 16 / 34 |
| 1.7B | Q5_K | 1.41 GB | 12.75% / 4.67% | 1.59 GB | 1.71 GB | 18 / 35 |
| 1.7B | Q4_K | 1.16 GB | 14.03% / 5.73% | 1.33 GB | 1.45 GB | 21 / 39 |

In the 0.6B model's Q6_K, Q5_K and Q4_K files, the encoder and the projector stay Q8_0.

## Speed

speech-bench on 2026-10-16: the 4,483 utterances of Common Voice 8.0's Japanese test set (4,479 kept by VAD), Q8_0,
one utterance at a time, the language left to the model, one round. The RTFx is the seconds of audio over the seconds
they took, so larger is faster.

| Model | p50 latency, M5 / RTX 2080 | p90 latency, M5 / RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|
| 0.6B | 0.107 s / 0.072 s | 0.188 s / 0.131 s | 32 / 47 |
| 1.7B | 0.257 s / 0.140 s | 0.456 s / 0.257 s | 13 / 24 |

Beside llama.cpp b11246 with ggml-org's Q8_0 files, on the first 500 of those utterances in three rounds on the same
days: on the M5, speech.cpp's p50 latencies are 4% to 6% lower and llama.cpp's p90 2% lower; on the RTX 2080,
speech.cpp's latencies are 10% to 31% lower. llama.cpp's peak memory on the M5 is 6.07 GB with the 0.6B model and
7.59 GB with the 1.7B, against speech.cpp's 1.01 GB and 2.35 GB, and its VRAM on the RTX 2080 1.95 GB and 3.32 GB,
against 1.13 GB and 2.47 GB. Its CER is 0.1 and 0.5 points lower.

Live transcription (`speech serve`'s `/v1/realtime` with `server_vad`, `speech asr --live`) gives each reading the
one before as its draft. On 2026-10-15, FLEURS's Japanese and English test clips, Q8_0: the p50 of the time from the
end of speech to the completed transcript is 0.81 s with the 0.6B model and 0.97 to 0.99 s with the 1.7B on the M5, and
0.76 to 0.78 s and 0.80 to 0.81 s on the RTX 2080, of which about 0.7 s is the detection's wait for the end of speech.
On the RTX 2080, 2 of 58 completed transcripts differed from the same audio recognized without a draft, at a near tie
(a full stop at the end, タイマー for タイマ).

On the M5 on 2026-10-07, 1338 s of audio took 120 s with the 0.6B model, nearly all of it the first part's 4096 tokens,
with a peak memory of 2.03 GB.
