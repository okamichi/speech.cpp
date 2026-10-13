# Qwen3-TTS

This page describes speech.cpp's port of Qwen's [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
CustomVoice models, which speak ten languages with nine named speakers and pass their audio as they make it.

| Name | Upstream | File |
|---|---|---|
| `qwen3-tts-0.6b` | [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice) | `Qwen3-TTS-12Hz-0.6B-CustomVoice-Q8_0.gguf`, 1.21 GB |
| `qwen3-tts-1.7b` | [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) | `Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf`, 2.29 GB |

```sh
speech tts qwen3-tts-0.6b --voice ono_anna --language ja -o out.wav "明日の東京は晴れです。"
speech tts qwen3-tts-1.7b --voice ryan --instructions "Speak slowly and softly." -o out.wav "Good night."
speech worker qwen3-tts-0.6b         # behind the worker protocol
speech serve qwen3-tts-0.6b          # behind OpenAI's speech API
```

## What it does

- It speaks with the model's nine speakers ([models.md](../models.md#voices)). The 1.7B model also follows an
  instruction of how to speak.
- Audio starts as soon as the first 0.08 s frame is made and keeps coming in small chunks, so a player can start on the
  first chunk. The chunks join without artifacts, and the same request with the same seed gives the same samples.
- A long text costs memory in proportion to its length.

Not supported: voice cloning and VoiceDesign, which are other Qwen3-TTS models.

## Options

| Option | Default | Notes |
|---|---|---|
| `voice` | required | one of the nine speakers |
| `language` | `auto` | `de` `en` `es` `fr` `it` `ja` `ko` `pt` `ru` `zh`; it steers the model |
| `seed` | drawn | 0 to 2^53 - 1 |
| `max_seconds` | none | stop the speech at this length |
| `instructions` | `""` | 1.7B only: how to speak, in words (`怒った口調で`, "Speak slowly and softly.") |
| `do_sample` | true | draw each token, or take the most likely one |
| `top_k` | 50 | the most likely tokens a draw keeps; 0 keeps all |
| `top_p` | 1 | the smallest set of tokens whose probability reaches it |
| `temperature` | 0.9 | what the logits are divided by |
| `repetition_penalty` | 1.05 | how much a token already taken is held back |
| `code_predictor_do_sample`, `code_predictor_top_k`, `code_predictor_top_p`, `code_predictor_temperature` | true, 50, 1, 0.9 | the same for the code predictor, which the official package calls `subtalker_dosample` and the rest |

The sampling options take transformers' names, and their defaults are the official ones. The model has no control of its
rate or length, so `speed`, `duration_scale` and `seconds` are refused, and so are `top_k`, `top_p` and `temperature`
when `do_sample` is false. A request stops at `max_seconds` or at the model's limit of 655 s, and the result says which.

## Accuracy

Every stage is checked against the official implementation's tensors:

| Check | Result |
|---|---|
| Codec decoder, CPU, F32 | 114 dB SNR from the official decoder |
| Codec decoder in a synthesis's chunks against the whole utterance | 132 dB SNR in F32 on the CPU, 64 dB with a Q8_0 file on Metal |
| Talker and code predictor, F32, CPU and Metal | the official choice on every frame; greedy decoding gives the official frames |
| Tokenizer | the official tokens on 27 texts, and the official text of 1191 sequences of ids |

## Weight types

Each model comes in four types; `NAME:TYPE` fetches one (`speech tts qwen3-tts-0.6b:q6_k`). speech-bench on 2026-10-12,
on an Apple M5 with Metal: the 20 sentences of its `prompts/speak-ja-JP.json` with `ono_anna` and seeds 1, 2 and 3, as
Qwen3-ASR 1.7B hears them. The memory is the peak of one request. The CER of one type moves by a few points from seed to
seed, so only a larger difference, such as the 0.6B model's Q4_K, tells two types apart.

| Type | 0.6B file | Memory | CER | 1.7B file | Memory | CER |
|---|---|---|---|---|---|---|
| Q8_0 | 1.21 GB | 1.37 GB | 6.14% | 2.29 GB | 2.44 GB | 6.25% |
| Q6_K | 0.99 GB | 1.15 GB | 6.03% | 1.82 GB | 1.98 GB | 4.42% |
| Q5_K | 0.87 GB | 1.03 GB | 7.35% | 1.57 GB | 1.72 GB | 6.02% |
| Q4_K | 0.76 GB | 0.92 GB | 13.60% | 1.33 GB | 1.48 GB | 7.19% |

## Speed

speech-bench on 2026-10-08: the 20 sentences of its `prompts/speak-ja-JP.json` with the speaker `ono_anna`, one
request at a time through the worker, Q8_0. The CER is of the speech as Qwen3-ASR 1.7B hears it.

| Model | Device | Median first audio | p90 first audio | Real-time factor | CER |
|---|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.038 s | 0.054 s | 0.31 | 5.8% |
| 1.7B | Apple M5, Metal | 0.060 s | 0.110 s | 0.42 | 3.0% |
| 0.6B | RTX 2080, Vulkan | 0.031 s | 0.039 s | 0.27 | 8.0% |
| 1.7B | RTX 2080, Vulkan | 0.039 s | 0.052 s | 0.32 | 2.8% |

On the RTX 2080 the 0.6B model takes 1.6 GB of video memory and the 1.7B 2.7 GB.

The first audio is the first chunk. The 0.6B model often begins its speech with silence, so the voice is heard later:
on the M5, half a second after the request in the median, from 0.24 s to 0.48 s by speaker and up to 1.9 s. The official
implementation does the same, 0.89 s in the median over 20 takes of `ono_anna`. The 1.7B model begins with 0.10 s.
