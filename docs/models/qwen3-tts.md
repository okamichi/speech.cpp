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

Each model comes in four types; `NAME:TYPE` fetches one (`speech tts qwen3-tts-0.6b:q6_k`). speech-bench on 2026-10-14,
two rounds, on an Apple M5 with Metal and an RTX 2080 with Vulkan: the 20 sentences of its `prompts/speak-ja-JP.json`
with `ono_anna` and seed 1, as Qwen3-ASR 1.7B hears them. The peak memory is the process's on the M5, whose GPU shares
it, and the VRAM the video memory in use on the RTX 2080. The CER of one type moves by a few points from seed to seed and
from one machine to the other, so only a larger difference, such as the 0.6B model's Q4_K, tells two types apart.

| Model | Type | File | CER, M5 / RTX 2080 | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| 0.6B | Q8_0 | 1.21 GB | 4.3% / 6.0% | 1.37 GB | 1.33 GB | 3.6 / 5.2 |
| 0.6B | Q6_K | 0.99 GB | 9.3% / 10.3% | 1.15 GB | 1.11 GB | 4.1 / 5.5 |
| 0.6B | Q5_K | 0.87 GB | 7.8% / 7.0% | 1.03 GB | 0.99 GB | 4.6 / 6.2 |
| 0.6B | Q4_K | 0.76 GB | 12.8% / 11.0% | 0.92 GB | 0.88 GB | 4.9 / 6.4 |
| 1.7B | Q8_0 | 2.29 GB | 4.3% / 8.8% | 2.45 GB | 2.41 GB | 2.5 / 4.3 |
| 1.7B | Q6_K | 1.82 GB | 3.5% / 5.3% | 1.99 GB | 1.94 GB | 2.9 / 4.9 |
| 1.7B | Q5_K | 1.57 GB | 3.8% / 3.7% | 1.73 GB | 1.69 GB | 3.3 / 5.5 |
| 1.7B | Q4_K | 1.33 GB | 7.6% / 8.0% | 1.49 GB | 1.45 GB | 3.9 / 5.8 |

## Speed

speech-bench on 2026-10-14: the 20 sentences of its `prompts/speak-ja-JP.json` with the speaker `ono_anna` and seeds 1
and 2, one request at a time through the worker, Q8_0, the medians of three rounds. The RTFx is the seconds of audio
over the seconds they took, so larger is faster. The CER is of the speech as Qwen3-ASR 1.7B hears it.

| Model | Device | Time to first audio, p50 | Time to first audio, p90 | RTFx | CER |
|---|---|---|---|---|---|
| 0.6B | Apple M5, Metal | 0.035 s | 0.051 s | 3.7 | 6.4% |
| 1.7B | Apple M5, Metal | 0.057 s | 0.107 s | 2.6 | 4.2% |
| 0.6B | RTX 2080, Vulkan | 0.024 s | 0.034 s | 5.2 | 6.0% |
| 1.7B | RTX 2080, Vulkan | 0.032 s | 0.046 s | 4.2 | 6.1% |

The time to first audio is the time to the first chunk. The 0.6B model often begins its speech with silence, so the
voice is heard later: on the M5 on 2026-10-08, half a second after the request in the median, from 0.24 s to 0.48 s by
speaker and up to 1.9 s. The official implementation does the same, 0.89 s in the median over 20 takes of `ono_anna`.
The 1.7B model begins with 0.10 s.
