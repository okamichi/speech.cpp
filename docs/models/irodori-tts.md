# Irodori-TTS

This page describes speech.cpp's port of Aratako's [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), which speaks
Japanese in the voice of a reference recording, a sentence at a time.

| Name | Upstream | File |
|---|---|---|
| `irodori-tts-mf` | [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | `Irodori-TTS-866M-MF-v4.1-F16.gguf`, 1.92 GB |
| `irodori-tts` | [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | `Irodori-TTS-859M-v4.1-F16.gguf`, 1.91 GB |

v4.1-Small-MF samples in 4 MeanFlow steps. v4.1-Small samples in Euler steps (40 by default) with guidance, which makes
it slower and lets a request tune more.

```sh
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o out.wav "明日の東京は晴れです。"
speech tts irodori-tts --add-voice me=me.voice.gguf --voice me --steps 16 --speed 1.2 -o out.wav "明日の東京は晴れです。"
```

## Local Anime and Large checkpoints

The conversion tools also take full-precision v4.1-Anime and v4-Large checkpoints. These models are not in the
catalog. Use the root `model.safetensors` and `tokenizer/` of each repository; torchao's quantized checkpoints cannot
be used as GGUF inputs. Large uses T5Gemma2 and Anime uses ModernBERT.

From the repository root, with the upstream folders beside it:

```sh
reference/irodori-tts/convert-local.sh anime --check-only
reference/irodori-tts/convert-local.sh anime
reference/irodori-tts/convert-local.sh large --check-only
reference/irodori-tts/convert-local.sh large
```

`--check-only` checks checkpoint metadata and tensor mappings without converting weights or loading the codec.
Conversion checks the checkpoint's size and SHA-256, writes F32 under `models/converted/<target>/`, and then writes
F16 as `models/Irodori-TTS-v4.1-Anime.gguf` or `models/Irodori-TTS-v4-Large.gguf`. Large also produces Q8_0 as
`models/Irodori-TTS-v4-Large-Quantized.gguf`. The F32 file remains for stage checks and other quantizations.

The scripts use uv's locked Python environment. `IRODORI_MODEL_DIR` replaces the source folder;
`IRODORI_CODEC_DIR` names a folder containing the pinned DACVAE `weights.pth`. Without the latter, conversion fetches
the pinned codec through Hugging Face. Temporary tensor data goes under `models/.tmp`; allow space for both the
temporary data and the final F32 file during conversion.

```sh
build/speech serve models/Irodori-TTS-v4.1-Anime.gguf --device MTL0
build/speech serve models/Irodori-TTS-v4-Large.gguf --device MTL0
build/speech serve models/Irodori-TTS-v4-Large-Quantized.gguf --device MTL0
```

These conversions write layout 3, which requires a build with the local conversion changes. Published layout 2
Small and Small-MF files remain readable. Large's T5Gemma2 has stage checks on synthetic official weights; its full
synthesis still needs comparison against the official runtime after conversion.

## What it does

A diffusion transformer makes a whole sentence's speech, its length predicted beforehand, and a 48 kHz codec decodes it.

- The text is normalized as the official runtime does it. The 56 emoji the model reads as directions are kept: 🤭 a
  giggle, 😮‍💨 a sigh, 👂 a whisper and the rest of the runtime's `ALLOWED_ANNOTATION_EMOJIS`.
- Without a reference, the model chooses the voice itself (the voice `none`).
- A caption describes the voice and the way of speaking (the option `instructions`, the runtime's VoiceDesign).
- A speaker-inversion embedding, which the official training learns for a speaker, becomes a voice file.
- The speech ends where the model goes silent, rather than at the predicted length.

Not supported: LoRA adapters and the watermark. A seed gives other audio than the same seed in the official runtime.

## Voices

A voice is one of these:

- **`none`**: the model chooses the voice. It is listed first among the voices.
- **A reference WAVE file** of at most 120 s, at any rate. Adding it encodes it each time, which takes 0.4 to 0.7 s for
  10 s of audio.
- **A voice file**, which `speech voice` writes once from one or more references. It loads in under 0.03 s and works
  with both models in any type; a model of another codec refuses it.
- **A voice file of a speaker-inversion embedding**, which `speech voice` makes from the `.speaker.safetensors` file the
  official training saves. It works only with the model it was made for.

A voice file may join several references, up to 120 s together. Each is brought to the model's loudness, -16 LUFS, or to
another (`--lufs`), or kept as recorded (`--keep-loudness`). `--add-voice` and the worker's `add_voice` take one file at
the model's loudness, so a voice of several references or another loudness is made into a voice file first.

```sh
speech voice irodori-tts-mf take-1.wav take-2.wav take-3.wav --lufs -23 three-takes.voice.gguf
speech voice Irodori-TTS-859M-v4.1-F16.gguf my-speaker.speaker.safetensors my-speaker.voice.gguf
```

## Options

| Option | Default | Range and notes |
|---|---|---|
| `voice` | required | `none`, or a voice added since loading |
| `language` | `auto` | `ja`; only checked |
| `seed` | drawn | 0 to 2^53 - 1 |
| `steps` | 4 (MF), 40 (RF) | 1 or more |
| `seconds` | none | 0.5 to 30: fixes the length instead of predicting it |
| `duration_scale` | 1 | above 0: multiplies the predicted length |
| `speed` | 1 | 0.25 to 4: divides the length, as [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) does with OpenAI's `speed` |
| `instructions` | `""` | the voice and the way of speaking in words, at most 512 tokens |
| `keep_tail` | false | keep the whole length, rather than end where the speech goes silent |
| `tail_window_size`, `tail_std_threshold`, `tail_mean_threshold` | 20, 0.05, 0.1 | how silence is found: this many frames with a spread and a mean under these |

v4.1-Small (RF) also takes the runtime's sampling settings. v4.1-Small-MF takes none of them: MeanFlow folded the guidance
into its training.

| Option | Default | What it does |
|---|---|---|
| `cfg_scale_text`, `cfg_scale_speaker`, `cfg_scale_instructions` | 3, 5, 3 | the guidance's scales; 0 turns that condition's guidance off |
| `cfg_guidance_mode` | `independent` | `joint` guides every condition in one branch at one scale; `alternating` guides one condition a step, in turn |
| `cfg_min_t`, `cfg_max_t` | 0.5, 1 | the guidance runs while the step's time lies between them |
| `speaker_uncond_mode` | `mask` | `noise` replaces the speaker with noise in the unguided branch |
| `sway_coeff` | 0 | Sway Sampling: below 0 the steps gather near the noise, above 0 near the speech; the runtime's own `sway` uses -1 |
| `truncation_factor` | none | multiplies the starting noise |
| `rescale_k`, `rescale_sigma` | none | temporal score rescaling; both or neither |
| `speaker_kv_scale` | 1 | above 1, the speech follows the voice more closely |
| `speaker_kv_min_t`, `speaker_kv_max_layers` | 0.9, 12 | the steps and layers that `speaker_kv_scale` applies to |

`speech info` shows each option's range. A request whose options contradict each other, or set one that another leaves
without effect, is refused before any work, naming the option: `seconds` with `duration_scale`, for example, or the
guidance's settings with both scales at 0.

A request speaks at most 30 s. A text whose speech the model predicts to last longer is refused with `out_of_range`
(`text`), with the predicted length in the message, and so is a text of more than 256 tokens. A few sentences can pass
30 s, so a request should be one sentence; `speech tts` and `speech serve` speak a longer text a sentence at a time, and
the worker cuts a text it is refused after its sentences, its commas or its spaces. A
`speed` above 1 that brings the length within 30 s is followed.

## Long sentences

v4.1-Small-MF can slur or garble a few words in the middle of a long sentence and then speak normally again. In our
listening (F16 on Metal, and the official runtime in F32), it began at about 18 s of speech, about 100 characters, and
was more frequent past 20 s; the official runtime does the same from the same noise. A sentence within 30 s is still
spoken in one request, so to avoid it:

- Cut a long sentence at its commas yourself. `speech tts` and `speech serve` cut a text at the ends of its sentences,
  and at its commas only when a sentence is too long for one request.
- Give a reference or a voice file, so that every part keeps one voice. With the voice `none`, the model may choose
  another voice for each part.

## Instructions

`instructions` describes the voice and the way of speaking: 「落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。」.
With a reference, the speech keeps the reference's voice and follows the description where it can. With the voice
`none`, the description alone chooses the voice:

```sh
speech tts irodori-tts-mf --voice none --instructions "低く落ち着いた男性の声で、ゆっくりと読み上げてください。" \
    -o out.wav "明日の東京は晴れです。"
```

On RF models, `irodori_compatible: true` explicitly selects the official caption truncation at the model's token
limit and accepts inactive caption and speaker guidance. It defaults to false; otherwise long captions and inactive
nondefault guidance are refused. [The server's bridge mode](../server.md#irodori-bridge-compatibility) selects this
option for nested `irodori` requests and reports truncation in the WAV response headers.

## Streaming

The model makes a sentence's whole speech before any audio, then decodes it in windows and passes each one on: the first
0.48 s, then windows sized to stay ahead of playback. The windows do not change the audio, so a seed repeats its
samples.

## Accuracy

Every stage is checked against the official implementation's tensors on an Apple M5:

| Check | CPU, F32 | Metal, F32 | Vulkan, F16 model and F32 codec |
|---|---|---|---|
| Normalization and tokens, 164 texts | all equal | all equal | all equal on the 51 texts without emoji |
| Decoded audio | 119 dB SNR | 68 dB | 68 dB |
| Whole synthesis from the official noise | 75 to 110 dB, the official length | 22 to 61 dB, the same length | 58 dB (MF), the same length |

On a GPU the audio is the same speech rather than the same waveform, since the GPU's matrix kernels round differently.

## Weight types

Each model comes in five types; `NAME:TYPE` fetches one (`speech tts irodori-tts:q8_0`). speech-bench on 2026-10-12, on
an Apple M5 with Metal: the 20 sentences of its `prompts/speak-ja-JP.json` in a voice file of one reference, seeds 1, 2
and 3, v4.1-Small with 16 steps, as Qwen3-ASR 1.7B hears them. The memory is the peak of one request. The codec stays
F32 in every type, which is why the files shrink less than the other models'. The CER of one type moves by a point or
two from seed to seed; v4.1-Small-MF's is higher in every quantized type than in F16.

| Type | v4.1-Small-MF file | Memory | CER | v4.1-Small file | Memory | CER |
|---|---|---|---|---|---|---|
| F16 | 1.92 GB | 2.04 GB | 5.92% | 1.91 GB | 2.03 GB | 3.43% |
| Q8_0 | 1.21 GB | 1.33 GB | 8.35% | 1.20 GB | 1.32 GB | 4.09% |
| Q6_K | 1.04 GB | 1.16 GB | 7.35% | 1.03 GB | 1.16 GB | 3.65% |
| Q5_K | 0.95 GB | 1.07 GB | 7.74% | 0.94 GB | 1.07 GB | 4.37% |
| Q4_K | 0.86 GB | 0.99 GB | 7.57% | 0.86 GB | 0.98 GB | 4.54% |

## Speed

speech-bench on 2026-10-08: the 20 sentences of its `prompts/speak-ja-JP.json` in a voice file of one reference, seed
1, one request at a time through the worker, F16. The CER is of the speech as Qwen3-ASR 1.7B hears it.

| Model | Device | Median first audio | p90 first audio | Real-time factor | CER |
|---|---|---|---|---|---|
| v4.1-Small-MF, 4 steps | Apple M5, Metal | 0.25 s | 0.52 s | 0.17 | 6.1% |
| v4.1-Small, 16 steps | Apple M5, Metal | 1.24 s | 3.40 s | 0.34 | 3.2% |
| v4.1-Small-MF, 4 steps | RTX 2080, Vulkan | 0.12 s | 0.20 s | 0.07 | 7.1% |
| v4.1-Small, 16 steps | RTX 2080, Vulkan | 0.52 s | 1.12 s | 0.13 | 3.2% |

The first audio waits for the whole sentence's sampling, so it grows with the sentence. v4.1-Small's default of 40
steps takes about 2.4 times as long as 16. Over the same 20 sentences on the M5, the worker's memory peaks at 2.06 GB
with F16 weights and 1.35 GB with Q8_0, which speaks as fast.
