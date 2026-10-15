# Irodori-TTS

This page describes speech.cpp's port of Aratako's [Irodori-TTS](https://github.com/Aratako/Irodori-TTS), which speaks
Japanese in the voice of a reference recording, a sentence at a time.

| Name | Upstream | File |
|---|---|---|
| `irodori-tts-mf` | [Aratako/Irodori-TTS-v4.1-Small-MF](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small-MF) | `Irodori-TTS-866M-MF-v4.1-F16.gguf`, 1.92 GB |
| `irodori-tts` | [Aratako/Irodori-TTS-v4.1-Small](https://huggingface.co/Aratako/Irodori-TTS-v4.1-Small) | `Irodori-TTS-859M-v4.1-F16.gguf`, 1.91 GB |
| `irodori-tts-large` | [Aratako/Irodori-TTS-v4-Large](https://huggingface.co/Aratako/Irodori-TTS-v4-Large) | `Irodori-TTS-3.4B-v4-Q8_0.gguf`, 3.88 GB |

v4.1-Small-MF samples in 4 MeanFlow steps. v4.1-Small samples in Euler steps (40 by default) with guidance, which makes
it slower and lets a request tune more. v4-Large samples as v4.1-Small does, with a model four times as large whose text
encoder is Google's T5Gemma 2: it follows a voice described in words and a reference more closely, reads kanji slightly
less accurately, and is two to four times slower. v4-Large is under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), the others under the MIT
License.

```sh
speech voice irodori-tts-mf me.wav me.voice.gguf
speech tts irodori-tts-mf --add-voice me=me.voice.gguf --voice me -o out.wav "明日の東京は晴れです。"
speech tts irodori-tts --add-voice me=me.voice.gguf --voice me --steps 16 --speed 1.2 -o out.wav "明日の東京は晴れです。"
```

## Local Anime and Large checkpoints

The conversion tools also take full-precision v4.1-Anime and v4-Large checkpoints. Anime is available as a local
conversion; Large is also available as `irodori-tts-large` in the catalog. Use the root `model.safetensors` and `tokenizer/` of each repository; torchao's quantized checkpoints cannot
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

These conversions write the upstream layout 3. Reconvert GGUF files made with the fork's `text.encoder` and
`tokenizer.type` keys before using them with this build. Published layout 1 and 2 files remain readable.

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
  with every model in any type; a model of another codec refuses it.
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
| `steps` | 4 (MF), 40 (the others) | 1 or more |
| `seconds` | none | 0.5 to 30: fixes the length instead of predicting it |
| `duration_scale` | 1 | above 0: multiplies the predicted length |
| `speed` | 1 | 0.25 to 4: divides the length, as [Irodori-TTS-Server](https://github.com/Aratako/Irodori-TTS-Server) does with OpenAI's `speed` |
| `instructions` | `""` | the voice and the way of speaking in words, at most 512 tokens |
| `keep_tail` | false | keep the whole length, rather than end where the speech goes silent |
| `tail_window_size`, `tail_std_threshold`, `tail_mean_threshold` | 20, 0.05, 0.1 | how silence is found: this many frames with a spread and a mean under these |

v4.1-Small and v4-Large also take the runtime's sampling settings. v4.1-Small-MF takes none of them: MeanFlow folded the
guidance into its training.

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
| `speaker_kv_min_t`, `speaker_kv_max_layers` | 0.9, 12 (24 for v4-Large) | the steps and layers that `speaker_kv_scale` applies to |

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
v4-Large is checked the same way: all 164 texts and 18 captions give the official tokens on every device, and the whole
synthesis is 76.7 dB from the official on the CPU in F32, 15 to 16 dB on Metal and 58.7 dB on Vulkan with F16 on a short
sentence.

## Weight types

Each model comes in five types; `NAME:TYPE` fetches one (`speech tts irodori-tts:q8_0`). speech-bench on 2026-10-14,
two rounds, on an Apple M5 with Metal and an RTX 2080 with Vulkan: the 20 sentences of its `prompts/speak-ja-JP.json`
in a voice file of one reference, seed 1, v4.1-Small with 16 steps, as Qwen3-ASR 1.7B hears them. The peak memory is the
process's on the M5, whose GPU shares it, and the VRAM the video memory in use on the RTX 2080. The codec stays F32 in
every type, which is why the files shrink less than the other models'. The CER of one type moves by a point or two from
seed to seed and from one machine to the other, and more for v4.1-Small-MF.

| Model | Type | File | CER, M5 / RTX 2080 | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|---|
| v4.1-Small-MF | F16 | 1.92 GB | 6.5% / 6.3% | 2.04 GB | 2.30 GB | 7.7 / 18 |
| v4.1-Small-MF | Q8_0 | 1.21 GB | 9.5% / 8.6% | 1.33 GB | 1.59 GB | 7.4 / 18 |
| v4.1-Small-MF | Q6_K | 1.04 GB | 8.3% / 8.1% | 1.16 GB | 1.44 GB | 7.4 / 18 |
| v4.1-Small-MF | Q5_K | 0.95 GB | 5.0% / 5.0% | 1.07 GB | 1.33 GB | 7.1 / 18 |
| v4.1-Small-MF | Q4_K | 0.86 GB | 7.1% / 8.8% | 0.99 GB | 1.25 GB | 7.5 / 18 |
| v4.1-Small, 16 steps | F16 | 1.91 GB | 3.2% / 3.2% | 2.04 GB | 2.39 GB | 3.5 / 8.4 |
| v4.1-Small, 16 steps | Q8_0 | 1.20 GB | 3.5% / 3.2% | 1.33 GB | 1.69 GB | 3.4 / 8.5 |
| v4.1-Small, 16 steps | Q6_K | 1.03 GB | 3.2% / 3.3% | 1.17 GB | 1.54 GB | 3.3 / 8.3 |
| v4.1-Small, 16 steps | Q5_K | 0.94 GB | 3.0% / 3.2% | 1.07 GB | 1.43 GB | 3.3 / 8.3 |
| v4.1-Small, 16 steps | Q4_K | 0.86 GB | 4.2% / 3.2% | 1.00 GB | 1.35 GB | 3.4 / 8.5 |

v4-Large, on the same sentences at 16 steps on the M5 on 2026-10-14, with seeds 1, 2 and 3 for the CER and seed 1 for
the RTFx; the RTX 2080 ran F16 and Q8_0 alone, with seed 1. The name fetches Q8_0, which hears as F16 does in half the
memory and leaves room on a GPU of 8 GB.

| Type | File | CER, M5 | Peak memory, M5 | VRAM, RTX 2080 | RTFx, M5 / RTX 2080 |
|---|---|---|---|---|---|
| F16 | 6.96 GB | 4.5% | 7.2 GB | 7.5 GB | 0.97 / 3.6 |
| Q8_0 | 3.88 GB | 4.8% | 4.1 GB | 4.4 GB | 0.97 / 3.4 |
| Q6_K | 3.27 GB | 4.5% | 3.5 GB | | 0.94 |
| Q5_K | 2.93 GB | 4.9% | 3.2 GB | | 0.94 |
| Q4_K | 2.62 GB | 5.1% | 2.9 GB | | 1.00 |

## Speed

speech-bench on 2026-10-14: the 20 sentences of its `prompts/speak-ja-JP.json` in a voice file of one reference, seeds
1 and 2, one request at a time through the worker, F16, the medians of three rounds. The RTFx is the seconds of audio
over the seconds they took, so larger is faster. The CER is of the speech as Qwen3-ASR 1.7B hears it.

| Model | Device | Time to first audio, p50 | Time to first audio, p90 | RTFx | CER |
|---|---|---|---|---|---|
| v4.1-Small-MF, 4 steps | Apple M5, Metal | 0.22 s | 0.45 s | 7.7 | 6.7% |
| v4.1-Small, 16 steps | Apple M5, Metal | 1.15 s | 3.03 s | 3.5 | 3.2% |
| v4.1-Small-MF, 4 steps | RTX 2080, Vulkan | 0.12 s | 0.19 s | 18 | 6.4% |
| v4.1-Small, 16 steps | RTX 2080, Vulkan | 0.52 s | 1.14 s | 8.5 | 3.2% |

The time to first audio includes the whole sentence's sampling, so it grows with the sentence. With v4.1-Small's
default of 40 steps, the RTFx on the M5 is 1.8 with F16 weights, the median of five runs on each of three sentences of
5 to 18 s on the same day.

v4-Large in Q8_0, the same 20 sentences with seed 1 and one round, on 2026-10-14:

| Steps | Device | Time to first audio, p50 | Time to first audio, p90 | RTFx | CER |
|---|---|---|---|---|---|
| 16 | Apple M5, Metal | 4.40 s | 14.4 s | 0.97 | 5.8% |
| 40 | Apple M5, Metal | 10.7 s | 35.5 s | 0.42 | 5.0% |
| 16 | RTX 2080, Vulkan | 1.24 s | 3.80 s | 3.4 | 5.8% |
| 40 | RTX 2080, Vulkan | 3.14 s | 10.0 s | 1.4 | 5.0% |

The official runtime takes 36 GB at FP32 on the M5, where its RTFx is 0.31, and runs on the RTX 2080 at BF16 alone, at
an RTFx of 0.03.
