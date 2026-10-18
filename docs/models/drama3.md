# Drama 3

This page describes speech.cpp's port of Fish Audio's [Drama 3](https://huggingface.co/fishaudio/drama-3), a speech
model of 35B parameters, 3B of them active, that speaks a text as one speaker or as a scene of several, with cues such as
laughter written in the text.

| Name | Upstream | File |
|---|---|---|
| `drama-3` | [fishaudio/drama-3](https://huggingface.co/fishaudio/drama-3) | `Drama-35B-A3B-v3-Q4_K.gguf`, 20.8 GB |

Built with Fish Audio. The weights are under the
[Fish Audio Community License 1.0](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE): commercial use is allowed
up to a yearly revenue of 500,000 US dollars, and a speech or voice-cloning service for others needs a separate agreement
with Fish Audio.

```sh
speech tts drama-3 --voice none -o out.wav "こんにちは、今日はいい天気ですね。"
speech tts drama-3 --voice none -o out.wav "<|speaker:0|> ねえ、聞いた？ <|speaker:1|> [laughs] うん、本当に驚いたよね。"
speech worker drama-3                         # behind the worker protocol
```

## Weight types

The model comes in five types; `drama-3` fetches Q4_K, and `NAME:TYPE` another (`speech tts drama-3:q8_0`). A model
needs its whole file in memory, the GPU's or, for the experts a GPU cannot hold, the CPU's
([Devices and memory](#devices-and-memory)), and holds the memory of its largest request from its load on, about 2.1 GB
of caches and graph buffers. speech-bench on 2026-10-16 on an H100 with Vulkan: the 20 sentences of its
`prompts/speak-ja-JP.json` in a voice file of one reference, seeds 1, 2 and 3, as Qwen3-ASR 1.7B hears them over the 60
takes. The CER of one type moves by a point or so from seed to seed, so the types are heard alike, and Q4_K speaks as
fast as Q8_0 in half the memory.

| Type | File | CER | The GPU's memory | RTFx |
|---|---|---|---|---|
| F16 | 71.6 GB | 1.1% | 69.1 GiB | 1.9 |
| Q8_0 | 38.5 GB | 1.3% | 38.2 GiB | 2.5 |
| Q6_K | 29.9 GB | 1.3% | 30.2 GiB | 2.5 |
| Q5_K | 25.2 GB | 0.9% | 25.8 GiB | 2.6 |
| Q4_K | 20.8 GB | 1.5% | 21.7 GiB | 2.6 |

## What it does

- **Speakers** are written in the text as `<|speaker:0|>` to `<|speaker:15|>`; text before the first tag is speaker 0's.
  The voice `none` gives every speaker a voice of the model's choosing, and a voice file gives its speakers the voices
  of its recordings ([Voices](#voices)).
- **Cues** in brackets, such as `[laughs]`, `[whispers]` or `[小声でささやく]`, and the words of the text steer how it is
  spoken, from the point where they are written. A cue that changes the voice, such as a whisper or a laugh, takes effect
  readily; a large change of feeling, such as shouting, follows the words of the text more than the cue.
- **A voice described in words**: with the voice `none`, a cue at the start such as `[deep, low male voice]` or
  `[明るく元気な若い女性の声]` steers the voice the model chooses for that speaker. A description of several traits may
  come out otherwise with some seeds; a voice file gives the same voice every time.
- **The language** is read from the text. `language` is checked against the model's languages and changes nothing.
- **A long text** is spoken in chunks of about 300 bytes of UTF-8 with at most three speakers, cut at sentence ends,
  then at commas, then at spaces, never inside a tag or a cue. Each chunk is prompted with the first and the latest chunk
  before it, so the voices and the way of speaking carry from one to the next.
- **Audio** is passed as it is made: the first piece after 4 frames, 0.19 s of speech, then pieces of 8, 16, 32, 64 and
  128 frames. The same request with the same seed gives the same samples.

## Voices

A voice file holds recordings of one or more speakers with the words each speaks, made by `speech voice`, and is added
with `--add-voice` as any voice is. Each recording needs its transcript and is speaker 0 unless `--speaker` gives
another; a voice file given to `speech voice` adds its recordings as they are, so a voice for a scene is put together
from voices made before:

```sh
speech voice drama-3 woman.wav --transcript "今日はいい天気ですね。" woman.voice.gguf
speech voice drama-3 man.wav --transcript "明日は雨の予報です。" man.voice.gguf
speech voice drama-3 woman.voice.gguf man.voice.gguf --speaker 0 --speaker 1 cast.voice.gguf
speech tts drama-3 --add-voice cast=cast.voice.gguf --voice cast -o scene.wav \
  "<|speaker:0|> 晴れたね。<|speaker:1|> 明日は雨だって。"
```

- The text names the speakers the voice file holds; a speaker it does not hold gets a voice of the model's choosing.
  The model information describes an added voice by its speakers (`speakers 0 and 1`).
- A recording is read at any rate, mixed to mono and resampled to 44.1 kHz. On the CPU, `speech voice`'s default, the
  codes of a 12.8 s recording were all the official encoder's; on Metal, 88 to 99% of each codebook's were.
- A voice file holds the codec's codes, not the audio, and works with every file of Drama 3 whose codec has the hash it
  carries, whatever its weight type.
- Every chunk of a text is prompted with all of the voice's recordings. The model runs them once and keeps them for the
  next chunks and requests in the same voice, until a request in another voice: with two recordings of 28 s in all on
  the Apple M5 with the Q4_K file, the first request's first audio came after 1.6 s and the next ones' after 0.25 to
  0.33 s. The same request with the same seed gives the same samples either way. A voice whose recordings leave a
  chunk no room for a text is refused.
- A WAVE file given to `--add-voice` is refused, since it lacks its words; make a voice file of it first. The page of
  `speech serve` makes no voices of Drama 3 for the same reason.

## Devices and memory

- **A GPU that holds the whole model computes it all**, as an Apple M5 with 32 GB does with the Q4_K file.
- **A GPU that does not hold it keeps the experts of as many layers as its free memory holds**, beside the rest of the
  model and what the largest request needs, and the CPU computes the experts of the other layers with the load's
  threads (`--threads`). The model then needs about 5.2 GB free on the GPU for its weights outside the experts (2.7 GB),
  the caches of the largest request and the graphs, and the CPU's memory holds the experts the GPU leaves out, 15 GB
  with an 8 GB GPU. The load says so:

  ```
  Drama 3 computes the experts of 35 of its 40 layers on the CPU, since Vulkan0 holds those of 5 beside the rest of the model; a GPU with more memory free speaks faster
  ```

- **`experts_on_cpu` false** refuses a request to such a model, for a caller that wants the GPU's speed or nothing.
- **Speed**: each layer whose experts the CPU computes costs about 1.15 ms of each frame's 46 ms of speech on an
  i9-9900K, so a GPU of 12 or 16 GB, which keeps about 18 or 27 layers' experts, speaks faster than an 8 GB one
  ([Speed](#speed)). A voice's recordings lengthen every chunk's prompt, which the CPU computes for the layers it holds:
  with an RTX 2080, a voice of about 8,000 rows of recordings took a minute before its first audio.

## Options

| Option | Default | Notes |
|---|---|---|
| `voice` | required | `none`, or a voice added from a voice file |
| `language` | `auto` | one of the model's languages; checked, not used |
| `seed` | drawn | 0 to 2^53 - 1 |
| `max_seconds` | 47.6 | the longest each chunk may speak, at most 95.1; a chunk that reaches it stops there, and the result says so |
| `top_k` | 30 | the most likely tokens a draw keeps, 1 to 4097 |
| `top_p` | 0.9 | the share of probability a draw keeps; the token that passes it is dropped |
| `temperature` | 1 | what the logits are divided by |
| `experts_on_cpu` | true | false refuses a request to a model whose GPU does not hold every expert ([Devices and memory](#devices-and-memory)) |

The defaults are those of the model card, and they apply to every draw of a frame. A text may have up to 8,192 tokens,
and a chunk's prompt up to 8,192 rows. A text is refused when it is empty, holds `<|im_start|>`, `<|im_end|>`,
`<|voice|>`, `<|text|>`, `<|interleave|>` or `<|semantic:`, which only the prompt writes, or names a speaker other than 0
to 15. `instructions`, `speed`, `seconds` and `repetition_penalty` are refused.

## Accuracy

Every stage is checked against the official implementation's tensors, on the CPU with the F32 file:

| Check | Result |
|---|---|
| Codec encoder and decoder | 96.9 dB SNR or more at every stage; the official codes of a 12.8 s reference |
| Text model, 40 layers from each one's own input | 120.2 dB or more at every stage; the official head's logits at 132.9 dB |
| Audio decoder | 107.7 dB or more in every layer and head |
| Sampler | the official probabilities within 3.2e-6 |
| Tokenizer, prompt, chunks | the official tokens, prompts and chunks of every case |

## Speed

speech-bench on 2026-10-16: the 20 sentences of its `prompts/speak-ja-JP.json` in a voice file of one reference of
12.8 s, seeds 1 to 3, one request at a time through the worker, the Q4_K file; the medians of the three seeds. The
voice's recordings run with the first request, and the later ones start from them. The RTFx is the seconds of audio over
the seconds they took, so larger is faster, and the CER is of the speech as Qwen3-ASR 1.7B hears it, over the 60 takes.

| Device | Experts on the CPU | Time to first audio, p50 | Time to first audio, p90 | RTFx | CER | Memory |
|---|---|---|---|---|---|---|
| Apple M5 (32 GB), Metal | none | 0.28 s | 0.32 s | 1.1 | 1.8% | 19.6 GiB |
| RTX 2080 (8 GB), Vulkan, i9-9900K | those of 35 of 40 layers | 0.58 s | 0.76 s | 0.57 | 2.4% | 6.9 GiB of the GPU's, 15.1 GiB of the CPU's |
| RTX 5090 (32 GB), Vulkan | none | 0.14 s | 0.16 s | 2.6 | 1.2% | 21.6 GiB of the GPU's |
| H100 (80 GB), Vulkan | none | 0.12 s | 0.14 s | 2.6 | 1.5% | 21.7 GiB of the GPU's |

On the same H100, the official implementation, fish-speech in BF16 with its experts in FP8, gives a CER of 2.2% with
seed 1 on the same sentences and reference, RTFx 1.9 with 44.2 GiB of the GPU's memory, and its first audio after
2.1 s, since it returns each sentence whole. SGLang Omni, the server of Fish Audio's own measurements, runs the official checkpoint with
CUDA at RTFx 6.9 and its first audio after 0.12 s, with a CER of 1.5% over three runs and 63 GiB of the GPU's memory
taken at its start; the unofficial int4 checkpoint in it gives RTFx 7.4 and 3.2%.
