# FastConformer

This page describes speech.cpp's port of NVIDIA NeMo's [FastConformer](https://arxiv.org/abs/2305.05084) recognizers:
two parakeet models and ReazonSpeech, which write the text of a recording.

| Name | Upstream | Languages | File |
|---|---|---|---|
| `reazonspeech-v2` | [reazon-research/reazonspeech-nemo-v2](https://huggingface.co/reazon-research/reazonspeech-nemo-v2) | ja, with punctuation | `reazonspeech-nemo-619M-v2-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt_ctc-0.6b-ja` | [nvidia/parakeet-tdt_ctc-0.6b-ja](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) | ja | `parakeet-tdt_ctc-0.6B-ja-Q8_0.gguf`, 0.66 GB |
| `parakeet-tdt-0.6b-v3` | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | 25 European languages, with punctuation and capitals | `parakeet-tdt-0.6B-v3-F16.gguf`, 1.26 GB |

parakeet-tdt-0.6b-v3's languages are those of its model card: `bg` `cs` `da` `de` `el` `en` `es` `et` `fi` `fr` `hr`
`hu` `it` `lt` `lv` `mt` `nl` `pl` `pt` `ro` `ru` `sk` `sl` `sv` `uk`.

```sh
speech asr parakeet-tdt_ctc-0.6b-ja utterance.wav                  # the text on stdout
speech asr parakeet-tdt-0.6b-v3 --timestamps utterance.wav         # with the times of its segments
speech asr reazonspeech-v2 --vad silero-vad meeting.wav            # a recording of minutes, by its regions of speech
speech asr reazonspeech-v2 --decoding greedy meeting.wav           # faster, with greedy decoding
speech worker parakeet-tdt_ctc-0.6b-ja                             # a recognition worker
speech serve parakeet-tdt-0.6b-v3                                  # POST /v1/audio/transcriptions
```

## What it does

- Each model writes the text NeMo's `transcribe()` writes by default: ReazonSpeech with a beam search, the parakeet models
  with greedy decoding.
- A request is recognized whole, however long, but a stretch that holds several sentences loses whole sentences, as in
  NeMo itself: on minutes of Common Voice ja clips joined with pauses, reazonspeech-v2 dropped 522 of 600 and
  parakeet-ja 266. Recognize a long recording by the regions where someone speaks, with `--vad silero-vad` or
  `chunking_strategy` (the [record](../adr/transcription-by-regions-recognizes-each-region-where-someone-speaks-alone-in-the-tools-and-joins-the-texts.md)
  has the measurement). ReazonSpeech's time and memory grow with the length; the parakeet models attend over the whole
  request, so theirs grow with its square.
- parakeet-v3 finds the language of the audio itself. None of the models takes a language: a request's `language` is only
  checked against the model's languages.
- With `timestamps`, the result also has the tokens and segments with their times in seconds. A segment ends at a full
  stop, a question mark or an exclamation mark.
- On a GPU, the encoder runs on the GPU, and the decoding runs on the CPU where the file's decoding weights are
  quantized (Q8_0, Q6_K, Q5_K and Q4_K files) and on the GPU where they are F16 or F32: each step of the decoding is a
  small computation, which the CPU finishes before a round trip to the GPU would with quantized weights but not with F16
  ones. `decode_on_cpu` chooses either way per request. When a request first decodes on the CPU, the model copies the
  weights the decoding reads into the CPU's memory and keeps them, 12 MB for the Q8_0 files of reazonspeech-v2 and
  parakeet-ja and 35 MB for the F16 file of parakeet-v3, and the CPU computes with `--threads`. The CPU's arithmetic
  differs from the GPU's, so a long recording can come out a word different where two words score almost the same.

Not supported: the CTC head of parakeet-tdt_ctc-0.6b-ja, which NeMo does not decode with by default.

## Options

| Option | Default | Notes |
|---|---|---|
| `language` | `auto` | one of the model's languages; only checked |
| `timestamps` | false | add the times of the tokens and segments |
| `decoding` | `beam` | reazonspeech-v2 only: `beam` or `greedy`. Greedy is faster and can write another text, often with fewer commas and full stops |
| `decode_on_cpu` | true for Q8_0 and K-quant files, false for F16 and F32 | for a model on a GPU, decode on the CPU (true) or on the GPU (false). A model on the CPU decodes there either way |

## Accuracy

Every stage is checked against NeMo's tensors on utterances of FLEURS' test split, on an Apple M5:

| Model | Inputs | Text from the audio, CPU F32, CPU F16 and Metal |
|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | 3 ja_jp utterances, 6 to 26 s | NeMo's text on all three |
| parakeet-tdt-0.6b-v3 | 12 utterances of en_us, de_de, fr_fr and es_419, 6 to 23 s | NeMo's text on all twelve |
| reazonspeech-nemo-v2 | 8 ja_jp utterances and 2 joined inputs of 65 s and 311 s | NeMo's text on all ten, with beam search and with greedy decoding |

The times of tokens and segments are NeMo's exactly with F32 weights, and with F16 weights decoding on Metal. Decoding on
the CPU with F16 weights, a token can fall one frame (80 ms) from NeMo's where two frames score almost the same: one
token in 2 of reazonspeech-nemo-v2's 20 decodings.

On the 4,483 utterances of Common Voice 8.0's Japanese test set, trimmed to the voice, the CER in F16 is 7.88% for
parakeet-tdt_ctc-0.6b-ja, 12.03% for reazonspeech-nemo-v2 with beam search and 12.49% with greedy decoding (2.98%,
7.13% and 7.69% accepting other spellings of the same words). parakeet-tdt-0.6b-v3's WER on 300 English utterances of
FLEURS is 8.76%.

## Weight types

Each model comes in five types; `NAME:TYPE` fetches one (`speech asr parakeet-tdt-0.6b-v3:q8_0`). speech-bench on
2026-10-12, on an Apple M5 with Metal: the CER on Common Voice 8.0's 4,483 Japanese test utterances, and after the slash
the CER accepting other spellings of the same words, and parakeet-tdt-0.6b-v3's WER on 300 English utterances of
FLEURS. The memory is the peak of one request. The file and the memory are the same for the three models within
0.03 GB.

| Type | File | Memory | parakeet-tdt_ctc-0.6b-ja | reazonspeech-nemo-v2, beam search | parakeet-tdt-0.6b-v3 |
|---|---|---|---|---|---|
| F16 | 1.24 GB | 1.37 GB | 7.88% / 2.98% | 12.02% / 7.13% | 8.76% |
| Q8_0 | 0.66 GB | 0.78 GB | 7.93% / 3.00% | 12.05% / 7.15% | 9.06% |
| Q6_K | 0.51 GB | 0.63 GB | 7.88% / 3.00% | 11.98% / 7.12% | 8.87% |
| Q5_K | 0.43 GB | 0.54 GB | 7.79% / 2.97% | 11.99% / 7.14% | 9.13% |
| Q4_K | 0.36 GB | 0.47 GB | 7.94% / 3.01% | 12.16% / 7.23% | 9.51% |

In the Q6_K, Q5_K and Q4_K files the prediction network and the joint stay Q8_0.

## Speed

speech-bench on 2026-10-12, one request at a time, with the files the names fetch: Q8_0 for parakeet-tdt_ctc-0.6b-ja
and reazonspeech-nemo-v2 on the first 100 utterances of Common Voice 8.0's Japanese test set (99 kept by VAD), and F16
for parakeet-tdt-0.6b-v3 on 300 English utterances of FLEURS (299 kept).

| Model | Utterances | Median, M5 / RTX 2080 | p90, M5 / RTX 2080 |
|---|---|---|---|
| parakeet-tdt_ctc-0.6b-ja | Common Voice ja | 0.051 s / 0.040 s | 0.072 s / 0.079 s |
| reazonspeech-nemo-v2, beam search | Common Voice ja | 0.081 s / 0.107 s | 0.127 s / 0.197 s |
| reazonspeech-nemo-v2, greedy | Common Voice ja | 0.056 s / 0.046 s | 0.083 s / 0.088 s |
| parakeet-tdt-0.6b-v3 | FLEURS en | 0.089 s / 0.100 s | 0.144 s / 0.143 s |

A long recording is recognized whole. ReazonSpeech and parakeet-tdt_ctc-0.6b-ja use global attention, whose memory
grows with the square of the input length. Use `speech asr --vad silero-vad` to transcribe long recordings by region.
