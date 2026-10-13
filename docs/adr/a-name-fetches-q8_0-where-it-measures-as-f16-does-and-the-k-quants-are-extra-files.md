# A name fetches Q8_0 where it measures as F16 does, and the K-quants are extra files

## Context

Each model's repository on Hugging Face holds a file of every type `speech quantize` writes from its F32 file but F32,
and the catalog gives the type a model's name alone fetches
([the record of the catalog](a-model-is-named-from-a-pinned-catalog-and-no-command-chooses-one-for-the-user.md)).
The machines speech.cpp runs on include laptops with 8 GB, and a smaller file takes less memory: a request's peak memory
footprint is the file's size and 0.1 to 0.17 GB more on an Apple M5 with Metal. Quantization costs accuracy unevenly,
measured by speech-bench on 2026-10-12 on that M5: on Common Voice 8.0's 4,483 Japanese test utterances, Q8_0 against
F16 moved the CER of parakeet-tdt_ctc-0.6b-ja by +0.05 points and of reazonspeech-nemo-v2 by +0.03; on 300 English
utterances of FLEURS it moved parakeet-tdt-0.6b-v3's WER by +0.30; on speech-bench's 20 Japanese sentences with seeds 1,
2 and 3, as Qwen3-ASR 1.7B hears them, it moved Irodori-TTS v4.1-Small-MF's CER from 5.92% to 8.35% and v4.1-Small's
(16 steps) from 3.43% to 4.09%, both past the range of F16's three seeds. Q4_K took Qwen3-TTS 0.6B's CER from 6.14% to
13.60% and Qwen3-ASR 0.6B's from 11.89% to 14.23%. A synthesis model's CER moves by a few points from seed to seed.

## Decision

- **A name fetches Q8_0 where Q8_0 measures as F16 does, and F16 where it does not.** For a recognizer, Q8_0 is within
  0.2 points of F16's error rate on the set its accuracy is reported on, and on Common Voice also within 0.2 points of
  the rate that accepts other spellings of the same words; for a synthesis model, Q8_0's CER over three seeds lies
  within the range of F16's seeds. reazonspeech-v2 and parakeet-tdt_ctc-0.6b-ja fetch Q8_0; parakeet-tdt-0.6b-v3 and
  both Irodori-TTS models fetch F16. Qwen3-TTS and Qwen3-ASR fetch Q8_0, and their repositories hold no F16
  file to measure it against.
- **Q6_K, Q5_K and Q4_K are extra files** of every model but Silero VAD, published with their size, memory and measured
  accuracy on the model card and the family's page, and never what a name alone fetches, even where they measure as
  well as Q8_0: `NAME:TYPE` asks for one.
- **Every type is published, the ones that measure worse included**, with their numbers, so that a user who needs the
  memory chooses knowing the cost.

The alternatives were turned down:

- F16 for every model that has one. reazonspeech-v2 and parakeet-tdt_ctc-0.6b-ja would take 1.37 GB of memory where
  0.78 GB gives the same error rate.
- Q8_0 for every model whatever it measures. Irodori-TTS's speech would be heard worse by default.
- A K-quant as what a name fetches where it measures as well. A type that departs on another set or another device
  would reach every user who did not choose it.
- Withholding a type that measures much worse, such as Qwen3-TTS 0.6B's Q4_K. A user short of memory would have no file
  of that model at all.

## Consequences

A model whose fetched type changes names another file in the catalog of the next release: the file of the earlier type
stays in the repository, and a release before keeps fetching it at the revision it pins. A model added to the catalog is
measured in F16 and Q8_0 before its type is chosen.
