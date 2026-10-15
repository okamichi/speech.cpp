# Irodori-TTS runs both v4.1-Small-MF and v4.1-Small

## Context

v4.1-Small-MF is the MeanFlow distillation of v4.1-Small: it speaks in 4 steps, where v4.1-Small takes 40 Euler steps
with guidance. The official runtime spoke the 20 sentences of speech-bench's prompts/speak-ja-JP.json in the voice of
bright-young-woman-10s.wav, and Qwen3-ASR 1.7B transcribed them:

| Official runtime | Character error rate |
|---|---|
| v4.1-Small, 40 steps, two seeds | 2.16% and 2.16% |
| v4.1-Small-MF, 4 steps, six seeds and noises | 2.99% to 5.80%, 4.8% on average |
| audio.cpp v0.8.2, v4.1-Small at 16 steps, same reference | 2.16% |

Every run had the same 13 errors that are not the model's (他 for ほか, katakana for Slack). MF added one to three
sentences a run with a word that came out wrong (Bluetooth as ウルブドゥス), more often in English words and long
sentences, though some of these may be the recognizer's. MPS and the CPU gave the same transcriptions from the same
noise, so the device was not the cause.

## Decision

speech.cpp runs both. The two share every stage but the sampler, so v4.1-Small adds its Euler steps and its guidance, a
batch with a branch without each guiding condition at the steps whose time lies in the guidance's range, and a file of
its own. The steps are a request option, `steps`, whose default the file gives: 4 for MF and 40 for v4.1-Small.

## Consequences

The catalog names the two `irodori-tts-mf` and `irodori-tts`. A caller chooses between MF's speed and v4.1-Small's
accuracy and settings: the p50 time to first audio over the same 20 sentences is 0.23 s with MF and 1.12 s with v4.1-Small at
16 steps on an Apple M5 (Metal, F16), and 0.13 s and 0.49 s on an RTX 2080 (Vulkan, F16). Only v4.1-Small takes the
guidance's options ([the record of the runtime's
settings](irodori-tts-requests-take-the-runtimes-guidance-schedule-and-tail-settings.md)).
