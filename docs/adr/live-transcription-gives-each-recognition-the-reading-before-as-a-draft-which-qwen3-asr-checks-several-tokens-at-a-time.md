# Live transcription gives each recognition the reading before as a draft, which Qwen3-ASR checks several tokens at a time

## Context

Live transcription recognizes an utterance again from its start each time audio comes (every 0.2 s of it while the
model keeps up), and once more whole when the utterance ends, so most of its work writes again the text the reading
before wrote. Qwen3-ASR writes a token a step, and its decoding is most of a recognition: 104 of 178 ms for 6.4 s of
Japanese with the 0.6B model in Q8_0 on an Apple M5. A step reads every weight once whatever the rows, so on a GPU a
step of five rows costs little more than one: a Q8_0 product of 4,096 × 14,336 took the same time for one to five
columns on the M5, and Qwen3-ASR's head 441 µs for one row and 626 µs for four on an RTX 2080 (2026-10-15).

A step that computes several rows rounds differently from a step of each, on every backend: Metal multiplies one column
with one kernel and two to eight with another, Vulkan on NVIDIA quantizes the activations of products of two to eight
columns to 8 bits and not those of one, and the CPU's products and attention take other paths for several rows. Where
two tokens score almost the same, the two can choose different ones.

## Decision

- **A recognition takes a request option `draft`, the text it is likely to write,** and Qwen3-ASR, which takes it,
  checks it: a step feeds the token written and the draft's next four tokens, picks on the device the token after each,
  keeps the draft's tokens up to the first its picks disagree with and the pick after them, and rewinds the cache past
  the rest. Where the text has gone astray of the draft, it takes up the draft again after the last two tokens it wrote,
  nearest where it was. The option steers nothing, and the text is the one written without it but at near ties.
- **Live transcription gives each reading the text of the reading before, and the recognition of the whole utterance
  its last reading**, where the model takes the option. A one-off recognition has no draft.
- **A draft may change a token where two score almost the same, and the documentation says so.** On the checks' dumps
  and readings simulated from them every 0.2 s (2026-10-15): with the 0.6B and 1.7B models in Q8_0, no reading of four
  recordings changed on the M5, 3 of 135 and 7 of 135 changed on an RTX 2080 with Vulkan, and 1 of 52 changed on the
  M5's CPU, each at the end of a reading cut in the middle of a word or between a comma and a full stop. Over
  /v1/realtime (below), 2 of the 58 completed transcripts on the RTX 2080 differed from the same samples recognized
  without a draft, a full stop at the end and タイマー for タイマ, and none of the 58 on the M5. The decoder's check
  compares the steps of a drafted decoding with the official implementation's as it compares those without a draft,
  and fails where they lie more than 12 dB further from it: the drafted steps lay up to 6.9 dB further on the RTX 2080,
  and 21.8 dB further with the rewind one row short.

measure/live_latency.py, FLEURS's Japanese and English test clips in two rounds of two sessions of five clips over
/v1/realtime with server_vad, Q8_0, the utterances the detection did not cut (2026-10-15): the p50 of the time from
the end of speech to the completed transcript went from 0.93 to 0.81 s (Japanese) and 0.93 to 0.82 s (English) with
the 0.6B model on an Apple M5, and from 1.29 to 0.99 s and 1.21 to 0.97 s with the 1.7B; on an RTX 2080 with Vulkan
from 0.82 to 0.76 s and 0.81 to 0.78 s with the 0.6B, and from 0.97 to 0.81 s and 0.93 to 0.80 s with the 1.7B. About
0.7 s of it is the detection's wait for the end of speech, so the recognition after the commit took about half the
time. In readings simulated from the checks' dumps, 86 to 95% of the drafted tokens were kept.

The alternatives were turned down:

- The model keeping the tokens it wrote last as the next recognition's draft, with no option. Requests would hold state
  between them, and a server's sessions on one model would draft from one another's text.
- Computing every step of the decoding with the same number of rows, padded, so that a draft changes nothing. Every
  recognition's text would change from today's at near ties, and a step without a draft would cost about a tenth more.
- Quantizing the activations of Vulkan's one-column Q8_0 products to 8 bits as those of several columns are, so that the
  two round alike on NVIDIA. It changes every model's output on Vulkan by the quantization of its activations.
- No drafts on Vulkan. Live transcription there would stay 1.7 to 2 times slower to its final text.

## Consequences

The option is part of the C API from 3.3; a model that does not take it refuses it, so the tools set it only where the
model takes it.
