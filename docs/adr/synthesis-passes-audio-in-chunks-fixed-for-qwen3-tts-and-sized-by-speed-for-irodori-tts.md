# Synthesis passes audio in chunks fixed for Qwen3-TTS and sized by speed for Irodori-TTS

## Context

A player starts on the first audio a synthesis passes and runs dry when the next comes later than the audio it holds
lasts. Measured through the worker on speech-bench's 20 Japanese sentences on 2026-10-07:

- Qwen3-TTS makes its speech a frame (0.08 s) at a time. With the first frame alone and then 4 frames a chunk, the
  second chunk of the 1.7B model arrived 0.096 s after the first on an RTX 2080 and 0.136 s after it on an Apple M5,
  where a frame takes about 24 ms and 34 ms to make, so a player that starts on the first chunk's 0.08 s ran dry for 16
  to 59 ms in every sentence. Its codec carries its state from one call to the next, but its samples still depend, by
  rounding, on how the frames are grouped into calls: decoding one sentence's codes in chunks of 1, 1, 2 and 4 frames
  instead of 1 and 4 changed 98% of its samples, by up to 1.7e-4 on Metal and 7.0e-4 on the CPU with the F16 weights of
  the codec. A call of the codec took about 16 ms on the M5's Metal whether it decoded 1 frame or 4 (a rough measure on
  a shared GPU).
- Irodori-TTS makes a sentence's whole latent, then its codec decodes it in windows and passes each on. With 12 frames
  (0.48 s) first and then 48 at a time, each window with 10 frames on either side decoded and dropped, the second
  window came 0.288 s after the first on the M5, with 0.19 s of the first window's audio left to play, and 0.32 s after
  it on the RTX 2080; a machine about half as fast as the M5 would run dry between them in every sentence. The decoder
  is convolutions without a cache whose receptive field reaches 8 frames on either side of a frame, so a window that
  decodes 8 frames on either side of its own and drops them gives the samples of one decode, bit for bit, whatever the
  sizes of the windows: 12 frames decode 28, 24 decode 40, 48 decode 64.

## Decision

- **Qwen3-TTS passes chunks of 1, 1, 2 and then 4 frames, fixed in advance** (`chunk_frames()`). The first frame alone
  lets the first audio come once one frame is made, the small chunks feed the player while the speech gets ahead of it,
  and 4 frames are the most a chunk holds, so that a cancel and the worker's silence last no longer than 4 frames. The
  same request with the same seed gives the same samples on every run, since the chunks do not depend on time.
- **Irodori-TTS sizes its decoder's later windows by its measured speed.** The first window is 12 frames. Each later
  window is the largest from 24 to 48 frames that the decoder expects to finish while the listener still has 0.1 s in
  hand, the listener taken to play from the first window on and the decoder's time per frame being the running estimate
  of the windows decoded, margins included. Where not even 24 frames finish in time, the window is 24 when decoding at
  it still gains on the listener and 48 when decoding is slower than real time, since smaller windows would only add
  margins to decode. 48 frames are never exceeded, so that a cancel and the silence between a worker's chunks stay
  bounded. The decision is one function without a clock (`next_window()`), which `irodori-window-check` checks on
  simulated machines.
- **Irodori-TTS's sizes may follow the clock because its audio does not depend on them**: `irodori-codec-check` checks
  seven patterns of windows against one decode, bit for bit, on the CPU, Metal and Vulkan (an RTX 2080), so a seed
  repeats its samples whatever the machine's load. Qwen3-TTS's samples depend on the chunking, so its sizes stay fixed.

The alternatives were turned down:

- Qwen3-TTS chunks that follow the measured speed. The grouping would change with the machine's load, so the same seed
  would give samples that differ by rounding from run to run, and every extra call costs the codec's 16 ms whatever it
  holds.
- A Qwen3-TTS codec whose samples do not depend on the grouping, with calls of one shape and keys and values kept by
  position. It rewrites the codec's state for a difference far below what anyone hears.
- A first chunk of 2 frames. The first audio would come a frame later.
- Chunks of 1 frame throughout. Each would pay the codec's 16 ms.
- A fixed later Irodori-TTS window smaller than 48, such as 24 everywhere. Fast machines would decode 25% more per
  second of audio for nothing and break the stream into twice the chunks.
- A floor of 12 for Irodori-TTS's later windows. It decodes 75% more per second of audio than 48, which a machine that
  decodes 48-frame windows barely faster than real time cannot afford.
- Later Irodori-TTS windows past 48 frames, up to 96, whenever their expected decoding fits the second window's wait.
  v4.1-Small-MF F16 spoke speech-bench's 20 sentences 7.0% faster on an Apple M5 (2026-10-13), but only by spending the
  listener's slack: its second window arrived with 0.05 to 0.1 s of audio left to play instead of 0.16 to 0.19 s, and a
  window seen taking 1.64 times its estimate on a shared GPU would have run the stream dry. Growing a window only once
  the listener holds the audio of the frames it decodes kept that slack but gained 1.4%, within the noise, since only
  sentences longer than about 6.2 s reach a third window, and each frame a window decodes adds about 3.5 MiB to the
  codec's buffers on the device (226 MiB at 48 frames, 394 MiB at 96).

## Consequences

`qwen3-tts-codec-check` compares Qwen3-TTS's decode in chunks with the whole decode as an SNR. A machine that makes
frames more slowly than the measured ones can still run a player dry, which then buffers. Irodori-TTS's windows of one
request depend on the machine and its load, and its samples do not.
