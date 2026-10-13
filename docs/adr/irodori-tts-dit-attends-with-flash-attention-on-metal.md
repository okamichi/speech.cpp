# Irodori-TTS's DiT attends with flash attention on Metal

## Context

Irodori-TTS's DiT attends from the latent's frames to themselves and to the text, speaker and caption conditions, in
twelve layers of 20 heads of 64, every step of the sampler, for every branch of RF's guidance at once. It can attend
through two matrix products and a softmax, or through ggml_flash_attn_ext. Metal's matrix kernel rounds both operands
to half precision, the keys and values included; its flash attention reads keys and values of float32 as they are,
rounds only the queries, and holds no scores. Vulkan's matrix product does not round the keys and values so. A request
that scales the speaker's keys and values in every layer makes the attention sensitive to the keys' rounding.

Measured on an Apple M5 on 2026-10-13, irodori-dit-check over the dumps and `speech worker` with v4.1-Small-MF (4 steps)
and v4.1-Small (16 steps) in F16 on the speech-bench sentences, the two builds alternating in four rounds:

| Metal | Products | Flash attention |
|---|---|---|
| Worst step over the RF dumps, least and median | 31.2 and 48.6 dB | 36.6 and 49.6 dB |
| rf-hai-speaker-kv-min1's worst step and sampled latent | 31.2 and 39.0 dB | 48.3 and 61.8 dB |
| The 20 sentences of 2 to 19 s, RTF, MF / RF | 0.1915 / 0.4057 | 0.1901 / 0.3948 |
| First audio of the three sentences of 16 to 19 s, MF / RF | 0.59 to 0.68 / 3.9 to 4.6 s | 0.56 to 0.63 / 3.6 to 4.3 s |
| 0.8 to 1.2 s of speech at a fixed length, total, MF / RF | 227 to 285 / 519 to 592 ms | 227 to 284 / 549 to 605 ms |

Qwen3-ASR 1.7B hears the same text in 58 of 60 of each model's audios on Metal (20 sentences, seeds 1 to 3), with a
character error rate of 4.75% and 4.69% for MF and 5.30% and 5.41% for RF.

Measured on an RTX 2080 with driver 591.86 on Vulkan, F16 files, the speech-bench sentences in three rounds:

| Vulkan | Products | Flash attention |
|---|---|---|
| mf-weather's worst step and sampled latent | 60.5 and 53.0 dB | 56.3 and 48.8 dB |
| rf-weather's worst step and sampled latent | 46.0 and 58.6 dB | 42.4 and 55.5 dB |
| rf-hai-speaker-kv-min1's worst step and sampled latent | 31.3 and 39.0 dB | 31.3 and 39.0 dB |
| The 20 sentences, RTF, MF / RF | 0.0720 / 0.1396 | 0.0718 / 0.1394 |

On the M5's CPU, MF in F32, the DiT sampled 27, 145 and 685 frames in 0.77, 3.36 and 14.9 s with the products and
0.82, 3.30 and 15.2 s with flash attention, and its error against the official was neither smaller nor larger.

## Decision

The DiT chooses its attention once, when the model loads, from its backend, and logs the choice: flash attention on
Metal where its backend computes ggml_flash_attn_ext for the DiT's heads with a half-precision mask
(`ggml_backend_supports_op`), and the two products on Vulkan, the CPU and any other backend. The mask that leaves a
condition out of a branch is in half precision for either attention. irodori-dit-check and irodori-synthesis-check take
an argument that forces either attention, and flash attention forced where the backend cannot compute it throws. The
speaker encoder, which runs once for a voice, keeps the two products: on Metal its condition lay 49.3 dB from the
official with flash attention and 50.9 dB with the products.

The alternatives were turned down:

- Flash attention on every GPU that computes it. On Vulkan it is no faster and about 4 dB further from the official,
  since the products there lose nothing to rounding the keys and values.
- Flash attention on every device. On the CPU it is no faster at any length the DiT meets, and no more precise.
- The products on every device. On Metal they round the keys and values to half precision, which costs 17 to 23 dB on
  a request that scales the speaker's keys, and they are slower for sentences of more than a few seconds.
- The products for short latents and flash attention for long ones. RF's guidance spends 15 to 30 ms more on 0.8 to
  1.2 s of speech with flash attention on Metal, and MF nothing; two attentions chosen by length would make the audio
  of a request depend on a threshold, for a few percent of a short request.

## Consequences

The audio of a request on Metal changes with this choice, within the rounding the checks allow: every Irodori stage
check passes with the same thresholds with either attention. On Vulkan and the CPU the audio is that of the products,
as before.
