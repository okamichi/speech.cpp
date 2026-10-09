# Irodori-TTS requests take the runtime's guidance, schedule and tail settings

## Context

The official runtime's request (`SamplingRequest` in `irodori_tts/inference_runtime.py` at 89f9d8f) takes 41 fields.
Those that need no other weights change only the sampler and the cut at the tail: the scales, the mode and the time
range of RF's guidance, what the branch without the speaker attends to, Sway Sampling, the truncation of the noise, the
temporal score rescaling, the scaling of the speaker's keys and values, and the cut's window and thresholds. The runtime
drops every RF setting for a MeanFlow checkpoint, some with a message and some without a word (truncation, rescaling,
the schedule, the time range). It ignores a setting that another one leaves without effect, clamps a length outside its
bounds, and cuts a text or a reference past its bounds.

The other fields return several takes from one batch (`num_candidates`, `decode_mode`), decide how the runtime computes
without changing the audio (`context_kv_cache`), repeat another field (`cfg_scale`, deprecated), are bounds of the model
(`min_seconds`, `max_seconds`, `max_ref_seconds`, `max_text_len`, `max_caption_len`), read Python's pickle
(`ref_latent`, `ref_latents`), or need weights or references of another kind (captions, no reference, several
references, speaker-inversion embeddings, LoRA adapters), which have records of their own.

Measurements of speech.cpp's Irodori-TTS hold only while a request that sets none of these settings gives the audio of
the runtime's default path that the port runs.

## Decision

- **The fields that need no other weights are options of the C API's vocabulary, by the runtime's names**:
  `cfg_scale_text`, `cfg_scale_speaker`, `cfg_guidance_mode`, `cfg_min_t`, `cfg_max_t`, `truncation_factor`,
  `rescale_k`, `rescale_sigma`, `speaker_uncond_mode`, `sway_coeff`, `keep_tail`, `tail_window_size`,
  `tail_std_threshold`, `tail_mean_threshold`, `speaker_kv_scale`, `speaker_kv_min_t` and `speaker_kv_max_layers`. Two
  differ from the runtime's: `t_schedule_mode` and `sway_coeff` are one option, since Sway Sampling at 0 is the linear
  schedule, bit for bit; and `trim_tail`, true by default, is `keep_tail`, false by default, since a boolean flag on the
  command line is true alone.
- **An RF file alone declares the guidance, the schedule and the speaker's scaling.** A MeanFlow file does not take
  them, so a request that sets one is `unsupported` there, as any option a model does not take; the cut at the tail is
  both models'.
- **Defaults come from the file**, where the checkpoint has its own (`irodori-tts.sampler.*`, `irodori-tts.tail.*`); the
  guidance mode and the speaker mode default to the runtime's default path. A default the file holds in float32 shows as
  its shortest decimal (0.05), and the request rounds it back.
- **The ranges stop at what a float32 holds**, since the family computes with these numbers as float32: at the largest
  float, and, for an option above 0, at the smallest normal float. A value past them, which would turn into infinity or
  0, is `out_of_range` when the request sets it, as any value outside a range, and the model information shows the
  bounds. Values within them whose products overflow are refused when the speech comes out not finite
  ([the record of finite audio](no-synthesis-passes-audio-that-is-not-finite.md)).
- **What the runtime would ignore or cannot run is refused before any work**, naming the option: one of the rescaling's
  two settings alone, `cfg_min_t` above `cfg_max_t`, the joint guidance with two scales that differ, a schedule that
  stops, and a value other than the default that another value leaves without effect (the guidance's mode, range and
  speaker mode with both scales at 0, speaker noise without a branch without the speaker, the tail's settings with
  `keep_tail`, the speaker's scaling's time and layers with a scale of 1). A caller that asks for something gets it or
  an error. The explicit RF option `irodori_compatible` accepts caption guidance without a caption and speaker
  guidance without a reference, matching the official runtime for callers that send these values unconditionally.
  It does not relax the other guidance checks.
- **The other fields are not offered.** A caller wanting several takes makes several requests with seeds of its own:
  each is a take of the same distribution, though not the runtime's candidates, as no seed of speech.cpp's generator
  gives the runtime's noise. The bounds stay the model's, from its file, past which speech.cpp refuses.
- **A request that sets none of these options runs the runtime's default path**, on every device: each default builds
  the graph and the order of float32 operations of a request without the option, which `checks/compare/same_audio.py`
  shows by comparing the callback's samples of two builds bit for bit. The noise of `speaker_uncond_mode` `noise` comes
  after the latent's from the request's seed, so that the latent's noise stays the seed's.

The alternatives were turned down:

- A check of each float32 option when the request runs, as Qwen3-TTS checks its divisors. The declared range would stay
  wider than what the model takes, and each option would need a check of its own.
- `t_schedule_mode` as an option of its own beside `sway_coeff`. Linear and Sway at 0 are the same schedule, and a
  coefficient without the mode, or the mode with 0, would be one more combination to refuse.
- `trim_tail`, true by default. The command line could not turn it off.
- Accepting RF's options on a MeanFlow file and ignoring them, as the runtime does. The caller gets audio other than it
  asked for and no sign of it.
- Ignoring a setting that has no effect, as the runtime does. The same, for a request that combines them.
- `num_candidates` as several streams of one request. Every entry point gives one stream of audio a request.

## Consequences

The worker, the server and `speech tts` take the options by their names without a change of their own, and the model
information lists them with their defaults and ranges. Each option's effect is checked against dumps of the official
runtime made with it (`reference/irodori-tts/dump.py` writes the options into `meta.json`, and `irodori-dit-check` and
`irodori-synthesis-check` read them). The default thresholds of the cut never cut v4.1's latent, whose silence has a
standard deviation near 0.7, so the cut only acts with thresholds a request raises.
