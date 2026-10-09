# The option instructions carries Qwen3-TTS's instruct and Irodori-TTS's caption

## Context

Two families take a description in words of how to speak. Qwen3-TTS's official `generate_custom_voice()` takes
`instruct` ("怒った口調で"): it tokenizes it as the turn `<|im_start|>user\n{instruct}<|im_end|>\n` and puts the turn's
text embeddings, with no codec embedding added, before the rows of the prompt. It drops the instruction of a model whose
`config.json` gives `tts_model_size` "0b6", the 0.6B, although the 0.6B's model card shows an example with one. The
converter writes `general.size_label` from `tts_model_size` and from nothing else, "0b6" as `0.6B` and "1b7" as `1.7B`,
and refuses any other.

Irodori-TTS v4.1 is trained with a caption, a sentence that describes the voice and the way of speaking, which the
official runtime's request takes as `caption` and calls VoiceDesign without a reference. The runtime strips it with
Python's `str.strip()` and nothing else, tokenizes it with the text's tokenizer after `<s>`, runs it through the
ModernBERT-ja it shares with the text and the caption's own projector and norm, lets the DiT attend to it after the
speaker, gives the duration predictor the mean of its tokens instead of the null caption, and for RF adds a branch
without it to the guidance at `cfg_scale_caption` (3). A caption that strips to nothing is masked whole, which computes
the same as none. The runtime cuts a caption past 512 tokens.

OpenAI's speech API names the same input `instructions`, which describes the voice and the way of speaking. Most
requests carry none, and they should not pay for it.

## Decision

- **One option, `instructions`, OpenAI's name**, a string whose neutral value `""` gives none. A model that takes an
  instruction declares the option, and any other refuses a value other than `""` as `unsupported`. The server takes
  OpenAI's member as the option, as it takes every option by its name.
- **A Qwen3-TTS file takes it unless its size label is `0.6B`.** The size label is the official `tts_model_size` under
  the GGUF specification's name, so the reader decides from it what the official code decides from `tts_model_size`. The
  official code, which speech.cpp follows, outweighs the 0.6B's model card. The special tokens split the turn and the
  tokenizer reads `user\n{instruction}` as one text, as the official processor does. The instruction takes positions of
  the talker that the text would otherwise have, so a text and an instruction together beyond the longest text are
  refused, naming `instructions`.
- **An Irodori-TTS file that holds the caption's encoder takes it as the caption**, and RF's scale is
  `cfg_scale_instructions`, its default the file's `irodori-tts.sampler.cfg_caption`. The file's keys keep the runtime's
  name, `irodori-tts.caption.*`, since they are facts of the model. A request without a caption computes nothing of it:
  the caption's encoder does not run, the DiT gets no caption keys, and RF's batch has no caption branch, so the graph
  is the one a file without the caption builds. A caption that strips to nothing is no caption. A caption past
  `irodori-tts.caption.max_tokens` is refused as `out_of_range`, naming `instructions`, where the runtime cuts it, and
  `cfg_scale_instructions` without a caption is refused but at its default and 0. An RF caller explicitly choosing
  `irodori_compatible` instead gets the official token-prefix truncation and inactive guidance handling, because
  the bridge accepts captions longer than the model's limit and sends guidance even without a caption. The result
  reports truncation so this choice is observable.

The alternatives were turned down:

- The official names, `instruct` and `caption`. OpenAI's clients send `instructions`, an OpenAI client would send
  `caption` in `extra_body` and have its own `instructions` refused, and one input would have a name per family.
- A key of a new layout that says whether a Qwen3-TTS model takes an instruction. Every published file would be
  converted and uploaded again for a fact the size label already holds, and the file would hold it twice.
- Taking the instruction on the 0.6B, or taking it and dropping it as the official code does. The first runs the model
  in a way its official code never does; the second is a silent fallback.
- Computing the caption always and masking it when there is none, as the runtime does. Every request would pay for a
  ModernBERT pass and the DiT's extra keys for nothing.
- Cutting a long caption as the runtime does. The caller would get speech that follows part of what it asked.

## Consequences

The Qwen3-TTS 1.7B files take instructions and the 0.6B files refuse them, and `general.size_label`, which otherwise
only describes a model, decides it. `qwen3-tts-talker-check` compares the instruction's tokens, the prompt and every
stage with a dump of the official 1.7B with an instruction and greedy decoding. An Irodori-TTS caption costs, as
estimated, about 10 ms of ModernBERT and under 5 ms of the DiT's keys over MeanFlow's four steps on an Apple M5 (4 to 6%
of the median first audio of 0.23 s), and with RF a fourth branch at each guided step, which raises the DiT's work by a
quarter while t ≥ 0.5 and the sampling time by an estimated 10 to 25%; `measure/irodori-caption-timing.py` measures it.
An Irodori-TTS file of layout 1 refuses instructions, naming the caption's encoder it lacks ([the record of layout
2](irodori-tts-layout-2-adds-what-new-requests-need-and-layout-1-reads-as-a-file-without-it.md)).
