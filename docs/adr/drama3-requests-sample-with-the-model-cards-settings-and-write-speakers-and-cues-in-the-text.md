# Drama 3 requests sample with the model card's settings and write speakers and cues in the text

## Context

Drama 3 has two official implementations, which take different requests. fish-speech's server (at 5415669) takes a
text, references or a stored reference's name, a seed, `temperature` and `top_p` (0.1 to 1), `max_new_tokens` for
each chunk, `repetition_penalty`, which it ignores, `single_chunk`, which speaks the whole text in one pass,
`chunk_length`, `normalize` and `latency`, which nothing of Drama 3 reads, `streaming`, `format` and
`use_memory_cache`; it samples with top-k 30, which a request cannot change, and takes temperature 0.7 and top-p 0.7
when a request sets none, the Fish Audio SDK's defaults for its older model too. SGLang-Omni takes `temperature`,
`top_p` and `top_k`, by default 1.0, 0.9 and 30, `max_new_tokens`, a `repetition_penalty` of 1 alone, and references
for speaker 0, and refuses a seed, a language and instructions. The model card measured the model with 1.0, 0.9 and 30.
The checkpoint has no generation configuration.

Both draw the main token of each frame from the end id and the 4,096 semantic ids alone, and draw it again with
temperature 1.0, top-k 30 and top-p 0.95 when the first draw repeats one of the last 16 main tokens, settings the
training recipe fixes. The text carries the speakers as `<|speaker:N|>` tags and the cues the model acts on, such as
`[sighs]` or `[laughs]`, as words in brackets; neither implementation takes a language.

## Decision

- **`temperature`, `top_p` and `top_k`, by default the model card's 1.0, 0.9 and 30**, under the names Qwen3-TTS's
  requests use ([the record of Qwen3-TTS's sampling](qwen3-tts-requests-set-sampling-under-transformers-names.md)).
  The file holds the defaults, and the converter names the model card as their source. The redraw's settings are the
  model's and no option.
- **`seed`**, as fish-speech takes it; the same seed gives the same speech on the same device.
- **`max_seconds` bounds each chunk's speech**, as `max_new_tokens` bounds a chunk in fish-speech: by default its 1,024
  frames (47.6 s), and at most 2,048 frames (95 s), the most it gives a text it speaks in one pass.
- **Speakers and cues are written in the text**, as both implementations take them: no option names a speaker or an
  instruction, and a request that sets `instructions` is refused, naming it.
- **A request's `language` is only checked** against the model's languages, as Irodori-TTS's and FastConformer's are
  ([the record of languages](languages-are-shortest-iso-639-codes-in-the-file-and-bcp-47-tags-in-requests.md)): the
  model reads the language from the text, and an OpenAI client sends one as a hint.
- **The references and the stored reference are the voice**
  ([the record of Drama 3's voice files](a-drama3-voice-file-holds-the-codes-and-transcripts-of-the-references-of-each-of-its-speakers.md)).
- **Not offered:** `repetition_penalty`, which neither implementation applies; `chunk_length`, `normalize` and
  `latency`, which nothing of Drama 3 reads; `streaming`, `format` and `use_memory_cache`, since the library always
  passes the audio on as it is made and the tools and the server choose the format; and `single_chunk`, since a long
  text is spoken in chunks
  ([the record of long texts](a-drama3-request-speaks-a-long-text-in-chunks-each-prompted-with-the-first-and-latest-chunks-before-it-as-fish-speechs-server-does.md)),
  which a later option could turn off without changing any request made before it.

The alternatives were turned down:

- fish-speech's defaults, temperature 0.7 and top-p 0.7. They are the SDK's for both of Fish Audio's models, while the
  model card measured Drama 3 with 1.0 and 0.9, which SGLang-Omni takes by default and the official implementation was
  measured with in speech-bench.
- A fixed top-k of 30, as fish-speech has it. SGLang-Omni takes it, and the sampler takes it at no cost.
- `instructions` carrying the cues. The model takes them in the text, at the place they act, which one instruction for
  the whole text cannot say.

## Consequences

A caller writes `<|speaker:1|>` and `[laughs]` into the text it sends, through every entry point alike.
