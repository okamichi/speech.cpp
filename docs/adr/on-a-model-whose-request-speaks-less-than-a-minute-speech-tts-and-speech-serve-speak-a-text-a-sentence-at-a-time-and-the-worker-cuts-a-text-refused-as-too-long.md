# On a model whose request speaks less than a minute, speech tts and speech serve speak a text a sentence at a time, and the worker cuts a text refused as too long

## Context

A synthesis request of the library speaks one text. Irodori-TTS speaks at most 30 s in a request, the upper bound of its
`seconds` option, and refuses a text whose speech it predicts to last longer
([the record of speed and length](a-synthesis-sets-speed-and-length-only-where-the-model-can-and-reports-its-seed-and-stop.md)).
A few sentences pass it: eight short Japanese sentences of 81 tokens take 38 s, and its limit of 256 tokens holds about
110 s of Japanese speech. Qwen3-TTS speaks up to 655 s (`max_seconds`) and takes 24565 tokens, so a paragraph is one
request. An OpenAI client sends `/v1/audio/speech` a paragraph or more (OpenAI's limit is 4096 characters), and
`speech tts` takes a whole text, while ASIST, through the worker, cuts its replies into sentences itself.
Irodori-TTS-Server, by the author of Irodori-TTS, cuts every input of 80 characters or more at its punctuation and joins
the audio directly.

Spoken alone and joined directly, two Irodori-TTS sentences leave 0.15 s of silence between them in the voice none and
0.59 to 0.67 s in a voice of a reference, which carries the reference's own silences. One request of six of the same
sentences pauses 1.04 to 1.12 s between them in the voice none and 0.79 to 0.85 s in that voice (both v4.1 models,
seeds 1 to 3, silence at or under -40 dBFS, Apple M5, 2026-10-08).

## Decision

- **The tools split, in `tools/common`, for `speech tts`, `speech serve` and the worker alike**; the library's request
  stays one text.
- **The worker speaks a synthesize as one request, since its caller chose the text, and cuts it only where the library
  refuses it as too long**: after its sentences first, each join holding the pause below, then as a refused sentence is
  cut. A text the worker spoke before sounds the same, and one it refused is spoken. ASIST cuts its replies into
  sentences, and a sentence past 50 characters at its next comma, yet a sentence whose commas all come in its first 50
  characters goes whole: 226 characters of Japanese were predicted at 33.0 s in one of ASIST's voices (2026-10-09).
- **A model whose longest request is under 60 s speaks a text of several sentences one sentence at a time.** The longest
  request is the upper bound of its `seconds` or `max_seconds` option from the model information, so no family is told
  by its name: Irodori-TTS, at 30 s, speaks by sentence, and Qwen3-TTS, at 655 s, reads a text whole, its prosody
  running across the sentences. A minute holds a paragraph of about a dozen short sentences. A text of one sentence, and any
  text on a model that reads whole, is one request, unchanged; one past the model's tokens is refused as before.
- **A sentence ends** after 。｡！？．, after . ! ? where a space or the end follows, so that 3.14 stays whole, and at a
  line break; closing brackets and quotation marks stay with the sentence before them.
- **A sentence the library refuses as too long is cut and spoken in pieces.** The library refuses a text too long for
  one request, in its tokens or in its predicted speech, as `out_of_range` naming the text before any audio. Such a
  sentence is cut after its commas (、､，； and , ; where a space or the end follows), or at its spaces where it has
  none, and each piece is spoken the same way in turn. A piece with no comma or space that is still refused fails the
  whole text, naming the piece and the library's reason: a cut inside a word reads wrongly, and ordinary text has no
  such run. HTTP Irodori compatibility requests explicitly allow Unicode-boundary bisection of a refused piece:
  the bridge permits a 500-codepoint run without punctuation, and refusing it would break that caller's contract.
  This can change pronunciation and is restricted to that splitting policy.
- **Every request takes the same model, voice and options, and one seed**: the seed given, or the one the first
  request drew, which the result reports, so that the seed repeats the whole text. The first request that stops other
  than complete ends the text with its stop. A cancel stops the request under way, and no other is made.
- **The audio passes on as it is made, through one callback**, so that a listener hears the first sentence while the
  rest are made. Each join after a sentence holds at least 0.9 s of silence: the silence after the last sound and before
  the next request's first sound is measured on the audio, and the join is topped up with zeros. 0.9 s lies between the
  pauses of one request in the two voices, so that neither is more than about 0.2 s off.

The alternatives were turned down:

- Putting consecutive sentences together into one request as long as they fit the model's tokens. An Irodori-TTS
  request of up to 256 tokens holds about 110 s of speech, which it refuses; a Qwen3-TTS one of 24565 tokens would stop
  at 655 s with `model_limit` and end the text there.
- Cutting every model's text into sentences. Qwen3-TTS would lose its prosody across the sentences of a paragraph, and
  a text it reads whole would sound otherwise than one request of it.
- Speaking the worker's texts a sentence at a time, as `speech tts` does. A text of several sentences that the worker
  spoke as one request would sound otherwise than it did, with no new option asking for it.
- Leaving a refused text to the worker's caller. The caller learns that a text is too long only from the refusal, after
  the library predicted its speech, and would have to rebuild these cuts.
- Telling the models that speak by sentence by their family. A family added with a short request would read a
  paragraph whole and be refused.
- Putting a refused sentence's pieces together as long as they fit. Whether a text fits is known only by running it,
  and a run that fits makes its audio.
- A fixed pause added at each join. The voice none lacks 0.9 to 0.96 s of the pause of one request and a voice of a
  reference 0.18 to 0.2 s, so a fixed pause leaves one of them about 0.7 s off.
- Joining directly, as Irodori-TTS-Server does. Sentences in the voice none follow each other after 0.15 s, where one
  request pauses about 1.1 s.

## Consequences

`speech serve` speaks an `input` of any length on Irodori-TTS, its first audio after the first sentence's, as fast as
one sentence alone; `speech tts` does the same, and reports how many requests a text took. A text of several sentences
on Irodori-TTS is spoken otherwise than one request of it would be, which the library refuses past 30 s. The pauses at
the joins depend on the silence the requests leave, so they differ by voice. The worker answers a text it cut with one
`end` for all its pieces, whose `samples` count the pauses at the joins, and runs its synthesis and its recognition
through the same cancellation as the other tools.
