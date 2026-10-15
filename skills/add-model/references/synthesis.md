# Synthesis models

## Voices

- **Built-in voices** (a speaker table in the checkpoint): each with its name, language, gender and description, from
  the checkpoint or the official code, shown in the model information.
- **Voices made from recordings:** a voice file holds what the model needs from a reference (a codec latent, an
  embedding) with the hash of the codec that made it, so a voice works with every model sharing that codec and is
  refused by another. Decide loudness normalization, the longest reference, and several references as the official
  runtime does them.
- **No reference:** if the official runtime speaks without one, the file carries what that needs and the model offers it
  as a voice (`none`).
- **A description of the voice or the way of speaking** is the option `instructions`, as OpenAI's speech API names it,
  declared only by files that hold what it needs.

## Length and speed

The text's tokens, the speech's length and its speed each have a range from the file. Past the range a request is
refused, never cut. A model that fixes the length before it speaks takes `seconds` and a scale of the predicted length;
one that decides as it goes takes `max_seconds`.

## Sampling

Defaults come from the file (the checkpoint's generation config). A setting that has no effect because of another is
refused with the option named. A value the float cannot hold, or whose products overflow, is out of range; no
synthesis returns non-finite audio as a success.

## Streaming

- **Time to first audio:** measure the time to the first chunk. A small first chunk or window brings it forward.
- **No gaps:** a chunk must arrive before the listener has played the previous one. Measure the arrival time of every
  chunk against the audio sent, on an Apple M5, an RTX 2080 and a slower machine (an M2), and record the smallest
  margin.
- **Chunks and samples:** state whether the decoder's samples depend on how it is cut (see checks.md). If they do, keep a
  fixed schedule; if they do not, sizes may follow the measured speed.
- **Interruption:** a cancel stops between steps; the largest chunk bounds how long a cancel waits.

## What the official output adds

A watermark or other post-processing the official code applies and speech.cpp does not is listed as not implemented,
in the model's page.
