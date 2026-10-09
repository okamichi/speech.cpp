# A synthesis sets speed and length only where the model can, and reports its seed and stop

## Context

The models differ in what they let a request say of the speech's rate and length. Irodori-TTS sets a sentence's length
before it makes it, and its official runtime lets a request fix the length (`seconds`) or scale the prediction
(`duration_scale`); Irodori-TTS-Server, by the same author, serves OpenAI's speech API on it and turns OpenAI's `speed`
(0.25 to 4) into those two by dividing both by it. Where the runtime goes on quietly, it clamps seconds outside 0.5 to
30 s, ignores the scale when seconds are given, and bounds a predicted length to the same 30 s. Speech squeezed into 30
s loses words: in the voice none, eight short Japanese sentences that take 38.3 s spoken one at a time were predicted at
30.7 s and, bounded to 30 s, lost two whole sentences, and three times that text, predicted at 47 s, gave 30 s in which
neither Qwen3-ASR 0.6B nor ReazonSpeech v2 recognized a word of it (v4.1-Small, seed 1, Apple M5, 2026-10-08;
v4.1-Small-MF predicts the same lengths). Each request completed. Qwen3-TTS's official implementation has no control of
the rate or of the length: the talker decides frame by frame when the speech ends, and `generate_custom_voice()` takes
no such argument.

On a short interjection Qwen3-TTS goes on talking: ASIST measured "あー。" between 0.6 and 6.5 s in eight generations with
`ono_anna` (2026-09-21). Its checkpoint's `max_new_tokens` allows 8192 frames, 655 s. A caller has to tell a sentence
the model finished from one a limit cut, and to repeat a request whose audio it liked.

## Decision

- **A request carries `speed`, `seconds` and `duration_scale`, and a model takes each only where its official
  implementation can.** A model that does not take one refuses any value of it as `unsupported`, but for the neutral
  speed and duration scale of 1.
- **Irodori-TTS follows the runtime for `seconds` and `duration_scale` and Irodori-TTS-Server for `speed`**, which
  divides whichever length applies. The bounds of the length and of the speed are the file's (`irodori-tts.length.*`).
  Where the runtime goes on quietly, speech.cpp refuses: seconds outside the bounds, seconds together with a duration
  scale, and a length that the speed or the scale takes outside the bounds.
- **Irodori-TTS refuses a text whose predicted length passes the longest**, `out_of_range` naming the text, before it
  samples, where the runtime bounds the length and squeezes the speech into it; the message gives the predicted length
  and the longest and says to split the text. A speed or a duration scale the caller sets that brings the length within
  the bounds is followed, since the caller asked for that rate; one that takes a predicted length within the bounds past
  them is refused naming that option, as above. Seconds the caller fixes keep their meaning. A prediction under the
  shortest length is raised to it, as the runtime raises it, which adds silence and loses nothing.
- **Qwen3-TTS takes `max_seconds`**, the longest the speech may be, and stops at its file's limit of frames otherwise.
  Its cache grows with the request, so that a sentence costs the memory of its own length. Irodori-TTS does not take
  `max_seconds`, since it fixes the length before it makes the speech. The HTTP Irodori compatibility mode alone
  accepts a whole-response cap and stops decoded output, including sentence pauses, at that bound. This preserves
  the bridge's bounded WAV contract without pretending to change the model's predicted latent length.
- **A synthesis reports its seed, its length in samples and why it stopped**: complete, at the request's `max_seconds`,
  at the model's limit, or cancelled. A request without a seed gets one drawn from 0 to 2^53 − 1, the integers a JSON
  reader in JavaScript holds exactly, so that a reported seed repeats the audio on the same device.

The alternatives were turned down:

- Changing Qwen3-TTS's rate after synthesis by resampling or a time stretch. Resampling moves the pitch with the rate,
  and a time stretch (WSOLA, a phase vocoder) adds its own artifacts to every request that uses it and is no part of the
  model; a caller that wants either can apply it to the audio it receives.
- Keeping `speed` and ignoring it where a model cannot follow it. The caller cannot tell that its request was not
  followed.
- Clamping seconds into the bounds as the runtime does. The caller gets a length other than the one it asked for, told
  only in a log.
- Bounding a predicted length past the longest to it, as the runtime does. The speech loses words, or all of them, and
  the request reports it complete; a caller cannot tell it from speech that holds the whole text.
- Refusing `speed` together with a length. Irodori-TTS-Server combines them by division, and a caller moving from the
  server gets the same length here.
- `max_seconds` for Irodori-TTS. A cap would cut a latent made to be longer; `seconds` asks for the length itself.
- A cap kept by the caller, which cancels the request. It cannot tell its stop from the model's, and wastes the frames
  after it.
- A limit of frames in the C++ below the checkpoint's. A long text the model would finish would be cut.

## Consequences

A caller learns from the model information which of the options a model takes, with their ranges, and from the upper
bound of `seconds` the longest an Irodori-TTS request speaks. A text of a few sentences can pass Irodori-TTS's 30 s, so a
caller speaks such a text a sentence at a time; the refusal comes before any progress or audio. A Qwen3-TTS request
without `max_seconds` can run to 655 s, so a caller that knows a plausible length sends it as `max_seconds`.
