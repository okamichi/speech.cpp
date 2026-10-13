# The page's live panel transcribes over /v1/realtime with server_vad, and needs a detection model

## Context

The page's live panel cut what it recorded into pieces of at most 20 s at a pause, and read the growing piece again for
its grey text. Each reading recognized the whole piece again, and a piece held several sentences, of which a
FastConformer model drops some. `/v1/realtime` now takes `server_vad`: the server finds each utterance with its
detection model, sends the text two readings agree on while it goes on, and the final text once a pause ends it
([the record of live transcription](live-transcription-cuts-utterances-by-the-detections-regions-and-sends-as-deltas-the-text-two-readings-agree-on.md)).

## Decision

- **The live panel opens a Realtime transcription session with `server_vad`** and appends what it records as it comes.
  The deltas grow the grey text of the utterance under way, and its final text replaces them. Stopping commits an
  utterance under way and waits for the last texts.
- **It needs a detection model**, which a picker of its own holds, as in the transcribe panel. Without one, it does not
  start, and says to choose one.
- **More options set the pause that ends an utterance** (`silence_duration_ms`), 500 ms unless changed, as OpenAI's
  default, and the browser keeps it. A shorter pause gives the final text sooner and cuts an utterance at a shorter
  pause, which depends on the speaker.

The alternatives were turned down:

- Pieces of a fixed length read again, as before. They hold several sentences each and are cut by length, not where
  someone stops speaking.
- Those pieces kept for a panel without a detection model. The same speech would be cut two ways and give two texts,
  and the page would keep a second way to find pauses for one case.

## Consequences

`checks/smoke/server_page_browser_smoke.mjs` runs the live panel with a microphone that plays the dumps with pauses
between them: the panel waits for a detection model, sends the pause set in its options and the microphone's audio
whole, in order and alone, shows grey text while an utterance is said, and ends with the text the API gives for the
audio it sent, with the same pause.
