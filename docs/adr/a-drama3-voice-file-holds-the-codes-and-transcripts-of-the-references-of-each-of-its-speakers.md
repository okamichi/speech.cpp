# A Drama 3 voice file holds the codes and transcripts of the references of each of its speakers

## Context

Fish Audio Drama 3 speaks in the voices of references. Each reference is a recording that the codec encodes to codes,
10 codebooks at 21.5 frames a second, with the transcript of what it says, and belongs to one of 16 speakers. The
prompt lists the references in its system turn, `[Reference 1]`, `Text: <|speaker:0|> …` and `Speech: ` followed by the
codes, and a text to speak chooses its speakers with the same tags, so that one text can be a scene of several people.
A speaker may have several references; a speaker the text names without a reference gets a voice the model chooses, and
with no reference at all every speaker does. fish-speech's server takes the references in each request, encodes them
and keeps their codes in memory by the audio's hash, or takes a stored reference by name; it refuses a reference
without a transcript. SGLang-Omni takes references for speaker 0 alone and refuses any other speaker's tag.

speech.cpp keeps what a voice needs from its references in a voice file, made once, rather than in a request
([the record of Irodori-TTS's voice files](an-irodori-tts-voice-file-holds-codec-latents-bound-to-the-codec-or-an-embedding-bound-to-its-model.md)):
a request, the worker's protocol and OpenAI's speech API each name one voice.

## Decision

- **A voice file holds one or more speakers, each with one or more references**: the codes of each reference and its
  transcript, in the order given. It is made from WAVE files with their transcripts, speaker 0 unless numbered, or from
  other Drama 3 voice files, whose references are copied under the speaker numbers given without being encoded again,
  so that a cast is put together from voices made before at no cost.
- **A request names one voice, and its text's tags choose among the voice's speakers.** A tag for a speaker the voice
  has no reference for lets the model choose that speaker's voice, as fish-speech does; the model information lists the
  speakers each voice holds.
- **The model speaks without a reference as its built-in voice `none`**, as Irodori-TTS does
  ([the record of the voice none](irodori-tts-speaks-without-a-reference-as-its-built-in-voice-none.md)), every speaker
  of the text in a voice of the model's choosing.
- **A voice binds to the codec by the SHA-256 of the official codec's tensors**, as Irodori-TTS's voices do, so that it
  works with every type of every model of that codec and a model of another codec refuses it.
- **A transcript is required.** A reference without one is refused when the voice is made, as fish-speech refuses it.

The alternatives were turned down:

- Assigning voices to speakers in the request, as an option that maps speaker numbers to voices. The request's voice
  would become a list, which the worker's protocol and OpenAI's speech API cannot carry, for the reason Irodori-TTS's
  references stay out of requests.
- One speaker per voice, cloning speaker 0 alone as SGLang-Omni does. A scene's other speakers would only ever have
  voices of the model's choosing, which change from text to text.
- Transcribing the references inside the library. The library would run a recognition model for a synthesis model; a
  caller transcribes with `speech asr` and hands the text over.

## Consequences

Making a voice takes the transcript of each recording, and its speaker number where it is not 0. A voice for a scene is
made once for the speakers it holds; another combination is another voice file, made from the voices of its speakers.
