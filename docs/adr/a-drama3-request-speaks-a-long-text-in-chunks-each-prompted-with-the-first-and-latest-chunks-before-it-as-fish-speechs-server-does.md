# A Drama 3 request speaks a long text in chunks, each prompted with the first and latest chunks before it, as fish-speech's server does

## Context

Drama 3 continues from its own earlier speech when the prompt holds it. fish-speech's server cuts a text into chunks of
about 300 bytes of UTF-8 with at most three speakers each, at sentence ends first, then at clause marks, then at
spaces, never inside a tag, and speaks the chunks one after another. Each chunk's prompt holds, after the references,
the first chunk and the latest one as earlier turns of the conversation, their text and the codes the model spoke, and
drops the latest and then the first while the prompt would exceed 8,192 rows. The voices and the way of speaking thus
carry from chunk to chunk. Without those turns each chunk starts afresh: a speaker without a reference, and every
speaker of the voice `none`, gets a voice of the model's choosing in each chunk, and the way of speaking breaks at every
chunk. The server can instead speak a whole text in one pass of at most 2,048 frames (95 s), which a request asks for.
SGLang-Omni speaks a text in one pass.

speech.cpp's tools speak a text whole on a model whose request speaks a minute or more, and the worker cuts a text the
library refuses as too long into requests that know nothing of each other
([the record of long texts](on-a-model-whose-request-speaks-less-than-a-minute-speech-tts-and-speech-serve-speak-a-text-a-sentence-at-a-time-and-the-worker-cuts-a-text-refused-as-too-long.md)).
The codes of the earlier chunks exist only inside the library.

## Decision

- **A request speaks its text as fish-speech's server does**: in chunks of about 300 bytes with at most three
  speakers, each prompted with the first and the latest chunk before it, and passes the audio on as it is made, chunk
  after chunk. A text that fits one chunk is one pass.
- **The rules are written from what the server does, and checked against its chunks.** fish-speech is run to make the
  reference dumps and not translated into the port, which is written from the weights, the technical report and
  SGLang-Omni's implementation; a check compares the chunks of the same texts, tagged and untagged, with the server's.
- **The chunks are the library's.** For the tools a Drama 3 request is one request that reads a text whole, as the record
  of long texts has it for a model that speaks a minute or more.
- **A request does not carry speech across requests.** Continuing from an earlier request's speech, which a caller that
  sends a sentence at a time would want, is a decision of its own when it is designed.

The alternatives were turned down:

- The tools' way alone: one pass of up to 95 s, and a longer text cut into requests that know nothing of each other. A
  voice of the model's choosing would change at every cut, and the way of speaking would break there.
- The tools cutting the text and passing each chunk's codes to the next request through the C API. Within one request
  the library holds the codes already; passing them between requests is the separate design above.
- Speaking every text in one pass, as SGLang-Omni does. A text longer than 95 s would be refused, and a long pass is
  not how Fish Audio serves the model.

## Consequences

A long text sounds as one speaker or one scene throughout, at the cost of the earlier chunks' rows in every prompt. The
request's text limit, not its speech, bounds how long it speaks.
