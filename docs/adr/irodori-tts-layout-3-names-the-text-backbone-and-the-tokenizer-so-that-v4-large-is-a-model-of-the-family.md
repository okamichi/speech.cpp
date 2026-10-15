# Irodori-TTS's layout 3 names the text backbone and the tokenizer, so that v4-Large is a model of the family

## Context

Irodori-TTS v4-Large (Aratako/Irodori-TTS-v4-Large, 3.29B parameters) is v4.1-Small's architecture made wider and deeper,
with one part replaced: the text and the caption are read by T5Gemma 2's text encoder, a Gemma 3 of 26 layers that
attends both ways, tokenized by Gemma's SentencePiece BPE tokenizer of 262,144 pieces, where v4.1-Small reads them with
ModernBERT-ja and its Unigram tokenizer. The projectors, the speaker encoder, the duration predictor, the DiT, the
sampler, the codec and the official runtime's request (`SamplingRequest` at 89f9d8f, which runs both) are the same, at
other widths, which the file's keys and tensors already give. Its references may be 120 s long, which the file's
`irodori-tts.reference.max_seconds` gives.

Layout 2 has no key that says which backbone or tokenizer a file holds: ModernBERT's tensors and the Unigram model's
scores are simply required.

## Decision

- **v4-Large is a model of the irodori-tts family**, read by the same reader, engine and sampler, with the same
  request options.
- **Layout 3** adds `irodori-tts.text.backbone`, `modernbert` or `t5gemma2`, and `irodori-tts.tokenizer.model`,
  `unigram` or `bpe`, which call for their own keys and tensors: T5Gemma 2's heads of keys and values, head width,
  sliding window before and after a token, RoPE scale of its full layers, `query_pre_attn_scalar` and the id of its
  token that ends an image with that token's embedding; and the BPE model's merges, in rank order, as the ids of their two
  pieces. Its `speech.requires` is 0.8.7: no file a release before it reads changes, so the release is a patch.
- **Gemma's RMS norms are stored as the factor they multiply by**, 1 + weight, which is the float32 sum the official
  code computes, so that every RMS norm of the family multiplies by its stored weight.
- **The upgrade sets the two keys of a layout 1 or 2 file to `modernbert` and `unigram`**, which every such file holds,
  so no user converts or downloads a v4.1 file again.
- **T5Gemma 2's tanh GELU is built from operations every backend computes in float32**, not `ggml_gelu()`, which on the
  CPU reads a table of half-precision values.
- **Nothing of the checkpoint is dropped**: the converter stops when a tensor of the checkpoint is left out of the file,
  the embedding of the token that ends an image included, which a text that holds the token's text reaches.

The alternatives were turned down:

- A family of its own for v4-Large. Every stage but the text backbone and the tokenizer would be copied, and the request
  options, voice files and checks would be kept twice.
- Telling the backbone apart by the tensors a file holds. The tensors follow from the keys alone, and a file missing one
  tensor would be read as the other backbone.
- Converting Gemma's tokenizer to a Unigram model. The two models cut a text differently, so the tokens would not be the
  ones the model was trained on.

- Maintaining a second layout 3 with `text.encoder`, `tokenizer.type` and string merge pairs. The same version would
  describe incompatible files, and every upstream change would require two readers and converters. The fork uses
  the upstream layout and reconverts its local Large and Anime files.

## Consequences

A release before 0.8.7 refuses a v4-Large file by its layout. Layout 3 files of v4.1 differ from their layout 2 files
only in the two keys, so the files on Hugging Face stay layout 2.
