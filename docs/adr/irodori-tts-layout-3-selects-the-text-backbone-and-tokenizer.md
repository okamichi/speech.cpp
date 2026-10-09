# Irodori-TTS layout 3 selects the text backbone and tokenizer

Irodori-TTS v4-Large uses T5Gemma2 and BPE, whereas v4.1-Small, Small-MF and Anime use ModernBERT and Unigram. A
reader of layout 2 cannot interpret Large's grouped attention, additional RMS norms or tokenizer merge ranks.

Layout 3 requires `irodori-tts.text.encoder` and `irodori-tts.tokenizer.type`, which select the metadata and tensors
of each. Layouts 1 and 2 upgrade to ModernBERT and Unigram in memory. The voice layout stays unchanged because the
codec, rather than the text backbone, binds reference latents to a model.

T5Gemma2's RMS weights store the official `1 + weight`, with its embedding scale, attention scale and RoPE scale
explicit in the file. Its end-of-image embedding is retained because its tokenizer can produce that token in text.
The model retains every checkpoint tensor; treating Large as a wider ModernBERT would compute different text and
caption conditions.
