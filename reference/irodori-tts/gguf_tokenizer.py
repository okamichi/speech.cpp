"""Writes the exact tokenizer.json formats the Irodori-TTS C++ tokenizer reads."""

import json
import os

from gguf import GGUFValueType


def add_array(w, key, values, element):
    """An array whose element type is explicit."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


def write_tokenizer(w, model_dir):
    """Writes Unigram or BPE pieces, added tokens and merge ranks, rejecting other normalization rules."""
    tokenizer = json.load(open(os.path.join(model_dir, "tokenizer", "tokenizer.json"), encoding="utf-8"))
    model = tokenizer["model"]
    assert model["byte_fallback"]
    w.add_string("irodori-tts.tokenizer.type", model["type"])
    if model["type"] == "Unigram":
        assert tokenizer["normalizer"] is None
        assert tokenizer["pre_tokenizer"] == {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "never", "split": False}
        tokens = [piece for piece, _ in model["vocab"]]
        add_array(w, "irodori-tts.tokenizer.scores", [float(score) for _, score in model["vocab"]], GGUFValueType.FLOAT64)
        unknown_id = int(model["unk_id"])
    else:
        assert model["type"] == "BPE" and model["dropout"] is None and not model["ignore_merges"]
        assert model["continuing_subword_prefix"] is None and model["end_of_word_suffix"] is None
        assert tokenizer["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
        assert tokenizer["pre_tokenizer"] == {"type": "Split", "pattern": {"String": " "}, "behavior": "MergedWithPrevious", "invert": False}
        tokens = [piece for piece, _ in sorted(model["vocab"].items(), key=lambda item: item[1])]
        assert [model["vocab"][piece] for piece in tokens] == list(range(len(tokens)))
        add_array(w, "irodori-tts.tokenizer.merges", [" ".join(pair) for pair in model["merges"]], GGUFValueType.STRING)
        unknown_id = int(model["vocab"][model["unk_token"]])
    for added in tokenizer["added_tokens"]:
        assert tokens[added["id"]] == added["content"] and not added["normalized"]
        assert not (added["lstrip"] or added["rstrip"] or added["single_word"])
    tokenizer_config = json.load(open(os.path.join(model_dir, "tokenizer", "tokenizer_config.json"), encoding="utf-8"))
    add_array(w, "irodori-tts.tokenizer.tokens", tokens, GGUFValueType.STRING)
    add_array(w, "irodori-tts.tokenizer.added_ids", [int(added["id"]) for added in tokenizer["added_tokens"]], GGUFValueType.INT32)
    w.add_uint32("irodori-tts.tokenizer.bos_id", tokens.index(tokenizer_config["bos_token"]))
    w.add_uint32("irodori-tts.tokenizer.unknown_id", unknown_id)
    return model["type"]
