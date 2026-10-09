"""Dumps a small official T5Gemma2 stage and real tokenizer cases without converting model weights.

Usage: uv run python text_stage.py OUT LARGE_FOLDER ANIME_FOLDER
The GGUF files contain check fixtures, not synthesis models.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFWriter
from transformers.models.t5gemma2.configuration_t5gemma2 import T5Gemma2TextConfig
from transformers.models.t5gemma2.modeling_t5gemma2 import T5Gemma2TextEncoder
from tokenizers import Tokenizer

from gguf_tokenizer import add_array, write_tokenizer
from gguf import GGUFValueType

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
torch.set_num_threads(2)
cfg = T5Gemma2TextConfig(vocab_size=64, hidden_size=24, intermediate_size=40, num_hidden_layers=3,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, sliding_window=4,
                         query_pre_attn_scalar=8, layer_types=["sliding_attention", "full_attention", "sliding_attention"],
                         rope_parameters={"full_attention": {"rope_type": "linear", "rope_theta": 1000000.0, "factor": 8.0},
                                          "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}})
cfg.dropout_rate = 0.0
cfg._attn_implementation = "eager"
model = T5Gemma2TextEncoder(cfg, eoi_token_index=63).eval()
with torch.no_grad():
    for name, param in model.named_parameters():
        if "norm" in name:
            param.uniform_(-0.2, 0.2)
        elif "eoi_embedding" in name:
            param.normal_(0, 0.1)
w = GGUFWriter(None, "irodori-text-stage")
w.add_uint32("speech.layout", 1)
w.add_string("speech.requires", "0.9.0")
p = "irodori-tts.text."
for key, value in {"hidden_size": 24, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "num_layers": 3, "window": 4, "eoi_id": 63}.items():
    w.add_uint32(p + key, value)
for key, value in {"norm_eps": cfg.rms_norm_eps, "attention_scale": 8 ** -0.5, "embedding_scale": 24 ** 0.5,
                   "rope_scale_global": 0.125, "rope_theta_global": 1000000.0, "rope_theta_local": 10000.0}.items():
    w.add_float32(p + key, value)
add_array(w, p + "layer_global", [0, 1, 0], GGUFValueType.INT32)
def add(name, value):
    w.add_tensor(name, value.detach().numpy().astype(np.float32))
add("text.embd", model.embed_tokens.weight)
add("text.eoi_embd", model.embed_tokens.eoi_embedding)
add("text.final_norm", 1 + model.norm.weight)
for i, layer in enumerate(model.layers):
    b = f"text.blk.{i}."
    for source, target in (("pre_self_attn_layernorm", "attn_norm"), ("post_self_attn_layernorm", "attn_post_norm"),
                           ("pre_feedforward_layernorm", "ffn_norm"), ("post_feedforward_layernorm", "ffn_post_norm")):
        add(b + target, 1 + getattr(layer, source).weight)
    for source, target in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
        add(b + "attn_" + target, getattr(layer.self_attn, source + "_proj").weight)
    for source in ("q", "k"):
        add(b + source + "_norm", 1 + getattr(layer.self_attn, source + "_norm").weight)
    for source, target in (("gate", "act"), ("up", "gate"), ("down", "down")):
        add(b + "ffn_" + target, getattr(layer.mlp, source + "_proj").weight)
w.write_header_to_file(out / "stage.gguf")
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
for name, ids in {"short": [2, 4, 7], "window": list(range(2, 30)), "eoi": [2, 63, 4, 63, 5]}.items():
    folder = out / name
    folder.mkdir(exist_ok=True)
    inputs = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        result = model(input_ids=inputs, attention_mask=torch.ones_like(inputs), output_hidden_states=True)
    np.save(folder / "input_ids.npy", np.array(ids, dtype=np.int32))
    np.save(folder / "text_layers.npy", torch.stack([x[0] for x in result.hidden_states]).numpy())
    np.save(folder / "text_backbone.npy", result.last_hidden_state[0].numpy())
texts = ["明日の東京は晴れです。", "a  b   c", "　全角と半角 space", "\t\nabc\r", "𠮷野家🦜", "é café Ελληνικά", "<bos><end_of_turn>",
         "<s></s><unk>", "<start_of_turn>model\n日本語", "a" * 180, "㍿Ⅳ℃ﬁ♥", "🤭😮‍💨👂", "", "hello world", "<eoi>"]
for name, model_dir in zip(("large", "anime"), sys.argv[2:]):
    tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
    w = GGUFWriter(None, "irodori-tokenizer-stage")
    w.add_uint32("speech.layout", 1)
    w.add_string("speech.requires", "0.9.0")
    write_tokenizer(w, model_dir)
    w.write_header_to_file(out / f"{name}-tokenizer.gguf")
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    with (out / f"{name}-tokenizer.tsv").open("w") as f:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            bos = 2 if name == "large" else 1
            f.write(text.encode().hex() + "\t" + " ".join(map(str, [bos] + ids)) + "\n")
print("Wrote official stage and tokenizer fixtures to", out)
