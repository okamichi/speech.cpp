"""Writes the cases the C++ caption strip, tokenizer and caption condition are checked against.

usage: uv run python caption_cases.py <large|anime|mf|rf> <out dir> [--model-dir DIR]

  caption-cases.tsv  one line per caption: its UTF-8 in hex, a tab, what the runtime keeps of it, str.strip() and
                     nothing else, in hex, a tab, and the ids the runtime's caption tokenizer gives it with <s>, none
                     for a caption that strips to nothing
  long-<i>/          for captions longer than ModernBERT's local window of 64 tokens on each side, the caption
                     condition as dump.py writes it: caption_ids and caption_state
  meta.json          in the folder and in each long-<i>/, the model, as dump.py names it

The captions reach each kind of character a caption may hold: Unicode's spaces at either end, which str.strip()
removes, and inside, which it keeps, Latin letters, digits, emoji, newlines, characters no piece covers, which fall
back to bytes, and captions of more than 128 tokens and of nearly 512, the longest the checkpoints take, which
in v4-Large are longer than T5Gemma 2's sliding window of 256 tokens before and 257 after.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import _load_checkpoint_for_inference
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer
from pins import MODELS, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--model-dir")
args = parser.parse_args()
name, out_dir = args.model, args.out_dir
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)
model_dir = args.model_dir if args.model_dir else snapshot(MODELS[name])


def write_meta(folder):
    json.dump({"model": MODELS[name]}, open(os.path.join(folder, "meta.json"), "w", encoding="utf-8"), indent=1)


write_meta(out_dir)
state, cfg_dict, inference_cfg, text_encoder_config = _load_checkpoint_for_inference(Path(model_dir) / "model.safetensors")
cfg = merge_dataclass_overrides(ModelConfig(), cfg_dict, section="model")
assert cfg.use_caption_condition and cfg.caption_tokenizer_repo_resolved == cfg.text_tokenizer_repo
tokenizer = PretrainedTextTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), add_bos=cfg.caption_add_bos_resolved,
                                                    local_files_only=True)
# The runtime's default_caption_max_len, from the checkpoint's configuration.
max_tokens = int(inference_cfg["max_caption_len"])

README = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"
captions = [
    README,
    "低く落ち着いた男性の声で、ゆっくりと読み上げてください。",
    "　全角の空白で囲んだ説明　",
    "  half-width spaces around  ",
    "\t\n改行とタブで囲んだ説明\r\n",
    "   様々な空白 　",
    "中に　全角と  半角の空白",
    "一行目\n二行目",
    "A calm female voice, reading softly at a close distance.",
    "ＡＢＣと１２３と12,345.6",
    "楽しそうに😊、ささやくように👂",
    "𠮷野家と🦜と́と",
    "a",
    "　",
    " \t\n　",
    "",
]
LIVELY = "明るく元気な若い女性の声で、はきはきと、少し早口に、語尾を上げて楽しそうに読み上げてください。"
# The second repeats a sentence as often as the model's tokens take it: 22 times in v4.1's 507 tokens.
repeats = max(k for k in range(1, 64) if len(tokenizer.encode(LIVELY * k)) < max_tokens)
long_captions = [README * 12, LIVELY * repeats]

with open(os.path.join(out_dir, "caption-cases.tsv"), "w") as f:
    for caption in captions + long_captions:
        kept = caption.strip()
        ids = tokenizer.encode(kept).tolist() if kept else []
        f.write(f"{caption.encode().hex()}\t{kept.encode().hex()}\t{' '.join(map(str, ids))}\n")
print(len(captions) + len(long_captions), "cases")

model = TextToLatentRFDiT(cfg, pretrained_backbone_config=text_encoder_config, load_pretrained_backbone_weights=False)
model.load_state_dict(state, assign=True)
model.eval()

for i, caption in enumerate(long_captions):
    ids, mask = tokenizer.batch_encode([caption.strip()], max_length=max_tokens)
    n = int(mask.sum())
    assert 128 < n < max_tokens, n
    caption_state = model.caption_norm(model.caption_encoder(model.pretrained_text_backbone, ids, mask))[0, :n]
    d = os.path.join(out_dir, f"long-{i}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "caption_ids.npy"), ids[0, :n].numpy().astype(np.int32))
    np.save(os.path.join(d, "caption_state.npy"), caption_state.numpy())
    write_meta(d)
    print(d, n, "tokens")
