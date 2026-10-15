"""Writes the cases the C++ text normalization, tokenizer and text encoder are checked against.

usage: uv run python text_cases.py <large|anime|mf|rf> <prompts.json> <out dir> [--model-dir DIR]

  text-cases.tsv     one line per text: the text's UTF-8 in hex, a tab, the official normalize_text() of it
                     (stripped, as the runtime does) in hex, a tab, and the ids the runtime gives it with <s>
  long-<i>/          for long texts, longer than ModernBERT's local window of 64 tokens on each side, the text
                     encoder's outputs as dump.py writes them: input_ids, text_layers and text_state
  meta.json          in the folder and in each long-<i>/, the model, as dump.py names it

The model's tokenizer and text encoder make the cases, so the checks take the cases of the model they check.

The texts are the prompts' sentences, texts that reach each rule of the normalization, the byte fallback
and the added tokens, and each of the emoji the runtime reads as a direction (a laugh, a sigh, a whisper), alone,
in a sentence and all together.
"""

import json
import os
import argparse
from pathlib import Path

import numpy as np
import torch

from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.duration import ALLOWED_ANNOTATION_EMOJIS
from irodori_tts.inference_runtime import _load_checkpoint_for_inference
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.text_normalization import normalize_text
from irodori_tts.tokenizer import PretrainedTextTokenizer
from pins import MODELS, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("prompts")
parser.add_argument("out_dir")
parser.add_argument("--model-dir")
args = parser.parse_args()
name, prompts_path, out_dir = args.model, args.prompts, args.out_dir
os.makedirs(out_dir, exist_ok=True)
torch.set_grad_enabled(False)
model_dir = args.model_dir if args.model_dir else snapshot(MODELS[name])


def write_meta(folder):
    json.dump({"model": MODELS[name]}, open(os.path.join(folder, "meta.json"), "w", encoding="utf-8"), indent=1)


write_meta(out_dir)
tokenizer = PretrainedTextTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"), add_bos=True,
                                                    local_files_only=True)

texts = [p["text"] for p in json.load(open(prompts_path, encoding="utf-8"))["prompts"]]
texts += [
    "ＡＳＩＳＴの設定でＶＯＩＣＥＶＯＸの話者を変えたい",
    "ｶﾀｶﾅとﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ、ｶﾞｷﾞｸﾞ",
    "わ～い〜、それは――違う―。",
    "えーと…………まあ...そうですね..",
    "本当？　すごい！！",
    "タブ\tと[n]改行の印\\[n]",
    "「全体を囲むかぎかっこ」",
    "『二重』",
    "（全角の括弧）",
    "(半角の括弧)",
    "【見出し】",
    "「前」と「後ろ」",
    "「「入れ子」」",
    "「閉じていない",
    "♥と●と◯と〇",
    "①番と②番；セミコロン▼♀♂《》≪≫",
    "ハイフン‐とダッシュ—とマイナス−と─罫線",
    "絵文字😀👍🏽と演出😊😮‍💨🙏",
    "記号★☆※と℃と㍿とⅣとx²とﬁ",
    "数字１２３と４５６、1,234.56円",
    "Ünïcödé café naïve résumé — Ελληνικά Русский 한국어 中文",
    "é combining and ﾞ alone",
    "𠮷野家と🦜",
    "  前後に空白  ",
    "a  b   c",
    "line one\nline two",
    "<s>と<|user|>と</s>",
    "a",
    "あ",
    "…",
    "?!",
]
texts += list(ALLOWED_ANNOTATION_EMOJIS)
texts += [f"えっと{e}、そうなんだ{e}。" for e in ALLOWED_ANNOTATION_EMOJIS]
texts += ["".join(ALLOWED_ANNOTATION_EMOJIS)]

with open(os.path.join(out_dir, "text-cases.tsv"), "w") as f:
    for text in texts:
        normalized = normalize_text(text).strip()
        ids = tokenizer.encode(normalized).tolist() if normalized else []
        f.write(f"{text.encode().hex()}\t{normalized.encode().hex()}\t{' '.join(map(str, ids))}\n")
print(len(texts), "cases")

state, cfg_dict, _, text_encoder_config = _load_checkpoint_for_inference(Path(model_dir) / "model.safetensors")
cfg = merge_dataclass_overrides(ModelConfig(), cfg_dict, section="model")
model = TextToLatentRFDiT(cfg, pretrained_backbone_config=text_encoder_config, load_pretrained_backbone_weights=False)
model.load_state_dict(state, assign=True)
model.eval()

long_texts = [
    "".join(texts[17:20]) * 4,
    "ログを見たところ、" * 20 + "TypeScriptの型エラーは、tsconfigのstrictを有効にしたことが原因でした。" * 4,
]
for i, text in enumerate(long_texts):
    normalized = normalize_text(text).strip()
    ids, mask = tokenizer.batch_encode([normalized], max_length=256)
    n = int(mask.sum())
    assert n > 65, n
    backbone = model.pretrained_text_backbone
    layers = backbone.backbone(input_ids=ids[:, :n], attention_mask=torch.ones(1, n, dtype=torch.long),
                               output_hidden_states=True, return_dict=True).hidden_states
    state = model.text_norm(model.text_encoder(backbone, ids, mask))[0, :n]
    d = os.path.join(out_dir, f"long-{i}")
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "input_ids.npy"), ids[0, :n].numpy().astype(np.int32))
    np.save(os.path.join(d, "text_layers.npy"), torch.stack([h[0] for h in layers]).numpy())
    np.save(os.path.join(d, "text_state.npy"), state.numpy())
    write_meta(d)
    print(d, n, "tokens")
