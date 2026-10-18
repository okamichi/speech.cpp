"""Writes the cases of drama3-prompt-check from fish-speech's own tokenizer, prompt builder and chunking, one JSON object
a line:

  {"kind": "encode", "text": ..., "ids": [...]}
      the ids the checkpoint's tokenizer.json gives a text, as build_drama3_prompt() tokenizes each piece
  {"kind": "prompt", "text": ..., "references": [...], "history": [...], "rows": [...] or "error": ...}
      the rows build_drama3_prompt() writes, flattened, for a text, references and chunks spoken before, each
      {"text": ..., "codes": [...], "frames": n} with codes [10, frames] flattened, or the error it raises
  {"kind": "chunks", "text": ..., "chunks": [...]}
      the chunks split_drama3_text() cuts a text into

The codes of the references and the turns are drawn at random with a fixed seed: the builder writes them as they are.
With a dump of dump.py, the prompt it holds is a case too, with its reference's transcript and codes, once the builder
gives the same rows again.

usage: uv run python prompt_cases.py <out.jsonl> [dump dir]
"""

import json
import os
import random
import sys
import unicodedata

import numpy as np
import torch
from tokenizers import Tokenizer

from fish_speech.models.text2semantic.errors import InvalidInputError
from fish_speech.models.text2semantic.fish_omni.chunking import split_drama3_text
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

import pins

folder = pins.snapshot()
tokenizer = Tokenizer.from_file(f"{folder}/tokenizer.json")
config = FishOmniConfig.from_dict(json.load(open(f"{folder}/config.json", encoding="utf-8")))
codebooks, codes_per_book = config.audio_decoder.num_codebooks, config.audio_decoder.vocab_size
rng = random.Random(0)
out = open(sys.argv[1], "w", encoding="utf-8")


def write(case):
    out.write(json.dumps(case, ensure_ascii=False) + "\n")


encode_texts = [
    "こんにちは、今日はいい天気ですね。",
    "明日の東京は晴れで、最高気温は二十四度の予報です。",
    "GitHubのプルリクエストのCIが落ちているか確認して",
    "It's 3:45 PM, and I'll be there in 10 minutes—don't worry!",
    "WE'RE HERE. They've gone. I'd say it's fine; she'll see.",
    "  leading spaces,   inner   spaces, and trailing spaces   ",
    "line one\nline two\r\n\nline four\n",
    "数字123と４５６、1,234.56円、2026年9月29日",
    "絵文字😀👍🏽と記号★☆※、「かぎかっこ」『二重』",
    "Ünïcödé café naïve résumé — Ελληνικά Русский 한국어 中文",
    "é combining and ｶﾀｶﾅ half-width",
    # Marks, which Qwen3.5's pattern takes with letters and Qwen2's apart from them.
    "नमस्ते दुनिया, हिन्दी में बात करें",
    "ภาษาไทยมีวรรณยุกต์ ก่ ก้ ก๊ ก๋",
    "à́b ç ́alone ⃝",
    "مَرْحَبًا بِكُمْ",
    unicodedata.normalize("NFD", "がぎぐげご ぱぴぷぺぽ ヴォイス"),
    "a>" + chr(0x338) + "b =" + chr(0x338) + " c",
    # Added tokens, which the tokenizer finds before anything else, special or not, the longest where two start alike.
    "<|im_start|>user\n<|speaker:0|> こんにちは<|im_end|>\n<|im_start|>assistant\n<|voice|>",
    "<|speaker:1|> A <|speaker:10|> B <|speaker:15|>C",
    "<think>考え中</think> <|fim_prefix|>x<|fim_suffix|>",
    "<|im_end|>" + chr(0x338) + " [laughs] [sighs] (whispering)",
    "<|speaker:3|>",
]
for text in encode_texts:
    write({"kind": "encode", "text": text, "ids": tokenizer.encode(text).ids})


def utterance(text, frames):
    return {"text": text, "codes": [rng.randrange(codes_per_book) for _ in range(codebooks * frames)], "frames": frames}


def tensor(u):
    return torch.tensor(u["codes"], dtype=torch.long).reshape(codebooks, u["frames"])


prompt_cases = [
    ("こんにちは、今日はいい天気ですね。", [], []),
    ("Hello there.", [utterance("札幌の今日の最高気温は十八度です。", 7)], []),
    ("<|speaker:1|> はい。<|speaker:0|> そうですか。",
     [utterance("  先頭に空白のある書き起こし", 3), utterance("<|speaker:1|>話者 1 の声", 4)], []),
    ("三つ目の塊です。", [utterance("参照", 2)], [utterance("一つ目の塊です。", 5), utterance("  <|speaker:2|> 二つ目", 3)]),
    ("[laughs] それは面白い！ [sighs] でも疲れた。", [], [utterance("最初", 2)]),
    ("　全角の空白で始まる", [], []),
    # Texts the builder refuses.
    ("   \n\t", [], []),
    ("<|im_start|>入れてはいけない", [], []),
    ("<|speaker:16|> 範囲の外", [], []),
    ("<|speaker:01|> 先頭のゼロ", [], []),
    ("<|speaker: 1|> 空白", [], []),
    ("話者のタグが壊れている <|speaker", [], []),
    ("正しい", [utterance("<|voice|>書き起こしに構造のトークン", 2)], []),
]
if len(sys.argv) > 2:
    dump = sys.argv[2]
    meta = json.load(open(os.path.join(dump, "meta.json"), encoding="utf-8"))
    codes = np.load(os.path.join(dump, "reference_codes.npy"))
    reference = {"text": meta["reference"]["prompt_text"], "codes": codes.reshape(-1).tolist(), "frames": int(codes.shape[1])}
    rows = build_drama3_prompt(meta["text"], tokenizer, config, [reference["text"]], [tensor(reference)])
    assert np.array_equal(rows.numpy(), np.load(os.path.join(dump, "prompt.npy"))), "the builder writes the dump's prompt otherwise"
    write({"kind": "prompt", "text": meta["text"], "references": [reference], "history": [], "rows": rows.reshape(-1).tolist()})

for text, references, history in prompt_cases:
    case = {"kind": "prompt", "text": text, "references": references, "history": history}
    try:
        rows = build_drama3_prompt(text, tokenizer, config, [r["text"] for r in references], [tensor(r) for r in references],
                                   [(h["text"], tensor(h)) for h in history])
        case["rows"] = rows.reshape(-1).tolist()
    except InvalidInputError as error:
        case["error"] = str(error)
    write(case)

long_japanese = "".join(f"これは{i}番目の文で、少し長めに書いてあります。" for i in range(1, 30))
long_english = " ".join(f"This is sentence number {i}, which is written to be a little longer than usual." for i in range(1, 20))
chunk_texts = [
    "短い文。",
    "   ",
    long_japanese,
    long_english,
    "円周率は3.14159265358979で、" * 12 + "おしまい。",
    "区切りのない長い文" * 40,
    "A" * 700,
    "[laughs] " + "とても長い笑いの説明。" * 30 + " [sighs]",
    "<whisper>" + "ささやき声で話す長い文章です、" * 20 + "</whisper>終わり。",
    "<b x=1>太字</b> と <|im_end|> と [cue] は切らない。" * 10,
    "<|speaker:0|> こんにちは。<|speaker:1|> こんばんは。<|speaker:0|> またね。",
    "<|speaker:0|> 一。<|speaker:1|> 二。<|speaker:2|> 三。<|speaker:3|> 四。<|speaker:1|> 五。",
    "前置き <|speaker:2|> " + "話者 2 が長く話します。" * 30 + "<|speaker:5|> 短く。",
    "全角の数字は３.１４、アラビア数字は٣.١٤です。" * 10,
    "Line one.\nLine two!\nLine three?\n" * 20,
]
for text in chunk_texts:
    write({"kind": "chunks", "text": text, "chunks": split_drama3_text(text)})
out.close()
print(len(encode_texts), len(prompt_cases), len(chunk_texts))
