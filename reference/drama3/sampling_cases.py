"""Runs the official sampler of Drama 3's audio decoder, fish-speech's sample_top_k_top_p(), on the logits of a dump of
dump.py, and saves the distribution each draw takes for drama3-sampler-check.

usage: uv run python sampling_cases.py <dump dir> <out dir>

Writes one folder per case under <out dir>, each with:
  meta.json          the settings, temperature, top_k and top_p, as "settings", or as "official" with "settings":
                     "file" for the model card's, which the model file holds, or "repeat" for the redraw of a repeated
                     main token, which it holds too
  logits.npy         the dump's logits, [rows, tokens]
  probabilities.npy  the probability of each token in the draw, 0 where the sampler drops it, [rows, tokens]

The main token's rows are the dump's main_logits, over the end id and the semantic ids; the codes' rows are its
head_logits. sample_top_k_top_p() draws with torch's exponential noise, which nothing outside torch reproduces, so the
draws themselves are not compared: the probabilities are the tensor it draws from, taken where it asks torch.empty_like()
for the noise, and put back in the order of the tokens with the ids its torch.sort() gave.
"""

import json
import os
import sys
from unittest import mock

import numpy as np
import torch
from fish_speech.models.text2semantic.fish_omni import tower

dump_dir, out_dir = sys.argv[1:3]
main_logits = np.load(os.path.join(dump_dir, "main_logits.npy"))
head_logits = np.load(os.path.join(dump_dir, "head_logits.npy"))
code_logits = head_logits.reshape(-1, head_logits.shape[-1])
CARD = {"temperature": 1.0, "top_k": 30, "top_p": 0.9}
REPEAT = {"temperature": tower.RAS_TEMPERATURE, "top_k": tower.RAS_TOP_K, "top_p": tower.RAS_TOP_P}

CASES = {
    "main-file": (main_logits, "file"),
    "main-repeat": (main_logits, "repeat"),
    "codes-file": (code_logits, "file"),
    "main-top-k-1": (main_logits, {"temperature": 1.0, "top_k": 1, "top_p": 0.9}),
    "codes-top-p-narrow": (code_logits, {"temperature": 1.0, "top_k": 30, "top_p": 0.1}),
    "codes-cold": (code_logits, {"temperature": 0.5, "top_k": 30, "top_p": 0.9}),
    "codes-hot": (code_logits, {"temperature": 1.5, "top_k": 100, "top_p": 1.0}),
    "main-every-token": (main_logits, {"temperature": 1.0, "top_k": 4097, "top_p": 1.0}),
}


def probabilities(row, settings):
    """The probabilities sample_top_k_top_p() draws `row` from, in the order of the tokens."""
    seen = {}
    sort, empty_like = torch.sort, torch.empty_like

    def recording_sort(*args, **kwargs):
        seen["ids"] = sort(*args, **kwargs)[1]
        return sort(*args, **kwargs)

    def recording_empty_like(t, *args, **kwargs):
        seen["probabilities"] = t.clone()
        return empty_like(t, *args, **kwargs)

    with mock.patch.object(torch, "sort", recording_sort), mock.patch.object(torch, "empty_like", recording_empty_like):
        tower.sample_top_k_top_p(torch.from_numpy(row)[None], settings["temperature"], settings["top_k"], settings["top_p"])
    out = torch.zeros(row.shape[-1], dtype=torch.float32)
    out[seen["ids"][0]] = seen["probabilities"][0]
    return out.numpy()


for name, (logits, settings) in CASES.items():
    values = CARD if settings == "file" else REPEAT if settings == "repeat" else settings
    folder = os.path.join(out_dir, name)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "logits.npy"), logits.astype(np.float32))
    np.save(os.path.join(folder, "probabilities.npy"), np.stack([probabilities(row, values) for row in logits]))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "official": values} if isinstance(settings, str) else {"settings": values}, f, indent=1)
    print(f"{name}: {len(logits)} rows")
