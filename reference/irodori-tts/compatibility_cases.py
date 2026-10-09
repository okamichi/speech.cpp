"""Dumps official caption limits and inactive CFG for the bridge compatibility check.

usage: uv run python compatibility_cases.py MODEL_DIRECTORY OUT
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from irodori_tts.config import ModelConfig, merge_dataclass_overrides
from irodori_tts.inference_runtime import _load_checkpoint_for_inference, resolve_cfg_scales
from irodori_tts.model import TextToLatentRFDiT
from irodori_tts.tokenizer import PretrainedTextTokenizer

torch.set_num_threads(8)
torch.set_grad_enabled(False)
source, out = map(Path, sys.argv[1:])
out.mkdir(parents=True, exist_ok=True)
state, config, inference, backbone = _load_checkpoint_for_inference(source / 'model.safetensors')
cfg = merge_dataclass_overrides(ModelConfig(), config, section='model')
limit = int(inference['max_caption_len'])
tokenizer = PretrainedTextTokenizer.from_pretrained(str(source / 'tokenizer'), add_bos=cfg.caption_add_bos_resolved, local_files_only=True)
# Spaces keep the repeated ASCII token separate, so each requested boundary has exactly that many tokens.
counts = [limit - 1, limit, limit + 1]
captions = ['', ' \t　', '落ち着いた女性の声で話してください。', '😊' * 1000]
for count in counts:
    text = 'a ' * (count - int(cfg.caption_add_bos_resolved))
    assert len(tokenizer.encode(text.strip())) == count
    captions.append(text)
model = TextToLatentRFDiT(cfg, pretrained_backbone_config=backbone, load_pretrained_backbone_weights=False)
model.load_state_dict(state, assign=True)
model.eval()
meta = {'limit': limit, 'captions': [], 'guidance': []}
for i, text in enumerate(captions):
    stripped = text.strip()
    ids, mask = tokenizer.batch_encode([stripped], max_length=limit)
    if not stripped:
        mask.zero_()
    n = int(mask.sum())
    raw = tokenizer.encode(stripped).tolist() if stripped else []
    kept = ids[0, :n].tolist()
    meta['captions'].append({'text': text, 'ids': kept, 'truncated': len(raw) > limit})
    if n:
        condition = model.caption_norm(model.caption_encoder(model.pretrained_text_backbone, ids, mask))[0, :n]
        np.save(out / f'caption-{i}.npy', condition.numpy())
    print(f'caption {i}: {len(raw)} -> {n} tokens', flush=True)
for reference in [False, True]:
    for caption in [False, True]:
        text, cap, speaker, messages = resolve_cfg_scales(cfg_guidance_mode='independent', cfg_scale_text=3,
            cfg_scale_caption=7, cfg_scale_speaker=8, cfg_scale=None, use_caption_condition=caption, use_speaker_condition=reference)
        # A masked caption creates no guidance branch even when the requested scale remains in the runtime object.
        meta['guidance'].append({'reference': reference, 'caption': caption, 'text': text, 'speaker': speaker,
                                 'instructions': cap if caption else 0})
(out / 'compatibility.json').write_text(json.dumps(meta, ensure_ascii=False))
