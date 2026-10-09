"""Runs the official Irodori-TTS on the CPU in float32 and saves the tensors the C++ port is checked against.

usage: uv run python dump.py <mf|rf|anime|large> <out dir> <text> (<reference.wav>... | --no-ref | --ref-embed f) [--seed n] [--steps n]
                             [--seconds s | --duration-scale x] [--speed x] [--ref-normalize-db x | --no-normalize]
                             [request options]

mf is v4.1-Small-MF with its 4 MeanFlow steps; rf is v4.1-Small with 40 Euler steps and the runtime's
default guidance (text 3.0 and speaker 5.0, each against a branch without it, while t >= 0.5).

--seconds fixes the length and --duration-scale scales the predicted one, as the runtime's request takes
them. --speed is OpenAI's speed, which Irodori-TTS-Server divides both by before it calls the runtime;
meta.json keeps the three as given. With --seconds the duration predictor does not run, and its two files
are not written. --no-ref speaks without a reference (the runtime's no_ref, speech.cpp's voice none), and
meta.json's "reference" is then null. Several references are the runtime's ref_wavs, each encoded on its own and
joined, and meta.json's "reference" lists them. --ref-normalize-db brings each to another loudness than -16 LUFS and
--no-normalize keeps it as recorded (ref_normalize_db None), which meta.json keeps as "ref_normalize_db".
--ref-embed speaks in a speaker-inversion embedding (a .speaker.safetensors file, the runtime's ref_embed), which
meta.json names as its "reference" with "embedding" true; its speaker_state is the embedding.

The request options are the C API's names of the runtime's SamplingRequest fields, in kebab-case: for rf
--cfg-scale-text, --cfg-scale-speaker, --cfg-guidance-mode, --cfg-min-t, --cfg-max-t, --truncation-factor,
--rescale-k, --rescale-sigma, --speaker-uncond-mode, --sway-coeff (Sway Sampling when not 0),
--speaker-kv-scale, --speaker-kv-min-t, --speaker-kv-max-layers and --cfg-scale-instructions, and for both
--instructions (the runtime's caption), --keep-tail, --tail-window-size, --tail-std-threshold and
--tail-mean-threshold. meta.json keeps the ones given under the C API's names in "options", and a check takes the
model's default for the others.

The official runtime.synthesize() runs once with its stages wrapped, so what is saved is what it computed.
The finer stages (the text encoder's layers, the speaker encoder, each DiT block of the first step, the
codec's blocks) are then recomputed from those inputs, and the recomputation is asserted to give the same
result as synthesize(). Writes <out dir>/*.npy and meta.json:

  input_ids            the text's tokens with <s>, [N]; the runtime pads them to 256 and masks the rest
  text_layers          ModernBERT's hidden states on the N tokens alone: embeddings then each layer, [26, N, 768]
  text_backbone        ModernBERT's final output on the padded input, [N, 768]
  text_state           the text condition after the projector and its norm, [N, 512]
  caption_ids          with instructions: the caption's tokens with <s>, [C]; the runtime pads them to 512
  caption_state        the caption condition after its projector and norm, [C, 512]
  ref_wav              one reference as read, mono float32 at 48 kHz
  ref_wav_normalized   the reference at its loudness, -16 LUFS but for the options above
  ref_encoder          the DACVAE encoder's output before the bottleneck, [1024, T], for one reference
  ref_latent           the DACVAE latent (the encoder's mean), [T, 32], of several references joined
  speaker_encoded      the speaker encoder's output on the latent in patches of 4, [T / 4, 768]
  speaker_state        after its norm, with the masked mean prepended, [1 + T / 4, 768]
  duration_features    the 14 features of the text, [14]
  duration_log_frames  the predictor's log(1 + frames), [1]
  noise                the sampler's draw of noise, before a truncation factor scales it, [S, 32]
  speaker_noise        rf with speaker_uncond_mode noise: the draw after it, [1 + T / 4, 768]
  speaker_uncond       the same, scaled by the speaker condition's spread: the branch without the speaker
  dit_t                each step's time, [steps]
  dit_delta            mf: each step's interval, [steps]
  dit_branches         rf: the DiT output of each branch of the guided steps, the branch with every condition
                       first, [guided steps, branches, S, 32]; meta.json's "branches" names the others by the
                       condition each leaves out (text, speaker, or all for the joint guidance)
  dit_velocity         the velocity of each step, after guidance and rescaling for rf, [steps, S, 32]
  dit_x                the latent after each step, [steps, S, 32]
  dit_step0_blocks     the output of each DiT block in the first step, [12, branches, S, 1280]
  dit_cond             the timestep condition of each step, [steps, 3840]
  codec_in             the codec input after its bottleneck projection, [1024, S]
  codec_conv_in        after the first convolution, [1536, S]
  codec_block{0..3}    after each upsampling block, [channels, samples so far]
  wav                  the decoded audio before the tail is trimmed, [S * 1920]
  audio                what synthesize() returned, trimmed at the tail
"""

import argparse
import json
import os

import numpy as np
import torch

from irodori_tts import inference_runtime as ir
from irodori_tts import rf
from irodori_tts.codec import DACVAECodec
from irodori_tts.model import get_timestep_embedding, patch_sequence_with_mask
from irodori_tts.speaker_inversion import load_speaker_inversion_payload
from irodori_tts.text_normalization import normalize_text
from pins import CODE, CODEC, MODELS, snapshot

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("text")
parser.add_argument("references", nargs="*")
parser.add_argument("--model-dir", help="local full-precision model folder")
parser.add_argument("--codec-dir", help="local folder containing weights.pth")
parser.add_argument("--no-ref", action="store_true")
parser.add_argument("--ref-embed")
parser.add_argument("--ref-normalize-db", type=float)
parser.add_argument("--no-normalize", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int)
parser.add_argument("--seconds", type=float, default=0.0)
parser.add_argument("--duration-scale", type=float, default=1.0)
parser.add_argument("--speed", type=float, default=1.0)
# The request options, by the C API's names, with the SamplingRequest field each sets.
RF_OPTIONS = {"cfg_scale_text": (float, "cfg_scale_text"), "cfg_scale_speaker": (float, "cfg_scale_speaker"),
              "cfg_guidance_mode": (str, "cfg_guidance_mode"), "cfg_min_t": (float, "cfg_min_t"),
              "cfg_max_t": (float, "cfg_max_t"), "truncation_factor": (float, "truncation_factor"),
              "rescale_k": (float, "rescale_k"), "rescale_sigma": (float, "rescale_sigma"),
              "speaker_uncond_mode": (str, "speaker_uncond_mode"), "sway_coeff": (float, "sway_coeff"),
              "speaker_kv_scale": (float, "speaker_kv_scale"), "speaker_kv_min_t": (float, "speaker_kv_min_t"),
              "speaker_kv_max_layers": (int, "speaker_kv_max_layers"), "cfg_scale_instructions": (float, "cfg_scale_caption")}
OPTIONS = {**RF_OPTIONS, "tail_window_size": (int, "tail_window_size"), "tail_std_threshold": (float, "tail_std_threshold"),
           "tail_mean_threshold": (float, "tail_mean_threshold"), "instructions": (str, "caption")}
for name, (kind, _) in OPTIONS.items():
    parser.add_argument("--" + name.replace("_", "-"), type=kind)
parser.add_argument("--keep-tail", action="store_true")
args = parser.parse_args()
# The runtime ignores a duration scale given with seconds; speech.cpp refuses that request.
assert not (args.seconds > 0 and args.duration_scale != 1.0), "give --seconds or --duration-scale, not both"
assert bool(args.references) + args.no_ref + bool(args.ref_embed) == 1, "give references, --no-ref or --ref-embed"
assert args.ref_normalize_db is None or not args.no_normalize, "give --ref-normalize-db or --no-normalize, not both"
options = {name: getattr(args, name) for name in OPTIONS if getattr(args, name) is not None}
if args.keep_tail:
    options["keep_tail"] = True
# MeanFlow's sampler takes none of RF's options, which the runtime drops with a message.
assert args.model != "mf" or not any(name in RF_OPTIONS for name in options), "the rf options are not mf's"
os.makedirs(args.out_dir, exist_ok=True)
torch.set_grad_enabled(False)

ir.SilentCipherWatermarker._load_backend = staticmethod(lambda **_: None)
runtime = ir.InferenceRuntime.from_key(
    ir.RuntimeKey(
        checkpoint=os.path.join(args.model_dir if args.model_dir else snapshot(MODELS[args.model]), "model.safetensors"),
        model_device="cpu",
        codec_repo=os.path.join(args.codec_dir if args.codec_dir else snapshot(CODEC), "weights.pth"),
        codec_device="cpu",
    )
)
model, codec = runtime.model, runtime.codec
saved = {}


def save(name, tensor):
    array = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    saved[name] = array.astype(np.int32 if array.dtype.kind in "iub" else np.float32)


def wrap(owner, name, record):
    original = getattr(owner, name)

    def wrapper(*a, **k):
        out = original(*a, **k)
        record(a, k, out)
        return out

    setattr(owner, name, wrapper)


calls = {"encode_conditions": [], "velocity": [], "normalized": [], "encode_in": [], "decode": []}
normalize_loudness = DACVAECodec._normalize_loudness
DACVAECodec._normalize_loudness = staticmethod(
    lambda *a, **k: calls["normalized"].append(normalize_loudness(*a, **k)) or calls["normalized"][-1])
wrap(codec, "encode_waveform", lambda a, k, out: calls["encode_in"].append((a[0], out)))
wrap(model, "encode_conditions", lambda a, k, out: calls["encode_conditions"].append((k, out)))
wrap(model, "predict_duration_log_frames", lambda a, k, out: saved.update(
    duration_features=k["duration_features"][0].numpy(), duration_log_frames=out.numpy()))


def record_velocity(a, k, out):
    # The sampler scales a request's speaker keys and values in place and restores them at speaker_kv_min_t, so the
    # first call's are copied for its blocks to be computed again below.
    if not calls["velocity"] and k.get("context_kv_cache") is not None:
        k = dict(k, context_kv_cache=[tuple(t.clone() for t in layer) for layer in k["context_kv_cache"]])
    calls["velocity"].append((k, out))


wrap(model, "forward_with_encoded_conditions", record_velocity)
wrap(codec, "decode_latent", lambda a, k, out: calls["decode"].append((a[0], out)))

# Irodori-TTS-Server's mapping of OpenAI's speed
# (Aratako/Irodori-TTS-Server@61012c760f22f7b4a6c21c5c5f8f9e148120b6f9, src/irodori_openai_tts/app.py).
seconds = args.seconds / args.speed if args.seconds > 0 else None
fields = {OPTIONS[name][1]: value for name, value in options.items() if name != "keep_tail"}
fields["trim_tail"] = not args.keep_tail
# speech.cpp's sway_coeff of 0 is the runtime's linear schedule, which Sway Sampling with 0 equals.
if options.get("sway_coeff", 0.0) != 0.0:
    fields["t_schedule_mode"] = "sway"
if len(args.references) == 1:
    fields["ref_wav"] = args.references[0]
elif args.references:
    fields["ref_wavs"] = args.references
if args.no_normalize or args.ref_normalize_db is not None:
    fields["ref_normalize_db"] = None if args.no_normalize else args.ref_normalize_db
if args.ref_embed:
    fields["ref_embed"] = args.ref_embed
request = ir.SamplingRequest(text=args.text, no_ref=args.no_ref, seed=args.seed, seconds=seconds,
                             duration_scale=args.duration_scale / args.speed, num_steps=args.steps, **fields)
result = runtime.synthesize(request)
normalized_text = normalize_text(args.text).strip()

# Text: the runtime pads to 256 tokens; the port runs the valid ones only.
kwargs, (text_state, text_mask, speaker_state, speaker_mask, _, _) = calls["encode_conditions"][-1]
n = int(text_mask[0].sum())
ids = kwargs["text_input_ids"][0, :n]
save("input_ids", ids)
backbone = model.pretrained_text_backbone
padded = backbone(kwargs["text_input_ids"], kwargs["text_mask"])[0, :n]
save("text_backbone", padded)
layers = backbone.backbone(input_ids=ids[None], attention_mask=torch.ones(1, n, dtype=torch.long),
                           output_hidden_states=True, return_dict=True).hidden_states
save("text_layers", torch.stack([h[0] for h in layers]))
unpadded_error = float((layers[-1][0] - padded).abs().max())
state = model.text_norm(model.text_encoder(backbone, kwargs["text_input_ids"], kwargs["text_mask"]))[0, :n]
assert torch.equal(state, text_state[0, :n])
save("text_state", state)

# Caption: the runtime pads it to 512 tokens and masks the rest, and masks all of one that strips to nothing.
caption_state, caption_mask = calls["encode_conditions"][-1][1][4:6]
has_caption = bool(caption_mask.any())
assert has_caption == bool(options.get("instructions", "").strip())
if has_caption:
    c = int(caption_mask[0].sum())
    save("caption_ids", kwargs["caption_input_ids"][0, :c])
    recomputed = model.caption_norm(model.caption_encoder(backbone, kwargs["caption_input_ids"], kwargs["caption_mask"]))[0, :c]
    assert torch.equal(recomputed, caption_state[0, :c])
    save("caption_state", caption_state[0, :c])

# Reference: loudness, the DACVAE encoder, the speaker encoder. Without a reference the runtime encodes a zero
# latent whose every position is masked, which the port leaves out.
if args.ref_embed:
    # The runtime attends to the embedding as it is, every token unmasked.
    embedding = load_speaker_inversion_payload(args.ref_embed)["speaker_embedding"]
    assert torch.equal(speaker_state[0], embedding) and bool(speaker_mask.all()) and not calls["encode_in"]
    save("speaker_state", speaker_state[0])
elif not args.no_ref:
    assert len(calls["encode_in"]) == len(args.references)
    latent = torch.cat([out for _, out in calls["encode_in"]], dim=1)
    if len(args.references) == 1:
        wav_in, _ = calls["encode_in"][0]
        save("ref_wav", wav_in.reshape(-1))
        if calls["normalized"]:
            processed = calls["normalized"][0]
        else:
            # Kept as recorded: the codec scales down a peak above 1 (ensure_max).
            peak = wav_in.reshape(-1).abs().max()
            processed = wav_in.reshape(-1) * (1.0 / float(peak)) if peak > 1 else wav_in.reshape(-1)
        save("ref_wav_normalized", processed)
        padded_wav = codec.model._pad(processed.reshape(1, 1, -1))
        encoded = codec.model.encoder(padded_wav)
        assert torch.equal(codec.model.quantizer.in_proj(encoded).chunk(2, dim=1)[0].transpose(1, 2), latent)
        save("ref_encoder", encoded[0])
    save("ref_latent", latent[0])
    patched, patch_mask = patch_sequence_with_mask(latent, torch.ones(latent.shape[:2], dtype=torch.bool),
                                                   model.cfg.speaker_patch_size)
    speaker_encoded = model.speaker_encoder(patched, patch_mask)
    save("speaker_encoded", speaker_encoded[0])
    recomputed, _ = model._prepend_masked_mean_token(model.speaker_norm(speaker_encoded), patch_mask)
    assert torch.equal(recomputed, speaker_state)
    save("speaker_state", speaker_state[0])
else:
    assert not bool(speaker_mask.any())

# Sampling: every step, and each block of the first. The steps are regrouped from the DiT's calls, one per step
# for MeanFlow, RF's independent guidance and an unguided step, two for RF's joint and alternating guidance.
calls_by_step = []
for k, out in calls["velocity"]:
    if calls_by_step and bool(calls_by_step[-1][0][0]["t"][0] == k["t"][0]):
        calls_by_step[-1].append((k, out))
    else:
        calls_by_step.append([(k, out)])
meanflow = model.cfg.flow_parameterization == "meanflow"
# A guided step's latent is repeated for each branch of the batch; the first is the sampler's.
first = calls_by_step[0][0][0]["x_t"][0:1]
S = first.shape[1]
# The draws of the runtime's generator, made again: the latent's noise, then the speaker's in noise mode.
rng = torch.Generator(device="cpu").manual_seed(args.seed)
noise = torch.randn((1, S, model.cfg.patched_latent_dim), generator=rng)
truncation = request.truncation_factor
assert torch.equal(noise * float(truncation) if truncation is not None else noise, first)
save("noise", noise[0])
mode = str(request.cfg_guidance_mode).strip().lower()
uncond_mode = str(request.speaker_uncond_mode).strip().lower()
if not meanflow and uncond_mode == "noise":
    speaker_noise = torch.randn(speaker_state.shape, generator=rng)
    speaker_uncond = speaker_noise * speaker_state.std().clamp_min(1e-6)
    save("speaker_noise", speaker_noise[0])
    save("speaker_uncond", speaker_uncond[0])
scales = {}
if not meanflow:
    text_scale, caption_scale, speaker_scale, _ = ir.resolve_cfg_scales(
        cfg_guidance_mode=mode, cfg_scale_text=request.cfg_scale_text, cfg_scale_caption=request.cfg_scale_caption,
        cfg_scale_speaker=request.cfg_scale_speaker, cfg_scale=request.cfg_scale, use_caption_condition=has_caption,
        use_speaker_condition=not args.no_ref)
    # The runtime's order of the guidance: text, speaker, caption.
    scales = {name: scale for name, scale in (("text", text_scale), ("speaker", speaker_scale),
                                              ("caption", caption_scale if has_caption else 0.0)) if scale > 0}
enabled = list(scales)
save("dit_t", torch.stack([group[0][0]["t"][0] for group in calls_by_step]))
velocities, branches, branch_names = [], [], []
x = first
xs = []
for i, group in enumerate(calls_by_step):
    k, out = group[0]
    assert torch.equal(k["x_t"][0:1], x)
    if len(group) == 1 and out.shape[0] == 1:
        v = out
    elif len(group) == 1:
        # The independent guidance: one branch without each enabled condition, added in the sampler's order.
        v = out[0:1]
        for name, branch in zip(enabled, out[1:], strict=True):
            v = v + scales[name] * (out[0:1] - branch[None])
        branches.append(out)
        branch_names.append(enabled)
    else:
        (_, cond), (_, uncond) = group
        name = "all" if mode == "joint" else enabled[i % len(enabled)]
        v = cond + scales[enabled[0] if mode == "joint" else name] * (cond - uncond)
        branches.append(torch.cat([cond, uncond]))
        branch_names.append([name])
    if request.rescale_k is not None:
        v = rf.temporal_score_rescale(v, x, k["t"][0], float(request.rescale_k), float(request.rescale_sigma))
    t_next = calls_by_step[i + 1][0][0]["t"][0] if i + 1 < len(calls_by_step) else torch.zeros_like(k["t"][0])
    x = x + v * (t_next - k["t"][0])
    velocities.append(v)
    xs.append(x[0])
if meanflow:
    save("dit_delta", torch.stack([group[0][0]["delta_t"][0] for group in calls_by_step]))
elif branches:
    save("dit_branches", torch.stack(branches))
save("dit_velocity", torch.cat(velocities))
save("dit_x", torch.stack(xs))
if not meanflow and uncond_mode == "noise" and mode == "independent" and "speaker" in enabled:
    batched = calls_by_step[0][0][0]["speaker_state"]
    assert torch.equal(batched[1 + enabled.index("speaker")], speaker_uncond[0])
steps = [group[0] for group in calls_by_step]
cond = []
for k, _ in steps:
    c = model.cond_module(get_timestep_embedding(k["t"][0:1], model.cfg.timestep_embed_dim))
    if meanflow:
        c = c + model.delta_cond_module(get_timestep_embedding(k["delta_t"][0:1], model.cfg.timestep_embed_dim))
    cond.append(c[0])
save("dit_cond", torch.stack(cond))
blocks = []
hooks = [b.register_forward_hook(lambda m, a, out: blocks.append(out)) for b in model.blocks]
again = model.forward_with_encoded_conditions(**steps[0][0])
for h in hooks:
    h.remove()
assert torch.equal(again, steps[0][1])
save("dit_step0_blocks", torch.stack(blocks))

# Codec: the decoder's stages on the final latent.
z, audio = calls["decode"][0]
assert torch.equal(z, xs[-1][None, : z.shape[1]])
dac = codec.model
h = dac.quantizer.out_proj(z.transpose(1, 2))
save("codec_in", h[0])
h = dac.decoder.model[0](h)
save("codec_conv_in", h[0])
for i, block in enumerate(dac.decoder.model[1:]):
    h = block(h)
    save(f"codec_block{i}", h[0])
h = dac.decoder.watermark(h)
assert torch.equal(h, audio)
save("wav", audio.reshape(-1))
save("audio", result.audio.reshape(-1))

for name, array in saved.items():
    np.save(os.path.join(args.out_dir, f"{name}.npy"), array)
meta = {
    "code": CODE, "model": MODELS[args.model], "codec": CODEC, "seed": args.seed,
    "text": args.text, "normalized_text": normalized_text, "tokens": n,
    "reference": None if args.no_ref else os.path.basename(args.ref_embed) if args.ref_embed
                 else (os.path.basename(args.references[0]) if len(args.references) == 1 else [os.path.basename(r) for r in args.references]),
    "embedding": bool(args.ref_embed),
    **({"ref_normalize_db": fields["ref_normalize_db"]} if "ref_normalize_db" in fields else {}),
    "seconds": args.seconds, "duration_scale": args.duration_scale, "speed": args.speed,
    "steps": len(calls_by_step), "options": options, "branches": branch_names,
    "latent_frames": int(z.shape[1]), "audio_samples": int(result.audio.shape[-1]),
    "unpadded_text_max_abs_diff": unpadded_error,
    "messages": result.messages,
}
json.dump(meta, open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in meta.items() if k not in ("code", "messages")}, ensure_ascii=False))
