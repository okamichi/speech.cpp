"""Converts a pinned official Irodori-TTS checkpoint, Small-MF, Small, Anime or Large, with its codec to the one GGUF file
the C++ port reads.

usage: uv run python convert.py <mf|rf|anime|large> <out dir> [--model-dir DIR] [--codec-dir DIR] [--check-only]

--check-only validates the local checkpoint mapping without reading tensor data or loading the codec.
Conversion stages tensors through a temporary file; allow space for it and the final GGUF.

Writes the model's file, named under GGUF's naming convention from its identity and the parameters of its tensors, in
layout 3: the tokenizer, ModernBERT-ja or T5Gemma2 and the projectors that make the text and the caption conditions from it, the
speaker encoder, the duration predictor with the null speaker of a request without a reference, the DiT with the
caption's keys and values, and Semantic-DACVAE-Japanese-32dim, the codec, with every constant the C++ reads, the hash
of the codec's tensors, which voice files carry, and the model's identity in the GGUF specification's general keys.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). Every tensor is written in float32; `speech quantize` makes the other weight types of the file, each
tensor in the type src/families/irodori-tts/layout.cpp gives it.

The codec's weight normalization is folded by the dacvae library itself (remove_weight_norm). Every convolution
weight is stored as numpy [k, out, in] (ne = [in, out, k]), so that tap k is a plain [in, out] matrix: the C++ side
runs a convolution as one matrix product per tap on channel-first activations. A transposed convolution's weight
[in, out, k] is stored the same way as [k, out, in]. A strided convolution of width 2s is stored as its two halves,
each [out, s * in] with the input channel fastest, so that it runs as two matrix products on the input cut into
frames of s samples.
"""

import argparse
import dataclasses
import hashlib
import inspect
import json
import os
import struct

import numpy as np
import torch
import yaml
from dacvae import DACVAE
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from irodori_tts import inference_runtime
from irodori_tts.model import precompute_freqs_cis
from safetensors import safe_open

from pins import CODEC, MODELS, snapshot
from gguf_tokenizer import write_tokenizer

ARCH = "irodori-tts"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0", 2: "0.8.0", 3: "0.9.0"}
LAYOUT = max(RELEASES)
# The SPDX identifier of each license a model card names.
LICENSES = {"mit": "MIT", "gemma": "LicenseRef-Gemma"}
# The parts of each model's name under GGUF's naming convention (ggml's docs/gguf.md) besides the size. The names' word
# for the size, Small, gives way to a size label counted from the parameters of the file's tensors, since the
# convention's size label is a number; MF is v4.1-Small distilled with MeanFlow.
NAMES = {"large": {"basename": "Irodori-TTS", "finetune": None, "version": "v4"},
         "anime": {"basename": "Irodori-TTS", "finetune": "Anime", "version": "v4.1"},
         "mf": {"basename": "Irodori-TTS", "finetune": "MF", "version": "v4.1"},
         "rf": {"basename": "Irodori-TTS", "finetune": None, "version": "v4.1"}}
# The bounds of OpenAI's speed, which Irodori-TTS-Server takes and divides the length by
# (Aratako/Irodori-TTS-Server@61012c760f22f7b4a6c21c5c5f8f9e148120b6f9, src/irodori_openai_tts/app.py).
MIN_SPEED, MAX_SPEED = 0.25, 4.0

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=sorted(MODELS))
parser.add_argument("out_dir")
parser.add_argument("--model-dir", help="local full-precision checkpoint and tokenizer folder")
parser.add_argument("--codec-dir", help="local folder containing the pinned DACVAE weights.pth")
parser.add_argument("--check-only", action="store_true")
parser.add_argument("--output-path-file", help="write the completed GGUF path to this UTF-8 file")
args = parser.parse_args()
if not args.check_only:
    os.makedirs(args.out_dir, exist_ok=True)

pin = MODELS[args.model]
model_dir = args.model_dir if args.model_dir else snapshot(pin)
if not args.check_only and "sha256" in pin:
    model_path = os.path.join(model_dir, "model.safetensors")
    if os.path.getsize(model_path) != pin["size"]:
        raise ValueError(f"Checkpoint size differs from the pinned model: {model_path}")
    digest = hashlib.sha256()
    with open(model_path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise ValueError(f"Checkpoint SHA-256 differs from the pinned model: {model_path}")
checkpoint = safe_open(os.path.join(model_dir, "model.safetensors"), "np")
config = json.loads(checkpoint.metadata()["config_json"])
text_config = json.loads(checkpoint.metadata()["text_encoder_config_json"])
assert config["text_encoder_type"] == "pretrained"
encoder_type = text_config["model_type"]
assert encoder_type in ("modernbert", "t5gemma2"), f"Unsupported text encoder: {encoder_type}"
if encoder_type == "t5gemma2":
    eoi_id = int(text_config["eoi_token_index"])
    text_config = text_config["encoder"]["text_config"]
    assert text_config["hidden_activation"] == "gelu_pytorch_tanh"
    assert text_config["attn_logit_softcapping"] is None and not text_config["attention_bias"]
    assert text_config["rope_parameters"]["full_attention"]["rope_type"] == "linear"
    assert text_config["rope_parameters"]["sliding_attention"]["rope_type"] == "default"
assert config["pretrained_projector_type"] == "residual_mlp" and config["latent_patch_size"] == 1
card = yaml.safe_load(open(os.path.join(model_dir, "README.md"), encoding="utf-8").read().split("---\n")[1])
license_id = LICENSES[card["license"]]
# The Anime card describes Japanese speech but omits the language field.
languages = ["ja"] if args.model == "anime" else sorted(card["language"])

# The official runtime's defaults: SamplingRequest's, and the steps synthesize() takes when a request names none.
defaults = {f.name: f.default for f in dataclasses.fields(inference_runtime.SamplingRequest)}
meanflow = config.get("flow_parameterization", "rf_velocity") == "meanflow"
assert "num_steps = (4 if is_meanflow else 40) if req.num_steps is None" in inspect.getsource(inference_runtime)
default_steps = 4 if meanflow else 40
# The time below which the speaker's keys and values are no longer scaled, when a request scales them and names none.
assert "0.9 if req.speaker_kv_min_t is None else float(req.speaker_kv_min_t)" in inspect.getsource(inference_runtime)
speaker_kv_min_t = 0.9

w = GGUFWriter(None, ARCH, use_temp_file=not args.check_only)


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "synthesis")
# Irodori-TTS takes no language with a request; a request's language is only checked against the model's.
w.add_string("speech.language_use", "checked")

p = f"{ARCH}."
w.add_string(p + "flow", "meanflow" if meanflow else "rf_velocity")
w.add_uint32(p + "latent_dim", int(config["latent_dim"]))
w.add_float32(p + "norm_eps", float(config["norm_eps"]))
# The RoPE base of the speaker encoder and the DiT, which both call precompute_freqs_cis() without one.
w.add_float32(p + "rope_theta", float(inspect.signature(precompute_freqs_cis).parameters["theta"].default))

w.add_string(p + "text.encoder", encoder_type)
w.add_uint32(p + "text.hidden_size", int(text_config["hidden_size"]))
w.add_uint32(p + "text.num_layers", int(text_config["num_hidden_layers"]))
w.add_uint32(p + "text.num_heads", int(text_config["num_attention_heads"]))
add_array(p + "text.layer_global", [int(t == "full_attention") for t in text_config["layer_types"]], GGUFValueType.INT32)
w.add_float32(p + "text.rope_theta_global", float(text_config["rope_parameters"]["full_attention"]["rope_theta"]))
w.add_float32(p + "text.rope_theta_local", float(text_config["rope_parameters"]["sliding_attention"]["rope_theta"]))
if encoder_type == "modernbert":
    assert text_config["hidden_activation"] == "gelu" and not text_config["norm_bias"]
    assert not text_config["attention_bias"] and not text_config["mlp_bias"]
    w.add_float32(p + "text.norm_eps", float(text_config["norm_eps"]))
    w.add_uint32(p + "text.window", int(text_config["local_attention"]) // 2)
else:
    w.add_float32(p + "text.norm_eps", float(text_config["rms_norm_eps"]))
    w.add_uint32(p + "text.window", int(text_config["sliding_window"]))
    w.add_uint32(p + "text.num_kv_heads", int(text_config["num_key_value_heads"]))
    w.add_uint32(p + "text.head_dim", int(text_config["head_dim"]))
    w.add_float32(p + "text.attention_scale", float(text_config["query_pre_attn_scalar"]) ** -0.5)
    w.add_float32(p + "text.embedding_scale", float(text_config["hidden_size"]) ** 0.5)
    w.add_float32(p + "text.rope_scale_global", 1.0 / float(text_config["rope_parameters"]["full_attention"]["factor"]))
    w.add_uint32(p + "text.eoi_id", eoi_id)
w.add_uint32(p + "text.dim", int(config["text_dim"]))
w.add_uint32(p + "text.max_tokens", int(config["max_text_len"]))

w.add_uint32(p + "speaker.dim", int(config["speaker_dim"]))
w.add_uint32(p + "speaker.num_layers", int(config["speaker_layers"]))
w.add_uint32(p + "speaker.num_heads", int(config["speaker_heads"]))
w.add_uint32(p + "speaker.patch_size", int(config["speaker_patch_size"]))
assert config["duration_architecture"] == "token_sum_dual_adarn_zero_no_aux"
w.add_uint32(p + "duration.num_layers", int(config["duration_layers"]))
# The predictor's speaker when a request has no reference (the runtime's no_ref), which a checkpoint of a speaker
# condition holds.
null_speaker = "duration_predictor.null_speaker" in checkpoint.keys()
w.add_bool(p + "duration.null_speaker", null_speaker)
# The caption condition: the shared ModernBERT-ja with the caption's own projector and norm, which the DiT attends to
# after the speaker and the duration predictor takes as the mean of its tokens. The width is the one
# ModelConfig.caption_dim_resolved gives, which the duration predictor's caption modulation takes whether or not the
# file holds the caption's encoder.
caption = bool(config["use_caption_condition"])
caption_dim = int(config["text_dim"] if config.get("caption_dim") is None else config["caption_dim"])
w.add_uint32(p + "caption.dim", caption_dim)
w.add_bool(p + "caption_condition", caption)
if caption:
    # The caption shares the text's tokenizer, <s> and ModernBERT, so it needs no tokenizer or encoder of its own.
    assert config["caption_tokenizer_repo"] in (None, config["text_tokenizer_repo"])
    assert config["caption_add_bos"] in (None, config["text_add_bos"]) and config["text_add_bos"]
    assert config["duration_caption_fusion"] == "adarn_zero" and config["duration_caption_pooling"] == "masked_mean"
    w.add_uint32(p + "caption.max_tokens", int(config["max_caption_len"]))
w.add_uint32(p + "dit.dim", int(config["model_dim"]))
w.add_uint32(p + "dit.num_layers", int(config["num_layers"]))
w.add_uint32(p + "dit.num_heads", int(config["num_heads"]))
w.add_uint32(p + "dit.timestep_dim", int(config["timestep_embed_dim"]))

# RF's guidance against a branch without the text, one without the speaker and one without the caption while t is in
# its range, and the time at which a request's scaling of the speaker's keys and values ends; MeanFlow folded the
# guidance into its training and takes none.
w.add_uint32(p + "sampler.default_steps", default_steps)
if not meanflow:
    w.add_float32(p + "sampler.cfg_text", float(defaults["cfg_scale_text"]))
    w.add_float32(p + "sampler.cfg_speaker", float(defaults["cfg_scale_speaker"]))
    w.add_float32(p + "sampler.cfg_min_t", float(defaults["cfg_min_t"]))
    w.add_float32(p + "sampler.cfg_max_t", float(defaults["cfg_max_t"]))
    w.add_float32(p + "sampler.speaker_kv_min_t", speaker_kv_min_t)
    if caption:
        w.add_float32(p + "sampler.cfg_caption", float(defaults["cfg_scale_caption"]))
w.add_float32(p + "length.min_seconds", float(defaults["min_seconds"]))
w.add_float32(p + "length.max_seconds", float(defaults["max_seconds"]))
w.add_float32(p + "length.min_speed", MIN_SPEED)
w.add_float32(p + "length.max_speed", MAX_SPEED)
w.add_float32(p + "reference.max_seconds", float(config["ref_max_seconds"]))
w.add_float32(p + "reference.lufs", float(defaults["ref_normalize_db"]))
w.add_uint32(p + "tail.window", int(defaults["tail_window_size"]))
w.add_float32(p + "tail.std_threshold", float(defaults["tail_std_threshold"]))
w.add_float32(p + "tail.mean_threshold", float(defaults["tail_mean_threshold"]))

# The SentencePiece Unigram tokenizer, with its scores in float64 as tokenizer.json has them, since the
# Viterbi path compares their sums.
tokenizer_type = write_tokenizer(w, model_dir)


used = set()


def tensor(key):
    """Reads one full-precision tensor, or only its shape during preflight."""
    used.add(key)
    if args.check_only:
        return np.broadcast_to(np.float32(0), checkpoint.get_slice(key).get_shape())
    return checkpoint.get_tensor(key).astype(np.float32, copy=False)


def add(out_name, data):
    if args.check_only:
        w.add_tensor_info(out_name, data.shape, np.dtype(np.float32), data.size * 4)
    else:
        w.add_tensor(out_name, np.ascontiguousarray(data, dtype=np.float32))


# ModernBERT-ja: the fused query, key and value projection is split, and so is the MLP's input projection
# into the half that goes through GELU and the half that gates it.
b = "pretrained_text_backbone.backbone."
hidden = int(text_config["hidden_size"])
inner = int(text_config["intermediate_size"])
if encoder_type == "modernbert":
    add("text.embd", tensor(b + "embeddings.tok_embeddings.weight"))
    add("text.embd_norm", tensor(b + "embeddings.norm.weight"))
    for i in range(int(text_config["num_hidden_layers"])):
        a, o = f"{b}layers.{i}.", f"text.blk.{i}."
        if i > 0:
            add(o + "attn_norm", tensor(a + "attn_norm.weight"))
        qkv = tensor(a + "attn.Wqkv.weight")
        add(o + "attn_q", qkv[:hidden])
        add(o + "attn_k", qkv[hidden : 2 * hidden])
        add(o + "attn_v", qkv[2 * hidden :])
        add(o + "attn_out", tensor(a + "attn.Wo.weight"))
        add(o + "ffn_norm", tensor(a + "mlp_norm.weight"))
        wi = tensor(a + "mlp.Wi.weight")
        add(o + "ffn_act", wi[:inner])
        add(o + "ffn_gate", wi[inner:])
        add(o + "ffn_down", tensor(a + "mlp.Wo.weight"))
    add("text.final_norm", tensor(b + "final_norm.weight"))
else:
    add("text.embd", tensor(b + "embed_tokens.weight"))
    add("text.eoi_embd", tensor(b + "embed_tokens.eoi_embedding"))
    add("text.final_norm", tensor(b + "norm.weight") + np.float32(1))
    for i in range(int(text_config["num_hidden_layers"])):
        a, o = f"{b}layers.{i}.", f"text.blk.{i}."
        for source, target in (("pre_self_attn_layernorm", "attn_norm"), ("post_self_attn_layernorm", "attn_post_norm"),
                               ("pre_feedforward_layernorm", "ffn_norm"), ("post_feedforward_layernorm", "ffn_post_norm")):
            add(o + target, tensor(a + source + ".weight") + np.float32(1))
        for source, target in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "out")):
            add(o + "attn_" + target, tensor(a + "self_attn." + source + "_proj.weight"))
        for source in ("q", "k"):
            add(o + source + "_norm", tensor(a + "self_attn." + source + "_norm.weight") + np.float32(1))
        for source, target in (("gate", "act"), ("up", "gate"), ("down", "down")):
            add(o + "ffn_" + target, tensor(a + "mlp." + source + "_proj.weight"))

def projector(source, out):
    """A PretrainedConditionProjector of ModernBERT's output: a linear map plus an RMS-normed residual MLP."""
    add(out + ".weight", tensor(source + "projector.weight"))
    add(out + ".bias", tensor(source + "projector.bias"))
    add(out + ".res_norm", tensor(source + "residual_norm.weight"))
    add(out + ".res_up.weight", tensor(source + "residual_up.weight"))
    add(out + ".res_up.bias", tensor(source + "residual_up.bias"))
    add(out + ".res_down.weight", tensor(source + "residual_down.weight"))
    add(out + ".res_down.bias", tensor(source + "residual_down.bias"))


projector("text_encoder.", "text.proj")
add("text.norm", tensor("text_norm.weight"))
if caption:
    projector("caption_encoder.", "caption.proj")
    add("caption.norm", tensor("caption_norm.weight"))


def swiglu(prefix_in, prefix_out):
    add(prefix_out + "ffn_gate", tensor(prefix_in + "w1.weight"))
    add(prefix_out + "ffn_up", tensor(prefix_in + "w3.weight"))
    add(prefix_out + "ffn_down", tensor(prefix_in + "w2.weight"))


# The speaker encoder: a pre-norm transformer on the reference latent in patches of four frames.
add("speaker.in_proj.weight", tensor("speaker_encoder.in_proj.weight"))
add("speaker.in_proj.bias", tensor("speaker_encoder.in_proj.bias"))
for i in range(int(config["speaker_layers"])):
    a, o = f"speaker_encoder.blocks.{i}.", f"speaker.blk.{i}."
    add(o + "attn_norm", tensor(a + "attention_norm.weight"))
    for x in ("q", "k", "v", "o"):
        add(o + f"attn_{x}", tensor(a + f"attention.w{x}.weight"))
    add(o + "attn_gate", tensor(a + "attention.gate.weight"))
    add(o + "q_norm", tensor(a + "attention.q_norm.weight"))
    add(o + "k_norm", tensor(a + "attention.k_norm.weight"))
    add(o + "ffn_norm", tensor(a + "mlp_norm.weight"))
    swiglu(a + "mlp.", o)
add("speaker.norm", tensor("speaker_norm.weight"))

# The duration predictor; without a caption its caption vector is the learned null one, and without a reference its
# speaker vector.
d = "duration_predictor."
add("duration.in_proj.weight", tensor(d + "token_input_proj.weight"))
add("duration.in_proj.bias", tensor(d + "token_input_proj.bias"))
for i in range(int(config["duration_layers"])):
    a, o = f"{d}token_blocks.{i}.", f"duration.blk.{i}."
    add(o + "norm", tensor(a + "norm.weight"))
    add(o + "mod.weight", tensor(a + "modulation.weight"))
    add(o + "mod.bias", tensor(a + "modulation.bias"))
    add(o + "caption_mod.weight", tensor(a + "caption_modulation.weight"))
    add(o + "caption_mod.bias", tensor(a + "caption_modulation.bias"))
    swiglu(a + "mlp.", o)
add("duration.out_norm", tensor(d + "token_out_norm.weight"))
add("duration.out_proj.weight", tensor(d + "token_out_proj.weight"))
add("duration.out_proj.bias", tensor(d + "token_out_proj.bias"))
add("duration.null_caption", tensor(d + "null_caption"))
if null_speaker:
    add("duration.null_speaker", tensor(d + "null_speaker"))

# The DiT.
for i, layer in enumerate((0, 2, 4)):
    add(f"dit.cond.{i}", tensor(f"cond_module.{layer}.weight"))
    if meanflow:
        add(f"dit.delta_cond.{i}", tensor(f"delta_cond_module.{layer}.weight"))
add("dit.in_proj.weight", tensor("in_proj.weight"))
add("dit.in_proj.bias", tensor("in_proj.bias"))
for i in range(int(config["num_layers"])):
    a, o = f"blocks.{i}.", f"dit.blk.{i}."
    for x in ("q", "k", "v", "o"):
        add(o + f"attn_{x}", tensor(a + f"attention.w{x}.weight"))
    add(o + "attn_gate", tensor(a + "attention.gate.weight"))
    for x in ("k", "v"):
        add(o + f"attn_{x}_text", tensor(a + f"attention.w{x}_text.weight"))
        add(o + f"attn_{x}_speaker", tensor(a + f"attention.w{x}_speaker.weight"))
        if caption:
            add(o + f"attn_{x}_caption", tensor(a + f"attention.w{x}_caption.weight"))
    add(o + "q_norm", tensor(a + "attention.q_norm.weight"))
    add(o + "k_norm", tensor(a + "attention.k_norm.weight"))
    swiglu(a + "mlp.", o)
    for ada, out in (("attention_adaln", "attn_ada"), ("mlp_adaln", "ffn_ada")):
        for part in ("shift", "scale", "gate"):
            add(o + f"{out}.{part}.down", tensor(a + f"{ada}.{part}_down.weight"))
            add(o + f"{out}.{part}.up.weight", tensor(a + f"{ada}.{part}_up.weight"))
            add(o + f"{out}.{part}.up.bias", tensor(a + f"{ada}.{part}_up.bias"))
add("dit.out_norm", tensor("out_norm.weight"))
add("dit.out_proj.weight", tensor("out_proj.weight"))
add("dit.out_proj.bias", tensor("out_proj.bias"))


unused = set(checkpoint.keys()) - used
assert not unused, f"Checkpoint tensors are not mapped: {sorted(unused)}"
if args.check_only:
    print(f"Checked {len(used)} checkpoint tensors: {encoder_type}, {tokenizer_type}; no GGUF was written.")
    raise SystemExit(0)

codec = DACVAE.load(os.path.join(args.codec_dir if args.codec_dir else snapshot(CODEC), "weights.pth")).eval()
encoder_rates = [int(r) for r in codec.encoder_rates]
decoder_rates = [int(r) for r in codec.decoder_rates]
assert int(codec.hop_length) == int(np.prod(encoder_rates))
assert int(codec.quantizer.codebook_dim) == int(config["latent_dim"]), "the model's latent is not the codec's"


def codec_sha256(model):
    """SHA-256 over the official codec's tensors as DACVAE.load() gives them, before the weight normalization is folded:
    for each in ascending order of its name, the name in UTF-8, a 0 byte, the number of dimensions as a little-endian
    u32, each dimension as a little-endian u64, and the values as little-endian float32 in row-major order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        values = np.ascontiguousarray(tensor.detach().cpu().float().numpy(), dtype="<f4")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(struct.pack("<I", values.ndim) + struct.pack(f"<{values.ndim}Q", *values.shape))
        digest.update(values.tobytes())
    return digest.hexdigest()


sha256 = codec_sha256(codec)
for module in codec.modules():
    try:
        torch.nn.utils.remove_weight_norm(module)
    except ValueError:
        pass

w.add_uint32("speech.sample_rate", int(codec.sample_rate))
w.add_uint32(p + "codec.hop_length", int(codec.hop_length))
add_array(p + "codec.encoder_rates", encoder_rates, GGUFValueType.INT32)
add_array(p + "codec.decoder_rates", decoder_rates, GGUFValueType.INT32)
w.add_string(p + "codec.sha256", sha256)

# The codec stays in float32 in every weight type.
def array(t):
    return t.detach().float().numpy()


def add_codec(out_name, data):
    w.add_tensor("codec." + out_name, np.ascontiguousarray(data, dtype=np.float32))


def conv(prefix, module):
    """A stride-1 Conv1d: weight [out, in, k] stored as [k, out, in]."""
    assert module.stride[0] == 1 and module.padding[0] == (module.kernel_size[0] - 1) * module.dilation[0] // 2
    add_codec(prefix + ".weight", np.transpose(array(module.weight), (2, 0, 1)))
    add_codec(prefix + ".bias", array(module.bias))


def snake(prefix, module):
    """Snake: x + sin(alpha x)^2 / (alpha + 1e-9), with the reciprocal computed in float32 as dacvae does."""
    alpha = array(module.alpha).reshape(-1)
    add_codec(prefix + ".alpha", alpha)
    add_codec(prefix + ".inv_alpha", np.float32(1.0) / (alpha + np.float32(1e-9)))


def residual(prefix, unit):
    snake(prefix + ".snake1", unit.block[0])
    conv(prefix + ".conv1", unit.block[1])
    snake(prefix + ".snake2", unit.block[2])
    conv(prefix + ".conv2", unit.block[3])


# The encoder and the mean of the bottleneck, which turn a reference voice into the latent the speaker encoder reads.
encoder = codec.encoder.block
conv("enc.conv_in", encoder[0])
for i, stride in enumerate(encoder_rates):
    block = encoder[i + 1].block
    e = f"enc.blk.{i}"
    # The three residual units of dilations 1, 3 and 9, which the C++ runs as the architecture's.
    for j in range(3):
        assert block[j].block[1].dilation[0] == 3**j
        residual(f"{e}.res.{j}", block[j])
    snake(e + ".snake", block[3])
    down = block[4]
    assert down.stride[0] == stride and down.kernel_size[0] == 2 * stride and down.padding[0] == stride // 2
    weight = array(down.weight)
    add_codec(e + ".down.first", np.transpose(weight[:, :, :stride], (0, 2, 1)).reshape(weight.shape[0], -1))
    add_codec(e + ".down.second", np.transpose(weight[:, :, stride:], (0, 2, 1)).reshape(weight.shape[0], -1))
    add_codec(e + ".down.bias", array(down.bias))
snake("enc.snake", encoder[len(encoder_rates) + 1])
conv("enc.conv_out", encoder[len(encoder_rates) + 2])

# The bottleneck's mean: the first half of in_proj's channels; Irodori-TTS encodes deterministically.
latent_dim = int(codec.quantizer.codebook_dim)
in_proj = codec.quantizer.in_proj
add_codec("bottleneck.mean.weight", array(in_proj.weight)[:latent_dim, :, 0])
add_codec("bottleneck.mean.bias", array(in_proj.bias)[:latent_dim])

# The decoder. Of each block's layers the forward pass uses the Snake, the transposed convolution and the
# residual units of dilation 1, 3 and 9 (the other layers belong to the watermark's path). Irodori-TTS
# replaces the watermark with the first Snake, convolution and tanh of its encoder block.
conv("dec.in_proj", codec.quantizer.out_proj)
conv("dec.conv_in", codec.decoder.model[0])
for i, stride in enumerate(decoder_rates):
    block = codec.decoder.model[i + 1].block
    e = f"dec.blk.{i}"
    snake(e + ".snake", block[0])
    up = block[1]
    assert up.stride[0] == stride and up.kernel_size[0] == 2 * stride and up.padding[0] == stride // 2
    assert stride % 2 == 0 and up.output_padding[0] == 0
    add_codec(e + ".up.weight", np.transpose(array(up.weight), (2, 1, 0)))
    add_codec(e + ".up.bias", array(up.bias))
    for j, layer in enumerate((4, 5, 8)):
        assert block[layer].block[1].dilation[0] == 3**j
        residual(f"{e}.res.{j}", block[layer])
watermark = codec.decoder.wm_model.encoder_block.pre
snake("dec.out_snake", watermark[0])
conv("dec.conv_out", watermark[1])

# The model's identity and languages in the GGUF specification's general keys, which name the file. They follow the
# tensors, whose parameters give the size label, and then go first after general.architecture, since GGUFWriter
# writes the keys in the order they were added.
parts = NAMES[args.model]
label = size_label(*w.get_total_parameter_count())
repository = f"https://huggingface.co/{pin['repository']}"
w.add_name(pin["repository"].split("/")[1])
w.add_organization(pin["repository"].split("/")[0])
w.add_basename(parts["basename"])
w.add_size_label(label)
if parts["finetune"]:
    w.add_finetune(parts["finetune"])
w.add_version(parts["version"])
w.add_license(license_id)
w.add_source_url(f"{repository}/tree/{pin['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(languages)
w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

path = os.path.join(args.out_dir, naming_convention(None, parts["basename"], parts["finetune"], parts["version"], label, "f32") + ".gguf")
w.write_header_to_file(path)
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
if args.output_path_file:
    with open(args.output_path_file, "w", encoding="utf-8") as f:
        f.write(os.path.abspath(path) + "\n")
print("wrote", path, "with the codec", sha256)
