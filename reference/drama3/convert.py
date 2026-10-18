"""Converts the pinned Fish Audio Drama 3 checkpoint with its codec to the one GGUF file the C++ port reads.

usage: uv run python convert.py <out dir>

Writes the model's file, named under GGUF's naming convention, in layout 1: the text model (a Qwen3.5-MoE backbone of
Gated DeltaNet and gated attention layers, each with 256 experts and a shared expert), its output head for the ids a
frame's main token is drawn from, the audio decoder that draws a frame's other codes, the tokenizer, and the codec, a
causal DAC at 44.1 kHz with transformers around its quantizer, with every constant the C++ reads, the hash of the
codec's tensors, which voice files carry, and the model's identity in the GGUF specification's general keys.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is (ne = [in,
out]). Every tensor is written in float32, about 143 GB, a tensor at a time, so that the conversion holds one layer's
experts in memory at most; `speech quantize` makes the other weight types, each tensor in the type
src/families/drama3/layout.cpp gives it.

- The FP8 experts are dequantized as fish-speech's dequantize_fp8_blocks() does, each 128 x 128 block of E4M3 values
  times its float32 scale, and each layer's 256 experts of a projection are stacked into one tensor [256, out, in],
  which ggml_mul_mat_id() reads.
- A zero-centred RMS norm, which scales by 1 + weight, is stored as 1 + weight, the float32 sum the official code
  computes on every call.
- An attention's query projection, whose rows give each head its query and then its output gate, is stored as two
  matrices; the audio decoder's fused [query | gate | key | value] projection as four.
- The output head keeps the rows of the end id and the semantic ids alone, in the order of their ids, and the audio
  decoder's code embeddings those of codebooks 1 to 9: nothing reads the others.
- The codec's weight normalization is folded as torch's weight_norm computes it. A convolution's weight is stored as
  numpy [k, out, in] (ne = [in, out, k]), so that tap k is a plain [in, out] matrix, and so is a transposed
  convolution's [in, out, k]; a strided convolution of width 2s is stored as its two halves, each [out, s * in] with
  the input channel fastest, and one whose width is its stride as one such matrix.
"""

import argparse
import hashlib
import json
import os
import struct

import numpy as np
import torch
import yaml
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention
from safetensors import safe_open

from fish_speech.models.text2semantic.fish_omni import generate as fish_generate
from fish_speech.models.text2semantic.fish_omni import chunking as fish_chunking
from fish_speech.models.text2semantic.fish_omni import tower as fish_tower
from fish_speech.models.text2semantic.fish_omni.config import FishOmniConfig
from fish_speech.models.text2semantic.inference import load_codec_model, resolve_codec_path
from fish_speech.models.text2semantic.sampling import DEFAULT_MAX_NEW_TOKENS

from pins import MODEL, snapshot

ARCH = "drama3"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.8.8"}
LAYOUT = max(RELEASES)
# The model's name under GGUF's naming convention (ggml's docs/gguf.md): the size label counts its parameters and the
# ones a token computes with, as Qwen3.5's names do.
BASENAME, VERSION, SIZE_LABEL = "Drama", "v3", "35B-A3B"
# The model card's sampling, with which Fish Audio measured the model and which SGLang-Omni takes by default.
CARD_SAMPLING = {"temperature": 1.0, "top_p": 0.9, "top_k": 30}
# The model card's languages that BCP 47 spells otherwise: Javanese is `jv`, which the card writes `jw`.
LANGUAGE_FIXES = {"jw": "jv"}

parser = argparse.ArgumentParser()
parser.add_argument("out_dir")
args = parser.parse_args()
os.makedirs(args.out_dir, exist_ok=True)

folder = snapshot()
config = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
text, audio = config["text_config"], config["audio_decoder_config"]
assert config["model_type"] == "fish_omni" and config["speech_template_version"] == "reference-bracket-v1"
assert config["quantization_config"]["weight_block_size"] == [128, 128]
assert text["layernorm_zero_centered_gamma"] and audio["layernorm_zero_centered_gamma"]
assert not text["rotary_interleaved"] and audio["rotary_interleaved"], "the backbone rotates halves and the audio decoder pairs"
assert audio["decoder_type"] == "codebook_mtp_shared" and not audio["use_moe"]
card = yaml.safe_load(open(os.path.join(folder, "README.md"), encoding="utf-8").read().split("---\n")[1])
languages = sorted({LANGUAGE_FIXES.get(code, code) for code in card["language"]})
assert all(2 <= len(code) <= 3 and code.isalpha() and code.islower() for code in languages)
tokenizer = json.load(open(os.path.join(folder, "tokenizer.json"), encoding="utf-8"))
assert tokenizer["model"]["type"] == "BPE" and tokenizer["normalizer"] == {"type": "NFC"}
index = json.load(open(os.path.join(folder, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
shards, read = {}, set()
# The values fish-speech takes where the checkpoint's configuration sets none, such as norm_topk_prob.
official = FishOmniConfig.from_dict(config)


def checkpoint(name):
    """A tensor of the checkpoint as torch reads it, in its own type."""
    shard = index[name]
    read.add(name)
    if shard not in shards:
        shards[shard] = safe_open(os.path.join(folder, shard), "pt")
    return shards[shard].get_tensor(name)


def f32(name):
    return checkpoint(name).to(torch.float32).numpy()


def zero_centred(name):
    """A zero-centred RMS norm's weight w, by which it scales as 1 + w; stored as 1 + w, the float32 sum it computes."""
    return np.float32(1.0) + f32(name)


def experts(layer, projection):
    """The 256 experts of one projection of a layer, [256, out, in], each 128 x 128 block of FP8 values times its scale."""
    stacked = []
    for e in range(int(text["num_experts"])):
        key = f"model.language_model.layers.{layer}.mlp.experts.{e}.{projection}"
        weight, scale = checkpoint(key + ".weight"), checkpoint(key + ".weight_scale_inv")
        rows, cols = weight.shape
        blocks = weight.view(rows // 128, 128, cols // 128, 128).to(torch.float32) * scale[:, None, :, None]
        stacked.append(blocks.reshape(rows, cols))
    return torch.stack(stacked).numpy()


codec_model = load_codec_model(resolve_codec_path(folder, os.path.join(folder, "codec.pth")), "cpu", torch.float32)
raw_codec = torch.load(os.path.join(folder, "codec.pth"), map_location="cpu", mmap=True, weights_only=True)
raw_codec = raw_codec.get("state_dict", raw_codec)
rebuilt = {f"{n}.{b}" if n else b for n, module in codec_model.named_modules() for b in module._non_persistent_buffers_set}


def codec_sha256():
    """SHA-256 over the official codec's tensors as codec.pth stores them, less the buffers the codec rebuilds when it
    loads (its RoPE tables and causal masks): for each in ascending order of its name, the name in UTF-8, a 0 byte, the
    number of dimensions as a little-endian u32, each dimension as a little-endian u64, and the values as little-endian
    float32 in row-major order."""
    digest = hashlib.sha256()
    for name in sorted(k for k in raw_codec if k not in rebuilt):
        values = np.ascontiguousarray(raw_codec[name].float().numpy(), dtype="<f4")
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(struct.pack("<I", values.ndim) + struct.pack(f"<{values.ndim}Q", *values.shape))
        digest.update(values.tobytes())
    return digest.hexdigest()


class Recorded(dict):
    """The codec's tensors, recording which the conversion takes, so that it can check it took them all."""

    def __init__(self, values):
        super().__init__(values)
        self.taken = set()

    def __getitem__(self, key):
        self.taken.add(key)
        return super().__getitem__(key)


codec_state = Recorded({k: v.numpy() for k, v in codec_model.state_dict().items() if k not in rebuilt})
qcfg = codec_model.quantizer
encoder_rates = [int(r) for r in codec_model.encoder_rates]
decoder_rates = [int(r) for r in codec_model.decoder_rates]
down_factors = [int(f) for f in qcfg.downsample_factor]
hop = int(np.prod(encoder_rates) * np.prod(down_factors))
assert codec_model.sample_rate == 44100 and hop == 2048, "the codec is not the 44.1 kHz one of 21.5 frames a second"

path = os.path.join(args.out_dir, naming_convention(None, BASENAME, None, VERSION, SIZE_LABEL, "f32") + ".gguf")
w = GGUFWriter(path, ARCH)
plan = []


def add_array(key, values, element):
    """An array with its element type given, not inferred from its first value."""
    w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)


def add(name, shape, make):
    """A tensor of numpy shape `shape` whose float32 values `make` gives when the tensor is written."""
    plan.append((name, tuple(int(d) for d in shape), make))


repository = f"https://huggingface.co/{MODEL['repository']}"
w.add_name("drama-3")
w.add_organization(MODEL["repository"].split("/")[0])
w.add_basename(BASENAME)
w.add_version(VERSION)
w.add_size_label(SIZE_LABEL)
# The Fish Audio Community License 1.0, which no SPDX identifier names.
w.add_license("LicenseRef-Fish-Audio-Community-License-1.0")
w.add_source_url(f"{repository}/tree/{MODEL['revision']}")
w.add_source_repo_url(repository)
w.add_file_type(LlamaFileType.ALL_F32)
w.add_languages(languages)
w.add_uint32("speech.layout", LAYOUT)
w.add_string("speech.requires", RELEASES[LAYOUT])
w.add_string("speech.task", "synthesis")
w.add_uint32("speech.sample_rate", int(codec_model.sample_rate))
# Drama 3 reads the language from the text; a request's language is only checked against the model's.
w.add_string("speech.language_use", "checked")

p = f"{ARCH}."
t = p + "text."
for key in ["hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
            "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "shared_expert_intermediate_size", "vocab_size", "max_position_embeddings"]:
    w.add_uint32(t + key, int(text[key]))
add_array(t + "layer_types", text["layer_types"], GGUFValueType.STRING)
assert set(text["layer_types"]) == {"linear_attention", "full_attention"} and set(text["mlp_layer_types"]) == {"moe"}
w.add_bool(t + "norm_topk_prob", bool(official.text.norm_topk_prob))
w.add_float32(t + "rms_norm_eps", float(text["rms_norm_eps"]))
w.add_float32(t + "rope_theta", float(text["rope_parameters"]["rope_theta"]))
w.add_uint32(t + "rotary_dim", int(text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"]))

a = p + "audio_decoder."
for key in ["dim", "n_head", "n_local_heads", "head_dim", "intermediate_size", "num_codebooks", "vocab_size",
            "shared_enrichment_layers", "vq_backbone_codebooks"]:
    w.add_uint32(a + key, int(audio[key]))
w.add_float32(a + "norm_eps", float(audio["norm_eps"]))
w.add_float32(a + "rope_base", float(audio["rope_base"]))
w.add_uint32(a + "rotary_dim", int(audio["head_dim"] * audio["rotary_percent"]))

for key in ["eos_token_id", "semantic_start_token_id", "semantic_end_token_id", "pad_token_id"]:
    w.add_uint32(p + key, int(config[key]))
candidates = sorted([int(config["eos_token_id"])] + list(range(config["semantic_start_token_id"], config["semantic_end_token_id"] + 1)))
assert len(candidates) == 1 + int(audio["vocab_size"])
add_array(p + "output.token_ids", candidates, GGUFValueType.INT32)

g = p + "sampling."
for key, value in CARD_SAMPLING.items():
    (w.add_uint32 if isinstance(value, int) else w.add_float32)(g + key, value)
# The redraw of a main token that repeats one of the last frames', which fish-speech's tower fixes from the training recipe.
r = p + "repeat."
w.add_uint32(r + "window", fish_tower.RAS_WINDOW_FRAMES)
w.add_float32(r + "temperature", fish_tower.RAS_TEMPERATURE)
w.add_uint32(r + "top_k", fish_tower.RAS_TOP_K)
w.add_float32(r + "top_p", fish_tower.RAS_TOP_P)
# fish-speech's server: the frames of a chunk when a request sets none, the most a text spoken in one pass takes, the
# rows of a prompt, and how it cuts a text into chunks with the turns before each.
q = p + "generation."
w.add_uint32(q + "default_frames", DEFAULT_MAX_NEW_TOKENS)
w.add_uint32(q + "max_frames", fish_generate.DRAMA3_DEFAULT_MAX_NEW_TOKENS)
w.add_uint32(q + "max_prompt_rows", fish_generate.DRAMA3_MAX_PROMPT_ROWS)
w.add_uint32(q + "chunk_bytes", fish_chunking.DRAMA3_CHUNK_BYTES)
w.add_uint32(q + "chunk_speakers", fish_chunking.DRAMA3_CHUNK_MAX_SPEAKERS)
w.add_uint32(q + "recent_turns", fish_chunking.DRAMA3_HISTORY_RECENT_TURNS)
w.add_uint32(q + "speakers", fish_generate.MAX_SPEAKER_ID + 1)
# The prompt of the checkpoint's template, the pieces build_drama3_prompt() writes around the texts and the codes: the
# system turn and its instruction, each reference as "\n\n[Reference <n>]\nText: <transcript>\nSpeech: <codes>", the end of
# a turn, a user turn, and the assistant's voice turn the frames follow. The system, reference and user pieces are
# literals of build_drama3_prompt() itself; the rest are its module's constants.
assert config["speech_template_version"] == "reference-bracket-v1", "the checkpoint's prompt template is not the one build_drama3_prompt() writes"
w.add_string(p + "prompt.template", config["speech_template_version"])
w.add_string(p + "prompt.system", "<|im_start|>system\n")
w.add_string(p + "prompt.instruction", fish_generate.DRAMA3_TTS_INSTRUCTION)
w.add_string(p + "prompt.reference_instruction", fish_generate.DRAMA3_REFERENCE_INSTRUCTION)
w.add_string(p + "prompt.reference_open", "\n\n[Reference ")
w.add_string(p + "prompt.reference_text", "]\nText: ")
w.add_string(p + "prompt.reference_speech", "\nSpeech: ")
w.add_string(p + "prompt.end", fish_generate.DRAMA3_MESSAGE_END)
w.add_string(p + "prompt.user", "<|im_start|>user\n")
w.add_string(p + "prompt.assistant", fish_generate.DRAMA3_GENERATION_START)
# The tokens only the prompt builder writes, which a text that contains one is refused for.
add_array(p + "prompt.reserved", list(fish_generate.DRAMA3_STRUCTURAL_TOKENS), GGUFValueType.STRING)

k = p + "tokenizer."
vocab = tokenizer["model"]["vocab"]
added = {entry["id"]: entry for entry in tokenizer["added_tokens"]}
size = max([*vocab.values(), *added]) + 1
tokens = [""] * size
for token, id_ in vocab.items():
    tokens[id_] = token
for id_, entry in added.items():
    assert not tokens[id_], f"the added token {entry['content']} takes the id of a token of the vocabulary"
    tokens[id_] = entry["content"]
assert all(tokens) and size <= int(text["vocab_size"])
add_array(k + "tokens", tokens, GGUFValueType.STRING)
add_array(k + "merges", [m if isinstance(m, str) else " ".join(m) for m in tokenizer["model"]["merges"]], GGUFValueType.STRING)
# Every added token, special or not, which the tokenizers library finds in a text before its pre-tokenizer runs.
assert all(not (e["lstrip"] or e["rstrip"] or e["single_word"] or e["normalized"]) for e in added.values()), \
    "an added token strips space, matches whole words or matches the normalized text, which the C++ does not"
add_array(k + "added_ids", sorted(added), GGUFValueType.INT32)
w.add_string(k + "pre_tokenizer", tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])

c = p + "codec."
w.add_string(c + "sha256", codec_sha256())
w.add_uint32(c + "hop_length", hop)
w.add_uint32(c + "encoder_dim", int(codec_model.encoder_dim))
add_array(c + "encoder_rates", encoder_rates, GGUFValueType.INT32)
w.add_uint32(c + "latent_dim", int(codec_model.latent_dim))
w.add_uint32(c + "decoder_dim", int(codec_model.decoder_dim))
add_array(c + "decoder_rates", decoder_rates, GGUFValueType.INT32)
add_array(c + "downsample_factors", down_factors, GGUFValueType.INT32)
w.add_uint32(c + "codebook_dim", int(qcfg.quantizer.quantizers[0].codebook_dim))
w.add_uint32(c + "codebook_size", int(qcfg.quantizer.codebook_size))
w.add_uint32(c + "semantic_codebook_size", int(qcfg.semantic_quantizer.codebook_size))
w.add_uint32(c + "residual_codebooks", int(qcfg.quantizer.n_codebooks))
for name, module in (("encoder", codec_model.encoder.block[4].block[5]), ("quantizer", qcfg.pre_module)):
    cfg = module.config
    w.add_uint32(c + f"{name}_transformer.layers", len(module.layers))
    w.add_uint32(c + f"{name}_transformer.heads", int(cfg.n_head))
    w.add_uint32(c + f"{name}_transformer.head_dim", int(cfg.head_dim))
    w.add_uint32(c + f"{name}_transformer.window", int(module.window_size))
    w.add_float32(c + f"{name}_transformer.rope_base", float(cfg.rope_base))
    w.add_float32(c + f"{name}_transformer.norm_eps", float(cfg.norm_eps))
assert codec_model.encoder.block[4].block[5].config.n_local_heads == codec_model.encoder.block[4].block[5].config.n_head
assert qcfg.pre_module.config.n_local_heads == qcfg.pre_module.config.n_head and qcfg.post_module.window_size == qcfg.pre_module.window_size

# The text model.
m = "model.language_model."
H, V = int(text["hidden_size"]), int(text["vocab_size"])
heads, kv, hd = int(text["num_attention_heads"]), int(text["num_key_value_heads"]), int(text["head_dim"])
E, I = int(text["num_experts"]), int(text["moe_intermediate_size"])
add("token_embd", (V, H), lambda: f32(m + "embed_tokens.weight"))
add("output", (len(candidates), H), lambda: f32("lm_head.weight")[candidates])
add("output_norm", (H,), lambda: zero_centred(m + "norm.weight"))
for i, kind in enumerate(text["layer_types"]):
    s, o = f"{m}layers.{i}.", f"blk.{i}."
    add(o + "attn_norm", (H,), lambda s=s: zero_centred(s + "input_layernorm.weight"))
    add(o + "post_attn_norm", (H,), lambda s=s: zero_centred(s + "post_attention_layernorm.weight"))
    if kind == "full_attention":
        a_ = s + "self_attn."
        add(o + "attn_q", (heads * hd, H), lambda a_=a_: f32(a_ + "q_proj.weight").reshape(heads, 2, hd, H)[:, 0].reshape(-1, H))
        add(o + "attn_gate", (heads * hd, H), lambda a_=a_: f32(a_ + "q_proj.weight").reshape(heads, 2, hd, H)[:, 1].reshape(-1, H))
        for part, name in (("attn_k", "k_proj"), ("attn_v", "v_proj")):
            add(o + part, (kv * hd, H), lambda a_=a_, name=name: f32(a_ + name + ".weight"))
        add(o + "attn_output", (H, heads * hd), lambda a_=a_: f32(a_ + "o_proj.weight"))
        add(o + "attn_q_norm", (hd,), lambda a_=a_: zero_centred(a_ + "q_norm.weight"))
        add(o + "attn_k_norm", (hd,), lambda a_=a_: zero_centred(a_ + "k_norm.weight"))
    else:
        l_ = s + "linear_attn."
        nk, nv = int(text["linear_num_key_heads"]), int(text["linear_num_value_heads"])
        dk, dv = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
        conv = 2 * nk * dk + nv * dv
        add(o + "ssm_in_qkv", (conv, H), lambda l_=l_: f32(l_ + "in_proj_qkv.weight"))
        add(o + "ssm_in_z", (nv * dv, H), lambda l_=l_: f32(l_ + "in_proj_z.weight"))
        add(o + "ssm_in_b", (nv, H), lambda l_=l_: f32(l_ + "in_proj_b.weight"))
        add(o + "ssm_in_a", (nv, H), lambda l_=l_: f32(l_ + "in_proj_a.weight"))
        add(o + "ssm_conv1d", (conv, int(text["linear_conv_kernel_dim"])), lambda l_=l_: f32(l_ + "conv1d.weight")[:, 0, :])
        add(o + "ssm_dt_bias", (nv,), lambda l_=l_: f32(l_ + "dt_bias"))
        add(o + "ssm_a_log", (nv,), lambda l_=l_: f32(l_ + "A_log"))
        add(o + "ssm_norm", (dv,), lambda l_=l_: f32(l_ + "norm.weight"))
        add(o + "ssm_out", (H, nv * dv), lambda l_=l_: f32(l_ + "out_proj.weight"))
    x = s + "mlp."
    add(o + "ffn_gate_inp", (E, H), lambda x=x: f32(x + "gate.weight"))
    add(o + "ffn_gate_exps", (E, I, H), lambda i=i: experts(i, "gate_proj"))
    add(o + "ffn_up_exps", (E, I, H), lambda i=i: experts(i, "up_proj"))
    add(o + "ffn_down_exps", (E, H, I), lambda i=i: experts(i, "down_proj"))
    S = int(text["shared_expert_intermediate_size"])
    add(o + "ffn_gate_shexp", (S, H), lambda x=x: f32(x + "shared_expert.gate_proj.weight"))
    add(o + "ffn_up_shexp", (S, H), lambda x=x: f32(x + "shared_expert.up_proj.weight"))
    add(o + "ffn_down_shexp", (H, S), lambda x=x: f32(x + "shared_expert.down_proj.weight"))
    add(o + "ffn_gate_inp_shexp", (1, H), lambda x=x: f32(x + "shared_expert_gate.weight"))

# The audio decoder.
D, CB, AV = int(audio["dim"]), int(audio["num_codebooks"]), int(audio["vocab_size"])
ah, akv, ahd, F = int(audio["n_head"]), int(audio["n_local_heads"]), int(audio["head_dim"]), int(audio["intermediate_size"])
add("ad.codebook_embd", ((CB - 1) * AV, D), lambda: f32("audio_decoder.codebook_embeddings.weight")[AV:])
layers = [(f"audio_decoder_mtp_shared.layers.{i}.", f"ad.shared.{i}.", "semantic_embedding") for i in range(int(audio["shared_enrichment_layers"]))]
layers += [(f"audio_decoder_mtp.layers.{i}.", f"ad.layer.{i}.", "codebook_embedding") for i in range(CB - 1)]
for s, o, embedding in layers:
    tl = s + "transformer_layer."
    add(o + "embd", (AV, D), lambda s=s, e=embedding: f32(s + e + ".weight"))
    for norm in ("enorm", "hnorm"):
        add(o + norm, (D,), lambda s=s, norm=norm: zero_centred(s + norm + ".weight"))
    add(o + "eh_proj", (D, 2 * D), lambda s=s: f32(s + "eh_proj.weight"))
    add(o + "final_norm", (D,), lambda s=s: zero_centred(s + "final_layernorm.weight"))
    add(o + "attn_norm", (D,), lambda tl=tl: zero_centred(tl + "attention_norm.weight"))
    add(o + "ffn_norm", (D,), lambda tl=tl: zero_centred(tl + "ffn_norm.weight"))
    split = np.cumsum([ah * ahd, ah * ahd, akv * ahd])
    for index_, part in enumerate(["attn_q", "attn_gate", "attn_k", "attn_v"]):
        rows = (ah if index_ < 2 else akv) * ahd
        add(o + part, (rows, D), lambda tl=tl, index_=index_: np.split(f32(tl + "attention.wqkv.weight"), split)[index_])
    add(o + "attn_q_norm", (ahd,), lambda tl=tl: zero_centred(tl + "attention.q_norm.weight"))
    add(o + "attn_k_norm", (ahd,), lambda tl=tl: zero_centred(tl + "attention.k_norm.weight"))
    add(o + "attn_output", (D, ah * ahd), lambda tl=tl: f32(tl + "attention.wo.weight"))
    add(o + "ffn_gate", (F, D), lambda tl=tl: f32(tl + "feed_forward.w1.weight"))
    add(o + "ffn_up", (F, D), lambda tl=tl: f32(tl + "feed_forward.w3.weight"))
    add(o + "ffn_down", (D, F), lambda tl=tl: f32(tl + "feed_forward.w2.weight"))
for i in range(CB - 1):
    add(f"ad.head.{i}", (AV, D), lambda i=i: f32(f"audio_decoder_mtp.heads.{i}.weight"))


# The codec.
def cw(name):
    """A convolution weight [out, in, k] as numpy [k, out, in]."""
    return np.ascontiguousarray(codec_state[name].transpose(2, 0, 1))


def tw(name):
    """A transposed convolution weight [in, out, k] as numpy [k, out, in]."""
    return np.ascontiguousarray(codec_state[name].transpose(2, 1, 0))


def halves(name, stride):
    """A strided convolution of width 2s, [out, in, 2s], as its two halves [out, s * in], the input channel fastest."""
    weight = codec_state[name]
    out, inp, width = weight.shape
    assert width == 2 * stride
    taps = weight.transpose(0, 2, 1).reshape(out, 2, stride * inp)
    return np.ascontiguousarray(taps[:, 0]), np.ascontiguousarray(taps[:, 1])


def framed(name):
    """A convolution whose width is its stride, [out, in, s], as one matrix [out, s * in], the input channel fastest."""
    weight = codec_state[name]
    return np.ascontiguousarray(weight.transpose(0, 2, 1).reshape(weight.shape[0], -1))


def add_codec(name, values):
    values = np.ascontiguousarray(values, dtype=np.float32)
    add("codec." + name, values.shape, lambda values=values: values)


def snake(src, dst):
    alpha = codec_state[src + ".alpha"].reshape(-1)
    add_codec(dst + ".alpha", alpha)
    # DAC's Snake divides by alpha + 1e-9, which the float32 reciprocal here gives as the official code computes it.
    add_codec(dst + ".inv_alpha", (np.float32(1.0) / (alpha + np.float32(1e-9))).astype(np.float32))


def conv(src, dst):
    add_codec(dst + ".weight", cw(src + ".conv.weight"))
    add_codec(dst + ".bias", codec_state[src + ".conv.bias"])


def residual(src, dst):
    snake(src + ".block.0", dst + ".snake1")
    conv(src + ".block.1", dst + ".conv1")
    snake(src + ".block.2", dst + ".snake2")
    conv(src + ".block.3", dst + ".conv2")


def transformer(src, dst, layers_n):
    for l in range(layers_n):
        s, o = f"{src}.layers.{l}.", f"{dst}.blk.{l}."
        q_, k_, v_ = np.split(codec_state[s + "attention.wqkv.weight"], 3)
        for part, values in (("attn_q", q_), ("attn_k", k_), ("attn_v", v_), ("attn_o", codec_state[s + "attention.wo.weight"]),
                             ("attn_norm", codec_state[s + "attention_norm.weight"]), ("attn_scale", codec_state[s + "attention_layer_scale.gamma"]),
                             ("ffn_norm", codec_state[s + "ffn_norm.weight"]), ("ffn_gate", codec_state[s + "feed_forward.w1.weight"]),
                             ("ffn_up", codec_state[s + "feed_forward.w3.weight"]), ("ffn_down", codec_state[s + "feed_forward.w2.weight"]),
                             ("ffn_scale", codec_state[s + "ffn_layer_scale.gamma"])):
            add_codec(o + part, values)
    add_codec(dst + ".norm", codec_state[src + ".norm.weight"])


def convnext(src, dst):
    add_codec(dst + ".dwconv.weight", codec_state[src + ".dwconv.conv.weight"][:, 0, :].T)
    add_codec(dst + ".dwconv.bias", codec_state[src + ".dwconv.conv.bias"])
    for part, name in (("norm.weight", "norm.weight"), ("norm.bias", "norm.bias"), ("pw1.weight", "pwconv1.weight"), ("pw1.bias", "pwconv1.bias"),
                       ("pw2.weight", "pwconv2.weight"), ("pw2.bias", "pwconv2.bias"), ("gamma", "gamma")):
        add_codec(f"{dst}.{part}", codec_state[f"{src}.{name}"])


conv("encoder.block.0", "enc.in_conv")
for b, stride in enumerate(encoder_rates):
    s, o = f"encoder.block.{b + 1}.block", f"enc.blk.{b}"
    for unit in range(3):
        residual(f"{s}.{unit}", f"{o}.res.{unit}")
    snake(f"{s}.3", f"{o}.snake")
    first, second = halves(f"{s}.4.conv.weight", stride)
    add_codec(f"{o}.down.first", first)
    add_codec(f"{o}.down.second", second)
    add_codec(f"{o}.down.bias", codec_state[f"{s}.4.conv.bias"])
    if f"{s}.5.layers.0.attention.wqkv.weight" in codec_state:
        transformer(f"{s}.5", f"{o}.tf", len(codec_model.encoder.block[b + 1].block[5].layers))
snake(f"encoder.block.{len(encoder_rates) + 1}", "enc.out_snake")
conv(f"encoder.block.{len(encoder_rates) + 2}", "enc.out_conv")
assert [i for i, block in enumerate(codec_model.encoder.block) if hasattr(block, "block") and not isinstance(block.block[5], torch.nn.Identity)] == [4]

for i in range(len(down_factors)):
    add_codec(f"q.down.{i}.conv.weight", framed(f"quantizer.downsample.{i}.0.conv.weight"))
    add_codec(f"q.down.{i}.conv.bias", codec_state[f"quantizer.downsample.{i}.0.conv.bias"])
    convnext(f"quantizer.downsample.{i}.1", f"q.down.{i}.convnext")
    add_codec(f"q.up.{i}.conv.weight", tw(f"quantizer.upsample.{i}.0.conv.weight"))
    add_codec(f"q.up.{i}.conv.bias", codec_state[f"quantizer.upsample.{i}.0.conv.bias"])
    convnext(f"quantizer.upsample.{i}.1", f"q.up.{i}.convnext")
transformer("quantizer.pre_module", "q.pre", len(qcfg.pre_module.layers))
transformer("quantizer.post_module", "q.post", len(qcfg.post_module.layers))
for src, dst, n in (("quantizer.semantic_quantizer", "q.semantic", 1), ("quantizer.quantizer", "q.residual", int(qcfg.quantizer.n_codebooks))):
    for j in range(n):
        s = f"{src}.quantizers.{j}."
        o = f"{dst}.{j}." if dst == "q.residual" else f"{dst}."
        add_codec(o + "in_proj.weight", codec_state[s + "in_proj.weight"][:, :, 0])
        add_codec(o + "in_proj.bias", codec_state[s + "in_proj.bias"])
        add_codec(o + "out_proj.weight", codec_state[s + "out_proj.weight"][:, :, 0])
        add_codec(o + "out_proj.bias", codec_state[s + "out_proj.bias"])
        add_codec(o + "codebook", codec_state[s + "codebook.weight"])

conv("decoder.model.0", "dec.in_conv")
for b, stride in enumerate(decoder_rates):
    s, o = f"decoder.model.{b + 1}.block", f"dec.blk.{b}"
    snake(f"{s}.0", f"{o}.snake")
    add_codec(f"{o}.up.weight", tw(f"{s}.1.conv.weight"))
    add_codec(f"{o}.up.bias", codec_state[f"{s}.1.conv.bias"])
    for unit in range(3):
        residual(f"{s}.{unit + 2}", f"{o}.res.{unit}")
snake(f"decoder.model.{len(decoder_rates) + 1}", "dec.out_snake")
conv(f"decoder.model.{len(decoder_rates) + 2}", "dec.out_conv")
names = [name for name, _, _ in plan]
assert len(names) == len(set(names)), "a tensor is written twice"
assert codec_state.taken == set(codec_state), f"the codec's tensors {sorted(set(codec_state) - codec_state.taken)[:5]} are not written"

# Every tensor's info first, so that the values are written a tensor at a time.
for name, shape, _ in plan:
    w.add_tensor_info(name, shape, np.dtype(np.float32), int(np.prod(shape)) * 4)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
for n, (name, shape, make) in enumerate(plan):
    values = np.ascontiguousarray(make(), dtype=np.float32)
    assert values.shape == shape, f"{name} is {values.shape}, not {shape}"
    w.write_tensor_data(values)
    if name.endswith("ffn_down_exps"):
        print(f"{n + 1}/{len(plan)} {name}", flush=True)
w.close()
assert read == set(index), f"the checkpoint's tensors {sorted(set(index) - read)[:5]} are not written"
print(path)
