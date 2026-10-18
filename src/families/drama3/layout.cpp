#include "layout.h"

#include <algorithm>
#include <climits>
#include <string>
#include <vector>

#include "error.h"
#include "qwen2-tokenizer.h"

namespace drama3 {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type. A matrix or embedding of the text model or the audio
 * decoder, which ggml_mul_mat(), ggml_mul_mat_id() and ggml_get_rows() alone read, takes the file's type. The codec's
 * convolutions and matrices take F16, or F32 in an F32 file, as Qwen3-TTS's codec does. The norms, the biases, the
 * codec's codebooks and the weights a quantized type would bend where the model is sensitive take F32: the router,
 * whose scores choose the experts, the shared expert's gate, and the Gated DeltaNet's convolution and the projections
 * that give its decay and its write strength.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, "0.8.8"},
                                           {GGML_TYPE_F16, "0.8.8"},
                                           {GGML_TYPE_Q8_0, "0.8.8"},
                                           {GGML_TYPE_Q6_K, "0.8.8"},
                                           {GGML_TYPE_Q5_K, "0.8.8"},
                                           {GGML_TYPE_Q4_K, "0.8.8"}});
const Storage kCodec = half_storage();
const Storage kF32 = float32_storage();

/**
 * The widths the official codec fixes in its code rather than in its configuration: 7 for the first and last
 * convolutions of the encoder and the decoder, a residual unit's first and a ConvNeXt block's depthwise one, 3 for the
 * encoder's last, 1 for a residual unit's second; and the fourfold widening of a ConvNeXt block and the threefold one
 * of the codec's feed-forward layers.
 */
constexpr int64_t kConvWidth = 7;
constexpr int64_t kEncoderOutWidth = 3;
constexpr int64_t kConvNextWidening = 4;
constexpr int64_t kCodecFfnWidening = 3;

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** An [i32] key of the strides of the codec's stages: each at least 1, and no more stages than the file holds tensors. */
std::vector<int32_t> strides(const ModelFile & m, const std::string & key) {
    const std::vector<int32_t> s = m.i32_array(key);
    require(!s.empty() && (int64_t) s.size() <= m.tensor_count() && std::all_of(s.begin(), s.end(), [](int32_t r) { return r >= 1; }), m,
            key + " is empty, holds a stride below 1 or names more stages than the file holds tensors");
    return s;
}

/** The codec's transformer of `prefix`, `layers` layers of width `dim` and its final norm. */
void add_codec_transformer(std::vector<TensorSpec> & t, const std::string & prefix, int layers, int64_t dim) {
    const int64_t ffn = kCodecFfnWidening * dim;
    add_numbered(t, prefix + ".blk.", layers,
                 {{"attn_norm", {dim}, kF32},        {"attn_q", {dim, dim}, kCodec},       {"attn_k", {dim, dim}, kCodec},
                  {"attn_v", {dim, dim}, kCodec},    {"attn_o", {dim, dim}, kCodec},       {"attn_scale", {dim}, kF32},
                  {"ffn_norm", {dim}, kF32},         {"ffn_gate", {dim, ffn}, kCodec},     {"ffn_up", {dim, ffn}, kCodec},
                  {"ffn_down", {ffn, dim}, kCodec},  {"ffn_scale", {dim}, kF32}});
    t.push_back({prefix + ".norm", {dim}, kF32});
}

/** A residual unit of `channels`: Snake, a convolution of width 7, Snake and one of width 1. */
void add_residual_units(std::vector<TensorSpec> & t, const std::string & prefix, int64_t channels) {
    add_numbered(t, prefix + ".res.", codec_residual_units,
                 {{"snake1.alpha", {channels}, kF32},                      {"snake1.inv_alpha", {channels}, kF32},
                  {"conv1.weight", {channels, channels, kConvWidth}, kCodec}, {"conv1.bias", {channels}, kF32},
                  {"snake2.alpha", {channels}, kF32},                      {"snake2.inv_alpha", {channels}, kF32},
                  {"conv2.weight", {channels, channels, 1}, kCodec},       {"conv2.bias", {channels}, kF32}});
}

/** A ConvNeXt block of `channels`. */
void add_convnext(std::vector<TensorSpec> & t, const std::string & prefix, int64_t channels) {
    const int64_t wide = kConvNextWidening * channels;
    add_block(t, prefix + ".", {{"dwconv.weight", {channels, kConvWidth}, kF32}, {"dwconv.bias", {channels}, kF32},
                                {"norm.weight", {channels}, kF32},              {"norm.bias", {channels}, kF32},
                                {"pw1.weight", {channels, wide}, kCodec},       {"pw1.bias", {wide}, kF32},
                                {"pw2.weight", {wide, channels}, kCodec},       {"pw2.bias", {channels}, kF32},
                                {"gamma", {channels}, kF32}});
}

/** The codec's tensors, from the keys of drama3.codec. */
void add_codec(std::vector<TensorSpec> & t, const ModelFile & m) {
    const std::string c = "drama3.codec.";
    const int64_t enc = m.size(c + "encoder_dim"), latent = m.size(c + "latent_dim"), dec = m.size(c + "decoder_dim");
    const std::vector<int32_t> enc_rates = strides(m, c + "encoder_rates"), dec_rates = strides(m, c + "decoder_rates");
    const std::vector<int32_t> factors = strides(m, c + "downsample_factors");
    int64_t hop = 1;
    for (const std::vector<int32_t> * stages : {&enc_rates, &factors}) {
        for (int32_t r : *stages) hop = std::min<int64_t>(hop * r, (int64_t) INT_MAX + 1);
    }
    require(hop == m.size(c + "hop_length"), m, "drama3.codec.encoder_rates and downsample_factors do not multiply to drama3.codec.hop_length");
    int64_t up = 1;
    for (int32_t r : dec_rates) up = std::min<int64_t>(up * r, (int64_t) INT_MAX + 1);
    for (int32_t f : factors) up = std::min<int64_t>(up * f, (int64_t) INT_MAX + 1);
    require(up == hop, m, "drama3.codec.decoder_rates and downsample_factors do not multiply to drama3.codec.hop_length");
    require(enc << enc_rates.size() == latent, m, "drama3.codec.latent_dim is not encoder_dim doubled by each encoder block");
    require(dec % (int64_t(1) << dec_rates.size()) == 0, m, "drama3.codec.decoder_dim cannot be halved by each decoder block");
    const int codebook_dim = m.size(c + "codebook_dim"), codebook = m.size(c + "codebook_size");
    const int semantic = m.size(c + "semantic_codebook_size"), residual = m.count(c + "residual_codebooks");
    m.str(c + "sha256");
    int encoder_layers = 0;
    for (const char * name : {"encoder_transformer", "quantizer_transformer"}) {
        const std::string k = c + name + ".";
        const int layers = m.count(k + "layers"), heads = m.size(k + "heads"), head_dim = m.size(k + "head_dim");
        require((int64_t) heads * head_dim == latent && head_dim % 2 == 0, m,
                k + "heads of " + k + "head_dim do not span the codec's width, or the heads are of an odd width, which RoPE turns in pairs");
        m.size(k + "window");
        m.f32(k + "rope_base");
        m.f32(k + "norm_eps");
        if (std::string(name) == "encoder_transformer") encoder_layers = layers;
    }

    add_block(t, "codec.enc.in_conv.", {{"weight", {1, enc, kConvWidth}, kCodec}, {"bias", {enc}, kF32}});
    for (size_t b = 0; b < enc_rates.size(); b++) {
        const int64_t in = enc << b, out = 2 * in, s = enc_rates[b];
        const std::string o = "codec.enc.blk." + std::to_string(b);
        add_residual_units(t, o, in);
        add_block(t, o + ".", {{"snake.alpha", {in}, kF32},      {"snake.inv_alpha", {in}, kF32},
                               {"down.first", {s * in, out}, kCodec}, {"down.second", {s * in, out}, kCodec},
                               {"down.bias", {out}, kF32}});
        // The official configuration gives the last encoder block alone a transformer.
        if (b + 1 == enc_rates.size()) add_codec_transformer(t, o + ".tf", encoder_layers, out);
    }
    add_block(t, "codec.enc.", {{"out_snake.alpha", {latent}, kF32},
                                {"out_snake.inv_alpha", {latent}, kF32},
                                {"out_conv.weight", {latent, latent, kEncoderOutWidth}, kCodec},
                                {"out_conv.bias", {latent}, kF32}});
    for (size_t i = 0; i < factors.size(); i++) {
        const std::string o = "codec.q.";
        add_block(t, o + "down." + std::to_string(i) + ".conv.",
                  {{"weight", {factors[i] * latent, latent}, kCodec}, {"bias", {latent}, kF32}});
        add_convnext(t, o + "down." + std::to_string(i) + ".convnext", latent);
        // The upsampling takes the downsampling's factors in reverse order.
        add_block(t, o + "up." + std::to_string(i) + ".conv.",
                  {{"weight", {latent, latent, factors[factors.size() - 1 - i]}, kCodec}, {"bias", {latent}, kF32}});
        add_convnext(t, o + "up." + std::to_string(i) + ".convnext", latent);
    }
    const int quantizer_layers = m.count(c + "quantizer_transformer.layers");
    add_codec_transformer(t, "codec.q.pre", quantizer_layers, latent);
    add_codec_transformer(t, "codec.q.post", quantizer_layers, latent);
    const auto add_quantizer = [&](const std::string & prefix, int64_t entries) {
        add_block(t, prefix, {{"in_proj.weight", {latent, codebook_dim}, kF32}, {"in_proj.bias", {codebook_dim}, kF32},
                              {"out_proj.weight", {codebook_dim, latent}, kF32}, {"out_proj.bias", {latent}, kF32},
                              {"codebook", {codebook_dim, entries}, kF32}});
    };
    add_quantizer("codec.q.semantic.", semantic);
    for (int j = 0; j < residual; j++) add_quantizer("codec.q.residual." + std::to_string(j) + ".", codebook);
    add_block(t, "codec.dec.in_conv.", {{"weight", {latent, dec, kConvWidth}, kCodec}, {"bias", {dec}, kF32}});
    for (size_t b = 0; b < dec_rates.size(); b++) {
        const int64_t in = dec >> b, out = in / 2, s = dec_rates[b];
        const std::string o = "codec.dec.blk." + std::to_string(b);
        add_block(t, o + ".", {{"snake.alpha", {in}, kF32}, {"snake.inv_alpha", {in}, kF32},
                               {"up.weight", {in, out, 2 * s}, kCodec}, {"up.bias", {out}, kF32}});
        add_residual_units(t, o, out);
    }
    const int64_t last = dec >> dec_rates.size();
    add_block(t, "codec.dec.", {{"out_snake.alpha", {last}, kF32},
                                {"out_snake.inv_alpha", {last}, kF32},
                                {"out_conv.weight", {last, 1, kConvWidth}, kCodec},
                                {"out_conv.bias", {1}, kF32}});
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "checked");
    const std::string p = "drama3.", x = p + "text.", a = p + "audio_decoder.";

    const int64_t h = m.size(x + "hidden_size"), vocab = m.size(x + "vocab_size");
    const int layers = m.count(x + "num_hidden_layers");
    const std::vector<std::string> types = m.str_array(x + "layer_types");
    require((int) types.size() == layers, m, "drama3.text.layer_types does not name a kind for each of the text model's layers");
    const int64_t heads = m.size(x + "num_attention_heads"), kv = m.size(x + "num_key_value_heads"), hd = m.size(x + "head_dim");
    require(heads % kv == 0, m, "the text model's attention heads do not share its key/value heads evenly");
    const int rotary = m.size(x + "rotary_dim");
    require(rotary <= hd && rotary % 2 == 0, m, "drama3.text.rotary_dim is odd or wider than a head");
    const int64_t nk = m.size(x + "linear_num_key_heads"), nv = m.size(x + "linear_num_value_heads");
    const int64_t dk = m.size(x + "linear_key_head_dim"), dv = m.size(x + "linear_value_head_dim");
    require(nv % nk == 0, m, "the Gated DeltaNet's value heads do not share its key heads evenly");
    require(dk == dv, m, "the Gated DeltaNet's key and value heads differ in width, which ggml_gated_delta_net() does not take");
    const int64_t conv_width = m.size(x + "linear_conv_kernel_dim"), conv = 2 * nk * dk + nv * dv;
    const int64_t experts = m.size(x + "num_experts"), used = m.size(x + "num_experts_per_tok");
    require(used <= experts, m, "drama3.text.num_experts_per_tok is more than the experts");
    const int64_t ffn = m.size(x + "moe_intermediate_size"), shared = m.size(x + "shared_expert_intermediate_size");
    m.size(x + "max_position_embeddings");
    m.boolean(x + "norm_topk_prob");
    m.f32(x + "rms_norm_eps");
    m.f32(x + "rope_theta");

    const int64_t d = m.size(a + "dim"), ah = m.size(a + "n_head"), akv = m.size(a + "n_local_heads"), ahd = m.size(a + "head_dim");
    require(ah % akv == 0, m, "the audio decoder's attention heads do not share its key/value heads evenly");
    const int64_t affn = m.size(a + "intermediate_size"), codes = m.size(a + "vocab_size");
    const int codebooks = m.count(a + "num_codebooks"), shared_layers = m.count(a + "shared_enrichment_layers");
    require(codebooks >= 2, m, "drama3.audio_decoder.num_codebooks is 1, where the audio decoder draws the codes after a frame's first");
    require(m.u32(a + "vq_backbone_codebooks") < (uint32_t) codebooks, m, "drama3.audio_decoder.vq_backbone_codebooks is not below num_codebooks");
    const int arotary = m.size(a + "rotary_dim");
    require(arotary <= ahd && arotary % 2 == 0, m, "drama3.audio_decoder.rotary_dim is odd or wider than a head");
    require(d == h, m, "the audio decoder's width is not the text model's, whose hidden state it takes");
    m.f32(a + "norm_eps");
    m.f32(a + "rope_base");

    const uint32_t eos = m.u32(p + "eos_token_id"), first = m.u32(p + "semantic_start_token_id"), end = m.u32(p + "semantic_end_token_id");
    m.u32(p + "pad_token_id");
    require(first <= end && end < (uint64_t) vocab && (int64_t) end - first + 1 == codes && (eos < first || eos > end), m,
            "the semantic ids are not drama3.audio_decoder.vocab_size ids of the vocabulary apart from the end id");
    const std::vector<int32_t> ids = m.i32_array(p + "output.token_ids");
    std::vector<int32_t> expected = {(int32_t) eos};
    for (uint32_t i = first; i <= end; i++) expected.push_back((int32_t) i);
    std::sort(expected.begin(), expected.end());
    require(ids == expected, m, "drama3.output.token_ids is not the end id and the semantic ids in order");

    m.f32(p + "sampling.temperature");
    m.f32(p + "sampling.top_p");
    m.u32(p + "sampling.top_k");
    m.size(p + "repeat.window");
    m.f32(p + "repeat.temperature");
    m.u32(p + "repeat.top_k");
    m.f32(p + "repeat.top_p");
    const int default_frames = m.size(p + "generation.default_frames"), max_frames = m.size(p + "generation.max_frames");
    require(default_frames <= max_frames, m, "drama3.generation.default_frames is more than max_frames");
    m.size(p + "generation.max_prompt_rows");
    m.size(p + "generation.chunk_bytes");
    m.size(p + "generation.chunk_speakers");
    m.u32(p + "generation.recent_turns");
    m.size(p + "generation.speakers");
    require(m.str(p + "prompt.template") == "reference-bracket-v1", m, "drama3.prompt.template names a prompt this release does not write");
    for (const char * key : {"system", "instruction", "reference_instruction", "reference_open", "reference_text", "reference_speech", "end",
                             "user", "assistant"}) {
        m.str(p + "prompt." + key);
    }
    m.str_array(p + "prompt.reserved");
    const std::vector<std::string> tokens = m.str_array(p + "tokenizer.tokens");
    require(tokens.size() <= (size_t) vocab, m, "drama3.tokenizer.tokens has more tokens than the text model's vocabulary");
    m.str_array(p + "tokenizer.merges");
    const std::vector<int32_t> added = m.i32_array(p + "tokenizer.added_ids");
    require(std::all_of(added.begin(), added.end(), [&](int32_t id) { return id >= 0 && (size_t) id < tokens.size() && !tokens[(size_t) id].empty(); }),
            m, "drama3.tokenizer.added_ids holds an id that is not one of the tokens");
    Qwen2Pattern pattern;
    require(qwen2_pattern(m.str(p + "tokenizer.pre_tokenizer"), &pattern), m, "drama3.tokenizer.pre_tokenizer is a pattern this release does not split by");

    std::vector<TensorSpec> t = {{"token_embd", {h, vocab}, kMatrix},
                                 {"output", {h, (int64_t) ids.size()}, kMatrix},
                                 {"output_norm", {h}, kF32}};
    for (int i = 0; i < layers; i++) {
        const std::string o = "blk." + std::to_string(i) + ".";
        add_block(t, o, {{"attn_norm", {h}, kF32}, {"post_attn_norm", {h}, kF32}});
        if (types[i] == "full_attention") {
            add_block(t, o, {{"attn_q", {h, heads * hd}, kMatrix},   {"attn_gate", {h, heads * hd}, kMatrix},
                             {"attn_k", {h, kv * hd}, kMatrix},      {"attn_v", {h, kv * hd}, kMatrix},
                             {"attn_output", {heads * hd, h}, kMatrix}, {"attn_q_norm", {hd}, kF32},
                             {"attn_k_norm", {hd}, kF32}});
        } else {
            require(types[i] == "linear_attention", m, "drama3.text.layer_types names " + types[i] + ", which is neither full_attention nor linear_attention");
            add_block(t, o, {{"ssm_in_qkv", {h, conv}, kMatrix},     {"ssm_in_z", {h, nv * dv}, kMatrix},
                             {"ssm_in_b", {h, nv}, kF32},            {"ssm_in_a", {h, nv}, kF32},
                             {"ssm_conv1d", {conv_width, conv}, kF32}, {"ssm_dt_bias", {nv}, kF32},
                             {"ssm_a_log", {nv}, kF32},              {"ssm_norm", {dv}, kF32},
                             {"ssm_out", {nv * dv, h}, kMatrix}});
        }
        add_block(t, o, {{"ffn_gate_inp", {h, experts}, kF32},      {"ffn_gate_exps", {h, ffn, experts}, kMatrix},
                         {"ffn_up_exps", {h, ffn, experts}, kMatrix}, {"ffn_down_exps", {ffn, h, experts}, kMatrix},
                         {"ffn_gate_shexp", {h, shared}, kMatrix},  {"ffn_up_shexp", {h, shared}, kMatrix},
                         {"ffn_down_shexp", {shared, h}, kMatrix},  {"ffn_gate_inp_shexp", {h, 1}, kF32}});
    }

    t.push_back({"ad.codebook_embd", {d, (codebooks - 1) * codes}, kMatrix});
    const std::initializer_list<TensorSpec> layer = {
        {"embd", {d, codes}, kMatrix},       {"enorm", {d}, kF32},                  {"hnorm", {d}, kF32},
        {"eh_proj", {2 * d, d}, kMatrix},    {"final_norm", {d}, kF32},             {"attn_norm", {d}, kF32},
        {"ffn_norm", {d}, kF32},             {"attn_q", {d, ah * ahd}, kMatrix},    {"attn_gate", {d, ah * ahd}, kMatrix},
        {"attn_k", {d, akv * ahd}, kMatrix}, {"attn_v", {d, akv * ahd}, kMatrix},   {"attn_q_norm", {ahd}, kF32},
        {"attn_k_norm", {ahd}, kF32},        {"attn_output", {ah * ahd, d}, kMatrix}, {"ffn_gate", {d, affn}, kMatrix},
        {"ffn_up", {d, affn}, kMatrix},      {"ffn_down", {affn, d}, kMatrix}};
    add_numbered(t, "ad.shared.", shared_layers, layer);
    add_numbered(t, "ad.layer.", codebooks - 1, layer);
    add_numbered(t, "ad.head.", codebooks - 1, {{"", {d, codes}, kMatrix}});

    add_codec(t, m);
    return t;
}

}  // namespace

const Layout model_layout = {"drama3", 1, "convert it again with reference/drama3/convert.py, or download it again from its Hugging Face repository",
                             tensors};

}  // namespace drama3
