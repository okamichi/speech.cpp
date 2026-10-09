#include "layout.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "codec.h"
#include "error.h"

namespace irodori {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type: a matrix or embedding of the model,
 * which ggml_mul_mat() and ggml_get_rows() alone read, in the file's type, which 0.7's reader of layout 1 took in Q8_0,
 * F16 and F32; and the codec, which encodes the voices' references and decodes the speech, the norms, the biases and
 * the rest in F32 in every file, as the released files hold them.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, kFirstLayoutRelease},
                                           {GGML_TYPE_F16, kFirstLayoutRelease},
                                           {GGML_TYPE_Q8_0, kFirstLayoutRelease},
                                           {GGML_TYPE_Q6_K, "0.8.0"},
                                           {GGML_TYPE_Q5_K, "0.8.0"},
                                           {GGML_TYPE_Q4_K, "0.8.0"}});
const Storage kF32 = float32_storage();

/**
 * The widths DACVAE fixes in its code rather than in its configuration: 7 for the first and the last convolution and
 * a residual unit's first, 3 for the encoder's last, and 1 for a residual unit's second and the decoder's input
 * projection.
 */
constexpr int64_t kConvWidth = 7;
constexpr int64_t kEncoderOutWidth = 3;

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** Checks that `width_key` splits into `heads_key` heads of an even width, which RoPE turns in pairs. */
void require_heads(const ModelFile & m, const std::string & width_key, int width, const std::string & heads_key, int heads) {
    require(width % heads == 0 && width / heads % 2 == 0, m,
            width_key + " does not split into " + heads_key + " heads of an even width, which RoPE turns in pairs");
}

/**
 * The strides of the codec's encoder or decoder blocks, whose convolutions of width 2s are padded by s / 2 on either
 * side: each even, and together the codec's hop length.
 */
std::vector<int32_t> strides(const ModelFile & m, const std::string & key, uint32_t hop) {
    const std::vector<int32_t> s = m.i32_array(key);
    int64_t product = 1;
    for (int32_t r : s) {
        require(r >= 2 && r % 2 == 0, m, key + " holds a stride that is odd or below 2");
        product = std::min<int64_t>(product * r, (int64_t) hop + 1);
    }
    require(product == hop, m, key + " does not multiply to irodori-tts.codec.hop_length");
    return s;
}

/** The residual units of a codec block of `c` channels, "<prefix>res.<j>." each. */
void add_residual_units(std::vector<TensorSpec> & t, const std::string & prefix, int64_t c) {
    add_numbered(t, prefix + "res.", kCodecResidualUnits,
                 {{"snake1.alpha", {c}, kF32}, {"snake1.inv_alpha", {c}, kF32}, {"conv1.weight", {c, c, kConvWidth}, kF32},
                  {"conv1.bias", {c}, kF32},   {"snake2.alpha", {c}, kF32},     {"snake2.inv_alpha", {c}, kF32},
                  {"conv2.weight", {c, c, 1}, kF32}, {"conv2.bias", {c}, kF32}});
}

/**
 * A condition made of ModernBERT's `hidden` channels by a PretrainedConditionProjector, "<prefix>proj.*", whose residual
 * MLP is `width` wide, to `out` channels, and its norm, "<prefix>norm".
 */
void add_projector(std::vector<TensorSpec> & t, const std::string & prefix, int64_t hidden, int64_t width, int64_t out) {
    add_block(t, prefix,
              {{"proj.weight", {hidden, out}, kMatrix},
               {"proj.bias", {out}, kF32},
               {"proj.res_norm", {hidden}, kF32},
               {"proj.res_up.weight", {hidden, width}, kMatrix},
               {"proj.res_up.bias", {width}, kF32},
               {"proj.res_down.weight", {width, out}, kMatrix},
               {"proj.res_down.bias", {out}, kF32},
               {"norm", {out}, kF32}});
}

/** Reads every key of the model layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> model_tensors(const ModelFile & m) {
    check_model_keys(m, "synthesis", "checked");
    const std::string p = "irodori-tts.";
    const bool meanflow = m.one_of(p + "flow", {"meanflow", "rf_velocity"}) == "meanflow";
    const int latent = m.size(p + "latent_dim");
    m.f32(p + "norm_eps");
    m.f32(p + "rope_theta");
    const int text_hidden = m.size(p + "text.hidden_size"), text_heads = m.size(p + "text.num_heads"), text_dim = m.size(p + "text.dim");
    const bool gemma = m.one_of(p + "text.encoder", {"modernbert", "t5gemma2"}) == "t5gemma2";
    const int text_head = gemma ? m.size(p + "text.head_dim") : text_hidden / text_heads;
    const int text_kv_heads = gemma ? m.size(p + "text.num_kv_heads") : text_heads;
    if (gemma) {
        require(text_heads % text_kv_heads == 0 && text_head % 2 == 0, m, "text heads must divide into KV groups with even head width");
        require((int64_t) text_heads * text_head <= INT_MAX, m, "text attention width exceeds the supported range");
        for (const char * key : {"attention_scale", "embedding_scale", "rope_scale_global"}) {
            const float value = m.f32(p + "text." + key);
            require(std::isfinite(value) && value > 0, m, p + "text." + key + " must be finite and positive");
        }
        m.u32(p + "text.eoi_id");
    } else {
        require_heads(m, p + "text.hidden_size", text_hidden, p + "text.num_heads", text_heads);
    }
    for (const char * key : {"window", "max_tokens"}) m.u32(p + "text." + key);
    for (const char * key : {"norm_eps", "rope_theta_global", "rope_theta_local"}) m.f32(p + "text." + key);
    const int text_layers = m.count(p + "text.num_layers");
    require(m.i32_array(p + "text.layer_global").size() == (size_t) text_layers, m, "irodori-tts.text.layer_global does not have one entry per layer");
    const int speaker_dim = m.size(p + "speaker.dim"), speaker_heads = m.size(p + "speaker.num_heads");
    require_heads(m, p + "speaker.dim", speaker_dim, p + "speaker.num_heads", speaker_heads);
    const int patch = m.size(p + "speaker.patch_size"), speaker_layers = m.count(p + "speaker.num_layers");
    const int duration_layers = m.count(p + "duration.num_layers");
    const int dit_dim = m.size(p + "dit.dim"), dit_heads = m.size(p + "dit.num_heads");
    require_heads(m, p + "dit.dim", dit_dim, p + "dit.num_heads", dit_heads);
    const int timestep = m.size(p + "dit.timestep_dim"), dit_layers = m.count(p + "dit.num_layers");
    const bool null_speaker = m.boolean(p + "duration.null_speaker");
    const int caption_dim = m.size(p + "caption.dim");
    const bool caption = m.boolean(p + "caption_condition");
    if (caption) m.size(p + "caption.max_tokens");
    m.size(p + "sampler.default_steps");
    if (!meanflow) {
        for (const char * key : {"cfg_text", "cfg_speaker", "cfg_min_t", "cfg_max_t", "speaker_kv_min_t"}) m.f32(p + "sampler." + key);
        if (caption) m.f32(p + "sampler.cfg_caption");
    }
    const float min_seconds = m.f32(p + "length.min_seconds"), max_seconds = m.f32(p + "length.max_seconds");
    const float min_speed = m.f32(p + "length.min_speed"), max_speed = m.f32(p + "length.max_speed");
    require(min_seconds > 0 && min_seconds <= max_seconds && std::isfinite(max_seconds) && min_speed > 0 && min_speed <= max_speed &&
                std::isfinite(max_speed),
            m, "the bounds of irodori-tts.length are not finite, above 0 and in order");
    m.f32(p + "reference.max_seconds");
    m.f32(p + "reference.lufs");
    m.u32(p + "tail.window");
    m.f32(p + "tail.std_threshold");
    m.f32(p + "tail.mean_threshold");
    const size_t pieces = m.str_array(p + "tokenizer.tokens").size();
    const bool bpe = m.one_of(p + "tokenizer.type", {"Unigram", "BPE"}) == "BPE";
    if (bpe) m.str_array(p + "tokenizer.merges");
    else require(m.f64_array(p + "tokenizer.scores").size() == pieces, m, "irodori-tts.tokenizer.scores does not have one score per piece");
    if (gemma) require(m.u32(p + "text.eoi_id") < pieces, m, "text.eoi_id is outside the vocabulary");
    for (int32_t id : m.i32_array(p + "tokenizer.added_ids")) require(id >= 0 && (size_t) id < pieces, m, "an added token is not a piece");
    require(m.u32(p + "tokenizer.bos_id") < pieces && m.u32(p + "tokenizer.unknown_id") < pieces, m,
            "the tokenizer's <s> or unknown piece is not a piece");
    const uint32_t hop = m.u32(p + "codec.hop_length");
    const std::vector<int32_t> encoder_rates = strides(m, p + "codec.encoder_rates", hop);
    const std::vector<int32_t> decoder_rates = strides(m, p + "codec.decoder_rates", hop);
    require(m.str(p + "codec.sha256").size() == 64, m, "irodori-tts.codec.sha256 is not a SHA-256 in hexadecimal");

    // The widths of the feed-forward layers of ModernBERT, its projectors, the speaker encoder and the DiT, of the
    // duration predictor, of the rank of the DiT's AdaLN, and of the codec's first convolution, its latent and its
    // decoder have no key: each is the width of one tensor, which the others are checked against.
    const int64_t text_ffn = m.width("text.blk.0.ffn_act", 1), projector = m.width("text.proj.res_up.weight", 1);
    const int64_t speaker_ffn = m.width("speaker.blk.0.ffn_gate", 1), duration = m.width("duration.in_proj.weight", 1);
    const int64_t dit_ffn = m.width("dit.blk.0.ffn_gate", 1), rank = m.width("dit.blk.0.attn_ada.shift.down", 1);
    const int64_t first = m.width("codec.enc.conv_in.weight", 1), codec_latent = m.width("codec.enc.conv_out.weight", 1);
    const int64_t decoder = m.width("codec.dec.conv_in.weight", 1);
    std::vector<int64_t> encoder_channels = {first}, decoder_channels = {decoder};
    for (int32_t r : encoder_rates) {
        require(encoder_channels.back() <= INT_MAX / r, m,
                "codec.enc.conv_in.weight has " + std::to_string(first) + " outputs, which the encoder's blocks widen past any tensor's");
        encoder_channels.push_back(2 * encoder_channels.back());
    }
    for (size_t b = 0; b < decoder_rates.size(); b++) {
        require(decoder_channels.back() % 2 == 0, m,
                "codec.dec.conv_in.weight has " + std::to_string(decoder) + " outputs, which the decoder's blocks cannot each halve");
        decoder_channels.push_back(decoder_channels.back() / 2);
    }

    const int64_t th = text_hidden, td = text_dim, cd = caption_dim, sd = speaker_dim, xd = dit_dim, ld = latent;
    std::vector<TensorSpec> t = {{"text.embd", {th, (int64_t) pieces}, kMatrix},
                                 {"text.final_norm", {th}, kF32},
                                 {"speaker.in_proj.weight", {ld * patch, sd}, kMatrix},
                                 {"speaker.in_proj.bias", {sd}, kF32},
                                 {"speaker.norm", {sd}, kF32},
                                 {"duration.in_proj.weight", {td, duration}, kMatrix},
                                 {"duration.in_proj.bias", {duration}, kF32},
                                 {"duration.out_norm", {duration}, kF32},
                                 {"duration.out_proj.weight", {duration, 1}, kF32},
                                 {"duration.out_proj.bias", {1}, kF32},
                                 {"duration.null_caption", {cd}, kF32},
                                 {"dit.in_proj.weight", {ld, xd}, kF32},
                                 {"dit.in_proj.bias", {xd}, kF32},
                                 {"dit.out_norm", {xd}, kF32},
                                 {"dit.out_proj.weight", {xd, ld}, kMatrix},
                                 {"dit.out_proj.bias", {ld}, kF32}};
    add_projector(t, "text.", th, projector, td);
    if (caption) add_projector(t, "caption.", th, m.width("caption.proj.res_up.weight", 1), cd);
    if (gemma) {
        t.push_back({"text.eoi_embd", {th}, kF32});
        const int64_t q = (int64_t) text_head * text_heads, kv = (int64_t) text_head * text_kv_heads;
        add_numbered(t, "text.blk.", text_layers,
                     {{"attn_q", {th, q}, kMatrix}, {"attn_k", {th, kv}, kMatrix}, {"attn_v", {th, kv}, kMatrix},
                      {"attn_out", {q, th}, kMatrix}, {"q_norm", {text_head}, kF32}, {"k_norm", {text_head}, kF32},
                      {"attn_norm", {th}, kF32}, {"attn_post_norm", {th}, kF32},
                      {"ffn_norm", {th}, kF32}, {"ffn_post_norm", {th}, kF32},
                      {"ffn_act", {th, text_ffn}, kMatrix}, {"ffn_gate", {th, text_ffn}, kMatrix},
                      {"ffn_down", {text_ffn, th}, kMatrix}});
    } else {
        t.push_back({"text.embd_norm", {th}, kF32});
        add_numbered(t, "text.blk.", text_layers,
                     {{"attn_q", {th, th}, kMatrix},         {"attn_k", {th, th}, kMatrix},           {"attn_v", {th, th}, kMatrix},
                      {"attn_out", {th, th}, kMatrix},       {"ffn_norm", {th}, kF32},                {"ffn_act", {th, text_ffn}, kMatrix},
                      {"ffn_gate", {th, text_ffn}, kMatrix}, {"ffn_down", {text_ffn, th}, kMatrix}});
        // ModernBERT normalizes the attention's input of every layer but the first.
        for (int l = 1; l < text_layers; l++) t.push_back({"text.blk." + std::to_string(l) + ".attn_norm", {th}, kF32});
    }
    const int64_t speaker_head = sd / speaker_heads;
    add_numbered(t, "speaker.blk.", speaker_layers,
                 {{"attn_norm", {sd}, kF32},
                  {"attn_q", {sd, sd}, kMatrix},
                  {"attn_k", {sd, sd}, kMatrix},
                  {"attn_v", {sd, sd}, kMatrix},
                  {"attn_o", {sd, sd}, kMatrix},
                  {"attn_gate", {sd, sd}, kMatrix},
                  {"q_norm", {speaker_head, speaker_heads}, kF32},
                  {"k_norm", {speaker_head, speaker_heads}, kF32},
                  {"ffn_norm", {sd}, kF32},
                  {"ffn_gate", {sd, speaker_ffn}, kMatrix},
                  {"ffn_up", {sd, speaker_ffn}, kMatrix},
                  {"ffn_down", {speaker_ffn, sd}, kMatrix}});
    if (null_speaker) t.push_back({"duration.null_speaker", {sd}, kF32});
    // A duration block's feed-forward layer is as wide as the block.
    add_numbered(t, "duration.blk.", duration_layers,
                 {{"norm", {duration}, kF32},
                  {"mod.weight", {sd, 3 * duration}, kMatrix},
                  {"mod.bias", {3 * duration}, kF32},
                  {"caption_mod.weight", {cd, 3 * duration}, kMatrix},
                  {"caption_mod.bias", {3 * duration}, kF32},
                  {"ffn_gate", {duration, duration}, kMatrix},
                  {"ffn_up", {duration, duration}, kMatrix},
                  {"ffn_down", {duration, duration}, kMatrix}});
    const auto add_condition = [&](const std::string & prefix) {
        add_block(t, prefix, {{"0", {timestep, xd}, kMatrix}, {"1", {xd, xd}, kMatrix}, {"2", {xd, 3 * xd}, kMatrix}});
    };
    add_condition("dit.cond.");
    if (meanflow) add_condition("dit.delta_cond.");
    const int64_t dit_head = xd / dit_heads;
    add_numbered(t, "dit.blk.", dit_layers,
                 {{"attn_q", {xd, xd}, kMatrix},
                  {"attn_k", {xd, xd}, kMatrix},
                  {"attn_v", {xd, xd}, kMatrix},
                  {"attn_o", {xd, xd}, kMatrix},
                  {"attn_gate", {xd, xd}, kMatrix},
                  {"attn_k_text", {td, xd}, kMatrix},
                  {"attn_v_text", {td, xd}, kMatrix},
                  {"attn_k_speaker", {sd, xd}, kMatrix},
                  {"attn_v_speaker", {sd, xd}, kMatrix},
                  {"q_norm", {dit_head, dit_heads}, kF32},
                  {"k_norm", {dit_head, dit_heads}, kF32},
                  {"ffn_gate", {xd, dit_ffn}, kMatrix},
                  {"ffn_up", {xd, dit_ffn}, kMatrix},
                  {"ffn_down", {dit_ffn, xd}, kMatrix}});
    if (caption) add_numbered(t, "dit.blk.", dit_layers, {{"attn_k_caption", {cd, xd}, kMatrix}, {"attn_v_caption", {cd, xd}, kMatrix}});
    for (const char * ada : {"attn_ada.", "ffn_ada."}) {
        for (const char * part : {"shift.", "scale.", "gate."}) {
            const std::string name = std::string(ada) + part;
            add_numbered(t, "dit.blk.", dit_layers,
                         {{name + "down", {xd, rank}, kMatrix}, {name + "up.weight", {rank, xd}, kMatrix}, {name + "up.bias", {xd}, kF32}});
        }
    }

    const int64_t e = encoder_channels.back(), out = decoder_channels.back();
    add_block(t, "codec.",
              {{"enc.conv_in.weight", {1, first, kConvWidth}, kF32},
               {"enc.conv_in.bias", {first}, kF32},
               {"enc.snake.alpha", {e}, kF32},
               {"enc.snake.inv_alpha", {e}, kF32},
               {"enc.conv_out.weight", {e, codec_latent, kEncoderOutWidth}, kF32},
               {"enc.conv_out.bias", {codec_latent}, kF32},
               {"bottleneck.mean.weight", {codec_latent, ld}, kF32},
               {"bottleneck.mean.bias", {ld}, kF32},
               {"dec.in_proj.weight", {ld, codec_latent, 1}, kF32},
               {"dec.in_proj.bias", {codec_latent}, kF32},
               {"dec.conv_in.weight", {codec_latent, decoder, kConvWidth}, kF32},
               {"dec.conv_in.bias", {decoder}, kF32},
               {"dec.out_snake.alpha", {out}, kF32},
               {"dec.out_snake.inv_alpha", {out}, kF32},
               {"dec.conv_out.weight", {out, 1, kConvWidth}, kF32},
               {"dec.conv_out.bias", {1}, kF32}});
    for (size_t b = 0; b < encoder_rates.size(); b++) {
        // A block's strided convolution is stored as its two halves, [s * in, out] each.
        const int64_t c = encoder_channels[b], s = encoder_rates[b];
        const std::string prefix = "codec.enc.blk." + std::to_string(b) + ".";
        add_block(t, prefix,
                  {{"snake.alpha", {c}, kF32},
                   {"snake.inv_alpha", {c}, kF32},
                   {"down.first", {s * c, 2 * c}, kF32},
                   {"down.second", {s * c, 2 * c}, kF32},
                   {"down.bias", {2 * c}, kF32}});
        add_residual_units(t, prefix, c);
    }
    for (size_t b = 0; b < decoder_rates.size(); b++) {
        const int64_t in = decoder_channels[b], o = decoder_channels[b + 1];
        const std::string prefix = "codec.dec.blk." + std::to_string(b) + ".";
        add_block(t, prefix,
                  {{"snake.alpha", {in}, kF32},
                   {"snake.inv_alpha", {in}, kF32},
                   {"up.weight", {in, o, 2 * (int64_t) decoder_rates[b]}, kF32},
                   {"up.bias", {o}, kF32}});
        add_residual_units(t, prefix, o);
    }
    return t;
}

/**
 * Layout 1 of a model file, which releases from 0.7.0 read, holds neither the null speaker nor the caption's encoder and has no time at
 * which a request's scaling of the speaker ends. A file of it speaks with a reference and no caption, its duration
 * predictor's caption is as wide as the text condition, as layout 1 checked it, and its RF model ends the scaling at the
 * runtime's 0.9 that layout 2 writes, since every layout 1 file was converted from the runtime at 89f9d8f.
 */
void upgrade(ModelFile & m) {
    if (m.layout_version() == 1) {
        m.upgrade_bool("irodori-tts.duration.null_speaker", false);
        m.upgrade_bool("irodori-tts.caption_condition", false);
        m.upgrade_u32("irodori-tts.caption.dim", m.u32("irodori-tts.text.dim"));
        if (m.str("irodori-tts.flow") == "rf_velocity") m.upgrade_f32("irodori-tts.sampler.speaker_kv_min_t", 0.9f);
    }
    m.upgrade_str("irodori-tts.text.encoder", "modernbert");
    m.upgrade_str("irodori-tts.tokenizer.type", "Unigram");
}

/**
 * Layout 1 of a voice file, which releases from 0.7.0 wrote, holds one reference brought to the model's loudness, which
 * every model file of layout 1 gives as the runtime's -16 LUFS.
 */
void voice_upgrade(ModelFile & m) {
    const std::string p = "irodori-tts-voice.";
    m.upgrade_str(p + "source", "references");
    m.upgrade_f32_array(p + "references.seconds", {m.f32(p + "reference_seconds")});
    m.upgrade_i32_array(p + "references.sample_rates", {(int32_t) m.u32(p + "reference_sample_rate")});
    m.upgrade_bool(p + "normalized", true);
    m.upgrade_f32(p + "lufs", -16.0f);
}

}  // namespace

const Layout model_layout = {"irodori-tts", 3,
                             "convert it again with reference/irodori-tts/convert.py, or download it again from its Hugging Face "
                             "repository",
                             model_tensors, upgrade};

Layout voice_layout(const ModelFile & model) {
    return {kVoiceArchitecture, kVoiceLayout, "make it again from its WAVE files with speech voice MODEL REFERENCE.wav... VOICE.gguf",
            [&model](const ModelFile & m) {
                const std::string p = "irodori-tts-voice.";
                if (m.one_of(p + "source", {"references", "embedding"}) == "embedding") {
                    // An embedding is learned against one model's DiT and duration predictor, which no other model shares.
                    const std::string made = m.str(p + "model"), own = model.str("general.source.url");
                    if (made != own) {
                        throw Error(Fault::InvalidArgument, m.path() + " was made of an embedding for the model " + made + ", and " +
                                                                model.str("general.name") + " is " + own +
                                                                "; make the voice again from an embedding learned against this model");
                    }
                    return std::vector<TensorSpec>{{"speaker", {model.size("irodori-tts.speaker.dim"), m.width("speaker", 1)}, kF32}};
                }
                const std::string codec = m.str(p + "codec_sha256"), own = model.str("irodori-tts.codec.sha256");
                require(codec.size() == 64, m, "irodori-tts-voice.codec_sha256 is not a SHA-256 in hexadecimal");
                if (codec != own) {
                    throw Error(Fault::InvalidArgument, m.path() + " was made with the codec of SHA-256 " + codec + ", and " + model.str("general.name") +
                                                            " has the codec " + own + "; make the voice again from its WAVE files with this model");
                }
                const std::vector<float> seconds = m.f32_array(p + "references.seconds");
                const std::vector<int32_t> rates = m.i32_array(p + "references.sample_rates");
                require(!seconds.empty() && seconds.size() == rates.size(), m, "the voice file does not give each of its recordings a length and a rate");
                for (size_t i = 0; i < seconds.size(); i++) {
                    require(seconds[i] > 0 && rates[i] > 0, m, "the voice file gives a recording no length or no rate");
                }
                if (m.boolean(p + "normalized")) m.f32(p + "lufs");
                m.one_of(p + "device_kind", {"cpu", "gpu", "igpu"});
                return std::vector<TensorSpec>{{"latent", {model.size("irodori-tts.latent_dim"), m.width("latent", 1)}, kF32}};
            },
            voice_upgrade};
}

}  // namespace irodori
