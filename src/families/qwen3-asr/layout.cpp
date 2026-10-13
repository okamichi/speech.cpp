#include "layout.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>

#include "error.h"
#include "qwen3-decoder.h"

namespace qwen3_asr {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type: a matrix of a linear layer or the token
 * embeddings, which are also the decoder's output matrix and which ggml_mul_mat() and ggml_get_rows() alone read, in
 * the file's type, which 0.7's reader took in Q8_0, F16 and F32; a convolution kernel, which ggml_im2col() reads in F16
 * or F32 alone, in F16, or F32 in an F32 file; and the norms, the biases and the frontend in F32. The encoder holds in
 * Q8_0: with the decoder's weights of the Q8_0 file, teacher-forced on the dumps' ids of every input of up to 30 s
 * (requests auto and auto-prompt), its output from a Q8_0 file moved the argmax of 1 or 2 of 605 steps with the 0.6B
 * model and of none of 604 with the 1.7B, on the CPU and on Metal of an Apple M5, where the official encoder's output
 * moved 2 and none (2026-10-06); each step it moved had a margin of 0.13 or less in the official model.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, kFirstLayoutRelease},
                                           {GGML_TYPE_F16, kFirstLayoutRelease},
                                           {GGML_TYPE_Q8_0, kFirstLayoutRelease},
                                           {GGML_TYPE_Q6_K, "0.8.0"},
                                           {GGML_TYPE_Q5_K, "0.8.0"},
                                           {GGML_TYPE_Q4_K, "0.8.0"}});
const Storage kConv = half_storage();
const Storage kF32 = float32_storage();

void require(bool condition, const ModelFile & m, const std::string & what) {
    if (!condition) throw Error(Fault::File, m.path() + ": " + what + "; " + m.remedy());
}

/** An [i32] key of token ids, each one of the `tokens` the tokenizer has; `sorted` asks for ascending ids without repeats. */
std::vector<int32_t> token_ids(const ModelFile & m, const std::string & key, size_t tokens, bool sorted) {
    const std::vector<int32_t> ids = m.i32_array(key);
    require(std::all_of(ids.begin(), ids.end(), [&](int32_t id) { return id >= 0 && (size_t) id < tokens; }), m,
            key + " holds an id that is not one of the tokenizer's tokens");
    require(!sorted || std::adjacent_find(ids.begin(), ids.end(), std::greater_equal<int32_t>()) == ids.end(), m,
            key + " is not sorted without repeats");
    return ids;
}

/** Reads every key of the layout, checks the ones that must agree, and names the tensors they call for. */
std::vector<TensorSpec> tensors(const ModelFile & m) {
    check_model_keys(m, "recognition", "steers");
    const std::string p = "qwen3-asr.";
    // The parse compares the prefix and the names with what the model writes in ASCII, where its case mapping is Python's,
    // and a name with what normalize_language_name() makes of it: its first letter raised and the others lowered.
    const auto ascii = [](const std::string & s) { return std::all_of(s.begin(), s.end(), [](char c) { return (unsigned char) c < 0x80; }); };
    const auto capital = [](char c) { return c >= 'A' && c <= 'Z'; };
    const auto normalized = [&](const std::string & name) {
        return !name.empty() && ascii(name) && !(name[0] >= 'a' && name[0] <= 'z') && std::none_of(name.begin() + 1, name.end(), capital);
    };
    const std::vector<std::string> names = m.str_array(p + "language_names");
    require(names.size() == m.str_array("general.languages").size() && std::all_of(names.begin(), names.end(), normalized), m,
            "qwen3-asr.language_names does not have one name per language of general.languages, in ASCII with only its first "
            "letter a capital");

    const int n_fft = m.size(p + "frontend.n_fft"), mels = m.size(p + "frontend.n_mels");
    m.size(p + "frontend.hop_length");
    // The frontend's real FFT packs the frame's values in pairs.
    require(n_fft % 2 == 0, m, "qwen3-asr.frontend.n_fft is odd");
    const float log_floor = m.f32(p + "frontend.log_floor"), range = m.f32(p + "frontend.dynamic_range");
    const float offset = m.f32(p + "frontend.log_offset"), divisor = m.f32(p + "frontend.log_divisor");
    require(log_floor > 0 && std::isfinite(log_floor) && range >= 0 && std::isfinite(range) && std::isfinite(offset) &&
                std::isfinite(divisor) && divisor != 0,
            m, "qwen3-asr.frontend.log_floor, dynamic_range, log_offset or log_divisor is out of its range");
    // The STFT reflects the audio by half a frame at either end, which takes more samples than that.
    const int min_samples = m.size(p + "audio.min_samples");
    require(min_samples > n_fft / 2, m, "qwen3-asr.audio.min_samples does not exceed half of qwen3-asr.frontend.n_fft");
    require(m.size(p + "audio.max_samples") >= min_samples, m, "qwen3-asr.audio.max_samples is below qwen3-asr.audio.min_samples");
    m.size(p + "audio.split_search_samples");
    m.size(p + "audio.split_window_samples");

    const int d = m.size(p + "encoder.d_model"), heads = m.size(p + "encoder.num_heads"), ffn = m.size(p + "encoder.ffn_dim");
    const int encoder_layers = m.count(p + "encoder.num_layers");
    // The positions' sinusoids take a sine and a cosine of each of half the channels, at d_model / 2 - 1 steps apart.
    require(d % heads == 0 && d % 2 == 0 && d >= 4, m,
            "qwen3-asr.encoder.d_model is not a multiple of qwen3-asr.encoder.num_heads, or odd, or below 4");
    const int chunk = m.size(p + "encoder.chunk_frames");
    require(m.size(p + "encoder.window_frames") % chunk == 0, m,
            "qwen3-asr.encoder.window_frames is not a multiple of qwen3-asr.encoder.chunk_frames");
    const float eps = m.f32(p + "encoder.norm_eps"), timescale = m.f32(p + "encoder.max_timescale");
    require(eps >= 0 && std::isfinite(eps) && timescale > 0 && std::isfinite(timescale), m,
            "qwen3-asr.encoder.norm_eps or max_timescale is out of its range");

    const Qwen3Shape decoder = read_qwen3_shape(m, p + "decoder");
    const int vocab = m.size(p + "decoder.vocab_size");
    const int positions = m.size(p + "decoder.max_position_embeddings");

    for (const char * key : {"before_context", "before_audio", "after_audio"}) m.str(p + "prompt." + key);
    require(ascii(m.str(p + "prompt.language_prefix")), m, "qwen3-asr.prompt.language_prefix is not ASCII");
    m.size(p + "output.repetition_threshold");
    m.size(p + "output.repetition_max_period");

    const std::vector<std::string> tokens = m.str_array(p + "tokenizer.tokens");
    m.str_array(p + "tokenizer.merges");
    require(tokens.size() <= (size_t) vocab, m, "qwen3-asr.tokenizer.tokens has more tokens than qwen3-asr.decoder.vocab_size");
    const std::vector<int32_t> added = token_ids(m, p + "tokenizer.added_ids", tokens.size(), true);
    const std::vector<int32_t> special = token_ids(m, p + "tokenizer.special_ids", tokens.size(), true);
    require(std::includes(added.begin(), added.end(), special.begin(), special.end()), m,
            "qwen3-asr.tokenizer.special_ids holds an id that is not one of qwen3-asr.tokenizer.added_ids");
    for (const char * key : {"asr_text", "audio_token"}) {
        const std::string token = m.str(p + "prompt." + key);
        require(std::any_of(added.begin(), added.end(), [&](int32_t id) { return tokens[(size_t) id] == token; }), m,
                p + "prompt." + key + " is not one of the tokenizer's added tokens");
    }
    require(!token_ids(m, p + "generation.eos_ids", tokens.size(), false).empty(), m, "qwen3-asr.generation.eos_ids is empty");
    require(m.size(p + "generation.max_new_tokens") < positions, m,
            "qwen3-asr.generation.max_new_tokens leaves the decoder no position for the prompt");

    // The convolutions' channels have no key: they are the width of one tensor, which the others are checked against.
    const int64_t channels = m.width("enc.conv.1.weight", 3), k = kConvWidth, h = decoder.hidden;
    std::vector<TensorSpec> t = {{"frontend.window", {n_fft}, kF32},
                                 {"frontend.filterbank", {n_fft / 2 + 1, mels}, kF32},
                                 {"enc.conv.1.weight", {k, k, 1, channels}, kConv},
                                 {"enc.conv_out.weight", {channels * after_convolutions(mels), d}, kMatrix},
                                 {"enc.norm.weight", {d}, kF32},
                                 {"enc.norm.bias", {d}, kF32},
                                 {"proj.1.weight", {d, d}, kMatrix},
                                 {"proj.1.bias", {d}, kF32},
                                 {"proj.2.weight", {d, h}, kMatrix},
                                 {"proj.2.bias", {h}, kF32},
                                 {"dec.token_embd", {h, vocab}, kMatrix}};
    for (int i = 1; i <= kConvLayers; i++) {
        if (i > 1) t.push_back({"enc.conv." + std::to_string(i) + ".weight", {k, k, channels, channels}, kConv});
        t.push_back({"enc.conv." + std::to_string(i) + ".bias", {channels}, kF32});
    }
    add_numbered(t, "enc.blk.", encoder_layers,
                 {{"attn_norm.weight", {d}, kF32},     {"attn_norm.bias", {d}, kF32},     {"attn_q.weight", {d, d}, kMatrix},
                  {"attn_q.bias", {d}, kF32},          {"attn_k.weight", {d, d}, kMatrix}, {"attn_k.bias", {d}, kF32},
                  {"attn_v.weight", {d, d}, kMatrix},  {"attn_v.bias", {d}, kF32},        {"attn_out.weight", {d, d}, kMatrix},
                  {"attn_out.bias", {d}, kF32},        {"ffn_norm.weight", {d}, kF32},    {"ffn_norm.bias", {d}, kF32},
                  {"ffn_up.weight", {d, ffn}, kMatrix}, {"ffn_up.bias", {ffn}, kF32},     {"ffn_down.weight", {ffn, d}, kMatrix},
                  {"ffn_down.bias", {d}, kF32}});
    add_qwen3_tensors(t, "dec", decoder, kMatrix);
    return t;
}

}  // namespace

const Layout layout = {"qwen3-asr", 1,
                       "convert it again with reference/qwen3-asr/convert.py, or download it again from its Hugging Face "
                       "repository",
                       tensors};

}  // namespace qwen3_asr
