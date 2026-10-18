#include "audio-decoder.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "error.h"

namespace drama3 {

namespace {

/**
 * The rows one graph of a prefill runs at most: a block's attention scores take rows × positions × heads × 4 bytes,
 * 268 MB for a block of the longest prompt, 8,192 rows of 16 heads.
 */
constexpr int64_t kBlockRows = 512;

std::string layer_name(int l, int shared) {
    return l < shared ? "ad.shared." + std::to_string(l) + "." : "ad.layer." + std::to_string(l - shared) + ".";
}

}  // namespace

/** What the layers of one graph read of its rows. */
struct AudioDecoder::Rows {
    int64_t first = 0, n = 0;
    ggml_tensor * positions = nullptr;
    ggml_tensor * mask = nullptr;
    /** 1 where the frame after the row is audio, whose codes the layers mix in, and 0 where it is text, [1, n]. */
    ggml_tensor * audio = nullptr;
    /** The summed embeddings of the row's own codes that the text model did not see, [dim, n]. */
    ggml_tensor * unseen = nullptr;
};

AudioDecoder::AudioDecoder(const ModelFile & m, ggml_backend_t backend) : backend_(backend), m_(m) {
    const std::string a = "drama3.audio_decoder.";
    d_ = m.size(a + "dim");
    heads_ = m.size(a + "n_head");
    kv_heads_ = m.size(a + "n_local_heads");
    head_dim_ = m.size(a + "head_dim");
    rotary_dim_ = m.size(a + "rotary_dim");
    codebooks_ = m.count(a + "num_codebooks");
    vocab_ = m.size(a + "vocab_size");
    shared_ = m.count(a + "shared_enrichment_layers");
    seen_ = (int) m.u32(a + "vq_backbone_codebooks");
    eps_ = m.f32(a + "norm_eps");
    rope_base_ = m.f32(a + "rope_base");
    semantic_begin_ = (int32_t) m.u32("drama3.semantic_start_token_id");
    semantic_end_ = (int32_t) m.u32("drama3.semantic_end_token_id");
    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
}

AudioDecoder::~AudioDecoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

void AudioDecoder::start(int64_t rows) {
    if (!cache_ || cache_->capacity() < rows) {
        cache_.reset();
        cache_ = std::make_unique<KeysValues>(backend_, layers(), head_dim_, kv_heads_, rows);
    }
    n_past_ = 0;
}

void AudioDecoder::rewind(int64_t rows) {
    if (!cache_ || rows > n_past_) throw std::invalid_argument("the audio decoder rewinds only to rows it has run");
    n_past_ = rows;
}

ggml_tensor * AudioDecoder::layer(Graph & g, int l, ggml_tensor * hidden, ggml_tensor * codes, const Rows & r) {
    ggml_context * c = g.ctx();
    const std::string p = layer_name(l, shared_);
    const auto norm = [&](ggml_tensor * x, const std::string & name) { return ggml_mul(c, ggml_rms_norm(c, x, eps_), m_.tensor(p + name)); };
    const int64_t n = r.n;

    ggml_tensor * e = ggml_mul(c, ggml_get_rows(c, m_.tensor(p + "embd"), codes), r.audio);
    if (l == 0 && r.unseen) e = ggml_add(c, e, r.unseen);
    ggml_tensor * x = mul_mat(c, m_.tensor(p + "eh_proj"), ggml_concat(c, norm(e, "enorm"), norm(hidden, "hnorm"), 0));

    ggml_tensor * h = norm(x, "attn_norm");
    ggml_tensor * q = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_q"), h), head_dim_, heads_, n);
    ggml_tensor * k = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_k"), h), head_dim_, kv_heads_, n);
    ggml_tensor * v = ggml_reshape_3d(c, mul_mat(c, m_.tensor(p + "attn_v"), h), head_dim_, kv_heads_, n);
    ggml_tensor * gate = ggml_sigmoid(c, mul_mat(c, m_.tensor(p + "attn_gate"), h));
    // RoPE turns the first rotary_dim channels of each head in adjacent pairs, ggml's normal mode.
    q = ggml_rope_ext(c, norm(q, "attn_q_norm"), r.positions, nullptr, rotary_dim_, GGML_ROPE_TYPE_NORMAL, 0, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(c, norm(k, "attn_k_norm"), r.positions, nullptr, rotary_dim_, GGML_ROPE_TYPE_NORMAL, 0, rope_base_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * o = ggml_mul(c, cache_->attend(g, l, q, k, v, r.first, r.mask), gate);
    x = ggml_add(c, x, mul_mat(c, m_.tensor(p + "attn_output"), o));

    h = norm(x, "ffn_norm");
    h = ggml_mul(c, ggml_silu(c, mul_mat(c, m_.tensor(p + "ffn_gate"), h)), mul_mat(c, m_.tensor(p + "ffn_up"), h));
    x = ggml_add(c, x, mul_mat(c, m_.tensor(p + "ffn_down"), h));
    return norm(x, "final_norm");
}

AudioDecoder::Rows AudioDecoder::rows(Graph & g, int64_t n, const int32_t * frames, const std::vector<float> & audio, bool unseen) const {
    Rows r;
    r.first = n_past_;
    r.n = n;
    r.positions = positions(g, r.first, n);
    r.mask = causal_mask(g, r.first, n);
    r.audio = g.input(audio, 1, n);
    if (!unseen) return r;

    const int width = 1 + codebooks_;
    std::vector<int32_t> ids;
    std::vector<float> own((size_t) n);
    for (int64_t i = 0; i < n; i++) own[(size_t) i] = semantic(frames[i * width]) ? 1.0f : 0.0f;
    // The file holds the embeddings of codebooks 1 to 9, codebook b's from row (b - 1) * vocab_size.
    for (int b = 1 + seen_; b < codebooks_; b++) {
        for (int64_t i = 0; i < n; i++) {
            const int32_t code = frames[i * width + 1 + b];
            if (code < 0 || code >= vocab_) throw std::invalid_argument("a frame's code is outside its codebook");
            ids.push_back((b - 1) * vocab_ + code);
        }
    }
    if (!ids.empty()) {
        ggml_context * c = g.ctx();
        ggml_tensor * e = ggml_get_rows(c, m_.tensor("ad.codebook_embd"), g.input(ids, (int64_t) ids.size()));
        ggml_tensor * sum = nullptr;
        for (int64_t j = 0; j < (int64_t) ids.size() / n; j++) {
            ggml_tensor * part = ggml_view_2d(c, e, d_, n, e->nb[1], (size_t) (j * n) * e->nb[1]);
            sum = sum ? ggml_add(c, sum, part) : part;
        }
        r.unseen = ggml_mul(c, sum, g.input(own, 1, n));
    }
    return r;
}

void AudioDecoder::prefill(const std::vector<float> & hidden, const std::vector<int32_t> & frames, int64_t n,
                           std::vector<std::vector<float>> * outputs) {
    const int width = 1 + codebooks_;
    if (n < 1 || (int64_t) hidden.size() != n * d_ || (int64_t) frames.size() != (n + 1) * width) {
        throw std::invalid_argument("the audio decoder's prefill takes a state of each row and the frames of the rows and the one after");
    }
    if (!cache_ || n_past_ + n > cache_->capacity()) throw std::invalid_argument("the audio decoder's rows outgrow the sequence it was started for");
    if (outputs) outputs->assign((size_t) layers(), {});
    for (int64_t from = 0; from < n; from += kBlockRows) {
        const int64_t rows_n = std::min(kBlockRows, n - from);
        const int32_t * own = frames.data() + from * width;
        Graph g;
        std::vector<ggml_tensor *> kept;
        build_prefill(g, g.input(std::vector<float>(hidden.begin() + from * d_, hidden.begin() + (from + rows_n) * d_), d_, rows_n), own,
                      outputs ? &kept : nullptr);
        g.compute(backend_, allocr_);
        for (size_t l = 0; l < kept.size(); l++) {
            const std::vector<float> out = Graph::read(kept[l]);
            (*outputs)[l].insert((*outputs)[l].end(), out.begin(), out.end());
        }
        n_past_ += rows_n;
    }
}

ggml_tensor * AudioDecoder::build_prefill(Graph & g, ggml_tensor * x, const int32_t * own, std::vector<ggml_tensor *> * kept) {
    const int width = 1 + codebooks_;
    const int64_t rows_n = x->ne[1];
    const int32_t * next = own + width;
    std::vector<float> audio((size_t) rows_n);
    for (int64_t i = 0; i < rows_n; i++) audio[(size_t) i] = semantic(next[i * width]) ? 1.0f : 0.0f;
    const Rows r = rows(g, rows_n, own, audio, true);
    for (int l = 0; l < layers(); l++) {
        // The shared layers mix in the first code of the frame after each row, and the layer of residual codebook b
        // the code of codebook b - 1, each clamped to its embeddings as the official layer clamps it.
        const int column = 1 + (l < shared_ ? 0 : l - shared_);
        std::vector<int32_t> codes((size_t) rows_n);
        for (int64_t i = 0; i < rows_n; i++) codes[(size_t) i] = std::clamp(next[i * width + column], 0, vocab_ - 1);
        x = layer(g, l, x, g.input(codes, rows_n), r);
        if (kept) {
            g.output(x);
            kept->push_back(x);
        }
    }
    g.output(x);
    return x;
}

void AudioDecoder::reserve(int64_t rows, int64_t prompt_rows) {
    if (prompt_rows < 2 || prompt_rows > rows) throw std::invalid_argument("the audio decoder reserves for a prompt within its sequence");
    start(rows);
    // The largest graph of such a sequence: a block of the prompt's rows that attends to every row before it.
    const int64_t block = std::min(kBlockRows, prompt_rows - 1);
    n_past_ = prompt_rows - 1 - block;
    const std::vector<int32_t> frames((size_t) ((block + 1) * (1 + codebooks_)), 0);
    Graph g;
    build_prefill(g, g.zeros(d_, block), frames.data(), nullptr);
    if (!ggml_gallocr_reserve(allocr_, g.cgraph())) {
        throw Error(Fault::OutOfMemory, "cannot reserve the memory of the audio decoder's graphs on " + std::string(ggml_backend_name(backend_)));
    }
    start(rows);
}

std::vector<int32_t> AudioDecoder::step(const std::vector<float> & hidden, const std::vector<int32_t> & frame, int32_t main,
                                        const ChooseCode & choose, std::vector<std::vector<float>> * outputs) {
    if ((int64_t) hidden.size() != d_ || (int64_t) frame.size() != 1 + codebooks_) {
        throw std::invalid_argument("the audio decoder's step takes the state and the frame of one row");
    }
    if (!cache_ || n_past_ + 1 > cache_->capacity()) throw std::invalid_argument("the audio decoder's rows outgrow the sequence it was started for");
    if (outputs) outputs->clear();
    const std::vector<float> audio = {semantic(main) ? 1.0f : 0.0f};
    std::vector<int32_t> codes = {std::clamp(main - semantic_begin_, 0, vocab_ - 1)};
    std::vector<float> state = hidden;
    // A graph for each residual codebook's layer, the first with the shared layers before it, since each layer mixes in
    // the code drawn from the head before it.
    for (int b = 0; b + 1 < codebooks_; b++) {
        Graph g;
        const Rows r = rows(g, 1, frame.data(), audio, b == 0);
        ggml_tensor * x = g.input(state, d_, 1);
        std::vector<ggml_tensor *> kept;
        for (int l = b == 0 ? 0 : shared_ + b; l <= shared_ + b; l++) {
            x = layer(g, l, x, g.input(std::vector<int32_t>{std::clamp(codes.back(), 0, vocab_ - 1)}, 1), r);
            if (outputs) {
                g.output(x);
                kept.push_back(x);
            }
        }
        ggml_tensor * logits = mul_mat(g.ctx(), m_.tensor("ad.head." + std::to_string(b)), x);
        g.output(x);
        g.output(logits);
        g.compute(backend_, allocr_);
        for (ggml_tensor * t : kept) outputs->push_back(Graph::read(t));
        state = Graph::read(x);
        codes.push_back(choose(b + 1, Graph::read(logits)));
    }
    n_past_++;
    return codes;
}

}  // namespace drama3
