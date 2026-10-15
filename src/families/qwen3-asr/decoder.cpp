#include "decoder.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "graph.h"

namespace qwen3_asr {

namespace {

/** The nodes of an embedding graph at most. */
constexpr int kEmbeddingGraphSize = 8;

}  // namespace

Decoder::Decoder(const ModelFile & m, ggml_backend_t backend, ggml_type cache_type, Qwen3Readback readback,
                 std::optional<Qwen3Attention> attention)
    : m_(m),
      backend_(backend),
      stack_(m, backend, "dec", read_qwen3_shape(m, "qwen3-asr.decoder"), m.u32("qwen3-asr.decoder.max_position_embeddings"), cache_type,
             readback, kQwen3BlockRows, attention),
      max_new_tokens_(m.u32("qwen3-asr.generation.max_new_tokens")),
      eos_(m.i32_array("qwen3-asr.generation.eos_ids")),
      allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))) {}

Decoder::~Decoder() {
    if (allocr_) ggml_gallocr_free(allocr_);
}

std::vector<float> Decoder::embeddings(const PromptIds & prompt, const std::vector<float> & audio) {
    const int64_t h = hidden(), n = (int64_t) prompt.ids.size(), audio_tokens = (int64_t) audio.size() / h;
    if (prompt.audio_start + audio_tokens > n) throw std::logic_error("the audio does not fit the prompt's audio tokens");
    std::vector<int32_t> text;
    for (int64_t i = 0; i < n; i++) {
        if (i < prompt.audio_start || i >= prompt.audio_start + audio_tokens) text.push_back(prompt.ids[(size_t) i]);
    }
    Graph g(kEmbeddingGraphSize);
    ggml_tensor * rows = ggml_get_rows(g.ctx(), m_.tensor("dec.token_embd"), g.input(text, (int64_t) text.size()));
    g.output(rows);
    g.compute(backend_, allocr_);
    const std::vector<float> text_rows = Graph::read(rows);
    std::vector<float> out((size_t) (n * h));
    const auto before = text_rows.begin() + (std::ptrdiff_t) (prompt.audio_start * h);
    std::copy(text_rows.begin(), before, out.begin());
    std::copy(audio.begin(), audio.end(), out.begin() + (std::ptrdiff_t) (prompt.audio_start * h));
    std::copy(before, text_rows.end(), out.begin() + (std::ptrdiff_t) ((prompt.audio_start + audio_tokens) * h));
    return out;
}

bool Decoder::prefill(const std::vector<float> & embeds, int64_t n, const std::function<bool(int64_t rows)> & keep_going) {
    const int64_t h = hidden();
    stack_.start(positions(n), n);
    for (int64_t from = 0; from < n; from += kQwen3BlockRows) {
        const int64_t count = std::min(kQwen3BlockRows, n - from);
        stack_.run(
            count,
            [&](Graph & g, int64_t, int64_t rows) {
                const auto first = embeds.begin() + (std::ptrdiff_t) (from * h);
                return g.input(std::vector<float>(first, first + (std::ptrdiff_t) (rows * h)), h, rows);
            },
            from + count == n ? m_.tensor("dec.token_embd") : nullptr);
        if (!keep_going(from + count)) return false;
    }
    return true;
}

void Decoder::step(int32_t id) {
    stack_.step(id, m_.tensor("dec.token_embd"), m_.tensor("dec.token_embd"));
}

void Draft::follow(const std::vector<int32_t> & written) {
    const int32_t last = written.back();
    if (!started_) {
        started_ = last == marker_;
        return;
    }
    if (at_ < ids_.size() && ids_[at_] == last) {
        at_++;
        return;
    }
    // The draft's two tokens that the decoding wrote last, nearest where it was.
    size_t found = kNowhere, distance = SIZE_MAX;
    if (written.size() >= 2) {
        const int32_t before = written[written.size() - 2];
        for (size_t j = 2; j <= ids_.size(); j++) {
            if (ids_[j - 1] != last || ids_[j - 2] != before) continue;
            const size_t d = at_ == kNowhere ? j : (j > at_ ? j - at_ : at_ - j);
            if (d < distance) {
                found = j;
                distance = d;
            }
        }
    }
    at_ = found;
}

std::vector<int32_t> Draft::ahead(size_t count) const {
    if (!started_ || at_ >= ids_.size()) return {};
    return std::vector<int32_t>(ids_.begin() + (std::ptrdiff_t) at_, ids_.begin() + (std::ptrdiff_t) std::min(ids_.size(), at_ + count));
}

std::optional<Generation> Decoder::generate(const std::function<bool(size_t tokens)> & keep_going, Draft * draft,
                                            std::vector<std::vector<float>> * seen) {
    const bool picks = stack_.readback() == Qwen3Readback::Pick;
    if (seen && picks) throw std::logic_error("a decoder that reads picks has no logits to give");
    const auto next = [&] {
        if (picks) return stack_.pick();
        const std::vector<float> & l = logits();
        if (seen) seen->push_back(l);
        return (int32_t) (std::max_element(l.begin(), l.end()) - l.begin());
    };
    Generation out;
    int32_t id = next();
    // Writes `token`, and says whether the decoding goes on: not after an end token, at the limit, or once stopped.
    std::optional<bool> stopped;
    const auto write = [&](int32_t token) {
        out.ids.push_back(token);
        if (std::find(eos_.begin(), eos_.end(), token) != eos_.end()) return false;
        if ((int64_t) out.ids.size() == max_new_tokens_) {
            out.limited = true;
            return false;
        }
        if (!keep_going(out.ids.size())) {
            stopped = true;
            return false;
        }
        if (draft) draft->follow(out.ids);
        return true;
    };
    for (;;) {
        if (!write(id)) break;
        std::vector<int32_t> rows{id};
        if (draft) {
            // Each token kept takes a position, and the pick after them one more: the rows stay within the limit.
            const size_t room = (size_t) (max_new_tokens_ - (int64_t) out.ids.size());
            for (int32_t d : draft->ahead(std::min(kDraftTokens, room))) rows.push_back(d);
        }
        if (rows.size() == 1) {
            step(id);
            id = next();
            continue;
        }
        const int64_t past = stack_.n_past();
        std::vector<float> all;
        const std::vector<int32_t> picked =
            stack_.verify(rows, m_.tensor("dec.token_embd"), m_.tensor("dec.token_embd"), seen ? &all : nullptr);
        size_t kept = 0;
        while (kept + 1 < rows.size() && picked[kept] == rows[kept + 1]) kept++;
        draft->checks++;
        draft->checked += (int) rows.size() - 1;
        draft->kept += (int) kept;
        stack_.rewind(past + 1 + (int64_t) kept);
        const size_t vocab = all.size() / rows.size();
        const auto saw = [&](size_t row) {
            if (seen) seen->emplace_back(all.begin() + (std::ptrdiff_t) (row * vocab), all.begin() + (std::ptrdiff_t) ((row + 1) * vocab));
        };
        bool goes_on = true;
        for (size_t j = 1; j <= kept && goes_on; j++) {
            saw(j - 1);
            goes_on = write(rows[j]);
        }
        if (!goes_on) break;
        saw(kept);
        id = picked[kept];
    }
    if (stopped) return std::nullopt;
    return out;
}

}  // namespace qwen3_asr
