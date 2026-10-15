#include "tokenizer.h"

#include "error.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <stdexcept>

#include "error.h"

namespace irodori {

namespace {

/** The Unigram path ends at a byte offset; `start` is where its last piece starts, -1 while unreached. */
struct PathEnd {
    int32_t id = -1;
    double score = 0;
    int64_t start = -1;
};

/** Hugging Face's penalty under the lowest piece score for a character no piece covers. */
constexpr double kUnknownPenalty = 10.0;

size_t utf8_length(unsigned char c) {
    return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : 4;
}

}  // namespace

Tokenizer::Tokenizer(const ModelFile & m) {
    bpe_ = m.one_of("irodori-tts.tokenizer.model", {"unigram", "bpe"}) == "bpe";
    tokens_ = m.str_array("irodori-tts.tokenizer.tokens");
    const std::vector<int32_t> added = m.i32_array("irodori-tts.tokenizer.added_ids");
    bos_ = (int32_t) m.u32("irodori-tts.tokenizer.bos_id");
    unknown_ = (int32_t) m.u32("irodori-tts.tokenizer.unknown_id");
    for (size_t i = 0; i < tokens_.size(); i++) {
        ids_.emplace(tokens_[i], (int32_t) i);
        longest_ = std::max(longest_, tokens_[i].size());
    }
    if (bpe_) {
        const std::vector<int32_t> left = m.i32_array("irodori-tts.tokenizer.merges_left");
        const std::vector<int32_t> right = m.i32_array("irodori-tts.tokenizer.merges_right");
        if (left.size() != right.size()) throw Error(Fault::File, m.path() + ": the tokenizer's merges do not each have two pieces");
        for (size_t r = 0; r < left.size(); r++) {
            const auto merged = ids_.find(tokens_.at((size_t) left[r]) + tokens_.at((size_t) right[r]));
            if (merged == ids_.end()) throw Error(Fault::File, m.path() + ": a merge of the tokenizer gives no piece");
            merges_.emplace((uint64_t) (uint32_t) left[r] << 32 | (uint32_t) right[r], Merge{(int32_t) r, merged->second});
        }
    } else {
        scores_ = m.f64_array("irodori-tts.tokenizer.scores");
        if (scores_.size() != tokens_.size()) throw Error(Fault::File, m.path() + ": the tokenizer does not have a score for each token");
        double lowest = 0;
        for (double score : scores_) lowest = std::min(lowest, score);
        unknown_score_ = lowest - kUnknownPenalty;
    }
    for (int32_t id : added) {
        const std::string & t = tokens_.at((size_t) id);
        if (t.empty()) throw Error(Fault::File, m.path() + ": an added token of the tokenizer is empty");
        added_[(unsigned char) t[0]].push_back(id);
    }
    for (int b = 0; b < 256; b++) {
        char name[8];
        std::snprintf(name, sizeof name, "<0x%02X>", b);
        const auto it = ids_.find(name);
        if (it == ids_.end()) throw Error(Fault::File, m.path() + ": the tokenizer has no byte token " + name);
        bytes_[b] = it->second;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids = {bos_};
    std::string piece;
    auto flush = [&]() {
        if (piece.empty()) return;
        if (bpe_) {
            encode_bpe(piece, ids);
        } else {
            encode_unigram(piece, ids);
        }
        piece.clear();
    };
    for (size_t i = 0; i < text.size();) {
        // The leftmost and then longest added token, matched on the text before anything else.
        int32_t added = -1;
        for (int32_t id : added_[(unsigned char) text[i]]) {
            const std::string & t = tokens_[id];
            if (text.compare(i, t.size(), t) == 0 && (added < 0 || t.size() > tokens_[added].size())) added = id;
        }
        if (added >= 0) {
            flush();
            ids.push_back(added);
            i += tokens_[added].size();
        } else if (text[i] == ' ') {
            piece += "\xE2\x96\x81";
            i++;
        } else {
            piece += text[i++];
        }
    }
    flush();
    return ids;
}

void Tokenizer::encode_unigram(const std::string & s, std::vector<int32_t> & ids) const {
    const size_t n = s.size();
    std::vector<PathEnd> best(n + 1);
    best[0].start = 0;
    for (size_t start = 0; start < n;) {
        const size_t char_length = utf8_length((unsigned char) s[start]);
        const double till_here = best[start].score;
        bool covered = false;
        for (size_t length = 1; length <= longest_ && start + length <= n; length++) {
            const auto it = ids_.find(s.substr(start, length));
            if (it == ids_.end()) continue;
            PathEnd & end = best[start + length];
            const double score = till_here + scores_[it->second];
            if (end.start < 0 || score > end.score) end = {it->second, score, (int64_t) start};
            if (length == char_length) covered = true;
        }
        if (!covered) {
            PathEnd & end = best[start + char_length];
            const double score = till_here + unknown_score_;
            if (end.start < 0 || score > end.score) end = {unknown_, score, (int64_t) start};
        }
        start += char_length;
    }
    // Back from the end; consecutive unknown characters are fused into one piece, then spelled in bytes.
    std::vector<std::pair<std::string, int32_t>> pieces;
    for (size_t end = n; end > 0;) {
        const PathEnd & p = best[end];
        const std::string text = s.substr((size_t) p.start, end - (size_t) p.start);
        if (p.id == unknown_ && !pieces.empty() && pieces.back().second == unknown_) {
            pieces.back().first = text + pieces.back().first;
        } else {
            pieces.push_back({text, p.id});
        }
        end = (size_t) p.start;
    }
    for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
        if (it->second != unknown_) {
            ids.push_back(it->second);
            continue;
        }
        const auto known = ids_.find(it->first);
        if (known != ids_.end()) {
            ids.push_back(known->second);
            continue;
        }
        for (unsigned char b : it->first) ids.push_back(bytes_[b]);
    }
}

void Tokenizer::encode_bpe(const std::string & s, std::vector<int32_t> & ids) const {
    // The pieces of the characters, linked to their neighbours; a piece merged into the one on its left is emptied.
    struct Symbol {
        int32_t id;
        int prev, next;
        bool merged;
    };
    std::vector<Symbol> symbols;
    for (size_t i = 0; i < s.size();) {
        const size_t length = std::min(utf8_length((unsigned char) s[i]), s.size() - i);
        const auto it = ids_.find(s.substr(i, length));
        const auto add = [&](int32_t id) { symbols.push_back({id, (int) symbols.size() - 1, (int) symbols.size() + 1, false}); };
        if (it != ids_.end()) {
            add(it->second);
        } else {
            for (size_t b = i; b < i + length; b++) add(bytes_[(unsigned char) s[b]]);
        }
        i += length;
    }
    if (symbols.empty()) return;
    symbols.back().next = -1;

    // The candidate merges, the lowest rank first and of one rank the leftmost; one whose pair has changed since it was
    // queued is passed over, as Hugging Face's Word::merge_all() does.
    struct Candidate {
        int32_t rank;
        int pos;
        int32_t id;
        bool operator<(const Candidate & o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; }
    };
    std::priority_queue<Candidate> queue;
    const auto find = [&](int32_t a, int32_t b) {
        const auto it = merges_.find((uint64_t) (uint32_t) a << 32 | (uint32_t) b);
        return it == merges_.end() ? nullptr : &it->second;
    };
    const auto offer = [&](int pos) {
        const int next = symbols[pos].next;
        if (next < 0) return;
        if (const Merge * m = find(symbols[pos].id, symbols[next].id)) queue.push({m->rank, pos, m->id});
    };
    for (int i = 0; i + 1 < (int) symbols.size(); i++) offer(i);
    while (!queue.empty()) {
        const Candidate c = queue.top();
        queue.pop();
        Symbol & left = symbols[c.pos];
        if (left.merged || left.next < 0) continue;
        Symbol & right = symbols[left.next];
        const Merge * m = find(left.id, right.id);
        if (m == nullptr || m->id != c.id) continue;
        left.id = c.id;
        right.merged = true;
        left.next = right.next;
        if (left.next >= 0) symbols[left.next].prev = c.pos;
        if (left.prev >= 0) offer(left.prev);
        offer(c.pos);
    }
    for (const Symbol & symbol : symbols) {
        if (!symbol.merged) ids.push_back(symbol.id);
    }
}

std::vector<int32_t> Tokenizer::encode_caption(const std::string & text, size_t limit, bool compatible, bool & truncated) const {
    auto ids = text.empty() ? std::vector<int32_t>{} : encode(text);
    truncated = ids.size() > limit;
    if (truncated && !compatible) {
        throw Error(Fault::OutOfRange, "the instructions are " + std::to_string(ids.size()) +
                                          " tokens long and Irodori-TTS takes at most " + std::to_string(limit) + "; shorten them",
                    "instructions");
    }
    if (truncated) ids.resize(limit);
    return ids;
}

}  // namespace irodori
