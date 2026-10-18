#include "qwen2-tokenizer.h"

#include <algorithm>
#include <climits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "error.h"
#include "unicode.h"
#include "utf8.h"

namespace {

struct CodepointRange {
    uint32_t first, last;
};

#include "unicode-ranges.inc"

template <size_t N>
bool in_ranges(const CodepointRange (&ranges)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < ranges[mid].first) hi = mid;
        else if (cp > ranges[mid].last) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t cp) { return in_ranges(kLetters, cp); }
bool is_number(uint32_t cp) { return in_ranges(kNumbers, cp); }
bool is_mark(uint32_t cp) { return in_ranges(kMarks, cp); }
bool is_space(uint32_t cp) { return in_ranges(kSpaces, cp); }
bool is_crlf(uint32_t cp) { return cp == '\r' || cp == '\n'; }

const char * const kQwen2Regex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
const char * const kQwen35Regex =
    "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

std::vector<uint32_t> code_points(const std::string & text) {
    std::optional<std::vector<uint32_t>> cps = decode_utf8(text);
    if (!cps) throw Error(Fault::InvalidArgument, "the text to tokenize is not valid UTF-8; give UTF-8");
    return std::move(*cps);
}

/** Length of the pre-token at `i`, following the alternatives of the pattern in order. */
size_t match_at(const std::vector<uint32_t> & cps, size_t i, Qwen2Pattern pattern) {
    const size_t n = cps.size();
    // A run of letters, of letters and marks in Qwen3.5's pattern, and what is neither whitespace, a number nor that.
    const bool marks = pattern == Qwen2Pattern::Qwen35;
    const auto is_word = [&](uint32_t c) { return is_letter(c) || (marks && is_mark(c)); };
    const auto is_other = [&](uint32_t c) { return !is_space(c) && !is_number(c) && !is_word(c); };
    auto lower = [](uint32_t c) { return c < 128 ? (uint32_t) std::tolower((int) c) : c; };
    // (?i:'s|'t|'re|'ve|'m|'ll|'d)
    if (cps[i] == '\'' && i + 1 < n) {
        const uint32_t a = lower(cps[i + 1]);
        const uint32_t b = i + 2 < n ? lower(cps[i + 2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    // [^\r\n\p{L}\p{N}]?\p{L}+, or [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
    {
        size_t j = i;
        if (!is_letter(cps[j]) && !is_crlf(cps[j]) && !is_number(cps[j]) && j + 1 < n && is_word(cps[j + 1])) j++;
        if (is_word(cps[j])) {
            while (j < n && is_word(cps[j])) j++;
            return j - i;
        }
    }
    // \p{N}
    if (is_number(cps[i])) return 1;
    // ' ?[^\s\p{L}\p{N}]+[\r\n]*', or ' ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*'
    {
        size_t j = i;
        if (cps[j] == ' ' && j + 1 < n && is_other(cps[j + 1])) j++;
        if (is_other(cps[j])) {
            while (j < n && is_other(cps[j])) j++;
            while (j < n && is_crlf(cps[j])) j++;
            return j - i;
        }
    }
    // The remaining alternatives all start with whitespace; take the whole run first.
    size_t run = i;
    while (run < n && is_space(cps[run])) run++;
    // \s*[\r\n]+ : ends after the last \r or \n of the run.
    for (size_t j = run; j > i; j--) {
        if (is_crlf(cps[j - 1])) return j - i;
    }
    // \s+(?!\S) : the run, less its last character when something other than whitespace follows.
    if (run == n) return run - i;
    if (run - i >= 2) return run - i - 1;
    // \s+
    return run - i;
}

}  // namespace

bool qwen2_pattern(const std::string & regex, Qwen2Pattern * pattern) {
    if (regex == kQwen2Regex) *pattern = Qwen2Pattern::Qwen2;
    else if (regex == kQwen35Regex) *pattern = Qwen2Pattern::Qwen35;
    else return false;
    return true;
}

Qwen2Tokenizer::Qwen2Tokenizer(const ModelFile & m, const std::string & prefix, Qwen2Pattern pattern)
    : pattern_(pattern), tokens_(m.str_array(prefix + ".tokens")) {
    for (size_t i = 0; i < tokens_.size(); i++) {
        if (!tokens_[i].empty()) vocab_[tokens_[i]] = (int32_t) i;
    }
    const std::vector<std::string> merges = m.str_array(prefix + ".merges");
    for (size_t r = 0; r < merges.size(); r++) {
        const size_t sp = merges[r].find(' ');
        ranks_[{merges[r].substr(0, sp), merges[r].substr(sp + 1)}] = (int) r;
    }
    // GPT-2's byte-to-unicode table: printable bytes map to themselves, the rest to 256 and up.
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        const bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        const uint32_t cp = printable ? (uint32_t) b : (uint32_t) (256 + extra++);
        byte_to_unicode_[b] = encode_utf8({cp});
        unicode_to_byte_[cp] = (unsigned char) b;
    }
}

std::vector<std::string> Qwen2Tokenizer::pre_tokenize(const std::string & text) const {
    const std::vector<uint32_t> cps = nfc(code_points(text), UnicodeVersion::V9);
    std::vector<std::string> pieces;
    for (size_t i = 0; i < cps.size();) {
        const size_t len = match_at(cps, i, pattern_);
        pieces.push_back(encode_utf8(std::vector<uint32_t>(cps.begin() + (std::ptrdiff_t) i, cps.begin() + (std::ptrdiff_t) (i + len))));
        i += len;
    }
    return pieces;
}

std::vector<int32_t> Qwen2Tokenizer::bpe(const std::string & piece) const {
    std::vector<std::string> symbols;
    for (unsigned char c : piece) symbols.push_back(byte_to_unicode_[c]);
    while (symbols.size() > 1) {
        int best = INT_MAX;
        size_t at = 0;
        for (size_t k = 0; k + 1 < symbols.size(); k++) {
            const auto it = ranks_.find({symbols[k], symbols[k + 1]});
            if (it != ranks_.end() && it->second < best) {
                best = it->second;
                at = k;
            }
        }
        if (best == INT_MAX) break;
        symbols[at] += symbols[at + 1];
        symbols.erase(symbols.begin() + at + 1);
    }
    std::vector<int32_t> ids;
    for (const std::string & s : symbols) {
        const auto it = vocab_.find(s);
        if (it == vocab_.end()) throw std::runtime_error("the tokenizer has no token for a byte sequence");
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<int32_t> Qwen2Tokenizer::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (const std::string & piece : pre_tokenize(text)) {
        const std::vector<int32_t> p = bpe(piece);
        ids.insert(ids.end(), p.begin(), p.end());
    }
    return ids;
}

std::string Qwen2Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::string bytes;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tokens_.size()) continue;
        const std::string & token = tokens_[id];
        const std::optional<std::vector<uint32_t>> chars = decode_utf8(token);
        std::string mapped;
        bool byte_level = chars.has_value();
        for (size_t i = 0; byte_level && i < chars->size(); i++) {
            const auto it = unicode_to_byte_.find((*chars)[i]);
            if (it == unicode_to_byte_.end()) byte_level = false;
            else mapped += (char) it->second;
        }
        bytes += byte_level ? mapped : token;
    }
    return replace_invalid_utf8(bytes);
}
