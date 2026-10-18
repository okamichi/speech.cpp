#include "chunking.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>

#include "error.h"
#include "utf8.h"

namespace drama3 {

namespace {

struct CodepointRange {
    uint32_t first, last;
};

#include "unicode.inc"

using Text = std::u32string;

template <size_t N>
bool in_table(const CodepointRange (&table)[N], char32_t c) {
    const CodepointRange * after = std::upper_bound(table, table + N, c, [](char32_t c, const CodepointRange & r) { return c < r.first; });
    return after != table && c <= (after - 1)->last;
}

/** The White_Space property of Unicode, which Python's str.strip() strips. */
bool is_white_space(char32_t c) {
    return (c >= 0x9 && c <= 0xD) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

/** What Python's regular expression \w matches. */
bool is_word(char32_t c) {
    return in_table(kWordCharacters, c);
}

/** What Python's str.isdigit() accepts. */
bool is_digit(char32_t c) {
    return in_table(kDigits, c);
}

bool is_ascii_letter(char32_t c) {
    return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z');
}

Text decode(const std::string & text) {
    std::optional<std::vector<uint32_t>> cps = decode_utf8(text);
    if (!cps) throw Error(Fault::InvalidArgument, "the text is not valid UTF-8; give UTF-8", "text");
    return Text(cps->begin(), cps->end());
}

std::string encode(const Text & text) {
    return encode_utf8(std::vector<uint32_t>(text.begin(), text.end()));
}

size_t utf8_length(char32_t c) {
    return c < 0x80 ? 1 : c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
}

/** The length of `text` in bytes of UTF-8, by which every limit counts. */
size_t length(const Text & text) {
    size_t bytes = 0;
    for (char32_t c : text) bytes += utf8_length(c);
    return bytes;
}

bool has_words(const Text & text) {
    return std::any_of(text.begin(), text.end(), [](char32_t c) { return !is_white_space(c); });
}

Text strip(const Text & text) {
    size_t begin = 0, end = text.size();
    while (begin < end && is_white_space(text[begin])) begin++;
    while (end > begin && is_white_space(text[end - 1])) end--;
    return text.substr(begin, end - begin);
}

/**
 * The end of a pair of tags such as <whisper>...</whisper> that opens at `at`, or 0 where none does. The name is an
 * ASCII letter and the word characters and hyphens after it, followed by a word boundary; where no closing tag follows,
 * the name is tried shorter, down to its first letter, as a regular expression backtracks.
 */
size_t tag_pair_end(const Text & t, size_t at) {
    const size_t n = t.size();
    if (at + 1 >= n || !is_ascii_letter(t[at + 1])) return 0;
    size_t name_end = at + 2;
    while (name_end < n && (is_word(t[name_end]) || t[name_end] == U'-')) name_end++;
    for (; name_end > at + 1; name_end--) {
        const bool word_before = is_word(t[name_end - 1]), word_after = name_end < n && is_word(t[name_end]);
        if (word_before == word_after) continue;
        const size_t open_end = t.find(U'>', name_end);
        if (open_end == Text::npos) return 0;
        const Text closing = U"</" + t.substr(at + 1, name_end - at - 1) + U">";
        const size_t close = t.find(closing, open_end + 1);
        if (close != Text::npos) return close + closing.size();
    }
    return 0;
}

/**
 * The end of the protected span that starts at `at`, or `at` where none does: a special token <|...|>, a pair of tags
 * or a [bracketed] cue, none of which is ever cut.
 */
size_t span_end(const Text & t, size_t at) {
    const size_t n = t.size();
    if (t[at] == U'<' && at + 1 < n && t[at + 1] == U'|') {
        size_t i = at + 2;
        while (i < n && t[i] != U'|' && t[i] != U'>') i++;
        if (i + 1 < n && t[i] == U'|' && t[i + 1] == U'>') return i + 2;
    }
    if (t[at] == U'<') {
        const size_t end = tag_pair_end(t, at);
        if (end) return end;
    }
    if (t[at] == U'[') {
        size_t i = at + 1;
        while (i < n && t[i] != U'[' && t[i] != U']') i++;
        if (i < n && t[i] == U']') return i + 1;
    }
    return at;
}

/** A `.` between two digits, as in 3.14, is a decimal point and ends no sentence. */
bool is_sentence_end(const Text & t, size_t i) {
    switch (t[i]) {
        case U'.':
            return !(i > 0 && i + 1 < t.size() && is_digit(t[i - 1]) && is_digit(t[i + 1]));
        case U'!':
        case U'?':
        case U';':
        case U'。':
        case U'！':
        case U'？':
        case U'；':
        case U'…':
        case U'\n':
            return true;
        default:
            return false;
    }
}

bool is_clause_mark(const Text & t, size_t i) {
    switch (t[i]) {
        case U',':
        case U':':
        case U'，':
        case U'：':
        case U'、':
            return true;
        default:
            return false;
    }
}

bool is_space(const Text & t, size_t i) {
    return t[i] == U' ';
}

using Mark = bool (*)(const Text &, size_t);

/** The marks a free piece is cut after, in the order they are tried on a piece still longer than the limit. */
const Mark kMarks[] = {is_sentence_end, is_clause_mark, is_space};

/** `piece` cut after each mark, which stays at the end of the part before the cut. */
std::vector<Text> cut_after(const Text & piece, Mark mark) {
    std::vector<Text> parts;
    size_t start = 0;
    for (size_t i = 0; i < piece.size(); i++) {
        if (!mark(piece, i)) continue;
        parts.push_back(piece.substr(start, i + 1 - start));
        start = i + 1;
    }
    if (start < piece.size()) parts.push_back(piece.substr(start));
    return parts;
}

/** `piece` cut into runs of whole code points, each the longest within `max` bytes and at least one code point. */
void cut_into_runs(const Text & piece, size_t max, std::vector<Text> & out) {
    size_t start = 0, bytes = 0;
    for (size_t i = 0; i < piece.size(); i++) {
        const size_t b = utf8_length(piece[i]);
        if (i > start && bytes + b > max) {
            out.push_back(piece.substr(start, i - start));
            start = i;
            bytes = 0;
        }
        bytes += b;
    }
    if (start < piece.size()) out.push_back(piece.substr(start));
}

/**
 * The parts of a free piece: cut after the marks of `stage`, which for sentence ends happens to every piece, and each
 * part still longer than `max` cut by the next stage, the last cutting anywhere.
 */
void cut_free(const Text & piece, size_t max, size_t stage, std::vector<Text> & out) {
    if (stage == std::size(kMarks)) {
        cut_into_runs(piece, max, out);
        return;
    }
    for (Text & part : cut_after(piece, kMarks[stage])) {
        if (length(part) <= max) {
            out.push_back(std::move(part));
        } else {
            cut_free(part, max, stage + 1, out);
        }
    }
}

/** The chunks of a text without speaker tags, each about `max` bytes long. */
std::vector<Text> untagged_chunks(const Text & text, size_t max) {
    if (!has_words(text)) return {};
    if (length(text) <= max) return {text};

    std::vector<Text> pieces;
    size_t free_start = 0, i = 0;
    while (i < text.size()) {
        const size_t end = span_end(text, i);
        if (end == i) {
            i++;
            continue;
        }
        if (i > free_start) cut_free(text.substr(free_start, i - free_start), max, 0, pieces);
        pieces.push_back(text.substr(i, end - i));
        free_start = i = end;
    }
    if (text.size() > free_start) cut_free(text.substr(free_start), max, 0, pieces);

    std::vector<Text> packed;
    Text current;
    size_t current_bytes = 0;
    for (const Text & piece : pieces) {
        const size_t bytes = length(piece);
        if (!current.empty() && current_bytes + bytes > max) {
            packed.push_back(std::move(current));
            current.clear();
            current_bytes = 0;
        }
        current += piece;
        current_bytes += bytes;
    }
    packed.push_back(std::move(current));

    std::vector<Text> chunks;
    for (const Text & chunk : packed) {
        Text stripped = strip(chunk);
        if (!stripped.empty()) chunks.push_back(std::move(stripped));
    }
    return chunks;
}

/** The length of the speaker tag <|speaker:N|> at `at`, N one or more ASCII digits, or 0 where none is. */
size_t speaker_tag_at(const Text & t, size_t at, Text & digits) {
    static const Text open = U"<|speaker:";
    if (t.compare(at, open.size(), open) != 0) return 0;
    size_t i = at + open.size();
    while (i < t.size() && t[i] >= U'0' && t[i] <= U'9') i++;
    if (i == at + open.size() || t.compare(i, 2, U"|>") != 0) return 0;
    digits = t.substr(at + open.size(), i - at - open.size());
    return i + 2 - at;
}

struct Segment {
    Text speaker;
    Text text;
};

/** The segments of `text` at its speaker tags, or nothing when it has no tag. */
std::optional<std::vector<Segment>> segments(const Text & text) {
    std::vector<Segment> found{{U"0", Text()}};
    size_t start = 0, i = 0;
    while (i < text.size()) {
        Text digits;
        const size_t tag = speaker_tag_at(text, i, digits);
        if (!tag) {
            i++;
            continue;
        }
        found.back().text = text.substr(start, i - start);
        found.push_back({std::move(digits), Text()});
        start = i = i + tag;
    }
    if (found.size() == 1) return std::nullopt;
    found.back().text = text.substr(start);
    return found;
}

}  // namespace

std::vector<std::string> split_text(const std::string & text, size_t max_bytes, size_t max_speakers) {
    const Text t = decode(text);
    std::vector<std::string> chunks;

    const std::optional<std::vector<Segment>> segs = segments(t);
    if (!segs) {
        for (const Text & chunk : untagged_chunks(t, max_bytes)) chunks.push_back(encode(chunk));
        return chunks;
    }

    std::vector<Text> items, speakers;
    size_t size = 0;
    auto finish = [&] {
        Text joined;
        for (size_t i = 0; i < items.size(); i++) joined += (i ? U"\n" : U"") + items[i];
        chunks.push_back(encode(joined));
        items.clear();
        speakers.clear();
        size = 0;
    };
    for (const Segment & seg : *segs) {
        if (!has_words(seg.text)) continue;
        for (const Text & piece : untagged_chunks(seg.text, max_bytes)) {
            Text item = U"<|speaker:" + seg.speaker + U"|> " + piece;
            const size_t bytes = length(item);
            const bool new_speaker = std::find(speakers.begin(), speakers.end(), seg.speaker) == speakers.end();
            if (!items.empty() && ((new_speaker && speakers.size() >= max_speakers) || size + bytes > max_bytes)) finish();
            if (std::find(speakers.begin(), speakers.end(), seg.speaker) == speakers.end()) speakers.push_back(seg.speaker);
            items.push_back(std::move(item));
            size += bytes;
        }
    }
    if (!items.empty()) finish();
    return chunks;
}

}  // namespace drama3
