#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model-file.h"

/**
 * The Qwen2 byte-level BPE (transformers' Qwen2Tokenizer) with the pre-tokenizer of the tokenizer.json that ships with
 * the model, Qwen2's or Qwen3.5's, whose tokens and merges a model file holds under one prefix.
 *
 * transformers 4.57.3, which the official Qwen3-TTS package pins, replaces that pre-tokenizer with Mistral's when
 * the model is loaded with fix_mistral_regex=True, as the package does; the two split text of Latin
 * letters in mixed case, contractions and '/' differently. This follows tokenizer.json, which is what
 * the model was trained with. Its normalizer brings the text to NFC before the pre-tokenizer splits it, by the tables
 * of Unicode 9.0 that the tokenizers library has, and so does this.
 */
/** The pre-tokenizers of the Qwen tokenizers, by the regex of their tokenizer.json. */
enum class Qwen2Pattern {
    /** (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+ */
    Qwen2,
    /**
     * Qwen3.5's, which takes marks as letters: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|
     * ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
     */
    Qwen35,
};

/** Whether `regex`, the pattern of a tokenizer.json's Split pre-tokenizer, is one of Qwen2Pattern's, and which. */
bool qwen2_pattern(const std::string & regex, Qwen2Pattern * pattern);

class Qwen2Tokenizer {
public:
    /**
     * The tokenizer of the keys `<prefix>.tokens`, every token in the order of its id, and `<prefix>.merges`, with the
     * pre-tokenizer `pattern`.
     */
    Qwen2Tokenizer(const ModelFile & m, const std::string & prefix, Qwen2Pattern pattern = Qwen2Pattern::Qwen2);

    /**
     * The ids of `text`, with special tokens written in the text taken literally. A text that is not UTF-8 throws an
     * Error that names no input, which the caller names.
     */
    std::vector<int32_t> encode(const std::string & text) const;

    /**
     * The text of `ids` as the ByteLevel decoder of tokenizer.json gives it, special tokens included: each token's
     * characters turned back into the bytes they stand for, or the token's own UTF-8 when one of its characters stands
     * for none, the bytes of all the tokens joined, and every sequence of them that is not UTF-8 replaced by U+FFFD
     * as Rust's String::from_utf8_lossy does, so that a character split between two tokens comes out whole and one cut
     * off by the end comes out as U+FFFD. An id with no token decodes to nothing, as the tokenizers library drops it.
     */
    std::string decode(const std::vector<int32_t> & ids) const;

    /** The pieces the pre-tokenizer splits the NFC of `text` into, as UTF-8. */
    std::vector<std::string> pre_tokenize(const std::string & text) const;

private:
    std::vector<int32_t> bpe(const std::string & piece) const;

    Qwen2Pattern pattern_;
    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> vocab_;
    std::map<std::pair<std::string, std::string>, int> ranks_;
    std::string byte_to_unicode_[256];
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_;
};
