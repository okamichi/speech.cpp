#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "model-file.h"
#include "qwen2-tokenizer.h"

/**
 * The added tokens of a tokenizer, which the tokenizers library finds in a text before anything else runs on it: at each
 * place the leftmost and, of those that start there, the longest, in the text as it is given, so that a text is split at
 * every added token, special or not, and the text between them goes through the normalizer, the pre-tokenizer and the
 * BPE. "<|im_end|>" followed by U+0338 stays the added token and U+0338, where the NFC of the whole would join '>' and
 * U+0338 into U+226F. It takes added tokens that strip no space, match anywhere and match the text as given, which is
 * what the converters check.
 */
class AddedVocabulary {
public:
    /** The tokens of `<prefix>.added_ids` among `<prefix>.tokens`, every token in the order of its id. */
    AddedVocabulary(const ModelFile & m, const std::string & prefix);

    /** The ids of `text`, split at its added tokens and the rest encoded by `bpe`. */
    std::vector<int32_t> encode(const std::string & text, const Qwen2Tokenizer & bpe) const;

    /** The id of the added token whose text is `text`; any other text throws. */
    int32_t id(const std::string & text) const;

private:
    /** The added tokens' texts and ids by their first byte, the longest first. */
    std::vector<std::vector<std::pair<std::string, int32_t>>> by_first_byte_;
};
