#include "added-vocabulary.h"

#include <algorithm>
#include <stdexcept>

AddedVocabulary::AddedVocabulary(const ModelFile & m, const std::string & prefix) : by_first_byte_(256) {
    const std::vector<std::string> tokens = m.str_array(prefix + ".tokens");
    for (int32_t id : m.i32_array(prefix + ".added_ids")) {
        const std::string & text = tokens[(size_t) id];
        by_first_byte_[(unsigned char) text[0]].push_back({text, id});
    }
    for (auto & same : by_first_byte_) {
        std::stable_sort(same.begin(), same.end(), [](const auto & a, const auto & b) { return a.first.size() > b.first.size(); });
    }
}

std::vector<int32_t> AddedVocabulary::encode(const std::string & text, const Qwen2Tokenizer & bpe) const {
    std::vector<int32_t> ids;
    const auto encode_between = [&](size_t from, size_t to) {
        if (to == from) return;
        const std::vector<int32_t> piece = bpe.encode(text.substr(from, to - from));
        ids.insert(ids.end(), piece.begin(), piece.end());
    };
    size_t last = 0;
    for (size_t i = 0; i < text.size();) {
        const auto & same = by_first_byte_[(unsigned char) text[i]];
        const auto match = std::find_if(same.begin(), same.end(), [&](const auto & a) { return text.compare(i, a.first.size(), a.first) == 0; });
        if (match == same.end()) {
            i++;
            continue;
        }
        encode_between(last, i);
        ids.push_back(match->second);
        i += match->first.size();
        last = i;
    }
    encode_between(last, text.size());
    return ids;
}

int32_t AddedVocabulary::id(const std::string & text) const {
    if (!text.empty()) {
        for (const auto & [token, id] : by_first_byte_[(unsigned char) text[0]]) {
            if (token == text) return id;
        }
    }
    throw std::logic_error(text + " is not an added token of the tokenizer, which the layout checks");
}
