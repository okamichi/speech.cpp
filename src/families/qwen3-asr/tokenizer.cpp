#include "tokenizer.h"

namespace qwen3_asr {

Tokenizer::Tokenizer(const ModelFile & m) : bpe_(m, "qwen3-asr.tokenizer"), added_(m, "qwen3-asr.tokenizer") {
    special_.assign(m.str_array("qwen3-asr.tokenizer.tokens").size(), false);
    for (int32_t id : m.i32_array("qwen3-asr.tokenizer.special_ids")) special_[(size_t) id] = true;
}

std::vector<int32_t> Tokenizer::encode(const std::string & text) const {
    return added_.encode(text, bpe_);
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids) const {
    std::vector<int32_t> kept;
    for (int32_t id : ids) {
        if (!special(id)) kept.push_back(id);
    }
    return bpe_.decode(kept);
}

bool Tokenizer::special(int32_t id) const {
    return id >= 0 && (size_t) id < special_.size() && special_[(size_t) id];
}

int32_t Tokenizer::added_id(const std::string & text) const {
    return added_.id(text);
}

}  // namespace qwen3_asr
