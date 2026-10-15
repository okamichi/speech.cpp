#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "model-file.h"

namespace irodori {

/**
 * The text encoder's SentencePiece tokenizer, as Hugging Face tokenizers runs its tokenizer.json: the added tokens are
 * split out of the text first, spaces become "▁", and the rest is cut into pieces by the model the file names, a
 * character no piece covers becoming its UTF-8 bytes. ModernBERT-ja's is a Unigram model, whose Viterbi path over the
 * pieces' scores picks the pieces; Gemma's is a BPE model, which merges the pair of the lowest rank, the leftmost first,
 * until no pair merges.
 */
class Tokenizer {
public:
    explicit Tokenizer(const ModelFile & m);

    /** The ids of normalized text, after <s>, as the runtime's batch_encode() gives them. */
    std::vector<int32_t> encode(const std::string & text) const;

    /** Encodes a stripped caption, enforcing its token limit or explicitly using the runtime's truncation. */
    std::vector<int32_t> encode_caption(const std::string & text, size_t limit, bool compatible, bool & truncated) const;

private:
    void encode_unigram(const std::string & piece, std::vector<int32_t> & ids) const;
    void encode_bpe(const std::string & piece, std::vector<int32_t> & ids) const;

    /** What a merge of two pieces gives: the merge's rank among the merges and the merged piece. */
    struct Merge {
        int32_t rank;
        int32_t id;
    };

    bool bpe_ = false;
    std::vector<std::string> tokens_;
    std::vector<double> scores_;
    std::unordered_map<uint64_t, Merge> merges_;
    std::unordered_map<std::string, int32_t> ids_;
    /** The added tokens by their first byte. */
    std::vector<int32_t> added_[256];
    size_t longest_ = 0;
    double unknown_score_ = 0;
    int32_t bos_ = 0, unknown_ = 0;
    int32_t bytes_[256];
};

}  // namespace irodori
