#include "spelling.h"

#include <algorithm>

namespace {

/** The edits between `a` and `b`, a swap of two neighbours counted as one (the optimal string alignment distance). */
size_t edits(const std::string & a, const std::string & b) {
    std::vector<std::vector<size_t>> d(a.size() + 1, std::vector<size_t>(b.size() + 1));
    for (size_t i = 0; i <= a.size(); i++) d[i][0] = i;
    for (size_t j = 0; j <= b.size(); j++) d[0][j] = j;
    for (size_t i = 1; i <= a.size(); i++) {
        for (size_t j = 1; j <= b.size(); j++) {
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1)});
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
        }
    }
    return d[a.size()][b.size()];
}

/** The word without the dashes of a flag, which every flag shares and so tell nothing apart. */
std::string bare(const std::string & word) {
    const size_t first = word.find_first_not_of('-');
    return first == std::string::npos ? "" : word.substr(first);
}

}  // namespace

std::string nearest(const std::string & word, const std::vector<std::string> & candidates) {
    const std::string typed = bare(word);
    const size_t most = typed.size() <= 4 ? 1 : 2;
    std::string best;
    size_t best_edits = most + 1;
    bool tied = false;
    for (const std::string & c : candidates) {
        const size_t e = edits(typed, bare(c));
        if (e < best_edits) {
            best = c;
            best_edits = e;
            tied = false;
        } else if (e == best_edits && c != best) {
            tied = true;
        }
    }
    return best_edits <= most && !tied ? best : "";
}
