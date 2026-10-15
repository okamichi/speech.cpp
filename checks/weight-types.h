#pragma once

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "model-file.h"

/**
 * Whether the model file at `path` holds weights of one of `types`, those whose arithmetic the check's bounds explain,
 * read from its metadata alone. For another type it prints which types the check takes and why, and the check exits
 * with a usage error, as for arguments it cannot take: a check that failed there would point to the weights and not
 * to a defect.
 */
inline bool takes_weights(const std::string & path, const Layout & layout, const std::vector<std::string> & types, const char * why) {
    const std::string held = read_identity(ModelFile(path, layout)).weight_type;
    if (std::find(types.begin(), types.end(), held) != types.end()) return true;
    std::string names;
    for (size_t i = 0; i < types.size(); i++) names += (i == 0 ? "" : i + 1 == types.size() ? " or " : ", ") + types[i];
    std::fprintf(stderr, "%s holds %s weights; the check takes %s weights: %s\n", path.c_str(), held.c_str(), names.c_str(), why);
    return false;
}
