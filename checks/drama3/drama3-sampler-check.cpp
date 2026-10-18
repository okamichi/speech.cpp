// Checks how Drama 3 samples against fish-speech's sample_top_k_top_p(): for every case of
// reference/drama3/sampling_cases.py, the probability the sampler gives each token against the one the official sampler
// draws it with, 0 where either drops it. A case samples with its own settings, or with the model file's, the model
// card's or the redraw's, which are then compared with the official ones. Draws are not compared with torch's, which
// draws with its own generator.
//
// The official sampler accumulates the probabilities in float32 and this one in double. Where a token's cumulative share
// lies within rounding of top_p, one keeps it and the other drops it; with a top_p of 1, the float32 sum passes 1 among
// the least likely tokens. A token kept by one alone passes when its share is that close to top_p or its probability is
// below 1e-6.
//
// It also checks which draw the main token takes: the first, unless it is a semantic id among the last main tokens of
// the window, and the redraw then.
//
// usage: drama3-sampler-check <drama3 model.gguf> <sampling cases dir>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "args.h"
#include "drama3/layout.h"
#include "drama3/sampler.h"
#include "json-reader.h"
#include "npy.h"

using namespace drama3;

namespace {

/** How close to top_p a cumulative share may lie for the float32 sum and the double one to part over the token. */
constexpr double kShareTolerance = 1e-6;
/** The probability below which a token kept by one sampler alone changes no draw that matters. */
constexpr double kNegligible = 1e-6;
/** The largest difference of a probability that float32's softmax against double's explains, with a margin. */
constexpr double kTolerance = 1e-5;

JsonValue read_meta(const std::filesystem::path & dir) {
    std::ifstream f(dir / "meta.json");
    if (!f) throw std::runtime_error("cannot open " + (dir / "meta.json").u8string());
    return parse_json(std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
}

const JsonValue & member(const JsonValue & object, const std::string & name) {
    const JsonValue * v = object.member(name);
    if (!v) throw std::runtime_error("meta.json has no " + name);
    return *v;
}

Sampling settings_of(const JsonValue & s) {
    Sampling p;
    p.temperature = std::stof(member(s, "temperature").text);
    p.top_k = std::stoi(member(s, "top_k").text);
    p.top_p = std::stof(member(s, "top_p").text);
    return p;
}

/** The cumulative share of each token by falling logit over the softmax of every logit, in double, by token. */
std::vector<double> shares(const std::vector<float> & logits, const Sampling & s) {
    const float t = std::max(s.temperature, 1e-5f);
    std::vector<int> order(logits.size());
    for (size_t i = 0; i < order.size(); i++) order[i] = (int) i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
    const double top = logits[(size_t) order[0]] / t;
    double sum = 0;
    for (float l : logits) sum += std::exp(l / t - top);
    std::vector<double> share(logits.size());
    double cumulative = 0;
    for (int i : order) share[(size_t) i] = cumulative += std::exp(logits[(size_t) i] / t - top) / sum;
    return share;
}

struct Result {
    double max_diff = 0;
    int parted = 0, rows = 0;
    bool ok = true;
};

Result check_case(const std::filesystem::path & dir, const Sampling & s) {
    const Npy logits = read_npy((dir / "logits.npy").u8string());
    const Npy official = read_npy((dir / "probabilities.npy").u8string());
    const size_t rows = (size_t) logits.shape[0], n = (size_t) logits.shape[1];
    Result r;
    r.rows = (int) rows;
    for (size_t row = 0; row < rows; row++) {
        const std::vector<float> l(logits.f32.begin() + (ptrdiff_t) (row * n), logits.f32.begin() + (ptrdiff_t) ((row + 1) * n));
        const Candidates c = candidates(l, s);
        double kept = 0;
        for (double w : c.weights) kept += w;
        std::vector<double> mine(n, 0.0);
        for (size_t i = 0; i < c.tokens.size(); i++) mine[(size_t) c.tokens[i]] = c.weights[i] / kept;
        const std::vector<double> share = shares(l, s);
        for (size_t i = 0; i < n; i++) {
            const double theirs = official.f32[row * n + i];
            if ((mine[i] > 0) != (theirs > 0)) {
                r.parted++;
                const bool explained = std::fabs(share[i] - s.top_p) <= kShareTolerance || std::max(mine[i], theirs) < kNegligible;
                if (!explained) {
                    std::printf("  row %zu token %zu: kept by %s alone with probability %.3e at a cumulative share of %.9f\n", row, i,
                                mine[i] > 0 ? "this sampler" : "the official one", std::max(mine[i], theirs), share[i]);
                    r.ok = false;
                }
            }
            r.max_diff = std::max(r.max_diff, std::fabs(mine[i] - theirs));
        }
    }
    r.ok = r.ok && r.max_diff <= kTolerance;
    return r;
}

/**
 * The main token's draws on the first row of `logits`: with a first draw of 0, the most likely token, and a redraw of
 * almost 1, the redraw's least likely kept token, the main token is the first draw unless the history's last window
 * holds it; and an end id, made the most likely, is never drawn again.
 */
bool check_redraw(const ModelFile & m, const std::filesystem::path & dir) {
    const Npy logits = read_npy((dir / "logits.npy").u8string());
    std::vector<float> l(logits.f32.begin(), logits.f32.begin() + (ptrdiff_t) logits.shape[1]);
    const FrameSampler sampler(m, file_sampling(m));
    const std::vector<int32_t> ids = m.i32_array("drama3.output.token_ids");
    const int window = (int) m.u32("drama3.repeat.window");
    const int32_t first = sampler.main(l, {}, 0.0, 0.0);
    const int32_t again = sampler.main(l, {first}, 0.0, 1.0 - 1e-12);
    std::vector<int32_t> old = {first};
    old.insert(old.end(), (size_t) window, ids[1] == first ? ids[2] : ids[1]);
    bool ok = true;
    const auto expect = [&](const char * what, bool pass) {
        std::printf("%-58s %s\n", what, pass ? "ok" : "FAIL");
        ok = ok && pass;
    };
    expect("a draw not in the window is kept", sampler.semantic(first) && sampler.main(l, old, 0.0, 1.0 - 1e-12) == first);
    expect("a draw in the window is drawn again", again != first && sampler.semantic(again));
    const int32_t eos = (int32_t) m.u32("drama3.eos_token_id");
    l[0] = *std::max_element(l.begin(), l.end()) + 50.0f;
    expect("an end id in the window is not drawn again", ids[0] == eos && sampler.main(l, {eos}, 0.0, 1.0 - 1e-12) == eos);
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <drama3 model.gguf> <sampling cases dir>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], model_layout);
    const std::filesystem::path root = std::filesystem::u8path(args[2]);
    std::vector<std::filesystem::path> cases;
    for (const auto & e : std::filesystem::directory_iterator(root)) {
        if (std::filesystem::is_regular_file(e.path() / "meta.json")) cases.push_back(e.path());
    }
    std::sort(cases.begin(), cases.end());
    if (cases.empty()) {
        std::fprintf(stderr, "no case of reference/drama3/sampling_cases.py is under %s\n", args[2].c_str());
        return 2;
    }
    bool ok = true;
    for (const auto & dir : cases) {
        const JsonValue meta = read_meta(dir);
        const JsonValue & settings = member(meta, "settings");
        Sampling s;
        if (settings.kind == JsonValue::Kind::String) {
            const std::string prefix = settings.text == "file" ? "drama3.sampling." : "drama3.repeat.";
            s.temperature = model.f32(prefix + "temperature");
            s.top_k = (int) model.u32(prefix + "top_k");
            s.top_p = model.f32(prefix + "top_p");
            const Sampling o = settings_of(member(meta, "official"));
            const bool same = s.temperature == o.temperature && s.top_k == o.top_k && s.top_p == o.top_p;
            std::printf("%-26s the file's %s settings against the official ones: %s\n", dir.filename().u8string().c_str(),
                        settings.text.c_str(), same ? "ok" : "FAIL");
            ok = ok && same;
        } else {
            s = settings_of(settings);
        }
        const Result r = check_case(dir, s);
        std::printf("%-26s temperature %.2f top-k %4d top-p %.2f: %3d rows, max |diff| %.2e, %d tokens parted: %s\n",
                    dir.filename().u8string().c_str(), s.temperature, s.top_k, s.top_p, r.rows, r.max_diff, r.parted, r.ok ? "ok" : "FAIL");
        ok = ok && r.ok;
    }
    ok = check_redraw(model, root / "main-file") && ok;
    return ok ? 0 : 1;
}
