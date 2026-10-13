// Checks how Qwen3-TTS samples against transformers' logits processors: for every case of
// reference/qwen3-tts/sampling_cases.py, the tokens the sampler draws from against the tokens the processors leave,
// their probabilities against the softmax of what the processors leave, and a greedy pick against its argmax. A case
// samples with its own settings, or with the model file's where it applies the official defaults, which are then
// compared with the official ones; the talker's tokens that a frame may not take are the model file's. Draws are not
// compared with torch's: torch.multinomial draws with torch's own generator. The draw is checked on its own: a pick at
// the edges of the cumulative shares, and the shares of many draws.
//
// The sampler computes the penalty and the temperature in float as torch does, and the probabilities in double from
// them, so it parts from the softmax of the processors' scores by the order of a sum alone. It also checks that a
// repetition penalty that takes a logit beyond a float is refused, naming the option, by a greedy pick as by a draw:
// an argmax among infinite logits takes the first of them rather than the largest.
//
// usage: qwen3-tts-sampler-check <qwen3-tts model.gguf> <sampling cases dir>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "args.h"
#include "error.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen3-tts/layout.h"
#include "qwen3-tts/sampler.h"

namespace {

/** The largest difference of a probability that the order of a sum of 3072 terms in double explains, with a margin. */
constexpr double kTolerance = 1e-12;

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

SamplingParams settings_of(const JsonValue & s) {
    SamplingParams p;
    p.greedy = !member(s, "do_sample").boolean;
    p.temperature = std::stof(member(s, "temperature").text);
    p.top_k = std::stoi(member(s, "top_k").text);
    p.top_p = std::stof(member(s, "top_p").text);
    p.repetition_penalty = std::stof(member(s, "repetition_penalty").text);
    return p;
}

bool same_settings(const SamplingParams & a, const SamplingParams & b) {
    return a.greedy == b.greedy && a.temperature == b.temperature && a.top_k == b.top_k && a.top_p == b.top_p &&
           a.repetition_penalty == b.repetition_penalty;
}

/**
 * Logits of 1 and 2, both taken before, under a repetition penalty of 1e-40, a float that divides both beyond the range
 * of a float: the greedy pick and the draw each throw out_of_range naming repetition_penalty.
 */
bool refuses_overflowing_penalty() {
    const std::vector<float> logits = {1.0f, 2.0f};
    const std::vector<int32_t> history = {0, 1};
    std::mt19937_64 rng(0);
    bool ok = true;
    for (const bool greedy : {true, false}) {
        SamplingParams p;
        p.greedy = greedy;
        p.repetition_penalty = 1e-40f;
        const char * how = greedy ? "greedy" : "drawing";
        try {
            const int32_t got = sample(logits, p, history, {}, rng);
            std::printf("%-36s picked %d where the penalty was to be refused: FAIL\n", how, got);
            ok = false;
        } catch (const Error & e) {
            const bool refused = e.fault() == Fault::OutOfRange && e.input() == "repetition_penalty";
            std::printf("%-36s a penalty of 1e-40 refused as out_of_range (%s): %s\n", how, e.input().c_str(), refused ? "ok" : "FAIL");
            ok &= refused;
        }
    }
    return ok;
}

/**
 * pick() at the edges of the cumulative shares of weights 1, 2, 3 and 4, where a draw equal to a share passes to the
 * next token, and the shares of 400000 draws of uniform() from seed 1, which stay within 0.004 of 0.1, 0.2, 0.3 and
 * 0.4, about five standard deviations of a fair draw.
 */
bool picks_by_cumulative_share() {
    const std::vector<double> weights = {1, 2, 3, 4};
    const std::vector<std::pair<double, int>> edges = {{0.0, 0}, {0.0999, 0}, {0.1, 1}, {0.2999, 1}, {0.3000001, 2}, {0.65, 3}, {1.0, 3}};
    bool ok = pick({5}, 1.0) == 0;
    for (const auto & e : edges) ok &= pick(weights, e.first) == e.second;
    std::mt19937_64 rng(1);
    std::vector<int> counts(weights.size(), 0);
    constexpr int kDraws = 400000;
    double worst = 0;
    for (int i = 0; i < kDraws; i++) {
        const double u = uniform(rng);
        ok &= u >= 0 && u <= 1;
        counts[pick(weights, u)]++;
    }
    for (size_t i = 0; i < weights.size(); i++) worst = std::max(worst, std::fabs((double) counts[i] / kDraws - weights[i] / 10));
    ok &= worst <= 0.004;
    std::printf("%-36s edges of the cumulative shares, and shares of %d draws within %.4f: %s\n", "draw", kDraws, worst, ok ? "ok" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <qwen3-tts model.gguf> <sampling cases dir>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], qwen3_tts_layout);
    const Generation generation(model);
    const int32_t eos = (int32_t) model.u32("qwen3-tts.talker.codec_eos_token_id");

    std::vector<std::filesystem::path> cases;
    for (const auto & e : std::filesystem::directory_iterator(std::filesystem::u8path(args[2]))) {
        if (e.is_directory() && std::filesystem::exists(e.path() / "meta.json")) cases.push_back(e.path());
    }
    std::sort(cases.begin(), cases.end());
    if (cases.empty()) {
        std::fprintf(stderr, "no case of sampling_cases.py is under %s\n", args[2].c_str());
        return 1;
    }

    bool ok = true;
    for (const auto & dir : cases) {
        const JsonValue meta = read_meta(dir);
        const bool talker = member(meta, "stack").text == "talker";
        const SamplingParams & file = talker ? generation.talker : generation.code_predictor;
        SamplingParams p = file;
        const JsonValue * applies = meta.member("settings");
        if (applies && applies->text == "file") {
            if (!same_settings(file, settings_of(member(meta, "official")))) {
                std::printf("%-36s FAIL: the model file's settings differ from the official defaults\n", dir.filename().u8string().c_str());
                ok = false;
                continue;
            }
        } else {
            p = settings_of(meta);
        }
        const Npy logits = read_npy((dir / "logits.npy").u8string());
        const Npy history = read_npy((dir / "history.npy").u8string());
        const Npy scores = read_npy((dir / "scores.npy").u8string());
        const int rows = (int) logits.shape[0], vocab = (int) logits.shape[1], longest = (int) history.shape[1];

        int support_differs = 0, argmax_differs = 0;
        double worst = 0;
        size_t kept = 0;
        std::mt19937_64 rng(0);
        for (int r = 0; r < rows; r++) {
            const std::vector<float> row(logits.f32.begin() + (size_t) r * vocab, logits.f32.begin() + (size_t) (r + 1) * vocab);
            const float * want = &scores.f32[(size_t) r * vocab];
            std::vector<int32_t> seen;
            for (int i = 0; i < longest && history.i32[(size_t) r * longest + i] >= 0; i++) seen.push_back(history.i32[(size_t) r * longest + i]);
            const std::vector<bool> banned = talker ? generation.banned(vocab, eos, (int) seen.size()) : std::vector<bool>();
            if (p.greedy) {
                const int32_t got = sample(row, p, seen, banned, rng);
                argmax_differs += got != (int32_t) (std::max_element(want, want + vocab) - want);
                continue;
            }
            const Candidates c = candidates(row, p, seen, banned);
            std::vector<double> got(vocab, 0.0);
            double sum = 0;
            for (double w : c.weights) sum += w;
            for (size_t i = 0; i < c.tokens.size(); i++) got[c.tokens[i]] = c.weights[i] / sum;
            const float top = *std::max_element(want, want + vocab);
            double total = 0;
            for (int i = 0; i < vocab; i++) total += want[i] == -INFINITY ? 0 : std::exp((double) want[i] - top);
            bool same_support = true;
            for (int i = 0; i < vocab; i++) {
                const bool in_want = want[i] != -INFINITY;
                same_support &= in_want == (got[i] > 0);
                const double expected = in_want ? std::exp((double) want[i] - top) / total : 0;
                worst = std::max(worst, std::fabs(got[i] - expected));
            }
            support_differs += !same_support;
            kept += c.tokens.size();
        }
        const bool pass = support_differs == 0 && argmax_differs == 0 && worst <= kTolerance;
        if (p.greedy) {
            std::printf("%-36s %4d rows, greedy, argmax differs in %d: %s\n", dir.filename().u8string().c_str(), rows, argmax_differs,
                        pass ? "ok" : "FAIL");
        } else {
            std::printf("%-36s %4d rows, %6.1f tokens kept on average, tokens kept differ in %d, largest difference of a probability %.1e: %s\n",
                        dir.filename().u8string().c_str(), rows, (double) kept / rows, support_differs, worst, pass ? "ok" : "FAIL");
        }
        ok &= pass;
    }
    ok &= refuses_overflowing_penalty();
    ok &= picks_by_cumulative_share();
    return ok ? 0 : 1;
}
