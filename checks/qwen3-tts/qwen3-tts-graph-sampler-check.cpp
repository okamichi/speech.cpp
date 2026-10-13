// Checks the picks GraphSampler builds into a graph against sample() on the CPU, on the code predictor's logits of the
// cases of reference/qwen3-tts/sampling_cases.py that a graph follows, with each case's settings or the model file's,
// and on the first case's logits with a top-k of 1 and of the whole vocabulary. For every row it draws 64 numbers, half
// spread evenly over [0, 1] and half from uniform(), and compares the device's pick of each with sample()'s with the same
// number: the two must agree where the number lies further than kMargin from every boundary between the cumulative
// shares of two candidates, the float weights of the graph parting from candidates()' double ones by their rounding, or
// the two picks have equal logits, which either sort may put in either order. It
// then draws 25,600 numbers from uniform() for the first row of each case, and the share of each candidate among the
// device's picks must lie within five standard deviations of a fair draw of its probability. A temperature that takes a
// logit beyond a float must be refused as candidates() refuses it, naming the temperature; in a Debug build, where ggml
// asserts, a softmax that reads the infinite logit aborts instead.
//
// usage: qwen3-tts-graph-sampler-check <qwen3-tts model.gguf> <sampling cases dir> [gpu|cpu]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "error.h"
#include "json-reader.h"
#include "npy.h"
#include "qwen3-tts/graph-sampler.h"
#include "qwen3-tts/layout.h"
#include "qwen3-tts/sampler.h"

namespace {

constexpr int kPicks = 64;
constexpr int kDistributionComputes = 400;

/**
 * The distance from a boundary between two candidates' cumulative shares within which the device may pick the other:
 * a float cumulative sum of 2048 weights rounds by 1.2e-4 at most, and by about 3e-6 in a typical row.
 */
constexpr double kMargin = 1e-4;

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

/** A graph of kPicks picks under one setting from one row of logits, which each compute gives anew. */
class Picks {
public:
    Picks(ggml_backend_t backend, const SamplingParams & p, int vocab)
        : backend_(backend), allocr_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend))), graph_(4096) {
        logits_ = graph_.input(std::vector<float>((size_t) vocab, 0.0f), vocab);
        sampler_ = std::make_unique<GraphSampler>(graph_, p, vocab, kPicks);
        for (int i = 0; i < kPicks; i++) sampler_->pick(graph_, i, logits_);
    }
    ~Picks() { ggml_gallocr_free(allocr_); }

    std::vector<int32_t> run(const std::vector<float> & row, const std::vector<double> & u) {
        graph_.set(logits_, row);
        sampler_->draw(graph_, sampler_->params().greedy ? std::vector<double>() : u);
        if (allocated_) graph_.compute_again(backend_);
        else graph_.compute(backend_, allocr_);
        allocated_ = true;
        return sampler_->read();
    }

private:
    ggml_backend_t backend_;
    ggml_gallocr_t allocr_;
    Graph graph_;
    ggml_tensor * logits_ = nullptr;
    std::unique_ptr<GraphSampler> sampler_;
    bool allocated_ = false;
};

/** The distance from `u` to the nearest boundary between the cumulative shares of the candidates of `c`. */
double distance_to_boundary(const Candidates & c, double u) {
    double sum = 0, cumulative = 0, nearest = INFINITY;
    for (double w : c.weights) sum += w;
    for (size_t i = 0; i + 1 < c.weights.size(); i++) {
        cumulative += c.weights[i] / sum;
        nearest = std::min(nearest, std::fabs(cumulative - u));
    }
    return nearest;
}

/** Compares the device's picks with sample()'s on every row of `logits` under `p`, and their shares on the first. */
bool check_case(ggml_backend_t backend, const std::string & name, const SamplingParams & p, const Npy & logits) {
    const int rows = (int) logits.shape[0], vocab = (int) logits.shape[1];
    Picks picks(backend, p, vocab);
    std::mt19937_64 rng(1);
    int away = 0, near = 0, tied = 0, total = 0;
    for (int r = 0; r < rows; r++) {
        const std::vector<float> row(logits.f32.begin() + (size_t) r * vocab, logits.f32.begin() + (size_t) (r + 1) * vocab);
        std::vector<double> u(kPicks);
        for (int i = 0; i < kPicks; i++) u[(size_t) i] = i < kPicks / 2 ? (i + 0.5) / (kPicks / 2) : uniform(rng);
        const std::vector<int32_t> got = picks.run(row, u);
        const Candidates c = p.greedy ? Candidates() : candidates(row, p, {}, {});
        for (int i = 0; i < kPicks; i++) {
            const int32_t want = sample(row, p, {}, {}, u[(size_t) i]);
            total++;
            if (got[(size_t) i] == want) continue;
            // Candidates of equal logits may come in either order, in candidates() as in the graph's sort.
            if (row[(size_t) got[(size_t) i]] == row[(size_t) want]) tied++;
            else if (!p.greedy && distance_to_boundary(c, u[(size_t) i]) < kMargin) near++;
            else away++;
        }
    }

    // The shares of the device's picks on the first row, against the candidates' probabilities.
    const std::vector<float> first(logits.f32.begin(), logits.f32.begin() + vocab);
    double worst = 0;
    if (!p.greedy) {
        const Candidates c = candidates(first, p, {}, {});
        std::vector<int> counts((size_t) vocab, 0);
        for (int n = 0; n < kDistributionComputes; n++) {
            std::vector<double> u(kPicks);
            for (double & x : u) x = uniform(rng);
            for (int32_t id : picks.run(first, u)) counts[(size_t) id]++;
        }
        const double draws = (double) kDistributionComputes * kPicks;
        double sum = 0;
        for (double w : c.weights) sum += w;
        std::vector<double> probability((size_t) vocab, 0.0);
        for (size_t i = 0; i < c.tokens.size(); i++) probability[(size_t) c.tokens[i]] = c.weights[i] / sum;
        for (int t = 0; t < vocab; t++) {
            const double q = probability[(size_t) t];
            // A pick of a token that is no candidate is a defect however rare.
            const double deviation = q == 0 && counts[(size_t) t] > 0 ? INFINITY
                                                                     : std::fabs(counts[(size_t) t] / draws - q) / (std::sqrt(q * (1 - q) / draws) + 1 / draws);
            worst = std::max(worst, deviation);
        }
    }
    const bool pass = away == 0 && worst <= 5;
    std::printf("%-36s %6d picks, %d others of an equal logit, %d others within %.0e of a boundary and %d further; shares within %.1f "
                "deviations: %s\n",
                name.c_str(), total, tied, near, kMargin, away, worst, pass ? "ok" : "FAIL");
    return pass;
}

/** A temperature of 1e-38 takes a logit of the first row beyond a float: the device refuses it as candidates() does. */
bool refuses_overflowing_temperature(ggml_backend_t backend, const Npy & logits) {
    const int vocab = (int) logits.shape[1];
    SamplingParams p;
    p.temperature = 1e-38f;
    p.top_k = 50;
    Picks picks(backend, p, vocab);
    const std::vector<float> first(logits.f32.begin(), logits.f32.begin() + vocab);
    bool ok = true;
    try {
        const std::vector<int32_t> got = picks.run(first, std::vector<double>(kPicks, 0.5));
        std::printf("%-36s picked %d where the temperature was to be refused: FAIL\n", "temperature 1e-38", got[0]);
        ok = false;
    } catch (const Error & e) {
        ok = e.fault() == Fault::OutOfRange && e.input() == "temperature";
        std::printf("%-36s refused as out_of_range (%s): %s\n", "temperature 1e-38", e.input().c_str(), ok ? "ok" : "FAIL");
    }
    try {
        sample(first, p, {}, {}, 0.5);
        std::printf("%-36s the CPU picked where it was to refuse: FAIL\n", "temperature 1e-38");
        ok = false;
    } catch (const Error &) {
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <qwen3-tts model.gguf> <sampling cases dir> [gpu|cpu]\n", args[0].c_str());
        return 2;
    }
    ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
    std::printf("backend: %s\n", ggml_backend_name(backend));
    const ModelFile model(args[1], qwen3_tts_layout);
    const Generation generation(model);

    std::vector<std::filesystem::path> cases;
    for (const auto & e : std::filesystem::directory_iterator(std::filesystem::u8path(args[2]))) {
        if (e.is_directory() && std::filesystem::exists(e.path() / "meta.json")) cases.push_back(e.path());
    }
    std::sort(cases.begin(), cases.end());

    bool ok = true;
    const Npy * first_logits = nullptr;
    std::vector<Npy> held;
    held.reserve(cases.size());
    for (const auto & dir : cases) {
        const JsonValue meta = read_meta(dir);
        if (member(meta, "stack").text != "code_predictor") continue;
        const JsonValue * applies = meta.member("settings");
        const SamplingParams p = applies && applies->text == "file" ? generation.code_predictor : settings_of(meta);
        const std::string name = dir.filename().u8string();
        if (!GraphSampler::follows(p)) {
            std::printf("%-36s picks step by step, as a graph does not follow its settings\n", name.c_str());
            continue;
        }
        held.push_back(read_npy((dir / "logits.npy").u8string()));
        if (!first_logits) first_logits = &held.back();
        ok &= check_case(backend, name, p, held.back());
    }
    if (!first_logits) {
        std::fprintf(stderr, "no case of the code predictor that a graph follows is under %s\n", args[2].c_str());
        return 1;
    }
    for (const int top_k : {1, 0}) {
        SamplingParams p = generation.code_predictor;
        p.top_k = top_k;
        ok &= check_case(backend, "the first case with top-k " + std::to_string(top_k), p, *first_logits);
    }
    ok &= refuses_overflowing_temperature(backend, *first_logits);
    ggml_backend_free(backend);
    std::printf("%s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
