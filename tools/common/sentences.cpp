#include "sentences.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

#include "unicode.h"

namespace {

/**
 * A model whose request speaks less than this, in seconds, speaks a text a sentence at a time: Irodori-TTS speaks 30 s,
 * which a few sentences pass, and Qwen3-TTS 655 s, which a paragraph does not.
 */
constexpr double kShortRequest = 60;

/**
 * The silence a join after a sentence holds at least, in seconds. Two Irodori-TTS sentences spoken alone and joined
 * leave 0.15 s between them in the voice none and 0.6 to 0.67 s in a voice of a reference, which carries the
 * reference's own silences, where one request of six sentences pauses 1.04 to 1.12 s and 0.79 to 0.85 s between them
 * (both v4.1 models, seeds 1 to 3, silence at or under kSilent, Apple M5, 2026-10-08). A fixed pause would leave one
 * of the voices 0.7 s off, so the join is topped up to a pause between the two.
 */
constexpr double kSentencePause = 0.9;

/** The largest magnitude of a sample counted as silence, -40 dBFS. */
constexpr float kSilent = 0.01f;

/** The code point at `at` of `text` and its length in bytes; a byte that begins no valid sequence stands for itself. */
std::pair<uint32_t, size_t> code_point(const std::string & text, size_t at) {
    const auto b = (unsigned char) text[at];
    const size_t n = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : (b >> 3) == 30 ? 4 : 0;
    if (n == 0 || at + n > text.size()) return {b, 1};
    uint32_t c = n == 1 ? b : b & (0x7F >> n);
    for (size_t i = 1; i < n; i++) {
        const auto x = (unsigned char) text[at + i];
        if ((x >> 6) != 2) return {b, 1};
        c = (c << 6) | (x & 0x3F);
    }
    return {c, n};
}

/** A closing bracket or quotation mark, which stays with the sentence or the clause before it. */
bool closes(uint32_t c) {
    static const uint32_t marks[] = {0x300D, 0x300F, 0xFF09, 0x3011, 0x3015, 0x3009, 0x300B, 0xFF3D, 0xFF5D, 0xFF63,
                                     0x201D, 0x2019, '"', '\'', ')', ']', '}'};
    return std::find(std::begin(marks), std::end(marks), c) != std::end(marks);
}

/** A mark that ends a sentence or a clause wherever it stands: the full-width and half-width forms. */
bool ends_anywhere(uint32_t c, Cut cut) {
    if (cut == Cut::Sentences) return c == 0x3002 || c == 0xFF61 || c == 0xFF01 || c == 0xFF1F || c == 0xFF0E;
    return c == 0x3001 || c == 0xFF64 || c == 0xFF0C || c == 0xFF1B;
}

/** A mark that ends a sentence or a clause where a space or the end follows it, so that 3.14 and 1,000 stay whole. */
bool ends_before_space(uint32_t c, Cut cut) {
    return cut == Cut::Sentences ? c == '.' || c == '!' || c == '?' : c == ',' || c == ';';
}

std::string trimmed(const std::string & text, size_t begin, size_t end) {
    while (begin < end) {
        const auto [c, n] = code_point(text, begin);
        if (!python_space(c)) break;
        begin += n;
    }
    while (end > begin) {
        size_t start = end - 1;
        while (start > begin && ((unsigned char) text[start] >> 6) == 2) start--;
        if (!python_space(code_point(text, start).first)) break;
        end = start;
    }
    return text.substr(begin, end - begin);
}

/** The first characters of `text` for a message, with an ellipsis where it goes on. */
std::string excerpt(const std::string & text) {
    size_t at = 0;
    for (int i = 0; i < 24 && at < text.size(); i++) at += code_point(text, at).second;
    return at < text.size() ? text.substr(0, at) + "…" : text;
}

/**
 * Passes the audio of the requests on, and gives the join after a sentence at least kSentencePause of silence: the
 * silence after the last sound passed on, and the silence a request begins with, which it holds back until the
 * request's first sound comes, topped up with zeros.
 */
class Joiner {
public:
    Joiner(speech_audio_callback out, void * user_data, int rate, uint64_t limit)
        : out_(out), user_data_(user_data), pause_((size_t) std::lround(kSentencePause * rate)), limit_(limit) {}

    /** The audio callback of every request, with the joiner as its user data. */
    static int receive(const float * s, size_t n, void * self) { return static_cast<Joiner *>(self)->take(s, n); }

    /** Begins a request, whose audio received() then counts. */
    void begin() { received_ = 0; }
    /** Says that a sentence ended, so that the next sound comes after its pause. */
    void sentence_ended() { pausing_ = true; }
    /** Passes on the silence that a request ended in while it was held back; nonzero when the callback stops. */
    int finish() {
        if (held_.empty()) return 0;
        trailing_ += held_.size();
        const int stop = pass(held_.data(), held_.size());
        held_.clear();
        return stop;
    }
    uint64_t received() const { return received_; }
    uint64_t passed() const { return passed_; }
    bool capped() const { return capped_; }

private:
    int take(const float * s, size_t n) {
        received_ += n;
        if (pausing_) {
            size_t first = 0;
            while (first < n && std::fabs(s[first]) <= kSilent) first++;
            if (first == n) {
                held_.insert(held_.end(), s, s + n);
                return 0;
            }
            pausing_ = false;
            const size_t silent = trailing_ + held_.size() + first;
            if (silent < pause_) {
                const std::vector<float> zeros(pause_ - silent, 0.0f);
                if (const int stop = pass(zeros.data(), zeros.size())) return stop;
            }
            if (const int stop = finish()) return stop;
        }
        size_t last = n;
        while (last > 0 && std::fabs(s[last - 1]) <= kSilent) last--;
        trailing_ = last == 0 ? trailing_ + n : n - last;
        return pass(s, n);
    }

    int pass(const float * s, size_t n) {
        if (limit_) n = (size_t) std::min<uint64_t>(n, limit_ - passed_);
        passed_ += n;
        if (n && out_(s, n, user_data_)) return 1;
        capped_ = limit_ && passed_ == limit_;
        return capped_ ? 1 : 0;
    }

    speech_audio_callback out_;
    void * user_data_;
    size_t pause_;
    uint64_t limit_ = 0;
    bool capped_ = false;
    bool pausing_ = false;
    std::vector<float> held_;
    uint64_t trailing_ = 0, received_ = 0, passed_ = 0;
};

/** A cut finer than `cut`, or none after the spaces. */
std::optional<Cut> finer(Cut cut) {
    if (cut == Cut::Sentences) return Cut::Clauses;
    if (cut == Cut::Clauses) return Cut::Words;
    return std::nullopt;
}

/**
 * The library refuses a text too long for one request, in its tokens or in its predicted speech, as out_of_range naming
 * the text, before the request passes any audio.
 */
bool too_long(const Failure & e) {
    return e.code() == speech_status_name(SPEECH_ERROR_OUT_OF_RANGE) && e.option() == "text";
}

/** One synthesis of a text, request after request. */
class Speaker {
public:
    Speaker(speech_model * model, const speech_model_info * info, const std::vector<RequestOption> & options, Split split,
            speech_audio_callback on_audio, speech_progress_callback on_progress, void * user_data, Cancellation & cancellation, uint64_t limit_samples)
        : model_(model),
          options_(options),
          split_(split),
          on_progress_(on_progress),
          user_data_(user_data),
          cancellation_(cancellation),
          joiner_(on_audio, user_data, speech_model_info_sample_rate(info), limit_samples),
          by_sentence_(speaks_by_sentence(info)),
          draws_seed_(speech_model_info_takes(info, SPEECH_OPT_SEED) &&
                      std::none_of(options.begin(), options.end(), [](const RequestOption & o) { return o.option == SPEECH_OPT_SEED; })) {}

    std::optional<Spoken> speak_text(const std::string & text) {
        const bool sentence_each = by_sentence_ && split_ != Split::Refused;
        const std::vector<std::string> sentences = sentence_each ? cut_text(text, Cut::Sentences) : std::vector<std::string>();
        if (sentences.size() > 1) {
            for (size_t i = 0; i < sentences.size(); i++) {
                if (i > 0) joiner_.sentence_ended();
                if (!speak(sentences[i], Cut::Clauses)) break;
            }
        } else {
            speak(text, !by_sentence_ ? std::nullopt : std::optional<Cut>(sentence_each ? Cut::Clauses : Cut::Sentences));
        }
        if (cancelled_) return std::nullopt;
        spoken_.samples = joiner_.passed();
        return spoken_;
    }

private:
    /**
     * Speaks `text` as one request, or, where the library refuses it as too long and a cut is left, its pieces at that
     * cut; false once the synthesis has ended, cancelled or stopped.
     */
    bool speak(const std::string & text, std::optional<Cut> cut) {
        const Request request = new_request(model_);
        speech_request * r = request.get();
        joiner_.begin();
        speech_status status = SPEECH_OK;
        try {
            check(speech_request_set_text(r, text.c_str()));
            apply_options(r, options_);
            if (on_progress_) check(speech_request_set_progress(r, on_progress_, user_data_));
            status = check(cancellation_.run(r, [&] { return speech_synthesize(r, Joiner::receive, &joiner_); }));
        } catch (const Failure & e) {
            if (!too_long(e) || joiner_.received() > 0) throw;
            if (cut) return speak_pieces(text, *cut, e);
            if (split_ == Split::Compatible) return bisect(text, e);
            throw;
        }
        const speech_result * result = speech_request_result(r);
        spoken_.instructions_truncated = spoken_.instructions_truncated || speech_result_instructions_truncated(result);
        const bool stopped = status == SPEECH_CANCELLED || joiner_.finish() != 0;
        if (stopped && (!joiner_.capped() || cancellation_.cancelled())) {
            cancelled_ = true;
            return false;
        }
        if (spoken_.requests++ == 0) {
            spoken_.seed = speech_result_seed(result);
            if (draws_seed_) options_.push_back({SPEECH_OPT_SEED, spoken_.seed});
        }
        spoken_.stop = joiner_.capped() ? SPEECH_STOP_MAX_SECONDS : speech_result_stop(result);
        return spoken_.stop == SPEECH_STOP_COMPLETE;
    }

    /**
     * Speaks the pieces of a text the library refused, cut at `cut` or, where that leaves it whole, finer; sentences
     * cut apart are joined with a sentence's pause, as a text spoken a sentence at a time.
     */
    bool speak_pieces(const std::string & text, Cut cut, const Failure & refused) {
        for (std::optional<Cut> at = cut; at; at = finer(*at)) {
            const std::vector<std::string> pieces = cut_text(text, *at);
            if (pieces.size() < 2) continue;
            for (size_t i = 0; i < pieces.size(); i++) {
                if (i > 0 && *at == Cut::Sentences) joiner_.sentence_ended();
                if (!speak(pieces[i], finer(*at))) return false;
            }
            return true;
        }
        if (split_ == Split::Compatible) return bisect(text, refused);
        throw Failure(refused.code(), refused.option(),
                      "\"" + excerpt(text) + "\" has no comma or space to cut it at, and is too long for one request: " + refused.what());
    }

    /** Retries only a refused text, preserving its bytes on Unicode code point boundaries. */
    bool bisect(const std::string & text, const Failure & refused) {
        std::vector<size_t> boundaries;
        for (size_t at = 0; at < text.size();) {
            at += code_point(text, at).second;
            boundaries.push_back(at);
        }
        if (boundaries.size() < 2) throw refused;
        const size_t middle = boundaries[boundaries.size() / 2 - 1];
        return speak(text.substr(0, middle), std::nullopt) && speak(text.substr(middle), std::nullopt);
    }

    speech_model * model_;
    std::vector<RequestOption> options_;
    Split split_;
    speech_progress_callback on_progress_;
    void * user_data_;
    Cancellation & cancellation_;
    Joiner joiner_;
    bool by_sentence_, draws_seed_;
    bool cancelled_ = false;
    Spoken spoken_;
};

}  // namespace

std::vector<std::string> cut_text(const std::string & text, Cut cut) {
    std::vector<std::string> pieces;
    const auto add = [&](size_t begin, size_t end) {
        std::string piece = trimmed(text, begin, end);
        if (!piece.empty()) pieces.push_back(std::move(piece));
    };
    size_t start = 0, at = 0;
    while (at < text.size()) {
        const auto [c, n] = code_point(text, at);
        size_t end = 0;
        if (cut == Cut::Words) {
            if (python_space(c)) end = at + n;
        } else if (cut == Cut::Sentences && (c == '\n' || c == '\r')) {
            end = at + n;
        } else if (ends_anywhere(c, cut) || ends_before_space(c, cut)) {
            bool anywhere = false;
            size_t after = at;
            while (after < text.size()) {
                const auto [d, m] = code_point(text, after);
                if (!ends_anywhere(d, cut) && !ends_before_space(d, cut) && !closes(d)) break;
                anywhere |= ends_anywhere(d, cut);
                after += m;
            }
            if (anywhere || after == text.size() || python_space(code_point(text, after).first)) end = after;
            else at = after;
            if (!end) continue;
        }
        if (end) {
            add(start, end);
            start = at = end;
        } else {
            at += n;
        }
    }
    add(start, text.size());
    return pieces;
}

double longest_request_seconds(const speech_model_info * info) {
    double longest = INFINITY;
    for (const speech_option o : {SPEECH_OPT_SECONDS, SPEECH_OPT_MAX_SECONDS}) {
        if (!speech_model_info_takes(info, o)) continue;
        double minimum = 0, maximum = 0;
        int exclusive = 0;
        check(speech_model_info_option_range(info, o, &minimum, &maximum, &exclusive));
        longest = std::min(longest, maximum);
    }
    return longest;
}

bool speaks_by_sentence(const speech_model_info * info) {
    return longest_request_seconds(info) < kShortRequest;
}

std::optional<Spoken> speak_text(speech_model * model, const std::string & text, const std::vector<RequestOption> & options, Split split,
                                 speech_audio_callback on_audio, speech_progress_callback on_progress, void * user_data,
                                 Cancellation & cancellation, uint64_t limit_samples) {
    const ModelInfo info = model_info(model);
    return Speaker(model, info.get(), options, split, on_audio, on_progress, user_data, cancellation, limit_samples).speak_text(text);
}
