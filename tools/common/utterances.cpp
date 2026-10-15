#include "utterances.h"

#include <chrono>
#include <stdexcept>
#include <variant>

#include "regions.h"

namespace utterances {

namespace {

/**
 * The audio an utterance under way gains before it is read again: the audio arrives every 0.1 to 0.2 s from a browser or
 * an OpenAI client and the detection takes it in chunks of 32 ms, so a reading of less new audio would read nothing new.
 */
constexpr double kReadAgainAfter = 0.2;

/**
 * How far before the audio heard a region not yet begun may begin, beyond its padding: it begins at a chunk the detection
 * has not yet taken, within a chunk of Silero VAD's 32 ms and what the resampler holds back, rounded up. The resampler
 * holds back 64 / (0.9476 · the lower rate) seconds, which grows as the rate falls: 4.2 ms from 16 kHz up, 8.4 ms from
 * kLowestRate, and 0.68 s from 100 Hz, which would leave a region's start among the audio dropped.
 */
constexpr double kReachMargin = 0.1;

/** The lowest rate a detection of the assembly takes, telephony's, whose resampler's delay kReachMargin covers. */
constexpr int kLowestRate = 8000;

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

Detection::Detection(speech_model * model, std::shared_ptr<const void> keep, std::vector<RequestOption> options, int rate)
    : model_(model), keep_(std::move(keep)), options_(std::move(options)), rate_(rate) {
    if (rate < kLowestRate) {
        throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "audio",
                      "audio at " + std::to_string(rate) + " Hz is below the " + std::to_string(kLowestRate) +
                          " Hz that detecting speech as it arrives takes; resample it to " + std::to_string(kLowestRate) + " Hz or more");
    }
    start();
    const ModelInfo info = model_info(model_);
    const auto milliseconds = [&](speech_option option) {
        for (const RequestOption & o : options_) {
            if (o.option == option) return (double) std::get<int64_t>(o.value);
        }
        int64_t value = 0;
        check(speech_model_info_option_default_int(info.get(), option, &value));
        return (double) value;
    };
    reach_back_ = milliseconds(SPEECH_OPT_SPEECH_PAD_MS) / 1000 + kReachMargin;
}

void Detection::start() {
    const Request request = new_request(model_);
    apply_options(request.get(), options_);
    speech_detection * raw = nullptr;
    check(speech_detection_start(request.get(), rate_, &raw));
    detection_.reset(raw);
}

void Detection::restart() {
    detection_.reset();
    start();
}

void Detection::push(const float * samples, size_t n) {
    check(speech_detection_push(detection_.get(), samples, n));
}

void Detection::end() {
    check(speech_detection_end(detection_.get()));
}

size_t Detection::region_count() const {
    return speech_detection_region_count(detection_.get());
}

std::pair<double, double> Detection::region(size_t index) const {
    double start = 0, end = 0;
    check(speech_detection_region(detection_.get(), index, &start, &end));
    return {start, end};
}

std::optional<std::pair<double, bool>> Detection::speaking() const {
    double start = 0;
    int kept = 0;
    if (!speech_detection_speaking(detection_.get(), &start, &kept)) return std::nullopt;
    return std::make_pair(start, kept != 0);
}

Assembly::Assembly(Recognizer & recognizer, int rate, std::function<void(const Event &)> emit)
    : recognizer_(recognizer), rate_(rate), emit_(std::move(emit)) {
    worker_ = std::thread([this] { work(); });
}

Assembly::~Assembly() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        closing_ = true;
        finals_.clear();
        if (reading_) reading_->cancel();
        if (final_) final_->cancel();
    }
    changed_.notify_all();
    worker_.join();
}

void Assembly::ask(std::optional<Asked> asked) {
    std::lock_guard<std::mutex> lock(mutex_);
    asked_ = std::move(asked);
    // Readings asked otherwise would not agree as two readings of the same request do.
    if (reading_) reading_->cancel();
    if (current_) current_->last_reading.clear();
    changed_.notify_all();
}

void Assembly::detect(std::unique_ptr<Detection> detection) {
    if (detection && detection->rate() != rate_) throw std::logic_error("a detection of another rate than its assembly's");
    std::lock_guard<std::mutex> lock(mutex_);
    end_detection();
    // An utterance the caller was to commit is left uncommitted: its audio stays for a commit, and the regions of the
    // audio that follows are utterances of their own.
    if (manual_ && current_) {
        if (reading_) reading_->cancel();
        current_.reset();
    }
    manual_ = false;
    detection_ = std::move(detection);
    origin_ = heard_;
    pushed_ = 0;
    regions_ = 0;
}

void Assembly::manual() {
    std::lock_guard<std::mutex> lock(mutex_);
    end_detection();
    manual_ = true;
    begin_manual_utterance();
    changed_.notify_all();
}

void Assembly::push(const float * samples, size_t n) {
    if (n == 0) return;
    if (detection_) {
        const auto t0 = std::chrono::steady_clock::now();
        detection_->push(samples, n);
        detecting_ += seconds_since(t0);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    audio_.insert(audio_.end(), samples, samples + n);
    heard_ += n;
    if (detection_) {
        pushed_ += n;
        take_regions();
    } else if (manual_) {
        begin_manual_utterance();
    }
    changed_.notify_all();
}

bool Assembly::commit() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (heard_ == buffer_) return false;
    // The commit is the utterance under way where its speech started has gone out, or the caller's, and otherwise one of
    // its own, the region under way dropped.
    uint64_t number = 0;
    if (current_ && current_->announced) {
        number = current_->number;
    } else {
        if (current_) drop_current();
        number = next_++;
    }
    commit_samples(number, buffer_, heard_);
    if (detection_) restart_detection();
    return true;
}

void Assembly::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (reading_) reading_->cancel();
    current_.reset();
    keep_from(heard_);
    if (detection_) restart_detection();
}

void Assembly::end() {
    std::unique_lock<std::mutex> lock(mutex_);
    end_detection();
    changed_.notify_all();
    changed_.wait(lock, [&] { return finals_.empty() && !busy_; });
}

size_t Assembly::held() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return audio_.size() + committed_;
}

size_t Assembly::committed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return committed_;
}

double Assembly::heard() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return (double) heard_ / rate_;
}

void Assembly::take_regions() {
    const auto at = [&](double seconds) { return origin_ + region_samples(Region{seconds, seconds}, rate_, pushed_).first; };
    for (const size_t count = detection_->region_count(); regions_ < count; regions_++) {
        const std::pair<double, double> region = detection_->region(regions_);
        const uint64_t first = at(region.first), last = at(region.second);
        if (current_ && current_->start != first) drop_current();
        if (!current_) current_ = Utterance{next_++, first, false, first, "", "", ""};
        announce(*current_);
        Event stopped;
        stopped.kind = Event::Kind::SpeechStopped;
        stopped.utterance = current_->number;
        stopped.at = (double) last / rate_;
        send(stopped);
        commit_samples(current_->number, first, last);
    }
    const std::optional<std::pair<double, bool>> speaking = detection_->speaking();
    if (!speaking) {
        if (current_) drop_current();
        // The audio before what a region still to come may reach back to lies outside every region.
        const uint64_t reach = (uint64_t) (detection_->reach_back() * rate_);
        if (heard_ - buffer_ > reach) keep_from(heard_ - reach);
        return;
    }
    const uint64_t start = at(speaking->first);
    if (current_ && current_->start != start) drop_current();
    if (!current_) current_ = Utterance{next_++, start, false, start, "", "", ""};
    keep_from(start);
    if (speaking->second) announce(*current_);
}

void Assembly::announce(Utterance & u) {
    if (u.announced) return;
    u.announced = true;
    Event started;
    started.kind = Event::Kind::SpeechStarted;
    started.utterance = u.number;
    started.at = (double) u.start / rate_;
    send(started);
    send_agreed(u);
}

void Assembly::send_agreed(Utterance & u) {
    if (!u.announced || u.agreed.size() <= u.sent.size() || u.agreed.compare(0, u.sent.size(), u.sent) != 0) return;
    Event delta;
    delta.kind = Event::Kind::Delta;
    delta.utterance = u.number;
    delta.delta = u.agreed.substr(u.sent.size());
    send(delta);
    u.sent = u.agreed;
}

void Assembly::commit_samples(uint64_t number, uint64_t first, uint64_t last) {
    Event committed;
    committed.kind = Event::Kind::Committed;
    committed.utterance = number;
    committed.recognized = asked_.has_value();
    send(committed);
    if (reading_) reading_->cancel();
    if (asked_) {
        Final f;
        f.number = number;
        if (current_ && current_->number == number) {
            f.sent = current_->sent;
            f.draft = current_->last_reading;
        }
        f.asked = *asked_;
        // The recognition takes its place as the utterance is committed, so that it runs before what comes later.
        try {
            f.reservation = recognizer_.reserve(f.asked);
            f.samples.assign(audio_.begin() + (long) (first - buffer_), audio_.begin() + (long) (last - buffer_));
            f.length = f.samples.size();
        } catch (...) {
            f.refused = std::current_exception();
        }
        committed_ += f.length;
        finals_.push_back(std::move(f));
    }
    current_.reset();
    keep_from(last);
    changed_.notify_all();
}

void Assembly::keep_from(uint64_t sample) {
    if (sample < buffer_) throw std::logic_error("a region begins in audio the assembly has dropped");
    audio_.erase(audio_.begin(), audio_.begin() + (long) (sample - buffer_));
    buffer_ = sample;
}

void Assembly::drop_current() {
    // A region is announced once it is certain to be kept, so only one not yet announced can be dropped.
    if (current_->announced) throw std::logic_error("the detection dropped a region it was certain to keep");
    if (reading_) reading_->cancel();
    current_.reset();
}

void Assembly::end_detection() {
    if (!detection_) return;
    const auto t0 = std::chrono::steady_clock::now();
    detection_->end();
    detecting_ += seconds_since(t0);
    take_regions();
    detection_.reset();
}

void Assembly::restart_detection() {
    detection_->restart();
    origin_ = heard_;
    pushed_ = 0;
    regions_ = 0;
}

void Assembly::begin_manual_utterance() {
    // The caller's utterances have no speech started, so their deltas go out at once.
    if (!current_ && heard_ > buffer_) current_ = Utterance{next_++, buffer_, true, buffer_, "", "", ""};
}

bool Assembly::reading_due() const {
    return asked_ && current_ && finals_.empty() && heard_ >= current_->read_to + (uint64_t) (kReadAgainAfter * rate_);
}

void Assembly::send(const Event & e) {
    if (!closing_) emit_(e);
}

void Assembly::work() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [&] { return closing_ || !finals_.empty() || reading_due(); });
        if (closing_) return;
        busy_ = true;
        if (!finals_.empty()) {
            Final f = std::move(finals_.front());
            finals_.pop_front();
            const std::shared_ptr<Cancellation> cancellation = final_ = std::make_shared<Cancellation>();
            lock.unlock();
            Event e;
            e.utterance = f.number;
            try {
                if (f.refused) std::rethrow_exception(f.refused);
                std::optional<Transcript> t =
                    recognizer_.transcribe(*f.reservation, f.asked, std::move(f.samples), rate_, *cancellation, f.number, false, f.draft);
                if (t) {
                    e.kind = Event::Kind::Completed;
                    e.transcript = std::move(*t);
                    e.duration = (double) f.length / rate_;
                }
            } catch (...) {
                e.kind = Event::Kind::Failed;
                e.failure = std::current_exception();
            }
            // The place passes as soon as the recognition has ended.
            f.reservation.reset();
            lock.lock();
            final_.reset();
            committed_ -= f.length;
            busy_ = false;
            // The deltas join into the final text wherever it goes on from what they gave, so that a client that builds
            // the text from them, as OpenAI's guide does, gets it whole; one that differs keeps them as they were.
            if (e.kind == Event::Kind::Completed && e.transcript.text.size() > f.sent.size() &&
                e.transcript.text.compare(0, f.sent.size(), f.sent) == 0) {
                Event rest;
                rest.kind = Event::Kind::Delta;
                rest.utterance = f.number;
                rest.delta = e.transcript.text.substr(f.sent.size());
                send(rest);
            }
            if (e.kind == Event::Kind::Completed || e.kind == Event::Kind::Failed) send(e);
            changed_.notify_all();
            continue;
        }
        const uint64_t number = current_->number;
        current_->read_to = heard_;
        const Asked asked = *asked_;
        // A reading takes its place once it is due, with the mutex held, so that a commit accepted before it runs first,
        // and one accepted after it, which this thread recognizes next, after it.
        std::unique_ptr<Reservation> reservation;
        try {
            reservation = recognizer_.reserve(asked);
        } catch (...) {
            // A reading refused a place adds no text; the recognition of the whole utterance reports why.
            busy_ = false;
            continue;
        }
        std::vector<float> samples(audio_.begin() + (long) (current_->start - buffer_), audio_.end());
        // Each reading reads the utterance from its start again, and the reading before is the likeliest text of this one.
        const std::string draft = current_->last_reading;
        const std::shared_ptr<Cancellation> cancellation = reading_ = std::make_shared<Cancellation>();
        lock.unlock();
        std::optional<Transcript> t;
        try {
            t = recognizer_.transcribe(*reservation, asked, std::move(samples), rate_, *cancellation, number, true, draft);
        } catch (...) {
            // A reading that fails adds no text; the recognition of the whole utterance reports the failure.
        }
        reservation.reset();
        lock.lock();
        reading_.reset();
        busy_ = false;
        changed_.notify_all();
        if (!t || cancellation->cancelled() || !current_ || current_->number != number) continue;
        current_->agreed = agreed_beginning(current_->last_reading, t->text);
        current_->last_reading = t->text;
        send_agreed(*current_);
    }
}

}  // namespace utterances
