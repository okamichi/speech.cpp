#pragma once

#include <mutex>

#include "speech.h"

/**
 * Stops, from any thread, a run of library requests made one after another: the request under way and every one after
 * it. A request runs through run(), which a cancel before it answers with SPEECH_CANCELLED without calling the library.
 */
class Cancellation {
public:
    /** Runs `call` on `request`, which it cancels while it runs; SPEECH_CANCELLED once the run is cancelled. */
    template <typename Call>
    speech_status run(speech_request * request, Call call) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (cancelled_) return SPEECH_CANCELLED;
            current_ = request;
        }
        try {
            const speech_status status = call();
            std::lock_guard<std::mutex> lock(mutex_);
            current_ = nullptr;
            return status;
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            current_ = nullptr;
            throw;
        }
    }

    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        cancelled_ = true;
        if (current_) speech_request_cancel(current_);
    }

    bool cancelled() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cancelled_;
    }

private:
    mutable std::mutex mutex_;
    speech_request * current_ = nullptr;
    bool cancelled_ = false;
};
