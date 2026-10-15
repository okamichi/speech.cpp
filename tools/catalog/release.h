#pragma once

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

// The latest release of speech.cpp, which `speech update` and the notice of a newer release ask github.com for with the
// system's curl. https://github.com/nyosegawa/speech.cpp/releases/latest redirects to the latest release's page,
// .../releases/tag/vX.Y.Z, which needs no API and counts against no rate limit, as install.sh finds it. The answer is
// kept in release.json in cache_dir(), {"checked": <seconds since 1970>, "latest": "X.Y.Z", or "" before any answer},
// with the time of every attempt, the failed and stopped ones too, so that the notice asks at most once a day. The
// library never reaches the network; only the command line asks.

/** Whether `s` is a version X.Y.Z. */
bool is_version(const std::string & s);

/** Whether `a` and `b` are versions X.Y.Z and `a` is the later one. */
bool later_version(const std::string & a, const std::string & b);

/**
 * Asks github.com for the latest release now, waiting up to 10 s, keeps the answer in release.json, and returns its
 * version. A failure throws an io Failure that says what to do.
 */
std::string latest_release();

/**
 * A check of the latest release that runs beside a subcommand: when release.json is missing or a day old, it starts curl
 * on a thread of its own, which keeps the answer in release.json as soon as curl ends, waiting up to 5 s. It never
 * throws and never prints: a check that fails is tried again a day later.
 */
class ReleaseCheck {
public:
    ReleaseCheck();
    ~ReleaseCheck();
    ReleaseCheck(const ReleaseCheck &) = delete;
    ReleaseCheck & operator=(const ReleaseCheck &) = delete;

    /** The latest release known: this check's answer once it came, or release.json's, or "" for none. */
    std::string latest() const;

    /**
     * Ends curl if it still runs, keeping the time of the attempt, and waits for the thread, which takes milliseconds;
     * then returns latest().
     */
    std::string finish();

private:
    void run();

    mutable std::mutex mutex_;
    std::string latest_;
    std::atomic<bool> stopping_{false};
    std::thread thread_;
};
