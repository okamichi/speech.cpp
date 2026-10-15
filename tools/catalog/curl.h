#pragma once

#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/types.h>
#endif

// Running the system's curl, so that the system's proxies and certificates apply: `curl` on the PATH on macOS and
// Linux, and System32's curl.exe on Windows, which Windows 10 1803 and later carry.

/** One end of a pipe from curl, read until curl closes it: a HANDLE on Windows, a file descriptor elsewhere. */
#ifdef _WIN32
using PipeEnd = void *;
#else
using PipeEnd = int;
#endif

/** Reads from a pipe into `buffer`, returning the bytes read, 0 at its end. */
size_t read_pipe(PipeEnd pipe, char * buffer, size_t size);

/** Reads a pipe to its end and closes it. */
std::string read_to_end(PipeEnd pipe);

void close_pipe(PipeEnd pipe);

/**
 * curl running with its stdout and stderr on pipes and its stdin on the null device, and no other handle or file
 * descriptor of this process, so that it holds no model file or socket open past its run. The caller reads and closes
 * both pipes.
 */
class Curl {
public:
    /** Starts curl with `args`; without curl it throws an io Failure whose message is `missing`. */
    Curl(const std::vector<std::string> & args, const std::string & missing);
    ~Curl();
    Curl(const Curl &) = delete;
    Curl & operator=(const Curl &) = delete;

    PipeEnd out() const { return out_; }
    PipeEnd err() const { return err_; }

    /** Waits up to `ms` milliseconds for curl to end, and returns whether it has, with its exit code in `code`. */
    bool wait(int ms, int & code);

    /** Ends curl; wait() then returns once it has ended. */
    void stop();

private:
    PipeEnd out_{}, err_{};
#ifdef _WIN32
    void * process_ = nullptr;
#else
    pid_t pid_ = 0;
#endif
};

/** The null device's name, for curl's --output: NUL on Windows, /dev/null elsewhere. */
const char * null_device();
