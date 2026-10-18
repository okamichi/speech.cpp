#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <thread>

// How the tools write for a person on a terminal: whether a stream is one, the colors, marks and logo they draw there,
// and a status line that shows a step under way. Off a terminal (a pipe, a file, a log), the same lines go out as plain
// text without a color, a mark or a spinner, so that a script or a log reads them a line at a time.

namespace terminal {

/**
 * Prepares the consoles of `out` and stderr on Windows: their virtual terminal processing, which draws colors and the
 * erasing of a line, and the output code page UTF-8, which the console had otherwise read the tools' UTF-8 in; the code
 * page is restored as the process ends, by Ctrl-C too. It does nothing elsewhere.
 */
void setup(FILE * out);

/** Whether `stream` is a terminal; on Windows a console, since _isatty() is true of any character device, NUL among them. */
bool is_terminal(FILE * stream);

/** Whether the tools draw on `stream` for a person: it is a terminal that erases a line, and TERM is not dumb. */
bool draws_on(FILE * stream);

/** Whether the tools draw on stderr (draws_on()). */
bool draws();

/** Whether what the tools draw on `stream` has colors: they draw, and NO_COLOR is unset or empty, as no-color.org asks. */
bool colors_on(FILE * stream);

/** Whether what the tools draw on stderr has colors (colors_on()). */
bool colors();

/** The columns of the terminal `stream` is, or 80. */
int columns(FILE * stream);

/** The beginning of `text` that a terminal draws in `width` columns, before an ellipsis where it does not fit whole. */
std::string head(const std::string & text, int width);

/** The end of `text` that a terminal draws in `width` columns, after an ellipsis where it does not fit whole. */
std::string tail(const std::string & text, int width);

/**
 * The colors of the tools, from the 256 of xterm: mid tones that read on a dark background and on a light one alike,
 * unlike the pale ones a dark theme would take.
 */
enum class Color { Mint, Pink, Lavender, Sky, Peach, Red, Gray, Bold };

/** `text` in `color` where `stream` has colors; otherwise `text` as it is. */
std::string paint(Color color, const std::string & text, FILE * stream = stderr);

/** The marks a line begins with where the tools draw: a step done, audio made, a fetch, and a warning. */
enum class Mark { Done, Audio, Fetch, Warning };

/** `text` after its mark and a space where draws() holds; otherwise `text` alone. */
std::string marked(Mark mark, const std::string & text);

/** The waveform a banner begins with, where the tools draw on `stream`; otherwise "". */
std::string logo(FILE * stream = stderr);

/** A bar `width` columns wide, its first `done` of `total` filled. */
std::string bar(uint64_t done, uint64_t total, int width);

/**
 * A step under way: where draws() holds, one line of stderr after `indent` spaces, which a spinner and the seconds since
 * the step began redraw ten times a second until done() or the end of the object; elsewhere nothing until done(). Nothing
 * else may write on stderr while one is shown.
 */
class Status {
public:
    /** Starts the step, `doing` saying what it is: "Loading qwen3-asr-0.6b". */
    explicit Status(std::string doing, int indent = 0);
    /** Erases the line of a step that ends without done(), so that a failure's message starts on a line of its own. */
    ~Status();
    Status(const Status &) = delete;
    Status & operator=(const Status &) = delete;

    /** Says what the step does as it goes on: "Speaking hello.wav, 3.1 s of audio". */
    void set(std::string doing);

    /** Ends the step with `line`: in place of the spinner, after `mark`, where draws() holds; on a line of its own elsewhere. */
    void done(const std::string & line, Mark mark = Mark::Done);

private:
    void draw(int frame);
    void stop();

    std::string doing_;
    int indent_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    std::mutex mutex_;
    std::condition_variable wake_;
    bool stopped_ = false;
    std::thread thread_;
};

}  // namespace terminal
