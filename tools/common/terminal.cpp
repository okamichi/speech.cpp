#include "terminal.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <io.h>
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#include "utf8.h"

namespace terminal {

namespace {

#ifdef _WIN32
HANDLE handle_of(FILE * stream) {
    return (HANDLE) _get_osfhandle(_fileno(stream));
}

/** The output code page the console had before setup(), or 0 where setup() left it. */
UINT g_code_page = 0;

void restore_code_page() {
    if (g_code_page) SetConsoleOutputCP(g_code_page);
}

BOOL WINAPI on_control(DWORD) {
    restore_code_page();
    // The handler after it, the system's, ends the process.
    return FALSE;
}
#endif

int columns_of(uint32_t cp) {
    if ((cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x200B && cp <= 0x200F) || (cp >= 0x20D0 && cp <= 0x20FF) || (cp >= 0xFE00 && cp <= 0xFE0F) ||
        (cp >= 0xFE20 && cp <= 0xFE2F)) {
        return 0;
    }
    const bool wide = (cp >= 0x1100 && cp <= 0x115F) || (cp >= 0x2E80 && cp <= 0x303E) || (cp >= 0x3041 && cp <= 0x33FF) ||
                      (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xA000 && cp <= 0xA4CF) ||
                      (cp >= 0xAC00 && cp <= 0xD7A3) || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0xFE30 && cp <= 0xFE4F) ||
                      (cp >= 0xFF00 && cp <= 0xFF60) || (cp >= 0xFFE0 && cp <= 0xFFE6) || (cp >= 0x1F300 && cp <= 0x1F64F) ||
                      (cp >= 0x1F900 && cp <= 0x1F9FF) || (cp >= 0x20000 && cp <= 0x3FFFD);
    return wide ? 2 : 1;
}

const char * const kEllipsis = "\xE2\x80\xA6";

/** An equalizer's bars, which the spinner of a Status plays a frame a tenth of a second. */
const char * const kFrames[] = {
    "\xE2\x96\x81\xE2\x96\x83\xE2\x96\x85", "\xE2\x96\x82\xE2\x96\x84\xE2\x96\x86", "\xE2\x96\x83\xE2\x96\x85\xE2\x96\x87",
    "\xE2\x96\x84\xE2\x96\x86\xE2\x96\x85", "\xE2\x96\x85\xE2\x96\x87\xE2\x96\x83", "\xE2\x96\x86\xE2\x96\x85\xE2\x96\x82",
    "\xE2\x96\x87\xE2\x96\x83\xE2\x96\x81", "\xE2\x96\x85\xE2\x96\x82\xE2\x96\x83", "\xE2\x96\x83\xE2\x96\x81\xE2\x96\x84",
    "\xE2\x96\x82\xE2\x96\x83\xE2\x96\x85",
};

/** Moves to the start of the line and erases it. */
const char * const kErase = "\r\033[K";

}  // namespace

void setup(FILE * out) {
#ifdef _WIN32
    bool console = false;
    for (FILE * stream : {out, stderr}) {
        DWORD mode = 0;
        if (!GetConsoleMode(handle_of(stream), &mode)) continue;
        console = true;
        SetConsoleMode(handle_of(stream), mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
    if (console && GetConsoleOutputCP() != CP_UTF8) {
        g_code_page = GetConsoleOutputCP();
        SetConsoleOutputCP(CP_UTF8);
        std::atexit(restore_code_page);
        SetConsoleCtrlHandler(on_control, TRUE);
    }
#else
    (void) out;
#endif
}

bool is_terminal(FILE * stream) {
#ifdef _WIN32
    DWORD mode = 0;
    return GetConsoleMode(handle_of(stream), &mode) != 0;
#else
    return isatty(fileno(stream)) != 0;
#endif
}

bool draws_on(FILE * stream) {
    const char * term = std::getenv("TERM");
    if (term && std::strcmp(term, "dumb") == 0) return false;
#ifdef _WIN32
    DWORD mode = 0;
    return GetConsoleMode(handle_of(stream), &mode) && (mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING);
#else
    return isatty(fileno(stream)) != 0;
#endif
}

bool draws() {
    return draws_on(stderr);
}

bool colors_on(FILE * stream) {
    const char * off = std::getenv("NO_COLOR");
    return draws_on(stream) && !(off && *off);
}

bool colors() {
    return colors_on(stderr);
}

int columns(FILE * stream) {
#ifdef _WIN32
    CONSOLE_SCREEN_BUFFER_INFO info;
    if (GetConsoleScreenBufferInfo(handle_of(stream), &info)) return info.srWindow.Right - info.srWindow.Left + 1;
#else
    winsize size{};
    if (ioctl(fileno(stream), TIOCGWINSZ, &size) == 0 && size.ws_col > 0) return size.ws_col;
#endif
    return 80;
}

std::string head(const std::string & text, int width) {
    const std::vector<uint32_t> cps = decode_utf8(text).value_or(std::vector<uint32_t>());
    int whole = 0;
    for (const uint32_t cp : cps) whole += columns_of(cp);
    if (whole <= width) return text;
    int used = 0;
    size_t end = 0;
    while (end < cps.size() && used + columns_of(cps[end]) <= width - 1) used += columns_of(cps[end++]);
    return encode_utf8(std::vector<uint32_t>(cps.begin(), cps.begin() + (long) end)) + kEllipsis;
}

std::string tail(const std::string & text, int width) {
    const std::vector<uint32_t> cps = decode_utf8(text).value_or(std::vector<uint32_t>());
    int whole = 0;
    for (const uint32_t cp : cps) whole += columns_of(cp);
    if (whole <= width) return text;
    int used = 0;
    size_t first = cps.size();
    while (first > 0 && used + columns_of(cps[first - 1]) <= width - 1) used += columns_of(cps[--first]);
    return kEllipsis + encode_utf8(std::vector<uint32_t>(cps.begin() + (long) first, cps.end()));
}

std::string paint(Color color, const std::string & text, FILE * stream) {
    if (!colors_on(stream)) return text;
    const char * code = "";
    switch (color) {
        case Color::Mint: code = "38;5;78"; break;
        case Color::Pink: code = "38;5;205"; break;
        case Color::Lavender: code = "38;5;141"; break;
        case Color::Sky: code = "38;5;75"; break;
        case Color::Peach: code = "38;5;209"; break;
        case Color::Red: code = "38;5;203"; break;
        case Color::Gray: code = "38;5;244"; break;
        case Color::Bold: code = "1"; break;
    }
    return std::string("\033[") + code + "m" + text + "\033[0m";
}

std::string marked(Mark mark, const std::string & text) {
    if (!draws()) return text;
    switch (mark) {
        case Mark::Done: return paint(Color::Mint, "\xE2\x9C\x93") + " " + text;
        case Mark::Audio: return paint(Color::Pink, "\xE2\x99\xAA") + " " + text;
        case Mark::Fetch: return paint(Color::Sky, "\xE2\x86\x93") + " " + text;
        case Mark::Warning: return paint(Color::Peach, "!") + " " + text;
    }
    return text;
}

std::string logo(FILE * stream) {
    if (!draws_on(stream)) return "";
    return paint(Color::Pink, "\xE2\x96\x81\xE2\x96\x83", stream) + paint(Color::Lavender, "\xE2\x96\x85\xE2\x96\x87", stream) +
           paint(Color::Sky, "\xE2\x96\x85\xE2\x96\x83\xE2\x96\x81", stream);
}

std::string bar(uint64_t done, uint64_t total, int width) {
    const int filled = total ? (int) (done * (uint64_t) width / total) : 0;
    std::string full, empty;
    for (int i = 0; i < filled; i++) full += "\xE2\x94\x81";
    for (int i = filled; i < width; i++) empty += "\xE2\x94\x80";
    return paint(Color::Lavender, full) + paint(Color::Gray, empty);
}

Status::Status(std::string doing, int indent) : doing_(std::move(doing)), indent_(indent) {
    if (!draws()) return;
    thread_ = std::thread([this] {
        std::unique_lock<std::mutex> lock(mutex_);
        for (int frame = 0; !stopped_; frame++) {
            draw(frame);
            wake_.wait_for(lock, std::chrono::milliseconds(100), [this] { return stopped_; });
        }
    });
}

Status::~Status() {
    if (!thread_.joinable()) return;
    stop();
    std::fputs(kErase, stderr);
    std::fflush(stderr);
}

void Status::set(std::string doing) {
    std::lock_guard<std::mutex> lock(mutex_);
    doing_ = std::move(doing);
}

void Status::done(const std::string & line, Mark mark) {
    if (!thread_.joinable()) {
        std::fprintf(stderr, "%*s%s\n", indent_, "", line.c_str());
        return;
    }
    stop();
    std::fprintf(stderr, "%s%*s%s\n", kErase, indent_, "", marked(mark, line).c_str());
    std::fflush(stderr);
}

void Status::draw(int frame) {
    char seconds[32];
    std::snprintf(seconds, sizeof seconds, "%.1f s", std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count());
    // The line stays narrower than the terminal, since a line that wraps is no longer the one "\r" returns to.
    const int room = columns(stderr) - 1 - indent_ - 4 - 2 - (int) std::strlen(seconds);
    const std::string doing = head(doing_ + kEllipsis, room);
    const std::string line = kErase + std::string(indent_, ' ') + paint(Color::Lavender, kFrames[frame % 10]) + " " + doing + "  " +
                             paint(Color::Gray, seconds);
    std::fputs(line.c_str(), stderr);
    std::fflush(stderr);
}

void Status::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
    }
    wake_.notify_all();
    thread_.join();
}

}  // namespace terminal
