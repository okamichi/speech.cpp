#include "update.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

#include "commands.h"
#include "failure.h"
#include "speech.h"

// speech update asks github.com for the latest release and, when this speech is the one the installer put in place,
// runs the installer with the option that leaves PATH alone, which downloads the release, checks it against the
// release's SHA256SUMS and replaces this version; any other speech it leaves alone and says how to update. The notice of
// a newer release that the other subcommands print is here too.

namespace fs = std::filesystem;

namespace {

const char * const kInstallSh = "curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh";
const char * const kInstallPs1 = "irm https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1 | iex";

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

bool notices_wanted() {
    const char * off = std::getenv("SPEECH_NO_UPDATE_CHECK");
    if (off && *off) return false;
#ifdef _WIN32
    // _isatty() is true of any character device, NUL among them; only a console has a console mode.
    DWORD mode = 0;
    return GetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), &mode) != 0;
#else
    return isatty(STDERR_FILENO) != 0;
#endif
}

std::string newer_release(const std::string & latest) {
    return "speech.cpp " + latest + " is out (this is " + speech_version() + ")";
}

#ifdef _WIN32
/** The path of a file or folder with every junction and link resolved, or an empty path when it cannot be opened. */
fs::path resolved(const fs::path & path) {
    HANDLE h = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                           FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) return {};
    std::vector<wchar_t> name(MAX_PATH);
    DWORD n = GetFinalPathNameByHandleW(h, name.data(), (DWORD) name.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (n >= name.size()) {
        name.resize(n + 1);
        n = GetFinalPathNameByHandleW(h, name.data(), (DWORD) name.size(), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    }
    CloseHandle(h);
    if (n == 0 || n >= name.size()) return {};
    return fs::path(std::wstring(name.data(), n));
}
#else
fs::path resolved(const fs::path & path) {
    std::error_code e;
    const fs::path out = fs::canonical(path, e);
    return e ? fs::path() : out;
}
#endif

/** This executable's path, with every link resolved. */
fs::path this_executable() {
#ifdef _WIN32
    std::vector<wchar_t> name(MAX_PATH);
    for (;;) {
        const DWORD n = GetModuleFileNameW(nullptr, name.data(), (DWORD) name.size());
        if (n == 0) throw io_failure("cannot find where this speech.exe is (Windows error " + std::to_string(GetLastError()) + ")");
        if (n < name.size()) {
            name.resize(n);
            break;
        }
        name.resize(name.size() * 2);
    }
    const fs::path path(std::wstring(name.begin(), name.end()));
#elif defined(__APPLE__)
    uint32_t size = 0;
    _NSGetExecutablePath(nullptr, &size);
    std::vector<char> name(size);
    if (_NSGetExecutablePath(name.data(), &size) != 0) throw io_failure("cannot find where this speech is");
    const fs::path path(name.data());
#else
    const fs::path path("/proc/self/exe");
#endif
    const fs::path out = resolved(path);
    if (out.empty()) throw io_failure("cannot resolve " + path.u8string() + ", where this speech is");
    return out;
}

/**
 * Whether `self` is a version the installer put in place: <root>/<X.Y.Z>/speech, under ~/.local/share/speech.cpp, or
 * <root>\<X.Y.Z>\speech.exe under %LOCALAPPDATA%\Programs\speech.cpp on Windows.
 */
bool put_by_installer(const fs::path & self) {
#ifdef _WIN32
    const wchar_t * base = _wgetenv(L"LOCALAPPDATA");
    if (!base || !*base) return false;
    const fs::path root = resolved(fs::path(base) / "Programs" / "speech.cpp");
    const char * name = "speech.exe";
#else
    const char * home = std::getenv("HOME");
    if (!home || !*home) return false;
    const fs::path root = resolved(fs::path(home) / ".local" / "share" / "speech.cpp");
    const char * name = "speech";
#endif
    const fs::path version = self.parent_path();
    return !root.empty() && self.filename() == name && version.parent_path() == root && is_version(version.filename().u8string());
}

/** Runs the installer with stdout and stderr on this process's stderr, waits for it, and returns its exit code. */
int run_installer() {
#ifdef _WIN32
    // System32's PowerShell, so that nothing in the current folder runs in its place.
    wchar_t system[MAX_PATH];
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) throw io_failure("cannot find the System32 folder, where PowerShell is");
    const std::wstring program = std::wstring(system) + L"\\WindowsPowerShell\\v1.0\\powershell.exe";
    std::wstring line = L"\"" + program +
                        L"\" -NoProfile -ExecutionPolicy Bypass -Command \"& ([scriptblock]::Create((irm "
                        L"https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.ps1))) -NoModifyPath\"";

    SECURITY_ATTRIBUTES inherit = {sizeof inherit, nullptr, TRUE};
    HANDLE err = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetStdHandle(STD_ERROR_HANDLE), GetCurrentProcess(), &err, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
        throw io_failure("cannot give stderr to the installer (Windows error " + std::to_string(GetLastError()) + ")");
    }
    HANDLE null = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    HANDLE handles[2] = {null, err};
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<char> list(bytes);
    STARTUPINFOEXW info = {};
    info.StartupInfo.cb = sizeof info;
    info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdInput = null;
    info.StartupInfo.hStdOutput = err;
    info.StartupInfo.hStdError = err;
    info.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list.data());
    InitializeProcThreadAttributeList(info.lpAttributeList, 1, 0, &bytes);
    UpdateProcThreadAttribute(info.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, sizeof handles, nullptr, nullptr);
    PROCESS_INFORMATION process = {};
    const BOOL started = CreateProcessW(program.c_str(), line.data(), nullptr, nullptr, TRUE, EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                                        &info.StartupInfo, &process);
    const DWORD error = GetLastError();
    DeleteProcThreadAttributeList(info.lpAttributeList);
    CloseHandle(null);
    CloseHandle(err);
    if (!started) throw io_failure("cannot start " + fs::path(program).u8string() + " to run the installer (Windows error " + std::to_string(error) + ")");
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return (int) code;
#else
    // `curl ... | sh` ends with sh's status, which is 0 when curl fails and sh reads nothing, and POSIX sh has no
    // pipefail; the installer is read first, so that a download that fails ends the script with curl's status.
    const std::string script =
        "installer=$(curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh) || exit $?\n"
        "printf '%s\\n' \"$installer\" | sh -s -- --no-modify-path";
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, 2, 1);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
#ifdef __APPLE__
    posix_spawn_file_actions_addinherit_np(&actions, 2);
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
#else
    posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
    char * argv[] = {const_cast<char *>("sh"), const_cast<char *>("-c"), const_cast<char *>(script.c_str()), nullptr};
    pid_t pid = 0;
    const int error = posix_spawn(&pid, "/bin/sh", &actions, &attributes, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    if (error != 0) throw io_failure("cannot start /bin/sh to run the installer: " + std::generic_category().message(error));
    int status = 0;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) throw io_failure("cannot wait for the installer: " + std::generic_category().message(errno));
    }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
}

int run_update(const CommandLine & line, FILE *) {
    const std::string current = speech_version();
    const std::string latest = latest_release();
    if (!later_version(latest, current)) {
        if (latest == current) std::fprintf(stderr, "speech.cpp %s is the latest release\n", current.c_str());
        else std::fprintf(stderr, "this is speech.cpp %s, and the latest release is %s\n", current.c_str(), latest.c_str());
        return 0;
    }
    if (line.has("--check")) {
        std::fprintf(stderr, "%s; speech update installs it\n", newer_release(latest).c_str());
        return 0;
    }
    const fs::path self = this_executable();
    if (!put_by_installer(self)) {
#ifdef _WIN32
        const std::string installer = kInstallPs1;
#else
        const std::string installer = kInstallSh;
#endif
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      newer_release(latest) + ", but this speech, " + self.u8string() +
                          ", was not put in place by speech.cpp's installer, so speech update leaves it alone; install the release with the "
                          "installer, `" + installer + "`, or pull and build speech.cpp again if you built it from source");
    }
    std::fprintf(stderr, "%s; running the installer\n", newer_release(latest).c_str());
    std::fflush(stderr);
    return run_installer();
}

}  // namespace

NewerReleaseNotice::NewerReleaseNotice(ReleaseNotice when) : when_(when) {
    if (when == ReleaseNotice::Never || !notices_wanted()) return;
    check_ = std::make_unique<ReleaseCheck>();
    const std::string latest = check_->latest();
    if (when == ReleaseNotice::AtStart && later_version(latest, speech_version())) {
        std::fprintf(stderr, "speech: %s; run speech update to install it\n", newer_release(latest).c_str());
    }
}

void NewerReleaseNotice::ended() {
    if (!check_) return;
    const std::string latest = check_->finish();
    if (when_ == ReleaseNotice::AtEnd && later_version(latest, speech_version())) {
        std::fprintf(stderr, "speech: %s; run speech update to install it\n", newer_release(latest).c_str());
    }
}

Command update_command() {
    Command c;
    c.name = "update";
    c.usage = "update [--check]";
    c.summary = "install the latest release of speech.cpp, or say whether one is out";
    c.description =
        "Asks github.com for the latest release of speech.cpp. When it is later than this one and this speech is the one\n"
        "speech.cpp's installer put in place, runs the installer, which downloads it, checks it against the release's\n"
        "SHA256SUMS and replaces this version, and exits with the installer's status. A speech built from source or put\n"
        "elsewhere is left alone, with a failure that says how to update it. Messages go to stderr.";
    c.flags = {
        {"--check", "", false, "only say whether a later release is out, and install nothing"},
    };
    c.run = run_update;
    return c;
}
