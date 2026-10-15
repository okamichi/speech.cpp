#include "curl.h"

#include <cerrno>
#include <chrono>
#include <system_error>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char ** environ;
#endif

#include "failure.h"
#include "speech.h"

namespace {

Failure io_failure(const std::string & message) {
    return Failure(speech_status_name(SPEECH_ERROR_IO), "", message);
}

}  // namespace

size_t read_pipe(PipeEnd pipe, char * buffer, size_t size) {
#ifdef _WIN32
    DWORD n = 0;
    return ReadFile(pipe, buffer, (DWORD) size, &n, nullptr) ? n : 0;
#else
    for (;;) {
        const ssize_t n = ::read(pipe, buffer, size);
        if (n >= 0) return (size_t) n;
        if (errno != EINTR) return 0;
    }
#endif
}

std::string read_to_end(PipeEnd pipe) {
    std::string out;
    char buffer[4096];
    while (const size_t n = read_pipe(pipe, buffer, sizeof buffer)) out.append(buffer, n);
    close_pipe(pipe);
    return out;
}

void close_pipe(PipeEnd pipe) {
#ifdef _WIN32
    CloseHandle(pipe);
#else
    ::close(pipe);
#endif
}

Curl::Curl(const std::vector<std::string> & args, const std::string & missing) {
#ifdef _WIN32
    // A bare "curl.exe" would be looked for in the current folder before System32.
    wchar_t system[MAX_PATH];
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    if (length == 0 || length >= MAX_PATH) throw io_failure("cannot find the System32 folder, where curl.exe is");
    const std::wstring program = std::wstring(system) + L"\\curl.exe";
    std::wstring line = L"\"" + program + L"\"";
    for (const std::string & a : args) line += L" \"" + std::wstring(a.begin(), a.end()) + L"\"";

    SECURITY_ATTRIBUTES inherit = {sizeof inherit, nullptr, TRUE};
    HANDLE out_write = nullptr, err_write = nullptr;
    if (!CreatePipe(&out_, &out_write, &inherit, 0) || !CreatePipe(&err_, &err_write, &inherit, 0)) {
        throw io_failure("cannot make a pipe for curl (Windows error " + std::to_string(GetLastError()) + ")");
    }
    SetHandleInformation(out_, HANDLE_FLAG_INHERIT, 0);
    SetHandleInformation(err_, HANDLE_FLAG_INHERIT, 0);
    HANDLE null = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
    HANDLE handles[3] = {null, out_write, err_write};
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<char> list(bytes);
    STARTUPINFOEXW info = {};
    info.StartupInfo.cb = sizeof info;
    info.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    info.StartupInfo.hStdInput = null;
    info.StartupInfo.hStdOutput = out_write;
    info.StartupInfo.hStdError = err_write;
    info.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(list.data());
    InitializeProcThreadAttributeList(info.lpAttributeList, 1, 0, &bytes);
    UpdateProcThreadAttribute(info.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, handles, sizeof handles, nullptr, nullptr);
    PROCESS_INFORMATION process = {};
    // CREATE_NO_WINDOW keeps a speech without a console, started by a program that hid it, from opening a window.
    const BOOL started = CreateProcessW(program.c_str(), line.data(), nullptr, nullptr, TRUE, EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
                                        nullptr, nullptr, &info.StartupInfo, &process);
    const DWORD error = GetLastError();
    DeleteProcThreadAttributeList(info.lpAttributeList);
    CloseHandle(null);
    CloseHandle(out_write);
    CloseHandle(err_write);
    if (!started) {
        close_pipe(out_);
        close_pipe(err_);
        if (error == ERROR_FILE_NOT_FOUND) throw io_failure(missing);
        throw io_failure("cannot start curl (Windows error " + std::to_string(error) + ")");
    }
    CloseHandle(process.hThread);
    process_ = process.hProcess;
#else
    int out[2], err[2];
    if (pipe(out) != 0 || pipe(err) != 0) throw io_failure("cannot make a pipe for curl: " + std::generic_category().message(errno));
    fcntl(out[0], F_SETFD, FD_CLOEXEC);
    fcntl(err[0], F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, out[1], 1);
    posix_spawn_file_actions_adddup2(&actions, err[1], 2);
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
#ifdef __APPLE__
    posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT);
#else
    posix_spawn_file_actions_addclosefrom_np(&actions, 3);
#endif
    std::vector<char *> argv{const_cast<char *>("curl")};
    for (const std::string & a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    const int error = posix_spawnp(&pid_, "curl", &actions, &attributes, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attributes);
    ::close(out[1]);
    ::close(err[1]);
    out_ = out[0];
    err_ = err[0];
    if (error != 0) {
        ::close(out_);
        ::close(err_);
        if (error == ENOENT) throw io_failure(missing);
        throw io_failure("cannot start curl: " + std::generic_category().message(error));
    }
#endif
}

Curl::~Curl() {
#ifdef _WIN32
    if (process_) CloseHandle(process_);
#endif
}

bool Curl::wait(int ms, int & code) {
#ifdef _WIN32
    if (WaitForSingleObject(process_, (DWORD) ms) != WAIT_OBJECT_0) return false;
    DWORD exit = 0;
    GetExitCodeProcess(process_, &exit);
    code = (int) exit;
    return true;
#else
    for (int waited = 0;; waited += 10) {
        int status = 0;
        if (waitpid(pid_, &status, WNOHANG) == pid_) {
            code = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
            return true;
        }
        if (waited >= ms) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#endif
}

void Curl::stop() {
#ifdef _WIN32
    TerminateProcess(process_, 1);
#else
    kill(pid_, SIGTERM);
#endif
}

const char * null_device() {
#ifdef _WIN32
    return "NUL";
#else
    return "/dev/null";
#endif
}
