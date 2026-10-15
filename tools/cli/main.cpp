// speech: speech synthesis, speech recognition and the detection of speech with every model speech.cpp runs, through
// its C API, as one executable with a subcommand per program: tts, asr and vad for the command line, voice, info and
// devices about models and devices, models, pull and rm about the models it fetches by name, quantize to write a model file in another
// weight type, update to install the latest release, serve for OpenAI's audio API over HTTP and worker for JSON Lines on
// stdin and stdout. On a terminal, the subcommands for people say on stderr when a later release is out, unless
// SPEECH_NO_UPDATE_CHECK is set; the worker never does.
//
// usage: speech <subcommand> [options]     speech <subcommand> --help     speech --version
//
// It exits with 0 when done, 1 on a failure, which it reports as "speech: <code> (<option>): <message>" with the
// library's category, 2 on a command line it cannot run, and 3 when `speech tts` or `speech asr` stopped a request at
// the most the model makes: the longest speech, or the most tokens of a text. Its stdout carries its output alone:
// whatever ggml or a driver prints there goes to stderr.

#include <cstdio>
#include <exception>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "args.h"
#include "commands.h"
#include "take-stdout.h"
#include "update.h"

namespace {

std::vector<Command> commands() {
    return {tts_command(), asr_command(), vad_command(), voice_command(), info_command(), devices_command(), models_command(), pull_command(), rm_command(),
            quantize_command(), update_command(), serve_command(), worker_command()};
}

std::string overview() {
    std::string out = "usage: speech <subcommand> [options]\n\n";
    for (const Command & c : commands()) out += "  " + c.name + std::string(10 - c.name.size(), ' ') + c.summary + "\n";
    out += "\n  speech <subcommand> --help   the subcommand's usage and options\n"
           "  speech --version             the release and the version of the C API\n";
    return out;
}

void report(const Failure & e) {
    std::fprintf(stderr, "speech: %s%s: %s\n", e.code().c_str(), e.option().empty() ? "" : (" (" + e.option() + ")").c_str(), e.what());
}

int run(const std::vector<std::string> & args, FILE * out) {
    if (args.size() < 2) {
        std::fputs(overview().c_str(), stderr);
        return 2;
    }
    const std::string & name = args[1];
    if (name == "--version") {
        std::fprintf(out, "speech.cpp %s, C API %d.%d\n", speech_version(), speech_api_version_major(), speech_api_version_minor());
        return 0;
    }
    if (name == "--help" || name == "-h") {
        std::fputs(overview().c_str(), out);
        return 0;
    }
    const std::vector<Command> all = commands();
    const Command * command = nullptr;
    for (const Command & c : all) {
        if (c.name == name) command = &c;
    }
    if (!command) {
        std::fprintf(stderr, "speech: there is no subcommand %s\n\n%s", name.c_str(), overview().c_str());
        return 2;
    }
    const std::vector<std::string> rest(args.begin() + 2, args.end());
    if (asks_for_help(rest)) {
        std::fputs(command_help(*command).c_str(), out);
        return 0;
    }
    std::optional<NewerReleaseNotice> notice;
    int status = 0;
    try {
        const CommandLine line = parse_command_line(*command, rest);
        notice.emplace(command->release_notice);
        status = command->run(line, out);
    } catch (const UsageError & e) {
        std::fprintf(stderr, "speech: %s\nRun speech %s --help for its usage and options.\n", e.what(), command->name.c_str());
        if (command->report) command->report(Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "", e.what()), out);
        return 2;
    } catch (const Failure & e) {
        report(e);
        if (command->report) command->report(e, out);
        status = 1;
    } catch (const std::exception & e) {
        const Failure internal(speech_status_name(SPEECH_ERROR_INTERNAL), "", e.what());
        report(internal);
        if (command->report) command->report(internal, out);
        status = 1;
    }
    // The subcommand's output is flushed first, so that a notice on a terminal comes after it.
    std::fflush(out);
    if (notice) notice->ended();
    return status;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    FILE * out = nullptr;
    try {
        out = take_stdout();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "speech: io: %s\n", e.what());
        return 1;
    }
    const int status = run(utf8_args(argc, argv), out);
    std::fflush(out);
    return status;
}
