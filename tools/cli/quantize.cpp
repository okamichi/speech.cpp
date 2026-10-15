#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>

#include "commands.h"

// speech quantize: writes a model file of F32 weights in another weight type through speech_quantize(), into a file
// or into a folder under the name GGUF's naming convention gives it, as the converters name theirs.

namespace {

/**
 * The name GGUF's naming convention gives the file of `model`'s model in `type`, as gguf-py's naming_convention() writes
 * it for the converters: <basename>-<size label>-<finetune>-<version>-<TYPE>.gguf without the parts the model has none of.
 */
std::string conventional_name(const speech_model_info * model, const std::string & type) {
    const char * finetune = speech_model_info_finetune(model), * version = speech_model_info_version(model);
    std::string name = std::string(speech_model_info_basename(model)) + "-" + speech_model_info_size_label(model);
    if (finetune) name += std::string("-") + finetune;
    if (version) name += std::string("-") + version;
    name += "-";
    for (char c : type) name += (char) std::toupper((unsigned char) c);
    return name + ".gguf";
}

int run_quantize(const CommandLine & line, FILE *) {
    const std::optional<std::string> type = line.value("--type");
    if (!type) throw UsageError("give the type to write with --type: f16, q8_0, q6_k, q5_k or q4_k");
    const std::string & in = line.args[0];
    std::string out = line.args[1];
    std::error_code error;
    if (std::filesystem::is_directory(std::filesystem::u8path(out), error)) {
        const ModelInfo info = file_info(in);
        out = (std::filesystem::u8path(out) / std::filesystem::u8path(conventional_name(info.get(), *type))).u8string();
    }
    const auto t0 = std::chrono::steady_clock::now();
    check(speech_quantize(in.c_str(), type->c_str(), out.c_str()));
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "wrote %s in %.1f s\n", out.c_str(), took);
    return 0;
}

}  // namespace

Command quantize_command() {
    Command c;
    c.name = "quantize";
    c.usage = "quantize MODEL OUT --type f16|q8_0|q6_k|q5_k|q4_k";
    c.summary = "write a model file of F32 weights in another weight type";
    c.description =
        "Writes MODEL, a model file of F32 weights as the converters write it, with its weights in another type, on the CPU,\n"
        "each tensor in the type its family's layout gives it in a file of that type: a matrix the model multiplies by in\n"
        "TYPE, or in Q8_0 instead of a K-quant where its rows are not whole blocks of 256 values and in F16 instead of Q8_0\n"
        "where they are not whole blocks of 32, and the rest as every file of the model holds it. OUT is the file to write,\n"
        "or a folder, in which the file takes the name GGUF's naming convention gives it, such as\n"
        "Qwen3-ASR-0.6B-Q8_0.gguf. A file of Q6_K, Q5_K or Q4_K needs speech.cpp 0.8.0 or later to read it.";
    c.flags = {
        {"--type", "TYPE", false, "the weight type to write: f16, q8_0, q6_k, q5_k or q4_k"},
    };
    c.min_args = 2;
    c.max_args = 2;
    c.run = run_quantize;
    c.release_notice = ReleaseNotice::AtEnd;
    return c;
}
