// Checks Drama 3's tokenizer, prompt and chunking against fish-speech's own on the cases of
// reference/drama3/prompt_cases.py: the ids of each text, the rows of each prompt, or that a prompt the official builder
// refuses is refused with an Error of the caller's argument, and the chunks of each text.
//
// usage: drama3-prompt-check <drama3 model.gguf> <prompt-cases.jsonl>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "args.h"
#include "drama3/chunking.h"
#include "drama3/layout.h"
#include "drama3/prompt.h"
#include "error.h"
#include "json-reader.h"

using namespace drama3;

namespace {

const JsonValue & member(const JsonValue & object, const std::string & name) {
    const JsonValue * v = object.member(name);
    if (!v) throw std::runtime_error("a case has no " + name);
    return *v;
}

std::vector<int32_t> ints(const JsonValue & array) {
    std::vector<int32_t> out;
    for (const JsonValue & v : array.items) out.push_back(std::stoi(v.text));
    return out;
}

std::vector<Utterance> utterances(const JsonValue & array) {
    std::vector<Utterance> out;
    for (const JsonValue & u : array.items) out.push_back({member(u, "text").text, ints(member(u, "codes")), std::stoll(member(u, "frames").text)});
    return out;
}

void print_ids(const char * what, const std::vector<int32_t> & ids) {
    std::printf("  %s", what);
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <drama3 model.gguf> <prompt-cases.jsonl>\n", args[0].c_str());
        return 2;
    }
    const ModelFile model(args[1], model_layout);
    const Prompt prompt(model);
    const size_t chunk_bytes = model.size("drama3.generation.chunk_bytes"), chunk_speakers = model.size("drama3.generation.chunk_speakers");
    std::ifstream in(std::filesystem::u8path(args[2]));
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", args[2].c_str());
        return 2;
    }
    int counts[3] = {0, 0, 0}, failed[3] = {0, 0, 0};
    std::string line;
    while (std::getline(in, line)) {
        const JsonValue c = parse_json(line);
        const std::string kind = member(c, "kind").text, text = member(c, "text").text;
        if (kind == "encode") {
            counts[0]++;
            const std::vector<int32_t> want = ints(member(c, "ids")), got = prompt.tokenize(text);
            if (got != want) {
                failed[0]++;
                std::printf("MISMATCH encoding: %s\n", text.c_str());
                print_ids("want:", want);
                print_ids("got: ", got);
            }
        } else if (kind == "prompt") {
            counts[1]++;
            const JsonValue * error = c.member("error");
            try {
                const std::vector<int32_t> got = prompt.rows(text, utterances(member(c, "references")), utterances(member(c, "history")));
                if (error || got != ints(member(c, "rows"))) {
                    failed[1]++;
                    std::printf("MISMATCH prompt: %s: %s\n", text.c_str(), error ? ("written where the official builder says: " + error->text).c_str() : "other rows");
                }
            } catch (const Error & e) {
                const bool refused = error && e.fault() == Fault::InvalidArgument;
                if (!refused) {
                    failed[1]++;
                    std::printf("MISMATCH prompt: %s: refused (%s) where the official builder writes it\n", text.c_str(), e.what());
                } else {
                    std::printf("refused as the official builder refuses it: %s\n  %s (%s)\n", text.c_str(), e.what(), e.input().c_str());
                }
            }
        } else if (kind == "chunks") {
            counts[2]++;
            std::vector<std::string> want;
            for (const JsonValue & v : member(c, "chunks").items) want.push_back(v.text);
            const std::vector<std::string> got = split_text(text, chunk_bytes, chunk_speakers);
            if (got != want) {
                failed[2]++;
                std::printf("MISMATCH chunks of %s\n", text.substr(0, 60).c_str());
                for (size_t i = 0; i < std::max(got.size(), want.size()); i++) {
                    std::printf("  %zu want: %s\n  %zu got:  %s\n", i, i < want.size() ? want[i].c_str() : "-", i, i < got.size() ? got[i].c_str() : "-");
                }
            }
        }
    }
    const char * what[3] = {"texts encode", "prompts are written or refused", "texts are cut into chunks"};
    for (int k = 0; k < 3; k++) std::printf("%d of %d %s as fish-speech's\n", counts[k] - failed[k], counts[k], what[k]);
    return failed[0] + failed[1] + failed[2] == 0 && counts[0] > 0 && counts[1] > 0 && counts[2] > 0 ? 0 : 1;
}
