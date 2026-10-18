#include "catalog.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>

#include "failure.h"
#include "json-reader.h"
#include "spelling.h"

/** catalog.json's bytes and a terminating zero, written into the build by CMake. */
extern const unsigned char speech_catalog_json[];

namespace {

const JsonValue & member(const JsonValue & object, const char * name, JsonValue::Kind kind) {
    const JsonValue * v = object.member(name);
    if (!v || v->kind != kind) throw std::runtime_error(std::string("the catalog built into speech lacks a well-formed \"") + name + "\"");
    return *v;
}

std::string text(const JsonValue & object, const char * name) {
    return member(object, name, JsonValue::Kind::String).text;
}

std::vector<std::string> texts(const JsonValue & object, const char * name) {
    std::vector<std::string> out;
    for (const JsonValue & v : member(object, name, JsonValue::Kind::Array).items) {
        if (v.kind != JsonValue::Kind::String) throw std::runtime_error(std::string("the catalog's \"") + name + "\" holds a value that is not a string");
        out.push_back(v.text);
    }
    return out;
}

std::vector<CatalogModel> read_catalog() {
    const JsonValue root = parse_json(reinterpret_cast<const char *>(speech_catalog_json));
    std::vector<CatalogModel> out;
    for (const JsonValue & m : member(root, "models", JsonValue::Kind::Array).items) {
        CatalogModel model;
        model.name = text(m, "name");
        model.repository = text(m, "repository");
        model.type = text(m, "type");
        model.start = texts(m, "start");
        model.revision = text(m, "revision");
        model.task = text(m, "task");
        model.languages = texts(m, "languages");
        model.voice_files = member(m, "voice_files", JsonValue::Kind::Bool).boolean;
        for (const JsonValue & f : member(m, "files", JsonValue::Kind::Array).items) {
            const JsonValue & size = member(f, "size", JsonValue::Kind::Number);
            if (!size.is_integer()) throw std::runtime_error("the catalog gives a file's size as a number with a fraction");
            model.files.push_back({text(f, "type"), text(f, "file"), std::stoull(size.text), text(f, "sha256")});
        }
        out.push_back(std::move(model));
    }
    return out;
}

/** The model whose name or repository is `name`, or nullptr. */
const CatalogModel * model_named(const std::string & name) {
    for (const CatalogModel & m : catalog()) {
        if (m.name == name || m.repository == name) return &m;
    }
    return nullptr;
}

/** NAME and TYPE of NAME[:TYPE], TYPE "" when there is none. A repository holds no ":", so the last one splits. */
std::pair<std::string, std::string> split_argument(const std::string & argument) {
    const size_t colon = argument.rfind(':');
    if (colon == std::string::npos) return {argument, ""};
    return {argument.substr(0, colon), argument.substr(colon + 1)};
}

bool fits(const CatalogModel & m, ModelKind kind) {
    switch (kind) {
        case ModelKind::Synthesis: return m.task == "synthesis";
        case ModelKind::Recognition: return m.task == "recognition";
        case ModelKind::Detection: return m.task == "detection";
        case ModelKind::VoiceFiles: return m.voice_files;
        default: return true;
    }
}

std::string joined(const std::vector<std::string> & items, const char * separator) {
    std::string out;
    for (size_t i = 0; i < items.size(); i++) out += (i ? separator : "") + items[i];
    return out;
}

/** The catalog's names, each task's together: "a, b (synthesis); c (recognition)", a task without models left out. */
std::string names_by_task() {
    std::string out;
    for (const char * task : {"synthesis", "recognition", "detection"}) {
        std::vector<std::string> names;
        for (const CatalogModel & m : catalog()) {
            if (m.task == task) names.push_back(m.name);
        }
        if (!names.empty()) out += (out.empty() ? "" : "; ") + joined(names, ", ") + " (" + task + ")";
    }
    return out;
}

}  // namespace

const std::vector<CatalogModel> & catalog() {
    static const std::vector<CatalogModel> models = read_catalog();
    return models;
}

bool is_model_path(const std::string & argument) {
    if (argument.size() < 5) return false;
    std::string extension = argument.substr(argument.size() - 5);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return extension == ".gguf";
}

std::string argument_label(const std::string & argument) {
    return is_model_path(argument) ? std::filesystem::u8path(argument).filename().u8string() : argument;
}

bool names_a_model(const std::string & argument) {
    return is_model_path(argument) || model_named(split_argument(argument).first);
}

CatalogChoice find_model(const std::string & argument) {
    const auto [name, type] = split_argument(argument);
    const CatalogModel * model = model_named(name);
    if (!model) {
        std::vector<std::string> names;
        for (const CatalogModel & m : catalog()) names.push_back(m.name);
        const std::string near = nearest(name, names);
        throw UsageError("\"" + argument + "\" is neither a model file, whose path ends in .gguf, nor a model of the catalog" +
                         (near.empty() ? ": " : "; did you mean " + near + "? The catalog holds ") + names_by_task() +
                         ". `speech models` lists them with their types and languages");
    }
    const std::string wanted = type.empty() ? model->type : type;
    std::vector<std::string> types;
    for (const CatalogFile & f : model->files) {
        if (f.type == wanted) return {model, &f};
        types.push_back(f.type);
    }
    throw UsageError(model->name + " has no " + wanted + " file in this release's catalog; its files are of the types " + joined(types, ", ") +
                     ", and " + model->name + " alone names its " + model->type);
}

std::string choice_name(const CatalogChoice & choice) {
    return choice.file->type == choice.model->type ? choice.model->name : choice.model->name + ":" + choice.file->type;
}

std::string file_url(const CatalogChoice & choice) {
    return "https://huggingface.co/" + choice.model->repository + "/resolve/" + choice.model->revision + "/" + choice.file->file;
}

std::string no_model_message(const std::string & usage, ModelKind kind) {
    std::string out = "the model is missing: speech " + usage +
                      "\nMODEL is a model file (.gguf) or the name of one in `speech models`, which is fetched the first time it is used.\n";
    std::vector<const CatalogModel *> listed;
    // A subcommand of speech detection lists every detection model, which takes no language, and `speech voice` every
    // model that takes voice files, whatever the language; the others list the models to start with.
    const bool every = kind == ModelKind::VoiceFiles || kind == ModelKind::Detection;
    for (const CatalogModel & m : catalog()) {
        if (fits(m, kind) && (every || !m.start.empty())) listed.push_back(&m);
    }
    if (listed.empty()) return out + "The catalog of this release names no such model; give a model file.";
    out += kind == ModelKind::VoiceFiles  ? "The models that take voice files:\n"
           : kind == ModelKind::Detection ? "The models that detect speech:\n"
                                          : "No model is chosen for you; the one to start with depends on the language:\n";
    size_t width = 0;
    for (const CatalogModel * m : listed) width = std::max(width, m->name.size());
    for (size_t i = 0; i < listed.size(); i++) {
        const CatalogModel & m = *listed[i];
        // A subcommand of either task names each task once, before its first model.
        const bool any = kind == ModelKind::Any, first = i == 0 || listed[i - 1]->task != m.task;
        const std::string label = any ? (first ? m.task : "") + std::string(13 - (first ? m.task.size() : 0), ' ') : "";
        out += "  " + label + m.name + std::string(width + 2 - m.name.size(), ' ') + joined(kind == ModelKind::VoiceFiles ? m.languages : m.start, " ") +
               "\n";
    }
    const size_t at = usage.find("MODEL");
    out += "For example: speech " + usage.substr(0, at) + listed.front()->name + usage.substr(at + 5);
    return out;
}
