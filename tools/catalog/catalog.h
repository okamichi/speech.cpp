#pragma once

#include <cstdint>
#include <string>
#include <vector>

// The catalog of the models a release fetches by name, read from tools/catalog/catalog.json as the build copies it into
// the executable: each model's name, its Hugging Face repository pinned at a revision, its files with their sizes and
// SHA-256, its task and languages, and the languages it is the model to start with. A model argument is a path to a
// GGUF file, or NAME[:TYPE], NAME being a model's name or its repository.

/** A file of a model in its repository at the catalog's revision. */
struct CatalogFile {
    /** The weight type in lower case, such as "q8_0" or "f16", which a model argument gives after ":". */
    std::string type;
    /** The file's name in the repository. */
    std::string file;
    uint64_t size = 0;
    /** The SHA-256 of its bytes, 64 lower-case hexadecimal digits. */
    std::string sha256;
};

struct CatalogModel {
    std::string name;
    /** The Hugging Face repository, "owner/name". */
    std::string repository;
    /** The repository's commit that the files are pinned at. */
    std::string revision;
    /** "synthesis", "recognition" or "detection", as the model information names it. */
    std::string task;
    /** The languages the model takes, as its file's general.languages names them. */
    std::vector<std::string> languages;
    /** Whether the model takes voice files, which `speech voice` makes. */
    bool voice_files = false;
    /** The languages for which it is the model of its task to start with. */
    std::vector<std::string> start;
    /** The type of the file a model argument without a type names. */
    std::string type;
    std::vector<CatalogFile> files;
};

/** A model of the catalog and one of its files. */
struct CatalogChoice {
    const CatalogModel * model = nullptr;
    const CatalogFile * file = nullptr;
};

/** Which models fit the first argument of a subcommand. */
enum class ModelKind { Any, Synthesis, Recognition, Detection, VoiceFiles };

/** The catalog of this release, in the order of catalog.json. */
const std::vector<CatalogModel> & catalog();

/** Whether a model argument is a path to a model file, which ends in ".gguf" in any case, rather than a name. */
bool is_model_path(const std::string & argument);

/** How a line names a model argument before the model is loaded: a path by its file's name, a name as it is given. */
std::string argument_label(const std::string & argument);

/** Whether a model argument is a path or a NAME of the catalog, whatever type follows it. */
bool names_a_model(const std::string & argument);

/**
 * The file that NAME[:TYPE] names: without a type, the model's `type`. A name or a type the catalog does not hold
 * throws a UsageError that lists what it holds.
 */
CatalogChoice find_model(const std::string & argument);

/** NAME[:TYPE] as `speech models` writes it: the name alone for the model's own type. */
std::string choice_name(const CatalogChoice & choice);

/** Where Hugging Face serves the file at the catalog's revision. */
std::string file_url(const CatalogChoice & choice);

/**
 * What a subcommand given no model says: what MODEL is, the models of `kind` to start with by language, and the
 * subcommand's usage with the first of them, for `usage` such as "asr MODEL [options] AUDIO.wav...".
 */
std::string no_model_message(const std::string & usage, ModelKind kind);
