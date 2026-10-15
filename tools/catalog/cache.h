#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "catalog.h"

// The folder the catalog's files are fetched into, the system's cache folder unless SPEECH_MODEL_DIR names another:
// <folder>/<owner>--<name>/<sha256>/<file>, with <file>.part while it is fetched and <file>.lock, which a process
// holds while it fetches or removes the file. A file is kept by its content, so a catalog that pins its repository at
// a later commit with the same file finds it where it is. Nothing in the folder is removed but on request: a file no
// entry of this release's catalog names is old, and `speech rm --old` removes it.

/**
 * speech.cpp's folder in the system's cache folder: ~/Library/Caches/speech.cpp on macOS, $XDG_CACHE_HOME/speech.cpp or
 * ~/.cache/speech.cpp on Linux, and %LOCALAPPDATA%\speech.cpp on Windows. It holds the models unless SPEECH_MODEL_DIR
 * names another folder, and what the command line keeps of the latest release, wherever the models are.
 */
std::filesystem::path cache_dir();

/** The folder: SPEECH_MODEL_DIR, an absolute path, or models in cache_dir(). */
std::filesystem::path model_dir();

/** Where a file of the catalog is, or goes once fetched. */
std::filesystem::path model_path(const CatalogChoice & choice);

/** The file that holds what has been fetched of `path` so far. */
std::filesystem::path partial_path(const std::filesystem::path & path);

/** How much of a catalog file is in the folder: the whole of it, or the bytes of a fetch that stopped, or nothing. */
struct Presence {
    bool whole = false;
    uint64_t partial = 0;
};

Presence presence(const CatalogChoice & choice);

/** A file in the folder that is no file of the catalog nor a partial fetch of one: an earlier release's. */
struct OldFile {
    std::filesystem::path path;
    uint64_t size = 0;
};

std::vector<OldFile> old_files();

/**
 * An exclusive lock on `<file>.lock`, which every process that fetches or removes the file holds, waiting for another
 * that holds it; `waiting` is called once if it has to wait. The lock file is created when it is taken and removed
 * before it is released, and one that another process removed while this one waited is made again, so that two
 * processes never hold different files of the same name.
 */
class FileLock {
public:
    FileLock(const std::filesystem::path & file, const std::function<void()> & waiting);
    ~FileLock();
    FileLock(const FileLock &) = delete;
    FileLock & operator=(const FileLock &) = delete;

private:
    std::filesystem::path path_;
#ifdef _WIN32
    void * handle_ = nullptr;
#else
    int fd_ = -1;
#endif
};

/** Removes a file of the folder and a partial fetch of it, holding its lock, and the folders that leaves empty. */
uint64_t remove_from_folder(const std::filesystem::path & path);

/**
 * The catalog with where each file goes and how much of it is there, and the old files, as one JSON object:
 * {"version", "directory", "models": [{"name", "repository", "revision", "task", "languages", "voice_files", "start",
 * "type", "files": [{"type", "file", "size", "sha256", "url", "path", "fetched", "partial"}]}], "old": [{"path", "size"}]}.
 */
std::string models_json();
