#pragma once

#include <ctime>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "jobs.h"
#include "library.h"
#include "voices.h"
#include "request-options.h"

// The models the server holds: at most one for speech synthesis, one for speech recognition and one for the detection of
// speech, which cuts the audio of a transcription into the regions where someone speaks, so that a page can speak and
// transcribe with one server while no more than one model of a task takes memory. A model given on the command line is
// loaded before the server listens; the page replaces the one of a task by a model of the catalog.

namespace server {

/** A model the server holds, which each request that runs on it keeps until it ends. */
class Served {
public:
    Served(Model model, std::string path, std::string catalog_name, double load_seconds);

    speech_model * get() const { return model_.get(); }
    /** The file the model was loaded from. */
    const std::string & path() const { return path_; }
    /** The catalog's NAME[:TYPE] the page loaded it by, or "" for a file the command line gave. */
    const std::string & catalog_name() const { return catalog_name_; }
    long long created() const { return created_; }
    /** The seconds its load took, its warm-up included. */
    double load_seconds() const { return load_seconds_; }
    /** The order in which the requests on this model take it. */
    Turns & turns() { return turns_; }

    /** The model's information as of its last voice. */
    std::shared_ptr<const speech_model_info> info() const;
    /** Adds a voice from a voice file or a WAVE file, and reads the information again, which then lists it. */
    void add_voice(const std::string & name, const std::string & file);

    /**
     * Gives the model the defaults of the command line's request options that it takes, checked against it, before the
     * server listens; a model the page loads later keeps its own.
     */
    void set_defaults(std::vector<RequestOption> defaults) { defaults_ = std::move(defaults); }
    const std::vector<RequestOption> & defaults() const { return defaults_; }
    /** A request's options on this model after its defaults (with_defaults()). */
    std::vector<RequestOption> with_defaults(const std::vector<RequestOption> & options) const {
        return ::with_defaults(defaults_, options);
    }

private:
    Model model_;
    std::string path_, catalog_name_;
    long long created_ = (long long) std::time(nullptr);
    double load_seconds_;
    Turns turns_;
    mutable std::mutex mutex_;
    std::shared_ptr<const speech_model_info> info_;
    std::vector<RequestOption> defaults_;
};

/**
 * The model of each task, and a replacement under way: while the page fetches a model, the one it replaces still serves;
 * once the file is there, the old model is let go, its requests end, and the new one is loaded in its place.
 */
class ServedModels {
public:
    /**
     * `loading` is how the synthesis and recognition models load: the device, the threads and the warm-up, which a
     * detection model takes on the CPU with one thread (detection_loading()); its voices go to no model.
     */
    explicit ServedModels(Loading loading, const std::string & directory = "", bool compatible = false);
    bool compatible() const { return compatible_; }
    /** Registers an upload on the current synthesis model, serialized with model replacement. */
    std::pair<std::shared_ptr<Served>, std::string> upload_voice(const std::string & name, const std::string & bytes);
    /** The health fields from one consistent snapshot of the synthesis place. */
    std::string health() const;

    /**
     * Loads a model given on the command line in the place of its task, which must be empty: a file, or the catalog's
     * file of `catalog_name`, "" for a file named by its path.
     */
    std::shared_ptr<Served> load_given(const std::string & path, const std::string & catalog_name);

    /** The model of a task, or nullptr while there is none or one is being loaded in its place. */
    std::shared_ptr<Served> of(speech_task task) const;
    /** Every model held: synthesis, recognition, then detection. */
    std::vector<std::shared_ptr<Served>> all() const;
    /** The NAME[:TYPE] of the model the page is replacing a task's with, or "". */
    std::string replacing(speech_task task) const;

    /** Marks a task's model as being replaced by `catalog_name`; false when another replacement of it is under way. */
    bool begin_replacing(speech_task task, const std::string & catalog_name);
    /** Ends a replacement that did not get as far as the load, leaving the model held as it was. */
    void cancel_replacing(speech_task task);
    /**
     * Lets the task's model go, waits for its requests to end, and loads the file at `path` in its place; a failure to
     * load leaves the task without a model and throws.
     */
    std::shared_ptr<Served> replace(speech_task task, const std::string & path);

private:
    struct Place {
        std::shared_ptr<Served> model;
        std::string replacing;
    };

    Loading loading_of(speech_task task) const;

    Place & place(speech_task task) {
        switch (task) {
            case SPEECH_TASK_SYNTHESIS: return synthesis_;
            case SPEECH_TASK_RECOGNITION: return recognition_;
            case SPEECH_TASK_DETECTION: return detection_;
        }
        throw std::logic_error("a task speech.h does not have");
    }
    const Place & place(speech_task task) const { return const_cast<ServedModels *>(this)->place(task); }

    Loading loading_;
    VoiceStore voices_;
    bool compatible_ = false;
    std::mutex changing_;
    mutable std::mutex mutex_;
    Place synthesis_, recognition_, detection_;
};

}  // namespace server
