#include "served-models.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>

#include "json.h"
#include "openai-error.h"

namespace server {

namespace {

std::shared_ptr<const speech_model_info> read_info(const speech_model * model) {
    ModelInfo info = model_info(model);
    return std::shared_ptr<const speech_model_info>(info.release(), [](const speech_model_info * i) {
        speech_model_info_free(const_cast<speech_model_info *>(i));
    });
}

}  // namespace

Served::Served(Model model, std::string path, std::string catalog_name)
    : model_(std::move(model)), path_(std::move(path)), catalog_name_(std::move(catalog_name)), info_(read_info(model_.get())) {}

std::shared_ptr<const speech_model_info> Served::info() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return info_;
}

void Served::add_voice(const std::string & name, const std::string & file) {
    Turn turn(turns_);
    turn.wait();
    const auto owner = std::make_shared<ModelInfo>(nullptr, speech_model_info_free);
    speech_model_info * raw = nullptr;
    check(speech_voice_add_with_info(model_.get(), name.c_str(), file.c_str(), &raw));
    owner->reset(raw);
    std::shared_ptr<const speech_model_info> info(owner, raw);
    std::lock_guard<std::mutex> lock(mutex_);
    info_ = std::move(info);
}

ServedModels::ServedModels(Loading loading, const std::string & directory, bool compatible)
    : loading_(std::move(loading)), voices_(directory), compatible_(compatible) {
    if (compatible && directory.empty()) throw UsageError("--irodori-compatible needs --voices-dir DIR to retain registered voices");
    loading_.voices.clear();
}

std::pair<std::shared_ptr<Served>, std::string> ServedModels::upload_voice(const std::string & name, const std::string & bytes) {
    std::lock_guard<std::mutex> operation(changing_);
    auto served = of(SPEECH_TASK_SYNTHESIS);
    if (!served) throw openai::ApiError{503, "No synthesis model is ready; load one before registering a voice.", "", "model_loading"};
    std::string metadata = voices_.add(*served, name, bytes);
    return {std::move(served), std::move(metadata)};
}

std::string ServedModels::health() const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto & served = synthesis_.model;
    const auto info = served ? served->info() : nullptr;
    const std::string model = served ? "{\"id\":" + json_string(compatible_ ? "irodori-tts" : speech_model_info_name(info.get())) +
                                      ",\"name\":" + json_string(speech_model_info_name(info.get())) + "}" : "null";
    return "{\"status\":\"ok\",\"model\":" + model + ",\"runtime\":{\"loaded\":" + (served ? "true" : "false") +
           ",\"loading\":" + (synthesis_.replacing.empty() ? "false" : "true") + ",\"checkpoint\":" +
           (served ? json_string(served->path()) : "null") + "}}";
}

Loading ServedModels::loading_of(speech_task task) const {
    return task == SPEECH_TASK_DETECTION ? detection_loading(loading_) : loading_;
}

void ServedModels::load_given(const std::string & path, const std::string & catalog_name) {
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task == SPEECH_TASK_SYNTHESIS && compatible_) require_irodori_compatible(file_info(path).get());
    auto served = std::make_shared<Served>(load_model(path, loading_of(task)), path, catalog_name);
    if (task == SPEECH_TASK_SYNTHESIS) voices_.restore(*served);
    std::lock_guard<std::mutex> lock(mutex_);
    Place & p = place(task);
    if (p.model) {
        throw UsageError(std::string("give one model of each task; ") + p.model->path() + " and " + path + " are both speech " + task_name(task) +
                         " models");
    }
    p.model = std::move(served);
}

std::shared_ptr<Served> ServedModels::of(speech_task task) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return place(task).model;
}

std::vector<std::shared_ptr<Served>> ServedModels::all() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Served>> out;
    for (const Place * p : {&synthesis_, &recognition_, &detection_}) {
        if (p->model) out.push_back(p->model);
    }
    return out;
}

std::string ServedModels::replacing(speech_task task) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return place(task).replacing;
}

bool ServedModels::begin_replacing(speech_task task, const std::string & catalog_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    Place & p = place(task);
    if (!p.replacing.empty()) return false;
    p.replacing = catalog_name;
    return true;
}

void ServedModels::cancel_replacing(speech_task task) {
    std::lock_guard<std::mutex> lock(mutex_);
    place(task).replacing.clear();
}

std::shared_ptr<Served> ServedModels::replace(speech_task task, const std::string & path) {
    std::lock_guard<std::mutex> operation(changing_);
    try {
        if (task == SPEECH_TASK_SYNTHESIS && compatible_) require_irodori_compatible(file_info(path).get());
    } catch (...) {
        cancel_replacing(task);
        throw;
    }
    std::weak_ptr<Served> old;
    std::string name;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Place & p = place(task);
        old = p.model;
        p.model.reset();
        name = p.replacing;
    }
    // The old model goes once the last request that runs on it lets it go, so that two models of a task never take
    // memory together.
    while (!old.expired()) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::shared_ptr<Served> served;
    try {
        served = std::make_shared<Served>(load_model(path, loading_of(task)), path, name);
        if (task == SPEECH_TASK_SYNTHESIS) voices_.restore(*served);
        if (speech_model_info_task(served->info().get()) != task) {
            throw Failure(speech_status_name(SPEECH_ERROR_MODEL_FILE), "", path + " is a speech " +
                          task_name(speech_model_info_task(served->info().get())) + " model, where the catalog says " + task_name(task));
        }
    } catch (...) {
        cancel_replacing(task);
        throw;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    Place & p = place(task);
    p.model = served;
    p.replacing.clear();
    return served;
}

}  // namespace server
