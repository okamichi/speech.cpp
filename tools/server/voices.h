#pragma once

#include <filesystem>
#include <string>

#include "speech.h"

namespace httplib { class Server; }

namespace server {

class ServedModels;
/** Installs the voice list and upload endpoints used by the Irodori bridge. */
void route_voices(httplib::Server & http, ServedModels & models);

class Served;

/** Refuses a synthesis model that cannot fulfill the bridge's Irodori requests. */
void require_irodori_compatible(const speech_model_info * info);

/** WAV files owned by the server, restored into every synthesis model before it is published. */
class VoiceStore {
public:
    explicit VoiceStore(const std::string & directory);
    bool persistent() const { return !directory_.empty(); }
    void restore(Served & served) const;
    /** Commits a recording and adds it to the model, returning prepared upload metadata. */
    std::string add(Served & served, const std::string & name, const std::string & recording) const;

private:
    std::filesystem::path directory_;
};

}  // namespace server
