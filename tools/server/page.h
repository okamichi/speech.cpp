#pragma once

#include <string>

#include "httplib.h"

#include "access.h"
#include "served-models.h"

// The page of `speech serve`, on which a user picks a model of each task from the catalog, fetches it and tries it:
// its files, written in tools/server/page/ and built into speech, and the endpoints that only the page calls, under
// /speech/: the catalog with the models held, loading a model of the catalog in place of its task's, and adding a
// voice made from a recording. The page speaks and transcribes through OpenAI's endpoints, as any client does. While
// the server listens elsewhere than on a loopback address, the page and its endpoints say how to reach them instead.

namespace server {

class Page {
public:
    Page(ServedModels & models, const Access & access) : models_(models), access_(access) {}

    void route(httplib::Server & http);

private:
    /** Whether the request may reach the page and its endpoints; otherwise answers why not. */
    bool admits(const httplib::Request & req, httplib::Response & res) const;

    /** GET /speech/models: the catalog, where each file is and how much of it is there, and the models held. */
    void models(httplib::Response & res) const;
    /** POST /speech/load: fetches a model of the catalog with its progress, then loads it in place of its task's. */
    void load(const httplib::Request & req, httplib::Response & res);
    /** POST /speech/voices: adds a voice made from a WAVE file to the synthesis model held. */
    void add_voice(const httplib::Request & req, httplib::Response & res);

    ServedModels & models_;
    const Access & access_;
};

/** The model's entry in GET /speech/models and in the load's last event: its information and where it came from. */
std::string served_json(const Served & served);

/** Opens the page's address in the system's browser, saying on stderr when it cannot. */
void open_in_browser(const std::string & url);

}  // namespace server
