#pragma once

#include <memory>

#include "httplib.h"

#include "served-models.h"

// POST /v1/audio/speech: OpenAI's create speech on the synthesis model held, its text spoken as speak_text() speaks it
// (sentences.h), a sentence at a time on a model whose request speaks only a short time. A wav waits for the whole
// speech; pcm and its SSE stream send the audio as it is made, from the first sentence's first audio on.

namespace server {

/**
 * Answers a create speech request on `served`: the library's refusal of a value at once, a refusal of the text before
 * any audio, the speech as wav, pcm or SSE events, and the request stopped for a client that goes away while it waits or
 * runs.
 */
void answer_speech(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res, bool compatible = false);

}  // namespace server
