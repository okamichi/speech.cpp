#pragma once

#include <string>

#include "json-reader.h"

namespace openai {

/** Rewrites an Irodori request into library option names and returns its whole-response time limit. */
double irodori_request(JsonValue & json, const std::string & model_name);

}  // namespace openai
