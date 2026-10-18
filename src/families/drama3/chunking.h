#pragma once

#include <string>
#include <vector>

namespace drama3 {

/**
 * The chunks fish-speech's split_drama3_text() cuts `text` into, in order, each spoken in a pass of its own. A text
 * without speaker tags is cut into chunks of about `max_bytes` bytes of UTF-8: at every sentence end, then, in a sentence
 * still longer, at clause marks, then at spaces, then anywhere, never inside a special token, a pair of tags such as
 * <whisper>...</whisper> or a [bracketed] cue, and the pieces are packed again up to max_bytes. A tagged text is cut
 * into its speakers' segments, a long one further with its tag on each piece, and a chunk takes segments while it has
 * at most `max_speakers` speakers and max_bytes bytes, so that A, B, A may share one. A text of no words has no chunk.
 */
std::vector<std::string> split_text(const std::string & text, size_t max_bytes, size_t max_speakers);

}  // namespace drama3
