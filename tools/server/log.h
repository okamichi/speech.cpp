#pragma once

#include <string>
#include <utility>

// The server's log on stderr once it listens: a line for each request, each realtime session and item, and each model
// or voice the page adds, after the time of day it was written at, so that a log kept in a file says when each happened.
// cpp-httplib's logger writes a request's line with the status sent, and what a model's run made where the run has
// ended by then; a run that ends after its handler has returned, a stream's or one whose client went away, writes its
// request's line itself.

namespace server {

/**
 * Writes `text` on stderr after the time of day: "10:42:13  text", the time in gray where colors are drawn. Every control
 * character in `text` is written as an escape ("\n" as \n, ESC as \x1b), since what a client sends reaches the log in
 * a request's path and in the messages of the errors it caused, and a line break or a terminal's escape there would
 * write lines of its own.
 */
void log(const std::string & text);

/** Has the lines of requests name their clients' addresses, as for a server that listens for other machines. */
void log_clients(bool name_them);

/**
 * Writes a request's line as log() writes text: its method and path, its status or how it ended ("gone" for a client
 * that went away, "failed"), colored by it where colors are drawn, then `what`, what it did, in gray, and the client's
 * address where log_clients() asks for it.
 */
void log_request(const std::string & method, const std::string & path, const std::string & status, const std::string & what,
                 const std::string & client);

/**
 * Leaves the logger, which cpp-httplib calls on the same thread once the handler has returned, what the request this
 * thread handles did, or with `skip` to leave the request out, as one whose run writes its own line.
 */
void note_request(std::string what, bool skip = false);

/** The note of the request this thread handles, which it then clears: what it did, and whether the logger leaves it out. */
std::pair<std::string, bool> take_request_note();

}  // namespace server
