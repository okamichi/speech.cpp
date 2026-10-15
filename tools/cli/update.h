#pragma once

#include <memory>

#include "command-line.h"
#include "release.h"

/**
 * The notice of a newer release around a subcommand: one line on stderr, `speech: speech.cpp X.Y.Z is out (this is
 * ...); run speech update to install it`, from the latest release known, at the time the subcommand's release_notice
 * says. It runs only when stderr is a terminal and SPEECH_NO_UPDATE_CHECK is unset or empty; otherwise it prints nothing
 * and nothing reaches the network, so that scripts and the programs that start speech never make a request.
 */
class NewerReleaseNotice {
public:
    /** Starts a ReleaseCheck, and prints the notice now for a subcommand that has it as it starts. */
    explicit NewerReleaseNotice(ReleaseNotice when);

    /** Stops the check, and prints the notice for a subcommand that has it once it ends. */
    void ended();

private:
    ReleaseNotice when_;
    std::unique_ptr<ReleaseCheck> check_;
};
