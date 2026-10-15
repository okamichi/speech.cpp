#pragma once

#include "command-line.h"

// The subcommands of `speech`, each declared with its flags and the function that runs it.

Command tts_command();
Command asr_command();
Command vad_command();
Command voice_command();
Command info_command();
Command devices_command();
Command models_command();
Command pull_command();
Command rm_command();
Command quantize_command();
Command update_command();
Command serve_command();
Command worker_command();
