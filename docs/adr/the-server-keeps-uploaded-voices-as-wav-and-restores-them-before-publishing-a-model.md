# The server keeps uploaded voices as WAV and restores them before publishing a model

## Context

The Irodori bridge registers recordings while the server runs and expects them after a restart. A voice latent is
bound to its codec, while the page can replace the synthesis model. Publishing a model before restoring its voices
would make a listed voice disappear temporarily, and a failed registration must not leave it in memory alone.

## Decision

The server owns a directory of original WAV recordings. It restores them through the C API before publishing a
synthesis model, and serializes uploads with model replacement. The library continues to own voices in memory alone.
Registration prepares the information snapshot before changing the engine and returns it as part of the same
operation. The server publishes the durable recording without replacing an existing file before registering it,
and removes the recording if registration fails.

The alternatives were turned down:

- Keeping GGUF latents alone. A different codec cannot restore them and the original recording is lost.
- Treating a derived GGUF cache as another authoritative store. Rebuilding from the WAV is sufficient for the
  bridge, and two authoritative files introduce mismatched state on failure.
- Registering in memory before saving. A failed save cannot be rolled back through the existing library API.
- Rebuilding model information after registration. Allocation failure would report an error after changing the
  library's voices and leave the server's listing stale.

## Consequences

Startup and model replacement encode each stored recording. A recording that cannot be restored fails the load
instead of silently disappearing from the voice list. The page and the compatibility endpoint use the same store.
