#!/bin/sh
# Converts a local Anime or Large checkpoint to F32, then F16 and, for Large, Q8_0.
# Usage: convert-local.sh anime|large [--check-only]
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
target=${1:?Give anime or large}
shift
case "$target" in
    anime) source=${IRODORI_MODEL_DIR:-"$root/../Irodori-TTS-v4.1-Anime"}; name=Irodori-TTS-v4.1-Anime ;;
    large) source=${IRODORI_MODEL_DIR:-"$root/../Irodori-TTS-v4-Large"}; name=Irodori-TTS-v4-Large ;;
    *) echo "Unknown conversion target: $target. Give anime or large." >&2; exit 2 ;;
esac
ref="$root/reference/irodori-tts"
out="$root/models/converted/$target"
if [ "${1:-}" = --check-only ] && [ "$#" -eq 1 ]; then
    exec uv run --frozen --project "$ref" python "$ref/convert.py" "$target" "$out" --model-dir "$source" --check-only
fi
if [ "$#" -ne 0 ]; then
    echo "Usage: $0 anime|large [--check-only]" >&2
    exit 2
fi
if [ ! -x "$root/build/speech" ]; then
    echo "Build executable not found: $root/build/speech. Build speech before conversion." >&2
    exit 1
fi
mkdir -p "$out" "$root/models/.tmp"
export TMPDIR="$root/models/.tmp"
if [ -n "${IRODORI_CODEC_DIR:-}" ]; then
    uv run --frozen --project "$ref" python "$ref/convert.py" "$target" "$out" --model-dir "$source" \
        --codec-dir "$IRODORI_CODEC_DIR" --output-path-file "$out/model.path"
else
    uv run --frozen --project "$ref" python "$ref/convert.py" "$target" "$out" --model-dir "$source" --output-path-file "$out/model.path"
fi
IFS= read -r f32 < "$out/model.path"
"$root/build/speech" quantize "$f32" "$root/models/$name.gguf" --type f16
if [ "$target" = large ]; then
    "$root/build/speech" quantize "$f32" "$root/models/Irodori-TTS-v4-Large-Quantized.gguf" --type q8_0
fi
