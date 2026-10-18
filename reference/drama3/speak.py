"""Speaks texts with the official Drama 3 on the CPU in float32, as fish-speech's server answers a /v1/tts request,
and writes each as a WAVE file with the time it took.

Usage: uv run python speak.py --out DIR --reference REF.wav --reference-text TEXT [--seed N] [--temperature T]
                              [--top-p P] TEXT...
"""

import argparse
import os
import time

import numpy as np
import soundfile as sf

from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest

import official


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("texts", nargs="+")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    started = time.perf_counter()
    _, engine = official.engine()
    print(f"loaded in {time.perf_counter() - started:.1f} s", flush=True)
    with open(args.reference, "rb") as f:
        reference = ServeReferenceAudio(audio=f.read(), text=args.reference_text)
    for index, text in enumerate(args.texts):
        request = ServeTTSRequest(
            text=text, references=[reference], seed=args.seed + index, temperature=args.temperature, top_p=args.top_p)
        started = time.perf_counter()
        final = [result for result in engine.inference(request) if result.code in ("final", "error")][-1]
        if final.code == "error":
            raise SystemExit(f"{text!r}: {final.error}")
        rate, audio = final.audio
        seconds = time.perf_counter() - started
        path = os.path.join(args.out, f"{index:02d}.wav")
        sf.write(path, np.asarray(audio, dtype=np.float32), rate, subtype="FLOAT")
        print(f"{path}: {len(audio) / rate:.2f} s of audio in {seconds:.1f} s, seed {args.seed + index}: {text}", flush=True)


if __name__ == "__main__":
    main()
