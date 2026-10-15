"""Times Irodori-TTS through `speech worker` with and without instructions (the runtime's caption): each sentence of a
speech-bench prompts file one request at a time after the worker is ready, in one voice, first without instructions and
then with each caption given. For each, the p50 and the p90 of the time to first audio, from the request to its first
chunk, and the median RTFx, the request's seconds of audio over the seconds it took. It prints one JSON object per line.

usage: python3 measure/irodori-caption-timing.py <speech> <model.gguf> <prompts.json> <voice NAME=FILE> [--steps N]
                                                 [--caption TEXT]... [-- worker options...]
"""

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checks", "smoke"))
from worker_client import Worker  # noqa: E402

# A caption of the length people write, the one the estimate of the cost was made for.
CAPTION = "落ち着いた女性の声で、近い距離感でやわらかく自然に読み上げてください。"

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) < 4:
    raise SystemExit(__doc__)
speech, model, prompts_path, voice, *rest = args
steps = None
captions = []
while rest:
    flag, value, *rest = rest
    if flag == "--steps":
        steps = int(value)
    elif flag == "--caption":
        captions.append(value)
    else:
        raise SystemExit(f"unknown option {flag}\n{__doc__}")
name, _ = voice.split("=", 1)
prompts = json.load(open(prompts_path, encoding="utf-8"))["prompts"]
worker = Worker(speech, model, ["--add-voice", voice, *options])
rate = worker.ready["model"]["sample_rate"]


def run(instructions):
    first, rtfx = [], []
    for i, prompt in enumerate(prompts):
        request = {"type": "synthesize", "id": str(i), "text": prompt["text"], "voice": name, "seed": 1}
        if steps:
            request["steps"] = steps
        if instructions:
            request["instructions"] = instructions
        sent = time.perf_counter()
        worker.request(request)
        messages = worker.until(str(i))
        end = messages[-1]
        if end["type"] != "end":
            raise SystemExit(f"{prompt['id']}: {end}")
        chunks = [m for m in messages if m["type"] == "chunk"]
        first.append(chunks[0]["_at"] - sent)
        rtfx.append(end["samples"] / rate / (end["_at"] - sent))
    return {"instructions": instructions, "sentences": len(prompts), "steps": steps,
            "time_to_first_audio_p50": statistics.median(first), "time_to_first_audio_p90": statistics.quantiles(first, n=10)[-1],
            "rtfx": statistics.median(rtfx)}


for instructions in ["", *(captions or [CAPTION])]:
    print(json.dumps(run(instructions), ensure_ascii=False), flush=True)
