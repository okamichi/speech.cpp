"""Speaks the same requests through two builds of libspeech, or two model files, and fails on the first request whose
audio differs, so that a change meant to leave the audio as it was is shown to: the float samples the C API's
callback passes, compared bit for bit, which the 16 bits of a WAVE file would round away.

Each side runs in a process of its own, which loads its library and model once and speaks every sentence of the
prompts file in the voice of --voice, a voice file it adds, or of --model-voice, one of the model's own, with every seed
and, for each --steps, that many sampler steps (none for the model's default). A side's --before-set and --after-set
give options of the C API's vocabulary to its requests alone, as NAME=JSON, so that a request that sets a default
explicitly can be compared with one that leaves it out.

usage: python3 checks/compare/same_audio.py <libspeech before> <model before> <libspeech after> <model after> <prompts.json>
                                            (--voice FILE | --model-voice NAME) [--device NAME] [--seeds 1,2,3] [--steps N]...
                                            [--before-set NAME=JSON]... [--after-set NAME=JSON]...
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile

STRING, INT, FLOAT, BOOL = 0, 1, 2, 3
AUDIO = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_void_p)


def speak_side(library, model_path, voice_file, voice, device, prompts, seeds, steps, settings, out_dir):
    """Speaks every request of one side into out_dir, one file of raw float32 samples per request."""
    lib = ctypes.CDLL(library)
    lib.speech_last_error.restype = ctypes.c_char_p
    lib.speech_version.restype = ctypes.c_char_p

    def check(status, what):
        if status < 0:
            raise SystemExit(f"{library}: {what}: {lib.speech_last_error().decode()}")

    params = ctypes.c_void_p()
    check(lib.speech_load_params_new(ctypes.byref(params)), "speech_load_params_new")
    check(lib.speech_load_params_set_device(params, device.encode()), "the device")
    model = ctypes.c_void_p()
    check(lib.speech_model_load(model_path.encode(), params, ctypes.byref(model)), model_path)
    if voice_file:
        check(lib.speech_voice_add(model, voice.encode(), voice_file.encode()), voice_file)

    options = []
    for name, value in settings:
        option = ctypes.c_int()
        check(lib.speech_option_from_name(name.encode(), ctypes.byref(option)), name)
        options.append((option.value, lib.speech_option_type(option.value), value))

    samples = bytearray()

    @AUDIO
    def collect(data, n, _):
        samples.extend(ctypes.string_at(data, n * 4))
        return 0

    for step_count in steps:
        for seed in seeds:
            for prompt in prompts:
                request = ctypes.c_void_p()
                check(lib.speech_request_new(model, ctypes.byref(request)), "speech_request_new")
                check(lib.speech_request_set_text(request, prompt["text"].encode()), prompt["id"])
                check(lib.speech_request_set_string(request, 0, voice.encode()), voice)
                check(lib.speech_request_set_int(request, 2, ctypes.c_int64(seed)), "seed")
                if step_count:
                    check(lib.speech_request_set_int(request, 6, ctypes.c_int64(step_count)), "steps")
                for option, kind, value in options:
                    if kind == STRING:
                        check(lib.speech_request_set_string(request, option, value.encode()), str(value))
                    elif kind == INT:
                        check(lib.speech_request_set_int(request, option, ctypes.c_int64(value)), str(value))
                    elif kind == FLOAT:
                        check(lib.speech_request_set_float(request, option, ctypes.c_double(value)), str(value))
                    else:
                        check(lib.speech_request_set_bool(request, option, 1 if value else 0), str(value))
                samples.clear()
                check(lib.speech_synthesize(request, collect, None), prompt["id"])
                lib.speech_request_free(request)
                with open(os.path.join(out_dir, f"{step_count}-{seed}-{prompt['id']}.f32"), "wb") as f:
                    f.write(samples)
    lib.speech_model_free(model)
    print(f"{library} ({lib.speech_version().decode()}) spoke {len(steps) * len(seeds) * len(prompts)} requests with {model_path} "
          f"on {device}", flush=True)


def settings_of(items):
    return [(item.split("=", 1)[0], json.loads(item.split("=", 1)[1])) for item in items]


parser = argparse.ArgumentParser()
parser.add_argument("before_library")
parser.add_argument("before_model")
parser.add_argument("after_library")
parser.add_argument("after_model")
parser.add_argument("prompts")
voices = parser.add_mutually_exclusive_group(required=True)
voices.add_argument("--voice")
voices.add_argument("--model-voice")
parser.add_argument("--device", default="auto")
parser.add_argument("--seeds", default="1,2,3")
parser.add_argument("--steps", type=int, action="append")
parser.add_argument("--before-set", action="append", default=[])
parser.add_argument("--after-set", action="append", default=[])
parser.add_argument("--side", help=argparse.SUPPRESS)
parser.add_argument("--out", help=argparse.SUPPRESS)
args = parser.parse_args()
prompts = json.load(open(args.prompts, encoding="utf-8"))["prompts"]
seeds = [int(s) for s in args.seeds.split(",")]
steps = args.steps or [0]

if args.side:
    before = args.side == "before"
    speak_side(args.before_library if before else args.after_library, args.before_model if before else args.after_model, args.voice,
               args.model_voice or "voice", args.device, prompts, seeds, steps, settings_of(args.before_set if before else args.after_set),
               args.out)
    sys.exit(0)

with tempfile.TemporaryDirectory() as tmp:
    dirs = {}
    for side in ("before", "after"):
        dirs[side] = os.path.join(tmp, side)
        os.makedirs(dirs[side])
        subprocess.run([sys.executable, __file__, *sys.argv[1:], "--side", side, "--out", dirs[side]], check=True)
    names = sorted(os.listdir(dirs["before"]))
    total = 0
    for name in names:
        a = open(os.path.join(dirs["before"], name), "rb").read()
        b = open(os.path.join(dirs["after"], name), "rb").read()
        if a != b:
            n = min(len(a), len(b)) // 4
            first = next((i for i in range(n) if a[4 * i : 4 * i + 4] != b[4 * i : 4 * i + 4]), n)
            raise SystemExit(f"FAIL: {name[:-4]}: {len(a) // 4} samples before and {len(b) // 4} after, the first difference at sample {first}")
        total += len(a) // 4
    print(f"the same audio, bit for bit: {len(names)} requests, {total} samples")
