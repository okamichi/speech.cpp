# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.2.6", "soxr==1.0.0"]
# ///
"""Measures how late live transcription is, as a client of `speech serve`'s Realtime transcription feels it, for one or
more builds of speech on the same recordings, so that two builds can be compared.

Sessions: for each language given (--set LANG=FOLDER), it draws clips from the folder's WAVE files (such as a FLEURS
test split), resamples each to 24000 Hz, cuts it to where Silero VAD finds speech with 0.3 s on either side, brings it
to an RMS of -23 dBFS, and joins a few into a session with a pause of 1.0 to 2.0 s between two clips, filled with white
noise at -60 dBFS, so that each clip ends at least one utterance. The reference of where someone speaks is Silero VAD's
regions of the whole session, computed offline by the first build's `speech vad` on the CPU with server_vad's options,
no padding and no longest region (threshold 0.5, min_silence_duration_ms 500, speech_pad_ms 0): the times below
therefore count the server's delay from where its own detection model hears the speech begin and end, not the error of
that model. The sessions and their reference are written to the work folder once and reused, so that other machines and
builds run on the same samples: copy `sessions/` to them.

Each session is sent to `speech serve MODEL silero-vad` over /v1/realtime with turn_detection server_vad and its
defaults, as the page's live panel sends it: 16-bit PCM at 24000 Hz, in chunks of --chunk-ms (100 ms, the page's tick,
unless given), each sent as soon as its audio would have been recorded in real time, so that the chunk's own length is
part of what is measured. The transcription language is the session's, where the model takes the option. For each
utterance, an item of the server, the reference's speech within its bounds gives its start and end of speech, and it
records:

- time to first partial: from the start of speech to the first delta that arrives before the utterance is committed,
  for the utterances that get one;
- finalization latency: from the end of speech to the completed transcript;
- speech started and committed, from the start and from the end of speech;
- whether it is cut: an utterance the detection cuts out of speech longer than its longest region (10 s) begins or ends
  inside the speech, at the cut, and is committed only once the stretch reaches that length, so its parts wait;
- whether the completed transcript is the text of the same samples, the region the server's detection cut found by
  `speech vad` with server_vad's options, sent afterwards to /v1/audio/transcriptions of the same server.

For each server, it records its peak memory: the peak physical footprint on macOS (as `/usr/bin/time -l` reports it),
the peak working set on Windows and VmHWM on Linux, none of which counts the GPU's own memory. Builds are run in turn
within each round (A B A B ...), every model in each. Each run appends its rows to <work dir>/results.jsonl, which keep
when each event arrived; the report computes the times from them and gives, per model, language and build, p50, p90 and
p99 with the count, over every row of the file, so runs of the same build label made at different times add up. The
server's processing time per chunk (latency per chunk) is not measured: the server answers an append with nothing a
client can time.

usage: uv run --script measure/live_latency.py <work dir> --build NAME=SPEECH... --set LANG=FOLDER...
                                               [--models qwen3-asr-0.6b ...] [--vad silero-vad] [--rounds 3]
                                               [--sessions 3] [--turns 6] [--chunk-ms 100] [-- serve options...]
       uv run --script measure/live_latency.py <work dir> --report

  NAME is the label of a build in the report and SPEECH its executable; LANG is the language tag of the clips in FOLDER,
  as the models list it (ja, en). Every model runs on the sets of its languages. Without --models: qwen3-asr-0.6b,
  qwen3-asr-1.7b, reazonspeech-v2, parakeet-tdt_ctc-0.6b-ja and parakeet-tdt-0.6b-v3. --set is needed only while the
  sessions of that language are not yet in the work folder, and --rounds 0 only makes them; --sessions and --turns
  apply only then. Options after -- go to every `speech serve`, except the detection model's, which the measure keeps
  at server_vad's defaults.

e.g. uv run --script measure/live_latency.py work --build 0.8.4=old/speech --build new=build/bin/speech \\
         --set ja=~/speech-bench-data/fleurs/<revision>/ja_jp/test --set en=~/speech-bench-data/fleurs/<revision>/en_us/test
"""

import base64
import ctypes
import json
import math
import os
import platform
import random
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy
import soxr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "checks", "smoke"))
from server_client import Server, WebSocket  # noqa: E402

RATE = 24000
MODELS = ["qwen3-asr-0.6b", "qwen3-asr-1.7b", "reazonspeech-v2", "parakeet-tdt_ctc-0.6b-ja", "parakeet-tdt-0.6b-v3"]
SERVER_VAD = ["--threshold", "0.5", "--min-silence-duration-ms", "500", "--max-speech-duration-s", "10", "--speech-pad-ms", "300"]
# Speech as server_vad hears it, without its padding and without a longest region, so that a stretch the server cuts
# stays whole.
REFERENCE = ["--threshold", "0.5", "--min-silence-duration-ms", "500", "--speech-pad-ms", "0"]
# The RMS of each clip, -23 dBFS, and of the noise of the pauses, -60 dBFS.
LEVEL, NOISE = 10 ** (-23 / 20), 10 ** (-60 / 20)
PAUSE = (1.0, 2.0)
EDGE = 0.3


def options(argv):
    """The work folder, and the options by name: lists for those that repeat or take several values."""
    named = {"--build": [], "--set": [], "--models": list(MODELS), "--vad": ["silero-vad"], "--rounds": ["3"],
             "--sessions": ["3"], "--turns": ["6"], "--chunk-ms": ["100"]}
    serve = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    plain, report, at, given = [], False, 0, set()
    while at < len(argv):
        name = argv[at]
        if name == "--report":
            report, at = True, at + 1
        elif name in named:
            end = at + 1
            while end < len(argv) and not argv[end].startswith("--"):
                end += 1
            if not argv[at + 1:end]:
                raise SystemExit(f"{name} needs a value")
            if name in ("--build", "--set"):
                named[name] += argv[at + 1:end]
            else:
                if name not in given:
                    named[name] = []
                named[name] += argv[at + 1:end]
            given.add(name)
            at = end
        else:
            plain.append(name)
            at += 1
    if len(plain) != 1 or (not report and not named["--build"]):
        raise SystemExit("usage:" + __doc__.split("usage:")[1].split("e.g.")[0].rstrip())
    pairs = {}
    for key in ("--build", "--set"):
        pairs[key] = {}
        for item in named[key]:
            if "=" not in item:
                raise SystemExit(f"{key} takes NAME=PATH, not {item!r}")
            k, v = item.split("=", 1)
            pairs[key][k] = os.path.expanduser(v)
    return plain[0], report, pairs["--build"], pairs["--set"], named, serve


def read_wave(path):
    """The samples of a WAVE file of 16-, 24- or 32-bit PCM or 32-bit float, its channels averaged, and its rate."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise SystemExit(f"{path} is not a WAVE file")
    at, fmt, samples = 12, None, None
    while at + 8 <= len(data):
        tag, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        body = data[at + 8:at + 8 + size]
        if tag == b"fmt ":
            kind, channels, rate = struct.unpack("<HHI", body[:8])
            bits = struct.unpack("<H", body[14:16])[0]
            if kind == 0xFFFE:
                kind = struct.unpack("<H", body[24:26])[0]
            fmt = (kind, channels, rate, bits)
        elif tag == b"data":
            samples = body
        at += 8 + size + (size & 1)
    if fmt is None or samples is None:
        raise SystemExit(f"{path} has no fmt or data chunk")
    kind, channels, rate, bits = fmt
    if kind == 3 and bits == 32:
        x = numpy.frombuffer(samples, "<f4").astype(numpy.float64)
    elif kind == 1 and bits in (16, 32):
        x = numpy.frombuffer(samples, f"<i{bits // 8}").astype(numpy.float64) / 2 ** (bits - 1)
    elif kind == 1 and bits == 24:
        b = numpy.frombuffer(samples[:len(samples) // 3 * 3], numpy.uint8).reshape(-1, 3).astype(numpy.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8) / 2 ** 23
    else:
        raise SystemExit(f"{path}: a WAVE format of tag {kind} and {bits} bits is not read; convert it to 16-bit PCM")
    return x[:len(x) // channels * channels].reshape(-1, channels).mean(axis=1), rate


def wave_bytes(pcm):
    """A WAVE file of 16-bit mono PCM at RATE."""
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, RATE, 2 * RATE, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def pcm16(x):
    return numpy.round(numpy.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def regions(speech, vad, paths, *flags):
    """`speech vad`'s regions of each file, (start, end) in seconds, on the CPU with one thread as the server's detection;
    it writes a line per file in the order given."""
    r = subprocess.run([speech, "vad", vad, "--device", "cpu", "--threads", "1", "--format", "json", *flags, *paths],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"speech vad exited with {r.returncode}: {r.stderr.decode(errors='replace')[-800:]}")
    found = [json.loads(line)["regions"] for line in r.stdout.decode().splitlines()]
    if len(found) != len(paths):
        raise SystemExit(f"speech vad wrote {len(found)} lines for {len(paths)} files")
    return [[(x["start"], x["end"]) for x in got] for got in found]


def make_sessions(work, language, folder, speech, vad, sessions, turns):
    """Writes the sessions of a language into <work>/sessions, unless some are there, and returns their names."""
    out = os.path.join(work, "sessions")
    made = sorted((n[:-len(".json")] for n in os.listdir(out) if n.endswith(".json") and n.rsplit("-", 1)[0] == language)
                  if os.path.isdir(out) else [], key=lambda n: int(n.rsplit("-", 1)[1]))
    if made:
        return made
    names = [f"{language}-{k + 1}" for k in range(sessions)]
    if not folder:
        raise SystemExit(f"the sessions of {language} are not in {out}: give --set {language}=FOLDER")
    os.makedirs(out, exist_ok=True)
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))
    chosen = random.Random(f"live-latency {language}").sample(files, sessions * turns)
    with tempfile.TemporaryDirectory() as temp:
        clips, paths = [], []
        for name in chosen:
            x, rate = read_wave(os.path.join(folder, name))
            x = soxr.resample(x, rate, RATE, "VHQ") if rate != RATE else x
            paths.append(os.path.join(temp, name))
            with open(paths[-1], "wb") as f:
                f.write(wave_bytes(pcm16(x)))
            clips.append(x)
        found = regions(speech, vad, paths, *REFERENCE)
        noise = numpy.random.default_rng(len(language))
        pause = random.Random(f"live-latency pauses {language}")
        for k, n in enumerate(names):
            parts, sources = [noise.standard_normal(RATE) * NOISE], []
            for name, x, speech_regions in list(zip(chosen, clips, found))[k * turns:(k + 1) * turns]:
                if not speech_regions:
                    raise SystemExit(f"Silero VAD finds no speech in {name}; choose another folder")
                first = max(0, round((speech_regions[0][0] - EDGE) * RATE))
                last = min(len(x), round((speech_regions[-1][1] + EDGE) * RATE))
                y = x[first:last]
                parts.append(y * (LEVEL / math.sqrt(float(numpy.mean(y * y)))))
                sources.append(name)
                parts.append(noise.standard_normal(round(pause.uniform(*PAUSE) * RATE)) * NOISE)
            parts.append(noise.standard_normal(RATE) * NOISE)
            path = os.path.join(out, n + ".wav")
            with open(path, "wb") as f:
                f.write(wave_bytes(pcm16(numpy.concatenate(parts))))
            reference = regions(speech, vad, [path], *REFERENCE)[0]
            with open(os.path.join(out, n + ".json"), "w") as f:
                json.dump({"language": language, "folder": folder, "clips": sources, "speech": reference}, f, indent=1)
    return names


class Socket(WebSocket):
    """A WebSocket whose frames may be sent from two threads: the sender's appends and the reader's answers to pings."""

    def __init__(self, sock):
        super().__init__(sock)
        self.lock = threading.Lock()

    def _frame(self, opcode, payload):
        with self.lock:
            super()._frame(opcode, payload)


class Session:
    """A Realtime transcription session whose server events are kept with the time each arrived."""

    def __init__(self, port, model, language):
        self.ws = Socket.open("127.0.0.1", port, "/v1/realtime", {})
        assert self.ws.event()["type"] == "session.created"
        transcription = {"model": model, **({"language": language} if language else {})}
        self.ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": RATE}, "transcription": transcription, "turn_detection": {"type": "server_vad"}}}}})
        updated = self.ws.event()
        if updated["type"] != "session.updated":
            raise SystemExit(f"the session was not configured: {updated}")
        self.events, self.changed = [], threading.Condition()
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        while True:
            try:
                text = self.ws.receive()
            except (ConnectionError, OSError):
                text = None
            with self.changed:
                self.events.append((time.perf_counter(), json.loads(text) if text is not None else None))
                self.changed.notify_all()
            if text is None:
                return

    def wait(self, done):
        """Waits until done(events) holds; a session the server ends or fails ends the measure, even where done holds."""
        with self.changed:
            while True:
                errors = [e for _, e in self.events if e and e["type"] == "error"]
                if errors:
                    raise SystemExit(f"the server answered an error: {errors[0]}")
                if done([e for _, e in self.events if e is not None]):
                    return
                if self.events and self.events[-1][1] is None:
                    raise SystemExit("the server closed the session")
                self.changed.wait(1)

    def stream(self, pcm, chunk):
        """Appends `pcm` a chunk of `chunk` samples at a time, each once its audio would have been recorded, then commits
        an utterance still under way and waits for every utterance's completed or failed. The time the session's audio
        began."""
        started = time.perf_counter()
        for k, at in enumerate(range(0, len(pcm) // 2, chunk)):
            time.sleep(max(0.0, started + min(at + chunk, len(pcm) // 2) / RATE - time.perf_counter()))
            self.ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[2 * at:2 * (at + chunk)]).decode()})
        # A session.update that changes nothing is answered after the events of the audio sent before it.
        self.ws.send({"type": "session.update", "session": {"type": "transcription"}})
        self.wait(lambda events: sum(e["type"] == "session.updated" for e in events) >= 1)
        kinds = [(e["type"], e.get("item_id")) for _, e in self.events if e]
        if any((t == "input_audio_buffer.speech_started" and ("input_audio_buffer.committed", item) not in kinds) for t, item in kinds):
            self.ws.send({"type": "input_audio_buffer.commit"})
            self.wait(lambda events: sum(e["type"] == "input_audio_buffer.committed" for e in events)
                      >= sum(e["type"] == "input_audio_buffer.speech_started" for e in events))
        self.wait(lambda events: sum(e["type"] == "input_audio_buffer.committed" for e in events)
                  == sum(e["type"].endswith((".completed", ".failed")) for e in events))
        return started

    def close(self):
        try:
            self.ws._frame(0x8, struct.pack(">H", 1000))
        except OSError:
            pass
        self.reader.join(10)
        self.ws.sock.close()


def milliseconds(seconds):
    """Milliseconds rounded half away from zero, as the server rounds audio_start_ms."""
    whole = math.floor(seconds * 1000)
    return whole + (1 if seconds * 1000 - whole >= 0.5 else 0)


def utterances(events, started, reference, cut):
    """One row per item of the session: its bounds, the reference's speech it overlaps, when each of its events arrived,
    in seconds from the start of the session's audio, and the region of the server's detection that it is."""
    rows, by_item = [], {}
    for t, e in events:
        if e and e.get("item_id"):
            by_item.setdefault(e["item_id"], []).append((t - started, e))
    for item, mine in by_item.items():
        first = {}
        for t, e in mine:
            first.setdefault(e["type"].rsplit(".", 1)[1], (t, e))
        begin = first["speech_started"][1]["audio_start_ms"] / 1000 if "speech_started" in first else 0.0
        end = first["speech_stopped"][1]["audio_end_ms"] / 1000 if "speech_stopped" in first else first["committed"][0]
        at = {kind: first[kind][0] for kind in ("speech_started", "speech_stopped", "committed", "completed", "failed") if kind in first}
        partial = [t for t, e in mine if e["type"].endswith(".delta") and t < at["committed"]]
        if partial:
            at["first_partial"] = partial[0]
        done = first.get("completed") or first.get("failed")
        region = [r for r in cut if milliseconds(r[0]) == first["speech_started"][1]["audio_start_ms"]] if "speech_started" in first else []
        rows.append({"item": item, "start": begin, "end": end, "heard": [r for r in reference if r[1] > begin and r[0] < end], "at": at,
                     "failed": "failed" in first, "text": done[1].get("transcript") if done else None, "region": region[0] if region else None})
    return rows


def latencies(row):
    """The times of an utterance that overlaps the reference's speech, and whether it is part of a longer stretch of it.

    Its speech is the reference's within its bounds: an utterance the detection cuts out of a stretch longer than its
    longest region begins or ends inside the reference's speech, and its own bounds are where that speech begins or ends
    for it.
    """
    heard, at = row["heard"], row["at"]
    speech_start, speech_end = max(heard[0][0], row["start"]), min(heard[-1][1], row["end"])
    times = {"time to first partial": ("first_partial", speech_start), "finalization latency": ("completed", speech_end),
             "speech started": ("speech_started", speech_start), "committed": ("committed", speech_end)}
    return {name: at[kind] - since if kind in at else None for name, (kind, since) in times.items()} | {
        "cut": heard[0][0] < row["start"] or heard[-1][1] > row["end"]}


def peak_memory(process):
    """The server's peak memory in bytes so far, as the system counts it for a process, or None where it cannot tell."""
    if sys.platform == "darwin":
        # struct rusage_info_v4 of <sys/resource.h>: a 16-byte uuid, then 64-bit fields, of which
        # ri_lifetime_max_phys_footprint is the 29th.
        buffer = (ctypes.c_uint64 * 64)()
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        if libc.proc_pid_rusage(process.pid, 4, ctypes.byref(buffer)) != 0:
            return None
        return int(buffer[2 + 28])
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
                                               "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        get = ctypes.WinDLL("kernel32").K32GetProcessMemoryInfo
        get.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_uint32]
        if not get(ctypes.c_void_p(int(process._handle)), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    try:
        with open(f"/proc/{process.pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def model_info(speech, model):
    """A model's information, from its file's metadata alone."""
    r = subprocess.run([speech, "info", model, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"speech info {model} exited with {r.returncode}: {r.stderr.decode(errors='replace')[-800:]}")
    return json.loads(r.stdout)


def check_serve(speech, vad, serve):
    """Refuses serve options that set a detection option: they would replace server_vad's defaults in the server, while
    the reference of the sessions and the regions the offline texts are made of assume those defaults."""
    names = {o["name"] for o in model_info(speech, vad)["options"]}
    names.update("vad." + name for name in tuple(names))
    given = [a for a in serve if a.startswith("--") and a[2:].split("=", 1)[0].replace("-", "_") in names]
    if given:
        raise SystemExit(f"{', '.join(given)} after -- would change the server's detection, which this measure keeps at "
                         "server_vad's defaults; leave the detection options out")


def run(work, label, speech, model, vad, serve, names, chunk_ms, round_number):
    """Runs a model's sessions on one server of a build, and appends the rows to results.jsonl."""
    server = Server(speech, [model, vad, *serve])
    held = json.loads(server.call("GET", "/v1/models")[2])["data"]
    info = next(m["speech"] for m in held if m["speech"]["task"] == "recognition")
    takes_language = any(o["name"] == "language" for o in info["options"])
    common = {"build": label, "model": model, "round": round_number, "chunk_ms": chunk_ms, "platform": f"{platform.system()} {platform.machine()}",
              "host": platform.node(), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rows = []
    for name in names:
        with open(os.path.join(work, "sessions", name + ".json")) as f:
            meta = json.load(f)
        path = os.path.join(work, "sessions", name + ".wav")
        cut_path = os.path.join(work, "cuts", label, name + ".json")
        if not os.path.exists(cut_path):
            os.makedirs(os.path.dirname(cut_path), exist_ok=True)
            cut = regions(speech, vad, [path], *SERVER_VAD)[0]
            with open(cut_path, "w") as f:
                json.dump(cut, f)
        with open(cut_path) as f:
            cut = json.load(f)
        with open(path, "rb") as f:
            pcm = f.read()[44:]
        language = meta["language"] if takes_language else None
        session = Session(server.port, info["name"], language)
        started = session.stream(pcm, RATE * chunk_ms // 1000)
        session.close()
        found = utterances(session.events, started, meta["speech"], cut)
        for u in found:
            if u["region"] is not None and u["text"] is not None:
                first, last = round(u["region"][0] * RATE), round(u["region"][1] * RATE)
                fields = [("response_format", "json")] + ([("language", language)] if language else [])
                status, _, body = server.transcribe(fields, [("file", "x.wav", wave_bytes(pcm[2 * first:2 * last]))])
                u["offline"] = json.loads(body)["text"] if status == 200 else None
                u["matches"] = u["offline"] == u["text"]
            rows.append(common | {"kind": "utterance", "language": meta["language"], "session": name} | u)
        timed = [latencies(u) for u in found if u["heard"]]
        print(f"  {name}: {len(found)} utterances, {sum(u.get('matches', False) for u in found)} match the offline text, "
              f"finalization latency p50 {fmt(percentile([t['finalization latency'] for t in timed], 50))}", flush=True)
    peak = peak_memory(server.process)
    server.stop()
    rows.append(common | {"kind": "server", "peak_memory": peak})
    with open(os.path.join(work, "results.jsonl"), "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def percentile(values, p):
    values = [v for v in values if v is not None]
    return float(numpy.percentile(values, p)) if values else None


def fmt(value):
    return "-" if value is None else f"{value:.2f}"


def spread(values):
    """p50/p90/p99 (count) in seconds."""
    values = [v for v in values if v is not None]
    return f"{fmt(percentile(values, 50))}/{fmt(percentile(values, 90))}/{fmt(percentile(values, 99))} ({len(values)})"


def report(work):
    path = os.path.join(work, "results.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no results in {path}")
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    groups = {}
    for r in rows:
        if r["kind"] == "utterance":
            groups.setdefault((r["model"], r["language"], r["build"]), []).append(r)
    print("\nTimes in seconds, p50/p90/p99 (count); time to first partial and speech started from the start of speech, "
          "finalization latency and committed from its end. Cut: the utterances the detection cut out of speech longer than "
          "its longest region, which wait for the cut; each model, language and build has a line for every utterance and "
          "one for those not cut.")
    names = ("time to first partial", "finalization latency", "speech started", "committed")
    head = ["model", "lang", "build", "chunk ms", "utterances", *names, "matches offline", "peak memory MiB"]
    table = [head]
    for (model, language, build), mine in sorted(groups.items()):
        timed = [latencies(u) for u in mine if u["heard"]]
        whole = [t for t in timed if not t["cut"]]
        peaks = [r["peak_memory"] for r in rows if r["kind"] == "server" and r["model"] == model and r["build"] == build and r["peak_memory"]]
        compared = [u for u in mine if "matches" in u]
        table.append([model, language, build, ",".join(sorted({str(u["chunk_ms"]) for u in mine})),
                      f"{len(mine)} ({len(timed) - len(whole)} cut, {len(mine) - len(timed)} without speech, "
                      f"{sum(u['failed'] for u in mine)} failed)",
                      *(spread([t[name] for t in timed]) for name in names),
                      f"{sum(u['matches'] for u in compared)}/{len(compared)}",
                      f"{numpy.median(peaks) / 2 ** 20:.0f} (max {max(peaks) / 2 ** 20:.0f})" if peaks else "-"])
        table.append(["", "", "", "", "not cut", *(spread([t[name] for t in whole]) for name in names), "", ""])
    widths = [max(len(str(r[k])) for r in table) for k in range(len(head))]
    for r in table:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)).rstrip())
    differ = [u for r in groups.values() for u in r if u.get("matches") is False]
    for u in differ[:20]:
        print(f"differs: {u['build']} {u['model']} {u['session']} {u['start']:.2f}-{u['end']:.2f} s: streamed {u['text']!r}, "
              f"offline {u['offline']!r}")


def main():
    # Transcripts in Japanese reach a Windows console or a file redirected there, whose code page cannot write them.
    sys.stdout.reconfigure(encoding="utf-8")
    work, only_report, builds, sets, named, serve = options(sys.argv[1:])
    if only_report:
        report(work)
        return
    for label, speech in builds.items():
        if not os.path.exists(speech):
            raise SystemExit(f"--build {label}: {speech} does not exist")
    vad = named["--vad"][0]
    first_build = next(iter(builds.values()))
    check_serve(first_build, vad, serve)
    folder = os.path.join(work, "sessions")
    made = {n[:-len(".json")].rsplit("-", 1)[0] for n in os.listdir(folder) if n.endswith(".json")} if os.path.isdir(folder) else set()
    names = []
    for language in sorted(set(sets) | made):
        names += make_sessions(work, language, sets.get(language), first_build, vad, int(named["--sessions"][0]), int(named["--turns"][0]))
    chunk_ms = int(named["--chunk-ms"][0])
    if chunk_ms <= 0 or RATE * chunk_ms % 1000:
        raise SystemExit("--chunk-ms must be a positive whole number of milliseconds")
    rounds = int(named["--rounds"][0])
    for round_number in range(rounds):
        for model in named["--models"]:
            spoken = set(model_info(first_build, model)["languages"])
            mine = [n for n in names if n.rsplit("-", 1)[0] in spoken]
            if not mine:
                print(f"{model} takes none of the sessions' languages ({', '.join(sorted(set(sets) | made))}); skipped", flush=True)
                continue
            for label, speech in builds.items():
                print(f"round {round_number + 1}: {model} on {label}", flush=True)
                run(work, label, speech, model, vad, serve, mine, chunk_ms, round_number + 1)
    if rounds:
        report(work)


if __name__ == "__main__":
    main()
