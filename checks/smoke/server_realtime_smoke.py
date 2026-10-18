"""Speaks OpenAI's Realtime transcription to `speech serve` over the WebSocket at /v1/realtime, the way a client of OpenAI's
API does, with the standard library alone: the session.created and session.updated of a transcription session, an
event written in two pieces 350 ms apart, two
commits of 24000 Hz PCM appended in pieces, each committed in order and transcribed into deltas that only add text and
the completed of the text, the languages and the duration of the same samples sent as a WAV file of that rate, a
cleared buffer, a commit with transcription null left untranscribed, an error event for each member and value it does
not take with the client's event_id, a session.update that changes only the members it gives, the bound on the audio a
session holds before it is transcribed, two commits and an HTTP request run in the order they arrived, a client gone
while its commit is transcribed, and the refusals of the upgrade: another model, another query member, another Origin
and another Host. As someone speaks: deltas of a commit appended in real time before it is committed, and a reading
under way stopped by the commit; with turn_detection server_vad, the dumps joined with silences and appended in pieces
of 20 ms cut into the regions that speech vad finds on the CPU, each with its speech_started and speech_stopped at the
region's bounds, its commit and its completed with the text of the region's samples sent as a WAV file, all of them
joined into the text of chunking_strategy with the same values; deltas of an utterance appended in real time before its
commit; a commit and a clear while speech is under way; minutes of silence held within the bound; server_vad's
refusals, on a server without a detection model too; and a detection flag given to the server (--threshold) as turn
detection's default, before OpenAI's, with a session's own member kept.

usage: python3 checks/smoke/server_realtime_smoke.py <speech> <recognition.gguf> <detection.gguf> <dump folder>... [-- serve options...]
"""

import array
import base64
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time

from server_client import Server, expect_error
from worker_client import joined_transcript, read_npy

args = sys.argv[1:]
options = args[args.index("--") + 1:] if "--" in args else []
args = args[:args.index("--")] if "--" in args else args
if len(args) < 4:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, model, vad, *dumps = args
ORIGIN = "http://localhost:5173"

# A server without a detection model refuses server_vad, saying to give it one.
alone = Server(speech, [model, *options])
ws = alone.websocket("/v1/realtime")
assert ws.event()["type"] == "session.created"
ws.send({"type": "session.update", "event_id": "vad", "session": {"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad"}}}}})
refused = ws.event()
assert refused["type"] == "error" and refused["error"]["param"] == "session.audio.input.turn_detection" and "silero-vad" in refused["error"]["message"], refused
ws.close()
assert alone.stop() == b""
print(f"realtime server_vad on a server without a detection model: error {refused['error']['code']}: {refused['error']['message'][:80]}")

# The detection options given to the server as flags are turn detection's defaults, before OpenAI's, and a session's own
# member wins.
tuned = Server(speech, [model, vad, *options, "--threshold", "0.7"])
ws = tuned.websocket("/v1/realtime")
assert ws.event()["type"] == "session.created"
for given, threshold in (({"type": "server_vad"}, 0.7), ({"type": "server_vad", "threshold": 0.4}, 0.4)):
    ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"turn_detection": given}}}})
    turns = ws.event()["session"]["audio"]["input"]["turn_detection"]
    assert turns == {"type": "server_vad", "threshold": threshold, "prefix_padding_ms": 300, "silence_duration_ms": 500}, turns
ws.close()
tuned.stop()
print("realtime server_vad on a server given --threshold 0.7: its threshold 0.7 with OpenAI's other defaults, and a session's 0.4 kept")

server = Server(speech, [model, vad, "--cors-origin", ORIGIN, *options])
transcribe = server.transcribe
info = json.loads(server.call("GET", "/v1/models")[2])["data"][0]["speech"]
assert info["task"] == "recognition", info["task"]
rate = info["sample_rate"]

# OpenAI's Realtime transcription at /v1/realtime: 16-bit PCM at 24000 Hz, which the server resamples to the model's
# rate as it does a WAV file of that rate, so a commit's text is that of the same samples sent as a file.
def pcm24(samples):
    n = len(samples) * 24000 // rate
    out = []
    for i in range(n):
        at = i * rate / 24000
        j = int(at)
        x = samples[j] + (samples[min(j + 1, len(samples) - 1)] - samples[j]) * (at - j)
        out.append(max(-32768, min(32767, round(x * 32768))))
    return array.array("h", out).tobytes()

def wav24(pcm):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)

def appended(ws, pcm, piece=9600):
    for at in range(0, len(pcm), piece):
        ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[at:at + piece]).decode()})

def realtime_error(ws, code, param, what, event_id=None):
    e = ws.event()
    assert e["type"] == "error" and e["error"]["code"] == code and e["error"]["param"] == param and e["error"]["event_id"] == event_id, (what, e)
    assert e["error"]["type"] == "invalid_request_error" and e["error"]["message"] and e["event_id"].startswith("event_"), e
    print(f"realtime {what}: error {code} ({param}) as expected: {e['error']['message'][:80]}")

def session(transcription, turn_detection=None, **members):
    return {"type": "session.update", **members, "session": {"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "transcription": transcription, "turn_detection": turn_detection}}}}

name = info["name"]
pcms = [pcm24(read_npy(os.path.join(d, "audio.npy"))) for d in (dumps * 2)[:2]]
wants = []
for pcm in pcms:
    status, _, body = transcribe([("response_format", "verbose_json")], [("file", "x.wav", wav24(pcm))])
    verbose = json.loads(body)
    wants.append((verbose["text"], verbose.get("language", "").split(",") if verbose.get("language") else []))
ws = server.websocket(f"/v1/realtime?model={name}")
created = ws.event()
assert created["type"] == "session.created" and created["event_id"].startswith("event_"), created
configured = created["session"]
assert configured["type"] == "transcription" and configured["object"] == "realtime.transcription_session" and configured["id"], configured
assert configured["audio"]["input"] == {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": name, "language": None},
                                        "noise_reduction": None, "turn_detection": None}, configured
ws.send(session({"model": name}))
updated = ws.event()
assert updated["type"] == "session.updated" and updated["session"] == configured, updated
# An event whose frame arrives in two pieces 350 ms apart is read whole: the connection waits for the rest of a message.
ws.send_split({"type": "input_audio_buffer.clear"}, 0.35)
assert ws.event()["type"] == "input_audio_buffer.cleared", "an event written in two pieces was not answered"
print("realtime: an event written in two pieces 350 ms apart, answered")
# Two commits in a row, each of its appends: both are committed at once, the second after the first, and each is
# transcribed in turn into one delta and its completed with the duration.
items = []
for pcm in pcms:
    appended(ws, pcm)
    ws.send({"type": "input_audio_buffer.commit"})
events = []
while sum(e["type"].endswith(".completed") for e in events) < len(pcms):
    events.append(ws.event())
committed = [e for e in events if e["type"] == "input_audio_buffer.committed"]
items = [e["item_id"] for e in committed]
assert [e["previous_item_id"] for e in committed] == [None, items[0]] and len(set(items)) == 2, committed
for item, pcm, (text, languages) in zip(items, pcms, wants):
    mine = [e for e in events if e.get("item_id") == item and e["type"] != "input_audio_buffer.committed"]
    deltas, done = mine[:-1], mine[-1]
    assert all(d["type"].endswith(".delta") and d["delta"] and d["content_index"] == 0 for d in deltas), mine
    # The deltas join into the final text: here none comes before the commit, so the rest is the whole of it.
    assert "".join(d["delta"] for d in deltas) == text, mine
    assert done["type"] == "conversation.item.input_audio_transcription.completed" and done["transcript"] == text, done
    assert done["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / 24000} and done["content_index"] == 0, done
    assert [x["code"] for x in done.get("languages", [])] == languages and done["stop"] == "complete", done
assert [e["type"] for e in events].index("conversation.item.input_audio_transcription.completed") > [e["type"] for e in events].index(
    "input_audio_buffer.committed"), events
print(f"realtime: session.created and session.updated as configured, two commits committed in order and transcribed as their "
      f"samples sent as a 24000 Hz file: {wants[0][0][:40]!r}")
# A buffer cleared leaves nothing to commit; a session that transcribes nothing commits without transcribing.
appended(ws, pcms[0][:4800])
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
ws.send({"type": "input_audio_buffer.commit", "event_id": "empty"})
realtime_error(ws, "input_audio_buffer_commit_empty", None, "commit of a cleared buffer", "empty")
ws.send(session(None))
assert ws.event()["session"]["audio"]["input"]["transcription"] is None
appended(ws, pcms[0])
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared", "a session with transcription null transcribed a commit"
ws.send(session({"model": name, "language": info["languages"][0]}))
assert ws.event()["session"]["audio"]["input"]["transcription"]["language"] == info["languages"][0]
print("realtime: a cleared buffer, and a commit with transcription null, which is committed and not transcribed")
# A session.update changes only the members it gives, and a member given as null takes its value away.
ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"transcription": {"model": name}}}}})
kept = ws.event()["session"]["audio"]["input"]["transcription"]
assert kept == {"model": name, "language": info["languages"][0]}, kept
ws.send({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"transcription": {"language": None}}}}})
kept = ws.event()["session"]["audio"]["input"]["transcription"]
assert kept == {"model": name, "language": None}, kept
print("realtime: a session.update of the model alone keeps the language, and a language of null takes it away")
for event, code, param, what in (
        (session({"model": name}) | {"session": {"type": "realtime"}}, "unsupported_value", "session.type", "a conversation session"),
        (session({"model": name}, {"type": "semantic_vad"}), "unsupported_value", "session.audio.input.turn_detection.type", "semantic_vad"),
        (session({"model": name}, {"threshold": 0.5}), "missing_required_parameter", "session.audio.input.turn_detection.type",
         "turn_detection without its type"),
        (session({"model": name}, {"type": "server_vad", "create_response": True}), "unsupported_parameter",
         "session.audio.input.turn_detection.create_response", "create_response"),
        (session({"model": name}, {"type": "server_vad", "interrupt_response": False}), "unsupported_parameter",
         "session.audio.input.turn_detection.interrupt_response", "interrupt_response"),
        (session({"model": name}, {"type": "server_vad", "idle_timeout_ms": 6000}), "unsupported_parameter",
         "session.audio.input.turn_detection.idle_timeout_ms", "idle_timeout_ms"),
        (session({"model": name}, {"type": "server_vad", "eagerness": "low"}), "unknown_parameter",
         "session.audio.input.turn_detection.eagerness", "a member of semantic_vad"),
        (session({"model": name}, {"type": "server_vad", "threshold": "high"}), "invalid_type", "session.audio.input.turn_detection.threshold",
         "a threshold that is no number"),
        (session({"model": name}, {"type": "server_vad", "threshold": 2}), "unsupported_value", "session.audio.input.turn_detection.threshold",
         "a threshold above 1"),
        (session({"model": name}, {"type": "server_vad", "silence_duration_ms": 0.5}), "invalid_type",
         "session.audio.input.turn_detection.silence_duration_ms", "a silence with a fraction"),
        (session({"model": name}, {"type": "server_vad", "prefix_padding_ms": -1}), "unsupported_value",
         "session.audio.input.turn_detection.prefix_padding_ms", "a negative padding"),
        (session({"model": "gpt-4o-transcribe"}), "model_not_found", "session.audio.input.transcription.model", "another model"),
        (session({"model": name, "language": "zz"}), "unsupported_value", "session.audio.input.transcription.language", "a language"),
        (session({"model": name, "languages": ["ja", "en"]}), "unsupported_value", "session.audio.input.transcription.languages",
         "two languages"),
        (session({"model": name, "keywords": ["渋谷"]}), "unsupported_parameter", "session.audio.input.transcription.keywords", "keywords"),
        (session({"model": name, "delay": "low"}), "unsupported_parameter", "session.audio.input.transcription.delay", "delay"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"noise_reduction": {"type": "near_field"}}}}},
         "unsupported_parameter", "session.audio.input.noise_reduction", "noise reduction"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcmu"}}}}},
         "unsupported_value", "session.audio.input.format.type", "G.711"),
        ({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 16000}}}}},
         "invalid_value", "session.audio.input.format.rate", "16000 Hz"),
        ({"type": "session.update", "session": {"type": "transcription", "include": ["item.input_audio_transcription.logprobs"]}},
         "unsupported_value", "session.include", "logprobs"),
        ({"type": "session.update", "session": {"type": "transcription", "model": name}}, "unknown_parameter", "session.model",
         "a member of a conversation session"),
        ({"type": "session.update"}, "missing_required_parameter", "session", "no session"),
        ({"type": "input_audio_buffer.append", "audio": "not base64!"}, "invalid_value", "audio", "audio that is not base64"),
        ({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"abc").decode()}, "invalid_value", "audio", "an odd byte"),
        ({"type": "input_audio_buffer.append"}, "missing_required_parameter", "audio", "no audio"),
        ({"type": "response.create"}, "unsupported_value", "type", "response.create"),
        ({"type": "input_audio_buffer.clear", "item_id": "x"}, "unknown_parameter", "item_id", "a member clear does not have"),
        ({"event_id": "x"}, "missing_required_parameter", "type", "no type")):
    ws.send(event | {"event_id": event.get("event_id", "e-" + what)})
    realtime_error(ws, code, param, what, event.get("event_id", "e-" + what))
ws.send("{not json")
realtime_error(ws, "invalid_json", None, "a message that is not JSON")
ws.send_binary(b"\0\0")
realtime_error(ws, "invalid_value", None, "a binary message")
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared", "the session did not go on after its errors"
# The audio a session holds before it is transcribed, buffered or committed, is bounded: an append past the bound is an
# error event and keeps nothing of itself, the buffer before it stays whole, and a commit that waits or runs counts until
# its transcription ends.
piece = b"\0" * (1 << 20)


def next_error():
    """The next error event, past the deltas of the readings of the buffer as it grows: a model can read words into
    silence, as reazonspeech-v2 on Vulkan read 23 MiB of zeros as ピンポン (RTX 2080, 2026-10-08)."""
    e = ws.event()
    while e["type"] == "conversation.item.input_audio_transcription.delta":
        e = ws.event()
    return e


def fill(limit_pieces):
    """Appends pieces of a mebibyte until an append is refused, each followed by an event that is always refused; the
    number of pieces taken, or None when none was refused."""
    for k in range(limit_pieces):
        ws.send({"type": "input_audio_buffer.append", "event_id": f"a{k}", "audio": base64.b64encode(piece).decode()})
        ws.send({"type": "no.such.event", "event_id": f"p{k}"})
        e = next_error()
        if e["error"]["event_id"] == f"a{k}":
            assert e["error"]["param"] == "audio" and e["error"]["code"] == "input_audio_buffer_full", e
            assert next_error()["error"]["event_id"] == f"p{k}"
            return k
        assert e["error"]["event_id"] == f"p{k}", e
    return None


taken = fill(512)
assert taken, "512 MiB of audio appended without a commit were all taken"
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
done = ws.event()
while not done["type"].endswith((".completed", ".failed")):
    done = ws.event()
assert done["type"].endswith(".completed") and done["usage"]["seconds"] == taken * (1 << 20) / 2 / 24000, done
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
appended(ws, piece * (taken * 3 // 5), 1 << 20)
ws.send({"type": "input_audio_buffer.commit"})
assert ws.event()["type"] == "input_audio_buffer.committed"
assert fill(taken) == taken - taken * 3 // 5, "a commit that waits or runs did not count toward the bound"
while not ws.event()["type"].endswith(".completed"):
    pass
ws.send({"type": "input_audio_buffer.clear"})
assert ws.event()["type"] == "input_audio_buffer.cleared"
print(f"realtime: {taken} MiB of audio held, the next append refused and the buffer committed whole; a commit that waits "
      f"or runs counts until it is transcribed")
# A commit takes its turn on the model when it is accepted, so an HTTP request that arrives after two commits runs
# after both, as the server runs every request in the order it arrives.
appended(ws, pcms[0] * 4)
ws.send({"type": "input_audio_buffer.commit"})
appended(ws, pcms[1])
ws.send({"type": "input_audio_buffer.commit"})
first, second = ws.event(), ws.event()
assert first["type"] == second["type"] == "input_audio_buffer.committed", (first, second)
assert transcribe([], [("file", "x.wav", wav24(pcms[1]))])[0] == 200
done = []
while len(done) < 2:
    e = ws.event()
    if e["type"].endswith((".completed", ".failed")):
        done.append(e)
assert [(e["type"], e["item_id"]) for e in done] == [("conversation.item.input_audio_transcription.completed", item) for item in
                                                    (first["item_id"], second["item_id"])], done
http = "POST /v1/audio/transcriptions "
heard = f"{len(pcms[1]) / 2 / 24000:.2f} s of audio at 24000 Hz"
for _ in range(50):
    lines = server.logged()
    order = [k for k, line in enumerate(lines)
             if line.startswith((f"realtime {first['item_id']}:", f"realtime {second['item_id']}:")) or line.startswith(http) and heard in line]
    if len(order) >= 3:
        break
    time.sleep(0.1)
ran = ["transcription" if lines[k].startswith(http) else lines[k].split(":")[0] for k in order][-3:]
assert ran == [f"realtime {first['item_id']}", f"realtime {second['item_id']}", "transcription"], ran
print("realtime: two commits and then an HTTP transcription run in that order")


def paced(ws, pcm):
    """Appends 24000 Hz PCM in pieces of 20 ms as fast as it is said."""
    started = time.perf_counter()
    for k, at in enumerate(range(0, len(pcm), 960)):
        time.sleep(max(0.0, started + k * 0.02 - time.perf_counter()))
        ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm[at:at + 960]).decode()})


def until_done(ws, count=1):
    """The events up to the count-th completed or failed."""
    events = []
    while sum(e["type"].endswith((".completed", ".failed")) for e in events) < count:
        events.append(ws.event())
    return events


def settle(ws):
    """The events of the audio appended so far: those that come before the answer to an event that is always refused,
    since a commit's goes out as its audio is taken, and then each commit's completed or failed."""
    ws.send({"type": "no.such.event", "event_id": "settle"})
    events = []
    while True:
        e = ws.event()
        if e["type"] == "error" and e["error"]["event_id"] == "settle":
            break
        events.append(e)
    while sum(e["type"] == "input_audio_buffer.committed" for e in events) > sum(e["type"].endswith((".completed", ".failed")) for e in events):
        events.append(ws.event())
    return events


def kinds_of(events, item):
    return [e["type"].rsplit(".", 1)[1] for e in events if e.get("item_id") == item]


# As someone speaks, a buffer appended in real time is read again for deltas, which only add text, before its commit.
longest = max(pcms, key=len)
paced(ws, longest)
ws.send({"type": "input_audio_buffer.commit"})
events = until_done(ws)
item = events[-1]["item_id"]
kinds = kinds_of(events, item)
deltas = [e["delta"] for e in events if e["type"].endswith(".delta")]
assert kinds.index("committed") > 0 and set(kinds[:kinds.index("committed")]) == {"delta"} and kinds[-1] == "completed" and all(deltas), kinds
# Where the final text goes on from what the deltas gave while it was said, one more delta gives the rest of it.
said = "".join(deltas[:kinds.index("committed")])
assert not events[-1]["transcript"].startswith(said) or "".join(deltas) == events[-1]["transcript"], (deltas, events[-1]["transcript"])
print(f"realtime: {kinds.index('committed')} deltas of {len(longest) / 48000:.1f} s appended in real time before its commit, "
      f"{''.join(deltas)[:30]!r}, then {events[-1]['transcript'][:30]!r}")
# A reading under way when its buffer is committed is stopped, and the commit is recognized whole.
ws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcms[0] * 6).decode()})
time.sleep(0.05)
ws.send({"type": "input_audio_buffer.commit"})
events = until_done(ws)
assert events[-1]["type"].endswith(".completed") and events[-1]["usage"]["seconds"] == len(pcms[0]) * 6 / 48000, events[-1]
for _ in range(50):
    if any(line.startswith("realtime a reading:") and "stopped: its utterance ended" in line for line in server.logged()):
        break
    time.sleep(0.1)
else:
    raise SystemExit("no reading stopped when its buffer was committed: " + "".join(list(server.lines)[-5:]))
print("realtime: a reading under way when its buffer was committed stopped, and the commit was recognized whole")

# server_vad: the dumps joined with silences of 1.5 s, appended in pieces of 20 ms, are cut into the regions speech vad
# finds in the same samples on the CPU, which the server's detection runs on too; each region is committed and
# recognized as its samples sent as a WAV file, and their texts join into what chunking_strategy gives the whole.
alls = [pcm24(read_npy(os.path.join(d, "audio.npy"))) for d in dumps]
silence = b"\0" * 2 * 36000
joined = b"".join(p + silence for p in alls)
folder = tempfile.mkdtemp()
path = os.path.join(folder, "joined.wav")
with open(path, "wb") as f:
    f.write(wav24(joined))
detection = ["--threshold", "0.5", "--speech-pad-ms", "300", "--min-silence-duration-ms", "500", "--max-speech-duration-s", "10"]
regions = json.loads(subprocess.run([speech, "vad", vad, "--device", "cpu", "--threads", "1", "--format", "json", *detection, path],
                                    capture_output=True, check=True).stdout)["regions"]
shutil.rmtree(folder)
assert len(regions) >= len(alls), regions
vws = server.websocket("/v1/realtime")
assert vws.event()["type"] == "session.created"
vws.send(session({"model": name}, {"type": "server_vad"}))
updated = vws.event()
assert updated["session"]["audio"]["input"]["turn_detection"] == {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
                                                                 "silence_duration_ms": 500}, updated
appended(vws, joined, 960)
events = settle(vws)
items = list(dict.fromkeys(e["item_id"] for e in events))
assert len(items) == len(regions), (len(items), regions)


def milliseconds(seconds):
    """Milliseconds rounded half away from zero, as C's llround() rounds them."""
    whole = math.floor(seconds * 1000)
    return whole + (1 if seconds * 1000 - whole >= 0.5 else 0)


parts = []
for k, (item, region) in enumerate(zip(items, regions)):
    mine = [e for e in events if e["item_id"] == item]
    kinds = [e["type"].rsplit(".", 1)[1] for e in mine if not e["type"].endswith(".delta")]
    assert kinds == ["speech_started", "speech_stopped", "committed", "completed"], kinds
    first, last = round(region["start"] * 24000), round(region["end"] * 24000)
    assert mine[0]["audio_start_ms"] == milliseconds(first / 24000) and mine[1]["audio_end_ms"] == milliseconds(last / 24000), (mine[:2], region)
    done = mine[-1]
    status, _, body = transcribe([("response_format", "verbose_json")], [("file", "x.wav", wav24(joined[2 * first:2 * last]))])
    verbose = json.loads(body)
    assert done["transcript"] == verbose["text"] and done["usage"]["seconds"] == (last - first) / 24000, (done, verbose["text"])
    parts.append((first / 24000, {"text": done["transcript"], "stop": done["stop"], **({"languages": [x["code"] for x in done["languages"]]}
                                                                                       if "languages" in done else {})}))
assert [e["previous_item_id"] for e in events if e["type"] == "input_audio_buffer.committed"] == [None] + items[:-1], events
status, _, body = transcribe([("chunking_strategy", "auto")], [("file", "x.wav", wav24(joined))])
assert status == 200 and json.loads(body)["text"] == joined_transcript(parts, False)["text"], (body, parts)
print(f"realtime server_vad: {len(items)} utterances of {len(joined) / 48000:.1f} s appended in pieces of 20 ms, each at the bounds of "
      f"speech vad's region and completed with the text of its samples as a file, joined into chunking_strategy's text")
# Appended as fast as it is said, an utterance gets deltas after its speech started and before its commit.
paced(vws, longest + silence)
events = settle(vws)
kinds = kinds_of(events, events[0]["item_id"])
assert kinds[0] == "speech_started" and "delta" in kinds[:kinds.index("speech_stopped")] and kinds[-1] == "completed", kinds
print(f"realtime server_vad: {kinds[:kinds.index('committed')].count('delta')} deltas of an utterance appended in real time before its commit")
# A commit while speech is under way commits the buffer as the item its speech_started named, without speech_stopped,
# and the detection begins again on the audio that follows; a clear drops the utterance under way. The longest region
# cut at its middle is speech under way that has lasted long enough to be certain.
widest = max(regions, key=lambda r: r["end"] - r["start"])
under_way = joined[2 * round(widest["start"] * 24000):2 * round((widest["start"] + widest["end"]) / 2 * 24000)]
appended(vws, under_way, 960)
vws.send({"type": "input_audio_buffer.commit"})
events = settle(vws)
started = [e for e in events if e["type"] == "input_audio_buffer.speech_started"]
committed = [e for e in events if e["type"] == "input_audio_buffer.committed"]
assert len(started) == len(committed) == 1 and started[0]["item_id"] == committed[0]["item_id"] == events[-1]["item_id"], events
assert not any(e["type"] == "input_audio_buffer.speech_stopped" for e in events) and events[-1]["type"].endswith(".completed"), events
appended(vws, silence, 960)
assert settle(vws) == [], "the detection did not begin again after the commit"
appended(vws, under_way, 960)
vws.send({"type": "input_audio_buffer.clear"})
events = [vws.event()]
while events[-1]["type"] != "input_audio_buffer.cleared":
    events.append(vws.event())
assert [e["type"].rsplit(".", 1)[1] for e in events if not e["type"].endswith(".delta")] == ["speech_started", "cleared"], events
appended(vws, silence, 960)
assert settle(vws) == [], "a cleared utterance went on"
print("realtime server_vad: a commit while speech was under way committed the item speech_started named; a clear dropped it")
# Silence is not held: 30 MiB of it, past the bound of what a session holds, are taken, and a commit after it holds
# only what the detection keeps for a region still to come.
for _ in range(30):
    vws.send({"type": "input_audio_buffer.append", "audio": base64.b64encode(piece).decode()})
assert settle(vws) == [], "silence past the bound was refused or cut into utterances"
vws.send({"type": "input_audio_buffer.commit"})
events = settle(vws)
assert events[-1]["type"].endswith(".completed") and events[-1]["usage"]["seconds"] <= 0.5, events[-1]
vws.close()
print(f"realtime server_vad: 30 MiB of silence taken within the bound, and a commit after it of {events[-1]['usage']['seconds']} s")
# A client that goes away while its commit is transcribed stops the transcription, gives up the turn of the commit
# that waits, and the model serves the next request.
for _ in range(2):
    appended(ws, pcms[0])
    ws.send({"type": "input_audio_buffer.commit"})
    assert ws.event()["type"] == "input_audio_buffer.committed"
ws.sock.close()
status, _, body = transcribe([], [("file", "x.wav", wav24(pcms[0]))])
assert status == 200 and json.loads(body)["text"] == wants[0][0], body
print("realtime: a session that goes on after its errors, and a client gone while its commit was transcribed")
expect_error(server.websocket("/v1/realtime?model=gpt-realtime"), 404, "model_not_found", "model", "/v1/realtime for another model")
expect_error(server.websocket("/v1/realtime?intent=transcription"), 400, "unknown_parameter", "intent", "/v1/realtime with intent")
expect_error(server.websocket("/v1/realtime", {"Origin": "http://elsewhere.test"}), 403, "origin_not_allowed", None,
             "/v1/realtime from another origin")
expect_error(server.websocket("/v1/realtime", {"Host": "elsewhere.test"}), 403, "host_not_allowed", None, "/v1/realtime for another Host")
ws = server.websocket("/v1/realtime", {"Origin": ORIGIN})
assert ws.event()["type"] == "session.created"
ws.close()
print("realtime: /v1/realtime from --cors-origin's origin opens")

out = server.stop()
if out:
    raise SystemExit(f"the server wrote to stdout: {out[:200]!r}")
print("ok")
