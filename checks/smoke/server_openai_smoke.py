# /// script
# requires-python = ">=3.10"
# dependencies = ["openai[realtime]==3.26.0"]
# ///
"""Drives `speech serve` with OpenAI's own Python client, openai-python, pinned by this script's lock, to show that a
program written for OpenAI's API works by changing the address alone.

It lists the models; transcribes the first dump's audio with chunking_strategy "auto" and with a server_vad object, as
the client sends them, against the same requests sent by hand; and opens a Realtime transcription session with
client.realtime.connect(model=...), whose events the client parses into its own types: session.created, a
session.update of a transcription session answered by session.updated, PCM appended in pieces and committed, answered
by input_audio_buffer.committed, deltas and the completed of the text that the same samples give as a 24000 Hz WAV
file; a session.update to turn_detection server_vad, after which the same PCM followed by silence, appended in pieces of
20 ms, gives input_audio_buffer.speech_started, .speech_stopped and .committed and the completed of each region where
speech vad finds speech, joined into the text of chunking_strategy "auto"; an error event for semantic_vad; and
input_audio_buffer.cleared.

usage: uv run --script checks/smoke/server_openai_smoke.py <speech> <recognition.gguf> <detection.gguf> <dump folder>
"""

import array
import base64
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import uuid

import openai
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server_client import Server  # noqa: E402
from worker_client import joined_transcript, read_npy  # noqa: E402

if len(sys.argv) != 5:
    raise SystemExit("usage:" + __doc__.split("usage:")[1].rstrip())
speech, model, vad, dump = sys.argv[1:]
server = Server(speech, [model, vad])
client = OpenAI(base_url=f"http://127.0.0.1:{server.port}/v1", api_key="unused")

listed = [m.id for m in client.models.list()]
name = listed[0]
print(f"openai {openai.__version__}: models.list() gives {listed}")


def wav(pcm, rate):
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


def by_hand(fields, content):
    boundary = uuid.uuid4().hex
    body = b"".join(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields)
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.wav\"\r\n\r\n").encode() + content
    body += f"\r\n--{boundary}--\r\n".encode()
    status, _, answer = server.call("POST", "/v1/audio/transcriptions", body, {"Content-Type": f"multipart/form-data; boundary={boundary}"})
    assert status == 200, answer
    return json.loads(answer)["text"]


samples = read_npy(os.path.join(dump, "audio.npy"))
rate = json.loads(server.call("GET", f"/v1/models/{name}")[2])["speech"]["sample_rate"]
pcm = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in samples]).tobytes()
for strategy, fields in (("auto", [("chunking_strategy", "auto")]),
                         ({"type": "server_vad", "threshold": 0.6, "silence_duration_ms": 700},
                          [("chunking_strategy[type]", "server_vad"), ("chunking_strategy[threshold]", "0.6"),
                           ("chunking_strategy[silence_duration_ms]", "700")])):
    got = client.audio.transcriptions.create(model=name, file=("x.wav", io.BytesIO(wav(pcm, rate)), "audio/wav"), chunking_strategy=strategy)
    assert got.text == by_hand(fields, wav(pcm, rate)) and got.text, (got.text, strategy)
print(f"audio.transcriptions.create() with chunking_strategy auto and a server_vad object: the text of the same forms sent by hand, "
      f"{got.text[:30]!r}")

# The dump's samples at 24000 Hz, the one rate of audio/pcm, by linear interpolation.
n = len(samples) * 24000 // rate
at24 = [samples[int(i * rate / 24000)] + (samples[min(int(i * rate / 24000) + 1, len(samples) - 1)] - samples[int(i * rate / 24000)])
        * (i * rate / 24000 - int(i * rate / 24000)) for i in range(n)]
pcm24 = array.array("h", [max(-32768, min(32767, round(x * 32768))) for x in at24]).tobytes()
want = by_hand([], wav(pcm24, 24000))

with client.realtime.connect(model=name) as connection:
    created = connection.recv()
    assert type(created).__name__ == "SessionCreatedEvent" and created.session.type == "transcription", created
    connection.session.update(session={"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                                                                                     "transcription": {"model": name}, "turn_detection": None}}})
    updated = connection.recv()
    assert type(updated).__name__ == "SessionUpdatedEvent" and updated.session.audio.input.transcription.model == name, updated
    for start in range(0, len(pcm24), 9600):
        connection.input_audio_buffer.append(audio=base64.b64encode(pcm24[start:start + 9600]).decode())
    connection.input_audio_buffer.commit()
    events = [connection.recv()]
    while events[-1].type != "conversation.item.input_audio_transcription.completed":
        events.append(connection.recv())
    kinds = [type(e).__name__ for e in events]
    assert kinds[0] == "InputAudioBufferCommittedEvent" and set(kinds[1:-1]) <= {"ConversationItemInputAudioTranscriptionDeltaEvent"}, kinds
    committed, completed = events[0], events[-1]
    assert all(e.item_id == committed.item_id for e in events) and all(e.delta for e in events[1:-1]) and completed.transcript == want, (events, want)
    assert completed.usage.type == "duration" and completed.usage.seconds == len(pcm24) / 2 / 24000, completed.usage
    # server_vad: the same samples and 1.5 s of silence, appended in pieces of 20 ms, cut into the regions speech vad finds
    # in them on the CPU, where the server's detection runs.
    connection.session.update(session={"type": "transcription", "audio": {"input": {"turn_detection": {"type": "server_vad",
                                                                                                        "silence_duration_ms": 500}}}})
    updated = connection.recv()
    assert type(updated).__name__ == "SessionUpdatedEvent" and updated.session.audio.input.turn_detection.type == "server_vad", updated
    spoken = pcm24 + b"\0" * 72000
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "spoken.wav")
    with open(path, "wb") as f:
        f.write(wav(spoken, 24000))
    regions = json.loads(subprocess.run([speech, "vad", vad, "--device", "cpu", "--threads", "1", "--format", "json", "--speech-pad-ms", "300",
                                         "--min-silence-duration-ms", "500", "--max-speech-duration-s", "10", path],
                                        capture_output=True, check=True).stdout)["regions"]
    os.remove(path)
    os.rmdir(folder)
    for start in range(0, len(spoken), 960):
        connection.input_audio_buffer.append(audio=base64.b64encode(spoken[start:start + 960]).decode())
    detected = []
    while sum(type(e).__name__ == "ConversationItemInputAudioTranscriptionCompletedEvent" for e in detected) < len(regions):
        detected.append(connection.recv())
    first = [type(e).__name__ for e in detected if e.item_id == detected[0].item_id and not e.type.endswith(".delta")]
    assert first == ["InputAudioBufferSpeechStartedEvent", "InputAudioBufferSpeechStoppedEvent", "InputAudioBufferCommittedEvent",
                     "ConversationItemInputAudioTranscriptionCompletedEvent"], first
    # The times count from the start of the session's audio, the commit's before the detection's.
    origin = len(pcm24) // 2
    ms = [math.floor((origin + round(regions[0][k] * 24000)) / 24 + 0.5) for k in ("start", "end")]
    assert [detected[0].audio_start_ms, detected[1].audio_end_ms] == ms, (detected[:2], ms)
    done = [e for e in detected if e.type.endswith(".completed")]
    parts = [(0, {"text": e.transcript, "stop": "complete"}) for e in done]
    assert joined_transcript(parts, False)["text"] == by_hand([("chunking_strategy", "auto")], wav(spoken, 24000)), (done, regions)
    connection.session.update(session={"type": "transcription", "audio": {"input": {"turn_detection": {"type": "semantic_vad"}}}})
    refused = connection.recv()
    assert type(refused).__name__ == "RealtimeErrorEvent" and refused.error.param == "session.audio.input.turn_detection.type", refused
    connection.input_audio_buffer.clear()
    assert type(connection.recv()).__name__ == "InputAudioBufferClearedEvent"
print(f"realtime.connect(): {', '.join(kinds)} for a commit, the transcript of the same samples as a 24000 Hz file: {want[:30]!r}; with "
      f"server_vad {', '.join(dict.fromkeys(type(e).__name__ for e in detected))} for {len(regions)} regions, joined into chunking_strategy's "
      f"text; and RealtimeErrorEvent for semantic_vad, parsed by the client")
out = server.stop()
assert out == b"", out[:200]
print("ok")
