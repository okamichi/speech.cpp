# Server and page

This page describes `speech serve`, which serves a synthesis model, a recognition model and a detection model over HTTP
with a subset of OpenAI's audio API and its Realtime transcription, and a page on which to try models.

```
speech serve [MODEL [MODEL [MODEL]]] [--open] [--host 127.0.0.1] [--port 8080] [--cors-origin ORIGIN|*]...
             [--add-voice NAME=FILE]... [--voices-dir DIR] [--irodori-compatible]
             [--device NAME] [--threads N] [--no-warmup] [request options]
```

```sh
speech serve --open                                   # the page, without a model until one is picked there
speech serve qwen3-tts-0.6b qwen3-asr-0.6b            # speech and transcriptions
speech serve reazonspeech-v2 silero-vad               # transcriptions, also by the regions where someone speaks
speech serve irodori-tts-mf --add-voice me=me.voice.gguf --cors-origin http://localhost:5173
speech serve irodori-tts-mf --steps 8 --add-voice me=me.voice.gguf  # 8 steps unless a request sets steps
speech serve qwen3-tts-0.6b qwen3-asr-0.6b --language ja --asr.language auto
```

The local `run-*.sh` scripts start the Metal server on port 8088. Pass
`--irodori-compatible --voices-dir ./voices` to a script for Irodori compatibility; the script warns on stderr
if either option is missing. `--voices-dir` can name another folder.

The server holds at most one model of each task: synthesis, recognition and detection (Silero VAD), whose model
transcriptions with `chunking_strategy` and Realtime sessions with `server_vad` use. It loads the models given, listens once they are ready, and logs to stderr.

| Option | Meaning |
|---|---|
| `--host ADDRESS` | the address to listen on, 127.0.0.1 unless given. The server has no authentication and no TLS, so a server reachable from other machines belongs behind a proxy that adds them. The page is served only on 127.0.0.1, `::1` or `localhost` |
| `--port N` | the port, 8080 unless given, or 0 for any free one |
| `--open` | open the page in the browser once the server listens; without a MODEL the server starts with none |
| `--cors-origin ORIGIN` | an origin a web page may call the server from, such as `http://localhost:5173`, or `*` for any; repeatable. A web page of any other origin is refused (403 `origin_not_allowed`); curl and scripts send no origin and are not |
| `--add-voice NAME=FILE` | add a voice to the synthesis model from a voice file or a WAVE file; repeatable |
| `--voices-dir DIR` | retain uploaded reference WAV files here and restore them when a synthesis model loads; creates the folder if needed |
| `--irodori-compatible` | accept the Irodori bridge request format and voice endpoints below; requires an Irodori RF model with captions, voice `none`, and `--voices-dir` |
| `--device`, `--threads`, `--no-warmup` | as for the worker, for the synthesis and recognition models the server loads, the page's included. A detection model runs on the CPU with one thread |
| a request option, such as `--steps 8` or `--language ja` | the default of every request that does not set it, for each model given that takes the option. A request's own value wins; an option that no model given takes, or a value out of its range, stops the server before it listens. The detection model's options, such as `--threshold`, come before OpenAI's server_vad defaults in `chunking_strategy` and Realtime turn detection |
| `--tts.`, `--asr.` or `--vad.` and a request option, such as `--asr.language en` | the same for the synthesis, recognition or detection model alone, winning over the option without the prefix. A model the page loads later keeps its own defaults |

## Endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | returns `status: "ok"`, the synthesis `model`, and `runtime` with `loaded`, `loading` and `checkpoint` (its GGUF path, or null) |
| `GET /v1/models` | lists the models held, synthesis, recognition and detection in that order, as OpenAI's model objects whose `id` is the model's name, with `defaults`, the request options the command line gave the model (`{}` without any), and `speech`, the model information ([c-api.md](c-api.md#model-information-as-json)), which keeps the model's own defaults |
| `GET /v1/models/{id}` | one model; another id is a 404 (`model_not_found`) |
| `POST /v1/audio/speech` | speaks a text, as [OpenAI's create speech](https://developers.openai.com/api/reference/resources/audio/subresources/speech/methods/create) does |
| `POST /v1/audio/transcriptions` | recognizes the speech in a WAV file, as [OpenAI's create transcription](https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create) does |
| `GET /v1/realtime` | a WebSocket that transcribes audio as it is sent, as [OpenAI's Realtime transcription](https://developers.openai.com/api/reference/resources/realtime/client-events) does ([below](#realtime-transcription)) |

An endpoint of a task the server holds no model of answers a 404 whose message says so. One whose model the page is
replacing answers a 503 (`model_loading`) with `Retry-After`.

Each model serves one request at a time, in the order they arrive. A client that disconnects stops its request.

## Speech

A speech request is a JSON object:

| Member | Meaning |
|---|---|
| `input` | the text, required |
| `model` | the synthesis model's `id`, or left out. Any other model is a 404 (`model_not_found`) |
| `voice` | one of the model's voices, required: a Qwen3-TTS speaker or a voice of `--add-voice` |
| `response_format` | `wav` (the default) or `pcm`. OpenAI's default is `mp3`, which speech.cpp does not encode; `mp3`, `opus`, `aac` and `flac` are refused |
| `stream_format` | `audio` (the default) or `sse`, which needs `pcm` |
| `speed`, `instructions` | OpenAI's members, as the options of the same names |
| any other option | every option of the vocabulary by its name ([c-api.md](c-api.md#options)), such as `language`, `seed`, `steps` or `temperature`, which the model checks |

A member speech.cpp does not take is refused rather than ignored, and so is `instructions` for a model that takes none,
such as Qwen3-TTS 0.6B. A request without `seed` gets one drawn, which `X-Speech-Seed` returns; the same request with that
seed gives the same audio on the same device.

Irodori-TTS speaks at most 30 s at a time, so the server speaks an `input` of several sentences one sentence at a time,
as `speech tts` does, with the same voice, options and seed, and a pause of 0.9 s between them. A `pcm` stream sends
the first sentence's audio as soon as it is made. A sentence too long for one request is cut at its commas, or at its
spaces; a run of text with neither that is too long is refused (400, `param` `input`). Qwen3-TTS speaks `input` whole,
up to 24565 tokens.

| Response | What it sends |
|---|---|
| `wav` | the whole file, 16-bit mono at the model's rate, once the speech is made, with the headers `X-Sample-Rate`, `X-Speech-Seed` and `X-Speech-Stop` (`complete`, `max_seconds` or `model_limit`) |
| `pcm` | raw 16-bit little-endian mono at the model's rate, `Content-Type: audio/pcm`, streamed as the model makes it, with `X-Sample-Rate` and `X-Speech-Seed`. The headers leave before the speech ends, so they cannot carry the stop reason |
| `pcm` with `stream_format: "sse"` | the same PCM as server-sent events: `{"type":"speech.audio.delta","audio":"<base64 PCM>"}` for each chunk, then `{"type":"speech.audio.done","seed":42,"samples":134400,"stop":"complete"}`. OpenAI's done event carries the usage in tokens, which speech.cpp does not count |

```sh
# A WAV file
curl http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna"}' -o out.wav

# PCM played as it arrives; the rate is the X-Sample-Rate header (24000 for Qwen3-TTS, 48000 for Irodori-TTS)
curl -sN http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm"}' |
    ffplay -nodisp -autoexit -f s16le -ar 24000 -ch_layout mono -i -
# On Linux with ALSA: ... | aplay -f S16_LE -r 24000 -c 1

# Server-sent events
curl -N http://127.0.0.1:8080/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm", "stream_format": "sse"}'
```

From Python with [requests](https://requests.readthedocs.io), streaming PCM into a WAV file as it arrives:

```python
import requests
import wave

with requests.post("http://127.0.0.1:8080/v1/audio/speech",
                   json={"input": "明日の東京は晴れです。", "voice": "ono_anna", "response_format": "pcm"},
                   stream=True) as r:
    r.raise_for_status()
    with wave.open("out.wav", "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(r.headers["X-Sample-Rate"]))
        for chunk in r.iter_content(chunk_size=None):
            w.writeframes(chunk)
```

## Irodori bridge compatibility

Start with an RF model that supports captions and reference voices:

```sh
build/speech serve models/Irodori-TTS-v4-Large.gguf --device MTL0 \
    --irodori-compatible --voices-dir ./voices --host 127.0.0.1 --port 8080
```

The bridge can keep its Irodori provider and use `http://127.0.0.1:8080`. Speech requests accept the model alias
`irodori-tts` and an `irodori` object:

```json
{"model":"irodori-tts","input":"準備はできました。","voice":"none","response_format":"wav","speed":1,
 "irodori":{"caption":"落ち着いた声","seed":1520596899881326291,"num_steps":20,"max_seconds":120,
            "chunking_enabled":false,"cfg_scale_text":3,"cfg_scale_caption":3,"cfg_scale_speaker":5}}
```

`caption`, `num_steps` and `cfg_scale_caption` map to `instructions`, `steps` and `cfg_scale_instructions`.
`seed`, `cfg_scale_text` and `cfg_scale_speaker` keep their names. Do not give an option both here and at the top
level. Unknown members are refused; null is absent. A nested seed must be an integer from 0 through 2^63−1 and is
reduced modulo 2^53; `X-Speech-Seed` returns this effective seed. Top-level seeds keep their usual range.

With an `irodori` object, `max_seconds` caps the whole output, including pauses, at 120 seconds by default. It must
be positive and at most 120. Reaching it returns a valid WAV with `X-Speech-Stop: max_seconds`.
`chunking_enabled` accepts either boolean: texts are split to fit the model in either case. A piece too long to fit
without punctuation or spaces is split at Unicode boundaries. This differs from Irodori-TTS-Server's whole-text
synthesis; joins retain speech.cpp's pause of at least 0.9 seconds, counting existing silence.

Captions are stripped and truncated at the model's token limit, as in the official runtime. WAV responses report
`X-Speech-Instructions-Truncated: true` when this happens, and the server logs it. Caption guidance with an empty
caption, and speaker guidance with `voice: "none"`, are accepted and have no effect. Native requests keep their
usual validation unless they explicitly set `irodori_compatible: true`.

### Reference voices

With `--irodori-compatible`, `GET /v1/audio/voices` returns:

```json
{"object":"list","data":[{"id":"none","object":"voice","no_ref":true}]}
```

Register a WAV recording with `POST /v1/audio/voices`:

```sh
curl http://127.0.0.1:8080/v1/audio/voices -F file=@reference.wav -F voice_id=narrator
```

Success is HTTP 201 with `id`, `object: "voice_file"`, `filename`, `bytes` and `created_at` (Unix seconds).
The ID defaults to the filename without its extension. IDs use ASCII letters, digits, underscores and hyphens;
`none`, `no_ref`, `no-ref`, `null` and `text-only` are reserved, regardless of case. Duplicate IDs return 409.
Recordings must be WAV files supported by the model; convert other formats first. Replacements and deletion over
HTTP are not provided.

Uploads are stored as `ID.wav` in `--voices-dir`. They appear in both voice listings and survive restarts and model
changes; a model encodes them when it loads. A corrupt or incompatible recording fails the load with its path.
The page's uploads also persist when this folder is configured. `--add-voice` alone remains temporary and must not
collide with a stored ID. Stop the server before editing or removing stored recordings.

Matching requests are repeatable on the same speech.cpp device, but their audio does not match the Python server's
waveform or necessarily its voice quality. Different 64-bit seeds can map to the same effective seed.

## Transcriptions

A transcription request is a `multipart/form-data` form, as OpenAI's API takes it. The upload may be up to 25 MB,
OpenAI's limit.

| Member | Meaning |
|---|---|
| `file` | the audio, required: a WAV file, 16-, 24- or 32-bit PCM or 32-bit float at any rate, its channels averaged and resampled to the model's rate. Any other file is refused with a 400 (`param` `file`); convert it first (`ffmpeg -i in.mp3 out.wav`) |
| `model` | the recognition model's `id`, or left out; any other model is a 404 (`model_not_found`) |
| `language` | a tag of one of the model's languages, or `auto` (the default) |
| `prompt` | what the model is told of the audio before it hears it, for a model that takes it (Qwen3-ASR) |
| `response_format` | `json` (the default), which answers `{"text":"..."}`; `text`, the text alone as `text/plain`; or `verbose_json` (below). `srt`, `vtt` and `diarized_json` are refused: speech.cpp gives neither subtitles nor speakers |
| `timestamp_granularities[]` | `segment`, with `verbose_json`; a model that gives no times refuses it. `word` is refused |
| `decoding` | speech.cpp's own: `beam` (the default) or `greedy`, for a model that takes it (reazonspeech-nemo-v2) |
| `chunking_strategy` | `auto`, or OpenAI's `server_vad` object: recognize each region where the detection model finds that someone speaks (below) |

`verbose_json` answers `{"task":"transcribe","language":…,"duration":…,"text":…,"segments":[{"id":0,"start":…,"end":…,"text":…}]}`.
`duration` is the file's length in seconds. `language` is the tag of the language Qwen3-ASR heard (`ja`, or `ja,en` for a
long recording in several), and is left out by FastConformer, which names none. `segments` come from a model that gives
times (FastConformer). Members the recognizers have no value for are left out.

Every answer carries `X-Speech-Stop`: `complete`, or `model_limit` when the model reached the most tokens it writes, with
the text written up to there. OpenAI's other members (`temperature`, `stream`, `include[]` and the rest) are refused with
a 400 rather than ignored.

```sh
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@utterance.wav -F response_format=text
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F response_format=verbose_json
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F language=ja -F prompt="Claude Code、渋谷"
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F decoding=greedy   # reazonspeech-nemo-v2
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav -F chunking_strategy=auto
curl http://127.0.0.1:8080/v1/audio/transcriptions -F file=@meeting.wav \
    -F 'chunking_strategy[type]=server_vad' -F 'chunking_strategy[silence_duration_ms]=800'
```

### By regions

With `chunking_strategy`, the detection model finds where someone speaks in the file, each region is recognized alone,
and the texts are joined in order, with a space between two regions unless either side is Japanese or Chinese text.
`segments` have the times of the whole file, and `language` lists the languages heard in order. A file in which no one
speaks gives an empty text. A recognizer given a long stretch with several sentences can drop whole sentences, and write
words where no one speaks; by regions it hears one or a few sentences at a time.

`chunking_strategy=auto` takes the defaults below. A `server_vad` object goes as OpenAI's SDKs send it in a form, one
field per member:

| Field | Default | Meaning |
|---|---|---|
| `chunking_strategy[type]` | | `server_vad`, required with the others |
| `chunking_strategy[threshold]` | 0.5 | the speech probability from which audio counts as speech, 0 to 1 |
| `chunking_strategy[prefix_padding_ms]` | 300 | the audio kept before and after each region |
| `chunking_strategy[silence_duration_ms]` | 500 | the silence that ends a region |

A region is at most 10 s long, cut at its longest silence. A request with `chunking_strategy` to a server without a
detection model is a 400 that says to give it one: `speech serve reazonspeech-v2 silero-vad`.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
with open("meeting.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="reazonspeech-nemo-v2", file=f, chunking_strategy="auto").text)
```

## Realtime transcription

`/v1/realtime` speaks OpenAI's Realtime API over a WebSocket, for a transcription session: the client appends audio as
it records it, and the server answers with the text of each utterance, which the client commits itself, or which, with
`turn_detection` `server_vad`, the detection model finds. OpenAI's client connects to it by its base URL alone:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
with client.realtime.connect(model="reazonspeech-nemo-v2") as connection:
    connection.session.update(session={"type": "transcription", "audio": {"input": {
        "format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"language": "ja"}, "turn_detection": None}}})
    pcm = open("utterance.raw", "rb").read()   # 16-bit little-endian mono PCM at 24000 Hz
    connection.input_audio_buffer.append(audio=base64.b64encode(pcm).decode())
    connection.input_audio_buffer.commit()
    for event in connection:
        if event.type == "conversation.item.input_audio_transcription.completed":
            print(event.transcript)
            break
```

The address is `ws://127.0.0.1:8080/v1/realtime`, with `?model=` the recognition model's `id` or nothing. The
session transcribes with the recognition model held, and starts with `turn_detection` `null`: the client commits each
utterance.

| Client event | What it does |
|---|---|
| `session.update` | sets `session.audio.input.transcription`: `model` (the recognition model's `id`), `language`, or `languages` with one tag, and `prompt`, each left as it was when left out and taken away by `null`; or `null`, after which commits are not transcribed. Sets `turn_detection`: `null`, or `{"type": "server_vad"}` with `threshold`, `prefix_padding_ms` and `silence_duration_ms` (below). `session.type` is `"transcription"`, the format `{"type": "audio/pcm", "rate": 24000}`, and `noise_reduction` `null` |
| `input_audio_buffer.append` | adds `audio`, base64 of 16-bit little-endian mono PCM at 24000 Hz |
| `input_audio_buffer.commit` | transcribes the buffer: what was appended since the last commit or clear |
| `input_audio_buffer.clear` | drops the buffer |

| Server event | When |
|---|---|
| `session.created`, `session.updated` | on connecting, and after each `session.update`, with the whole configuration |
| `input_audio_buffer.speech_started` | with `server_vad`, once speech has begun and lasted long enough to be kept, with `audio_start_ms` and the `item_id` of the utterance |
| `input_audio_buffer.speech_stopped` | with `server_vad`, once the utterance has ended, with `audio_end_ms` and its `item_id` |
| `input_audio_buffer.committed` | at once for a commit, or with `server_vad` after `speech_stopped`, with its `item_id` and the `previous_item_id` |
| `input_audio_buffer.cleared` | for a clear |
| `conversation.item.input_audio_transcription.delta` | while the utterance goes on: text that adds to the end of what came before (below) |
| `conversation.item.input_audio_transcription.completed` | once the utterance is recognized whole, with the `transcript`, the whole text, which replaces the deltas; `usage` `{"type": "duration", "seconds": …}`, `languages` where the model names them (Qwen3-ASR), and speech.cpp's own `stop`, `complete` or `model_limit` |
| `conversation.item.input_audio_transcription.failed` | an utterance the model could not transcribe, with OpenAI's error object |
| `error` | an event the server does not take, with OpenAI's error object and the client's `event_id` |

While an utterance goes on, the server recognizes it again as audio comes, as often as the model keeps up, and the
deltas carry the beginning that two of these readings in a row agree on. They may stop short of the text, or differ
from it where a later reading changed its mind: a client shows them while someone speaks and replaces them with the
completed `transcript`, as it does for OpenAI's `gpt-live-transcribe`. With Qwen3-ASR, each reading and the completed
`transcript` are given the reading before as their `draft`, which makes them sooner where the text stays the same.

Utterances are transcribed one after another, each in its turn among the server's other transcriptions, taken when it
is committed. A session holds at most 25 MB of audio not yet transcribed, in its buffer or committed, which is 546 s;
an append past that is an `error` event (`input_audio_buffer_full`) and adds nothing. A member or a value
speech.cpp does not take is an `error` event rather than ignored: a conversation session (`session.type: "realtime"`),
`semantic_vad`, the members of `server_vad` that steer a response (`create_response`, `interrupt_response`,
`idle_timeout_ms`), `noise_reduction`, the formats `audio/pcmu` and `audio/pcma` and rates other than 24000, `include`
(log probabilities), `keywords`, `delay`, more than one language, and every other client event.

### Turn detection

With `turn_detection` `server_vad`, the detection model finds where someone speaks, as `chunking_strategy` does in a
file, and each region is an utterance, committed and transcribed as it ends: the same audio gives the same regions and
texts as `chunking_strategy` with the same values. A server without a detection model answers `server_vad` with an
`error` event that says to give it one: `speech serve reazonspeech-v2 silero-vad`.

| Member | Default | Meaning |
|---|---|---|
| `threshold` | 0.5 | the speech probability from which audio counts as speech, 0 to 1 |
| `prefix_padding_ms` | 300 | the audio kept before and after each utterance |
| `silence_duration_ms` | 500 | the silence that ends an utterance |

An utterance is at most 10 s long, cut at its longest silence. `speech_started` comes about 0.26 s after the speech
begins, once it has lasted long enough to be kept, and the utterance is committed about 0.6 s after the speech ends. The buffer
keeps only the utterance under way, or the last moments of silence, so a session can stay open through silence; a
commit while `server_vad` runs transcribes it as an utterance, the one that `speech_started` named if speech is under
way, and a clear drops it, and in both cases turn detection begins again on the audio that follows.

A connection that the server refuses gets an HTTP error before the WebSocket opens: a server without a recognition
model (404, or 503 while the page loads one), a model other than the recognition model held (404), a query member other
than `model` (400), a web page of an origin `--cors-origin` does not allow (403),
and, while the server listens on 127.0.0.1, `::1` or `localhost`, a Host other than those names with the server's port
(403), as for the page. While the page loads another recognition model, a commit fails with `model_loading`; once it is
loaded, a session that named no model goes on with it, and one that named the previous model, in the address or in
`session.update`, fails with `model_not_found`. A session with `server_vad` lets the detection model go when the page
replaces it, committing the utterance under way, and goes on with the new one.

## Errors

Errors have OpenAI's shape, `{"error":{"message":...,"type":...,"param":...,"code":...}}`, and `param` is the input at
fault:

| Category | HTTP | `type` | `code` |
|---|---|---|---|
| `invalid_argument` | 400 | `invalid_request_error` | `invalid_value` |
| `unsupported` | 400 | `invalid_request_error` | `unsupported_parameter` |
| `out_of_range` | 400 | `invalid_request_error` | `unsupported_value` |
| `model_file`, `device`, `out_of_memory`, `io`, `internal` | 500 | `server_error` | the category's name |

So an unknown voice is a 400 `unsupported_value` with `param` `voice`, and a `speed` on Qwen3-TTS a 400
`unsupported_parameter`. A member the server does not have is a 400 `unknown_parameter`.

A refused request gets its error status before any audio. An error while a stream runs ends a `pcm` stream early, which
the client reads as a broken transfer, and an SSE stream with `{"type":"error","error":{...}}`.

## The page

`speech serve --open` opens a page on which to try models without writing a request.

- Three tabs: Speak, Transcribe and Live. At the top of each, a picker lists the catalog's models of its task, each in
  every type, with their sizes, whether they are fetched and their languages. Picking one fetches it, with progress that can be
  cancelled and is resumed the next time, and loads it in place of the model of its task. Transcribe and Live use the
  same recognition model and the same detection model. The options' fields, Live's pause among them, start at the
  defaults the command line gave a model, and a field left as it is keeps them.
- Speak takes a text and the model's options, plays the speech as the server streams it, and can play it again, save it
  as a WAVE file or hand it to Transcribe. For Irodori-TTS it makes a voice from a recording dropped on it or recorded in
  the browser.
- Transcribe takes an audio file in any format the browser decodes, or a recording, and shows the text, the language
  heard, the time it took and the segments. A second picker there holds a detection model: with one, the audio is
  transcribed by the regions where someone speaks (`chunking_strategy`); without one, audio up to a minute is
  transcribed whole, and longer audio asks for one, since a FastConformer model loses sentences of a long stretch and
  parakeet's memory grows with the square of its length.
- Live transcribes the microphone while you speak, over `/v1/realtime` with `server_vad`, so it needs a detection model,
  which a second picker holds as in Transcribe. The text of an utterance under way is shown in grey and grows as the
  readings agree; once a pause ends the utterance, its final text replaces it. More options set that pause, 500 ms
  unless changed.
- Transcribe sends a file whole, or, where it is larger than the server takes in a request, in pieces of about
  12 minutes cut at a pause, whose texts are joined.
- The page is built into `speech` and needs nothing from the network.

The server prints the page's address, `http://127.0.0.1:8080/`. Other web pages cannot use it: the server refuses a
request from another origin than its own, and a Host other than 127.0.0.1, `localhost` or `[::1]` with its port. It is
served while the server listens on 127.0.0.1, `::1` or `localhost`. To use it from another machine, forward the port with SSH, the same number at both
ends (`ssh -L 8080:127.0.0.1:8080 host`), and open the printed address in your browser. The page loads only models of the catalog,
and removes none; `speech rm` does.
