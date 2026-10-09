"""Checks the bridge's requests, bounded audio and durable voice uploads through HTTP.

usage: python3 checks/smoke/irodori_compat_smoke.py <speech> <RF model.gguf> [--device NAME]
"""
import concurrent.futures
import copy
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave

from server_client import Server

speech, model, *options = sys.argv[1:]
speech, model = os.path.abspath(speech), os.path.abspath(model)
metadata = json.loads(subprocess.check_output([speech, 'info', model, '--json', '--meta']))['meta']


def upload(server, name, audio, page=False, duplicate=False):
    boundary = uuid.uuid4().hex
    field = 'name' if page else 'voice_id'
    part = f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{name}\r\n'.encode()
    body = part * (2 if duplicate else 1)
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    body += audio + f'\r\n--{boundary}--\r\n'.encode()
    return server.call('POST', '/speech/voices' if page else '/v1/audio/voices', body,
                       {'Content-Type': f'multipart/form-data; boundary={boundary}'})


def wav_samples(body):
    assert body[:4] == b'RIFF' and struct.unpack('<I', body[4:8])[0] + 8 == len(body)
    with wave.open(io.BytesIO(body)) as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1
        assert 0 < w.getnframes() <= w.getframerate() * 120
        return w.getnframes(), w.getframerate()


with tempfile.TemporaryDirectory(prefix='speech-compatible-') as temp:
    directory = Path(temp) / 'voices'
    args = [model, '--no-warmup', '--irodori-compatible', '--voices-dir', str(directory), *options]
    server = Server(speech, args)
    payload = {'model': 'irodori-tts', 'input': '準備はできました。', 'voice': 'none', 'response_format': 'wav', 'speed': 1,
               'irodori': {'caption': '', 'seed': 1520596899881326291, 'num_steps': 1, 'max_seconds': 120,
                           'chunking_enabled': False, 'cfg_scale_text': 3, 'cfg_scale_caption': 7, 'cfg_scale_speaker': 8}}
    status, _, body = server.call('GET', '/health')
    health = json.loads(body)
    assert status == 200 and health['runtime']['loaded'] and health['runtime']['checkpoint'] == model
    assert json.loads(server.call('GET', '/v1/audio/voices')[2])['data'] == [{'id': 'none', 'object': 'voice', 'no_ref': True}]
    status, headers, wav = server.post_json('/v1/audio/speech', payload)
    assert status == 200, wav
    wav_samples(wav)
    assert int(headers['x-speech-seed']) == payload['irodori']['seed'] % 2**53
    assert server.post_json('/v1/audio/speech', payload)[2] == wav
    native = {'input': payload['input'], 'voice': 'none', 'seed': payload['irodori']['seed'] % 2**53,
              'steps': 1, 'cfg_scale_text': 3, 'cfg_scale_instructions': 0, 'cfg_scale_speaker': 0}
    assert server.post_json('/v1/audio/speech', native)[2] == wav, 'inactive guidance changed the sampling output'
    assert server.call('GET', '/v1/audio/voices', headers={'Origin': 'https://foreign.invalid'})[0] == 403
    for token in ('1e2', '1.0'):
        encoded = json.dumps(payload).replace(str(payload['irodori']['seed']), token).encode()
        assert server.call('POST', '/v1/audio/speech', encoded, {'Content-Type': 'application/json'})[0] == 400
    for seed in (0, 2**53-1, 2**53, 2**63-1):
        case = copy.deepcopy(payload)
        case['irodori'].update(seed=seed, max_seconds=0.1)
        status, h, audio = server.post_json('/v1/audio/speech', case)
        assert status == 200 and int(h['x-speech-seed']) == seed % 2**53, audio
        n, rate = wav_samples(audio)
        assert n == int(rate * 0.1) and h['x-speech-stop'] == 'max_seconds'
    for seed in (-1, 2**63, 1.5):
        case = copy.deepcopy(payload); case['irodori']['seed'] = seed
        assert server.post_json('/v1/audio/speech', case)[0] == 400
    for key, value in [('num_steps', 'x'), ('chunking_enabled', 'false'), ('max_seconds', 0), ('max_seconds', 121), ('unknown', 1)]:
        case = copy.deepcopy(payload); case['irodori'][key] = value
        assert server.post_json('/v1/audio/speech', case)[0] == 400, (key, value)
    case = copy.deepcopy(payload); case['seed'] = 1
    assert server.post_json('/v1/audio/speech', case)[0] == 400
    for text in ('一つ目です。二つ目です。三つ目です。', '一つ目です、二つ目です、三つ目です'):
        case = copy.deepcopy(payload); case['input'] = text
        status, _, full = server.post_json('/v1/audio/speech', case)
        assert status == 200, full
        for seconds in (0.1, 1.1, 2.3):
            case['irodori']['max_seconds'] = seconds
            status, h, cut = server.post_json('/v1/audio/speech', case)
            assert status == 200, cut
            n, rate = wav_samples(cut)
            assert cut[44:] == full[44:44+n*2], 'a cap changed preceding audio or sentence order'
            assert n == min(int(seconds * rate), wav_samples(full)[0])
            if n == int(seconds * rate): assert h['x-speech-stop'] == 'max_seconds'
    for text in ('冒険の準備を進めながら仲間と話しています' * 25, '😊日本語の音声を確認します' * 25):
        case = copy.deepcopy(payload); case['input'] = text[:500]; case['irodori']['max_seconds'] = 0.15
        status, h, audio = server.post_json('/v1/audio/speech', case)
        assert status == 200 and h['x-speech-stop'] == 'max_seconds', audio
        n, rate = wav_samples(audio); assert n <= int(rate * 0.15)
    case = copy.deepcopy(payload)
    case['irodori'].update(caption='😊' * 1000, max_seconds=0.1)
    status, h, audio = server.post_json('/v1/audio/speech', case)
    assert status == 200 and h['x-speech-instructions-truncated'] == 'true', audio
    print('speech: seed boundaries, reproducibility, inactive guidance, Unicode splitting, caption truncation and limits pass', flush=True)
    assert upload(server, 'broken', b'not a wav')[0] == 400
    assert not (directory / 'broken.wav').exists()
    long_file = io.BytesIO()
    with wave.open(long_file, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
        tone = b''.join(struct.pack('<h', int(12000 * math.sin(2 * math.pi * 440 * i / 8000))) for i in range(8000))
        w.writeframes(tone * (int(metadata['irodori-tts.reference.max_seconds']) + 1))
    assert upload(server, 'too-long', long_file.getvalue())[0] == 400
    assert not (directory / 'too-long.wav').exists()
    directory.chmod(0o555)
    try:
        assert upload(server, 'unwritable', wav)[0] == 500
        assert not (directory / 'unwritable.wav').exists()
    finally:
        directory.chmod(0o755)
    assert {v['id'] for v in json.loads(server.call('GET', '/v1/audio/voices')[2])['data']} == {'none'}
    assert upload(server, '../escape', wav)[0] == 400
    assert upload(server, 'none', wav)[0] == 400
    assert upload(server, 'duplicate-field', wav, duplicate=True)[0] == 400
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: upload(server, 'reference', wav)[0], range(2)))
    assert sorted(results) == [201, 409], results
    assert time.monotonic() - started < 30
    assert (directory / 'reference.wav').read_bytes() == wav
    assert upload(server, 'page-reference', wav, page=True)[0] == 200
    case = copy.deepcopy(payload); case['voice'] = 'reference'
    status, _, before = server.post_json('/v1/audio/speech', case)
    assert status == 200, before
    server.stop()
    server = Server(speech, args)
    ids = {v['id'] for v in json.loads(server.call('GET', '/v1/audio/voices')[2])['data']}
    assert ids == {'none', 'reference', 'page-reference'}, ids
    status, _, after = server.post_json('/v1/audio/speech', case)
    assert status == 200 and after == before
    server.stop()
    conflict = subprocess.run([speech, 'serve', *args, '--add-voice', f'reference={directory / "reference.wav"}', '--port', '0'],
                              capture_output=True, timeout=120)
    assert conflict.returncode != 0 and b'already' in conflict.stderr, conflict.stderr
    (directory / 'broken.wav').write_bytes(b'bad')
    failed = subprocess.run([speech, 'serve', *args, '--port', '0'], capture_output=True, timeout=120)
    assert failed.returncode != 0 and b'broken.wav' in failed.stderr, failed.stderr
    print('voices: validation, concurrent duplicate, page upload, restart, identical restored voice and corrupt-store failure pass', flush=True)
print('Irodori bridge compatibility: OK', flush=True)
