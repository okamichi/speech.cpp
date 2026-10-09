"""Checks persistent voices and health while the page replaces an Irodori model.

usage: python3 checks/smoke/irodori_compat_lifecycle.py SPEECH INITIAL.gguf CATALOG_RF.gguf REFERENCE.wav [--device NAME]
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from server_client import Server

speech, initial, replacement, reference, *options = sys.argv[1:]
speech, initial, replacement = map(os.path.abspath, (speech, initial, replacement))
with tempfile.TemporaryDirectory(prefix='speech-voice-lifecycle-') as temp:
    folder = Path(temp)
    env = {**os.environ, 'SPEECH_MODEL_DIR': str(folder / 'models')}
    catalog = json.loads(subprocess.check_output([speech, 'models', '--json'], env=env))
    digest = hashlib.sha256()
    with open(replacement, 'rb') as f:
        while chunk := f.read(16 << 20): digest.update(chunk)
    matches = [(m, f) for m in catalog['models'] for f in m['files'] if f['sha256'] == digest.hexdigest()]
    assert len(matches) == 1, 'Give a local RF file listed in the catalog; this check downloads nothing.'
    entry, file = matches[0]
    target = Path(file['path'])
    target.parent.mkdir(parents=True)
    target.symlink_to(replacement)
    name = entry['name'] if entry['type'] == file['type'] else entry['name'] + ':' + file['type']
    server = Server(speech, [initial, '--no-warmup', '--irodori-compatible', '--voices-dir', str(folder / 'voices'), *options], env)
    boundary = 'speech-lifecycle-boundary'
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="voice_id"\r\n\r\npersisted\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="voice.wav"\r\n\r\n').encode()
    body += Path(reference).read_bytes() + f'\r\n--{boundary}--\r\n'.encode()
    assert server.call('POST', '/v1/audio/voices', body, {'Content-Type': f'multipart/form-data; boundary={boundary}'})[0] == 201
    payload = {'input': 'モデルを切り替える間も、この音声は最後まで生成します。', 'voice': 'persisted',
               'irodori': {'seed': 42, 'num_steps': 500}}
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        speech_job = pool.submit(server.post_json, '/v1/audio/speech', payload)
        time.sleep(0.15)
        loading = pool.submit(server.post_json, '/speech/load', {'model': name})
        saw_loading = False
        deadline = time.monotonic() + 120
        while not loading.done() and time.monotonic() < deadline:
            health = json.loads(server.call('GET', '/health')[2])
            if health['runtime']['loading'] and not health['runtime']['loaded']:
                assert health['runtime']['checkpoint'] is None and health['model'] is None
                status, headers, _ = server.call('GET', '/v1/audio/voices')
                assert status == 503 and headers.get('retry-after')
                saw_loading = True
                break
            time.sleep(0.02)
        assert saw_loading, 'No loading state observed while an old synthesis held the model.'
        assert speech_job.result(timeout=120)[0] == 200
        status, _, data = loading.result(timeout=120)
        events = [json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: ')]
        assert status == 200 and any(e['type'] == 'loaded' for e in events), events
    health = json.loads(server.call('GET', '/health')[2])
    assert health['runtime'] == {'loaded': True, 'loading': False, 'checkpoint': str(target)}, health
    ids = {v['id'] for v in json.loads(server.call('GET', '/v1/audio/voices')[2])['data']}
    assert ids == {'none', 'persisted'}, ids
    payload['irodori']['num_steps'] = 1
    assert server.post_json('/v1/audio/speech', payload)[0] == 200
    assert (folder / 'voices/persisted.wav').read_bytes() == Path(reference).read_bytes()
    server.stop()
print('voice restoration, in-flight synthesis, loading health and voice-list retry: OK')
