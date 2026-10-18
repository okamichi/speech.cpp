// Drives the page of speech serve in a headless Chrome as a person does, and fails on a defect: the tabs, with the model
// pickers moving to the panel in view; the transcribe panel without a detection model, which transcribes audio up to a
// minute whole, as the API does, and asks for one for longer audio; the detection model chosen in its picker and loaded
// from the catalog; longer audio then transcribed by its regions, giving the text chunking_strategy "auto" gives; and the
// live panel, which waits for a detection model, transcribing a microphone that plays the dumps with pauses between
// them over /v1/realtime with the pause set in its options: grey text while an utterance is said, the microphone's audio
// sent whole, in order and alone, and at the end the text the API gives for the audio sent, with the same pause. The
// text is not compared with the API's for the file the microphone plays: the page sends it resampled to 24 kHz and some
// milliseconds late, which moves Silero VAD's chunks of 32 ms over the speech, and the regions and the text with them;
// the file delayed by 4 to 28 ms gave reazonspeech-nemo-v2 F16 texts 8.7 to 17.4% CER from the undelayed one's (Apple
// M5, Metal, 2026-10-12). The audio is
// 16-bit WAVE made of the dumps of reference/fastconformer/dump.py or reference/qwen3-asr/dump.py. Chrome is the one at
// $CHROME, or macOS's; the server and Chrome are stopped however the script ends.
// usage: node checks/smoke/server_page_browser_smoke.mjs <speech> <work dir> <recognition model> <detection model's catalog name> <dump folder>...
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';

const [speech, work, recognition, detection, ...dumps] = process.argv.slice(2);
if (!dumps.length) {
  console.error('usage: node checks/smoke/server_page_browser_smoke.mjs <speech> <work dir> <recognition model> <detection model\'s catalog name> <dump folder>...');
  process.exit(2);
}
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

/** A child process, and a promise that rejects when it cannot start or exits before it is stopped. */
function start(command, args, options) {
  const child = spawn(command, args, options);
  let stopping = false;
  const failed = new Promise((_, reject) => {
    child.on('error', (e) => reject(new Error(`${command} could not start: ${e.message}`)));
    child.on('exit', (code) => stopping || reject(new Error(`${command} exited with ${code}`)));
  });
  failed.catch(() => {});
  return { child, failed, stop: () => ((stopping = true), child.kill()) };
}

/** `promise`, or the first failure of `processes`, or a timeout. */
function racing(promise, processes, ms, what) {
  let timer;
  const timeout = new Promise((_, reject) => (timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms)));
  return Promise.race([promise, timeout, ...processes.map((p) => p.failed)]).finally(() => clearTimeout(timer));
}

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  await new Promise((resolve) => server.close(resolve));
  return port;
}

/** The float32 samples of a dump's audio.npy. */
function readNpy(file) {
  const data = fs.readFileSync(file);
  const headerLength = data.readUInt16LE(8);
  const header = data.subarray(10, 10 + headerLength).toString();
  if (!header.includes("'<f4'")) throw new Error(`${file} is not float32`);
  const body = data.subarray(10 + headerLength);
  return new Float32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
}

/**
 * Where `sent` holds `played`, both at `rate`: the lag of the sent audio's start, from 0 to `most` samples, found for the
 * half second around the loudest sample played, and the largest difference, in steps of 16-bit PCM, between a sample of
 * `sent` and the sample of `played` at that lag, or silence where `played` has none, and between a sample of `played`
 * past the end of `sent` and silence. A piece dropped, repeated or held back moves what follows off the lag, and audio
 * added anywhere, before or after `played` too, differs from the silence there.
 */
function alignment(sent, played, rate, most) {
  const loudest = played.reduce((at, x, i) => (Math.abs(x) > Math.abs(played[at]) ? i : at), 0);
  const from = Math.max(0, Math.min(played.length - rate / 2, loudest - rate / 4));
  const block = played.subarray(from, from + rate / 2);
  let lag = 0, best = -Infinity;
  for (let k = 0; k <= most; k++) {
    let dot = 0, norm = 0;
    for (let i = 0; i < block.length; i++) {
      const s = sent[from + k + i] ?? 0;
      dot += s * block[i];
      norm += s * s;
    }
    const score = dot / Math.sqrt(norm || 1);
    if (score > best) [lag, best] = [k, score];
  }
  let largest = 0, at = 0;
  for (let i = 0; i < Math.max(sent.length, lag + played.length); i++) {
    const differs = Math.abs((sent[i] ?? 0) - (played[i - lag] ?? 0));
    if (differs > largest) [largest, at] = [differs, i];
  }
  return { lag, largest: 32768 * largest, at };
}

/** A WAVE file of one channel of 16-bit samples, as the page writes one. */
function wav(samples, rate) {
  const out = Buffer.alloc(44 + 2 * samples.length);
  out.write('RIFF', 0);
  out.writeUInt32LE(36 + 2 * samples.length, 4);
  out.write('WAVEfmt ', 8);
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(1, 22);
  out.writeUInt32LE(rate, 24);
  out.writeUInt32LE(2 * rate, 28);
  out.writeUInt16LE(2, 32);
  out.writeUInt16LE(16, 34);
  out.write('data', 36);
  out.writeUInt32LE(2 * samples.length, 40);
  samples.forEach((x, i) => out.writeInt16LE(Math.max(-32768, Math.min(32767, Math.round(x * 32768))), 44 + 2 * i));
  return out;
}

const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'page-smoke-chrome-'));
const running = [];
let ws = null;
try {
  // Audio: the first dump alone, under a minute, and every dump with a second of silence after each, repeated past one.
  const rate = 16000;
  const first = readNpy(path.join(dumps[0], 'audio.npy'));
  if (first.length > 50 * rate) throw new Error(`${dumps[0]} is longer than 50 s; give a shorter dump first`);
  const once = [];
  for (const d of dumps) once.push(readNpy(path.join(d, 'audio.npy')), new Float32Array(rate));
  const spoken = new Float32Array(once.reduce((n, p) => n + p.length, 0));
  once.reduce((at, p) => (spoken.set(p, at), at + p.length), 0);
  const parts = [];
  for (let n = 0; n < 75 * rate; ) {
    for (const d of dumps) {
      const samples = readNpy(path.join(d, 'audio.npy'));
      parts.push(samples, new Float32Array(rate));
      n += samples.length + rate;
    }
  }
  const joined = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
  parts.reduce((at, p) => (joined.set(p, at), at + p.length), 0);
  fs.mkdirSync(work, { recursive: true });
  const short = path.join(work, 'page-smoke-short.wav');
  const long = path.join(work, 'page-smoke-long.wav');
  fs.writeFileSync(short, wav(first, rate));
  fs.writeFileSync(long, wav(joined, rate));
  const microphone = path.join(work, 'page-smoke-microphone.wav');
  fs.writeFileSync(microphone, wav(spoken, rate));

  const server = start(speech, ['serve', recognition, '--port', String(await freePort())], { stdio: ['ignore', 'pipe', 'pipe'] });
  running.push(server);
  let log = '';
  const address = await racing(new Promise((resolve) => {
    const read = (chunk) => {
      log += chunk;
      const m = /Ready at (http:\/\/127\.0\.0\.1:\d+\/)/.exec(log);
      if (m) resolve(m[1]);
    };
    server.child.stdout.on('data', read);
    server.child.stderr.on('data', read);
  }), [server], 120000, 'speech serve to listen').catch((e) => {
    throw new Error(`${e.message}\n${log}`);
  });
  const origin = new URL(address).origin;

  /** The text of the API for `file`, with the form's other fields. */
  const api = async (file, fields = {}) => {
    const form = new FormData();
    form.append('file', new Blob([fs.readFileSync(file)], { type: 'audio/wav' }), 'x.wav');
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    const response = await fetch(`${origin}/v1/audio/transcriptions`, { method: 'POST', body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(`the API answered ${response.status}: ${JSON.stringify(body)}`);
    return body.text;
  };

  const chrome = start(CHROME, [
    '--headless=new', '--remote-debugging-port=0', `--user-data-dir=${profile}`, '--no-first-run',
    '--hide-scrollbars', '--lang=en-US', '--accept-lang=en-US', '--window-size=960,900',
    // A click the script makes is no gesture of a person's, which an audio context needs to run.
    '--autoplay-policy=no-user-gesture-required',
  ], { stdio: 'ignore' });
  running.push(chrome);
  const port = await racing((async () => {
    for (;;) {
      try {
        return fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0];
      } catch {
        // Chrome writes the file once it listens.
        await wait(100);
      }
    }
  })(), [chrome], 30000, 'Chrome to listen');
  const targets = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await racing(new Promise((resolve) => ws.addEventListener('open', resolve)), [chrome], 30000, 'Chrome\'s page');
  let id = 0;
  const pending = new Map();
  const errors = [];
  ws.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      pending.get(message.id)(message);
      pending.delete(message.id);
    } else if (message.method === 'Runtime.exceptionThrown') {
      errors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
    }
  });
  const send = (method, params = {}) => racing(new Promise((resolve) => {
    const n = ++id;
    pending.set(n, resolve);
    ws.send(JSON.stringify({ id: n, method, params }));
  }), running, 300000, method);
  /** Runs `body`, an async function's body, in the page with $ and until(check, what) defined. */
  const page = async (body) => {
    const expression = `(async () => {
      const $ = (id) => document.getElementById(id);
      const until = async (check, what, ms = 240000) => {
        const t = Date.now();
        while (!check()) {
          if (Date.now() - t > ms) throw new Error('timed out waiting for ' + what);
          await new Promise((r) => setTimeout(r, 100));
        }
      };
      ${body}
    })()`;
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description ?? JSON.stringify(r.result.exceptionDetails));
    return r.result?.result?.value;
  };
  /** Gives `file` to the transcribe panel's file input, as a person choosing it does. */
  const choose = async (file) => {
    const input = await send('Runtime.evaluate', { expression: "document.querySelector('#transcribe-audio .audio-file')" });
    await send('DOM.setFileInputFiles', { files: [path.resolve(file)], objectId: input.result.result.objectId });
  };
  /**
   * Transcribes the audio given in the transcribe panel and returns the card's text, or that the panel refused it, with
   * the requests it sent.
   */
  const transcribe = () => page(`
    await until(() => !$('transcribe-start').disabled, 'the transcribe button');
    window.__transcriptions.length = 0;
    $('transcribe-start').click();
    const card = document.querySelector('#transcribe-transcript .transcript');
    await until(() => !$('transcribe-start').disabled && !$('transcribe-status').textContent, 'the transcription');
    const requests = [...window.__transcriptions];
    return $('transcribe-error').hidden ? { text: card.querySelector('.transcript-text').textContent, requests } : { refused: true, requests };
  `);

  await send('Runtime.enable');
  await send('Page.enable');
  // The microphone plays the dumps once, in real time, and then silence. It plays at the rate the page records, OpenAI's
  // rate of audio/pcm, so that the page receives the samples decodeAudioData gives at that rate, a whole number of
  // samples late. Through a context at the browser's own rate, the delay had a fraction of a sample in one run of four,
  // which left 17 steps of 16-bit PCM differing in the speech of reazonspeech-nemo-v2's dumps. It plays a second of
  // silence first: the page connects its recorder only once getUserMedia has returned and the recorder's worklet has
  // loaded, and the samples played before that are not recorded, which were the first 240 of a dump that speaks at once.
  // The check of the audio sent finds the speech lost if the recorder ever takes longer.
  const pcmRate = 24000;
  const lead = 1;
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__microphone = ${JSON.stringify(fs.readFileSync(microphone).toString('base64'))};
    navigator.mediaDevices.getUserMedia = async () => {
      const bytes = Uint8Array.from(atob(window.__microphone), (c) => c.charCodeAt(0));
      const context = new AudioContext({ sampleRate: ${pcmRate} });
      const source = context.createBufferSource();
      source.buffer = await context.decodeAudioData(bytes.buffer);
      const out = context.createMediaStreamDestination();
      source.connect(out);
      source.start(context.currentTime + ${lead});
      return out.stream;
    };
  ` });
  // The session.updates the page sends and the audio it appends, and the page's transcription requests, each with the
  // fields of its form other than the file.
  await send('Page.addScriptToEvaluateOnNewDocument', { source: `
    window.__sessions = [];
    window.__appended = [];
    const sent = WebSocket.prototype.send;
    WebSocket.prototype.send = function (data) {
      const event = JSON.parse(data);
      if (event.type === 'session.update') window.__sessions.push(event.session);
      if (event.type === 'input_audio_buffer.append') window.__appended.push(event.audio);
      return sent.call(this, data);
    };
    window.__transcriptions = [];
    const fetched = window.fetch;
    window.fetch = (url, init) => {
      if (String(url).endsWith('/v1/audio/transcriptions') && init?.body instanceof FormData) {
        window.__transcriptions.push(Object.fromEntries([...init.body].filter(([name]) => name !== 'file')));
      }
      return fetched(url, init);
    };
  ` });
  const pageLoaded = new Promise((resolve) => ws.addEventListener('message', (event) => {
    if (JSON.parse(event.data).method === 'Page.loadEventFired') resolve();
  }));
  await send('Page.navigate', { url: address.replace('127.0.0.1', 'localhost') });
  // Page.navigate answers once the new document is committed, which can be before the page's modules have run and
  // given the tabs their clicks; a click then is lost.
  await racing(pageLoaded, running, 60000, 'the page to load');
  // A picker sits only in the panel in view.
  await page(`
    $('tab-transcribe').click();
    await until(() => document.querySelector('#transcribe .picker-slot[data-task="recognition"] .picker-button:not(.empty)'), 'the recognition model');
  `);

  // The tabs: each picker sits in the panel in view, and the transcribe panel has the detection model's, empty.
  const tabs = await page(`
    const where = (task) => document.querySelector('.picker-slot[data-task="' + task + '"] .picker')?.closest('.panel')?.id ?? null;
    const seen = {};
    for (const tab of ['speak', 'transcribe', 'live']) {
      $('tab-' + tab).click();
      seen[tab] = { recognition: where('recognition'), detection: where('detection'), synthesis: where('synthesis'), hidden: $(tab).hidden };
    }
    $('tab-transcribe').click();
    const picker = document.querySelector('#transcribe .picker-slot[data-task="detection"] .picker');
    return { seen, empty: picker.querySelector('.picker-button').classList.contains('empty'), liveStarts: !$('live-start').disabled };
  `);
  if (tabs.seen.transcribe.recognition !== 'transcribe' || tabs.seen.live.recognition !== 'live' || tabs.seen.transcribe.detection !== 'transcribe'
      || tabs.seen.speak.synthesis !== 'speak' || Object.values(tabs.seen).some((s) => s.hidden)) {
    throw new Error(`the pickers did not follow the tabs: ${JSON.stringify(tabs.seen)}`);
  }
  if (!tabs.empty) throw new Error('the detection picker shows a model the server does not hold');
  if (tabs.liveStarts) throw new Error('the live panel starts without a detection model');
  console.log('tabs: the recognition picker in the transcribe and live panels, the detection picker in the transcribe panel, empty');

  // Without a detection model: audio under a minute whole, as the API transcribes it; longer audio refused.
  await choose(short);
  const whole = await transcribe();
  const wantWhole = await api(short);
  if (whole.text !== wantWhole || whole.requests.length !== 1 || 'chunking_strategy' in whole.requests[0]) {
    throw new Error(`the page without a detection model: ${JSON.stringify(whole)}, where the API gives ${JSON.stringify(wantWhole)} for one request without chunking_strategy`);
  }
  console.log(`transcribe without a detection model: ${(first.length / rate).toFixed(1)} s whole, the API's text: ${wantWhole.slice(0, 40)}`);
  await choose(long);
  const refused = await transcribe();
  if (!refused.refused || refused.requests.length) throw new Error(`audio over a minute without a detection model: ${JSON.stringify(refused)}`);
  console.log(`transcribe without a detection model: ${(joined.length / rate).toFixed(1)} s refused, asking for one`);

  // The detection model chosen in its picker, fetched already, and loaded in its place.
  const loaded = await page(`
    const picker = document.querySelector('#transcribe .picker-slot[data-task="detection"] .picker');
    picker.querySelector('.picker-button').click();
    const use = [...picker.querySelectorAll('.model')].find((m) => m.querySelector('.model-name').textContent === ${JSON.stringify(detection)})?.querySelector('.model-use');
    if (!use) throw new Error('the detection picker does not list ${detection}');
    use.click();
    await until(() => picker.querySelector('.picker-name').textContent === ${JSON.stringify(detection)} && picker.querySelector('.picker-progress').hidden, 'the detection model');
    return !picker.querySelector('.picker-button').classList.contains('empty');
  `);
  if (!loaded) throw new Error('the detection picker shows no model after loading one');
  console.log(`the detection picker loaded ${detection}`);

  // With it, the longer audio by its regions, as chunking_strategy "auto" gives it.
  const regions = await transcribe();
  const wantRegions = await api(long, { chunking_strategy: 'auto' });
  if (regions.text !== wantRegions || !regions.requests.length || regions.requests.some((r) => r.chunking_strategy !== 'auto')) {
    throw new Error(`the page by regions: ${JSON.stringify(regions)}, where the API gives ${JSON.stringify(wantRegions)} with chunking_strategy "auto"`);
  }
  console.log(`transcribe with ${detection}: ${(joined.length / rate).toFixed(1)} s by its regions, the text of chunking_strategy "auto": ${wantRegions.slice(0, 40)}`);
  // The live panel: the microphone transcribed as it plays, grey text while an utterance is said, then the whole text.
  const live = await page(`
    $('tab-live').click();
    await until(() => !$('live-start').disabled, 'the live panel to start');
    $('live-silence').value = '400';
    $('live-silence').dispatchEvent(new Event('change'));
    $('live-start').click();
    const card = document.querySelector('#live-transcript .transcript');
    let grey = 0;
    const end = Date.now() + ${Math.ceil((lead + spoken.length / rate) * 1000) + 2500};
    while (Date.now() < end) {
      if (!$('live-error').hidden) return { error: $('live-error').textContent };
      if (card.querySelector('.provisional')?.textContent) grey++;
      await new Promise((r) => setTimeout(r, 100));
    }
    $('live-start').click();
    await until(() => !$('live-start').disabled && !$('live-status').textContent, 'the last words');
    if (!$('live-error').hidden) return { error: $('live-error').textContent };
    return {
      text: card.querySelector('.transcript-text').textContent, grey, provisional: card.querySelector('.provisional')?.textContent ?? '',
      // Stop's session.update changes nothing and carries no audio.
      turns: window.__sessions.filter((s) => s.audio).map((s) => s.audio.input.turn_detection),
    };
  `);
  // The audio the page appended, and the microphone's as the browser decodes it at the same rate, in base64.
  const heard = await page(`
    const played = await new OfflineAudioContext(1, 1, ${pcmRate}).decodeAudioData(Uint8Array.from(atob(window.__microphone), (c) => c.charCodeAt(0)).buffer);
    const bytes = new Uint8Array(played.getChannelData(0).slice().buffer);
    let text = '';
    for (let at = 0; at < bytes.length; at += 0x8000) text += String.fromCharCode(...bytes.subarray(at, at + 0x8000));
    return { appended: window.__appended, played: btoa(text) };
  `);
  const pcm = Buffer.concat(heard.appended.map((a) => Buffer.from(a, 'base64')));
  const sent = Float32Array.from({ length: pcm.length / 2 }, (_, i) => pcm.readInt16LE(2 * i) / 32768);
  // The page sends each sample clamped to [-1, 1], times 32767 and rounded, which the server reads over 32768; Chrome's
  // resampling takes a loud dump past 1.
  const played = Float32Array.from(new Float32Array(Uint8Array.from(Buffer.from(heard.played, 'base64')).buffer),
                                   (x) => Math.round(32767 * Math.max(-1, Math.min(1, x))) / 32768);
  // The recorder starts within the microphone's second of silence.
  const { lag, largest, at } = alignment(sent, played, pcmRate, 2 * lead * pcmRate);
  // The audio sent was the microphone's bit for bit (Chrome 155, 2026-10-12), where a dump's speech or 128 samples dropped
  // left samples hundreds of steps off; a sample computed otherwise before its rounding would be a step off.
  if (largest > 1) {
    throw new Error(`the live panel sent ${(sent.length / pcmRate).toFixed(1)} s that hold the microphone's ${(played.length / pcmRate).toFixed(1)} s `
                    + `from ${(lag / pcmRate).toFixed(3)} s on with a sample ${(at / pcmRate).toFixed(3)} s in differing by ${largest.toFixed(0)} `
                    + 'steps of 16-bit PCM');
  }
  const appended = path.join(work, 'page-smoke-appended.wav');
  fs.writeFileSync(appended, wav(sent, pcmRate));
  const wantLive = await api(appended, { 'chunking_strategy[type]': 'server_vad', 'chunking_strategy[silence_duration_ms]': '400' });
  if (live.error || !live.grey || live.provisional || live.text !== wantLive
      || JSON.stringify(live.turns) !== JSON.stringify([{ type: 'server_vad', silence_duration_ms: 400 }])) {
    throw new Error(`the live panel: ${JSON.stringify(live)}, where the API gives ${JSON.stringify(wantLive)} for the audio it sent`);
  }
  console.log(`live: ${(spoken.length / rate).toFixed(1)} s from the microphone, sent from ${(lag / pcmRate).toFixed(3)} s on with `
              + `the largest difference ${largest.toFixed(0)} steps of 16-bit PCM, grey text in ${live.grey} of its looks, the API's text for the audio sent: ${live.text.slice(0, 40)}`);
  if (errors.length) throw new Error(`the page threw: ${errors.join('; ')}`);
  console.log('ok');
} finally {
  ws?.close();
  for (const child of running.reverse()) child.stop();
  await wait(500);
  fs.rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
}
