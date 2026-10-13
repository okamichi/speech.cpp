// The live panel: transcribes the microphone while the user speaks, over the server's Realtime transcription. The text
// of an utterance under way is shown in the muted colour as it grows, and replaced by its final text when it ends.

import { Recorder } from './audio.js';
import { RATE, RealtimeTranscription } from './live-realtime.js';
import { OptionForm, withServerDefaults } from './options.js';
import { length, TranscriptView } from './transcript.js';

const $ = (id) => document.getElementById(id);

/** How often the samples recorded go to the server and the time listened shows. */
const TICK_MS = 100;
/** The options of the model that a Realtime transcription session carries. */
const SESSION_OPTIONS = new Set(['language', 'prompt']);
const SETTINGS_KEY = 'speech.cpp live settings';

/** The value of a number field, within its bounds, or its default when it holds no number. */
function bounded(input) {
  const value = Number(input.value);
  return Number.isFinite(value) ? Math.min(Number(input.max), Math.max(Number(input.min), value)) : Number(input.defaultValue);
}

export class LivePanel {
  #held = null;
  #detection = null;
  #options = null;
  #recorder = null;
  /** The recorder of a run whose microphone the browser has not yet given; a change of model ends it. */
  #starting = null;
  #live = null;
  #ticking = 0;
  #busy;
  #view = new TranscriptView($('live-transcript'));
  /** The pause the page gives a detection model the server was not started with a pause for: OpenAI's server_vad default. */
  #silence = $('live-silence').defaultValue;
  /** The range of pauses the field offers, which widens to take the one the server was started with. */
  #bounds = [Number($('live-silence').min), Number($('live-silence').max)];

  /** `busy(running)` follows the start and the end of each run, which the panel's tab shows. */
  constructor(busy) {
    this.#busy = busy;
    $('live-form').addEventListener('submit', (event) => {
      event.preventDefault();
      if (this.#recorder) this.#stop();
      else this.#start();
    });
    try {
      const kept = JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}');
      if (Number.isFinite(kept.silence)) $('live-silence').value = kept.silence;
    } catch {
      // Storage the browser refuses, or a value of another shape, leaves the setting as the page gives it.
    }
    $('live-silence').addEventListener('change', () => {
      try {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify({ silence: bounded($('live-silence')) }));
      } catch {
        // Without storage the setting lasts as long as the page.
      }
    });
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none; a run going on ends. */
  show(held) {
    this.#end();
    this.#held = held;
    if (held) {
      const kept = this.#options?.values() ?? {};
      this.#options = new OptionForm(withServerDefaults(held), $('live-options'), null, SESSION_OPTIONS);
      this.#options.restore(kept);
    } else {
      $('live-options').replaceChildren();
      this.#options = null;
    }
    this.#ready();
  }

  /**
   * Shows the detection model held, which finds where each utterance starts and ends; a run going on ends. The pause
   * starts at the one the server was started with, unless the user set one.
   */
  showDetection(held) {
    this.#end();
    this.#detection = held;
    const silence = $('live-silence');
    silence.defaultValue = held?.defaults?.min_silence_duration_ms ?? this.#silence;
    silence.min = Math.min(this.#bounds[0], silence.defaultValue);
    silence.max = Math.max(this.#bounds[1], silence.defaultValue);
    this.#ready();
  }

  /** The button starts once there are both models, stops while listening, and waits while the last words are transcribed. */
  #ready() {
    const button = $('live-start');
    button.textContent = this.#recorder ? 'Stop' : 'Start';
    button.classList.toggle('primary', !this.#recorder);
    button.classList.toggle('stop', this.#recorder !== null);
    button.disabled = !this.#held || !this.#detection || this.#starting !== null || (this.#live !== null && !this.#recorder);
    button.title = !this.#held ? 'Choose a model first' : !this.#detection ? 'Choose a detection model first' : '';
    $('live-dot').hidden = !this.#recorder;
  }

  #status(text) {
    $('live-status').textContent = text;
  }

  /** Shows a failure next to its cause: an option's field, or below the form. */
  #fail(e) {
    const field = e.param?.split('.').pop();
    if (!(field && this.#options?.showError(field, e.message))) {
      $('live-error').textContent = e.message;
      $('live-error').hidden = false;
    }
  }

  async #start() {
    if (this.#live || this.#starting || !this.#held || !this.#detection) return;
    $('live-error').hidden = true;
    this.#options.clearErrors();
    const model = this.#held.model;
    const live = new RealtimeTranscription(
      (transcript, provisional) => this.#view.render(transcript, model, { provisional, done: false }),
      (e) => {
        this.#end();
        this.#fail(e);
        this.#ready();
      },
    );
    const recorder = new Recorder(RATE);
    this.#starting = recorder;
    this.#ready();
    // Both are let to settle, so that a microphone the browser opens after the session failed is still turned off.
    const [microphone, session] = await Promise.allSettled([
      recorder.start(),
      live.open(model.name, this.#options.values(), bounded($('live-silence'))),
    ]);
    if (microphone.status === 'rejected' || session.status === 'rejected') {
      live.abort();
      await recorder.stop();
      if (this.#starting !== recorder) return;
      this.#starting = null;
      const e = microphone.status === 'rejected' ? microphone.reason : session.reason;
      this.#fail(e.name === 'NotAllowedError' ? new Error('The browser was not allowed to use the microphone.') : e);
      this.#ready();
      return;
    }
    if (this.#starting !== recorder) {
      live.abort();
      await recorder.stop();
      return;
    }
    this.#starting = null;
    this.#recorder = recorder;
    this.#live = live;
    this.#busy(true);
    this.#ready();
    this.#view.render({ text: '', segments: [], languages: [], seconds: 0, stop: 'complete' }, model, { done: false });
    let heard = 0;
    const tick = () => {
      const samples = recorder.take();
      heard += samples.length;
      live.push(samples);
      this.#status(`Listening, ${length(heard / RATE)}`);
    };
    tick();
    this.#ticking = setInterval(tick, TICK_MS);
  }

  /** Ends the recording and shows the whole text once the last words are transcribed. */
  async #stop() {
    const live = this.#live;
    const model = this.#held.model;
    live.push(await this.#release());
    this.#status('Transcribing the last words');
    this.#ready();
    try {
      this.#view.render(await live.finish(), model, { live: true });
    } catch (e) {
      this.#fail(e);
    } finally {
      if (this.#live === live) this.#live = null;
      this.#status('');
      this.#busy(false);
      this.#ready();
    }
  }

  /** Ends a run that is starting or going on, without its last words. */
  #end() {
    this.#starting = null;
    if (!this.#live) return;
    this.#live.abort();
    this.#live = null;
    if (this.#recorder) this.#release();
    this.#status('');
    this.#busy(false);
  }

  /** Turns the microphone off, and returns the samples recorded since the last tick. */
  async #release() {
    clearInterval(this.#ticking);
    const recorder = this.#recorder;
    this.#recorder = null;
    return recorder.stop();
  }
}
