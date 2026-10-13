// The transcribe panel: audio dropped, chosen, recorded or handed over from the speak panel, the options the
// transcription form takes, and the text as a card. With a detection model the server transcribes the regions where
// someone speaks, one at a time; without one it transcribes the audio whole, which a FastConformer model does well for
// an utterance and parakeet's memory allows for a few minutes at most, so longer audio asks for a detection model.

import * as api from './api.js';
import { AudioInput } from './audio-input.js';
import { OptionForm, withServerDefaults } from './options.js';
import { Transcript } from './pieces.js';
import { length, TranscriptView } from './transcript.js';
import { wavFile } from './wav.js';

const $ = (id) => document.getElementById(id);

/** The options of the model that OpenAI's transcription form carries; timestamps follow from verbose_json. */
export const FORM_OPTIONS = new Set(['language', 'prompt', 'decoding']);

/**
 * The most bytes of audio a request carries, below the server's 25 MB, OpenAI's limit, with room for the form: about
 * 12 minutes at 16 kHz. Longer audio goes in pieces cut at a pause.
 */
const MAX_BYTES = 24e6;
/** The longest audio transcribed whole, without a detection model. */
const MAX_WHOLE_SECONDS = 60;

export class TranscribePanel {
  #held = null;
  #detection = null;
  #options = null;
  #controller = null;
  #view = new TranscriptView($('transcribe-transcript'));
  #audio = new AudioInput($('transcribe-audio'), () => {
    $('transcribe-error').hidden = true;
    this.#ready();
  });

  constructor() {
    $('transcribe-form').addEventListener('submit', (event) => {
      event.preventDefault();
      this.#transcribe();
    });
    $('transcribe-stop').addEventListener('click', () => this.#controller?.abort());
  }

  /** Shows the recognition model held, `held` of GET /speech/models, or that there is none. */
  show(held) {
    this.#held = held;
    if (held) {
      const kept = this.#options?.values() ?? {};
      this.#options = new OptionForm(withServerDefaults(held), $('transcribe-options'), null, FORM_OPTIONS);
      this.#options.restore(kept);
    } else {
      $('transcribe-options').replaceChildren();
      this.#options = null;
    }
    this.#ready();
  }

  /** Shows the detection model held, `held` of GET /speech/models, or that there is none. */
  showDetection(held) {
    this.#detection = held;
    $('transcribe-detection-hint').textContent = held
      ? 'The audio is transcribed a region at a time, where someone speaks.'
      : `Without one, audio up to ${MAX_WHOLE_SECONDS} s is transcribed whole; longer audio needs one, such as silero-vad.`;
  }

  /** Takes audio from elsewhere on the page, the speak panel's speech. */
  use(file, label) {
    this.#audio.use(file, label);
  }

  /** The button transcribes once there is a model and audio, and says what is missing until then. */
  #ready() {
    const start = $('transcribe-start');
    start.disabled = !this.#held || !this.#audio.given || this.#controller !== null || this.#audio.recording;
    start.title = !this.#held ? 'Choose a model first' : !this.#audio.given ? 'Give the audio first' : '';
    $('transcribe-stop').hidden = this.#controller === null;
  }

  #status(text) {
    $('transcribe-status').textContent = text;
  }

  /** Shows a failure next to its cause: the audio, an option's field, or below the form. */
  #fail(e) {
    if (e.name === 'AbortError') {
      // Stopped by the user, who sees the button come back.
    } else if (e.param === 'file') {
      this.#audio.fail(e.message);
    } else if (!(e.param && this.#options?.showError(e.param, e.message))) {
      $('transcribe-error').textContent = e.message;
      $('transcribe-error').hidden = false;
    }
  }

  /**
   * Transcribes the audio given, by its regions where the server holds a detection model, in as many requests as its
   * size needs, showing the text as each comes back.
   */
  async #transcribe() {
    if (this.#controller) return;
    $('transcribe-error').hidden = true;
    this.#options.clearErrors();
    this.#controller = new AbortController();
    this.#ready();
    const model = this.#held.model;
    // A detection model loaded while the audio is read does not change how this transcription goes.
    const byRegions = this.#detection !== null;
    const fields = { ...this.#options.values(), ...(byRegions ? { chunking_strategy: 'auto' } : {}) };
    const transcript = new Transcript();
    const started = performance.now();
    try {
      this.#status('Preparing the audio');
      for await (const piece of this.#audio.pieces(model.sample_rate, MAX_BYTES)) {
        if (!byRegions && piece.total > MAX_WHOLE_SECONDS) {
          throw new Error(`This audio is ${length(piece.total)} long. Without a detection model, audio up to ${MAX_WHOLE_SECONDS} s is ` +
                          'transcribed whole; choose one, such as silero-vad, to transcribe longer audio by the regions where someone speaks.');
        }
        this.#status(piece.start ? `Transcribing, ${length(piece.start)} of ${length(piece.total)} done` : 'Transcribing');
        const wav = wavFile(piece.samples, piece.rate);
        const { result, stop } = await api.transcribe(wav, fields, this.#controller.signal);
        transcript.add(result, stop, piece.start);
        this.#view.render(transcript, model, { done: false });
      }
      this.#view.render(transcript, model, { took: (performance.now() - started) / 1000 });
    } catch (e) {
      this.#fail(e);
    } finally {
      this.#status('');
      this.#controller = null;
      this.#ready();
    }
  }
}
