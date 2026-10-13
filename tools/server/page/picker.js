// The model picker at the top of a panel: a button that shows the model of the task in use, and a dialog that lists
// the task's models of the catalog with their size, whether they are fetched, the languages they take and those they
// are the one to start with. Picking one fetches it with its progress, which can be cancelled, and has the server load
// it in place of the task's model.

import * as api from './api.js';
import { languageList, languageNames, startPhrase } from './languages.js';

const TITLES = { synthesis: 'Models to speak with', recognition: 'Models to transcribe with', detection: 'Models to find speech with' };
const ROLES = { synthesis: 'Model to speak with', recognition: 'Model to transcribe with', detection: 'Model to find speech with' };
const LABELS = { synthesis: 'Speak with', recognition: 'Transcribe with', detection: 'Find speech with' };
/** What the button says while the task has no model: a panel can work without a detection model. */
const NONE = { synthesis: 'Choose a model', recognition: 'Choose a model', detection: 'None' };

function gigabytes(bytes) {
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

function megabytes(bytes) {
  return `${Math.round(bytes / 1e6)} MB`;
}

/** The catalog's NAME[:TYPE] of a file of a model. */
function fileName(model, file) {
  return file.type === model.type ? model.name : `${model.name}:${file.type}`;
}

/** The end of a path, as much as a picker's footer has room for, with the whole path in its title. */
function shortPath(path) {
  const parts = path.split(/[\\/]/).filter(Boolean);
  return parts.length > 3 ? `…/${parts.slice(-3).join('/')}` : path;
}

export class ModelPicker {
  #task;
  #state = null;
  #loading = null;

  /**
   * Builds the picker of `task` from the page's picker template as `element`, which the slot of the panel in view takes;
   * `loaded(held)` is called when the page has loaded a model in place of the task's, and `changed()` when what is
   * fetched may have changed.
   */
  constructor(task, loaded, changed) {
    this.#task = task;
    this.loaded = loaded;
    this.changed = changed;
    this.element = document.querySelector('#picker').content.firstElementChild.cloneNode(true);
    const $ = (selector) => this.element.querySelector(selector);
    this.button = $('.picker-button');
    this.name = $('.picker-name');
    this.size = $('.picker-size');
    this.progressRow = $('.picker-progress');
    this.progress = $('.picker-progress progress');
    this.progressText = $('.picker-progress-text');
    this.cancel = $('.picker-cancel');
    this.error = $('.picker > .error');
    this.dialog = $('dialog');
    this.list = $('.model-list');
    this.foot = $('.dialog-foot');
    $('.picker-role').textContent = LABELS[task];
    $('dialog h3').textContent = TITLES[task];
    $('dialog h3').id = `${task}-models-heading`;
    this.dialog.setAttribute('aria-labelledby', `${task}-models-heading`);

    this.button.addEventListener('click', () => this.dialog.showModal());
    $('.dialog-close').addEventListener('click', () => this.dialog.close());
    // A click on the backdrop lands on the dialog itself, outside its content.
    this.dialog.addEventListener('click', (event) => event.target === this.dialog && this.dialog.close());
  }

  /** Whether this page is fetching or loading a model of the task. */
  get busy() {
    return this.#loading !== null;
  }

  /** Shows the state that GET /speech/models gave. */
  show(state) {
    this.#state = state;
    const { held, replacing } = state[this.#task];
    if (held) {
      this.name.textContent = held.name ?? held.path.split(/[\\/]/).pop();
      this.size.textContent = held.name ? gigabytes(held.model.file_bytes) : 'given on the command line';
    } else {
      this.name.textContent = NONE[this.#task];
      this.size.textContent = '';
    }
    this.button.classList.toggle('empty', !held);
    this.button.setAttribute('aria-label', `${ROLES[this.#task]}: ${held ? `${this.name.textContent}, ${this.size.textContent}` : 'none'}. Choose one`);
    if (!this.busy && replacing) this.#showProgress(`Loading ${replacing}`, null, false);
    else if (!this.busy) this.progressRow.hidden = true;
    this.#fillList();
  }

  #fillList() {
    const { catalog } = this.#state;
    const { held, replacing } = this.#state[this.#task];
    this.list.replaceChildren();
    for (const model of catalog.models) {
      if (model.task !== this.#task) continue;
      for (const file of model.files) this.list.append(this.#item(model, file, held, replacing));
    }
    this.foot.textContent = `Fetched into ${shortPath(catalog.directory)}`;
    this.foot.title = catalog.directory;
  }

  #item(model, file, held, replacing) {
    const item = document.querySelector('#model-item').content.firstElementChild.cloneNode(true);
    const $ = (selector) => item.querySelector(selector);
    const name = fileName(model, file);
    const inUse = held?.name === name;
    $('.model-name').textContent = name;
    $('.in-use').hidden = !inUse;
    const fetched = file.fetched ? 'fetched' : file.partial ? `${megabytes(file.partial)} of it fetched` : 'not fetched yet';
    $('.model-meta').textContent = `${gigabytes(file.size)}, ${fetched}`;
    const languages = $('.model-languages');
    // The file a name alone fetches is the one to start with, not the model's other types.
    const start = file.type === model.type ? model.start : [];
    if (start.length) {
      const badge = document.createElement('p');
      badge.className = 'badge start';
      badge.textContent = `Start here for ${startPhrase(start)}`;
      badge.title = languageNames(start).join(', ');
      languages.append(badge);
    }
    // A model that is the one to start with for every language it takes needs its badge alone.
    if (start.length !== model.languages.length) languages.append(languageList(model.languages));
    item.dataset.state = inUse ? 'held' : file.fetched ? 'fetched' : 'remote';

    const use = $('.model-use');
    if (inUse) {
      use.remove();
      return item;
    }
    use.textContent = file.fetched ? 'Use' : file.partial ? 'Resume and use' : 'Fetch and use';
    use.setAttribute('aria-label', `${use.textContent} ${name}`);
    if (!file.fetched) use.title = `Fetches ${gigabytes(file.size - file.partial)} from Hugging Face`;
    use.disabled = this.busy || replacing !== null;
    use.addEventListener('click', () => {
      this.dialog.close();
      this.#use(name);
    });
    return item;
  }

  #showProgress(text, value, cancellable) {
    this.progressRow.hidden = false;
    this.progressText.textContent = text;
    if (value === null) this.progress.removeAttribute('value');
    else this.progress.value = value;
    this.cancel.hidden = !cancellable;
  }

  async #use(name) {
    const controller = new AbortController();
    this.#loading = controller;
    this.error.hidden = true;
    this.button.disabled = true;
    this.cancel.onclick = () => controller.abort();
    this.#showProgress(`Starting ${name}`, null, true);
    try {
      for await (const event of api.load(name, controller.signal)) {
        if (event.type === 'fetch') {
          this.#showProgress(`Fetching ${name}: ${megabytes(event.done)} of ${megabytes(event.total)}`, event.done / event.total, true);
        } else if (event.type === 'note') {
          this.#showProgress(event.message, null, true);
        } else if (event.type === 'load') {
          this.#showProgress(`Loading ${name}`, null, false);
        } else if (event.type === 'loaded') {
          this.loaded(event.held);
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        this.error.textContent = e.message;
        this.error.hidden = false;
      }
    } finally {
      this.#loading = null;
      this.button.disabled = false;
      this.progressRow.hidden = true;
    }
    this.changed();
  }
}
