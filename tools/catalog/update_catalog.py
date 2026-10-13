"""Writes tools/catalog/catalog.json again, pinning each model at its Hugging Face repository's current revision.

What a person chose stays as catalog.json has it: each model's name, its repository, the type of the file its name
alone names, and the languages it is the model to start with. The rest is read from Hugging Face: the repository's
commit, each GGUF file at that commit with its size and SHA-256 from the Hub's API, and the type, task, languages and
whether the model takes voice files from the output of `speech info --json` that the repository holds beside each
file. A release lists the files it was checked with, so this runs whenever a repository's files change, and its diff
is reviewed like code. It refuses a catalog it would write wrong: a name a model argument cannot hold, a type the
repository has no file of, files of one model that disagree on its task or languages, a language to start with that
the model does not take or that two models of a task claim, and a file whose JSON gives another size.

To add a model, add its name, repository, type and start to catalog.json and run this.

usage: python3 tools/catalog/update_catalog.py
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request

CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.json")
CHOSEN = ["name", "repository", "type", "start"]
TASKS = ["synthesis", "recognition", "detection"]
# A model's files in the order `speech models` and the page list them: the type its name alone means, then the others
# from the widest.
WIDEST_FIRST = ["f32", "f16", "q8_0", "q6_k", "q5_k", "q4_k"]


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def pin(model):
    repository = model["repository"]
    api = fetch_json(f"https://huggingface.co/api/models/{repository}/revision/main?blobs=true")
    revision = api["sha"]
    files, about = [], []
    for sibling in sorted(api["siblings"], key=lambda s: s["rfilename"]):
        name = sibling["rfilename"]
        if not name.endswith(".gguf"):
            continue
        if "/" in name or not sibling.get("lfs"):
            raise SystemExit(f"{repository}: {name} is not a GGUF file at the top of the repository, stored in LFS")
        info = fetch_json(f"https://huggingface.co/{repository}/resolve/{revision}/{urllib.parse.quote(name)}.json")
        if info["file_bytes"] != sibling["lfs"]["size"]:
            raise SystemExit(f"{repository}: {name}.json gives {info['file_bytes']} bytes, the file has {sibling['lfs']['size']}")
        files.append({"type": info["weight_type"].lower(), "file": name, "size": sibling["lfs"]["size"], "sha256": sibling["lfs"]["sha256"]})
        about.append((info["task"], info["languages"], info.get("voice_files", False)))
    if not files:
        raise SystemExit(f"{repository} holds no GGUF file at {revision}")
    if len({a for a in map(json.dumps, about)}) != 1:
        raise SystemExit(f"{repository}: the files disagree on the task, the languages or voice files: {about}")
    types = [f["type"] for f in files]
    if len(set(types)) != len(types):
        raise SystemExit(f"{repository} holds two files of one type: {types}")
    if model["type"] not in types:
        raise SystemExit(f"{model['name']}: its type {model['type']} is not among the repository's, {types}")
    unknown = sorted(set(types) - set(WIDEST_FIRST))
    if unknown:
        raise SystemExit(f"{repository} holds a file of a type the catalog does not order: {unknown}")
    files.sort(key=lambda f: (f["type"] != model["type"], WIDEST_FIRST.index(f["type"])))
    task, languages, voice_files = about[0]
    if task not in TASKS:
        raise SystemExit(f"{repository}: a task {task!r}")
    if not set(model["start"]) <= set(languages):
        raise SystemExit(f"{model['name']} does not take {sorted(set(model['start']) - set(languages))}, which it is to start with")
    print(f"{model['name']}: {repository} at {revision}, {', '.join(f['file'] for f in files)}", file=sys.stderr)
    return {**{k: model[k] for k in CHOSEN}, "revision": revision, "task": task, "languages": languages, "voice_files": voice_files,
            "files": files}


def check(models):
    names = [m["name"] for m in models]
    for name in names:
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
            raise SystemExit(f"{name!r}: a name is lower-case letters, digits, '.', '_' and '-', since ':' starts the type")
    for values in (names, [m["repository"] for m in models]):
        if len(set(values)) != len(values):
            raise SystemExit(f"a name or repository appears twice: {values}")
    for task in TASKS:
        claimed = {}
        for m in models:
            for language in m["start"] if m["task"] == task else []:
                if language in claimed:
                    raise SystemExit(f"{language}: both {claimed[language]} and {m['name']} are the {task} model to start with")
                claimed[language] = m["name"]


def write(models):
    """catalog.json with one line per member, and each list and file on one line, so that a diff shows what changed."""
    def line(value):
        return json.dumps(value, ensure_ascii=False, separators=(", ", ": "))

    out = ['{\n  "models": [']
    for i, m in enumerate(models):
        members = [f'      "{k}": {line(m[k])}' for k in m if k != "files"]
        files = ",\n".join(f"        {line(f)}" for f in m["files"])
        members.append(f'      "files": [\n{files}\n      ]')
        out.append("    {\n" + ",\n".join(members) + "\n    }" + ("," if i + 1 < len(models) else ""))
    out.append("  ]\n}\n")
    with open(CATALOG, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


with open(CATALOG, encoding="utf-8") as f:
    models = [pin(m) for m in json.load(f)["models"]]
check(models)
write(models)
print(f"wrote {CATALOG}", file=sys.stderr)
