"""The pinned checkpoint every script in this folder reads, each large file checked against its size and SHA-256, and
the commit of the code pyproject.toml pins."""

import hashlib
import importlib.metadata
import json
import os

from huggingface_hub import snapshot_download

_installed = json.loads(importlib.metadata.distribution("fish-speech").read_text("direct_url.json"))
CODE = {"repository": _installed["url"], "commit": _installed["vcs_info"]["commit_id"]}
MODEL = {
    "repository": "fishaudio/drama-3",
    "revision": "e0a9218aa48888270f06ef1ed9ea67622c0e17d2",
    "files": {
        "codec.pth": (1871099728, "74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f"),
        "model-00001-of-00008.safetensors": (5000202800, "e486b4f438d0b2df28846e41e599bd2077a64a825893ed0371b332a40dc4b49b"),
        "model-00002-of-00008.safetensors": (5000496824, "a67b726fe950ad57ebc15696009bde4a7dbbdc3645b42fa7d77e3a5fa1d70262"),
        "model-00003-of-00008.safetensors": (5000988520, "acfdeb67b8c4cb4784d887339e85bddbc5d7b9be3bbee82e40fa10ffe305d965"),
        "model-00004-of-00008.safetensors": (5000496392, "fa2960bddf128f51c3d4d74a1a942e34f61cc7ea294d10aec27f79dc10455c9f"),
        "model-00005-of-00008.safetensors": (5000227096, "a49ed8717c47e12b354a711c71ecde8be153bad704ca0a25d4805534f4405971"),
        "model-00006-of-00008.safetensors": (4984792768, "8867c0e3d600ebf28cf4734fc433dc98061fb990940e07b192ec0356ba1ff223"),
        "model-00007-of-00008.safetensors": (4977946704, "51e7d473f8a270bf79b2896a6ea4f19162eeec5b1ea1c5a2ba7dc7750511dbda"),
        "model-00008-of-00008.safetensors": (4638423240, "c5b065cce3b3b851369dfdb5a3b60f6abf2699121071c2b1a9dfd84acb7b8f7d"),
        "tokenizer.json": (20791070, "0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6"),
    },
}


def check(path, size, sha256):
    """Fails unless the file at `path` has the pinned size and SHA-256."""
    actual = os.path.getsize(path)
    if actual != size:
        raise SystemExit(f"{path} has {actual} bytes, not the pinned {size}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {sha256}")


def snapshot():
    """The local folder of the pinned checkpoint with its configuration, tokenizer, weights, codec, model card and license,
    downloaded and checked on first use; a folder already checked holds a file that says so, since hashing 41.5 GB takes
    minutes."""
    folder = snapshot_download(
        MODEL["repository"], revision=MODEL["revision"], allow_patterns=["*.json", "*.safetensors", "codec.pth", "README.md", "LICENSE"])
    stamp = os.path.join(folder, ".pins-checked")
    if not os.path.exists(stamp):
        for name, (size, sha256) in MODEL["files"].items():
            check(os.path.join(folder, name), size, sha256)
        with open(stamp, "w") as f:
            f.write(MODEL["revision"] + "\n")
    return folder


if __name__ == "__main__":
    print(snapshot())
