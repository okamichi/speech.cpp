"""The pinned weights every script in this folder runs, and the commit of the code pyproject.toml pins."""

import importlib.metadata
import json

from huggingface_hub import snapshot_download

_installed = json.loads(importlib.metadata.distribution("irodori-tts").read_text("direct_url.json"))
CODE = {"repository": _installed["url"], "commit": _installed["vcs_info"]["commit_id"]}
MODELS = {
    "large": {"repository": "Aratako/Irodori-TTS-v4-Large", "revision": "2e0c55428ce97268a507f1feeb2478f8d9148e8b",
              "size": 13152893348, "sha256": "0f3de3cdcb8cb82ac2d0089f2c051ab99e12eb7270625d06e3d1d303bad9e63c"},
    "anime": {"repository": "phasefield-audio/Irodori-TTS-v4.1-Anime", "revision": "6b259f5baa5e236b3d14cbd1f8555ca87d92b530",
              "size": 3064295596, "sha256": "5630aa0a661930ff678a1d1f1893876e24520f4776d4fcc25ca54e6ab9f41f37"},
    "mf": {"repository": "Aratako/Irodori-TTS-v4.1-Small-MF", "revision": "ccc78f5d480b6e51b69b2d5042a14c4da04fea6e"},
    "rf": {"repository": "Aratako/Irodori-TTS-v4.1-Small", "revision": "2b28324dc263ed5e6638b3cf3dd94c82ead07b4b"},
}
CODEC = {"repository": "Aratako/Semantic-DACVAE-Japanese-32dim", "revision": "47376ee24834d7a05a48ebabfe3cde29b3c5e214"}


def snapshot(pin):
    """The local folder of a pinned Hugging Face repository, downloaded on first use."""
    return snapshot_download(pin["repository"], revision=pin["revision"])
