# SPDX-License-Identifier: Apache-2.0
"""Content hashes for the frozen model and imported multimodal source files."""

import hashlib
import inspect
from pathlib import Path


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def provenance(model, model_path, prepare):
    folder = Path(model_path)
    files = [p for p in folder.iterdir() if p.suffix in (".safetensors", ".json", ".model")]
    owners = (type(model), type(model.model), type(model.get_vision_tower()),
              type(model.get_model().get_video_tower()), prepare)
    source_paths = {Path(inspect.getfile(owner)).resolve() for owner in owners}
    return {
        "checkpoint_files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
                             for p in sorted(files)},
        "source_files": {str(p): digest(p) for p in sorted(source_paths)},
    }
