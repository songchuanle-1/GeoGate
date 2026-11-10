# SPDX-License-Identifier: Apache-2.0
"""Immutable per-question commits for interrupted multi-condition experiments."""

import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class QuestionJournal:
    def __init__(self, directory, manifest, question_ids, conditions):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.ids = tuple(str(q) for q in question_ids)
        self.conditions = tuple(conditions)
        if len(set(self.ids)) != len(self.ids) or not self.ids:
            raise ValueError("Question IDs must be unique and nonempty")
        if len(set(self.conditions)) != len(self.conditions) or not self.conditions:
            raise ValueError("Conditions must be unique and nonempty")
        expected = {"provenance": manifest, "question_ids": self.ids,
                    "conditions": self.conditions}
        encoded = canonical(expected).encode()
        path = self.directory / "manifest.json"
        if path.exists():
            if path.read_bytes() != encoded:
                raise ValueError("Manifest, exact question list, or conditions changed")
        else:
            if any(self.directory.iterdir()):
                raise ValueError("Nonempty journal without manifest")
            with path.open("xb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        self.paths = {q: self.directory / (hashlib.sha256(q.encode()).hexdigest() + ".json")
                      for q in self.ids}
        allowed = {p.name for p in self.paths.values()} | {"manifest.json"}
        if any(p.name not in allowed and not p.name.endswith(".pending")
               for p in self.directory.iterdir()):
            raise ValueError("Unknown journal artifact")
        for q, path in self.paths.items():
            if path.exists():
                self.validate(q, json.loads(path.read_text(encoding="utf-8")))

    def validate(self, question_id, rows):
        if question_id not in self.paths:
            raise ValueError("Unknown question")
        if not isinstance(rows, list) or len(rows) != len(self.conditions):
            raise ValueError("Incomplete condition group")
        if any(str(r["question_id"]) != question_id for r in rows):
            raise ValueError("Mixed question IDs")
        if sorted(r["condition"] for r in rows) != sorted(self.conditions):
            raise ValueError("Duplicate or incorrect condition")
        canonical(rows)

    def completed(self):
        return {q for q, path in self.paths.items() if path.exists()}

    def commit(self, question_id, rows):
        question_id = str(question_id)
        self.validate(question_id, rows)
        target = self.paths[question_id]
        payload = canonical(rows).encode()
        if target.exists():
            if target.read_bytes() != payload:
                raise ValueError("Conflicting committed output")
            return
        # Exclusive pending names preserve interrupted writes for diagnosis.
        import uuid
        pending = target.with_name(target.name + "." + uuid.uuid4().hex + ".pending")
        with pending.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and refuses replacement of a commit.
        os.link(pending, target)
        pending.unlink()

    def export(self, path):
        if len(self.completed()) != len(self.ids):
            raise ValueError("Cannot export an incomplete experiment")
        with Path(path).open("x", encoding="utf-8") as output:
            for q in self.ids:
                rows = json.loads(self.paths[q].read_text(encoding="utf-8"))
                self.validate(q, rows)
                for row in rows:
                    output.write(canonical(row) + "\n")
