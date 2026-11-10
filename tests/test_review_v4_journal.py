# SPDX-License-Identifier: Apache-2.0
import tempfile
import unittest
from pathlib import Path

from analysis.review_v4_journal import QuestionJournal


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "journal"
        self.manifest = {"model_sha256": "abc", "protocol_sha256": "def"}
        self.rows = [{"question_id": "q1", "condition": c} for c in ("a", "b")]

    def journal(self, **kwargs):
        return QuestionJournal(self.root, kwargs.get("manifest", self.manifest),
                               kwargs.get("ids", ["q1", "q2"]), ["a", "b"])

    def test_resume_and_manifest_gate(self):
        self.journal().commit("q1", self.rows)
        self.assertEqual(self.journal().completed(), {"q1"})
        with self.assertRaises(ValueError):
            self.journal(manifest={"model_sha256": "changed"})
        with self.assertRaises(ValueError):
            self.journal(ids=["q1", "q3"])

    def test_reject_partial_duplicate_and_unknown(self):
        journal = self.journal()
        for rows in (self.rows[:1], [self.rows[0], self.rows[0]]):
            with self.assertRaises(ValueError):
                journal.commit("q1", rows)
        with self.assertRaises(ValueError):
            journal.commit("unknown", self.rows)

    def test_pending_not_completed_and_full_export(self):
        journal = self.journal()
        (self.root / "interrupted.pending").write_bytes(b'{"partial')
        self.assertEqual(self.journal().completed(), set())
        with self.assertRaises(ValueError):
            journal.export(Path(self.temp.name) / "out.jsonl")
        journal.commit("q1", self.rows)
        journal.commit("q2", [r | {"question_id": "q2"} for r in self.rows])
        output = Path(self.temp.name) / "out.jsonl"
        journal.export(output)
        self.assertEqual(len(output.read_text().splitlines()), 4)

    def test_existing_commit_never_overwritten(self):
        journal = self.journal()
        journal.commit("q1", self.rows)
        journal.commit("q1", self.rows)
        with self.assertRaises(ValueError):
            journal.commit("q1", [r | {"text": "changed"} for r in self.rows])
