# SPDX-License-Identifier: Apache-2.0
import unittest
from analysis.review_v3_protocol import CONDITIONS, onset, prompt_variants, rotation, sample_seed


class FrozenProtocolTests(unittest.TestCase):
    def test_rewrites_only_final_suffix(self):
        variants = prompt_variants("What color? Answer the question simply.")
        self.assertEqual(variants["no_suffix"], "What color?")
        self.assertEqual(variants["paraphrase"], "What color? Give a short answer.")

    def test_missing_suffix_rejected(self):
        with self.assertRaises(ValueError):
            prompt_variants("What color?")

    def test_rotation_balances_order(self):
        orders = [rotation(CONDITIONS, i) for i in range(3)]
        for column in range(3):
            self.assertEqual({x[column] for x in orders}, set(CONDITIONS))

    def test_last_layer_trigger_is_not_fallback(self):
        frac, coverage = [0.0] * 32, [1.0] * 32
        for layer in (17, 18):
            frac[layer], coverage[layer] = 0.2, 0.05
        self.assertEqual(onset(frac, coverage), {"layer": 18, "fallback": False})

    def test_missing_trigger_is_fallback(self):
        self.assertEqual(onset([0] * 32, [1] * 32), {"layer": 18, "fallback": True})

    def test_seed_independent_of_execution_order(self):
        self.assertEqual(sample_seed("q1"), sample_seed("q1"))
        self.assertNotEqual(sample_seed("q1"), sample_seed("q2"))


if __name__ == "__main__":
    unittest.main()
