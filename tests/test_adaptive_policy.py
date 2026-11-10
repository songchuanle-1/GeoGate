# SPDX-License-Identifier: Apache-2.0
import unittest

from adaptive_token.adaptive_policy import question_keep_ratio


class AdaptivePolicyTest(unittest.TestCase):
    def test_attribute_question_uses_base_budget(self):
        self.assertEqual(question_keep_ratio("What color is the chair?"), 0.10)

    def test_spatial_question_gets_double_budget(self):
        self.assertEqual(question_keep_ratio("Where is the chair?"), 0.20)

    def test_count_question_gets_larger_budget(self):
        self.assertEqual(question_keep_ratio("How many chairs are there?"), 0.25)

    def test_budget_is_capped(self):
        self.assertEqual(
            question_keep_ratio("How many chairs are there?", max_ratio=0.20),
            0.20,
        )


if __name__ == "__main__":
    unittest.main()
