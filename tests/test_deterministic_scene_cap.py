# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from adaptive_token.deterministic_scene_cap import (
        audit_scene_cap_trace,
        deterministic_permutation,
        enable_deterministic_scene_cap,
        scene_cap_seed,
        set_deterministic_scene_cap,
    )
except ModuleNotFoundError:
    torch = None


class FakeModel:
    def __init__(self, token_count, slots):
        self.token_count = token_count
        self.slots = slots
        self.capped = []

    def prepare_inputs_labels_for_multimodal(self):
        self.capped = [
            torch.randperm(self.token_count)[:2560]
            for _ in range(self.slots)
        ]
        return self.capped


class FakeGenerationModel(FakeModel):
    def prepare_inputs_labels_for_multimodal(self, input_ids):
        if input_ids.shape[1] == 1:
            return None
        return super().prepare_inputs_labels_for_multimodal()


@unittest.skipIf(torch is None, "PyTorch is required for scene-cap tests")
class DeterministicSceneCapTest(unittest.TestCase):
    def test_seed_is_stable_and_scene_specific(self):
        self.assertEqual(
            scene_cap_seed("scene/a", 10, 3000),
            scene_cap_seed("scene/a", 10, 3000),
        )
        self.assertNotEqual(
            scene_cap_seed("scene/a", 10, 3000),
            scene_cap_seed("scene/b", 10, 3000),
        )

    def test_permutation_does_not_consume_global_rng(self):
        torch.manual_seed(7)
        before = torch.rand(1)
        deterministic_permutation(3000, "scene/a", 10)
        after = torch.rand(1)
        torch.manual_seed(7)
        self.assertTrue(torch.equal(before, torch.rand(1)))
        self.assertTrue(torch.equal(after, torch.rand(1)))

    def test_slots_models_and_repeated_calls_share_indices(self):
        models = [FakeModel(3000, 4), FakeModel(3000, 4)]
        selected = []
        for model in models:
            enable_deterministic_scene_cap(model)
            set_deterministic_scene_cap(model, "scene/a", 10)
            model.prepare_inputs_labels_for_multimodal()
            audit = audit_scene_cap_trace(model, 4, 3000)
            selected.append(audit["selected_sha256"])
            self.assertTrue(all(
                torch.equal(model.capped[0], row)
                for row in model.capped[1:]
            ))
        self.assertEqual(selected[0], selected[1])

        models[0].prepare_inputs_labels_for_multimodal()
        repeated = audit_scene_cap_trace(models[0], 4, 3000)
        self.assertEqual(selected[0], repeated["selected_sha256"])

    def test_uncapped_scene_is_unchanged_and_has_no_trace(self):
        model = FakeModel(2000, 2)
        enable_deterministic_scene_cap(model)
        set_deterministic_scene_cap(model, "scene/a", 10)
        model.prepare_inputs_labels_for_multimodal()
        audit = audit_scene_cap_trace(model, 2, 2000)
        self.assertFalse(audit["applied"])
        self.assertFalse(torch.equal(model.capped[0], model.capped[1]))

    def test_context_is_required(self):
        model = FakeModel(3000, 2)
        enable_deterministic_scene_cap(model)
        with self.assertRaisesRegex(RuntimeError, "context"):
            model.prepare_inputs_labels_for_multimodal()

    def test_decode_call_preserves_prefill_trace(self):
        model = FakeGenerationModel(3000, 2)
        enable_deterministic_scene_cap(model)
        set_deterministic_scene_cap(model, "scene/a", 10)
        model.prepare_inputs_labels_for_multimodal(torch.ones(2, 3))
        before = audit_scene_cap_trace(model, 2, 3000)
        model.prepare_inputs_labels_for_multimodal(torch.ones(2, 1))
        after = audit_scene_cap_trace(model, 2, 3000)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
