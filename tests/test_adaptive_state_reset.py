# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from adaptive_token.geogate_v0 import (
        enable_geogate,
        reset_adaptive_state,
        set_span,
    )
except ModuleNotFoundError:
    torch = None
    reset_adaptive_state = None


@unittest.skipIf(torch is None, "PyTorch is required for adaptive state tests")
class AdaptiveStateResetTest(unittest.TestCase):
    def test_prefill_reset_contract(self):
        class State:
            _gg_adaptive_onset = True
            _gg_onset_layer = 12
            _gg_onset_hits = 2
            _gg_attn_history = [torch.tensor([1.0])]
            _gg_keep_ratio = 0.1
            _gg_stages = {12: 0.1}

        state = State()
        reset_adaptive_state(state)

        self.assertIsNone(state._gg_onset_layer)
        self.assertEqual(state._gg_onset_hits, 0)
        self.assertEqual(state._gg_attn_history, [])
        self.assertEqual(state._gg_stages, {-1: 0.1})

    def test_repeated_prefills_reset_adaptive_state(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval()
        enable_geogate(
            model,
            prune_layer=-1,
            keep_ratio=0.5,
            scorer="stable_attn",
            adaptive_onset=True,
            stability_window=2,
            min_onset_layer=0,
            max_onset_layer=2,
            onset_patience=1,
            onset_min_alpha=0.0,
            onset_max_coverage=1.0,
        )

        observed = []
        handle = model.model.layers[0].register_forward_pre_hook(
            lambda module, args: observed.append(model.model._gg_onset_layer)
        )
        try:
            for length in (12, 13):
                set_span(model, 2, 10)
                model.model._gg_onset_layer = 2
                model.model._gg_onset_hits = 3
                with torch.inference_mode():
                    model.generate(
                        torch.arange(1, length + 1).unsqueeze(0) % 32,
                        max_new_tokens=1,
                        do_sample=False,
                    )
        finally:
            handle.remove()

        self.assertTrue(observed)
        self.assertTrue(all(value is None for value in observed))
        self.assertIsNotNone(model.model._gg_onset_layer)

    def test_repeated_prefills_prune_each_sample(self):
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=1024,
            bos_token_id=1,
            eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval()
        enable_geogate(
            model,
            prune_layer=-1,
            keep_ratio=0.5,
            scorer="stable_attn",
            adaptive_onset=True,
            stability_window=2,
            min_onset_layer=2,
            max_onset_layer=2,
            onset_patience=1,
            onset_min_alpha=0.0,
            onset_max_coverage=1.0,
        )

        for length in (12, 13, 15):
            set_span(model, 2, 10)
            calls = []
            handles = []
            for index, block in enumerate(model.model.layers):
                def before(module, args, index=index):
                    calls.append((index, args[0].shape[1]))
                handles.append(block.register_forward_pre_hook(before))
            with torch.inference_mode():
                model.generate(
                    torch.arange(1, length + 1).unsqueeze(0) % 32,
                    max_new_tokens=3,
                    do_sample=False,
                )
            for handle in handles:
                handle.remove()

            expected = (
                [(0, length), (1, length), (2, length), (3, length - 4)]
                + [(index, 1) for _ in range(2) for index in range(4)]
            )
            self.assertEqual(calls, expected)
            self.assertEqual(model.model._gg_onset_layer, 2)
            self.assertEqual(len(model.model._gg_orig_map), length - 4)
            self.assertIsNone(model.model._gg_span)


if __name__ == "__main__":
    unittest.main()
