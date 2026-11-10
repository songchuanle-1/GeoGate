# SPDX-License-Identifier: Apache-2.0
"""CPU transformer integration tests; no checkpoint or GPU allocation."""

import unittest

try:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from adaptive_token.geogate_v0 import enable_geogate, set_span, set_keep_ratio
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch and Transformers required")
class RepeatedPrefillDecodeTests(unittest.TestCase):
    def test_stage_deletion_switching(self):
        from adaptive_token.stage_arm_v4 import configure_arm
        from analysis.stage_protocol_v4 import arms
        from analysis.stage_trace_v4 import StageForwardTrace, validate_stage_trace
        config = LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, bos_token_id=1, eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval()
        enable_geogate(model, prune_layer=14, keep_ratio=0.1)
        trace = StageForwardTrace(model)
        try:
            for arm in arms("delete10"):
                torch.manual_seed(arm["seed"])
                counts = configure_arm(model, arm, 2, 10)
                trace.reset()
                with torch.inference_mode():
                    model.generate(torch.arange(1, 15).unsqueeze(0),
                                   max_new_tokens=2, do_sample=False)
                validate_stage_trace(
                    trace.layers, trace.caches, 14, 2, 10, counts["keep"],
                    arm["layer"], model.model._gg_orig_map.tolist())
        finally:
            trace.close()

    def test_switch_layer_between_questions(self):
        config = LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, bos_token_id=1, eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval()
        enable_geogate(model, prune_layer=14, keep_ratio=0.25)
        for layer, keep in ((2, 6), (8, 4), (14, 2), (2, 6)):
            model.model._gg_prune_layer = layer
            set_keep_ratio(model, keep / 8)
            set_span(model, 2, 10)
            calls = []
            handles = []
            for index, block in enumerate(model.model.layers):
                def before(module, args, index=index):
                    calls.append((index, args[0].shape[1]))
                handles.append(block.register_forward_pre_hook(before))
            with torch.inference_mode():
                model.generate(torch.arange(1, 15).unsqueeze(0),
                               max_new_tokens=3, do_sample=False)
            for handle in handles:
                handle.remove()
            expected = [14] * (layer + 1) + [6 + keep] * (31 - layer)
            self.assertEqual(calls[:32], list(enumerate(expected)))
            self.assertTrue(all(length == 1 for _, length in calls[32:]))
            self.assertEqual(len(model.model._gg_orig_map), 6 + keep)

    def test_repeated_prefills_and_single_token_decodes(self):
        config = LlamaConfig(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=32, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=1024, bos_token_id=1, eos_token_id=None,
            pad_token_id=0,
        )
        model = LlamaForCausalLM(config).eval()
        enable_geogate(model, prune_layer=14, keep_ratio=0.25)
        for length in (14, 17, 15):
            set_span(model, 2, 10)
            inputs = torch.arange(1, length + 1).unsqueeze(0) % 32
            calls = []
            handle = model.model.layers[0].register_forward_pre_hook(
                lambda module, args: calls.append(args[0].shape[1]))
            with torch.inference_mode():
                output = model.generate(inputs, max_new_tokens=3, do_sample=False)
            handle.remove()
            self.assertEqual(calls, [length, 1, 1])
            self.assertEqual(output.shape[1], length + 3)
            self.assertEqual(len(model.model._gg_orig_map), length - 6)
            self.assertIsNone(model.model._gg_span)


if __name__ == "__main__":
    unittest.main()
