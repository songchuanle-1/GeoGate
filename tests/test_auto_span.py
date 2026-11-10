# SPDX-License-Identifier: Apache-2.0
import unittest

try:
    import torch
    from adaptive_token.geogate_v0 import enable_auto_span
except ModuleNotFoundError:
    torch = None
    enable_auto_span = None


@unittest.skipIf(torch is None, "PyTorch is required for auto-span tests")
class AutoSpanTest(unittest.TestCase):
    def test_sets_span_from_expanded_embeddings(self):
        class LanguageModel:
            _gg_span = None

        class Model:
            def __init__(self):
                self.model = LanguageModel()

            def prepare_inputs_labels_for_multimodal(self, input_ids, *args, **kwargs):
                embeds = torch.zeros(1, 9, 4)
                return None, None, None, None, embeds, None

        model = Model()
        enable_auto_span(model, -200)
        input_ids = torch.tensor([[1, -200, 4, 5]])
        model.prepare_inputs_labels_for_multimodal(input_ids)
        self.assertEqual(model.model._gg_span, (1, 7))

    def test_decode_without_image_token_does_not_reset_span(self):
        class LanguageModel:
            _gg_span = (2, 8)

        class Model:
            def __init__(self):
                self.model = LanguageModel()

            def prepare_inputs_labels_for_multimodal(self, input_ids, *args, **kwargs):
                embeds = torch.zeros(1, input_ids.shape[1], 4)
                return None, None, None, None, embeds, None

        model = Model()
        enable_auto_span(model, -200)
        model.prepare_inputs_labels_for_multimodal(torch.tensor([[42]]))
        self.assertEqual(model.model._gg_span, (2, 8))


if __name__ == "__main__":
    unittest.main()
