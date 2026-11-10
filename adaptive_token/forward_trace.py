# SPDX-License-Identifier: Apache-2.0
"""Minimal forward/KV trace hooks shared by paired quality runners."""

from __future__ import annotations


class ForwardTrace:
    def __init__(self, model):
        self.calls = []
        self.layers = []
        self.model = model
        self.handles = [
            model.model.register_forward_hook(self.after_lm),
        ]
        for index, layer in enumerate(model.model.layers):
            def before(module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.layers.append((index, int(hidden.shape[1])))

            self.handles.append(
                layer.register_forward_pre_hook(before, with_kwargs=True)
            )

    def after_lm(self, module, args, output):
        cache = output.past_key_values
        lengths = (
            [
                int(cache.get_seq_length(index))
                for index in range(len(module.layers))
            ]
            if hasattr(cache, "get_seq_length")
            else [int(layer[0].shape[-2]) for layer in cache]
        )
        self.calls.append(lengths)

    def reset(self):
        self.calls.clear()
        self.layers.clear()
