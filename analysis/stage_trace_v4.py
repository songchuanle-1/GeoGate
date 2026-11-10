# SPDX-License-Identifier: Apache-2.0
"""Dependency-free stage trace gate for the forthcoming 3D-LLaVA runner."""


class StageForwardTrace:
    """Capture the actual decoder inputs and returned per-layer cache sizes."""

    def __init__(self, model):
        self.layers = []
        self.caches = []
        self.handles = []
        decoder = model.model
        if len(decoder.layers) != 32:
            raise ValueError("Frozen protocol requires 32 decoder layers")
        for index, block in enumerate(decoder.layers):
            def before(_module, args, kwargs, index=index):
                hidden = args[0] if args else kwargs["hidden_states"]
                self.layers.append((index, int(hidden.shape[1])))
            self.handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(decoder.register_forward_hook(self.after))

    def after(self, module, _args, output):
        cache = output.past_key_values
        if cache is None:
            raise ValueError("Stage experiment requires returned KV cache")
        lengths = ([int(cache.get_seq_length(i)) for i in range(len(module.layers))]
                   if hasattr(cache, "get_seq_length") else
                   [int(entry[0].shape[-2]) for entry in cache])
        self.caches.append(lengths)

    def reset(self):
        self.layers.clear()
        self.caches.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def validate_stage_trace(layers, caches, full, spatial_start, spatial_count,
                         keep, prune_layer, selected):
    if not (0 <= prune_layer < 32 and 0 <= spatial_start
            and spatial_start + spatial_count <= full and 1 <= keep <= spatial_count):
        raise ValueError("Invalid stage geometry")
    reduced = full - spatial_count + keep
    expected = [full] * (prune_layer + 1) + [reduced] * (31 - prune_layer)
    if layers[:32] != list(enumerate(expected)):
        raise ValueError("Unexpected prefill")
    if not caches or len(layers) != 32 * len(caches):
        raise ValueError("Incomplete forward trace")
    for step, cache in enumerate(caches):
        if cache != [reduced + step] * 32:
            raise ValueError("Unexpected KV growth")
        if step and layers[step * 32:(step + 1) * 32] != list(enumerate([1] * 32)):
            raise ValueError("Decode is not single-token")
    if (selected != sorted(set(selected)) or len(selected) != reduced
            or any(i < 0 or i >= full for i in selected)):
        raise ValueError("Invalid selected map")
    spatial_end = spatial_start + spatial_count
    expected_text = set(range(spatial_start)) | set(range(spatial_end, full))
    if not expected_text.issubset(selected):
        raise ValueError("Text token removed")
    if sum(spatial_start <= i < spatial_end for i in selected) != keep:
        raise ValueError("Actual spatial budget differs")
    return reduced
