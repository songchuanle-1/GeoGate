# GeoGate

**Training-free late spatial-token pruning for 3D large multimodal models.**

GeoGate applies FastV's head-averaged last-prompt attention ranking after a
selected decoder block. The default schedule processes blocks L0–L14 at full
length, retains 10% of spatial tokens before L15, preserves text and original
rotary positions, and compacts every layer's prompt KV cache before decoding.
Layer indices are zero-based. The reported intervention includes both prefill
pruning and all-layer prompt-cache compaction.

This initial release contains the current method implementation, LLaVA-3D
experiment entry points, analysis utilities, and synthetic tests. It is a source
release, not a complete archive of the historical experiments. See
[reproduction status](docs/reproduction.md) for dependencies and unavailable
execution assets. No model weights, datasets, raw predictions, or manuscript
sources are distributed here.

## Quick start: core checks

Use Python 3.10 in a separate environment. These dependencies are for the
synthetic CPU checks; a real 3D backbone also needs its own environment.

```bash
git clone https://github.com/songchuanle-1/GeoGate.git
cd GeoGate
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
```

The CPU checks use small, randomly initialized Llama models. They do not
download checkpoints or validate benchmark accuracy. Optional geometry tests
require `torch_scatter` and are skipped when it is unavailable.

## Use the pruning patch

Given a compatible, already loaded Llama-based model:

```python
from adaptive_token.geogate_v0 import enable_geogate, set_span

enable_geogate(model, prune_layer=14, keep_ratio=0.10)
# Set the half-open spatial span in the EXPANDED multimodal prompt.
# Recompute it for each input; do not use the unexpanded text-token positions.
set_span(model, spatial_start, spatial_end)
answer_ids = model.generate(**inputs, do_sample=False, num_beams=1, use_cache=True)
```

For LLaVA-3D, `enable_auto_span(model, IMAGE_TOKEN_INDEX)` obtains the span from
the real multimodal preparation call. The implementation patches an internal
Transformers interface and is not a general adapter for every model family or
Transformers version. The reference test target is Transformers 4.37.2.

Default GeoGate uses `scorer='attn'`, a single pruning layer, and no geometry,
merging, or adaptive-onset option. Additional options in the research source
represent ablations; their availability does not imply a quality guarantee.

## Run a LLaVA-3D ScanQA evaluation

Install [LLaVA-3D](https://github.com/ZCMax/LLaVA-3D) separately using its official
instructions. Obtain the model and ScanNet/ScanQA assets under their own terms.
Use a CUDA environment compatible with that backbone, not the CPU test
environment above. Put both repositories on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:/path/to/LLaVA-3D:${PYTHONPATH:-}"
mkdir -p outputs
python -m adaptive_token.run_scanqa_geogate \
  --model-path /path/to/LLaVA-3D-7B \
  --question-file /path/to/scanqa_val_questions.json \
  --video-folder /path/to/scene-data \
  --answers-file outputs/scanqa_r10.jsonl \
  --prune-layer 14 --keep-ratio 0.10 --seed 0
```

Set `--prune-layer -1` and choose a different output file for the native
baseline. The question file uses the upstream `question_id`, `text`, and
`video` fields; its prepared `text` includes the intended answer suffix.
Use identical inputs, seeds, prompt, and decoding settings for comparisons.
This basic runner is an entry point, not a substitute for the historical
shared-input protocol or evidence of equal preparation between separate runs.

## Experiments and analysis

| Purpose | Entry point |
| --- | --- |
| Spatial-attention profiles | `analysis/e1_attention_profile.py` |
| Native / L14-R10 evaluation | `adaptive_token/run_scanqa_geogate.py` |
| Shared-input native / 3DZip / K128 comparison | `adaptive_token/run_shared_input_quality_v3.py` |
| Analytical cost-matched L2 / L8 controls | `analysis/freeze_cost_controls_v4.py`, `adaptive_token/run_cost_controls_v4.py` |
| Paired layer sweep | `adaptive_token/run_layer_sweep_paired.py` |
| B1 timing with explicit boundaries | `adaptive_token/run_scanqa_timing_boundaries_v4.py` |
| Analytical LM FLOPs and prompt KV | `analysis/flops_estimate.py` |
| Paired EM and scene bootstrap | `analysis/paired_scanqa_bootstrap.py` |
| CIDEr and other paired metrics | `analysis/paired_cider_bootstrap.py`, `analysis/paired_scene_metric_bootstrap.py` |
| Fixed-set deletion interface | `adaptive_token/geogate_v0_fixedset_v21.py` |

Some advanced runners require historical upstream modifications and complete
manifests that are not supplied. In particular, the shared-input comparison
expects a backbone pooling interface with a `3dzip` mode; a stock LLaVA-3D
checkout alone does not provide the full tested stack. Consult the status table
before running them.

The analytical example needs no model or dataset:

```bash
python -m analysis.flops_estimate --prune-layer 14 --keep-ratio 0.10
```

It reports approximately 42.1% lower analytical LM FLOPs and 78.5% lower initial
prompt-cache storage for its default lengths. These figures exclude vision,
selection, and cache-indexing cost and are not measured application speedups.
R10 is a fraction; K128 is a fixed count. They are separate configurations.

## Evidence boundaries

The schedule was selected on ScanQA validation. Attention concentration and
deletion damage do not by themselves demonstrate that retaining a token set
preserves quality. The study found transfer losses and only 0.8–1.4% application
time reduction in the measured single-H100 setting. SQA3D validation did not
pass the prespecified non-inferiority tests. Matching final token counts with
3DZip is not a match of complete measured cost.

The current source hashes differ from some historical execution records.
[The source manifest](docs/source_manifest.json) identifies this release
snapshot; it does not certify that every historical result can be reproduced
from these files. The original numerical results have not been rerun for this
release.

## License and attribution

The GeoGate core and independent utilities are distributed under
[Apache-2.0](LICENSE). Files marked `CC-BY-NC-SA-4.0` retain the non-commercial,
share-alike conditions associated with the LLaVA-3D integration. The entire
repository must not be treated as uniformly Apache-licensed or unrestricted
for commercial use. See [THIRD_PARTY.md](THIRD_PARTY.md) and the per-file notices.

The ranking rule builds on [FastV](https://github.com/pkunlp-icler/FastV);
the decoder/cache interface builds on
[Hugging Face Transformers](https://github.com/huggingface/transformers).
Backbones, datasets, and weights retain their own terms.
