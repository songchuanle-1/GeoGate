# Third-party attribution and license scope

The root Apache-2.0 license covers the GeoGate core and independent code carrying
that SPDX identifier. Per-file identifiers take precedence for integrations
marked with another license. Model weights, data and separately installed
dependencies are not relicensed by this repository.

## Hugging Face Transformers

`adaptive_token/geogate_v0.py` and its fixed-set variant
`adaptive_token/geogate_v0_fixedset_v21.py` adapt the Llama forward/cache interface provided
by [Transformers 4.37.2](https://github.com/huggingface/transformers/tree/v4.37.2),
distributed under Apache-2.0. GeoGate adds spatial scoring, token selection,
position tracking and cache compaction. The original upstream copyright
notice is retained in `NOTICE`. No complete Transformers distribution is
vendored here.

## FastV

The last-prompt attention ranking follows
[FastV](https://github.com/pkunlp-icler/FastV). GeoGate studies the pruning
schedule and execution of this existing scoring rule; it does not claim to
invent that rule. The FastV repository is an external reference, not a bundled
dependency.

## LLaVA-3D

[ZCMax/LLaVA-3D](https://github.com/ZCMax/LLaVA-3D) is by Chenming Zhu, Tai Wang,
Wenwei Zhang, Jiangmiao Pang and Xihui Liu. Its repository README states
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.

Files carrying `SPDX-License-Identifier: CC-BY-NC-SA-4.0` are the LLaVA-3D
integration and evaluation/profile adaptations. They add GeoGate configuration,
scoring/selection, tracing, paired comparisons or timing to the upstream model
interface. Their complete list appears in `docs/source_manifest.json`.
The [license text](licenses/CC-BY-NC-SA-4.0.txt) is included. Preserve attribution,
the change notices, and the applicable non-commercial/share-alike conditions.
No endorsement by the upstream authors is implied.

This repository's license statement concerns code. A model card's license
label must not be used to override the code repository's stated terms.

## 3D-LLaVA

[djiajunustc/3D-LLaVA](https://github.com/djiajunustc/3D-LLaVA) is a separately
maintained backbone. On 2026-11-10, a repository-level redistribution license
was not found in the inspected root. This release does not redistribute the
adapted 3D-LLaVA dense-captioning/referring evaluators or the upstream source.
Contact the upstream authors about rights not expressly granted. The generic
GeoGate interface is not evidence that every upstream evaluator is packaged or
validated here.

## Data and additional dependencies

ScanNet/ScanQA, SQA3D, Scan2Cap, ScanRefer, Multi3DRefer, MMScan and OpenEQA
assets are not included. Follow each official dataset's access requirements.
PyTorch, NumPy, pycocoevalcap, torch_scatter, pretrained Llama-family models,
vision encoders and tokenizers retain their own licenses.
