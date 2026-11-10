# Reproduction status

## What this release can establish

The source snapshot is dated 2026-11-10. Packaging added license notices,
package markers, documentation, and portable replacements for machine-specific
examples. It did not change the pruning mathematics or rerun benchmark
inference. `source_manifest.json` records source and release hashes for the
copied research files.

| Component | Included | Additional requirements or limitation |
| --- | --- | --- |
| GeoGate core, cache mapping, span handling | Yes | PyTorch and Transformers 4.37.2; synthetic CPU tests supplied |
| LLaVA-3D basic ScanQA runner | Yes | Separately installed backbone, checkpoint, licensed scene assets and formatted questions |
| LLaVA-3D attention profile | Yes | Same backbone/assets; sample selection and prompt must match the intended analysis |
| Paired layer and analytical-cost controls | Current source | Complete question/length manifests, matching backbone preparation and configuration |
| Shared-input official-3DZip comparison | Current source | Historical modified pooling interface and official 3DZip stack are not bundled |
| B1 timing | Current source | CUDA, suitable GPU, complete matching questions, explicit output/manifest/status paths |
| Fixed-set deletion | Patch/interface | Historical full experiment assets are not bundled |
| Paired statistical analysis | Yes | Paired prediction files, annotations and scene metadata; CIDEr additionally uses pycocoevalcap |
| 3D-LLaVA full benchmark evaluators | No | Upstream-adapted scripts are withheld pending a verified redistribution license |
| B2/B4, MMScan and OpenEQA full evaluation stacks | No | This initial source release does not package every historical experiment |
| Weights, raw data, annotations and historical predictions | No | Obtain authorized assets independently |

## Environment

The synthetic integration target uses Python 3.10, PyTorch 2.6.0 and
Transformers 4.37.2. This is a CPU contract-test environment, not a complete
reconstruction of either 3D backbone's original environment. Inference also
requires the matching upstream vision encoder, preprocessing, and any compiled
point-cloud operators. Do not upgrade the internal Transformers API silently.

Install each real backbone in a separate environment according to its official
instructions. Do not install both unrelated packages named `llava` into one
environment and assume the import selects the intended model.

## Run order and data contracts

1. Acquire the backbone, model weights, ScanNet-derived inputs and benchmark
   annotations using their official distribution instructions and licenses.
2. Check a small example against the native upstream evaluator before enabling
   pruning. Preserve the prompt, image/point-cloud preprocessing and decode
   settings. This release does not supply a validated cross-version fallback.
3. Set the spatial span from the expanded multimodal prompt for every sample.
   Preserve text tokens and original positions; use one-token cached decoding.
4. Write separate baseline/candidate predictions. Preserve question IDs and
   scene IDs. Pair the same questions within the same runner; do not silently
   compare only an intersection when records are missing.
5. Score the complete intended split and report the selection history. ScanQA
   validation was used to select the configuration, so it is not an independent
   confirmation set.

The shared-input runner expects 4,675 unique ScanQA validation questions and
three conditions per question (`native`, `zip128`, `geogate128`). It validates
complete triples on resume. The cost-control builder consumes native token
lengths from that format and constructs L2/L8 budgets matching analytical LM
cost for L14/K128. Those budgets do not match measured latency.

Use `python -m <module> --help` for the actual command-line options, after
installing that module's dependencies. Use module invocation from the repository
root so local `analysis` and `adaptive_token` imports resolve correctly.

## Statistics and timing

Paired quality analysis must retain the question-to-scene map. Scene bootstrap
resamples whole scenes and computes a question-weighted mean. An interval
containing zero establishes neither equivalence nor non-inferiority. The
prespecified test margins and the historical selection history cannot be
reconstructed merely from a generic bootstrap utility.

For CIDEr, preserve the reference corpus and its IDF convention. Report
analytical LM estimates separately from measured generation, application time
and total GPU memory. Shared timings in mixed-scene batches induce dependence;
scene-only intervals are not automatically validated for that setting.

## Historical reproducibility gaps

Some recorded historical runner hashes differ from the available current
sources, and portions of the historical execution stack have not been
recovered. Installing the latest upstream code does not resolve that mismatch.
This release therefore provides inspectable implementation and entry points,
not a claim that every paper table can be rebuilt exactly from this checkout.

A complete measured native/official-3DZip/GeoGate quality–cost comparison and
independent quality preservation remain separate evidence requirements.
No completed OpenEQA-HM3D preservation result is claimed or supplied.
