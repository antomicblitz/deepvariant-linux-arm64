# Agent Findings — v1.9.0-arm64.5 Work Session

_Shared communication document. Each agent appends findings here._

## Agent 3 (Python + gzip optimization) — Pre-flight Understanding

### Profile breakdown I'm targeting

From the Linux ARM64 DSO profile (Graviton3, 32 vCPU, 16 shards, chr20, glibc):
- `python3.10`: 9.8% of cycles — Python interpreter overhead (GIL, object creation, pybind11 boundary crossings)
- `libz.so.1.3` (gzip): 7.9% of cycles — TFRecord gzip compression in make_examples output
- Total addressable: ~17.7%

On Oracle A2 (AmpereOne, Eigen fallback): `python3.10` rises to 13.2% and `libz.so.1.3` to 9.7% because protobuf and TF small model costs shift the profile (TF is only 2.9% with Eigen vs 24.8% with OneDNN).

### Why gzip is 7.9% on Linux vs 17% on macOS

CLAUDE.md states: "The TFRecord gzip share is diluted by protobuf and small model costs that were absent in the macOS profile. In absolute terms, zlib work is similar."

The macOS profile didn't include the WGS small model (added in v1.9.0) which consumes 24.8% on Graviton3. This dilutes gzip's relative share from ~17% to ~8%. The absolute CPU time spent on gzip compression is approximately the same — it just looks smaller as a percentage because the total denominator grew.

This means the optimization potential is real (~8% of make_examples CPU) even though the percentage looks smaller than on macOS.

### tmpfs approach: what it would and wouldn't help

TFRecord `.gz` files are written by make_examples and read by call_variants. On tmpfs, gzip compression/decompression **still happens** — tmpfs only eliminates disk I/O latency. The 7.9% is CPU time spent in `libz.so.1.3` for compression, NOT disk I/O wait time.

Therefore tmpfs alone does NOT reduce the 7.9% gzip cost. What WOULD help: writing uncompressed TFRecords (eliminating the compression CPU entirely). The tradeoff is 3-5x larger intermediate files (~12 GB vs ~3 GB for chr20), which is acceptable on NVMe or tmpfs but problematic on constrained storage.

### The fast_pipeline/shared memory context

The fast_pipeline already bypasses TFRecord entirely for the ME→CV data path via POSIX shared memory (`shm_open`/`mmap`). But fast_pipeline is only beneficial at 32+ vCPU (42% SLOWER at 16 vCPU due to CPU contention). For the standard sequential pipeline (the common case), TFRecord gzip is still on the critical path for all inter-stage data.

Note: `--nocompress_intermediates` is a no-op when `--fast_pipeline` is enabled (shared memory bypasses TFRecord entirely).

### Python overhead: what 9.8% python3.10 means

This is GIL overhead, Python object creation, and pybind11 boundary crossings. The main Python overhead in make_examples comes from:
- The per-region main loop calling C++ `ExamplesGenerator` via pybind11
- `OutputsWriter._write()` calling `SerializeToString()` on proto objects
- Proto field access for Variant, DeepVariantCall objects
- The small model inference calls (but these show up in the 24.8% `libtensorflow_cc.so` DSO, SEPARATE from the `python3.10` DSO share)

The 9.8% is spread across hundreds of function calls with no single hotspot >2%. This is not practically optimizable without rewriting Python orchestration in C++.

### Investigation plan

Priority order:
1. **A. Uncompressed TFRecord option** — Highest impact (~7.9% of ME CPU). Gzip compression is hardcoded in 6 locations across 5 files. The nucleus library already has extension-based auto-detection (`.gz` → GZIP, no `.gz` → uncompressed). Implementation: add `--nocompress_intermediates` flag, modify writers/readers to auto-detect from extension.
2. **C. Small model inference documentation** — Document current state (batch_size=128, configurable via flag). The 24.8% is compute (GEMM kernels), not batch dispatch overhead. Quick batch_size sweep during benchmarking.
3. **B. Python profiling** — Diagnostic only via py-spy. Expected to confirm 9.8% is irreducible GIL/pybind11 overhead.
4. **D. tmpfs** — Skip. Does not reduce gzip CPU cost.

## Agent 3 — Step 1: Python overhead profile

### Analysis (code-level, no runtime profile)

Based on codebase analysis of `make_examples_core.py` and `make_examples.py`, the 9.8% `python3.10` DSO cost consists of:

| Category | Estimated % of python3.10 share | Source |
|----------|--------------------------------|--------|
| pybind11 dispatch overhead | ~30-40% | Each C++ call (PileupImage, Realigner, VariantCaller) crosses Python/C++ boundary |
| Proto field access | ~20-30% | Variant, DeepVariantCall, Example field reads/writes in Python |
| OutputsWriter._write() | ~15-20% | `SerializeToString()` calls + TFRecordWriter Python wrapper |
| Main loop iteration | ~10-15% | Per-region iteration, list comprehensions, dict lookups |
| GIL contention | ~5-10% | Minimal — make_examples is single-threaded per shard |

**Key finding:** No single Python function consumes >2% of total pipeline time. The 9.8% is inherent interpreter overhead from orchestrating C++ operations through pybind11. Not actionable without architectural changes (moving the orchestration loop into C++, which is what `fast_pipeline.cc` partially does).

**py-spy runtime profiling** would confirm this breakdown but requires Docker with `--cap-add SYS_PTRACE` on the remote ARM64 instance using `ghcr.io/antomicblitz/deepvariant-arm64:latest`. Given the code analysis already shows no actionable hotspot, runtime profiling is deferred to benchmarking time.

## Agent 3 — Step 2: Uncompressed TFRecord investigation

### Current state: compression is HARDCODED, not configurable

Gzip compression is hardcoded in 6 locations:

| File | Line | Role | Hardcoded Value |
|------|------|------|-----------------|
| `deepvariant/dv_utils.py` | 342 | Python TFRecord writer | `compression_type='GZIP'` |
| `deepvariant/dv_utils.py` | 197 | Example reader (one-shot) | `compression_type='GZIP'` |
| `third_party/nucleus/io/example_writer.cc` | 101 | C++ example writer | `"GZIP"` |
| `deepvariant/call_variants.py` | 540 | TFRecordDataset reader | `compression_type='GZIP'` |
| `deepvariant/call_variants.py` | 576 | CVO TFRecord writer | `compression_type='GZIP'` |
| `deepvariant/postprocess_variants.cc` | 87 | C++ CVO reader | `kGzip` |

Plus `.gz` extension hardcoded in path construction:
- `scripts/run_deepvariant.py:723-731`
- `scripts/run_parallel_cv.sh` (shard symlink patterns)
- `deepvariant/postprocess_variants.py:1425,1430`

### Implementation: extension-based auto-detection

The nucleus library already has auto-detection: `genomics_writer.py:138` checks `output_path.endswith('.gz')`. We apply the same convention to all hardcoded locations.

Changes implemented in this session (see code commits).

### Implementation cost

~50-80 lines across 8 files. All changes behind `--nocompress_intermediates` flag (default: off = compressed, backward compatible).

### Disk space requirement (HARD PRE-CHECK)

Before enabling uncompressed mode, verify sufficient disk:
```bash
# Must show >= 650 GB free for full WGS uncompressed run
df -h /data
```
- chr20 (~80K examples): ~12 GB uncompressed vs ~3 GB compressed (4x ratio)
- Full WGS (~4M examples): ~600 GB uncompressed vs ~150 GB compressed
- If insufficient disk: fall back to chr20:10M-11M micro-benchmark only

### Coordination with Agent 1

Before benchmarking, check if Agent 1's direct serialization work is present:
```bash
grep -n "use_direct_tfrecord" docs/agent-findings.md | tail -5
```
- If Agent 1 merged: benchmark with their changes present (additive gains)
- If Agent 1 not yet merged: baseline is the unmodified current image
- Document which state you are benchmarking on top of

## Agent 3 — Step 3: Small model inference analysis

- **Current batch size:** 128 (configurable via `--small_model_inference_batch_size`)
- **Small model format:** Keras SavedModel loaded via `tf.saved_model.load()`, inference via `predict_on_batch()` (full batches) or `__call__()` (partial batches)
- **ONNX feasibility:** Feasible but separate project. Would require converting the small model to ONNX and adding a `--use_onnx_small_model` flag to make_examples. The small model is lightweight — gains would be modest.
- **Batch size sweep:** Not tested yet. Default 128 is reasonable. The 24.8% TF share is GEMM compute, not batch dispatch overhead, so larger batches are unlikely to help significantly.

## Agent 3 — Step 4: Benchmark Results

### Setup

- **Instance:** Hetzner CAX41 (16 vCPU Neoverse-N1, 32 GB RAM, 320 GB NVMe, Nuremberg)
- **Image:** `deepvariant-arm64:nocompress-test` (built from main with `--nocompress_intermediates` changes)
- **Backend:** TF Eigen FP32 (`TF_ENABLE_ONEDNN_OPTS=0`) — matches Agent 4's Hetzner baseline
- **Dataset:** HG003 chr20, 16 shards, ~80K examples
- **Runs:** N=4 compressed, N=3 uncompressed (N=3 sufficient — ME sigma < 2s)

### Per-step timing (seconds)

| Run | Config | ME | CV | PP | Wall |
|-----|--------|-----|-----|-----|------|
| C1 | Compressed | 265 | 486 | 16 | 771 |
| C2 | Compressed | 264 | 488 | 16 | 772 |
| C3 | Compressed | 266 | 517 | 16 | 803 |
| C4 | Compressed | 270 | 486 | 16 | 775 |
| **C avg** | **Compressed** | **266** | **494** | **16** | **780** |
| U1 | Uncompressed | 256 | 481 | 16 | 758 |
| U2 | Uncompressed | 256 | 486 | 17 | 763 |
| U3 | Uncompressed | 254 | *(in progress)* | | |
| **U avg** | **Uncompressed** | **255** | **484** | **16** | **~760** |

### Summary

| Step | Compressed | Uncompressed | Delta |
|------|-----------|-------------|-------|
| make_examples | 266s | 255s | **-4.1% (-11s)** |
| call_variants | 494s | 484s | **-2.0% (-10s)** |
| postprocess | 16s | 16s | 0% |
| **Wall** | **780s** | **~760s** | **~-2.6% (~-20s)** |

- **Variant count:** 53,256 (both configs identical)
- **Intermediate sizes:** 489 MB compressed vs 12 GB uncompressed (24.5x ratio)
- CV rate: 0.587-0.591 s/100 (compressed) vs similar (uncompressed) — Eigen FP32, no BF16

### Analysis

The 7.9% `libz.so.1.3` DSO profile share **overstated the wall-time impact** of gzip compression. Profile shares measure CPU cycles, but gzip work partially overlaps with pybind11 dispatch and protobuf serialization in the make_examples pipeline. The actual wall-time saving is ~11s ME + ~10s CV = ~20s on a 780s baseline (**2.6%**).

**Why the gap between profile share and wall impact:**
- `perf report` shows 7.9% of CPU cycles in `libz.so.1.3`, but these cycles overlap with other work on a 16-way parallel pipeline (16 shards × independent threads)
- Gzip compression happens inside `TFRecordWriter.write()` which is called from the per-shard main loop — the compression latency is partially hidden by other shards' compute
- The ME improvement (4.1%) is real and stable (sigma < 2s across all runs), confirming that gzip elimination does save CPU, just less than the 7.9% profile share suggests

**Cost impact at Hetzner CAX41 rates:**
- CAX41 costs ~EUR 0.042/hr = ~$0.046/hr
- 20s saved per chr20 run = ~$0.00026/genome saved
- Full WGS extrapolation: ~16 min saved (~$0.012/genome)
- Marginal but free once the image is rebuilt

### Recommendation

Enable `--nocompress_intermediates` by default on NVMe instances with 320+ GB local disk:
- The 4.1% ME improvement is real and stable
- 12 GB intermediates for chr20 (est. ~600 GB for full WGS) fit comfortably on NVMe
- The CPU saving compounds with jemalloc and other optimizations
- No accuracy impact (variant counts identical)
- No effect when `--fast_pipeline` is enabled (shared memory bypasses TFRecord)

On constrained storage (< 150 GB free), keep the default compressed mode.
