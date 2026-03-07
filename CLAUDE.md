# CLAUDE.md — DeepVariant Linux ARM64 with Hardware Acceleration

## Project Overview

This project ports DeepVariant to Linux ARM64 with hardware-accelerated inference, creating an open-source variant caller that runs on cheap ARM cloud instances (Graviton, Ampere) and embedded ARM boards (RK3588). The goal is a viable alternative to proprietary solutions like Sentieon DNAscope — possibly somewhat slower, but dramatically cheaper per genome.

**Upstream repo:** [google/deepvariant](https://github.com/google/deepvariant) v1.9.0 (Bazel 5.3.0, TF 2.13.1)
**macOS ARM64 reference:** [antomicblitz/deepvariant-macos-arm64-metal](https://github.com/antomicblitz/deepvariant-macos-arm64-metal) — all C++ optimizations transfer directly
**License:** BSD-3-Clause (same as upstream)
**Current release:** v1.9.0-arm64.4 (2026-03-06)

***

## Strategic Context

### Why This Matters

- DeepVariant is the accuracy leader for variant calling (SNP F1 >99.9%, best-in-class indels)[1][2]
- No official ARM64 Linux build exists — the Docker image is x86-only
- Sentieon (proprietary) already proved 35% cost savings on Graviton vs x86
- Oracle Ampere A1 costs $0.01/OCPU-hour — 78% cheaper than equivalent x86[3]
- Graviton4 instances are 20-40% cheaper than comparable x86 with 60% less energy[4]
- TensorFlow has official aarch64 wheels since v2.9.0[5]

### Target Platforms (Priority Order)

| Platform | CPU | GPU/Accelerator | Memory Model | Use Case |
|----------|-----|-----------------|-------------|----------|
| AWS Graviton3/4 | Neoverse V1/V2 | None (CPU-only) | Discrete | Cloud cost savings |
| Oracle Ampere A2 | AmpereOne (Siryn) | None (CPU-only) | Discrete | Ultra-cheap cloud |
| Oracle Ampere A1 | Altra/Altra Max | None (CPU-only) | Discrete | Ultra-cheap cloud (capacity-limited) |
| NVIDIA Jetson Orin | Cortex-A78AE | Ampere GPU (CUDA) | Unified | Edge/on-premise |
| Rockchip RK3588 | Cortex-A76/A55 | Mali G610 (OpenCL) + 6 TOPS NPU | Shared | Low-resource labs |

**Platform compatibility notes:**
- **Graviton4 (c8g):** TF+OneDNN BF16 works but requires 64+ GB RAM (TF SavedModel uses ~26 GB RSS; OOM-killed on 32 GB machines when forking postprocess). Use ONNX backend on 32 GB instances.
- **Oracle A2 (AmpereOne/Siryn):** OneDNN+ACL (compiled for Neoverse-N1) triggers SIGILL on AmpereOne in both make_examples and call_variants. ISA cap also fails under high concurrency. Must use `TF_ENABLE_ONEDNN_OPTS=0` (Eigen fallback) for all binaries. Use INT8 ONNX for call_variants ($2.32/genome). BF16 permanently blocked — TF source rebuild with ACL v23.08 patches fixed 2 of 4 cascading bugs but Conv2D still falls to unaccelerated `gemm:ref` and Grappler remapping crashes inner_product. See `docs/onednn-ampereone.md`.
- **Oracle A1 (Altra/N1):** **Benchmarked — $1.04/genome (new cost leader).** INT8 ONNX + jemalloc, 486s chr20 at $0.16/hr (16 OCPU). 55% cheaper than A2. No BF16/i8mm — INT8 uses NEON GEMM. Higher run-to-run variance (sigma=14s). Capacity-limited in some regions. See `docs/oracle-a1-benchmark.md`.

***

## Architecture Decision: Inference Backend

The `call_variants` step runs Inception V3 (23.9M params) CNN inference on 100×221×7 uint8 pileup images — standard image classification with no custom ops. This is the bottleneck without GPU (83% of CPU-only wall time).

### Backend Comparison (Updated with Benchmark Results)

| Backend | Target Hardware | Maturity | Measured/Expected Speedup | Effort |
|---------|----------------|----------|--------------------------|--------|
| **TF OneDNN+ACL** | All ARM64 CPUs | Production | **Baseline winner** on Neoverse-N1[7] | Low |
| **TF OneDNN+ACL + BF16 fast math** | Graviton3+ (Neoverse V1/V2) | Production | **38% measured (1.61x call_variants)** on Graviton3[7] | Config |
| **INT8 quantization (ONNX dynamic)** | All ARM64 CPUs (ORT >= 1.17) | **BROKEN** | ConvInteger op unsupported on ARM64 CPUExecutionProvider | N/A |
| **INT8 quantization (ONNX static)** | All ARM64 CPUs (ORT >= 1.17) | Production | **2.3x over ONNX FP32 (measured)**, matches BF16 | Done |
| **ONNX Runtime (CPUExecutionProvider)** | All ARM64 CPUs | Production | **24% slower** than TF+OneDNN (measured)[6] | Done |
| **ONNX Runtime (MLAS BF16)** | Graviton3+ only | Production | 35-65% over ONNX CPU (NLP workloads; CNN gains lower) | Config |
| **EfficientNet-B3 (model swap)** | Any backend | **DEAD END** | **3x slower** on CPU (measured) — depthwise convs have poor GEMM density | Done |
| **MobileNetV2 / any depthwise-separable model** | Any backend | **DEAD END** | Same architecture class as EfficientNet — same CPU slowdown | N/A |
| **HELLO (1D CNN pipeline)** | Any backend | Research | Different pipeline entirely — not a model swap | Very High |
| **Apache TVM + OpenCL** | Mali G610/G710 GPUs | Alpha for Mali Valhall | Poor real results on G610[8][9] | High |
| **CUDA (Jetson)** | Jetson Orin only | Production | 3-5x | Low (TF native) |

**Key insight:** FLOPs do not predict CPU inference speed. Kernel efficiency and memory access patterns dominate. Dense Conv2D operations (InceptionV3, ResNet) map to single large GEMM calls with high arithmetic intensity. Depthwise separable convolutions (EfficientNet, MobileNet) produce many small kernel launches with poor data reuse. This is a general principle — not specific to any one model.

**BF16 vs INT8:** These are mutually exclusive for quantized layers. BF16 accelerates FP32 GEMM via BFMMLA; INT8 replaces those same GEMMs with SMMLA. Only mixed-precision (some layers FP32/BF16, others INT8) combines both.

### Recommended Strategy: Revised Tiered Approach

**Phase 1 (CPU-only, COMPLETE):** TF aarch64 wheel + OneDNN+ACL + C++ optimizations (haplotype cap, ImageRow flat buffer, query cache). Benchmarked at 12m57s on chr20:1-30M (GCP t2a-standard-8).

**Phase 2 (Inference acceleration — revised after literature review + experiments):**
- **2A: TF OneDNN warmup (DONE)** — SavedModel warmup gives ~4% call_variants improvement. KMP_AFFINITY and TF_ONEDNN_USE_SYSTEM_ALLOCATOR are **HARMFUL** (30% regression, reverted).
- **2B: BF16 on Graviton3+ (DONE)** — 1.61x call_variants speedup on Graviton3, zero accuracy loss. See section 2.2b.
- **2C: INT8 quantization of InceptionV3 (DONE)** — Static INT8 via ONNX Runtime gives 2.3x over ONNX FP32 but matches BF16 (no additional gain on Graviton3). Viable alternative for non-BF16 platforms. Dynamic INT8 unsupported on ARM64 (ConvInteger op missing).
- **~~2D: EfficientNet-B3~~** — **DEAD END.** 3x slower on CPU. Depthwise separable convolutions have poor GEMM density. This generalizes to ALL "efficient" CNN architectures (MobileNetV2, MnasNet, etc.). See `TRAINING_EXPERIMENT.md`.

**Phase 2 ONNX status:** The `--use_onnx` flag is implemented and works. On Neoverse-N1 (no BF16), ONNX CPUExecutionProvider is slower than TF+OneDNN. On Graviton3+ with BF16, ONNX MLAS BF16 is configured but INT8 ONNX matches BF16 TF+OneDNN speed (0.225 vs 0.232 s/100) — no additional gain from ONNX BF16.

**Phase 3 (GPU/NPU, optional):** TFLite + OpenCL for Mali, CUDA for Jetson. Only pursue after Phase 2 is validated.

***

## Phase 1: CPU-Only ARM64 Linux Build

### Goal
A working DeepVariant ARM64 Linux Docker image that produces bit-identical VCF output to the x86 reference build. No GPU acceleration yet — just make it run correctly on Graviton/Ampere.

### 1.1 Build System Modifications

The official Dockerfile hardcodes x86-specific paths (`bazel-out/k8-opt/bin/`). The macOS port already solved the Bazel 5.3.0 + TF 2.13.1 build on ARM64.

**Key changes needed:**

```
# settings.sh already handles aarch64 detection [cite:118]:
# - CUDNN_INSTALL_PATH="/usr/lib/aarch64-linux-gnu" for aarch64
# - DV_COPT_FLAGS exclude -march=corei7 for non-x86

# Dockerfile changes:
# 1. Replace FROM ubuntu:22.04 with multi-arch base
ARG TARGETARCH
FROM ubuntu:22.04

# 2. Replace k8-opt with aarch64-opt in COPY paths
COPY --from=builder /opt/deepvariant/bazel-out/aarch64-opt/bin/deepvariant/make_examples.zip .
# ... (all .zip and fast_pipeline binaries)

# 3. Use TF aarch64 wheel instead of x86
RUN pip install tensorflow-cpu-aws-graviton==2.14.1  # or official tensorflow-aarch64
```

**Build tasks:**

- [ ] Create `Dockerfile.arm64` based on upstream Dockerfile
- [ ] Port macOS Bazel patches to Linux ARM64 context (remove Apple-specific: zlib `TARGET_OS_MAC` guard not needed on Linux, no Boost Homebrew paths)
- [ ] Set `--config=mkl_aarch64_threadpool` for OneDNN+ACL backend[7]
- [ ] Cross-compile or native-build on Graviton instance (native preferred — 64+ cores makes build fast)
- [ ] Verify `int64_t` / `long` type issues — macOS port already fixed these but Linux ARM64 has `int64_t` = `long` (unlike macOS where it's `long long`), so these patches may need conditional compilation
- [ ] htslib ARM64 config: verify `config.h` settings (no SSE, NEON via `HAVE_NEON`, libdeflate should work natively)
- [ ] Protobuf 21.9 + CLIF Python-C++ bindings — test on aarch64 (CLIF may need patching)

### 1.2 TensorFlow Runtime Configuration

For optimal CPU-only inference on Graviton:[7]

```bash
# Enable OneDNN+ACL backend (critical — default Eigen is much slower)
export TF_ENABLE_ONEDNN_OPTS=1

# Enable BF16 fast math on Graviton3+ (has BF16 support)
grep -q bf16 /proc/cpuinfo && export DNNL_DEFAULT_FPMATH_MODE=BF16

# Thread configuration
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=false
export OMP_PLACES=cores

# TF session config
intra_op_parallelism_threads = num_vcpus
inter_op_parallelism_threads = 1
```

### 1.3 Port C++ Optimizations from macOS Build

All six optimizations from the macOS port are platform-independent and transfer directly:

| Optimization | Impact | Linux Portability |
|-------------|--------|-------------------|
| Haplotype cap (≤8 DeBruijn paths) | -14.7% make_examples | Direct — pure C++ algorithmic change |
| ImageRow flat buffer | -8.4% total | Direct — contiguous memory layout, benefits any ARM64 L1 cache (64B lines) |
| InMemorySamReader query cache | -2.1% total | Direct — C++ data structure, no platform API |
| Fast pipeline (shared memory IPC) | 2.25x (concurrent ME+CV) | Direct — POSIX `shm_open`/`mmap` works on Linux, no Apple-specific APIs |
| Smith-Waterman NEON (via sse2neon) | Already in upstream libssw | Verify native NEON path compiles (skip sse2neon translation on Linux ARM64 — use native NEON intrinsics if available) |
| TFRecord bypass in fast pipeline | 0% additional | Already implemented in fast_pipeline.cc |

**Critical note on fast pipeline:** The fast_pipeline binary uses Boost.Process v1 and POSIX shared memory. On Linux ARM64, replace Homebrew Boost paths with system packages (`libboost-dev`). The semaphore cleanup code already handles POSIX semantics.

### 1.4 Validation

**Accuracy must match x86 DeepVariant exactly.** Use the GIAB benchmark:

- [ ] Run HG003 chr20 case study on ARM64 build
- [ ] Compare VCF output against x86 Docker reference using `rtg vcfeval`
- [ ] Run `hap.py` (Illumina) for direct comparison with Google's published numbers[2][14]
- [ ] Target: SNP F1 ≥0.9995, INDEL F1 ≥0.9945 (published x86 reference)
- [ ] Verify all PASS variants identical between ARM64 and x86 builds (zero tolerance for high-confidence variant discrepancies)

***

## Phase 2: Inference Acceleration (Revised)

### Status

ONNX integration is **complete** (`--use_onnx` flag, model conversion, Docker images). Benchmarking revealed TF+OneDNN outperforms ONNX CPUExecutionProvider on Neoverse-N1. EfficientNet-B3 model swap is **dead** (3x slower on CPU). Strategy revised to focus on BF16 fast math (Graviton3+) and INT8 quantization of InceptionV3.

### 2.0 Benchmark Results (chr20:1-30M, ~46K examples, GCP t2a-standard-8)

| Config | make_examples | call_variants (rate) | Total |
|--------|-------------|---------------------|-------|
| Original baseline (TF) | 6m12s | 6m58s (0.88s/100) | 13m33s |
| + C++ optimizations | 5m35s | 7m04s (0.88s/100) | **12m57s** |
| + C++ opts + ONNX | 5m37s | 8m29s (1.09s/100) | 14m23s |

**Key finding:** ONNX CPUExecutionProvider is 24% slower than TF+OneDNN at steady-state inference. The "28-51% speedup" cited in literature refers to MLAS BF16 kernels on Graviton3+ (Neoverse V1), not the ACL ExecutionProvider. The ACL EP is community-maintained (16 operators, fragile builds, no pre-built wheels) and not worth pursuing.

### 2.1 ONNX Integration (COMPLETE)

- [x] `--use_onnx` and `--onnx_model` flags in `call_variants.py`
- [x] ONNX session loading with lazy import, provider fallback
- [x] Skip TF model loading when `--use_onnx` (reads shape from example_info.json)
- [x] Model conversion via tf2onnx in Docker image
- [x] `--call_variants_extra_args` passthrough working

### 2.2 TF OneDNN Tuning (PARTIAL — Warmup Only)

**SavedModel warmup (DONE, ~4% gain):** Running a dummy inference pass before the main loop triggers Grappler BatchNorm folding, persistent kernel caching, and memory layout optimization. Benchmarked at 0.512s/100 vs 0.532s/100 baseline (3.8% improvement on 16-vCPU). Already committed.

**KMP_AFFINITY and TF_ONEDNN_USE_SYSTEM_ALLOCATOR — PROVEN HARMFUL:**
Benchmarked `KMP_AFFINITY=granularity=core,compact,1,0` + `TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1` on t2a-standard-16: **30% regression** (0.692s/100 vs 0.532s/100). Reverted in commit `cb0c37a3`. Do not re-attempt.

**ONNX BF16 on Graviton3+ (configured, no additional gain):** `mlas.enable_gemm_fastmath_arm64_bfloat16` is set in `call_variants.py` ONNX session setup. Benchmarked on Graviton3: INT8 ONNX (0.225 s/100) already matches TF+OneDNN BF16 (0.232 s/100), so ONNX BF16 provides no further improvement.

**Diagnostic:** Run `DNNL_VERBOSE=1` to confirm ACL kernels are active in oneDNN. If output shows `ref` or `cpp` instead of `acl` primitives, BF16 env var is being silently ignored.

### 2.2b BF16 Fast Math on Graviton3+ (DONE — 1.61x call_variants speedup)

Benchmarked on AWS c7g.4xlarge (16 vCPU Graviton3, Neoverse V1). Full chr20, 2 runs averaged:

| Config | make_examples | call_variants (rate) | postprocess | Total |
|--------|--------------|---------------------|-------------|-------|
| FP32 | 255s | 298s (0.379s/100) | 29s | **582s (9m41s)** |
| BF16 | 278s | 185s (0.232s/100) | 24s | **487s (8m06s)** |

- **call_variants speedup: 1.61x (38% faster)**
- **Total wall speedup: 1.20x**
- **Accuracy: BF16 = FP32** (SNP F1=0.9974, INDEL F1=0.9940 — identical)
- DNNL_VERBOSE confirmed 196 ACL kernel references — ACL is active
- Graviton3 FP32 is already 26% faster than GCP Neoverse-N1 FP32 (0.379 vs 0.512s/100)
- BF16 env: `ONEDNN_DEFAULT_FPMATH_MODE=BF16`, `TF_ENABLE_ONEDNN_OPTS=1`, `OMP_NUM_THREADS=$(nproc)`

**Note:** BF16 and INT8 are mutually exclusive for quantized layers. BF16 accelerates FP32 GEMM via BFMMLA; INT8 replaces those same GEMMs with INT8 SMMLA. Only mixed-precision combines both.

### 2.2c INT8 Quantization of InceptionV3 (DONE — 2.3x over ONNX FP32, matches BF16)

Benchmarked static INT8 quantization via ONNX Runtime on AWS c7g.4xlarge (16 vCPU Graviton3).

**Quantization details:**
- Static INT8 with Percentile calibration (99.99), QDQ format, 500 calibration samples
- Dynamic INT8 **does not work** on ARM64 — produces `ConvInteger(10)` ops unsupported by CPUExecutionProvider
- Model size: 83.1 MB (FP32) → 21.2 MB (INT8), 74% smaller
- ORT 1.23.2, `CPUExecutionProvider`

**Benchmark results (isolated ONNX, chr20 ~80K images):**

| Model | bs=64 | bs=128 | bs=256 | bs=512 |
|-------|-------|--------|--------|--------|
| FP32 ONNX | 0.521 s/100 | 0.518 s/100 | 0.517 s/100 | 0.520 s/100 |
| INT8 static | 0.233 s/100 | 0.225 s/100 | 0.226 s/100 | 0.225 s/100 |
| **Speedup** | **2.24x** | **2.30x** | **2.29x** | **2.31x** |

- **INT8 vs BF16 (TF+OneDNN): ~same** (0.225 vs 0.232 s/100, ~3% faster)
- Batch size has minimal impact — INT8 SMMLA is efficient at all sizes tested

**Full pipeline (chr20, c7g.4xlarge 16 vCPU, single run):**

| Step | INT8 (pre-OMP fix) | INT8 (post-OMP fix, 3-run avg) | BF16 |
|------|------|------|------|
| make_examples | 307s | 299s | 278s |
| call_variants | 195s (0.238s/100) | 194s (0.237s/100) | 185s (0.232s/100) |
| postprocess | 14s | 14s | 24s |
| **Total** | **516s** | **507s** | **487s** |

> **OMP fix results:** Scoping OMP vars per-subprocess recovered 8s in make_examples (307→299s). The remaining 21s gap vs BF16 ME (278s) is baseline variance, not OMP-related. Postprocess 14s (INT8) vs 24s (BF16) is confirmed real (stable across 3 runs) — likely due to ONNX output format differences reducing postprocess work.

**Accuracy (rtg vcfeval, chr20 GIAB HG003):**

| Metric | INT8 | BF16 | Gate | Status |
|--------|------|------|------|--------|
| SNP F1 | **0.9978** | 0.9977 | ≥0.9974 | **PASS** |
| INDEL F1 | **0.9962** | 0.9961 | ≥0.9940 | **PASS** |

**Stratified region validation (DONE — INT8 PASSES all regions):**

| Region | INT8 SNP | BF16 SNP | INT8 INDEL | BF16 INDEL |
|--------|----------|----------|------------|------------|
| Aggregate chr20 | 0.9978 | 0.9977 | 0.9962 | 0.9961 |
| Homopolymers (≥7bp) | 0.9985 | 0.9985 | 0.9967 | 0.9963 |
| Simple Repeats | 0.9994 | 0.9994 | 0.9967 | 0.9961 |
| Tandem Repeats (201-10000bp) | 0.9983 | 0.9983 | 0.9926 | 0.9926 |
| Segmental Duplications | 0.9802 | 0.9744 | 0.9814 | 0.9814 |

INT8 matches or exceeds BF16 in all tested stratification regions. No localized degradation detected in homopolymers, tandem repeats, or segmental duplications. The production caveat is cleared.

**Key finding:** INT8 via ONNX gives essentially the **same performance as BF16 via TF+OneDNN** on Graviton3. INT8 is 2.3x over ONNX FP32, but since TF+OneDNN FP32 is faster than ONNX FP32, the net effect vs TF BF16 is negligible.

**Required fix:** INT8 quantization error causes some predictions to not sum to 1.0 (e.g., [0.992, 0.0, 0.0]). Fixed by renormalizing ONNX output: `predictions = predictions / predictions.sum(axis=1, keepdims=True)`. Without this fix, `round_gls()` in `call_variants.py` crashes.

**Conclusion:** INT8 is a viable alternative to BF16 for platforms without BF16 support (Neoverse-N1, Ampere Altra). On Graviton3+ where BF16 is available, INT8 offers no additional speedup. The main remaining levers are **more cores** (16→32 vCPU) and **fast_pipeline**.

**Scaling projections (32 vCPU, c7g.8xlarge, $1.15/hr):**

| Config | WGS Time | Cost/Genome | vs Google |
|--------|----------|-------------|-----------|
| BF16 sequential | ~3.7 hr | $4.25 | 2.8x slower, 15% cheaper |
| BF16 + fast_pipeline | ~2.8 hr | $3.27 | 2.2x slower, 35% cheaper |
| Google x86 reference | ~1.3 hr | $5.01 | baseline |

**Resolved:** AWS vCPU quota increased to 32 — c7g.8xlarge and c8g.8xlarge benchmarks complete.

### 2.2d Phase 2D: Scaling + Platform Expansion (COMPLETE)

**Priority order (maximize progress per hour):**

1. **Fix OMP make_examples overhead (DONE)** — OMP env vars scoped per-subprocess in `run_deepvariant.py` via explicit `env=` dicts. Benchmarked with 3 runs: make_examples 299s avg (down from 307s, ~2.6% gain), call_variants 194s avg, postprocess 14s avg, total 507s avg (down from 516s). The OMP fix recovered only 8s of the 29s gap vs BF16 make_examples (278s). The remaining ~21s is baseline variance, not OMP-related. Postprocess 14s vs 24s (BF16) confirmed real across 3 runs — likely due to ONNX output format differences.

2. **Graviton4 (c8g) benchmark (DONE)** — Benchmarked on c8g.4xlarge (16 vCPU, 32 GB, Neoverse V2).
   - **INT8 ONNX (2-run avg):** ME ~194s, CV ~158s (0.197 s/100), PP ~6s, total **~366s (6m06s)**. CV rate is **14% faster** per-core than Graviton3 (0.225 s/100). ME is **35% faster** (194s vs 299s). Total 28% faster than Graviton3 INT8 (366s vs 507s).
   - **ONNX FP32 (2-run avg):** ME 232s, CV 360s (0.446 s/100), PP 10s, total 602s.
   - **TF SavedModel OOM on 32 GB:** TF allocates ~26 GB RSS for InceptionV3 SavedModel. When call_variants forks postprocess subprocess, copy-on-write pages push total >32 GB → Linux OOM killer. Needs c8g.8xlarge (64 GB) for TF BF16 full pipeline.
   - **Standalone TF BF16 CV:** 0.328 s/100 — 29% faster than Graviton3 FP32 (0.379) but 41% slower than Graviton3 BF16 (0.232).
   - **Cost:** $0.68/hr × ~4.9hr WGS = ~$3.33/genome (INT8 ONNX).

3. **Oracle A2 (AmpereOne) benchmark (DONE)** — Benchmarked on VM.Standard.A2.Flex (16 OCPU, 32 GB). Oracle A1 (Altra) had no capacity in Frankfurt free tier.
   - **SIGILL with OneDNN+ACL:** ACL (compiled for Neoverse-N1) triggers SIGILL on AmpereOne in BOTH make_examples AND call_variants. ISA cap (`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD`) also fails under 16-way concurrency. Only fix: `TF_ENABLE_ONEDNN_OPTS=0` (Eigen fallback) for all binaries. See `docs/onednn-ampereone.md`.
   - **INT8 ONNX (2-run avg):** ME ~280s, CV ~245s (0.358 s/100), PP ~17s, total **~542s (9m02s)**. Cost: **~$2.32/genome** — cheapest INT8 config.
   - **TF Eigen FP32:** ME 287s, CV 325s (0.387 s/100), PP 17s, total **629s**. Rate matches Graviton3 FP32 (0.379 s/100).
   - **ONNX FP32:** ME 277s, CV 613s (0.759 s/100), PP 17s, total **907s**. ONNX is 1.96x slower than TF Eigen on AmpereOne.
   - **Cost winner:** Oracle A2 at **$2.32/genome** (INT8 ONNX) is the cheapest tested platform. At $0.32/hr for 16 OCPUs, the low hourly rate dominates.
   - **BF16 blocked:** Requires TF source rebuild with `-mcpu=ampere1` to recompile ACL for AmpereOne ISA. Projected ~$1.44/genome with BF16.

4. **Oracle A1 (Altra/N1) benchmark (DONE)** — Benchmarked on VM.Standard.A1.Flex (16 OCPU = 16 vCPU, 32 GB, $0.16/hr). New cost leader.
   - **INT8 ONNX jemalloc OFF (4-run avg):** ME 245s, CV 247s (0.309 s/100), PP 14s, wall **509s**. Cost: **$1.09/genome**.
   - **INT8 ONNX jemalloc ON (3-run avg):** ME 219s, CV 250s (0.313 s/100), PP 14s, wall **486s**. Cost: **$1.04/genome**.
   - **TF Eigen FP32 (1 run):** ME 247s, CV 470s (0.588 s/100), PP 14s, wall **735s**. Cost: $1.57/genome.
   - **ONNX FP32 (2-run avg):** ME 240s, CV ~569s (0.711 s/100), PP 14s, wall **828s**. Cost: $1.77/genome.
   - **A1 vs A2:** A1 is **55% cheaper** ($1.04 vs $2.32/genome) AND 11% faster (486s vs 544s). A1's INT8 CV rate is 21% faster than A2 (0.309 vs 0.389 s/100) — N1 NEON GEMM is well-optimized (primary ACL target). A1 costs 2x less per OCPU ($0.01 vs $0.04).
   - **Variance:** Higher than other platforms (sigma=14s). Two of 7 INT8 runs hit slow CV (267s vs ~240s). Recommend N>=6 for future A1 benchmarks.
   - **jemalloc benefit:** ME -10.6%, CV ~0%, wall -4.5%. Smaller than A2 (-6.9%) or Graviton3 (-9.0%), consistent with N1's lower malloc contention.
   - See `docs/oracle-a1-benchmark.md` for full results.

5. **Stratified region validation (DONE)** — INT8 passes all GIAB stratification regions. Production caveat cleared. See section 2.2c above.

5. **fast_pipeline at 16 vCPU (DONE — SLOWER than sequential)** — Benchmarked on Graviton3 c7g.4xlarge with BF16. Run1: 696s, Run2: 689s, avg **~693s** — **42% slower than sequential 487s**. CV rate degraded from 0.232 to 0.290 s/100 (25% slower) due to CPU contention between concurrent ME+CV. fast_pipeline only benefits with 32+ vCPU where ME and CV can have dedicated CPU cores.

6. **Oracle A2 32-vCPU scaling (DONE)** — Tested on 16 OCPU (32 vCPU), 64 GB RAM, INT8 ONNX.
   - **Test C (sequential, 32 shards):** ME 167s, CV 310s (0.358 s/100), PP 12s, wall **489s**. Cost: **$4.19/genome**.
   - **Test D (sequential, 16 shards):** ME 297s, CV 312s (0.384 s/100), PP 16s, wall **629s**. Cost: $5.38/genome.
   - **Test A (fast_pipeline, partitioned cores 0-15 ME, 16-31 CV):** ME ~437s, CV ~414s (0.489 s/100 with streaming stalls), PP **FAILS** (CVO ordering issue), wall 483s.
   - **ME scales well:** 297s (16 shards) → 167s (32 shards), 1.78x with 2x shards.
   - **CV does NOT scale beyond 16 threads:** INT8 ONNX rate 0.358 s/100 at both 16 and 32 threads. This is the bottleneck.
   - **fast_pipeline verdict:** <1% wall improvement over sequential, PP broken, CV rate degrades with streaming stalls. **Not worth pursuing on Oracle A2.**
   - **Best 32-vCPU config:** Sequential, 32 shards = **489s**.
   - **Scaling efficiency:** Doubling vCPUs (16→32) gives 1.11x speedup (542→489s, 10%) — poor scaling because CV is the bottleneck.

7. **Graviton3/4 32 vCPU (DONE)** — Benchmarked on c7g.8xlarge and c8g.8xlarge (both 32 vCPU, 64 GB).
   - **Graviton3 BF16+jemalloc 32 shards:** ME 131s, CV 141s, PP 8s, wall **283s**. $/genome = $4.35.
   - **Graviton4 BF16+jemalloc 32 shards:** ME 100s, CV 126s, PP 5s, wall **232s**. $/genome = $4.22.
   - **Graviton4 INT8+jemalloc 32 shards:** ME 96s, CV 129s, PP 5s, wall **233s**. $/genome = $4.24.
   - **Key insight:** BF16 and INT8 converge at 32 vCPU (~232s) because both hit the CV floor. Backend choice is irrelevant once ME < CV. CV is the hard bottleneck — doesn't improve beyond 16 ORT threads.
   - **Solution:** Parallel call_variants (Phase 2F) breaks through the CV floor.

8. **SVE Smith-Waterman** (deferred) — Only relevant if ME becomes the bottleneck. With parallel CV, ME is again the bottleneck on some platforms.

### 2.2e Phase 2E: Runtime Optimizations (inter-op done, jemalloc VERIFIED)

**ONNX Inter-op Parallelism Sweep (Oracle A2 32-vCPU, INT8 ONNX) — NO IMPROVEMENT:**

Tested whether splitting ORT threads into intra-op (GEMM parallelism within operators) + inter-op (parallelism between InceptionV3's parallel branches) improves throughput. Swept 6 configs:

| Config | intra_op | inter_op | CV Rate (s/100) | vs Baseline |
|--------|----------|----------|-----------------|-------------|
| Baseline | 32 | 1 | 0.367 | — |
| Best | 28 | 2 | 0.363 | -1.1% (noise) |
| | 24 | 4 | 0.401 | +9.3% worse |
| | 20 | 6 | 0.368 | +0.3% (same) |
| | 16 | 8 | 0.410 | +11.7% worse |
| | 16 | 16 | 0.397 | +8.2% worse |

**Verdict:** Inter-op parallelism does not help InceptionV3. GEMM intra-op parallelism dominates — taking threads away from intra-op to give to inter-op hurts performance. Flags added to `call_variants.py` (`--onnx_intra_op_threads`, `--onnx_inter_op_threads`) but defaults remain optimal (all intra, 1 inter).

**jemalloc (VERIFIED on Graviton3 and Oracle A2):**

jemalloc integrated into Docker image via `DV_USE_JEMALLOC=1`. Verified results:

- **Graviton3 (N=2):** ME -13.8%, CV +1.6% (noise), wall -9.0% (487→443s)
- **Oracle A2 (N=4):** ME -17.0%, CV within noise, wall -6.9% (584→544s)

ME improvement is the dominant factor — jemalloc's per-thread arenas reduce malloc contention in make_examples' C++ allocations. CV sees minimal benefit because ONNX Runtime and TF have their own internal allocators.

### 2.2f Phase 2F: Parallel call_variants (DONE — breaks CV floor)

**Problem:** INT8 ONNX call_variants rate doesn't improve beyond 16 ORT threads (0.358 s/100 at both 16 and 32 threads). On 32-vCPU machines, CV is the hard bottleneck — ME finishes in ~100-130s but CV takes 126-283s. No single-process optimization can break this floor.

**Solution:** Split ME output shards across N independent call_variants workers running in separate Docker containers, each using `$(nproc)/N` OMP threads. Merge outputs via postprocess_variants using `@N` sharded file pattern.

**Implementation:** `scripts/benchmark_parallel_cv.sh` — zero code changes to DeepVariant. Uses shard renumbering via symlinks (original shard `make_examples.tfrecord-00005-of-00032.gz` → worker-local `examples.tfrecord-00000-of-00008.gz`) and `@N` notation for both CV input and output.

**Critical postprocess flags (discovered via debugging):**
- `--regions` must match the region used in make_examples
- `--small_model_cvo_records` must point to make_examples' small model outputs
- Without these, postprocess partitions across the whole genome and a pre-existing CVO sanity check bug silently kills the chr20 partition worker in `multiprocessing.Pool` (producing only ~125 RefCall variants instead of 207,799)

**Benchmark results (32 vCPU, INT8 ONNX, jemalloc ON):**

| Platform | Sequential CV | 2-way CV | 4-way CV | 4-way Speedup | N (4-way) |
|----------|--------------|----------|----------|---------------|-----------|
| **Graviton4** (c8g.8xlarge) | 128s | 83s | **61s** | **2.10x** | 3 |
| **Graviton3** (c7g.8xlarge) | 141s | 101s | **74s** | **1.90x** | 4 |
| **Oracle A2** (16 OCPU) | 283s | 164s | **114s** | **2.47x** | 2* |

*All variant counts match sequential baseline exactly (207,799).*

**Projected full pipeline with 4-way parallel CV:**

| Platform | ME | CV (4-way) | PP | Wall (proj) | $/hr | $/genome |
|----------|-----|-----------|-----|-------------|------|----------|
| **Oracle A2** | 113s | 114s | 10s | **~250s** | $0.64 | **$2.14** |
| **Graviton4** | 100s | 61s | 6s | **~172s** | $1.36 | **$3.13** |
| **Graviton3** | 131s | 74s | 8s | **~218s** | $1.15 | **$3.35** |

**Why 4-way scales nearly linearly on Oracle A2 (2.47x) but less on Graviton (1.90-2.10x):**
Oracle A2's sequential CV (283s at 32 ORT threads) wastes 16 threads' worth of GEMM parallelism (rate doesn't improve past 16 threads). Four 8-thread workers each operate near the GEMM saturation point. On Graviton3/4, BF16 BFMMLA kernels have higher arithmetic intensity and better thread scaling, so the single-process baseline is already more efficient, leaving less room for parallel improvement.

**Memory budget:** INT8 ONNX uses ~3 GB RSS per CV worker. 4 workers × 3 GB = 12 GB + ME headroom = ~16 GB total. Safe on 64 GB instances. TF SavedModel (~26 GB per process) would OOM with 2+ workers on 64 GB — parallel CV only works with ONNX backend.

### 2.2g Phase 2G: EncodeExample Serialization Optimization (CLOSED — 0% impact)

**Analysis:** Profiling shows `_message.so` (protobuf) at 29-43% of make_examples CPU — but this covers ALL protobuf ops in the process (pybind11 proto conversions, Read access, Variant field access), not just `EncodeExample` serialization. Function-level breakdown unavailable (Docker strips debug symbols).

**Investigated and rejected:**
- **Protobuf arena allocation:** Tested `Arena::CreateMessage<tensorflow::Example>(arena)` — 0% impact on Linux ARM64 (wall -0.2%, cache misses -0.5%, both within noise). Reverted. The `_message.so` share is dominated by varint serialization and proto field access, not allocation. See `docs/protobuf-serialization-notes.md` for full benchmark data.
- **SerializeToArray swap:** `SerializeToString` already does single allocation internally. Switching to `SerializeToArray` saves one `resize()` call = 0.008% of runtime.
- **Direct TFRecord serialization:** Implemented (ee9a910d). Bypasses `tensorflow::Example` proto entirely — writes protobuf wire format bytes directly via `CodedOutputStream`. Eliminates one ~154KB image copy per example. Benchmarked on both Graviton3 (34.59s vs 34.65s, -0.2%) and Oracle A2 (46.29s vs 46.39s, -0.2%). **0% measurable impact** — the ~20µs/example savings from one memcpy is far below the noise floor. The `_message.so` share is dominated by Read/Variant proto operations and pybind11 boundary crossings, not Example serialization.

**Changes retained (code is cleaner):**
1. **`std::ostringstream` → `absl::StrCat`** (make_examples_native.cc): Variant range string formatting.
2. **Reusable pileup buffer** (make_examples_native.cc/h): Persistent `encode_buffer_` member eliminates ~80K malloc/free cycles of ~154KB buffers per genome.
3. **Direct wire format serialization** (make_examples_native.cc): Removes dependency on `tensorflow/core/example/{example,feature}.pb.h`. Uses `CodedOutputStream` to write tf::Example wire bytes directly from raw data.

**Benchmark (2026-03-07, chr20:10M-11M, 2878 examples, 3 runs each):**

| Platform | Baseline (proto) | Direct serial | Delta |
|----------|-----------------|---------------|-------|
| Graviton3 (BF16+jemalloc) | 34.65s ± 0.08s | 34.59s ± 0.08s | -0.2% (noise) |
| Oracle A2 (Eigen, no jemalloc) | 46.39s ± 0.07s | 46.29s ± 0.00s | -0.2% (noise) |

**Conclusion:** The EncodeExample serialization path is fully optimized. The protobuf bottleneck in `_message.so` is NOT in Example creation — it's in Read/Variant proto field access and pybind11 boundary crossings across the entire make_examples pipeline. No further optimization is possible without architectural changes (moving proto access into C++).

### 2.3 EfficientNet-B3 Model (DEAD END)

**Attempted and measured.** Despite 3.2x fewer FLOPs, EfficientNet-B3 is **3x slower** than InceptionV3 on CPU inference. The full training pipeline was built and validated (see `TRAINING_EXPERIMENT.md`), but the speed result kills this approach.

**Root cause:** EfficientNet's depthwise separable convolutions and squeeze-and-excitation blocks produce many small kernel launches with poor data reuse, while InceptionV3's dense `Conv2D` operations map to single large GEMM calls with high arithmetic intensity. FLOPs ≠ speed on CPUs.

| Model | FLOPs | Params | CPU img/s (batch=128) | Relative Speed |
|-------|-------|--------|-----------------------|----------------|
| InceptionV3 | 5.7G | 23.9M | 591 | **1.0x** |
| EfficientNet-B3 | 1.8G | 12.3M | 185 | **0.31x** |

### 2.4 Cumulative Performance

| Optimization | Impact | Status | Best Measured |
|-------------|--------|--------|---------------|
| C++ opts (haplotype cap, flat buffer, query cache) | make_examples -10% | **DONE** | 7m33s total (8-core: 12m57s) |
| TF warmup (dummy inference pass) | call_variants -4% | **DONE** | 7m22s total (0.512s/100) on GCP 16-vCPU |
| BF16 on Graviton3+ | **call_variants -38% (1.61x)** | **DONE** | 8m06s total (0.232s/100) on Graviton3 16-vCPU |
| INT8 static quantization (ONNX) | **2.3x over ONNX FP32** | **DONE** | 0.225s/100 — matches BF16, no additional gain on Graviton3 |
| OMP env scoping (per-subprocess) | ME: 307→299s (2.6%), total 516→507s | **DONE** | 3-run avg; remaining 21s gap vs BF16 ME is baseline variance |
| Graviton4 INT8 ONNX (Neoverse V2) | **28% faster total** vs Graviton3 INT8 | **DONE** | 366s (0.197s/100), ME 194s — ~$3.33/genome |
| Oracle A2 INT8 ONNX (AmpereOne) | **$2.32/genome** | **DONE** | 542s (0.358s/100); OneDNN SIGILL, needs Docker rebuild |
| **Oracle A1 INT8 ONNX (Altra/N1)** | **$1.04/genome** (new cost leader) | **DONE** | 486s (0.309 s/100 CV), jemalloc ON. 55% cheaper than A2. See `docs/oracle-a1-benchmark.md` |
| fast_pipeline at 16 vCPU | **42% SLOWER** (CPU contention) | **DONE** | 693s vs 487s sequential — needs 32+ vCPU |
| Oracle A2 32-vCPU sequential 32 shards | **489s** (10% faster than 16 vCPU) | **DONE** | ME scales (167s), CV doesn't scale beyond 16 threads |
| Oracle A2 32-vCPU fast_pipeline | **<1% improvement, PP broken** | **DONE** | 483s wall, PP fails on CVO ordering — not worth pursuing |
| Graviton3 32 vCPU BF16+jemalloc | **283s** chr20, ME=131, CV=141, PP=8 | **DONE** | $4.35/genome. CV is hard floor — doesn't improve beyond 16 threads. |
| Graviton4 32 vCPU BF16+jemalloc | **232s** chr20, ME=100, CV=126, PP=5 | **DONE** | $4.22/genome. BF16 and INT8 converge (233s INT8). |
| jemalloc (`DV_USE_JEMALLOC=1`) | **ME -14-17%, CV ~0%, Wall -7-9%** | **VERIFIED** | Graviton3: 487→443s (2 runs). Oracle A2: 584→544s (4 runs). Universal ARM64 benefit. |
| **Parallel call_variants (4-way)** | **CV 2.0-2.5x faster** | **DONE** | Graviton4: CV 128→61s (2.10x). Graviton3: CV 141→74s (1.90x). Oracle A2: CV 283→114s (2.47x). |
| ONNX inter-op parallelism | **No improvement** | **DONE** | intra-op GEMM dominates; inter-op hurts |
| EncodeExample: StrCat + buffer reuse + direct serial | **0%** | **CLOSED** | G3: 34.59 vs 34.65s (-0.2%); A2: 46.29 vs 46.39s (-0.2%). See `docs/protobuf-serialization-notes.md` |
| Protobuf arena allocation (tf::Example) | **0%** | **CLOSED** | Tested and reverted. Wall -0.2%, cache -0.5% (noise). See `docs/protobuf-serialization-notes.md` |
| KMP_AFFINITY + system allocator | **30% REGRESSION** | **REVERTED** | Do not re-attempt |
| ~~EfficientNet-B3~~ | **3x SLOWER** | **DEAD END** | Depthwise separable conv penalty |
| ~~MobileNetV2~~ | **Same architecture class** | **DEAD END** | Same depthwise conv penalty |

***

## Phase 3: Model Modernization — ABANDONED

### EfficientNet-B3: Attempted and Failed

The full training pipeline was built and a model was trained (see `TRAINING_EXPERIMENT.md`). However, CPU inference benchmarking showed EfficientNet-B3 is **3x slower** than InceptionV3 despite having 3.2x fewer FLOPs. This is because EfficientNet's depthwise separable convolutions and squeeze-and-excitation blocks have poor computational density on CPUs — many small operations with poor cache reuse vs InceptionV3's large dense GEMM calls.

**The ISPRAS paper's accuracy improvements may be real, but are irrelevant if the model is 3x slower on our target hardware (ARM64 CPUs without GPU).**

EfficientNet-B3 might still be viable for:
- GPU-accelerated inference (Jetson Orin, desktop GPUs) where parallelism hides kernel dispatch overhead
- NPU inference (RK3588) where the smaller model fits in limited memory budgets
- But these are niche use cases that don't justify the complexity

### What Was Built (Preserved for Reference)
- `deepvariant/keras_modeling.py` — `efficientnetb3()` function, name-based weight transfer
- `scripts/train_efficientnet_b3.sh` — Full training data pipeline
- `scripts/export_efficientnet_b3.py` — Checkpoint → SavedModel export
- Training config, dataset configs, protobuf compilation

***

## Phase 4: GPU/NPU Acceleration (Optional, Hardware-Specific)

### 4.1 Jetson Orin (CUDA — easiest path)

Jetson Orin has unified memory (like Apple Silicon) and CUDA support. TensorFlow already supports CUDA on ARM64 via NVIDIA's builds.

- [ ] Use NVIDIA's TF wheel for Jetson (`tensorflow-aarch64` with CUDA)
- [ ] No model conversion needed — SavedModel runs directly
- [ ] Expected 3-5x speedup on call_variants (similar to desktop NVIDIA GPUs)
- [ ] Unified memory eliminates CPU→GPU copy overhead

### 4.2 Mali G610 / OpenCL (RK3588)

Real-world testing shows Mali G610 OpenCL performance is disappointing for ML inference — approximately 75% of CPU capability in some benchmarks, with TVM autotuning showing most tasks failing on Valhall architecture.[9][8]

**Recommended approach for RK3588:**
- Use the 6 TOPS NPU via RKNN-Toolkit2 (Rockchip's proprietary SDK) rather than the GPU
- Convert ONNX model → RKNN format
- INT8 quantization is mandatory for the NPU
- This is a high-effort integration with uncertain payoff — defer to Phase 4

### 4.3 TFLite + GPU Delegate

NXP's i.MX 95 platform demonstrates TFLite GPU Delegate working with Mali G310 via OpenCL. A similar approach could work for Mali G610:[10]

- [ ] Convert SavedModel → TFLite (with quantization)
- [ ] Enable GPU Delegate for OpenCL
- [ ] Benchmark vs CPU-only TFLite
- [ ] This is the most portable GPU path but may have limited speedup on Mali

### 4.4 Future: ExecuTorch + Arm Neural Technology

Arm's 2026 Mali GPUs with neural technology will support ExecuTorch via the VGF backend. This is the long-term best path for Mali GPU acceleration but hardware isn't available yet.[11][12]

***

## Build & CI Infrastructure

### Docker Multi-Architecture Build

```dockerfile
# Dockerfile.arm64
FROM arm64v8/ubuntu:22.04 AS builder

# Install Bazel 5.3.0 for aarch64
RUN wget https://github.com/bazelbuild/bazel/releases/download/5.3.0/bazel-5.3.0-linux-arm64 \
    -O /usr/local/bin/bazel && chmod +x /usr/local/bin/bazel

# Install TF aarch64 dependencies
RUN pip install tensorflow==2.14.1  # official aarch64 wheel

# Build with OneDNN+ACL
RUN bazel build --config=mkl_aarch64_threadpool \
    //deepvariant:make_examples.zip \
    //deepvariant:call_variants.zip \
    //deepvariant:postprocess_variants.zip \
    //deepvariant:fast_pipeline
```

### CI Pipeline (GitHub Actions)

```yaml
# .github/workflows/arm64-build.yml
jobs:
  build-arm64:
    runs-on: ubuntu-latest  # with QEMU for ARM64, or self-hosted Graviton runner
    steps:
      - uses: docker/setup-qemu-action@v3
        with:
          platforms: arm64
      - name: Build ARM64 Docker image
        run: docker buildx build --platform linux/arm64 -f Dockerfile.arm64 .
      
  test-accuracy:
    needs: build-arm64
    runs-on: [self-hosted, arm64]  # Graviton runner
    steps:
      - name: Run HG003 chr20 benchmark
        run: bash scripts/benchmark_arm64.sh --accuracy
      - name: Compare VCF against x86 reference
        run: rtg vcfeval --baseline ref_x86.vcf.gz --calls arm64.vcf.gz
```

***

## Known Pitfalls (From macOS Port — Do NOT Repeat)

These optimizations were investigated on macOS and found to have zero impact. They are architecture-independent dead ends:

| Attempted Optimization | Result | Why |
|----------------------|--------|-----|
| htslib decompression threads | 0% | libdeflate already fast, bottleneck is downstream |
| Protobuf Arena allocation | 0% | Confirmed 0% on both macOS and Linux ARM64 (Neoverse-N1). Arena::CreateMessage\<tensorflow::Example\> tested and benchmarked: cache misses -0.5%, wall -0.2% (noise). `_message.so` 49% share is Smith-Waterman + serialization, not allocation. See `docs/protobuf-serialization-notes.md` |
| TFRecord gzip removal | 0% in fast pipeline | Shared memory bypasses TFRecord entirely |
| Channel object caching/Reset | 0% | Object creation isn't the bottleneck |
| B-array fast-skip in aux parsing | 0% | Illumina WGS BAMs don't have ML/MM tags |
| Batch sizes > 1024 | Slower | Memory pressure exceeds throughput gain |
| `-O3` / `-march=native` globally | 0% | Already the default; breaks BoringSSL on ARM64 |
| Mixed precision inference | 0% | Only useful if backend handles it automatically |
| ConvertToPb vectorization | 0% | Only 0.12% of CPU time |
| **EfficientNet-B3 model swap** | **3x slower** | Depthwise separable convs + SE blocks have poor CPU GEMM density; FLOPs ≠ speed |
| **MobileNetV2 / any depthwise-separable model** | **Dead end** | Same architecture class as EfficientNet — same CPU GEMM density problem |
| **KMP_AFFINITY tuning** | **30% regression** | `granularity=core,compact,1,0` + `TF_ONEDNN_USE_SYSTEM_ALLOCATOR` cause massive slowdown on ARM64 Neoverse |
| **ONNX ACL ExecutionProvider** | Not worth it | Community-maintained, 16 operators only, fragile version pinning, no pre-built wheels |
| **ONNX dynamic INT8 quantization** | **Broken on ARM64** | `ConvInteger(10)` op not implemented in CPUExecutionProvider — use static INT8 (QDQ format) instead |
| **INT8 on Graviton3+ (vs BF16)** | **No additional gain** | INT8 ONNX static = 0.225s/100, TF+OneDNN BF16 = 0.232s/100 — essentially same speed |
| **TF SavedModel on 32 GB machines** | **OOM kill** | TF allocates ~26 GB RSS for InceptionV3; forking postprocess pushes >32 GB → OOM. Use ONNX backend or 64+ GB instances |
| **OneDNN+ACL on AmpereOne (Siryn)** | **SIGILL** | ACL compiled for Neoverse-N1 triggers SIGILL on AmpereOne in both ME and CV. ISA cap also fails at 16-way concurrency. Use `TF_ENABLE_ONEDNN_OPTS=0`. BF16 source rebuild attempted: fixed 2 bugs (SVE filter, indirect_gemm null ptr) but uncovered 2 more (Conv2D falls to gemm:ref, Grappler remapping crash). BF16 permanently blocked — 4 cascading ACL/OneDNN bugs. See `docs/onednn-ampereone.md` |
| **ONNX on AmpereOne** | **1.96x slower than TF Eigen** | ONNX CPUExecutionProvider much worse than TF Eigen on AmpereOne (0.759 vs 0.387 s/100) — use INT8 ONNX for CV only |
| **fast_pipeline on Oracle A2 32-vCPU** | **<1% improvement, PP broken** | CV stalls on streaming, PP fails on CVO ordering. Sequential 32 shards (489s) is better than fast_pipeline (483s wall, no VCF) |
| **INT8 ONNX CV scaling beyond 16 threads** | **No improvement** | 0.358 s/100 at 16 threads, 0.384 s/100 at 32 threads with 16 shards. ORT GEMM parallelism saturates at ~16 threads |
| **ONNX inter-op parallelism (inter_op_threads > 1)** | **No improvement** | Tested intra/inter splits (28/2, 24/4, 20/6, 16/8, 16/16). All equal or worse than 32/1 baseline. InceptionV3 branches don't benefit from inter-op threading |
| **Direct TFRecord serialization (bypass tf::Example)** | **0%** | Writes protobuf wire format directly via CodedOutputStream. Eliminates one 154KB image copy per example. G3: -0.2%, A2: -0.2% (both noise). The `_message.so` bottleneck is Read/Variant proto ops, not Example serialization |

### What DOES Matter (Profile First)

**Linux ARM64 profiling (Graviton3 c7g.8xlarge, 32 vCPU, 16 shards, chr20):**

Measured via `perf stat -a` and `perf record -a` (system-wide during make_examples Docker run). See `scripts/profile_make_examples.sh`.

**Hardware counters:**

| Metric | glibc (default) | jemalloc | Delta |
|--------|----------------|----------|-------|
| Wall time | 237s | 212s | **-10.7%** |
| IPC (insn/cycle) | 2.61 | 2.73 | +4.6% |
| L1-dcache miss rate | 0.92% | 0.78% | **-15.2%** |
| Cache ref miss rate | 0.93% | 0.79% | **-15.1%** |
| Branch misprediction | 1.28% | 1.29% | ~same |

**DSO-level CPU breakdown (`perf report --sort=dso --no-children`, Self%):**

| DSO | Default | jemalloc | Category |
|-----|---------|----------|----------|
| `_message.so` (protobuf upb) | **29.4%** | 28.9% | Protobuf serialization |
| `libtensorflow_cc.so.2` | **24.8%** | 25.7% | TF small model inference |
| `libc.so.6` | **18.1%** | **4.4%** | malloc/free + libc |
| `python3.10` | 9.8% | 10.3% | Python interpreter |
| `libz.so.1.3` | **7.9%** | 8.2% | gzip (TFRecord compression) |
| `libtensorflow_framework.so.2` | 3.5% | 3.7% | TF framework |
| `[kernel.kallsyms]` | 3.1% | **9.4%** | Kernel (syscalls, page faults) |
| `libjemalloc.so.2` | — | **4.7%** | jemalloc allocator |
| `libstdc++.so.6` | 1.3% | 0.8% | C++ STL |
| Other | ~2.1% | ~3.9% | numpy, pywrap_tf, ld-linux |

**Key findings (vs macOS reference):**

1. **Protobuf serialization is the #1 hotspot (29.4%).** `_message.so` (Google's upb C protobuf library) dominates. On macOS, Instruments attributed this to calling functions (pileup/realigner). The real cost is proto object creation/serialization for Read, Example, and Variant protos.

2. **TF small model inference is #2 (24.8%).** The WGS small model (CNN for candidate filtering, added in v1.9.0) runs during make_examples. This was not in the macOS profiling baseline. On non-WGS models without small model, this ~25% would shift to pileup/realigner.

3. **malloc/free is exactly 18.1% (matches macOS 18%).** With jemalloc: `libc.so.6` drops from 18.1% → 4.4%, `libjemalloc.so.2` adds 4.7%, net reduction ~9% absolute. jemalloc's per-thread arenas reduce L1 cache misses by 15% and increase IPC by 4.6% — it's better memory locality, not just faster malloc.

4. **gzip compression is 7.9% (vs macOS 17%).** The TFRecord gzip share is diluted by protobuf and small model costs that were absent in the macOS profile. In absolute terms, zlib work is similar.

5. **Kernel overhead jumps 3x with jemalloc (3.1% → 9.4%).** jemalloc uses more mmap/munmap for arena management. Net effect is still positive (wall -10.7%).

6. **The macOS profile categories are misleading for optimization.** The macOS profile (pileup ~33%, Smith-Waterman ~30%) reflected function-level attribution. On Linux, the DSO-level view reveals that protobuf serialization and TF small model inference are the dominant costs — pileup/realigner code generates data, but serializing it costs more CPU than computing it.

**Hardware counters (perf stat, Graviton3, 16 shards, chr20):**

| Metric | glibc (default) | jemalloc | Delta |
|--------|----------------|----------|-------|
| Wall time | 237s | 205s | **-13.5%** |
| IPC | 2.61 | 2.77 | +6.1% |
| L1-dcache miss rate | 0.92% | 0.79% | -14.1% |
| Cache ref miss rate | 0.93% | 0.80% | -14.0% |
| Branch misprediction | 1.28% | 1.31% | +2.3% |

jemalloc's ME speedup is explained by the 14% reduction in L1 cache misses (per-thread arenas reduce cross-thread cache line sharing in malloc metadata). IPC improves 6% as a result. Branch misprediction is unchanged — the benefit is purely memory subsystem.

**Comparison with macOS reference profile:**

| Component | macOS CPU % | Linux ARM64 CPU % | Notes |
|-----------|-------------|-------------------|-------|
| Pileup image generation | ~33% | (inside protobuf + libc) | Pileup creates protos — cost shows up in `_message.so` |
| Smith-Waterman realigner | ~30% | (inside protobuf + libc) | Realigner creates alignment protos |
| Protobuf serialization | (attributed to callers) | **29.4%** | Hidden in macOS profile |
| TF small model inference | (not in macOS baseline) | **24.8%** | New in v1.9.0 WGS model |
| malloc/free overhead | ~18% | **18.1%** | Consistent across platforms |
| TFRecord gzip | ~17% | **7.9%** | Lower on Linux — libz more efficient or less data |
| Python interpreter | (not separated) | **9.8%** | GIL overhead, object creation |
| call_variants (TF inference) | ~83% of total | ~83% of total | Still the overall pipeline bottleneck |

**To re-collect:** `sudo bash scripts/profile_make_examples.sh --data-dir /data --num-shards 16`. See the script for full options. Note: `perf report` on the 1.8 GB recording takes 5-10 minutes per report; function-level symbols are unresolved (Docker overlay strips debug info) but DSO-level is sufficient for categorization.

**Linux ARM64 profiling (Oracle A2 AmpereOne, 32 vCPU / 16 OCPU, 16 shards, chr20):**

Measured with `TF_ENABLE_ONEDNN_OPTS=0` (OneDNN+ACL causes SIGILL on AmpereOne — Eigen fallback).

**Hardware counters:**

| Metric | glibc (default) | jemalloc | Delta |
|--------|----------------|----------|-------|
| Wall time | 256.1s | 200.5s | **-21.7%** |
| IPC (insn/cycle) | 1.72 | 1.77 | +2.9% |
| L1-dcache miss rate | 1.16% | 0.92% | **-20.7%** |
| Cache ref miss rate | 1.16% | 0.92% | **-20.7%** |
| Branch misprediction | 1.62% | 1.75% | +8.0% |

**DSO-level CPU breakdown (`perf report --sort=dso --no-children`, Self%):**

| DSO | Default | jemalloc | Category |
|-----|---------|----------|----------|
| `_message.so` (protobuf upb) | **42.6%** | 49.7% | Protobuf serialization |
| `libc.so.6` | **23.2%** | **4.9%** | malloc/free + libc |
| `python3.10` | 13.2% | 15.5% | Python interpreter |
| `libz.so.1.3` | 9.7% | 11.3% | gzip (TFRecord compression) |
| `libtensorflow_cc.so.2` | 2.9% | 3.6% | TF Eigen inference (no OneDNN) |
| `libtensorflow_framework.so.2` | 2.6% | 2.8% | TF framework |
| `libstdc++.so.6` | 2.2% | 1.3% | C++ STL |
| `[kernel.kallsyms]` | 1.1% | 1.4% | Kernel |
| `libjemalloc.so.2` | — | **6.5%** | jemalloc allocator |
| Other | ~2.5% | ~3.0% | pywrap_tf, numpy, ld-linux |

**Cross-platform comparison (Graviton3 vs Oracle A2):**

| Metric | Graviton3 (Neoverse V1) | Oracle A2 (AmpereOne) | Significance |
|--------|------------------------|----------------------|--------------|
| IPC (default) | **2.61** | 1.72 | G3 has 52% higher IPC — better out-of-order execution for this workload |
| IPC (jemalloc) | **2.73** | 1.77 | jemalloc IPC gain: G3 +4.6%, A2 +2.9% |
| L1 miss (default) | **0.92%** | 1.16% | A2 has 26% higher L1 miss rate |
| L1 miss (jemalloc) | 0.78% | 0.92% | jemalloc reduces: G3 -15%, A2 -21% (A2 benefits more) |
| `libc.so.6` share (default) | 18.1% | **23.2%** | A2 has 28% higher malloc share — confirms larger jemalloc benefit |
| `libc.so.6` share (jemalloc) | 4.4% | 4.9% | Both converge to ~5% with jemalloc |
| `_message.so` share | 29.4% | **42.6%** | Protobuf dominates more on A2 (TF Eigen is lightweight vs OneDNN GEMM) |
| TF inference share | **24.8%** | 5.5% | G3 OneDNN+ACL GEMM kernels are compute-dense; A2 Eigen is lightweight |
| Wall time (default) | **237s** | 256.1s | G3 is 7.5% faster |
| Wall time (jemalloc) | **212s** | 200.5s | A2 catches up with jemalloc (jemalloc wall gain: G3 -10.7%, A2 -21.7%) |

**Key cross-platform findings:**

1. **Oracle A2's malloc contention is 28% higher than Graviton3** (23.2% vs 18.1% of cycles in `libc.so.6`). This quantitatively explains why jemalloc's wall time improvement is ~2x larger on A2 (21.7% vs 10.7%). AmpereOne's cache hierarchy produces more cross-thread contention in glibc's malloc metadata.

2. **jemalloc's L1 cache miss reduction is larger on A2** (20.7% vs 15.2%). Both platforms converge to similar L1 miss rates with jemalloc (~0.78-0.92%), but A2 starts from a worse baseline. Per-thread arenas eliminate the cross-thread cache line sharing that AmpereOne's memory subsystem handles less efficiently.

3. **TF inference profile is radically different.** On Graviton3, OneDNN+ACL GEMM kernels make `libtensorflow_cc.so` the #2 hotspot (24.8%) — compute-dense matrix multiplications. On A2 with Eigen fallback, the same small model inference uses lightweight code paths (5.5% total), shifting the profile toward protobuf (42.6%) and malloc (23.2%). An AmpereOne Docker rebuild with native OneDNN would likely shift 15-20% of cycles from protobuf/malloc into efficient GEMM, improving both IPC and wall time.

4. **AmpereOne IPC (1.72) is 34% lower than Graviton3 (2.61).** This reflects both the Eigen-vs-OneDNN backend difference and AmpereOne's microarchitectural characteristics. The Eigen code paths produce more memory-bound stalls. This IPC gap is the primary reason A2 is slower despite having the same vCPU count.

5. **The AmpereOne Docker rebuild is doubly motivated.** Not just for BF16 BFMMLA support, but because OneDNN+ACL would transform the profile: shifting cycles from memory-bound protobuf/malloc (low IPC) into compute-bound GEMM (high IPC), while simultaneously reducing malloc pressure by using OneDNN's internal memory pools. Projected improvement: 25-35% make_examples speedup on top of jemalloc gains.

### Operational Pitfalls (Docker + Cloud Instances)

These are common issues encountered when running benchmarks on cloud ARM64 instances:

| Issue | Impact | Fix |
|-------|--------|-----|
| **Docker output files owned by root** | Benchmark scripts with `rm -rf` and `set -euo pipefail` crash on re-runs | Always `sudo chown -R ubuntu:ubuntu /data/output/` before re-running, or run Docker with `--user $(id -u):$(id -g)` |
| **Repacking `.zip` Python apps breaks native `.so` files** | `python3 -m zipfile -c` includes `.so` files that become invalid; call_variants crashes with "invalid ELF header" | Never repack zip-based Python apps. Instead, copy the entire `.zip` from a working image, or build a proper Docker layer |
| **`/data/flags/` or `/data/output/` missing on fresh instance** | Benchmark scripts crash at `mkdir -p` if parent `/data/` is root-owned | Create data dirs with `sudo mkdir -p /data/{flags,output,...} && sudo chown -R ubuntu:ubuntu /data` |
| **GCS reference download fails with `curl -sO`** | Returns XML error page instead of file (redirect issue) | Use `wget -q` instead of `curl -sO` for GCS public data |
| **AWS vCPU limit blocks larger instances** | Cannot launch c7g.8xlarge/c8g.8xlarge without quota increase (resolved — quota increased to 32) | Request via AWS Console (IAM user needs `servicequotas:RequestServiceQuotaIncrease` permission for CLI) |
| **`--memory=28g` Docker flag required** | TF allocator grabs ALL available RAM without this limit | Always set `--memory=28g` (or appropriate limit) |
| **Docker entrypoint sets OMP vars** | `docker_entrypoint.sh` sets `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES` — leaks to all children in fast_pipeline | Override entrypoint with `--entrypoint /bin/bash` and unset OMP vars before fast_pipeline |
| **fast_pipeline at 16 vCPU is SLOWER than sequential** | CPU contention between concurrent ME+CV degrades CV rate from 0.232 to 0.291 s/100 (25% slower), total 693s vs 487s | fast_pipeline only benefits with 32+ vCPU where ME and CV can use separate cores |
| **ghcr.io Docker pull requires auth on fresh instances** | `docker pull ghcr.io/antomicblitz/...` fails without login on new instances | Run `echo "$PAT" \| docker login ghcr.io -u antomicblitz --password-stdin` first |
| **GCS reference genome URL changed** | `genomics-public-data` bucket returns 404 for `.fna.gz` | Use `deepvariant/case-study-testdata` bucket instead |
| **Docker image ONNX model is FP32, not INT8** | `/opt/models/wgs/model.onnx` (84 MB) is the FP32 ONNX model converted by tf2onnx. INT8 static model (21 MB) must be mounted from host at `/data/model_int8_static.onnx`. Using the wrong model gives 2x slower CV (0.44 vs 0.22 s/100). | Always pass `--onnx_model=/data/model_int8_static.onnx` for INT8, not the in-image model |
| **postprocess_variants missing `--regions` with parallel CV** | Without `--regions`, postprocess partitions across the whole genome. A pre-existing CVO sanity check bug in `merge_predictions()` (line 1058) crashes the chr20 partition worker, but `multiprocessing.Pool.apply_async` silently swallows the exception — producing only ~125 RefCall variants instead of 207,799. | Always pass `--regions` and `--small_model_cvo_records` to postprocess when running parallel CV outside `run_deepvariant` |

**Important: Monitor instance setup scripts.** When spinning up new cloud instances and running setup scripts (Docker install, data download, image pull), check for errors every 60 seconds for the first 5 minutes. Many failures (wrong URLs, auth issues, permission errors) happen within the first minute — waiting 10+ minutes before checking wastes time and money.

***

## Data Transfer: Unified vs Discrete Memory

Apple Silicon's unified memory makes CPU→GPU data transfer free. On Linux ARM64:

| Platform | Memory Model | Transfer Cost | Mitigation |
|----------|-------------|---------------|------------|
| Graviton + no GPU | N/A | N/A | CPU-only; ONNX+ACL is the "accelerator" |
| Jetson Orin | Unified | Free | No mitigation needed |
| RK3588 (Mali G610) | Shared | Minimal | SoC shares memory — similar to Apple Silicon |
| Graviton + attached GPU | Discrete | Significant | Batch larger; double-buffer; overlap transfer with compute |

***

## Release Milestones

### v0.1.0 — CPU-Only ARM64 Build (COMPLETE)
- [x] DeepVariant builds and runs on ARM64 (Hetzner CAX31 + GCP t2a-standard-8)
- [x] All C++ optimizations ported (haplotype cap, flat buffer, query cache)
- [x] Pipeline validated — 103,516 variants on chr20:1-30M
- [x] Docker images on ghcr.io (`deepvariant-arm64:latest`, `deepvariant-arm64:optimized`)
- [x] Benchmark: chr20:1-30M in 12m57s on 8-core Ampere (optimized TF)
- [x] ONNX inference path implemented (`--use_onnx` flag)

### v0.2.0 — Inference Acceleration (COMPLETE — released as v1.9.0-arm64.4)
- [x] ONNX model conversion and integration (complete, but slower than TF+OneDNN on N1)
- [x] TF OneDNN warmup (~4% call_variants improvement)
- [x] EfficientNet-B3 training pipeline built and model trained — **DEAD END: 3x slower on CPU**
- [x] KMP_AFFINITY + system allocator — **HARMFUL** (30% regression, reverted)
- [x] Graviton3+ BF16: **1.61x call_variants speedup**, zero accuracy loss (measured on c7g.4xlarge)
- [x] INT8 static quantization: 2.3x over ONNX FP32, matches BF16 speed (0.225 vs 0.232 s/100)
- [x] INT8 accuracy: SNP F1=0.9978, INDEL F1=0.9962 (matches BF16, passes gate)
- [x] OMP env var scoping: per-subprocess env dicts in run_deepvariant.py (benchmarked: 307→299s ME, 2.6% gain)
- [x] Stratified region validation: INT8 passes all GIAB regions (homopolymers, STRs, segdups)
- [x] Graviton4 (c8g) benchmark: INT8 ONNX 366s (0.197s/100), ONNX FP32 602s, TF BF16 OOM on 32 GB
- [x] Oracle A2 (AmpereOne) benchmark: INT8 ONNX 542s ($2.32/genome), TF Eigen 629s ($2.49/genome), BF16 SIGILL confirmed
- [x] fast_pipeline on Linux ARM64: works but **42% SLOWER at 16 vCPU** (693s vs 487s sequential — CPU contention)
- [x] Oracle A2 32-vCPU scaling: sequential 32 shards = 489s, fast_pipeline broken (PP fails)
- [x] ONNX inter-op parallelism sweep: no improvement (intra-op GEMM dominates)
- [x] jemalloc: verified on Graviton3 (2 runs) and Oracle A2 (4 runs). ME -14-17%, wall -7-9%.
- [x] Graviton3/4 32 vCPU benchmarks: c7g.8xlarge 283s, c8g.8xlarge 232s. CV floor confirmed.
- [x] **Parallel call_variants (4-way):** CV 2.0-2.5x speedup on all 3 platforms. Graviton4 61s (N=3), Graviton3 74s (N=4), Oracle A2 114s (N=2). Variant counts match exactly. See `scripts/benchmark_parallel_cv.sh`.
- [ ] Graviton4 BF16 full pipeline on c8g.8xlarge (64 GB) — TF OOM on 32 GB
- [x] ~~Oracle A2 ACL rebuild for AmpereOne BF16~~ — **BLOCKED**: 4 cascading ACL/OneDNN bugs. See `docs/onednn-ampereone.md`

### v0.3.0 — GPU/NPU Acceleration (Future, Optional)
- [ ] Jetson Orin CUDA path working
- [ ] (Optional) TFLite + OpenCL for Mali
- [ ] (Optional) RKNN integration for RK3588 NPU

### v1.0.0 — Production Release
- [ ] Upstream PR to google/deepvariant with ARM64 support
- [ ] Multi-arch Docker image (amd64 + arm64)
- [x] Published benchmark results on Graviton3, Graviton4, Oracle A1, Oracle A2
- [x] Stratified GIAB validation on all backends (INT8, BF16, FP32)
- [x] Documentation and quickstart guide

***

## Cost Projections (Measured + Projected)

| Platform | vCPU | $/hr | chr20 Time | WGS Time | Cost/Genome | Source |
|----------|------|------|-----------|----------|-------------|--------|
| Google x86 (official) | 96 | $3.81 | — | ~1.3 hr | **$5.01** | [Official](https://github.com/google/deepvariant/blob/r1.9/docs/metrics.md) |
| **Graviton3 FP32** | 16 | $0.58 | 9m41s | ~7.8 hr | **$4.50** | Measured (2-run avg 582s)* |
| **Graviton3 BF16** | 16 | $0.58 | 8m06s | ~6.5 hr | **$3.77** | Measured (2-run avg 487s)* |
| **Graviton3 INT8 ONNX** | 16 | $0.58 | ~8m27s | ~6.8 hr | **$3.92** | Measured (3-run avg 507s) |
| **Graviton4 INT8 ONNX** | 16 | $0.68 | 6m06s | ~4.9 hr | **$3.33** | Measured (2-run avg 366s)* |
| **Graviton4 ONNX FP32** | 16 | $0.68 | 10m02s | ~8.0 hr | **$5.47** | Measured (2-run avg 602s)* |
| **Graviton4 BF16** (standalone CV) | 16 | $0.68 | ~8m32s | ~6.8 hr | ~**$4.65** | Partial (CV measured, ME from ONNX run)* |
| **Oracle A2 INT8 ONNX** | 16 vCPU (8 OCPU) | $0.32 | 9m44s | ~7.8 hr | **$2.49** | Measured (4-run avg 584s, OneDNN OFF) |
| **Oracle A2 TF Eigen FP32** | 16 vCPU (8 OCPU) | $0.32 | 10m29s | ~8.4 hr | **$2.69** | Measured (2-run avg 629s, Eigen: OneDNN SIGILL)* |
| **Graviton3 BF16 + jemalloc** | 16 | $0.58 | 7m23s | ~5.9 hr | **$3.43** | Measured (2-run avg 443s, jemalloc ON)* |
| **Oracle A2 INT8 + jemalloc** | 16 vCPU (8 OCPU) | $0.32 | 9m04s | ~7.3 hr | **$2.32** | Measured (4-run avg 544s, OneDNN OFF) |
| **Oracle A1 INT8 ONNX** | 16 (16 OCPU) | $0.16 | 8m29s | ~6.8 hr | **$1.09** | Measured (4-run avg 509s, OneDNN OFF) |
| **Oracle A1 INT8 + jemalloc** | 16 (16 OCPU) | $0.16 | 8m06s | ~6.5 hr | **$1.04** | Measured (3-run avg 486s, OneDNN OFF) |
| **Oracle A1 TF Eigen FP32** | 16 (16 OCPU) | $0.16 | 12m15s | ~9.8 hr | **$1.57** | Measured (1 run 735s, OneDNN OFF)* |
| **Oracle A1 ONNX FP32** | 16 (16 OCPU) | $0.16 | 13m48s | ~11.1 hr | **$1.77** | Measured (2-run avg 828s)* |
| **Graviton3 BF16+jemalloc 32 shards** | 32 | $1.15 | 4m43s | ~3.8 hr | **$4.35** | Measured (2-run avg 283s, c7g.8xlarge)* |
| **Graviton4 BF16+jemalloc 32 shards** | 32 | $1.36 | 3m52s | ~3.1 hr | **$4.22** | Measured (2-run avg 232s, c8g.8xlarge)* |
| **Graviton4 INT8+jemalloc 32 shards** | 32 | $1.36 | 3m53s | ~3.1 hr | **$4.24** | Measured (2-run avg 233s, c8g.8xlarge)* |
| **Oracle A2 INT8+jemalloc 32 shards** | 32 (16 OCPU) | $0.64 | 6m58s | ~5.6 hr | **$3.57** | Measured (sequential, wall=418s)* |
| **Graviton4 4-way parallel CV** | 32 | $1.36 | ~2m52s | ~2.3 hr | **~$3.13** | Projected (ME=100+CV=61+PP=6=~172s) |
| **Graviton3 4-way parallel CV** | 32 | $1.15 | ~3m38s | ~2.9 hr | **~$3.35** | Projected (ME=131+CV=74+PP=8=~218s) |
| **Oracle A2 4-way parallel CV** | 32 (16 OCPU) | $0.64 | ~4m10s | ~3.3 hr | **~$2.14** | Projected (ME=113+CV=114+PP=10=~250s) |
| Graviton3 fast_pipeline BF16 16 vCPU | 16 | $0.58 | 11m33s | ~9.3 hr | $5.37 | Measured (2-run avg 693s)* — **42% SLOWER** |

*All $/genome use formula: `chr20_wall_s × 48.1 / 3600 × $/hr`. Oracle A1 pricing: $0.01/OCPU/hr — 16 OCPU = $0.16/hr. Oracle A2 pricing: $0.04/OCPU/hr — 16-vCPU rows priced at 8 OCPU ($0.32/hr), 32-vCPU rows at 16 OCPU ($0.64/hr). Measured values averaged over N runs (N noted in Source column). Rows marked with \* have N<4 runs (wider confidence interval). Projected values marked with ~. WGS time extrapolation has ~15-20% uncertainty. Wall time includes ~4-5s Docker startup/inter-stage overhead beyond ME+CV+PP sum. INT8 matches BF16 speed on Graviton3 (no additional gain); INT8 is for non-BF16 platforms. fast_pipeline at 16 vCPU is slower due to CPU contention — needs 32+ vCPU.*

*Graviton4 BF16 full pipeline OOM-killed on 32 GB (c8g.4xlarge). TF SavedModel uses ~26 GB RSS; forking postprocess pushes total >32 GB. Standalone CV rate measured at 0.328 s/100 (BF16). ME time (232s) taken from ONNX run. Needs c8g.8xlarge (64 GB) for full TF BF16 pipeline.*

*Oracle A2 (AmpereOne/Siryn) uses TF Eigen fallback or ONNX because OneDNN+ACL (compiled for Neoverse-N1) causes SIGILL on AmpereOne's ISA. INT8 ONNX is the fastest working backend. 32-vCPU (16 OCPU) scaling: ME benefits from 32 shards (167s vs 297s), but INT8 ONNX CV does not scale beyond ~16 ORT threads (0.358 s/100 at both 16 and 32 threads). fast_pipeline tested on Oracle A2 32-vCPU: <1% wall improvement, PP broken (CVO ordering issue) — not worth pursuing.*

*jemalloc (via `DV_USE_JEMALLOC=1`): Verified benefit on both ARM64 platforms. Graviton3: ME -13.8%, CV -3.2%, wall -9.0% (2 runs each). Oracle A2: ME -17.0%, CV within noise, wall -6.9% (4 runs each). ME improvement is the dominant factor (jemalloc's per-thread arenas reduce malloc contention in make_examples' C++ allocations). CV improvement is small/zero — ONNX Runtime and TF have their own internal allocators that bypass glibc malloc.*

*Parallel call_variants: 4-way parallel CV splits 32 ME shards across 4 call_variants workers in separate Docker containers (8 shards each, OMP_NUM_THREADS=nproc/4). Measured CV speedups: Graviton4 2.10x (128→61s, N=3, σ=0), Graviton3 1.90x (141→74s, N=4), Oracle A2 2.47x (283→114s, N=2). All variant counts match sequential baseline exactly (207,799). Projected wall times combine measured sequential ME+PP with measured parallel CV. See `scripts/benchmark_parallel_cv.sh`. Only works with ONNX backend (~3 GB/worker); TF SavedModel (~26 GB/worker) would OOM.*

***

## Development Environment

### Remote ARM64 Development (VS Code Remote SSH)

All development happens on a remote ARM64 instance via VS Code Remote SSH. You edit code locally on your Apple Silicon Mac, but all compilation, Docker builds, and benchmarks run on the remote ARM64 box.

**Active instance:** Hetzner CAX31 (8 vCPU Ampere Altra Neoverse-N1, 16 GB RAM, 150 GB NVMe, ~$14/mo)
- **IP:** `178.104.17.88`
- **SSH Host:** `hetzner-arm64` (configured in `~/.ssh/config`)
- **OS:** Ubuntu 24.04.3 LTS (aarch64)
- **Repo path:** `/root/deepvariant-linux-arm64`
- **CPU features:** NEON, CRC32, AES, ASIMD (no BF16 — that's Graviton3+/Neoverse V1)

**SSH config** (`~/.ssh/config` on Mac):
```
Host hetzner-arm64
    HostName 178.104.17.88
    User root
    IdentityFile ~/.ssh/id_ed25519
```

**VS Code connection:** `Cmd+Shift+P` → "Remote-SSH: Connect to Host" → `hetzner-arm64` → Open `/root/deepvariant-linux-arm64`

**Workflow:**
- Edit files in VS Code on your Mac (feels fully local)
- Build in the VS Code integrated terminal (runs on Hetzner ARM64)
- Benchmark natively — the instance IS the target architecture
- **Power off** the instance from Hetzner console when not in use (stops compute charges, keeps disk)

**Installed on remote:**
- Docker (verified with `arm64v8/ubuntu:22.04`)
- Build essentials (gcc, g++, make, pkg-config, zip, unzip)
- Libraries: Boost, zlib, OpenSSL, libbz2, liblzma
- Python 3 + pip + venv
- TF environment variables auto-loaded via `/etc/profile.d/deepvariant-arm64.sh`
- Upstream remote: `git remote upstream` → `https://github.com/google/deepvariant.git`

***

## File Structure

```
deepvariant-linux-arm64/
├── CLAUDE.md                           # This file
├── Dockerfile                          # Upstream x86 Dockerfile (reference)
├── Dockerfile.arm64                    # ARM64 Linux Docker build
├── settings.sh                         # Upstream x86 settings (reference)
├── settings_arm64.sh                   # ARM64 settings (no -march=corei7, OneDNN+ACL, BF16)
├── build-prereq.sh                     # Upstream x86 build prereqs (reference)
├── build-prereq-arm64.sh              # ARM64 build prereqs (ARM64 Bazel, system Boost)
├── build_release_binaries.sh           # Upstream x86 build (reference)
├── build_release_binaries_arm64.sh    # ARM64 build (aarch64-opt paths)
├── .github/
│   └── workflows/
│       └── arm64-build.yml            # CI: QEMU build + self-hosted accuracy test
├── scripts/
│   ├── setup_graviton.sh              # One-command ARM64 instance setup
│   ├── benchmark_arm64.sh             # HG003 chr20 benchmark (wall time + system info)
│   ├── validate_accuracy.sh           # hap.py accuracy validation (SNP/INDEL F1 targets)
│   ├── convert_model_onnx.py          # TF SavedModel → ONNX conversion (Phase 2)
│   └── convert_model_tflite.py        # TF SavedModel → TFLite (Phase 4)
├── deepvariant/                        # Upstream source (modifications for ARM64 TBD)
│   ├── call_variants.py               # Future: --use_onnx flag (Phase 2)
│   ├── fast_pipeline.cc               # Future: remove Homebrew Boost paths (Phase 1)
│   └── ...                            # All upstream DeepVariant source
├── models/                             # Future: retrained/converted models
│   ├── efficientnet_b3/               # Phase 3: EfficientNet-B3 weights
│   └── onnx/                          # Phase 2: ONNX converted models
└── docs/                               # Upstream docs + future ARM64 docs
```

***

## Key Technical Decisions

1. **TF 2.14.1 (not 2.13.1) for ARM64 Linux.** The macOS port uses TF 2.13.1 because `tensorflow-macos` is pinned there. Linux ARM64 should use the latest official wheel (2.14.1) which includes OneDNN+ACL optimizations. The C++ headers must match — rebuild against v2.14.1 tag.[7]

2. **Native NEON, not sse2neon.** The macOS port compiles libssw with `sse2neon.h` because the upstream code is SSE2. On Linux ARM64, prefer native NEON intrinsics (available in newer libssw forks) or verify the sse2neon translation layer compiles correctly with GCC/Clang on aarch64-linux.

3. **Boost from system packages, not Homebrew.** The macOS port uses `/opt/homebrew/include`. On Linux: `apt install libboost-dev` and update `WORKSPACE`/`BUILD` paths accordingly.

4. **`int64_t` type handling.** On Linux ARM64, `int64_t` = `long` (same as x86_64 Linux). The macOS port fixed `int64_t` = `long long` mismatches. These patches may not be needed on Linux ARM64 but should be verified — use `static_cast<int64_t>()` universally for safety.

5. **TF+OneDNN is the primary inference backend.** Benchmarking showed ONNX Runtime CPUExecutionProvider is 24% slower than TF+OneDNN on Neoverse-N1. The ONNX ACL ExecutionProvider is community-maintained (16 operators, fragile builds) and not worth pursuing. On Graviton3+, INT8 ONNX matches BF16 TF+OneDNN speed (0.225 vs 0.232 s/100). The `--use_onnx` flag is implemented as an opt-in alternative, and is required for parallel CV (low memory footprint).

6. **BF16 fast math on Graviton3+.** Graviton3 and newer support BF16 operations. Enable via `ONEDNN_DEFAULT_FPMATH_MODE=BF16`. Accuracy validated: BF16 = FP32 (SNP F1=0.9977, INDEL F1=0.9961 — identical). 38% call_variants speedup on Graviton3, zero accuracy loss.

7. **EfficientNet-B3 is NOT faster than InceptionV3 on CPU.** Benchmarked at 0.31x the speed of InceptionV3 (3x slower) despite 3.2x fewer FLOPs. Depthwise separable convolutions and squeeze-and-excitation blocks produce many small operations with poor data reuse, while InceptionV3's dense Conv2D operations map to single large GEMM calls. FLOP count does not predict CPU inference speed — kernel efficiency and memory access patterns matter more. See `TRAINING_EXPERIMENT.md`.