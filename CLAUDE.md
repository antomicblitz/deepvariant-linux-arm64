# CLAUDE.md — DeepVariant Linux ARM64 with Hardware Acceleration

## Project Overview

This project ports DeepVariant to Linux ARM64 with hardware-accelerated inference, creating an open-source variant caller that runs on cheap ARM cloud instances (Graviton, Ampere) and embedded ARM boards (RK3588). The goal is a viable alternative to proprietary solutions like Sentieon DNAscope — possibly somewhat slower, but dramatically cheaper per genome.

**Upstream repo:** [google/deepvariant](https://github.com/google/deepvariant) v1.9.0 (Bazel 5.3.0, TF 2.13.1)
**macOS ARM64 reference:** [antomicblitz/deepvariant-macos-arm64-metal](https://github.com/antomicblitz/deepvariant-macos-arm64-metal) — all C++ optimizations transfer directly
**License:** BSD-3-Clause (same as upstream)

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
| Oracle Ampere A1 | Altra/Altra Max | None (CPU-only) | Discrete | Ultra-cheap cloud |
| NVIDIA Jetson Orin | Cortex-A78AE | Ampere GPU (CUDA) | Unified | Edge/on-premise |
| Rockchip RK3588 | Cortex-A76/A55 | Mali G610 (OpenCL) + 6 TOPS NPU | Shared | Low-resource labs |

***

## Architecture Decision: Inference Backend

The `call_variants` step runs Inception V3 (23.9M params) CNN inference on 100×221×7 uint8 pileup images — standard image classification with no custom ops. This is the bottleneck without GPU (83% of CPU-only wall time).

### Backend Comparison (Updated with Benchmark Results)

| Backend | Target Hardware | Maturity | Measured/Expected Speedup | Effort |
|---------|----------------|----------|--------------------------|--------|
| **TF OneDNN+ACL** | All ARM64 CPUs | Production | **Baseline winner** on Neoverse-N1[7] | Low |
| **ONNX Runtime (CPUExecutionProvider)** | All ARM64 CPUs | Production | **24% slower** than TF+OneDNN (measured)[6] | Done |
| **ONNX Runtime (MLAS BF16)** | Graviton3+ (Neoverse V1/V2 only) | Production | 35-65% over ONNX CPU (config flag only) | Config |
| **ONNX Runtime (ACL EP, source build)** | All ARM64 CPUs | Community-maintained | Unknown; fragile build, 16 ops only | High |
| **EfficientNet-B3 (model swap)** | Any backend | **DEAD END** | **3x slower** on CPU (measured) — depthwise convs have poor GEMM density | Done |
| **Apache TVM + OpenCL** | Mali G610/G710 GPUs | Alpha for Mali Valhall | Poor real results on G610[8][9] | High |
| **CUDA (Jetson)** | Jetson Orin only | Production | 3-5x | Low (TF native) |

### Recommended Strategy: Revised Tiered Approach

**Phase 1 (CPU-only, COMPLETE):** TF aarch64 wheel + OneDNN+ACL + C++ optimizations (haplotype cap, ImageRow flat buffer, query cache). Benchmarked at 12m57s on chr20:1-30M (GCP t2a-standard-8).

**Phase 2 (TF tuning):**
- **2A: TF OneDNN tuning** — `get_concrete_function()` for Grappler BatchNorm folding, `KMP_AFFINITY`, `TF_ONEDNN_USE_SYSTEM_ALLOCATOR`. Low effort, 10-25% call_variants gain.
- **~~2B: EfficientNet-B3~~** — **DEAD END.** Benchmarked at 0.31x speed of InceptionV3 on CPU (3x slower despite 3.2x fewer FLOPs). Depthwise separable convolutions and SE blocks have poor computational density on CPUs compared to InceptionV3's dense GEMM-friendly convolutions. See `TRAINING_EXPERIMENT.md` for full details.

**Phase 2 ONNX status:** The `--use_onnx` flag is implemented and works. On Neoverse-N1 (no BF16), ONNX CPUExecutionProvider is slower than TF+OneDNN. On Graviton3+ (BF16), enabling `mlas.enable_gemm_fastmath_arm64_bfloat16` may reverse this — needs benchmarking on actual Graviton hardware.

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

ONNX integration is **complete** (`--use_onnx` flag, model conversion, Docker images). Benchmarking revealed TF+OneDNN outperforms ONNX CPUExecutionProvider on Neoverse-N1. Strategy revised to focus on TF tuning + EfficientNet-B3.

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

### 2.2 TF OneDNN Tuning (TODO)

**`call_variants.py`** — Add `get_concrete_function()` after SavedModel load:
```python
if use_saved_model and not use_onnx:
    concrete_fn = model.signatures['serving_default'].get_concrete_function(
        tf.TensorSpec(shape=[None, 100, 221, num_channels], dtype=tf.float32)
    )
```
This triggers Grappler BatchNorm folding and persistent kernel reuse (5-15% speedup).

**`docker_entrypoint.sh`** — Add missing environment variables:
```bash
export KMP_AFFINITY="${KMP_AFFINITY:-granularity=core,compact,1,0}"
export TF_ONEDNN_USE_SYSTEM_ALLOCATOR=1
```

**ONNX BF16 on Graviton3+** — Add to ONNX session setup:
```python
sess_options.add_session_config_entry(
    "mlas.enable_gemm_fastmath_arm64_bfloat16", "1"
)
```

**Diagnostic:** Run `DNNL_VERBOSE=1` once to confirm ACL kernels are active in oneDNN.

### 2.3 EfficientNet-B3 Model (DEAD END)

**Attempted and measured.** Despite 3.2x fewer FLOPs, EfficientNet-B3 is **3x slower** than InceptionV3 on CPU inference. The full training pipeline was built and validated (see `TRAINING_EXPERIMENT.md`), but the speed result kills this approach.

**Root cause:** EfficientNet's depthwise separable convolutions and squeeze-and-excitation blocks produce many small kernel launches with poor data reuse, while InceptionV3's dense `Conv2D` operations map to single large GEMM calls with high arithmetic intensity. FLOPs ≠ speed on CPUs.

| Model | FLOPs | Params | CPU img/s (batch=128) | Relative Speed |
|-------|-------|--------|-----------------------|----------------|
| InceptionV3 | 5.7G | 23.9M | 591 | **1.0x** |
| EfficientNet-B3 | 1.8G | 12.3M | 185 | **0.31x** |

### 2.4 Expected Cumulative Performance

| Optimization | Impact | Cumulative (chr20:1-30M, 8-core) |
|-------------|--------|----------------------------------|
| C++ opts (done) | make_examples -10% | 12m57s |
| TF tuning (concrete_fn + env vars) | call_variants -10-25% | ~11m-12m |
| ~~EfficientNet-B3~~ | ~~call_variants -40-50%~~ | **DEAD END: 3x slower** |
| On Graviton3+ with BF16 | additional -30-50% on inference | ~9-10m |

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
| Protobuf Arena allocation | 0% | Allocation isn't the bottleneck for Read objects |
| TFRecord gzip removal | 0% in fast pipeline | Shared memory bypasses TFRecord entirely |
| Channel object caching/Reset | 0% | Object creation isn't the bottleneck |
| B-array fast-skip in aux parsing | 0% | Illumina WGS BAMs don't have ML/MM tags |
| Batch sizes > 1024 | Slower | Memory pressure exceeds throughput gain |
| `-O3` / `-march=native` globally | 0% | Already the default; breaks BoringSSL on ARM64 |
| Mixed precision inference | 0% | Only useful if backend handles it automatically |
| ConvertToPb vectorization | 0% | Only 0.12% of CPU time |
| **EfficientNet-B3 model swap** | **3x slower** | Depthwise separable convs + SE blocks have poor CPU GEMM density; FLOPs ≠ speed |

### What DOES Matter (Profile First)

From macOS profiling:

| Component | CPU % | Action |
|-----------|-------|--------|
| Smith-Waterman realigner | ~30% | Already NEON-vectorized; haplotype cap reduces alignment count |
| Pileup image generation | ~33% wall | ImageRow flat buffer optimization (-8.4%) |
| malloc/free overhead | ~18% | Distributed — no single hotspot; flat buffer helps |
| TFRecord gzip | ~17% | Eliminated by fast pipeline shared memory |
| call_variants (TF inference) | ~83% of total | THE bottleneck — all GPU/ONNX work targets this |

**On Linux ARM64:** Profile with `perf stat` for L1 cache miss rates and `perf record` for hotspot identification. The hot path is the same on any architecture.

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

### v0.2.0 — Inference Acceleration (In Progress)
- [x] ONNX model conversion and integration (complete, but slower than TF+OneDNN on N1)
- [x] EfficientNet-B3 training pipeline built and model trained — **DEAD END: 3x slower on CPU**
- [ ] TF OneDNN tuning (concrete_fn, KMP_AFFINITY, system allocator)
- [ ] Graviton3+ benchmark with BF16 (ONNX MLAS or TF OneDNN)
- [ ] INT8 quantization of InceptionV3

### v0.3.0 — GPU/NPU Acceleration (Future, Optional)
- [ ] Jetson Orin CUDA path working
- [ ] (Optional) TFLite + OpenCL for Mali
- [ ] (Optional) RKNN integration for RK3588 NPU

### v1.0.0 — Production Release
- [ ] Upstream PR to google/deepvariant with ARM64 support
- [ ] Multi-arch Docker image (amd64 + arm64)
- [ ] Published benchmark results on Graviton4, Ampere A1, RK3588
- [ ] hap.py validation matching Google's published accuracy numbers
- [ ] Documentation and quickstart guide

***

## Cost Projections

| Platform | Instance | $/hr | Est. chr20 Time | Est. Cost/chr20 | Est. Cost/Genome |
|----------|----------|------|----------------|----------------|-----------------|
| GCP n2-standard-16 (x86, baseline) | 16 vCPU | $0.76 | 14m 28s | $0.18 | $8.70 |
| AWS Graviton3 (c7g.4xlarge, 16 vCPU) | 16 vCPU | $0.48 | ~12 min (est.) | $0.10 | $4.60 |
| AWS Graviton4 (c8g.4xlarge, 16 vCPU) | 16 vCPU | $0.54 | ~9 min (est.) | $0.08 | $3.85 |
| Oracle Ampere A1 (16 OCPU) | 16 OCPU | $0.16 | ~14 min (est.) | $0.04 | $1.73 |
| RK3588 board ($100 hardware) | 8 cores | $0 (electricity) | ~45 min (est.) | ~$0.01 | ~$0.50 |

*Genome estimates: chr20 × 48.1 scaling factor. Graviton estimates assume OneDNN+ACL+BF16 optimizations. Oracle A1 estimate assumes ONNX Runtime + ACL. All estimates are pre-EfficientNet-B3 model optimization.*

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

5. **TF+OneDNN is the primary inference backend.** Benchmarking showed ONNX Runtime CPUExecutionProvider is 24% slower than TF+OneDNN on Neoverse-N1. The ONNX ACL ExecutionProvider is community-maintained (16 operators, fragile builds) and not worth pursuing. ONNX may outperform TF on Graviton3+ via MLAS BF16 kernels — needs testing. The `--use_onnx` flag is implemented as an opt-in alternative.

6. **BF16 fast math on Graviton3+.** Graviton3 and newer support BF16 operations. Enable via `DNNL_DEFAULT_FPMATH_MODE=BF16`. Validate that variant calling accuracy is unaffected (BF16 has 8 mantissa bits vs FP32's 23 — sufficient for Inception V3/EfficientNet classification but must be verified).[7]

7. **EfficientNet-B3 is NOT faster than InceptionV3 on CPU.** Benchmarked at 0.31x the speed of InceptionV3 (3x slower) despite 3.2x fewer FLOPs. Depthwise separable convolutions and squeeze-and-excitation blocks produce many small operations with poor data reuse, while InceptionV3's dense Conv2D operations map to single large GEMM calls. FLOP count does not predict CPU inference speed — kernel efficiency and memory access patterns matter more. See `TRAINING_EXPERIMENT.md`.