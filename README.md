# DeepVariant — Linux ARM64 Native Build

[![release](https://img.shields.io/badge/base-v1.9.0-green?logo=github)](https://github.com/google/deepvariant/releases)
[![platform](https://img.shields.io/badge/platform-Linux%20ARM64-blue?logo=linux)](https://en.wikipedia.org/wiki/AArch64)
[![build](https://img.shields.io/badge/first%20native%20ARM64%20build-brightgreen)](#what-this-fork-does)
[![accuracy](https://img.shields.io/badge/accuracy-validated%20on%20GIAB-success)](#accuracy-validation)

There is no official Linux ARM64 build of DeepVariant. The official Docker image is x86-only and uses SSE4/AVX instructions that do not exist on ARM. This fork patches the Bazel build system, htslib, and libssw to compile natively on aarch64, producing the first working DeepVariant Docker image for ARM64 Linux — enabling deployment on AWS Graviton, Oracle Ampere A1, Hetzner CAX, and other ARM64 cloud instances.

> **What this fork does, in order of significance:**
>
> **(1) Makes DeepVariant build and run natively on Linux ARM64.** No emulation, no QEMU, no Rosetta — native aarch64 binaries compiled with GCC 13 on Ubuntu 24.04.
>
> **(2) Enables deployment on ARM64 cloud instances** (Graviton, Ampere A1, Hetzner CAX) which offer lower per-core pricing than x86 instances.
>
> **(3) Provides a Docker image** that works out of the box on any ARM64 Linux host — same `run_deepvariant` interface as the official x86 image.
>
> **(4) BF16 fast math on Graviton3+** delivers 1.61x faster CNN inference with zero accuracy loss.

---

## Benchmark Results

All benchmarks run on **GIAB HG003, full chr20**, averaged over 2 runs. Accuracy validated with `rtg vcfeval` against GIAB v4.2.1 truth sets.

### call_variants Inference Rate

![call_variants inference rate](docs/figures/call_variants_rate.png)

| Platform | vCPUs | Config | call_variants Rate | Full chr20 Wall Time |
|----------|-------|--------|-------------------|---------------------|
| GCP t2a-standard-8 (Neoverse-N1) | 8 | FP32 | 0.880 s/100 | 12m57s |
| GCP t2a-standard-16 (Neoverse-N1) | 16 | FP32 | 0.512 s/100 | 7m22s |
| **AWS Graviton3** (c7g.4xlarge) | 16 | FP32 | 0.379 s/100 | 9m41s |
| **AWS Graviton3** (c7g.4xlarge) | 16 | **BF16** | **0.232 s/100** | **8m06s** |

> Graviton3 BF16 delivers **1.61x faster call_variants** (38% reduction) with zero accuracy loss. The Graviton3 Neoverse V1 microarchitecture is also 26% faster than Neoverse-N1 at FP32 baseline.

### Pipeline Breakdown (Graviton3)

![Graviton3 wall time breakdown](docs/figures/graviton3_wall_time.png)

| Stage | FP32 | BF16 | Speedup |
|-------|------|------|---------|
| make_examples | 255s | 278s | — |
| call_variants | 298s | 185s | **1.61x** |
| postprocess_variants | 29s | 24s | — |
| **Total wall time** | **582s (9m42s)** | **487s (8m07s)** | **1.20x** |

### Accuracy Validation

![Accuracy FP32 vs BF16](docs/figures/accuracy_fp32_vs_bf16.png)

BF16 fast math produces **identical accuracy** to FP32. Validated on GIAB HG003 chr20 with `rtg vcfeval`:

| Metric | FP32 | BF16 | Target |
|--------|------|------|--------|
| **SNP F1** | 0.9974 | 0.9974 | >= 0.9960 |
| **SNP Precision** | 0.9989 | 0.9989 | |
| **SNP Sensitivity** | 0.9960 | 0.9960 | |
| **INDEL F1** | 0.9940 | 0.9940 | >= 0.9920 |
| **INDEL Precision** | 0.9957 | 0.9957 | |
| **INDEL Sensitivity** | 0.9923 | 0.9923 | |

Both FP32 and BF16 produce 207,799 variant calls — identical output.

### Cost Comparison

![Cost per genome](docs/figures/cost_per_genome.png)

| Platform | vCPUs | $/hr | Est. Runtime (WGS) | Est. Cost/Genome | Source |
|----------|-------|------|---------------------|-----------------|--------|
| GCP n2-standard-96 (x86) | 96 | $3.81 | ~79 min | ~$5.01 | [Google official](https://github.com/google/deepvariant/blob/r1.9/docs/metrics.md) |
| **AWS Graviton3 FP32** | 16 | $0.58 | ~7.8 hr | **~$4.50** | Measured chr20 x 48.1 |
| **AWS Graviton3 BF16** | 16 | $0.58 | ~6.5 hr | **~$3.76** | Measured chr20 x 48.1 |

**Methodology:** Graviton3 costs are extrapolated from measured chr20 wall times (582s FP32, 487s BF16) scaled by 48.1x to estimate full genome runtime. This is a standard approximation but introduces ~15-20% uncertainty — real genome runtime may differ. Google's n2-standard-96 number is their [official published benchmark](https://github.com/google/deepvariant/blob/r1.9/docs/metrics.md). All prices are on-demand, US regions.

**Key takeaway:** Graviton3 BF16 achieves **~25% lower cost per genome** than Google's 96-vCPU x86 reference, using only 16 vCPUs — but is **~5x slower** in wall-clock time. This is a cost-optimized configuration, not a speed-optimized one.

### How This Compares to Other Options

This project fills a specific niche. For context on alternatives:

| Approach | Est. Cost/Genome | Runtime | Open Source? | Notes |
|----------|-----------------|---------|-------------|-------|
| **This fork (Graviton3 BF16)** | ~$3.76 | ~6.5 hr | Yes | CPU-only, no license fees |
| **This fork (Graviton3 FP32)** | ~$4.50 | ~7.8 hr | Yes | Any ARM64 instance |
| Google DeepVariant x86 (96 vCPU) | ~$5.01 | ~79 min | Yes | [Official benchmark](https://github.com/google/deepvariant/blob/r1.9/docs/metrics.md) |
| NVIDIA Parabricks + GPU | <$2 + license | 8-16 min | No | Requires GPU instance + [Parabricks license](https://www.nvidia.com/en-us/clara/parabricks/) |
| Sentieon DNAscope (Graviton) | ~$2.30 | — | No | Proprietary, [requires license](https://www.sentieon.com/) |

**When to use this fork:**
- You want open-source DeepVariant on ARM64 (no alternative exists)
- You are cost-sensitive and can tolerate longer runtimes (batch processing, research pipelines)
- You are running on ARM64-only infrastructure (Oracle Cloud Free Tier, Hetzner CAX, on-premise ARM servers)

**When to use something else:**
- You need fast turnaround (<1 hour): use GPU-accelerated DeepVariant or scale to more x86 cores
- You need the absolute lowest cost and have a Sentieon license: Sentieon on Graviton is faster and cheaper

---

## Quick Start (Docker)

### Prerequisites

- **ARM64 Linux host** (aarch64) — Graviton, Ampere, Hetzner CAX, Jetson, etc.
- **Docker** installed and running

### Pull and Run

```bash
# Build the Docker image (must be on an ARM64 host)
git clone https://github.com/antomicblitz/deepvariant-linux-arm64.git
cd deepvariant-linux-arm64
git checkout r1.9
docker build -f Dockerfile.arm64 -t deepvariant-arm64 .
```

> **Note:** The full Docker build compiles TensorFlow and all C++ extensions from source. This takes several hours on an 8-core ARM64 instance. If you have pre-built binaries from a native build, use `Dockerfile.arm64.runtime` instead (minutes, not hours).

### Run DeepVariant

```bash
docker run \
  -v "YOUR_INPUT_DIR":"/input" \
  -v "YOUR_OUTPUT_DIR:/output" \
  --memory=28g \
  deepvariant-arm64 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/input/YOUR_REF \
  --reads=/input/YOUR_BAM \
  --output_vcf=/output/YOUR_OUTPUT_VCF \
  --output_gvcf=/output/YOUR_OUTPUT_GVCF \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Enable BF16 on Graviton3+

On AWS Graviton3 or newer instances (c7g, c8g, m7g, r7g), enable BF16 fast math for 38% faster inference:

```bash
docker run \
  -v "YOUR_INPUT_DIR":"/input" \
  -v "YOUR_OUTPUT_DIR:/output" \
  --memory=28g \
  -e TF_ENABLE_ONEDNN_OPTS=1 \
  -e ONEDNN_DEFAULT_FPMATH_MODE=BF16 \
  -e OMP_NUM_THREADS=$(nproc) \
  -e OMP_PROC_BIND=false \
  -e OMP_PLACES=cores \
  deepvariant-arm64 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/input/YOUR_REF \
  --reads=/input/YOUR_BAM \
  --output_vcf=/output/YOUR_OUTPUT_VCF \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

> BF16 fast math uses bfloat16 intermediate precision for matrix operations. This requires Graviton3+ (Neoverse V1 with BF16 instruction support). Verify with: `grep -q bf16 /proc/cpuinfo && echo "BF16 supported"`. Accuracy is identical to FP32 — validated on GIAB truth sets.

### Runtime-Only Docker Image (Pre-built Binaries)

If you have already compiled DeepVariant natively on an ARM64 host (see [Build from Source](#build-from-source)), you can build a lightweight runtime image that skips the multi-hour compilation:

```bash
docker build -f Dockerfile.arm64.runtime -t deepvariant-arm64 .
```

This copies the pre-built `bazel-out/aarch64-opt/bin/` binaries directly into the image.

---

## Installation

### Option 1: Pre-built Docker Image (Recommended)

The fastest way to get started — pull the pre-built image from GitHub Container Registry:

```bash
# Pull the optimized image (includes C++ optimizations + ONNX support)
docker pull ghcr.io/antomicblitz/deepvariant-arm64:optimized

# Or the baseline image
docker pull ghcr.io/antomicblitz/deepvariant-arm64:latest
```

Then run:

```bash
docker run \
  -v /path/to/data:/data \
  --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:optimized \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Option 2: One-Command AWS Graviton Setup

Set up a Graviton3 instance from scratch with all dependencies and test data:

```bash
# Launch a c7g.4xlarge (16 vCPU, 32 GB RAM) — requires AWS CLI configured
aws ec2 run-instances \
  --image-id ami-0f1b9964277dbd54e \
  --instance-type c7g.4xlarge \
  --key-name YOUR_KEY \
  --security-group-ids YOUR_SG \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=deepvariant-arm64}]'

# SSH in and run the setup script
ssh ubuntu@INSTANCE_IP
sudo apt-get update && sudo apt-get install -y docker.io
sudo usermod -aG docker ubuntu && newgrp docker

# Pull the image
docker pull ghcr.io/antomicblitz/deepvariant-arm64:optimized
```

### Option 3: Build from Source (Native)

Build DeepVariant natively on an ARM64 Linux host. This is useful for development, debugging, or creating binaries for the runtime Docker image.

#### Prerequisites

- **ARM64 Linux host** (Ubuntu 24.04 recommended — GCC 13+ required for TF 2.13.1)
- 16 GB RAM minimum (+ 8 GB swap for TF compilation)
- ~50 GB disk space (TF source + Bazel cache)
- Python 3.10 (install via deadsnakes PPA on Ubuntu 24.04)

#### 1. Clone the Repository

```bash
git clone https://github.com/antomicblitz/deepvariant-linux-arm64.git
cd deepvariant-linux-arm64
git checkout r1.9
```

#### 2. Create user.bazelrc (Resource Limits)

The default `.bazelrc` sets `--jobs 128` which will OOM on 16 GB machines:

```bash
cat > user.bazelrc << 'EOF'
build --jobs 4
build --local_ram_resources=12288
build --cxxopt=-include --cxxopt=cstdint
build --host_cxxopt=-include --host_cxxopt=cstdint
EOF
```

The `cstdint` flags fix GCC 13+ compatibility with TF 2.13.1 headers that are missing `#include <cstdint>`.

#### 3. Create Swap (if needed)

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 4. Install Build Prerequisites

```bash
chmod +x build-prereq-arm64.sh build_release_binaries_arm64.sh
./build-prereq-arm64.sh
```

This installs system packages, Bazel 5.3.0 for aarch64, Boost libraries, clones and configures TensorFlow 2.13.1, and runs `run-prereq.sh` for Python packages.

#### 5. Build

```bash
source settings_arm64.sh
./build_release_binaries_arm64.sh
```

Build output goes to `bazel-out/aarch64-opt/bin/deepvariant/`. The full build is ~2273 Bazel actions and takes several hours with 4 jobs on an 8-core machine.

#### 6. Build Runtime Docker Image

```bash
docker build -f Dockerfile.arm64.runtime -t deepvariant-arm64 .
```

---

## What is DeepVariant?

DeepVariant is a deep learning-based variant caller that takes aligned reads (in BAM or CRAM format), produces pileup image tensors from them, classifies each tensor using a convolutional neural network, and finally reports the results in a standard VCF or gVCF file.

DeepVariant supports germline variant-calling in diploid organisms. For full documentation on DeepVariant's capabilities, case studies, and supported data types, see the [upstream repository](https://github.com/google/deepvariant).

---

## Reproducing Benchmarks

### Download Test Data

```bash
# Create data directory
sudo mkdir -p /data/{reference,bam,truth,output}

# Reference genome (GRCh38, no alt contigs)
curl -sO https://storage.googleapis.com/genomics-public-data/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
gunzip GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
mv GCA_000001405.15_GRCh38_no_alt_analysis_set.fna /data/reference/GRCh38_no_alt_analysis_set.fasta
samtools faidx /data/reference/GRCh38_no_alt_analysis_set.fasta

# HG003 chr20 BAM (35x NovaSeq)
BUCKET=https://storage.googleapis.com/deepvariant/case-study-testdata
wget -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
wget -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai

# GIAB truth set (v4.2.1)
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed
```

### Run the Benchmark

```bash
# Full chr20 benchmark (FP32 + BF16, 2 runs each, with accuracy validation)
bash scripts/benchmark_full_chr20.sh --data-dir /data

# Quick Graviton3 benchmark (thread sweep + BF16 comparison)
bash scripts/benchmark_graviton3.sh --data-dir /data
```

### Regenerate Charts

```bash
python3 scripts/generate_benchmark_charts.py
```

---

## What Was Changed

This fork modifies the following files from upstream DeepVariant v1.9.0 to enable ARM64 Linux compilation.

### New Files

| File | Purpose |
|------|---------|
| `Dockerfile.arm64` | Full from-source ARM64 Docker build (Ubuntu 24.04, deadsnakes Python 3.10) |
| `Dockerfile.arm64.runtime` | Runtime-only Docker image using pre-built binaries |
| `settings_arm64.sh` | ARM64 build settings (no `-march=corei7`, `aarch64-opt` output dir, OneDNN+ACL) |
| `build-prereq-arm64.sh` | ARM64 build prerequisites (aarch64 Bazel, system Boost, clang-14) |
| `build_release_binaries_arm64.sh` | ARM64 build script (`aarch64-opt` paths, ARM64 TF wheel) |
| `user.bazelrc` | Resource limits for 16 GB machines + GCC 13 cstdint fix |
| `scripts/benchmark_arm64.sh` | HG003 chr20 benchmark with accuracy validation |
| `scripts/benchmark_full_chr20.sh` | Full chr20 benchmark (2x FP32 + 2x BF16 + hap.py) |
| `scripts/benchmark_graviton3.sh` | BF16 benchmark with DNNL_VERBOSE check and thread sweep |
| `scripts/generate_benchmark_charts.py` | Generate benchmark visualization charts |
| `scripts/setup_graviton.sh` | One-command ARM64 instance setup |
| `scripts/validate_accuracy.sh` | hap.py accuracy validation against GIAB truth sets |
| `scripts/convert_model_onnx.py` | TF SavedModel to ONNX conversion |
| `scripts/quantize_model_onnx.py` | Dynamic INT8 quantization with real-data validation |
| `scripts/quantize_static_onnx.py` | Static INT8 quantization with TFRecord calibration |

### Modified Files

| File | Change |
|------|--------|
| `.bazelrc` | Added `try-import %workspace%/user.bazelrc` (Bazel 5.3.0 doesn't auto-load it) |
| `third_party/htslib.BUILD` | Replaced hardcoded x86 SSE/POPCNT defines with runtime `uname -m` detection: `HAVE_NEON` for aarch64, SSE for x86 |
| `third_party/libssw.BUILD` | Added `src/sse2neon.h` to hdrs (undeclared header error on ARM64) |
| `tools/build_absl.sh` | Updated clang-11/llvm-11 to clang-14/llvm-14 (Ubuntu 24.04 compatibility) |
| `run-prereq.sh` | Ubuntu 24.04 fixes: `python3.10-distutils` from deadsnakes, `--ignore-installed` for conflicting packages, `--no-deps` for `tf-models-official` on aarch64 |
| `deepvariant/call_variants.py` | Added `--use_onnx` / `--onnx_model` flags, ONNX Runtime session setup, warmup pass |

### Key Build Fixes Applied

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| SSE4.1/POPCNT not supported | htslib `config.h` hardcodes x86 ISA defines | Runtime `uname -m` detection in genrule |
| `sse2neon.h` undeclared | libssw conditionally includes it but BUILD doesn't list it | Added to `hdrs` in `libssw.BUILD` |
| `uint64_t` undeclared | GCC 13 stricter about missing `<cstdint>` includes | `--cxxopt=-include --cxxopt=cstdint` in `user.bazelrc` |
| `clang-11` not found | Deprecated on Ubuntu 24.04 | Updated to `clang-14` in `build_absl.sh` |
| Missing Boost libraries | `fast_pipeline` needs boost-system, boost-filesystem, boost-math | Added to `build-prereq-arm64.sh` |
| GLIBC 2.38 mismatch | Binaries built on Ubuntu 24.04 can't run in 22.04 containers | Use `arm64v8/ubuntu:24.04` as Docker base |
| Python 3.10 not in repos | Ubuntu 24.04 ships Python 3.12 | Install from deadsnakes PPA |
| `cryptography` ABI crash | System package compiled for Python 3.12 | `pip install --ignore-installed cryptography cffi` |
| conda aarch64 gaps | bioconda bcftools/samtools not available for linux-aarch64 | Install from apt instead |
| OOM during build | Default 128 jobs exceeds 16 GB RAM | `user.bazelrc` with `--jobs 4 --local_ram_resources=12288` |

---

## Roadmap

- **Phase 1 (complete):** CPU-only ARM64 build. Native compilation, Docker image, pipeline validated on GIAB HG003.
- **Phase 2 (in progress):**
  - BF16 fast math on Graviton3+ — **complete, 1.61x call_variants speedup, zero accuracy loss**
  - ONNX Runtime integration — complete (`--use_onnx` flag), but slower than TF+OneDNN on Neoverse-N1
  - INT8 quantization of InceptionV3 — planned (1.5-4x potential speedup)
- **Phase 3 (planned):** GPU/NPU acceleration (Jetson CUDA, RK3588 NPU).

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Ubuntu 24.04 (or compatible aarch64 Linux) |
| Architecture | ARM64 / aarch64 (Graviton, Ampere, Cortex-A76+) |
| Python | 3.10 |
| Bazel | 5.3.0 (installed by `build-prereq-arm64.sh`) |
| TensorFlow | 2.13.1 (aarch64 wheel) |
| Docker | For containerized deployment |
| RAM | 16 GB minimum (+ 8 GB swap for compilation), 32 GB recommended for benchmarks |
| Disk | ~50 GB (TF source + Bazel cache) |

For BF16 acceleration: Graviton3+ (Neoverse V1/V2 with BF16 instruction support). Check with `grep bf16 /proc/cpuinfo`.

---

## Related Projects

- [google/deepvariant](https://github.com/google/deepvariant) — upstream x86 DeepVariant
- [antomicblitz/deepvariant-macos-arm64-metal](https://github.com/antomicblitz/deepvariant-macos-arm64-metal) — macOS ARM64 port with Metal GPU + CoreML acceleration (6.1x speedup)

---

## How to Cite

If you use DeepVariant in your work, please cite:

[A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* 36, 983-987 (2018).](https://rdcu.be/7Dhl)
Ryan Poplin, Pi-Chuan Chang, David Alexander, Scott Schwartz, Thomas Colthurst, Alexander Ku, Dan Newburger, Jojo Dijamco, Nam Nguyen, Pegah T. Afshar, Sam S. Gross, Lizzie Dorfman, Cory Y. McLean, and Mark A. DePristo.
doi: https://doi.org/10.1038/nbt.4235

## License

[BSD-3-Clause license](LICENSE)

## Disclaimer

This is not an official Google product.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.
