# DeepVariant — Linux ARM64 Native Build

[![release](https://img.shields.io/badge/base-v1.9.0-green?logo=github)](https://github.com/google/deepvariant/releases)
[![platform](https://img.shields.io/badge/platform-Linux%20ARM64-blue?logo=linux)](https://en.wikipedia.org/wiki/AArch64)
[![accuracy](https://img.shields.io/badge/accuracy-validated%20on%20GIAB-success)](#accuracy)

> This fork tracks google/deepvariant v1.9.0. Version tags use the format
> `v{upstream}-arm64.{n}` to indicate upstream compatibility.
> For the upstream project, see https://github.com/google/deepvariant

There is no official Linux ARM64 build of [DeepVariant](https://github.com/google/deepvariant). This fork patches the build system to compile natively on aarch64, producing a working Docker image for AWS Graviton, Oracle Ampere A1, Hetzner CAX, and other ARM64 instances.

---

## Why ARM64?

ARM64 cloud instances are 25-50% cheaper per vCPU than x86 equivalents. This fork makes DeepVariant — the most accurate open-source variant caller — available on this hardware for the first time.

| Solution | Speed (30x WGS) | Cost/Genome | License |
|----------|-----------------|-------------|---------|
| Google DeepVariant x86 (96 vCPU) | ~1.3 hr | ~$5.01 | Open source |
| Sentieon DNAscope (Graviton) | ~1-2 hr | ~$2-4 + **per-sample license** | Proprietary |
| NVIDIA Parabricks (GPU) | 8-16 min | <$2 + **license** | Proprietary |
| **This fork (16 vCPU Graviton3, BF16)** | **~6.5 hr** | **~$3.77** | **Open source** |
| **This fork (16 vCPU Graviton4, INT8)** | **~4.9 hr** | **~$3.33** | **Open source** |
| **This fork (16 OCPU Oracle A2, INT8)** | **~7.3 hr** | **~$2.32†** | **Open source** |

> †Oracle A2 $2.32/genome with jemalloc enabled (`-e DV_USE_JEMALLOC=1`); baseline without jemalloc is $2.49/genome.

> **Already cheaper than Google's x86 reference** — Oracle A2 INT8 at $2.32/genome (with jemalloc), Graviton3 BF16 at $3.77/genome (vs $5.01). With scaling to 32+ vCPU and fast_pipeline, targeting ~2.5 hr at ~$3/genome on Graviton. Oracle A2 with a rebuilt Docker image (enabling BF16) could push below $2/genome.

**Use this fork when** you want open-source DeepVariant on ARM64, or you are cost-sensitive and can tolerate longer runtimes (batch processing, research pipelines). **Use GPU-accelerated DeepVariant** when you need fast turnaround.

---

## Benchmarks (chr20, Multiple Platforms)

All benchmarks: GIAB HG003, full chr20, accuracy validated with `rtg vcfeval`. Graviton3 averaged over 2-3 runs; Graviton4 and Oracle A2 averaged over 2 runs.

### Inference Rate

| Platform | vCPUs | Config | call_variants Rate | chr20 Wall Time |
|----------|-------|--------|-------------------|-----------------|
| GCP t2a (Neoverse-N1) | 8 | FP32 | 0.880 s/100 | 12m57s |
| GCP t2a (Neoverse-N1) | 16 | FP32 | 0.512 s/100 | 7m22s |
| AWS Graviton3 | 16 | FP32 | 0.379 s/100 | 9m41s |
| **AWS Graviton3** | **16** | **BF16** | **0.232 s/100** | **8m06s** |
| **AWS Graviton3** | **16** | **INT8 ONNX** | **0.238 s/100** | **~8m36s** |
| **AWS Graviton4** | **16** | **INT8 ONNX** | **0.197 s/100** | **6m06s** |
| AWS Graviton4 | 16 | ONNX FP32 | 0.446 s/100 | 10m02s |
| AWS Graviton4 | 16 | BF16 (standalone CV) | 0.328 s/100 | ~8m32s* |
| **Oracle A2 (AmpereOne)** | **16 OCPU** | **INT8 ONNX** | **0.389 s/100** | **9m44s** |
| Oracle A2 (AmpereOne) | 16 OCPU | TF Eigen FP32 | 0.387 s/100 | 10m29s |

BF16 and INT8 achieve nearly identical call_variants rates on Graviton3. INT8 is the better choice on platforms **without** BF16 support (Neoverse-N1, Ampere Altra), where it provides a 2.3x speedup over FP32 ONNX (isolated benchmark: 0.225 s/100 vs 0.517 s/100).

> **Note on isolated vs pipeline rates:** The isolated ONNX benchmark measures 0.225 s/100 for INT8, while the full pipeline measures 0.238 s/100. The difference is due to pipeline overhead (TF environment initialization, dataset loading, writer process coordination). The pipeline rate is the operationally relevant number.

> **Graviton4 caveats:** *TF BF16 full pipeline OOM-killed on c8g.4xlarge (32 GB) — TF SavedModel uses ~26 GB RSS. INT8 ONNX works perfectly on 32 GB (~2-3 GB RSS). Needs c8g.8xlarge (64 GB) for full TF BF16 pipeline.* Oracle A2 uses ONNX or TF Eigen (OneDNN+ACL causes SIGILL on AmpereOne); a Docker rebuild would enable BF16.

### Pipeline Breakdown

**Graviton3 (c7g.4xlarge, 16 vCPU):**

| Stage | FP32 | BF16 | INT8 ONNX (3-run avg) |
|-------|------|------|-----------|
| make_examples | 255s | 278s | 299s |
| call_variants | 298s (0.379s/100) | 185s (0.232s/100) | 194s (0.237s/100) |
| postprocess | 29s | 24s | 14s |
| **Total** | **582s** | **487s** | **507s** |

**Cross-platform comparison (16 vCPU, best available backend):**

| Platform | Backend | jemalloc | ME | CV (rate) | PP | Total | $/hr | $/genome | N |
|----------|---------|----------|-----|-----------|-----|-------|------|----------|---|
| Graviton3 (c7g) | BF16 | off | 278s | 185s (0.232) | 24s | **487s** | $0.58 | **$3.77** | 2* |
| Graviton3 (c7g) | BF16 | **on** | 242s | 188s (0.235) | 9s | **443s** | $0.58 | **$3.43** | 2* |
| Graviton3 (c7g) | INT8 ONNX | off | 299s | 194s (0.237) | 14s | **507s** | $0.58 | **$3.92** | 3 |
| **Graviton4 (c8g)** | **INT8 ONNX** | **off** | **194s** | **158s (0.197)** | **6s** | **366s** | **$0.68** | **$3.33** | 2* |
| Graviton4 (c8g) | ONNX FP32 | off | 232s | 360s (0.446) | 10s | **602s** | $0.68 | $5.07 | 2* |
| **Oracle A2 (AmpereOne)** | **INT8 ONNX** | **off** | **253s** | **315s (0.389)** | **11s** | **584s** | **$0.32** | **$2.49** | **4** |
| **Oracle A2 (AmpereOne)** | **INT8 ONNX** | **on** | **210s** | **318s (0.393)** | **12s** | **544s** | **$0.32** | **$2.32** | **4** |
| Oracle A2 (AmpereOne) | TF Eigen FP32 | off | 287s | 325s (0.387) | 17s | **629s** | $0.32 | $2.69 | 2* |

> *N<4 runs; wider confidence interval. Wall time includes ~4-5s Docker startup and inter-stage overhead not captured in individual ME/CV/PP timings. All $/genome use formula: `chr20_wall_s × 48.1 / 3600 × $/hr`. jemalloc: enable with `-e DV_USE_JEMALLOC=1` (verified 14-17% ME, ~0% CV improvement; see `scripts/benchmark_jemalloc_ablation.sh`).

> Graviton4 INT8 ONNX is the fastest ARM64 configuration. Oracle A2 INT8 ONNX + jemalloc is the cheapest at $2.32/genome. Graviton4 TF BF16 OOM-killed on 32 GB — needs 64 GB instance. Oracle A2 OneDNN SIGILL — needs Docker rebuild for BF16. Both platforms have headroom for improvement.

### Accuracy

All three configurations produce **equivalent accuracy** on GIAB HG003 chr20 (207,799 variant calls, validated with `rtg vcfeval`):

**Aggregate F1:**

| Metric | FP32 | BF16 | INT8 ONNX |
|--------|------|------|-----------|
| SNP F1 | 0.9977 | 0.9977 | 0.9978 |
| INDEL F1 | 0.9961 | 0.9961 | 0.9962 |

**Stratified region validation** (GIAB difficult regions, chr20, confidence-restricted):

| Region | INT8 SNP | BF16 SNP | INT8 INDEL | BF16 INDEL |
|--------|----------|----------|------------|------------|
| Homopolymers (≥7bp) | 0.9985 | 0.9985 | 0.9967 | 0.9963 |
| Simple Repeats | 0.9994 | 0.9994 | 0.9967 | 0.9961 |
| Tandem Repeats (201-10000bp) | 0.9983 | 0.9983 | 0.9926 | 0.9926 |
| Segmental Duplications | 0.9802 | 0.9744 | 0.9814 | 0.9814 |

INT8 matches or exceeds BF16 accuracy in all tested stratification regions, including the difficult contexts where quantization commonly degrades (homopolymers, tandem repeats, segmental duplications).

---

## Quick Start

**Prerequisites:** ARM64 Linux host + Docker.

```bash
# Pull pre-built image
docker pull ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2

# Run
docker run \
  -v /path/to/data:/data \
  --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Enable BF16 (Graviton3+)

38% faster call_variants inference on Graviton3/4 instances (c7g, c8g, m7g, r7g). Total pipeline speedup is 1.20x at 16 vCPU — make_examples becomes the bottleneck at this core count.

```bash
docker run \
  -v /path/to/data:/data \
  --memory=28g \
  -e TF_ENABLE_ONEDNN_OPTS=1 \
  -e ONEDNN_DEFAULT_FPMATH_MODE=BF16 \
  -e OMP_NUM_THREADS=$(nproc) \
  -e OMP_PROC_BIND=false \
  -e OMP_PLACES=cores \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

Check BF16 support: `grep -q bf16 /proc/cpuinfo && echo "BF16 supported"`

### Optional: jemalloc Allocator

Reduces malloc contention under concurrent shards. Enable with:

```bash
docker run -e DV_USE_JEMALLOC=1 ...
```

To use a custom jemalloc path: `-e DV_JEMALLOC_PATH=/path/to/libjemalloc.so`.

Benchmark data: see `scripts/benchmark_jemalloc_ablation.sh`.

### Auto-configure for your ARM64 CPU

Not sure which backend to use? Run autoconfig to get a recommended configuration:

```bash
docker run --rm ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2 \
  bash /opt/deepvariant/scripts/autoconfig.sh
```

Or enable automatic configuration for every run:

```bash
docker run -e DV_AUTOCONFIG=1 ...
```

Autoconfig detects your CPU (Graviton3/4, AmpereOne, Neoverse-N1) and selects the optimal backend, thread counts, and safety settings automatically. User-provided environment variables always take precedence.

---

## Build from Source

<details>
<summary>Click to expand build instructions</summary>

### Prerequisites

- ARM64 Linux host (Ubuntu 24.04, GCC 13+)
- 16 GB RAM + 8 GB swap, ~50 GB disk
- Python 3.10 (deadsnakes PPA)

### Steps

```bash
git clone https://github.com/antomicblitz/deepvariant-linux-arm64.git
cd deepvariant-linux-arm64

# Resource limits for 16 GB machines + GCC 13 fix
cat > user.bazelrc << 'EOF'
build --jobs 4
build --local_ram_resources=12288
build --cxxopt=-include --cxxopt=cstdint
build --host_cxxopt=-include --host_cxxopt=cstdint
EOF

# Install prerequisites and build
chmod +x build-prereq-arm64.sh build_release_binaries_arm64.sh
./build-prereq-arm64.sh
source settings_arm64.sh
./build_release_binaries_arm64.sh

# Build runtime Docker image from compiled binaries
docker build -f Dockerfile.arm64.runtime -t deepvariant-arm64 .
```

The full build takes several hours on an 8-core machine (~2273 Bazel actions).

</details>

---

## Reproducing Benchmarks

<details>
<summary>Click to expand benchmark instructions</summary>

### Download Test Data

```bash
sudo mkdir -p /data/{reference,bam,truth,output}

# Reference genome
curl -sO https://storage.googleapis.com/genomics-public-data/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
gunzip GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
mv GCA_000001405.15_GRCh38_no_alt_analysis_set.fna /data/reference/GRCh38_no_alt_analysis_set.fasta
samtools faidx /data/reference/GRCh38_no_alt_analysis_set.fasta

# HG003 chr20 BAM + GIAB truth set
BUCKET=https://storage.googleapis.com/deepvariant/case-study-testdata
wget -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
wget -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
wget -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed
```

### Run

```bash
bash scripts/benchmark_full_chr20.sh --data-dir /data
```

</details>

---

## What Was Changed

This fork modifies upstream DeepVariant v1.9.0 for ARM64 Linux compilation.

**New files:** `Dockerfile.arm64`, `settings_arm64.sh`, `build-prereq-arm64.sh`, `build_release_binaries_arm64.sh`, benchmark and quantization scripts in `scripts/`.

**Key modifications:** `third_party/htslib.BUILD` (NEON detection), `third_party/libssw.BUILD` (sse2neon header), `tools/build_absl.sh` (clang-14), `run-prereq.sh` (Ubuntu 24.04 fixes), `deepvariant/call_variants.py` (ONNX inference, INT8 output normalization, SavedModel warmup).

**INT8 normalization fix:** INT8 quantization error causes some predictions to have probability distributions that don't sum to 1.0 (e.g., `[0.992, 0.0, 0.0]`). The ONNX inference path renormalizes outputs (`predictions / row_sums`) before passing to `round_gls()`. Without this fix, `postprocess_variants` crashes with `ValueError: Invalid genotype likelihoods do not sum to one`. This is specific to quantized models — FP32 and BF16 outputs always sum to 1.0.

<details>
<summary>Full list of build fixes</summary>

| Issue | Fix |
|-------|-----|
| SSE4.1/POPCNT not supported | Runtime `uname -m` detection in htslib genrule |
| `sse2neon.h` undeclared | Added to `hdrs` in `libssw.BUILD` |
| `uint64_t` undeclared (GCC 13) | `--cxxopt=-include --cxxopt=cstdint` |
| `clang-11` not found (Ubuntu 24.04) | Updated to `clang-14` |
| Missing Boost libraries | Added to `build-prereq-arm64.sh` |
| GLIBC 2.38 mismatch | `arm64v8/ubuntu:24.04` Docker base |
| Python 3.10 not in repos | deadsnakes PPA |
| OOM during build | `user.bazelrc` with `--jobs 4` |

</details>

---

## Roadmap

- **Phase 1 (complete):** Native ARM64 build, Docker image, GIAB-validated pipeline.
- **Phase 2A (complete):** BF16 on Graviton3+ — 1.61x call_variants speedup, zero accuracy loss.
- **Phase 2B (complete):** INT8 static quantization — 2.3x over ONNX FP32, matches BF16 call_variants rate, stratified region validation passed.
- **Phase 2C (complete):** OMP env fix, stratified validation, Graviton4 ONNX FP32 benchmark, Oracle A2 TF Eigen benchmark — $2.49/genome cheapest tested.
- **Phase 2D (in progress):** Graviton4 INT8 ONNX (366s, $3.33/genome), Oracle A2 INT8 ONNX (542s, $2.32/genome — new cheapest), fast_pipeline tested (needs 32+ vCPU). Remaining: 32 vCPU + fast_pipeline, Graviton4 BF16 on 64 GB, Oracle A2 Docker rebuild for BF16. Target: ~2.5 hr at ~$3/genome on Graviton, <$2/genome on Oracle A2.
- **Phase 3 (future):** GPU/NPU acceleration (Jetson CUDA, RK3588 NPU).

> **Note:** This project targets Illumina short-read WGS/WES workflows. For long-read ONT or PacBio data, consider [Clair3](https://github.com/HKU-BAL/Clair3) which has community ARM64 support.

---

## How to Cite

[A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* 36, 983-987 (2018).](https://rdcu.be/7Dhl)

## License

[BSD-3-Clause](LICENSE). This is not an official Google product. Not intended as a medical device or for clinical use.
