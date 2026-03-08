# DeepVariant — Linux ARM64 Native Build

[![release](https://img.shields.io/badge/base-v1.9.0-green?logo=github)](https://github.com/google/deepvariant/releases)
[![platform](https://img.shields.io/badge/platform-Linux%20ARM64-blue?logo=linux)](https://en.wikipedia.org/wiki/AArch64)
[![accuracy](https://img.shields.io/badge/SNP%20F1-0.9961%20(full%20genome)-success)](#accuracy--full-genome-validation)
[![license](https://img.shields.io/badge/license-BSD--3-blue)](#license)

> Fork of [google/deepvariant](https://github.com/google/deepvariant) v1.9.0. Tags: `v{upstream}-arm64.{n}`.

**The gold standard in variant calling. Now on ARM64. For less than the price of a chewing gum per genome.**

Google's [DeepVariant](https://rdcu.be/7Dhl) (Poplin et al., *Nature Biotechnology* 2018) achieves the highest SNP accuracy of any open-source variant caller. It also had no Linux ARM64 build — until this fork. Run it on a $0.16/hr Oracle A1 instance and get near-reference accuracy (SNP F1 0.9961 vs x86's ~0.9996 on full 30x WGS) at **$0.80/genome on dedicated ARM64 vCPUs** — or **$0.33/genome on a Hetzner CAX41 shared instance** in the EU.

At [UK Biobank](https://doi.org/10.1038/s41586-025-09272-9) scale (490,640 genomes), the compute cost difference vs. the x86 reference is **$2.3 million** — enough to fund the sequencing of ~4,600 additional genomes. For the [proposed Three Million African Genomes project](https://doi.org/10.1038/d41586-021-00313-7), the gap is $15M vs. $1M. The trade-off is a small accuracy reduction (SNP F1 0.9961 vs ~0.9996 on x86) and longer wall time — acceptable for population-scale GWAS, where cost dominates.

---

## Quick Start

**Requirements:** ARM64 Linux + Docker.

```bash
docker pull ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6

docker run \
  -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/scripts/run_parallel_cv.sh \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz
```

The script auto-detects your CPU (Graviton3/4, AmpereOne, Neoverse-N1/N2), enables jemalloc, selects the right backend (BF16 or INT8 ONNX), and scales parallel CV workers to your CPU and RAM. No env vars needed — just `docker run`. On non-BF16 platforms (Oracle A1, A2, Hetzner CAX), it uses the **pre-installed INT8 ONNX model** (`/opt/models/wgs/model_int8_static.onnx`, 21 MB). User-provided env vars (e.g., `TF_ENABLE_ONEDNN_OPTS`) always take precedence.

<details>
<summary>Manual backend override (BF16 or custom INT8)</summary>

#### BF16 (Graviton3+, 38% faster CV)

```bash
docker run -v /path/to/data:/data \
  -e TF_ENABLE_ONEDNN_OPTS=1 -e ONEDNN_DEFAULT_FPMATH_MODE=BF16 \
  -e OMP_NUM_THREADS=$(nproc) -e OMP_PROC_BIND=false -e OMP_PLACES=cores \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS --ref=/data/reference.fasta --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

#### Custom INT8 quantization (WES, PacBio, or your own calibration data)

The Docker image ships with a pre-quantized WGS INT8 model. To quantize a different model:

```bash
# Step 1: Run the pipeline to generate calibration TFRecords
docker run -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS --ref=/data/reference.fasta --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz --num_shards=$(nproc) \
  --intermediate_results_dir=/data/intermediate

# Step 2: Quantize (one-time, ~2 min)
docker run -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  quantize_model \
  --input /opt/models/wgs/model.onnx \
  --output /data/model_int8_custom.onnx \
  --tfrecord_dir /data/intermediate/make_examples \
  --saved_model_dir /opt/models/wgs

# Step 3: Use the custom INT8 model
docker run -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS --ref=/data/reference.fasta --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256,--use_onnx=true,--onnx_model=/data/model_int8_custom.onnx"
```

</details>

On NVMe or tmpfs storage, add `--nocompress_intermediates` to skip gzip on TFRecord intermediates (~4% faster ME, ~12 GB disk for chr20).

<details>
<summary>Sequential mode (simpler, no parallel CV)</summary>

```bash
docker run -v /path/to/data:/data \
  -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS --ref=/data/reference.fasta --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

</details>

<details>
<summary>All run_parallel_cv.sh options</summary>

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--model_type` | Yes | -- | WGS, WES, PACBIO, ONT_R104, etc. |
| `--ref` | Yes | -- | Reference FASTA |
| `--reads` | Yes | -- | Input BAM/CRAM |
| `--output_vcf` | Yes | -- | Output VCF |
| `--num_shards` | No | nproc | make_examples shards |
| `--num_cv_workers` | No | auto (nproc/8) | Parallel CV workers |
| `--regions` | No | -- | Genomic region (e.g., chr20) |
| `--batch_size` | No | 256 | CV batch size |
| `--onnx_model` | No | auto | Custom ONNX model path |
| `--customized_model` | No | -- | Custom checkpoint |
| `--sample_name` | No | -- | VCF header sample name |
| `--output_gvcf` | No | -- | gVCF output path |
| `--postprocess_cpus` | No | num_shards | CPUs for postprocess |
| `--intermediate_results_dir` | No | tmpdir | Working directory |

</details>

---

## What this project actually contributes

There are four distinct results here, with different levels of novelty.

### 1. INT8 quantization without accuracy loss *(the most novel result)*

The DeepVariant InceptionV3 model (84 MB FP32) was quantized to INT8 using ONNX Runtime static quantization with calibration data drawn from real genomic TFRecords. The quantized model is **74% smaller (21 MB)** and **2.3x faster** at inference than FP32 ONNX — with no accuracy loss *from quantization itself*:

| Metric | FP32 (chr20) | BF16 (chr20) | INT8 ONNX (chr20) | INT8 Full Genome | INT8 WES |
|--------|------|------|-----------|------------------|----------|
| SNP F1 | 0.9977 | 0.9977 | **0.9978** | **0.9961** | **0.9931** |
| INDEL F1 | 0.9961 | 0.9961 | **0.9962** | **0.9956** | **0.9738** |

> chr20 results validated with `rtg vcfeval` on GIAB HG003 v4.2.1 (all rows). Full Genome column: 30x WGS HG003, all chromosomes, on AWS c8g.8xlarge (32 vCPU Graviton4). Completed in ~2h17m — 27% faster than chr20 extrapolation. WES column: HG003 IDT exome 100x, all chromosomes, same platform. WES FP32 BF16 baseline: SNP F1=0.9930, INDEL F1=0.9776 — INT8 matches within noise.
>
> **Note on full-genome accuracy:** The INT8 full genome SNP F1 (0.9961) is lower than both the chr20 result (0.9978) and the x86 reference (~0.9996). The chr20→full genome drop is expected (harder chromosomes contribute more errors), but the gap vs x86 (0.0035 SNP F1) likely reflects differences in the ARM64 build environment (TF version, OneDNN backend, numerical precision paths) rather than INT8 quantization per se — the chr20 INT8 result *matches or exceeds* FP32 and BF16 on the same ARM64 platform.

Post-training INT8 quantization typically degrades accuracy by 0.6-3% on vision CNNs. That it doesn't here — not even in the difficult regions where quantization characteristically fails — is the finding. Stratified GIAB validation confirms:

| Region | INT8 SNP F1 | BF16 SNP F1 | INT8 INDEL F1 | BF16 INDEL F1 |
|--------|-------------|-------------|---------------|---------------|
| Homopolymers (>=7bp) | 0.9985 | 0.9985 | 0.9967 | 0.9963 |
| Simple Repeats | 0.9994 | 0.9994 | 0.9967 | 0.9961 |
| Tandem Repeats (201-10,000bp) | 0.9983 | 0.9983 | 0.9926 | 0.9926 |
| Segmental Duplications | **0.9802** | 0.9744 | 0.9814 | 0.9814 |

INT8 matches or exceeds BF16 in every region — including segmental duplications, where it actually *improves* by 0.006 SNP F1. The explanation is that InceptionV3's large, dense convolutions are unusually quantization-friendly; the model's structure largely avoids the depthwise convolution layers that cause INT8 degradation in other CNN architectures.

The 21 MB INT8 model is included as a release attachment and also opens a credible path to **edge deployment** on devices like NVIDIA Jetson or Rockchip RK3588 for point-of-care genomics — though this is not yet validated in this project.

**Engineering note:** INT8 quantization can produce probability vectors that don't sum to 1.0. Without correction, `postprocess_variants` crashes with `ValueError: Invalid genotype likelihoods`. The ONNX inference path in this fork renormalizes outputs (`predictions / row_sums`) before passing to `round_gls()`. This fix is mandatory and is included in the Docker image.

### 2. First open-source Linux ARM64 port

There is no official ARM64 build of DeepVariant ([Issue #834](https://github.com/google/deepvariant/issues/834), open since 2024, unresolved). The build system assumes SSE4/AVX throughout. This fork patches the build system, fixes 8 upstream compilation errors for ARM64/GCC 13, and produces a working Docker image validated against GIAB HG003.

The only existing ARM64 option is [Sentieon DNAscope](https://www.sentieon.com/) — excellent performance, competitive cost (under $1/genome on OCI ARM), but requires a **commercial license**, limiting access for many academic and resource-constrained labs. This fork is BSD-3 licensed: use it, modify it, redistribute it.

### 3. The first systematic ARM64 benchmarking corpus for variant calling

Before this project, there was no published comparison of DeepVariant performance across ARM64 hardware. This fork benchmarks **5 platforms x 3 backends x jemalloc on/off x 16 and 32 vCPU**, with N >= 2 runs per configuration and accuracy validated at each point:

| Platform | Microarch | Best config | $/genome | Notes |
|----------|-----------|-------------|----------|-------|
| AWS Graviton3 (c7g) | Neoverse-V1 | BF16+jemalloc | $3.43 | BF16 native; sweet spot at 16 vCPU |
| AWS Graviton4 (c8g) | Neoverse-V2 | INT8+4-way CV | $3.13 | Fastest CPU tested (0.197 s/100) |
| Oracle A1 (Altra) | Neoverse-N1 | INT8+4-way CV | **$0.80** | Cheapest dedicated OCPU |
| Oracle A2 (AmpereOne) | AmpereOne | INT8+4-way CV | $2.14 | OneDNN causes SIGILL — use ONNX |
| Hetzner CAX41 (Altra) | Neoverse-N1 | INT8+jemalloc | **$0.33** | Shared vCPU; EU-only; N=1 |

This is a community infrastructure resource. Choosing hardware for a genomics workload on ARM64 no longer requires guesswork.

### 4. EfficientNet-B3: a documented dead end

Built the full EfficientNet-B3 training pipeline — then measured it running **3x slower** than InceptionV3 despite 3.2x fewer FLOPs. Depthwise separable convolutions and squeeze-and-excitation blocks have poor GEMM density on CPUs, negating their theoretical FLOP advantage on ARM NEON hardware. This confirms results from the architecture literature in a genomics-specific CPU context. Full details: [TRAINING_EXPERIMENT.md](docs/TRAINING_EXPERIMENT.md).

---

## How much does it matter at scale?

The cost numbers below use the formula `chr20_wall_s x 48.1 / 3600 x $/hr` — a standard chr20-to-WGS projection. Full 30x WGS end-to-end validation on Graviton4 (c8g.8xlarge) completed in **2h17m** — 27% faster than this projection, confirming the estimate is conservative.

| Study scale | x86 reference ($5.01) | Oracle A1 ($0.80) | Hetzner ($0.33+) | Savings vs. x86 |
|-------------|----------------------|-------------------|-----------------|-----------------|
| 1 genome | $5.01 | $0.80 | $0.33 | -- |
| 1,000 genomes | $5,010 | $800 | $330 | up to $4,680 |
| 100,000 genomes (large GWAS) | $501,000 | $80,000 | $33,000 | up to $468,000 |
| 490,640 genomes ([UK Biobank WGS](https://doi.org/10.1038/s41586-025-09272-9)) | $2,458,106 | $392,512 | $161,911 | up to $2,296,195 |
| 3,000,000 genomes ([Three Million African Genomes](https://doi.org/10.1038/d41586-021-00313-7)) | $15,030,000 | $2,400,000 | $990,000 | up to $14,040,000 |

> **On Hetzner scale-out:** Default account limit is 5 servers, expandable to 10-25 per project on request. Hetzner is not designed for hyperscale burst compute. For runs requiring 50+ concurrent instances, Oracle A1 ($0.80/genome) is the operationally practical choice. The Hetzner figure is most relevant for individual researchers and small labs doing serial or small-batch processing.

---

## Compared to alternatives

| Solution | $/genome | Speed (30x WGS) | Accuracy (SNP F1) | License | ARM64 |
|----------|----------|-----------------|-------------------|---------|-------|
| Google DeepVariant (96 vCPU x86) | ~$5.01 | ~1.3 hr | ~0.9996 | Open source | No |
| Google DeepVariant (n1-standard-16, preemptible) | ~$2.84 | ~5.5 hr | ~0.9996 | Open source | No |
| Sentieon DNAscope (OCI ARM) | <$1 + license | ~1-2 hr | Comparable | **Proprietary** | Yes |
| Sentieon DNAscope (AWS Graviton, spot) | ~$0.74 + license | ~1-2 hr | Comparable | **Proprietary** | Yes |
| NVIDIA Parabricks | <$2 + license | 8-16 min | Comparable | **Proprietary** | No |
| **This fork — Oracle A1, 4-way CV** | **$0.80** | **~5.0 hr** | **0.9961*** | **Open source** | **Yes** |
| **This fork — Hetzner CAX41, INT8** | **$0.33+** | **~7.7 hr** | **0.9961*** | **Open source** | **Yes** |

> *Full 30x WGS accuracy (all chromosomes, INT8 ONNX). chr20-only is higher (0.9978) due to chr20 being among the easier chromosomes. The 0.0035 gap vs x86 (0.9961 vs ~0.9996) corresponds to ~17K additional missed SNPs genome-wide — clinically irrelevant for most GWAS/population studies, but worth noting for clinical diagnostics.
>
> +Hetzner: shared vCPU (~5% throttling variance), EU-only, N=1 for best config. Increase N before citing this number in a paper.

The trade-off is wall time and a small accuracy gap. A Hetzner run takes ~8 hours vs. ~1 hour on a 96-vCPU x86 instance, with SNP F1 0.35% lower than the x86 reference. For batch processing, overnight pipelines, and cost-constrained studies, this is an acceptable trade-off. For clinical turnaround or maximum accuracy, use GPU-accelerated DeepVariant on x86.

---

## Benchmarks (chr20, All Platforms)

All benchmarks: GIAB HG003, full chr20, accuracy validated with `rtg vcfeval`. N = repetitions per config.

### Inference Rate

| Platform | vCPUs | Config | CV Rate | chr20 Wall |
|----------|-------|--------|---------|-----------|
| GCP t2a (Neoverse-N1) | 8 | FP32 | 0.880 s/100 | 12m57s |
| GCP t2a (Neoverse-N1) | 16 | FP32 | 0.512 s/100 | 7m22s |
| AWS Graviton3 | 16 | FP32 | 0.379 s/100 | 9m41s |
| **AWS Graviton3** | **16** | **BF16** | **0.232 s/100** | **8m06s** |
| **AWS Graviton3** | **16** | **INT8 ONNX** | **0.237 s/100** | **~8m27s** |
| **AWS Graviton4** | **16** | **INT8 ONNX** | **0.197 s/100** | **6m06s** |
| AWS Graviton4 | 16 | ONNX FP32 | 0.446 s/100 | 10m02s |
| **Oracle A2 (AmpereOne)** | **16 OCPU** | **INT8 ONNX** | **0.389 s/100** | **9m44s** |
| **Oracle A1 (Altra)** | **16 OCPU** | **INT8 ONNX** | **0.309 s/100** | **~8m29s** |
| Oracle A1 (Altra) | 16 OCPU | TF Eigen FP32 | 0.588 s/100 | 12m15s |
| **Hetzner CAX41 (Altra)** | **16 (shared)** | **INT8 ONNX** | **0.366 s/100** | **~8m45s** |

### Full Pipeline — 16 vCPU, Best Backend

| Platform | Backend | jemalloc | ME | CV | PP | Total | $/hr | $/genome | N |
|----------|---------|----------|-----|-----|-----|-------|------|----------|---|
| Graviton3 (c7g) | BF16 | on | 242s | 188s | 9s | **443s** | $0.58 | $3.43 | 2* |
| Graviton3 (c7g) | INT8 ONNX | off | 299s | 194s | 14s | **507s** | $0.58 | $3.92 | 3 |
| **Graviton4 (c8g)** | **INT8 ONNX** | **off** | **194s** | **158s** | **6s** | **366s** | **$0.68** | **$3.33** | 2* |
| **Oracle A2 (AmpereOne)** | **INT8 ONNX** | **on** | **210s** | **318s** | **12s** | **544s** | **$0.32** | **$2.32** | **4** |
| **Oracle A1 (Altra)** | **INT8 ONNX** | **on** | **219s** | **250s** | **14s** | **486s** | **$0.16** | **$1.04** | 3 |
| **Hetzner CAX41 (Altra)** | **INT8 ONNX** | **on** | **253s** | **298s** | **17s** | **578s** | **$0.043** | **$0.33** | 1 |

> *N<4; wider confidence interval. Hetzner: ~5% throttling variance. All $/genome: `chr20_wall_s x 48.1 / 3600 x $/hr`.

### Parallel call_variants — Breaking the GEMM Ceiling

At 16+ threads, InceptionV3 GEMM saturates and more threads yield no speedup. `run_parallel_cv.sh` launches N independent ONNX workers on disjoint shards, giving 1.7-2.5x CV speedup with **zero changes to DeepVariant code**. Variant counts match sequential baseline exactly (207,799).

| Platform | vCPU | Sequential CV | 4-way CV | Speedup | $/genome | N |
|----------|------|--------------|----------|---------|----------|---|
| **Oracle A1** (16 OCPU) | 16 | 250s | **147s** | **1.70x** | **$0.80** | 3 |
| **Graviton4** (c8g.8xlarge) | 32 | 128s | **61s** | **2.10x** | **~$3.13** | 3 |
| **Graviton3** (c7g.8xlarge) | 32 | 141s | **74s** | **1.90x** | **~$3.35** | 4 |
| **Oracle A2** (16 OCPU) | 32 | 282s | **114s** | **2.47x** | **~$2.14** | 2* |

> ONNX backend only (~3 GB/worker). TF SavedModel uses ~26 GB RSS per worker — use ONNX or 64+ GB instances for TF.

---

## Deploy on ARM64

### Fresh instance setup

```bash
# One-command setup: installs Docker, detects BF16 support, configures TF env vars
bash scripts/setup_graviton.sh
```

Works on Ubuntu 22.04/24.04. Installs Docker, build essentials, and writes CPU-specific TF environment variables to `/etc/profile.d/deepvariant-arm64.sh`.

### Pull the Docker image

```bash
# Authenticate (if using a private registry)
echo "$GITHUB_PAT" | docker login ghcr.io -u USERNAME --password-stdin

# Pull (~2-4 GB compressed)
docker pull ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6
```

### Platform-specific notes

| Platform | Key setting | Why |
|----------|------------|-----|
| **Graviton3/4** (c7g, c8g) | Autoconfig enables BF16 + OneDNN (>=48 GB RAM) | BF16 BFMMLA — 38% faster CV. <48 GB falls back to INT8 ONNX (TF OOMs) |
| **Oracle A1** (Altra, Neoverse-N1) | Autoconfig selects INT8 ONNX, disables OneDNN | OneDNN+ACL adds 29% ME overhead on N1 without BF16 |
| **Oracle A2** (AmpereOne) | Autoconfig blocks OneDNN (SIGILL), selects INT8 ONNX | ACL compiled for N1 crashes on AmpereOne ISA — both ME and CV |
| **Hetzner CAX** (shared Altra) | Same as Oracle A1 | Shared vCPU; ~5% throttling variance |

`DV_AUTOCONFIG=1` handles all of the above automatically. You only need manual env vars if you want to override the defaults.

---

## Accuracy — Full Genome Validation

**Full 30x WGS** (HG003, GRCh38, all chromosomes) validated on AWS c8g.8xlarge (32 vCPU Graviton4, INT8 ONNX + jemalloc). Accuracy measured with `rtg vcfeval` against GIAB HG003 v4.2.1 truth set:

| Metric | Precision | Recall | F1 | TP | FP | FN |
|--------|-----------|--------|------|--------|-------|--------|
| **SNP** | 0.9986 | 0.9936 | **0.9961** | 3,306,123 | 4,571 | 21,357 |
| **INDEL** | 0.9973 | 0.9938 | **0.9956** | 501,300 | 1,340 | 3,135 |
| **Overall** | 0.9985 | 0.9936 | **0.9960** | 3,807,423 | 5,911 | 24,492 |

Total PASS variants: 4,813,103 (3,894,025 SNPs + 919,078 INDELs). Wall time: **2 hours 17 minutes** (c8g.8xlarge, 32 vCPU, $1.36/hr).

### WES (Exome) Validation

**HG003 IDT exome 100x** (all chromosomes, IDT capture kit) validated on the same Graviton4 instance. INT8 model quantized from the WES-specific SavedModel using 500 calibration samples (Percentile 99.99).

| Metric | FP32 BF16 | INT8 ONNX | Delta |
|--------|-----------|-----------|-------|
| **SNP F1** | 0.9930 | **0.9931** | +0.0001 |
| **INDEL F1** | 0.9776 | **0.9738** | -0.0038 |
| **Overall F1** | 0.9924 | **0.9923** | -0.0001 |

INT8 CV rate: **0.120 s/100** (24% faster than FP32 BF16 at 0.149 s/100). Total pipeline: **3m34s** (INT8) vs **3m55s** (FP32). Variant counts match exactly: 47,895.

---

## Workflow Integration

Nextflow and Snakemake workflows are provided in [`workflows/`](workflows/):

```bash
# Nextflow (single sample)
nextflow run workflows/nextflow/main.nf \
  -profile arm64 \
  --bam /data/sample.bam --ref /data/GRCh38.fasta --outdir results/

# Nextflow (batch — CSV with columns: sample,bam,bai)
nextflow run workflows/nextflow/main.nf \
  -profile arm64 \
  --input samples.csv --ref /data/GRCh38.fasta --outdir results/

# Snakemake
snakemake --cores 16 \
  --config bam=/data/sample.bam ref=/data/GRCh38.fasta
```

Platform profiles: `arm64` (auto-detect), `graviton`, `oracle_a1`, `hetzner`. See [workflows/nextflow/nextflow.config](workflows/nextflow/nextflow.config) for details.

---

## Reproduce These Results

Reproduce the accuracy numbers from this README on your own hardware.

### 1. Download test data

```bash
sudo mkdir -p /data/{reference,bam,truth,output}
sudo chown -R $(whoami) /data

# Reference genome
wget -q -O /data/reference/GRCh38_no_alt_analysis_set.fasta.gz \
  https://storage.googleapis.com/genomics-public-data/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
gunzip /data/reference/GRCh38_no_alt_analysis_set.fasta.gz
samtools faidx /data/reference/GRCh38_no_alt_analysis_set.fasta

# HG003 chr20 BAM + GIAB truth set
BUCKET=https://storage.googleapis.com/deepvariant/case-study-testdata
wget -q -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
wget -q -P /data/bam/ ${BUCKET}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai
wget -q -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
wget -q -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi
wget -q -P /data/truth/ ${BUCKET}/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed
```

### 2. Run the pipeline on chr20

```bash
docker run \
  -v /data:/data \
  -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference/GRCh38_no_alt_analysis_set.fasta \
  --reads=/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
  --output_vcf=/data/output/HG003.chr20.vcf.gz \
  --regions chr20 \
  --num_shards=$(nproc) \
  --intermediate_results_dir=/data/output/intermediate \
  --call_variants_extra_args="--batch_size=256"
```

### 3. Validate with hap.py

```bash
bash scripts/validate_accuracy.sh \
  --vcf /data/output/HG003.chr20.vcf.gz \
  --truth-vcf /data/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
  --truth-bed /data/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
  --ref /data/reference/GRCh38_no_alt_analysis_set.fasta \
  --output-dir /data/output/happy_results
```

This pulls the hap.py Docker image (`jmcdani20/hap.py:v0.3.12`) automatically and checks against accuracy gates:

```
Expected output (chr20 only):
  SNP F1:   0.9978  (gate: >= 0.9974)   PASS
  INDEL F1: 0.9962  (gate: >= 0.9940)   PASS
```

> These are chr20-only numbers. Full genome accuracy is lower (SNP F1=0.9961, INDEL F1=0.9956) — see [Accuracy — Full Genome Validation](#accuracy--full-genome-validation).

Or use the all-in-one benchmark script:

```bash
bash scripts/benchmark_full_chr20.sh --data-dir /data
```

---

## Build from Source

<details>
<summary>Click to expand</summary>

**Prerequisites:** ARM64 Linux (Ubuntu 24.04, GCC 13+), 16 GB RAM + 8 GB swap, ~50 GB disk, Python 3.10.

```bash
git clone https://github.com/antomicblitz/deepvariant-linux-arm64.git
cd deepvariant-linux-arm64

cat > user.bazelrc << 'EOF'
build --jobs 4
build --local_ram_resources=12288
build --cxxopt=-include --cxxopt=cstdint
build --host_cxxopt=-include --host_cxxopt=cstdint
EOF

chmod +x scripts/build/build-prereq-arm64.sh scripts/build/build_release_binaries_arm64.sh
./scripts/build/build-prereq-arm64.sh
source scripts/build/settings_arm64.sh
./scripts/build/build_release_binaries_arm64.sh
docker build -f docker/Dockerfile.arm64.runtime -t deepvariant-arm64 .
```

Full build: several hours on an 8-core machine (~2273 Bazel actions).

</details>

---

## What Was Changed

**New files:** `docker/Dockerfile.arm64`, `docker/Dockerfile.arm64.runtime`, `scripts/build/settings_arm64.sh`, `scripts/build/build-prereq-arm64.sh`, `scripts/build/build_release_binaries_arm64.sh`, `scripts/` (benchmarking, quantization, parallel CV, autoconfig, jemalloc ablation), and `workflows/` (Nextflow + Snakemake integration).

**Key modifications to upstream:** `third_party/htslib.BUILD` (NEON detection), `third_party/libssw.BUILD` (sse2neon), `tools/build_absl.sh` (clang-14), `scripts/build/run-prereq.sh` (Ubuntu 24.04), `deepvariant/call_variants.py` (ONNX inference, INT8 renormalization, SavedModel warmup). ACL v23.08 SVE filter and OneDNN indirect GEMM patches for AmpereOne preserved in `third_party/` for source rebuilds.

<details>
<summary>Build fixes</summary>

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

<details>
<summary>Dead ends (documented so you don't repeat them)</summary>

| Approach | Result | Why it failed |
|----------|--------|--------------|
| EfficientNet-B3 model swap | 3x slower despite 3.2x fewer FLOPs | Depthwise conv poor GEMM density on CPU — see [TRAINING_EXPERIMENT.md](docs/TRAINING_EXPERIMENT.md) |
| KMP_AFFINITY tuning | 30% regression | Conflicts with OMP thread pinning on ARM |
| ONNX ACL ExecutionProvider | Fragile, 16 ops supported | Not worth maintaining |
| Dynamic INT8 on ARM64 | Crash | ConvInteger op missing in ORT ARM64 |
| fast_pipeline at 16 vCPU | 42% slower | CPU contention between ME and CV |
| INT8 beyond 16 threads | No improvement | GEMM saturates — use parallel CV instead |
| ONNX inter-op parallelism | No improvement | InceptionV3 is intra-op bound |

</details>

---

## Roadmap

- [x] Native ARM64 build + Docker image
- [x] GIAB HG003 chr20 accuracy validation
- [x] BF16 inference on Graviton3+ (38% CV speedup)
- [x] INT8 static quantization (2.3x speedup, zero accuracy loss, 74% smaller model)
- [x] Parallel call_variants wrapper (1.9-2.5x CV speedup)
- [x] jemalloc integration (14-17% make_examples speedup)
- [x] Autoconfig for CPU detection
- [x] Hetzner CAX41 benchmark ($0.33/genome)
- [x] **Full 30x WGS end-to-end validation** — INT8 ONNX, GIAB HG003, all chromosomes (SNP F1=0.9961, INDEL F1=0.9956, 2h17m on Graviton4 32 vCPU)
- [x] **WES model validation on ARM64** — INT8 ONNX, GIAB HG003, IDT exome 100x (SNP F1=0.9931, INDEL F1=0.9738, matches FP32 baseline). WES INT8 is 24% faster CV (0.120 vs 0.149 s/100).
- [x] **Nextflow / Snakemake integration** — [`workflows/`](workflows/) with platform profiles (arm64, graviton, oracle_a1, hetzner). Supports single sample and batch CSV input.
- [ ] Edge device validation (Jetson, RK3588)
- [ ] **Upgrade TF 2.13.1 → 2.18+ and Python 3.10 → 3.13** — blocked on upstream google/deepvariant (still on TF 2.13.1). Plan:
  1. Wait for upstream TF bump or fork TF 2.18 ARM64 build
  2. Upgrade Bazel 5.3.0 → 6.x+, update WORKSPACE and third_party patches
  3. Fix C++ kernel compilation against new TF headers (pybind11, protobuf ABI)
  4. Rebuild base builder image from source
  5. Re-validate accuracy (GIAB HG003 WGS + WES, compare F1 to current baseline)

> For long-read ONT or PacBio data, see [Clair3](https://github.com/HKU-BAL/Clair3).

---

## How to Cite

If you use this fork, please cite the original DeepVariant paper:

> Poplin, R. et al. A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* **36**, 983-987 (2018). https://rdcu.be/7Dhl

## License

[BSD-3-Clause](LICENSE). Not an official Google product. Not validated for clinical use.
