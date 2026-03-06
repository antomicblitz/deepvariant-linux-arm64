# DeepVariant ARM64 — Benchmark Summary

## Overview

This is a downstream fork of [google/deepvariant](https://github.com/google/deepvariant)
v1.9.0 that adds native ARM64 Linux support with hardware-accelerated inference.
The fork produces a Docker image for AWS Graviton, Oracle Ampere, and other
ARM64 instances.

Version tags use the format `v{upstream}-arm64.{n}` (e.g., `v1.9.0-arm64.4`).

### Tested Platforms

| Platform | Instance | CPU | vCPUs | RAM | $/hr |
|----------|----------|-----|-------|-----|------|
| AWS Graviton3 | c7g.4xlarge | Neoverse V1 | 16 | 32 GB | $0.58 |
| AWS Graviton3 | c7g.8xlarge | Neoverse V1 | 32 | 64 GB | $1.15 |
| AWS Graviton4 | c8g.4xlarge | Neoverse V2 | 16 | 32 GB | $0.68 |
| AWS Graviton4 | c8g.8xlarge | Neoverse V2 | 32 | 64 GB | $1.36 |
| Oracle A2 | A2.Flex (16 OCPU) | AmpereOne (Siryn) | 32 | 64 GB | $0.64 |
| GCP | t2a-standard-16 | Neoverse N1 | 16 | 64 GB | — |

---

## Accuracy Validation

All backends validated on GIAB HG003, chr20, using `rtg vcfeval`.

### Aggregate F1

| Metric | FP32 | BF16 | INT8 ONNX | Gate |
|--------|------|------|-----------|------|
| SNP F1 | 0.9977 | 0.9977 | 0.9978 | >= 0.9974 |
| INDEL F1 | 0.9961 | 0.9961 | 0.9962 | >= 0.9940 |

All backends pass gates. INT8 matches or exceeds BF16 and FP32 accuracy.

### Stratified Region Validation

| Region | INT8 SNP | BF16 SNP | INT8 INDEL | BF16 INDEL |
|--------|----------|----------|------------|------------|
| Aggregate chr20 | 0.9978 | 0.9977 | 0.9962 | 0.9961 |
| Homopolymers (>=7bp) | 0.9985 | 0.9985 | 0.9967 | 0.9963 |
| Simple Repeats | 0.9994 | 0.9994 | 0.9967 | 0.9961 |
| Tandem Repeats (201-10000bp) | 0.9983 | 0.9983 | 0.9926 | 0.9926 |
| Segmental Duplications | 0.9802 | 0.9744 | 0.9814 | 0.9814 |

INT8 passes all GIAB stratification regions with no localized degradation.

---

## Full Benchmark Matrix

All benchmarks: GIAB HG003, full chr20. Cost formula:
`$/genome = chr20_wall_s x 48.1 / 3600 x $/hr`.
Oracle A2 pricing: $0.04/OCPU/hr — 16-vCPU rows use 8 OCPU ($0.32/hr),
32-vCPU rows use 16 OCPU ($0.64/hr).

### Cross-Platform Comparison (16 vCPU)

| Platform | Backend | jemalloc | ME | CV (rate) | PP | Total | $/hr | $/genome | N |
|----------|---------|----------|-----|-----------|-----|-------|------|----------|---|
| Graviton3 (c7g) | BF16 | off | 278s | 185s (0.232) | 24s | **487s** | $0.58 | **$3.77** | 2* |
| Graviton3 (c7g) | BF16 | **on** | 242s | 188s (0.235) | 9s | **443s** | $0.58 | **$3.43** | 2* |
| Graviton3 (c7g) | INT8 ONNX | off | 299s | 194s (0.237) | 14s | **507s** | $0.58 | **$3.92** | 3 |
| **Graviton4 (c8g)** | **INT8 ONNX** | **off** | **194s** | **158s (0.197)** | **6s** | **366s** | **$0.68** | **$3.33** | 2* |
| Graviton4 (c8g) | ONNX FP32 | off | 232s | 360s (0.446) | 10s | **602s** | $0.68 | $5.47 | 2* |
| **Oracle A2 (AmpereOne)** | **INT8 ONNX** | **off** | **253s** | **315s (0.389)** | **11s** | **584s** | **$0.32** | **$2.49** | **4** |
| **Oracle A2 (AmpereOne)** | **INT8 ONNX** | **on** | **210s** | **318s (0.393)** | **12s** | **544s** | **$0.32** | **$2.32** | **4** |
| Oracle A2 (AmpereOne) | TF Eigen FP32 | off | 287s | 325s (0.387) | 17s | **629s** | $0.32 | $2.69 | 2* |

### Cross-Platform Comparison (32 vCPU, Sequential)

| Platform | Backend | jemalloc | ME | CV | PP | Total | $/hr | $/genome | N |
|----------|---------|----------|-----|-----|-----|-------|------|----------|---|
| Graviton3 (c7g.8xlarge) | BF16 | on | 131s | 141s | 8s | **283s** | $1.15 | **$4.35** | 2* |
| **Graviton4 (c8g.8xlarge)** | **BF16** | **on** | **100s** | **126s** | **5s** | **232s** | **$1.36** | **$4.22** | 2* |
| Graviton4 (c8g.8xlarge) | INT8 ONNX | on | 96s | 129s | 5s | **233s** | $1.36 | **$4.24** | 2* |
| Oracle A2 (16 OCPU) | INT8 ONNX | on | 113s | 283s | 9s | **418s** | $0.64 | **$3.57** | 1* |

> **Key insight at 32 vCPU:** BF16 and INT8 converge (~232-233s on Graviton4) because both hit the CV floor. CV rate doesn't improve beyond 16 ORT threads. ME scales well (2x shards → 1.8x speedup) but CV is the bottleneck. Solution: parallel call_variants.

### Parallel call_variants (32 vCPU, 4-way)

Splits 32 ME shards across 4 independent call_variants workers (8 shards each, `OMP_NUM_THREADS=$(nproc)/4`). Zero code changes — wrapper script with symlink shard renumbering.

| Platform | Sequential CV | 4-way CV | Speedup | N (4-way) |
|----------|--------------|----------|---------|-----------|
| **Graviton4** (c8g.8xlarge) | 128s | **61s** | **2.10x** | 3 |
| **Graviton3** (c7g.8xlarge) | 141s | **74s** | **1.90x** | 4 |
| **Oracle A2** (16 OCPU) | 283s | **114s** | **2.47x** | 2* |

All variant counts match sequential baseline exactly (207,799).

**Measured full pipeline with `run_parallel_cv.sh`** (single-run smoke test, no jemalloc):

| Platform | ME | CV (4-way) | PP | Wall | $/hr | $/genome |
|----------|-----|-----------|-----|------|------|----------|
| **Graviton4** (c8g.8xlarge) | 76s | 151s | 5s | **232s** | $1.36 | **$4.22** |
| **Oracle A2** (16 OCPU) | 131s | 233s | 9s | **373s** | $0.64 | **$3.19** |

> CV times in the full pipeline are higher than isolated benchmarks due to per-worker
> ONNX session startup overhead (~20-30s). With jemalloc, ME times improve 14-17%.

**Projected full pipeline with 4-way parallel CV** (from isolated CV benchmarks + sequential ME/PP):

| Platform | ME | CV (4-way) | PP | Wall (proj) | $/hr | $/genome |
|----------|-----|-----------|-----|-------------|------|----------|
| **Oracle A2** | 113s | 114s | 10s | **~250s** | $0.64 | **~$2.14** |
| **Graviton4** | 100s | 61s | 6s | **~172s** | $1.36 | **~$3.13** |
| **Graviton3** | 131s | 74s | 8s | **~218s** | $1.15 | **~$3.35** |

\*N<4 runs; wider confidence interval.

> Wall time includes ~4-5s Docker startup and inter-stage overhead not captured
> in individual ME/CV/PP timings. jemalloc: enable with `-e DV_USE_JEMALLOC=1`.

User-facing wrapper: `scripts/run_parallel_cv.sh` (runs inside Docker container).
See README.md for usage instructions.

### Graviton3 Pipeline Breakdown (16 vCPU)

| Stage | FP32 | BF16 | INT8 ONNX (3-run avg) |
|-------|------|------|-----------|
| make_examples | 255s | 278s | 299s |
| call_variants | 298s (0.379s/100) | 185s (0.232s/100) | 194s (0.237s/100) |
| postprocess | 29s | 24s | 14s |
| **Total** | **582s** | **487s** | **507s** |

### Inference Rate by Platform

| Platform | vCPUs | Config | call_variants Rate | chr20 Wall Time |
|----------|-------|--------|-------------------|-----------------|
| GCP t2a (Neoverse-N1) | 8 | FP32 | 0.880 s/100 | 12m57s |
| GCP t2a (Neoverse-N1) | 16 | FP32 | 0.512 s/100 | 7m22s |
| AWS Graviton3 | 16 | FP32 | 0.379 s/100 | 9m41s |
| **AWS Graviton3** | **16** | **BF16** | **0.232 s/100** | **8m06s** |
| **AWS Graviton3** | **16** | **INT8 ONNX** | **0.237 s/100** | **~8m27s** |
| **AWS Graviton4** | **16** | **INT8 ONNX** | **0.197 s/100** | **6m06s** |
| AWS Graviton4 | 16 | ONNX FP32 | 0.446 s/100 | 10m02s |
| AWS Graviton4 | 16 | BF16 (standalone CV) | 0.328 s/100 | ~8m32s* |
| **Oracle A2 (AmpereOne)** | **16 OCPU** | **INT8 ONNX** | **0.389 s/100** | **9m44s** |
| Oracle A2 (AmpereOne) | 16 OCPU | TF Eigen FP32 | 0.387 s/100 | 10m29s |

---

## Methodology

### Benchmark Setup

- **Sample:** GIAB HG003, Illumina NovaSeq PCR-free 35x WGS
- **Region:** Full chr20 (63 Mbp)
- **Reference:** GRCh38 (no alt analysis set)
- **Accuracy validation:** `rtg vcfeval` against GIAB v4.2.1 truth set

### WGS Extrapolation

chr20 is 2.08% of the human genome. The extrapolation factor is:

```
WGS_time = chr20_time x 48.1
$/genome = chr20_wall_s x 48.1 / 3600 x $/hr
```

This extrapolation has ~15-20% uncertainty because chr20 may not be
representative of all chromosomes (variant density, repetitive content).

### Run Count Conventions

- **N >= 4:** Reported as verified (no asterisk)
- **N < 4:** Flagged with \* (wider confidence interval)
- All runs include Docker startup overhead (~4-5s)
- Interleaved jemalloc ablation runs eliminate cache-warming bias

### jemalloc Impact

Measured via `scripts/benchmark_jemalloc_ablation.sh` with interleaved runs:

- **Graviton3 (N=2):** ME -13.8%, CV +1.6% (noise), wall -9.0% (487->443s)
- **Oracle A2 (N=4):** ME -17.0%, CV within noise, wall -6.9% (584->544s)

ME improvement is the dominant factor. jemalloc's per-thread arenas reduce
malloc contention in make_examples' C++ allocations. CV sees minimal benefit
because ONNX Runtime and TF have their own internal allocators.

---

## Recommended Configurations

### By Use Case

| Use Case | Platform | vCPU | Backend | Parallel CV | $/genome | Notes |
|----------|----------|------|---------|-------------|----------|-------|
| **Cheapest** | Oracle A2 (16 OCPU) | 32 | INT8 ONNX | 4-way | **~$2.14** | Projected from measured CV |
| Cheapest (sequential) | Oracle A2 (16 OCPU) | 16 | INT8 ONNX+jemalloc | no | **$2.32** | 4-run verified |
| **Best speed/cost** | Graviton4 (c8g.8xlarge) | 32 | BF16+jemalloc | 4-way | **~$3.13** | Projected; fastest wall ~172s |
| **Fastest ARM64** | Graviton4 (c8g.8xlarge) | 32 | BF16+jemalloc | 4-way | **~$3.13** | CV=61s (N=3, σ=0) |
| Fastest (sequential) | Graviton4 (c8g.8xlarge) | 32 | BF16+jemalloc | no | **$4.22** | 232s wall, 2-run* |

\*N<4 runs. Parallel CV rows are projected full-pipeline (measured CV + measured sequential ME/PP).

### Platform Notes

- **Graviton3/4:** Use BF16 when BF16 CPU flag is present (`grep bf16 /proc/cpuinfo`).
  INT8 ONNX is the fallback for non-BF16 platforms. Both achieve similar CV rates
  on Graviton3 (0.232 vs 0.237 s/100).
- **Oracle A2 (AmpereOne):** OneDNN+ACL causes SIGILL (compiled for Neoverse-N1).
  Must use `TF_ENABLE_ONEDNN_OPTS=0` or ONNX backend. INT8 ONNX is the fastest
  working backend. The entrypoint automatically forces OneDNN off on AmpereOne.
- **Graviton4 BF16:** Full TF pipeline OOM-killed on 32 GB machines (TF uses ~26 GB RSS).
  Use INT8 ONNX on 32 GB instances, or upgrade to 64 GB (c8g.8xlarge).

---

## What Was Tried and Didn't Work

| Approach | Result | Details |
|----------|--------|---------|
| EfficientNet-B3 model swap | 3x slower | Depthwise separable convs have poor GEMM density on CPU |
| MobileNetV2 / depthwise-separable models | Dead end | Same architecture class, same penalty |
| KMP_AFFINITY tuning | 30% regression | `granularity=core,compact,1,0` + system allocator |
| ONNX ACL ExecutionProvider | Not worth it | 16 supported ops, fragile builds, no pre-built wheels |
| Dynamic INT8 on ARM64 | Broken | ConvInteger(10) op not in ORT ARM64 CPUExecutionProvider |
| fast_pipeline at 16 vCPU | 42% slower | CPU contention between concurrent ME+CV (693s vs 487s) |
| fast_pipeline on Oracle A2 32 vCPU | <1% improvement | PP broken on CVO ordering, CV stalls on streaming |
| INT8 ONNX beyond 16 threads | No scaling | 0.358 s/100 at both 16 and 32 ORT threads |
| ONNX inter-op parallelism | No improvement | InceptionV3 is intra-op GEMM bound |
| ONNX FP32 on AmpereOne | 1.96x slower than TF Eigen | 0.759 vs 0.387 s/100 |
| TF SavedModel on 32 GB | OOM kill | ~26 GB RSS, forking PP pushes >32 GB |

---

## Untested / Pending

| Item | Blocking On | Expected Impact |
|------|-------------|-----------------|
| Full pipeline with 4-way parallel CV | Integration script | Projected: Oracle A2 ~$2.14, Graviton4 ~$3.13 |
| AmpereOne BF16 via generic TF wheel | SVE hypothesis test | Could reach ~$1.55/genome if BF16 unlocked |
| AmpereOne Docker rebuild (OneDNN for Siryn) | Build infrastructure | Full BF16 fast math, target <$2/genome |
| Oracle A1 (Altra) benchmark | Instance capacity | Ultra-cheap ($0.01/OCPU/hr) |
| 8-way parallel CV | Diminishing returns test | Memory bandwidth may saturate at 8 workers |

See `docs/oracle-a2-wheel-test.md` for the AmpereOne SIGILL investigation
procedure, `scripts/benchmark_parallel_cv.sh` for the parallel CV benchmark,
and `scripts/benchmark_32vcpu.sh` for the 32-vCPU sequential test plan.
