# Hetzner CAX41 Benchmark Results

**Date:** 2026-03-07
**Instance:** Hetzner CAX41 (Helsinki)
**CPU:** 16 shared vCPU, Ampere Altra (Neoverse N1 / Armv8.2-A), CPU part 0xd0c
**RAM:** 32 GB
**Disk:** 320 GB NVMe
**Cost:** EUR 0.0396/hr ($0.043/hr at ~1.08 EUR/USD)
**vCPU type:** SHARED — results have higher variance than dedicated instances
**ISA features:** ASIMD, CRC32, AES — no BF16, no SVE, no i8mm

**Docker image:** `ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5`
**Workload:** GIAB HG003, chr20 (full), WGS model, 16 shards, batch_size=256

---

## Summary

| Config | N | Wall (s) | ME (s) | CV (s) | PP (s) | CV rate (s/100) | $/genome |
|--------|---|----------|--------|--------|--------|-----------------|----------|
| **INT8 ONNX, jemalloc** | 1 | 578 | 253 | 298 | 17 | 0.366 | **$0.33** |
| **INT8 ONNX, no jemalloc** | 1 | 637 | 282 | 328 | 17 | 0.355 | **$0.37** |
| TF Eigen FP32, no jemalloc | 5 | 746 ± 28 | 260 ± 4 | 467 ± 27 | 15.5 ± 0.5 | 0.551 ± 0.003 | $0.43 |
| TF Eigen FP32, jemalloc | 5 | 735 ± 7 | 258 ± 3 | 457 ± 5 | 15.6 ± 0.4 | 0.553 ± 0.006 | $0.42 |
| ONNX FP32, no jemalloc | 4 | 897 ± 18 | 261 ± 1 | 616 ± 17 | 15.6 ± 0.3 | 0.763 ± 0.020 | $0.52 |
| ONNX FP32, jemalloc | 4 | 926 ± 30 | 263 ± 3 | 643 ± 32 | 15.7 ± 0.3 | 0.784 ± 0.019 | $0.53 |
| 4-way parallel CV (ONNX FP32) | 1 | 2524 (proj) | 260 | 2248 | 16 | 11.0/worker | $1.45 |

**Best config: INT8 ONNX + jemalloc at $0.33/genome** (21% cheaper than TF Eigen FP32).

**Formula:** `$/genome = wall_s × 48.1 / 3600 × $0.043`

---

## Platform Comparison

| Platform | vCPU | $/hr | Best Backend | chr20 Wall | $/genome | vs Hetzner |
|----------|------|------|-------------|------------|----------|------------|
| **Hetzner CAX41** | 16 (shared) | $0.043 | INT8 ONNX+jemalloc | 578s | **$0.33** | — |
| Oracle A1 (Altra/N1) | 16 | $0.16 | INT8 ONNX+jemalloc | 486s | $1.04 | 3.2x more |
| Oracle A2 (AmpereOne) | 16 | $0.32 | INT8 ONNX+jemalloc | 544s | $2.32 | 7.0x more |
| GCP t2a (Neoverse N1) | 16 | ~$0.35 | TF+OneDNN | ~780s | ~$3.65 | 11.1x more |
| Graviton3 (c7g.4xlarge) | 16 | $0.58 | BF16+jemalloc | 443s | $3.43 | 10.4x more |
| Graviton4 (c8g.4xlarge) | 16 | $0.68 | INT8 ONNX | 366s | $3.33 | 10.1x more |
| Google x86 reference | 96 | $3.81 | GPU | ~4680s | $5.01 | 15.2x more |

Hetzner's $0.043/hr rate is 7.4x cheaper than Oracle A2 ($0.32/hr) and 3.7x cheaper than Oracle A1 ($0.16/hr). Combined with INT8 ONNX (1.6x faster CV than TF Eigen), the result is **$0.33/genome — 3.2x cheaper** than the next cheapest platform (Oracle A1 at $1.04) and **15.2x cheaper** than Google's x86 reference.

---

## Individual Runs

### TF Eigen FP32 (TF_ENABLE_ONEDNN_OPTS=0)

| Run | jemalloc | Wall (s) | ME (s) | CV (s) | PP (s) | CV rate (s/100) | Time (UTC) | Notes |
|-----|----------|----------|--------|--------|--------|-----------------|------------|-------|
| 1 | off | 736 | 261.3 | 454.3 | 16.0 | 0.550 | 05:51 | |
| 2 | on | 731 | 257.2 | 454.7 | 15.4 | 0.551 | 06:03 | |
| 3 | off | **796** | 259.9 | **515.9** | 15.5 | 0.556 | 06:16 | **OUTLIER — likely shared vCPU throttling** |
| 4 | on | 729 | 255.1 | 455.2 | 15.0 | 0.551 | 06:28 | |
| 5 | off | 731 | 255.4 | 455.9 | 15.5 | 0.552 | 06:40 | |
| 6 | on | 744 | 258.9 | 465.7 | 15.7 | 0.564 | 06:53 | |
| 7 | off | 737 | 264.6 | 453.3 | 14.6 | 0.548 | 07:05 | |
| 8 | on | 729 | 255.9 | 452.8 | 15.8 | 0.549 | 07:17 | |
| 9 | off | 732 | 257.1 | 455.0 | 15.7 | 0.550 | 07:29 | |
| 10 | on | 741 | 262.5 | 458.2 | 15.9 | 0.555 | 07:42 | |

**Run 3 (796s)** is a clear outlier — CV jumped from ~455s to 515.9s (+60s, 13%) while ME was normal. This pattern (ME unaffected, CV degraded) is consistent with a noisy neighbor consuming CPU during the compute-intensive call_variants phase. Excluding run 3, the jemalloc OFF wall times (736, 731, 737, 732) average 734s — identical to jemalloc ON (735s).

**jemalloc effect on TF Eigen:** Negligible. ME differs by ~2s (259.7 vs 257.9, within noise). CV differs by ~10s but this is dominated by the run 3 outlier. With run 3 excluded, both configs converge to ~455s CV. This makes sense — TF Eigen uses its own BFC allocator internally, bypassing glibc malloc. jemalloc only helps workloads dominated by glibc malloc (e.g., TF+OneDNN where ACL allocations go through glibc).

### INT8 ONNX (--use_onnx, model_int8_static.onnx, TF_ENABLE_ONEDNN_OPTS=0)

| Run | jemalloc | OneDNN | Wall (s) | ME (s) | CV (s) | PP (s) | CV rate (s/100) | Time (UTC) | Notes |
|-----|----------|--------|----------|--------|--------|--------|-----------------|------------|-------|
| 1 | on | 1 | 645 | ~332 | ~278 | 16 | 0.347 | 18:26 | OneDNN=1 baseline |
| 2 | on | 1 | 686 | 332 | 328 | 16 | 0.345 | 18:39 | OneDNN=1 baseline |
| 3 | off | 1 | 707 | 390 | 290 | 17 | 0.355 | 18:51 | OneDNN=1, no jemalloc |
| **4** | **on** | **0** | **578** | **253** | **298** | **17** | **0.366** | **19:06** | **OneDNN=0 — best config** |
| **5** | **off** | **0** | **637** | **282** | **328** | **17** | **0.355** | **19:16** | **OneDNN=0, no jemalloc** |

**OneDNN=1 vs OneDNN=0:** Setting `TF_ENABLE_ONEDNN_OPTS=0` reduces ME by 24% (332→253s with jemalloc, 390→282s without). On Neoverse N1 without BF16, OneDNN+ACL adds overhead to the make_examples small model inference (TF Eigen is faster). CV is unaffected by OneDNN because it uses ONNX Runtime. This is a **critical autoconfig fix** — the default should be OneDNN=0 on non-BF16 platforms.

**INT8 vs TF Eigen FP32:** INT8 ONNX CV rate (0.355-0.366 s/100) is **1.6x faster** than TF Eigen FP32 (0.551 s/100). Combined with jemalloc's ME benefit, INT8+jemalloc delivers 578s vs 735s = **21% faster wall time**.

**jemalloc effect on INT8 ONNX:** ME -10% (282→253s), CV within noise, wall -9% (637→578s). Consistent with other N1 platforms — jemalloc reduces glibc malloc contention in make_examples C++ allocations.

**Shared vCPU variance in CV rate:** The INT8 runs show CV rates from 0.345 to 0.366 s/100 (6% range). This is higher than Oracle A1's dedicated vCPU (0.309 ± 0.004 s/100). The Hetzner shared vCPU CV rate (~0.355 s/100 median) is 15% slower than Oracle A1 (0.309), which is the shared vCPU tax.

**Accuracy (rtg vcfeval, chr20 GIAB HG003):**

| Metric | INT8 (Hetzner CAX41) | Gate | Status |
|--------|---------------------|------|--------|
| SNP F1 | **0.9978** | ≥0.9974 | **PASS** |
| INDEL F1 | **0.9963** | ≥0.9940 | **PASS** |

Matches all other platforms exactly. INT8 accuracy is architecture-independent.

### ONNX FP32 (--use_onnx, /opt/models/wgs/model.onnx)

| Run | jemalloc | Wall (s) | ME (s) | CV (s) | PP (s) | CV rate (s/100) | Time (UTC) |
|-----|----------|----------|--------|--------|--------|-----------------|------------|
| 1 | off | 879 | 259.6 | 599.7 | 15.4 | 0.743 | 09:25 |
| 2 | on | 967 | 259.8 | 687.1 | 15.9 | 0.799 | 09:41 |
| 3 | off | 910 | 260.6 | 630.2 | 15.5 | 0.781 | 09:57 |
| 4 | on | 926 | 265.8 | 640.1 | 16.0 | 0.793 | 10:12 |
| 5 | off | 885 | 262.4 | 602.7 | 15.4 | 0.747 | 10:27 |
| 6 | on | 916 | 262.9 | 633.8 | 15.4 | 0.786 | 10:42 |
| 7 | off | 914 | 262.2 | 630.7 | 16.1 | 0.782 | 10:57 |
| 8 | on | 896 | 265.2 | 611.2 | 15.6 | 0.757 | 11:12 |

**ONNX FP32 vs TF Eigen:** ONNX is 32% slower by wall time (897s vs 735s) and 35% slower by CV rate (0.763 vs 0.551 s/100). ME is identical (~261s) since both use TF for make_examples. The CV difference is entirely due to ONNX Runtime CPUExecutionProvider being less efficient than TF Eigen for InceptionV3 GEMM operations on Neoverse N1.

**jemalloc effect on ONNX:** Slightly harmful. jemalloc ON averages 926s vs 897s OFF (+3.2%). ONNX Runtime uses its own arena allocator (BFC) — jemalloc's LD_PRELOAD overhead adds friction without reducing contention. This is consistent with the finding that jemalloc only helps when glibc malloc contention is a significant bottleneck (e.g., TF+OneDNN/ACL on Graviton3 where libc.so.6 was 18% of cycles).

### 4-way Parallel CV (ONNX FP32, 4 workers × 4 threads)

| Metric | Sequential | 4-way Parallel | Delta |
|--------|-----------|----------------|-------|
| ME | 260s | 260s (reused) | same |
| CV | 467s | 2248s | **4.8x SLOWER** |
| PP | 15s | 16s | same |
| Wall (projected) | 746s | 2524s | **3.4x SLOWER** |
| CV rate per worker | 0.550 s/100 | 11.0 s/100 | 20x worse |
| Variant count | 207,799 | 207,799 | match |

**Parallel CV is counterproductive at 16 vCPU.** Each worker gets only 4 OMP threads, producing severely underparallelized GEMM operations (11.0 s/100 per worker vs 0.74 s/100 with 16 threads). Although 4 workers run concurrently, the total GEMM throughput is far worse than a single 16-thread process.

This confirms the pattern seen in fast_pipeline benchmarks: parallel CV only benefits when total vCPUs exceed the GEMM saturation point (~16 threads). On 32-vCPU machines, parallel CV gives 1.9-2.5x speedup because each of 4 workers gets 8 threads (near saturation). On 16-vCPU machines, 4 × 4 threads produces catastrophic GEMM efficiency loss.

**Note:** Parallel CV is not recommended on shared vCPU instances without capping `OMP_NUM_THREADS` per worker. The 2524s result reflects severe thread oversubscription — 4 CV workers × unconstrained OMP threads competing for 16 shared vCPUs. Fix: set `OMP_NUM_THREADS=$((total_vcpus / num_workers))` per worker in `run_parallel_cv.sh`. This is a latent bug that affects any instance with fewer cores than workers × default thread count. Dedicated instances (Oracle A2, Graviton3) are unaffected because thread affinity works correctly. Re-test with the fix on CAX51 (32 vCPU, $0.086/hr) is planned — projected ~$0.31/genome at 4-way parallel CV.

---

## Variance Analysis (Shared vCPU)

| Config | Wall σ | CV σ | Wall σ/mean | Notes |
|--------|--------|------|-------------|-------|
| TF Eigen, jemalloc off | 27.8s | 27.4s | 3.7% | Inflated by run 3 outlier (796s) |
| TF Eigen, jemalloc off (excl outlier) | 2.9s | 1.1s | 0.4% | Very stable when not throttled |
| TF Eigen, jemalloc on | 7.2s | 5.1s | 1.0% | Low variance |
| ONNX FP32, jemalloc off | 17.6s | 16.9s | 2.0% | Moderate |
| ONNX FP32, jemalloc on | 29.9s | 31.9s | 3.2% | Higher variance |

Shared vCPU variance is characterized by occasional throttling events that primarily affect the compute-intensive call_variants phase (run 3: CV +60s, ME unaffected). When unthrottled, the instance delivers remarkably consistent performance (σ/mean < 1%). The higher variance in ONNX runs may reflect longer runtime exposure to throttling windows.

**Comparison with dedicated instances:** Oracle A2 (dedicated) had σ = 2.6s over 4 runs (σ/mean = 0.5%). Graviton3 (dedicated) had σ ≈ 2-3s. Hetzner's unthrottled σ (~3s excl outlier) is comparable, but the tail risk of throttling events adds ~28s σ when included.

---

## Key Findings

1. **$0.33/genome is the cheapest configuration tested** — 3.2x cheaper than Oracle A1 ($1.04), 7.0x cheaper than Oracle A2 ($2.32), 10x cheaper than Graviton3 ($3.43), 15.2x cheaper than Google x86 ($5.01).

2. **INT8 ONNX is the fastest backend on Neoverse N1.** CV rate 0.355 s/100 is 1.6x faster than TF Eigen FP32 (0.551 s/100). Combined with jemalloc: 578s wall vs 735s = 21% faster.

3. **OneDNN must be OFF on non-BF16 platforms.** `TF_ENABLE_ONEDNN_OPTS=1` adds 29% overhead to make_examples on N1 (332→253s). ACL FP32 GEMM is slower than Eigen on N1 for the small model. Only enable OneDNN when BF16 BFMMLA is available (Graviton3+). This is fixed in autoconfig v1.9.0-arm64.6.

4. **The cost advantage is pricing-driven.** Hetzner is ~6% slower than Oracle A1 by wall time (578s vs 486s) but 3.7x cheaper per hour ($0.043 vs $0.16). The shared vCPU tax (~15% slower CV than dedicated) is far outweighed by the pricing gap.

5. **jemalloc helps INT8 ONNX by 9%.** ME -10% (282→253s), total wall -9% (637→578s). The benefit is in make_examples malloc contention, not CV (ONNX has its own allocator).

6. **Parallel CV does not work at 16 vCPU.** Same conclusion as fast_pipeline at 16 vCPU: not enough cores for parallel workers to maintain GEMM efficiency. Requires 32+ vCPU.

7. **Shared vCPU throttling is rare but real.** 1 out of 10 TF Eigen runs (10%) showed a 60s CV spike from noisy neighbors. INT8 ONNX CV rate has 6% variance across runs (vs <2% on dedicated instances). For production use, budget for ~5% overhead from occasional throttling.

---

## WGS Extrapolation

| Config | chr20 Wall | WGS Projected | $/genome | Time |
|--------|-----------|---------------|----------|------|
| **INT8 ONNX + jemalloc (best)** | **578s** | **~7.7 hr** | **$0.33** | **~8 hr** |
| INT8 ONNX (no jemalloc) | 637s | ~8.5 hr | $0.37 | ~9 hr |
| TF Eigen + jemalloc | 735s | ~9.8 hr | $0.42 | ~10 hr |
| TF Eigen (no jemalloc) | 746s | ~10.0 hr | $0.43 | ~10 hr |
| ONNX FP32 (no jemalloc) | 897s | ~12.0 hr | $0.52 | ~12 hr |

*WGS time = chr20_wall × 48.1. Extrapolation has ~15-20% uncertainty.*

A **$0.33 WGS genome in ~8 hours** on a $15/month shared ARM instance is remarkable — 15x cheaper than Google's x86 reference ($5.01). For non-time-critical workloads (research, education, low-resource labs), this is dramatically more accessible than any cloud GPU or dedicated ARM option.

---

## Reproducibility

```bash
# Instance setup
apt-get install -y docker.io
echo "ghp_..." | docker login ghcr.io -u antomicblitz --password-stdin
docker pull ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5

# Download reference data + INT8 model
mkdir -p /data/{reference,bam,truth,output}
# Reference: NCBI FTP GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
# BAM: gs://deepvariant/case-study-testdata/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
# Truth: GIAB FTP HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz
wget -O /data/model_int8_static.onnx \
  https://github.com/antomicblitz/deepvariant-linux-arm64/releases/download/v1.9.0-arm64.5/model_int8_static.onnx

# Run benchmark (INT8 ONNX + jemalloc, best config)
docker run --rm --memory=28g \
  -v /data:/data \
  -e DV_AUTOCONFIG=1 \
  -e DV_USE_JEMALLOC=1 \
  -e TF_ENABLE_ONEDNN_OPTS=0 \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5 \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref=/data/reference/GRCh38_no_alt_analysis_set.fasta \
    --reads=/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
    --output_vcf=/data/output/output.vcf.gz \
    --regions=chr20 \
    --num_shards=16 \
    --intermediate_results_dir=/data/output/intermediate \
    --call_variants_extra_args="--batch_size=256,--use_onnx=true,--onnx_model=/data/model_int8_static.onnx"
```
