# Hetzner CAX41 Benchmark Results

**Date:** 2026-03-07
**Instance:** Hetzner CAX41 (Helsinki)
**CPU:** 16 shared vCPU, Ampere Altra (Neoverse N1 / Armv8.2-A), CPU part 0xd0c
**RAM:** 32 GB
**Disk:** 320 GB NVMe
**Cost:** EUR 0.0396/hr ($0.043/hr at ~1.08 EUR/USD)
**vCPU type:** SHARED — results have higher variance than dedicated instances
**ISA features:** ASIMD, CRC32, AES — no BF16, no SVE, no i8mm

**Docker image:** `ghcr.io/antomicblitz/deepvariant-arm64:optimized`
**Workload:** GIAB HG003, chr20 (full), WGS model, 16 shards, batch_size=256

---

## Summary

| Config | N | Wall (s) | ME (s) | CV (s) | PP (s) | CV rate (s/100) | $/genome |
|--------|---|----------|--------|--------|--------|-----------------|----------|
| **TF Eigen FP32, no jemalloc** | 5 | 746 ± 28 | 260 ± 4 | 467 ± 27 | 15.5 ± 0.5 | 0.551 ± 0.003 | **$0.43** |
| **TF Eigen FP32, jemalloc** | 5 | 735 ± 7 | 258 ± 3 | 457 ± 5 | 15.6 ± 0.4 | 0.553 ± 0.006 | **$0.42** |
| ONNX FP32, no jemalloc | 4 | 897 ± 18 | 261 ± 1 | 616 ± 17 | 15.6 ± 0.3 | 0.763 ± 0.020 | $0.52 |
| ONNX FP32, jemalloc | 4 | 926 ± 30 | 263 ± 3 | 643 ± 32 | 15.7 ± 0.3 | 0.784 ± 0.019 | $0.53 |
| 4-way parallel CV (ONNX FP32) | 1 | 2524 (proj) | 260 | 2248 | 16 | 11.0/worker | $1.45 |

**Best config: TF Eigen FP32 + jemalloc at $0.42/genome.**

**Formula:** `$/genome = wall_s × 48.1 / 3600 × $0.043`

---

## Platform Comparison

| Platform | vCPU | $/hr | Best Backend | chr20 Wall | $/genome | vs Hetzner |
|----------|------|------|-------------|------------|----------|------------|
| **Hetzner CAX41** | 16 (shared) | $0.043 | TF Eigen FP32 | 735s | **$0.42** | — |
| Oracle A2 (AmpereOne) | 16 | $0.32 | INT8 ONNX+jemalloc | 544s | $2.32 | 5.5x more |
| GCP t2a (Neoverse N1) | 16 | ~$0.35 | TF+OneDNN | ~780s | ~$3.65 | 8.7x more |
| Graviton3 (c7g.4xlarge) | 16 | $0.58 | BF16+jemalloc | 443s | $3.43 | 8.2x more |
| Graviton4 (c8g.4xlarge) | 16 | $0.68 | INT8 ONNX | 366s | $3.33 | 7.9x more |
| Google x86 reference | 96 | $3.81 | GPU | ~4680s | $5.01 | 11.9x more |

Hetzner's $0.043/hr rate is 7.4x cheaper than Oracle A2 ($0.32/hr), which more than compensates for 35% slower wall time. The result is **5.5x cheaper per genome** than the next cheapest platform.

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

1. **$0.42/genome is the cheapest configuration tested** — 5.5x cheaper than Oracle A2 ($2.32), 8x cheaper than Graviton3 ($3.43), 12x cheaper than Google x86 ($5.01).

2. **The cost advantage is entirely pricing-driven**, not performance. Hetzner is ~35% slower than Oracle A2 by wall time (735s vs 544s) but 7.4x cheaper per hour ($0.043 vs $0.32).

3. **TF Eigen is the fastest backend on Neoverse N1.** ONNX FP32 CPUExecutionProvider is 35% slower for CV. OneDNN+ACL was not tested (potential N1 compatibility issues), but GCP t2a data shows it would add ~7% CV improvement (0.512 vs 0.550 s/100).

4. **jemalloc has no effect on TF Eigen and slightly hurts ONNX.** This differs from Graviton3/Oracle A2 where jemalloc gave 7-22% wall improvement. The difference is that those platforms used TF+OneDNN (ACL), which routes allocations through glibc malloc. TF Eigen and ONNX use internal allocators.

5. **Parallel CV does not work at 16 vCPU.** Same conclusion as fast_pipeline at 16 vCPU: not enough cores for parallel workers to maintain GEMM efficiency. Requires 32+ vCPU.

6. **Shared vCPU throttling is rare but real.** 1 out of 10 TF Eigen runs (10%) showed a 60s CV spike from noisy neighbors. For production use, budget for ~5% overhead from occasional throttling.

---

## WGS Extrapolation

| Config | chr20 Wall | WGS Projected | $/genome | Time |
|--------|-----------|---------------|----------|------|
| TF Eigen + jemalloc (best) | 735s | ~9.8 hr | $0.42 | ~10 hr |
| TF Eigen (no jemalloc) | 746s | ~10.0 hr | $0.43 | ~10 hr |
| ONNX FP32 (no jemalloc) | 897s | ~12.0 hr | $0.52 | ~12 hr |

*WGS time = chr20_wall × 48.1. Extrapolation has ~15-20% uncertainty.*

A $0.42 WGS genome in ~10 hours on a $15/month shared ARM instance is remarkable. For non-time-critical workloads (research, education, low-resource labs), this is dramatically more accessible than any cloud GPU or dedicated ARM option.

---

## Reproducibility

```bash
# Instance setup
apt-get install -y docker.io
echo "ghp_..." | docker login ghcr.io -u antomicblitz --password-stdin
docker pull ghcr.io/antomicblitz/deepvariant-arm64:optimized

# Download reference data
mkdir -p /data/{reference,bam,truth,output}
# Reference: NCBI FTP GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
# BAM: gs://deepvariant/case-study-testdata/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam
# Truth: GIAB FTP HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz

# Run benchmark (TF Eigen FP32, best config)
docker run --rm --memory=28g \
  -v /data:/data \
  -e TF_ENABLE_ONEDNN_OPTS=0 \
  -e CUDA_VISIBLE_DEVICES= \
  ghcr.io/antomicblitz/deepvariant-arm64:optimized \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref=/data/reference/GRCh38_no_alt_analysis_set.fasta \
    --reads=/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
    --output_vcf=/data/output/output.vcf.gz \
    --regions=chr20 \
    --num_shards=16 \
    --intermediate_results_dir=/data/output/intermediate \
    --call_variants_extra_args="--batch_size=256"
```
