## DeepVariant ARM64 v1.9.0-arm64.3

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**Parallel call_variants benchmarks** — proves that splitting CV into 4 workers
gives 1.9-2.5x CV speedup at 32 vCPU. Benchmark script only (not a user-facing
tool). User-facing wrapper shipped in v1.9.0-arm64.4.

**32-vCPU benchmarks** — Graviton3 (c7g.8xlarge), Graviton4 (c8g.8xlarge),
and Oracle A2 (16 OCPU). BF16 and INT8 converge at 32 vCPU (~232s on
Graviton4) because CV rate doesn't improve beyond 16 ORT threads.

**Corrected benchmark data** — fixed $/genome calculation errors in Graviton4
ONNX FP32 ($5.07->$5.47) and BF16 standalone (~$4.31->~$4.65). INT8 Graviton3
rate updated to 3-run average (0.237 s/100).

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| Cheapest (projected) | Oracle A2 (16 OCPU, 32 vCPU) | INT8 + jemalloc + 4-way CV | ~$2.14 |
| Cheapest (sequential, verified) | Oracle A2 (8 OCPU, 16 vCPU) | INT8 + jemalloc | $2.32 |
| Fastest ARM64 (projected) | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc + 4-way CV | ~$3.13 |
| Fastest sequential | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc | $4.22* |

\*N<4 runs. Parallel CV rows are projected (measured CV + measured sequential ME/PP).

### Quick start

```bash
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.3 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Accuracy

All backends pass GIAB HG003 gates (SNP F1 >= 0.9974, INDEL F1 >= 0.9940)
including stratified validation across homopolymers, tandem repeats, and
segmental duplications.

### Known issues (fixed in v1.9.0-arm64.4)

- **Parallel CV is benchmark-only** — `scripts/benchmark_parallel_cv.sh` is
  hardcoded for HG003/chr20 testing. A user-facing wrapper
  (`scripts/run_parallel_cv.sh`) ships in v1.9.0-arm64.4.
- **`DV_AUTOCONFIG=1` does not work** — inherited from arm64.2. Fixed in arm64.4.

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
