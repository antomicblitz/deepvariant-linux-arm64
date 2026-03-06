## DeepVariant ARM64 v1.9.0-arm64.2

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**jemalloc allocator integration** — reduces glibc malloc contention under
parallel shards. Verified 14-17% make_examples speedup on Graviton3 and
AmpereOne. Enable with `-e DV_USE_JEMALLOC=1`.

**CPU-aware autoconfig** — automatically selects backend, thread counts, and
safety settings for your ARM64 CPU. Run `scripts/autoconfig.sh` or enable
with `-e DV_AUTOCONFIG=1`.

**Verified benchmark data** — all cost/performance numbers are now 4-run means
with standard deviations. Previous 2-run estimates corrected.

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| Cheapest | Oracle A2 (16 OCPU) | INT8 + jemalloc | $2.32 |
| Best speed/cost | Graviton3 (16 vCPU) | BF16 + jemalloc | $3.43* |
| Fastest ARM64 | Graviton4 (16 vCPU) | INT8 + jemalloc | ~$3.03* |

*jemalloc gain on Graviton3/4 projected from verified AmpereOne data;
Graviton3 N=2 confirmation in progress.

### Quick start

```bash
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2 \
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

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
