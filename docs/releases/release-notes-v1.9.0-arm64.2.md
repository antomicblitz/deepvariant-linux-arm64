## DeepVariant ARM64 v1.9.0-arm64.2

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**CPU-aware autoconfig** — automatically selects backend, thread counts, and
safety settings for your ARM64 CPU. Run `scripts/autoconfig.sh` or enable
with `-e DV_AUTOCONFIG=1`.

**jemalloc allocator integration** — reduces glibc malloc contention under
parallel shards. Verified 14-17% make_examples speedup on Graviton3 and
AmpereOne. Enable with `-e DV_USE_JEMALLOC=1`.

**Verified benchmark data** — all cost/performance numbers include run counts
and asterisks for N<4. Oracle A2 INT8+jemalloc verified at N=4.

**AmpereOne safety** — entrypoint auto-disables OneDNN on AmpereOne to prevent
SIGILL. Autoconfig enforces this as a hard safety rule.

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| Cheapest (verified) | Oracle A2 (8 OCPU, 16 vCPU) | INT8 + jemalloc | $2.32 |
| Fastest sequential | Graviton4 (c8g.4xlarge, 16 vCPU) | INT8 ONNX | $3.33* |
| Best value (Graviton3) | Graviton3 (c7g.4xlarge, 16 vCPU) | BF16 + jemalloc | $3.43* |

\*N<4 runs.

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

### Known issues (fixed in v1.9.0-arm64.4)

- **`DV_AUTOCONFIG=1` does not work** — `autoconfig.sh` was not included in the
  Docker image. The entrypoint silently skips it. Fixed in v1.9.0-arm64.4.
- `DV_USE_JEMALLOC=1` works correctly.

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
