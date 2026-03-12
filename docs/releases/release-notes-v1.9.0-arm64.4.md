## DeepVariant ARM64 v1.9.0-arm64.4

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**User-facing parallel call_variants** — `run_parallel_cv.sh` is a drop-in
replacement for `run_deepvariant` that splits call_variants into N parallel
workers. 1.9-2.5x CV speedup on 32+ vCPU machines with ONNX backend.
Runs inside the Docker container — no host-level orchestration needed.

**Fixed autoconfig** — `DV_AUTOCONFIG=1` now works. `autoconfig.sh` was missing
from the Docker image in arm64.2 and arm64.3 (silently non-functional). The
script is now correctly included at `/opt/deepvariant/scripts/autoconfig.sh`.

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| Cheapest (projected) | Oracle A2 (16 OCPU, 32 vCPU) | INT8 + jemalloc + 4-way CV | ~$2.14 |
| Cheapest (sequential, verified) | Oracle A2 (8 OCPU, 16 vCPU) | INT8 + jemalloc | $2.32 |
| Fastest ARM64 (projected) | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc + 4-way CV | ~$3.13 |
| Fastest sequential | Graviton4 (c8g.8xlarge, 32 vCPU) | BF16 + jemalloc | $4.22* |

\*N<4 runs. Parallel CV rows are projected (measured CV + measured sequential ME/PP).

### Quick start — parallel call_variants (32+ vCPU, recommended)

Splits call_variants into N parallel workers for 1.9-2.5x CV speedup.
Requires ONNX backend (used automatically) and 32+ vCPU.

```bash
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=56g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.4 \
  /opt/deepvariant/scripts/run_parallel_cv.sh \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=32 \
  --num_cv_workers=4
```

**Requirements:**
- `--num_shards` must be divisible by `--num_cv_workers`
- `--memory=56g` recommended for 4 workers (vs 28g for sequential)
- ONNX backend only (TF SavedModel uses ~26 GB/worker — would OOM)

**Additional flags:** `--regions`, `--batch_size`, `--onnx_model`,
`--sample_name`, `--output_gvcf`, `--postprocess_cpus`,
`--intermediate_results_dir`, `--customized_model`. Run with `--help`
for full usage.

### Quick start — sequential (16 vCPU or simpler setup)

```bash
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.4 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"
```

### Smoke test results

Both platforms tested with full chr20 (GIAB HG003), 32 shards, 4-way CV:

| Platform | ME | CV (4-way) | PP | Total | Variants |
|----------|-----|-----------|-----|-------|----------|
| Graviton4 (c8g.8xlarge) | 76s | 151s | 5s | **232s** | 207,799 |
| Oracle A2 (16 OCPU) | 131s | 233s | 9s | **373s** | 207,799 |

Variant counts match sequential baseline exactly.

### Accuracy

All backends pass GIAB HG003 gates (SNP F1 >= 0.9974, INDEL F1 >= 0.9940)
including stratified validation across homopolymers, tandem repeats, and
segmental duplications.

### Fixes from previous releases

- `DV_AUTOCONFIG=1` now works (was silently broken in arm64.2 and arm64.3)
- Parallel CV now has a user-facing wrapper (was benchmark-only in arm64.3)

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
