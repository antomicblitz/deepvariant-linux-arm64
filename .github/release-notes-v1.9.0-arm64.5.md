## DeepVariant ARM64 v1.9.0-arm64.5

Compatible with google/deepvariant v1.9.0. Native ARM64 Linux build with
hardware-accelerated inference.

### What's new in this release

**Oracle A1: $0.80/genome (new cost leader)** — Oracle A1 (Ampere Altra,
Neoverse N1) benchmarked at **$0.80/genome** with 4-way parallel CV + INT8 +
jemalloc (3-run avg, 376s chr20). Sequential: $1.04/genome (486s). 6.3x cheaper
than Google's x86 reference ($5.01). See `docs/oracle-a1-benchmark.md`.

**INT8 quantization tools in Docker image** — `quantize_model` command now
available inside the container. Run `quantize_model --help` for usage. Quantize
the FP32 ONNX model to INT8 using calibration data from your make_examples
output. One-time step (~2 minutes), then use the INT8 model for all subsequent
runs with 2.3x faster call_variants inference.

**`--nocompress_intermediates` flag** — Skip gzip compression for TFRecord
intermediates. Saves ~4% make_examples time on fast storage (NVMe/tmpfs).
Trade-off: ~12 GB vs ~3 GB intermediates for chr20. Default: off.

**AmpereOne safety improvements** — OneDNN now disabled for ALL binaries on
AmpereOne (was previously only call_variants). Prevents SIGILL in make_examples
under high concurrency. BF16 on AmpereOne confirmed permanently blocked after
investigating 4 cascading ACL/OneDNN bugs. See `docs/onednn-ampereone.md`.

**Docker build fixes** — 3 critical fixes that allow building from source:
- cryptography ABI mismatch between builder and runtime stages
- pip install ordering (cryptography must come after build-prereq)
- httplib2 pinned to <0.22 (pyparsing 2.2.2 compatibility)

**C++ code improvements** — Direct TFRecord serialization (bypasses
tf::Example proto), reusable pileup buffer, `absl::StrCat` for string
formatting. Benchmarked at 0% measurable impact but cleaner code.

**ACL v23.08 patches committed** — SVE filter and OneDNN indirect GEMM patches
for AmpereOne preserved in `third_party/` for reference/source rebuilds.

### Recommended configurations

| Use case | Platform | Config | $/genome |
|---|---|---|---|
| **Cheapest (measured)** | **Oracle A1 (16 OCPU)** | **INT8 + jemalloc + 4-way CV** | **$0.80** |
| Cheapest (sequential) | Oracle A1 (16 OCPU) | INT8 + jemalloc | $1.04 |
| Cheapest A2 (projected) | Oracle A2 (16 OCPU, 32 vCPU) | INT8 + jemalloc + 4-way CV | ~$2.14 |
| Fastest ARM64 (projected) | Graviton4 (c8g.8xlarge) | BF16 + jemalloc + 4-way CV | ~$3.13 |

### Quick start

```bash
# Sequential (16 vCPU)
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=28g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5 \
  /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=$(nproc) \
  --call_variants_extra_args="--batch_size=256"

# Parallel CV (32+ vCPU, 1.9-2.5x CV speedup)
docker run -e DV_AUTOCONFIG=1 -e DV_USE_JEMALLOC=1 \
  -v /path/to/data:/data --memory=56g \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5 \
  /opt/deepvariant/scripts/run_parallel_cv.sh \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --num_shards=32 \
  --num_cv_workers=4
```

### Accuracy

All backends pass GIAB HG003 gates (SNP F1 >= 0.9974, INDEL F1 >= 0.9940)
including stratified validation across homopolymers, tandem repeats, and
segmental duplications.

### Changes since v1.9.0-arm64.4

- Oracle A1 parallel CV benchmarks ($0.80/genome, 3-run measured)
- `quantize_static_onnx.py` and `convert_model_onnx.py` shipped in Docker image
- `quantize_model` shell wrapper in `/opt/deepvariant/bin/`
- `--nocompress_intermediates` flag for make_examples
- call_variants sharding + validator for uncompressed .tfrecord output
- AmpereOne OneDNN safety block extended to all binaries
- OMP_NUM_THREADS cap per CV worker in run_parallel_cv.sh
- Direct TFRecord serialization in C++ (bypass tf::Example proto)
- Reusable pileup buffer + absl::StrCat optimizations
- ACL v23.08 SVE filter + OneDNN indirect GEMM patches
- 3 Docker build fixes (cryptography, pip ordering, httplib2)
- AmpereOne BF16 investigation concluded — permanently blocked

### Full changelog

See [CHANGELOG.md](../CHANGELOG.md) for the complete list of changes.
