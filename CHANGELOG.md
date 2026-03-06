# Changelog

All notable changes to the DeepVariant ARM64 fork are documented here.
Upstream compatibility: google/deepvariant v1.9.0

## [v1.9.0-arm64.2] — 2026-03-06

### Added
- `scripts/autoconfig.sh` — CPU-aware config advisor. Detects Graviton3/4,
  AmpereOne, Neoverse-N1/N2. Recommends backend, thread counts, jemalloc.
  Enforces AmpereOne OneDNN hard safety (prevents SIGILL).
- `DV_AUTOCONFIG=1` entrypoint integration — auto-applies recommended env vars
  without overriding user-provided values.
- `DV_USE_JEMALLOC=1` opt-in jemalloc allocator integration. Verified 14-17%
  make_examples speedup on Graviton3 and AmpereOne (N=2 and N=4 respectively).
- `scripts/benchmark_jemalloc_ablation.sh` — interleaved ablation benchmark
  with 1s RSS polling, startup overhead instrumentation, and JSON output.
- `scripts/request_aws_quota.sh` — AWS vCPU quota checker across 6 regions.
- `docs/oracle-a2-wheel-test.md` — AmpereOne SIGILL investigation procedure.
- INT8 static ONNX backend: 2.3x speedup over ONNX FP32, matches BF16 speed.
- Stratified GIAB validation: all backends pass homopolymers, tandem repeats,
  segmental duplications.

### Changed
- Cost tables corrected: Oracle A2 baseline is $2.49/genome (4-run verified),
  $2.32/genome with jemalloc enabled.
- All $/genome cells now include: $/hr rate, N runs, jemalloc state, formula.
- N<4 run rows flagged with asterisk in benchmark tables.

### Fixed
- `benchmark_jemalloc_ablation.sh` timing parser — fixed to parse `real XmYs`
  output from time command (was looking for non-existent log patterns).
- `--tf-onednn-opts` flag added to ablation script — Oracle A2 requires
  OneDNN disabled to prevent SIGILL on AmpereOne.
- Ablation runs now interleaved (off/on/off/on) to eliminate cache-warming
  ordering bias.

### Dead Ends (Documented)
- EfficientNet-B3: 3x slower than InceptionV3 on CPU (depthwise conv penalty).
- KMP_AFFINITY tuning: 30% regression.
- ONNX ACL ExecutionProvider: fragile, 16 supported ops, not worth maintaining.
- Dynamic INT8 on ARM64: ConvInteger op missing in ORT ARM64.
- fast_pipeline at 16 vCPU: 42% slower than sequential (CPU contention).
- fast_pipeline on Oracle A2 32 vCPU: PP broken on streaming CVO, <1% wall improvement.
- INT8 ONNX beyond 16 threads: CV rate does not improve (GEMM saturates).
- ONNX inter-op parallelism: no improvement (InceptionV3 is intra-op bound).

## [v1.9.0-arm64.1] — 2026-01-15

### Added
- Initial ARM64 Linux port (Bazel 5.3.0, TF 2.13.1).
- All C++ optimizations from macOS port: haplotype cap, ImageRow flat buffer,
  query cache.
- Docker images on ghcr.io, validated on chr20 GIAB HG003.
- TF+OneDNN BF16 on Graviton3: 38% CV speedup, zero accuracy loss.
- OMP env scoping in run_deepvariant.py: 2.6% ME improvement.
- ONNX Runtime integration (`--use_onnx` flag, model conversion).
- INT8 output renormalization fix for quantized ONNX models.
- Graviton4 INT8 ONNX benchmark: 366s, $3.33/genome.
- Oracle A2 INT8 ONNX benchmark: 542s, $2.32/genome (cheapest tested).
