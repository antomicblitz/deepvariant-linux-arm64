# Changelog

All notable changes to the DeepVariant ARM64 fork are documented here.
Upstream compatibility: google/deepvariant v1.9.0

## [Unreleased]

## [v1.9.0-arm64.7] — 2026-03-09

### Fixed
- `--ignore-installed` → `--reinstall-package` in Dockerfiles and `run-prereq.sh` for uv 0.10
- `get-pip.py` bootstrap → `uv pip install` in `build_absl.sh` (PEP 668 on Ubuntu 24.04)
- `uv pip install --python` targeting venv explicitly in `build_absl.sh` sudo context
- `PYTHON_LIB_PATH` override in `settings_arm64.sh` for uv venv `site-packages`

## [v1.9.0-arm64.6] — 2026-03-08

### Added
- `.bumpversion.toml`: automated fork version bumping (`bump-my-version`)
- `.github/workflows/bump-version.yml`: workflow_dispatch version bump
- `Dockerfile.arm64.builder`: standalone builder-stage Dockerfile for native
  ARM64 Bazel compilation (~1hr native vs ~4hr+ QEMU)
- `Dockerfile.arm64`: 3-stage build (builder → models → runtime) with
  `BUILDER_IMAGE` build-arg defaulting to pre-built ghcr.io base
- CI `build-builder` job: manual workflow_dispatch to rebuild base builder image
- `docker_entrypoint.sh`: THP via `GLIBC_TUNABLES=glibc.malloc.hugetlb=2`
  (glibc >= 2.35, ~6% SPEC improvement on AArch64, reduces TLB pressure)
- `docker_entrypoint.sh`: jemalloc `MALLOC_CONF` with background_thread and
  metadata_thp — matches run_parallel_cv.sh for consistency
- `autoconfig.sh`: JSON output now includes `DV_USE_JEMALLOC`, `MALLOC_CONF`,
  and `GLIBC_TUNABLES` so `DV_AUTOCONFIG=1` auto-enables jemalloc
- `call_variants.py`: ORT `graph_optimization_level = ORT_ENABLE_ALL` for
  extended operator fusion (Conv+BN, GEMM+Activation, constant folding)

### Changed
- Moved `TRAINING_EXPERIMENT.md` and `TRAINING_RESUME.md` from root to `docs/`
- Updated all cross-references to moved files (README, CLAUDE.md, architecture, paper)
- Replaced pip with uv for all Python package management in ARM64 Docker builds
- Replaced deadsnakes PPA with `uv venv --python 3.10` in all Docker stages
- Pinned uv to `0.10` (was `latest`)
- Split lint job from `arm64-build.yml` into dedicated `ci.yml` workflow
- `Dockerfile.arm64`: 36 ADD layers → single wget loop in models stage (~50→15 layers)
- `Dockerfile.arm64`: merged 2 apt-get + 3 pip install calls into single layers
- `Dockerfile.arm64`: ONNX conversion moved to models stage (cached independently)
- CI `build-arm64` timeout reduced from 240min to 30min (pre-built builder base)

### Removed
- deadsnakes PPA dependency (Python 3.10 now managed by uv)
- `python3-testresources` apt package (pip-era workaround)
- `get-pip.py` bootstrap in `run-prereq.sh`
- `Dockerfile.arm64.runtime`: superseded by `Dockerfile.arm64` with BUILDER_IMAGE arg
- `Dockerfile.arm64`: removed inline builder-local fallback (use .builder file instead)

### Fixed
- `call_variants.py`: ORT version parse crash on `.post1` suffixes
  (e.g. `1.17.0.post1` caused `int()` ValueError)
- `call_variants.py`: `np.ascontiguousarray` wrapping ONNX batch input to
  prevent silent per-batch copy on non-contiguous tensors
- `docker_entrypoint.sh`: ONNX auto-selection now warns that ONT/MasSeq/Hybrid
  models are TF-only (Dockerfile only converts WGS/WES/PacBio to ONNX)
- `settings_arm64.sh`: removed misleading `TF_ENABLE_ONEDNN_OPTS=1` from
  build-time settings (runtime env var, no effect during Bazel build)

### Added
- CLAUDE.md: slim entry point (`@AGENTS.md`), original content preserved in
  `docs/architecture-arm64.md`
- AGENTS.md: agent behavioral rules, decision framework, dead ends, quality
  thresholds
- CONTRIBUTING.md: dev workflow, command reference, coding standards
- AGENT_LEARNINGS.md: seeded with dead-end patterns from benchmarking history
- `pyproject.toml`: PEP 440 version, uv dependency-groups, ruff + pyright +
  pytest + complexipy config
- `Makefile`: setup_dev, lint, type_check, test, validate, docker_build,
  benchmark (all via uv)
- `.claude/settings.json`: marketplace plugins
- `.devcontainer/devcontainer.json`: Python 3.10, docker-in-docker, uv + dev deps
- `.github/workflows/codeql.yaml`: CodeQL Python code scanning on push/PR/schedule
- `.github/dependabot.yaml`: weekly pip dependency updates
- `.github/PULL_REQUEST_TEMPLATE.md`: structured PR template
- CI lint-and-typecheck job with `astral-sh/setup-uv` + cache in `arm64-build.yml`

### Changed
- CI `build-arm64` timeout reduced from 240min to 30min (pre-built builder base)
- `Dockerfile.arm64` refactored: two-stage build now references external builder
  image instead of inline Bazel compilation

### Changed
- `setup.py` extracted to `scripts/build_proto.py` (standalone proto generation)
- Ruff auto-fixes applied to fork scripts (f-strings, import sorting)
- Pyright scoped to `scripts/build_proto.py` (only stdlib-only fork script);
  upstream `deepvariant/` ignored

### Removed
- `__init__.py` (unused with Bazel build)
- `.github/PULL_REQUEST_TEMPLATE` (replaced by `.md` version)

## [v1.9.0-arm64.4] — 2026-03-06

### Added
- `scripts/run_parallel_cv.sh` — user-facing parallel call_variants wrapper.
  Runs inside the Docker container as a drop-in replacement for run_deepvariant.
  Splits call_variants into N parallel workers with scoped OMP_NUM_THREADS.
  Requires ONNX backend (TF SavedModel would OOM with multiple workers).

### Fixed
- `autoconfig.sh` now included in Docker image. `DV_AUTOCONFIG=1` was silently
  non-functional in arm64.2 and arm64.3 because the script was never COPY'd
  into the image.
- `run_parallel_cv.sh` included in Docker image at
  `/opt/deepvariant/scripts/run_parallel_cv.sh`.

## [v1.9.0-arm64.3] — 2026-03-06

### Added
- `scripts/benchmark_parallel_cv.sh` — parallel call_variants benchmark.
  Splits 32 ME shards across N workers (2-way, 4-way), merges via
  postprocess_variants `@N` sharded pattern. Zero DeepVariant code changes.
- 32-vCPU benchmarks on Graviton3 (c7g.8xlarge), Graviton4 (c8g.8xlarge),
  and Oracle A2 (16 OCPU). CV floor confirmed at 16 threads on all platforms.
- 4-way parallel CV results: Graviton4 61s (2.10x, N=3), Graviton3 74s
  (1.90x, N=4), Oracle A2 114s (2.47x, N=2). Variant counts match exactly.
- Projected $/genome with parallel CV: Oracle A2 ~$2.14, Graviton4 ~$3.13,
  Graviton3 ~$3.35.

### Changed
- Corrected $/genome calculations: Graviton4 ONNX FP32 $5.07→$5.47,
  Graviton4 BF16 standalone ~$4.31→~$4.65.
- INT8 Graviton3 rate updated to 3-run average: 0.238→0.237 s/100.
- Removed stale status labels and superseded data across all docs.

### Known Issues
- Parallel CV is benchmark-only (`benchmark_parallel_cv.sh`), not user-facing.
  User wrapper shipped in v1.9.0-arm64.4.
- `DV_AUTOCONFIG=1` non-functional (inherited from arm64.2). Fixed in arm64.4.

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

### Known Issues
- `DV_AUTOCONFIG=1` non-functional — `autoconfig.sh` not included in Docker
  image. Entrypoint silently skips it. Fixed in v1.9.0-arm64.4.

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
