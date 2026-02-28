# DeepVariant — macOS ARM64 (Apple Silicon) Native Build

[![release](https://img.shields.io/badge/base-v1.9.0-green?logo=github)](https://github.com/google/deepvariant/releases)
[![platform](https://img.shields.io/badge/platform-macOS%20ARM64-blue?logo=apple)](https://support.apple.com/en-us/116943)

This is a fork of [Google DeepVariant](https://github.com/google/deepvariant) v1.9.0 that builds and runs **natively on macOS with Apple Silicon** (M1, M2, M3, M4) — no Docker, no Rosetta, no remote server.

> **Why this matters:** DeepVariant cannot run on Apple Silicon Macs via Docker. The official Docker image [crashes immediately](https://github.com/google/deepvariant/issues/657) because TensorFlow requires AVX instructions that Rosetta 2 cannot translate inside Docker's Linux VM. There is no official macOS build. Before this fork, Mac users had no local option — they needed a remote Linux x86_64 server. This fork patches the Bazel build system, C++ source, and third-party dependencies to compile and run natively on macOS ARM64 with Apple Clang.

## What is DeepVariant?

DeepVariant is a deep learning-based variant caller that takes aligned reads (in BAM or CRAM format), produces pileup image tensors from them, classifies each tensor using a convolutional neural network, and finally reports the results in a standard VCF or gVCF file.

DeepVariant supports germline variant-calling in diploid organisms. For full documentation on DeepVariant's capabilities, case studies, and supported data types, see the [upstream repository](https://github.com/google/deepvariant).

---

## Quick Install (Pre-built Binaries)

Install pre-built binaries with a single command. No build tools required.

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **conda**, **mamba**, or **micromamba** — native ARM64 (e.g. [Miniforge](https://github.com/conda-forge/miniforge))

### Install

```bash
curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | USE_CONDA=1 bash
```

This creates a `deepvariant` conda environment with Python 3.10, GNU parallel, and all dependencies. Activate it with `conda activate deepvariant`.

Alternatively, create the environment manually from the included `environment.yml`:

```bash
git clone https://github.com/antomicblitz/deepvariant-macos-arm64-metal.git
cd deepvariant-macos-arm64-metal
conda env create -f environment.yml
conda activate deepvariant
# Then install --no-deps packages:
pip install --no-deps tensorflow-hub==0.14.0 tensorflow-model-optimization==0.7.5 tf-models-official==2.13.1
```

### venv Install (Alternative)

If you already have **Python 3.10** installed (e.g., `brew install python@3.10`), the installer can use a lightweight venv instead of conda:

```bash
curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | bash
```

> **Note:** Python 3.10 specifically is required — `tensorflow-macos 2.13.1` does not support other Python versions. You also need GNU parallel installed separately (`brew install parallel`).

### Environment Variables

Customize the install with environment variables:

```bash
# Install to a custom location
curl -fsSL ... | DEEPVARIANT_HOME=/path/to/dir bash

# Download specific models (WGS WES PACBIO ONT_R104 HYBRID MASSEQ ALL NONE)
curl -fsSL ... | MODEL_TYPES="WGS WES PACBIO" bash

# Force conda or venv
curl -fsSL ... | USE_CONDA=1 bash
curl -fsSL ... | USE_CONDA=0 bash   # force venv, fail if no Python 3.10

# Custom conda env name (default: deepvariant)
curl -fsSL ... | CONDA_ENV_NAME=dv19 USE_CONDA=1 bash

# Skip environment creation entirely (if you manage your own)
curl -fsSL ... | SKIP_ENV=1 bash
```

### After Installation

Open a new terminal and run:

```bash
run_deepvariant \
  --model_type=WGS \
  --ref=reference.fasta \
  --reads=input.bam \
  --output_vcf=output.vcf \
  --num_shards=$(sysctl -n hw.ncpu)
```

Download additional models any time:

```bash
deepvariant-download-model WES PACBIO ONT_R104
deepvariant-download-model WGS --deeptrio
```

### Quicktest

Verify your installation with a small end-to-end test (requires GNU parallel):

```bash
$DEEPVARIANT_HOME/scripts/quicktest.sh
```

This runs all three DeepVariant steps on a 10kb region of chr20, confirms Metal GPU detection, and produces a VCF output.

### Uninstalling

Run the uninstall script:

```bash
deepvariant-uninstall
```

This removes the install directory, conda/venv environment, shell profile entries, and quicktest data. It shows what will be removed and asks for confirmation before proceeding.

---

## Build from Source

If you prefer to build from source instead of using pre-built binaries:

### Prerequisites

- **macOS** on Apple Silicon (M1/M2/M3/M4)
- **Homebrew** — [https://brew.sh](https://brew.sh)
- **Python 3.10** — `brew install python@3.10`
- **Xcode Command Line Tools** — `xcode-select --install`
- ~30 GB disk space (TensorFlow source + Bazel cache)

### 1. Clone the Repository

```bash
git clone https://github.com/antomicblitz/deepvariant-macos-arm64-metal.git
cd deepvariant-macos-arm64-metal
git checkout r1.9
```

### 2. Install Build Prerequisites

```bash
./build-prereq-macos.sh
```

This installs Homebrew packages, Bazel 5.3.0, abseil-cpp, CLIF C++ runtime, clones/configures TensorFlow 2.13.1 source, and runs `run-prereq-macos.sh` for Python packages.

### 3. Patch zlib in Bazel Cache (Required After `bazel clean`)

After the first Bazel build starts downloading external deps, you must patch zlib's `zutil.h`. This is needed because modern macOS defines `TARGET_OS_MAC`, which causes zlib to set `fdopen` to `NULL`:

```bash
ZUTIL=$(find $(bazel info output_base)/external/zlib -name zutil.h 2>/dev/null)
gsed -i 's/#if defined(MACOS) || defined(TARGET_OS_MAC)/#if defined(MACOS) \&\& !defined(__APPLE__)/' "$ZUTIL"
```

> **Important:** This patch lives in the Bazel cache and must be reapplied after `bazel clean` or any cache wipe.

### 4. Build

```bash
source settings.sh
./build_release_binaries.sh
```

### 5. Set Up a Python Virtual Environment

```bash
python3.10 -m venv ~/dv-venv
source ~/dv-venv/bin/activate
pip install tensorflow-macos==2.13.1 tensorflow-metal==1.0.0
pip install --no-deps tensorflow-hub==0.14.0 tensorflow-model-optimization==0.7.5 tf-models-official==2.13.1
pip install absl-py protobuf==4.21.9 pysam==0.20.0 contextlib2 etils typing_extensions \
  importlib_resources sortedcontainers==2.1.0 intervaltree==3.1.0 ml_collections \
  clu==0.0.9 joblib psutil pandas==1.3.4 Pillow==9.5.0 scikit-learn==1.0.2 \
  jax==0.4.35 opencv-python-headless markupsafe==2.0.1
```

### 6. Package for Distribution (Optional)

```bash
./scripts/package_release.sh           # create tarball only
./scripts/package_release.sh --release # create tarball + GitHub Release
```

---

## Benchmarks

We benchmarked DeepVariant v1.9.0 on an **Apple M1 Max** (8 performance cores, 32-core GPU, 32 GB RAM) using the standard HG003 chr20 WGS case study and compared against published GCP metrics.

### Performance: M1 Max vs GCP Instances

| Stage | M1 Max | GCP 16-vCPU (est.) | GCP 96-vCPU | M1 Max vs 16-vCPU |
|-------|--------|---------------------|-------------|---------------------|
| `make_examples` | 4m32s | 4m49s | 57s | **1.06x faster** |
| `call_variants` | 3m15s | 58s | 21s | 0.30x |
| `postprocess_variants` | 16s | 10s | 9s | 0.62x |
| **Total** | **8m03s** | **5m57s** | **1m39s** | **0.74x** |

*GCP 16-vCPU times are estimated from [published scaling data](https://pmc.ncbi.nlm.nih.gov/articles/PMC7481958/) (16/32/64/96 CPU counts), adjusted for v1.9 improvements. GCP 96-vCPU times are from [docs/metrics.md](docs/metrics.md), scaled from full genome to chr20 (64M / 3.1G bases). n2-standard-16 has 8 physical Intel Cascade Lake cores with hyperthreading (16 vCPUs), matching the M1 Max's 8 physical performance cores.*

### Key Findings

- **`make_examples` (CPU-bound, embarrassingly parallel):** M1 Max matches or slightly beats an equivalent-core GCP instance. Apple Silicon's high per-core performance compensates for the lower core count.

- **`call_variants` (TensorFlow inference):** Metal GPU provides **no measurable speedup** in v1.9. The "small model" optimization in v1.9 pre-screens easy variants on CPU and only sends hard sites to the full CNN, dramatically reducing the GPU-acceleratable workload. This is a fundamental change from earlier versions where GPU provided significant benefit. The M1 Max is ~3x slower than the estimated 16-vCPU GCP for this stage, likely because the estimated GCP number inherits favorable scaling assumptions from older DeepVariant versions.

- **`postprocess_variants`:** Mostly single-threaded; comparable across platforms.

- **Overall:** The M1 Max processes HG003 chr20 in ~8 minutes. It is competitive on a per-core basis for CPU-bound stages but cannot match cloud instances with many more cores. The 96-core GCP instance is ~5x faster overall, as expected given the 12:1 core ratio.

### Why Run DeepVariant on Apple Silicon?

The alternative is **not running it at all**. The official DeepVariant Docker image [crashes on Apple Silicon](https://github.com/google/deepvariant/issues/657) because TensorFlow's binaries require AVX instructions, which Rosetta 2 cannot translate inside Docker's Linux VM. QEMU-based emulation technically works but is 10-20x slower and impractical. There is no official macOS build. Before this fork, Mac users needed a remote Linux server.

With this native build, Apple Silicon Macs become viable for:

1. **Local development and testing.** Run the full pipeline on your laptop without Docker, cloud instances, or network access. Valuable for pipeline development, parameter tuning, and education.

2. **Small datasets and targeted regions.** Exome, panel, or single-chromosome analyses complete in minutes — fast enough for interactive workflows.

3. **Privacy and data sovereignty.** Clinical or restricted datasets that cannot leave your facility can be processed locally.

4. **Cost.** No cloud compute charges. The Mac you already own can run DeepVariant.

5. **Reproducibility.** A self-contained local environment with no Docker or cloud dependencies.

**Not recommended for:**

- Full whole-genome sequencing at scale (30x WGS). A 96-core cloud instance at ~79 minutes is more practical than the estimated 6-12 hours on a Mac.
- High-throughput batched processing. Use cloud instances or HPC clusters.

### Metal GPU Status

TensorFlow Metal GPU is installed and available on all Apple Silicon Macs, but benchmarks show it does not meaningfully accelerate `call_variants` in DeepVariant v1.9. We include `tensorflow-metal` because:
- It is required by `tensorflow-macos` for full platform support
- Future DeepVariant versions or different model types may benefit from GPU
- It does not hurt performance (no measurable overhead vs CPU-only)

### Running the Benchmark Yourself

```bash
# Full benchmark (downloads ~5 GB of data on first run)
bash scripts/benchmark.sh

# Quick run, skip accuracy evaluation
bash scripts/benchmark.sh --skip-happy --runs 1

# Visualize results
python3 scripts/benchmark_viz.py ~/deepvariant-benchmark/benchmark_results.json --show
```

---

## TensorFlow Package Setup

### Correct Package Combination

| Package | Version | Notes |
|---------|---------|-------|
| `tensorflow-macos` | 2.13.1 | Apple's TF build for macOS ARM64 |
| `tensorflow-metal` | 1.0.0 | Metal GPU plugin |

**Do NOT install the standard `tensorflow` pip package alongside `tensorflow-metal`.** Both register the Metal platform, causing a fatal "platform already registered" crash. Use `tensorflow-macos` instead.

### Dependencies that Pull in `tensorflow`

Some packages (e.g., `tensorflow-hub`, `tensorflow-model-optimization`, `tf-models-official`) list `tensorflow` as a dependency and will install the regular package, breaking your setup. Install these with `--no-deps`:

```bash
pip install --no-deps tensorflow-hub==0.14.0
pip install --no-deps tensorflow-model-optimization==0.7.5
pip install --no-deps tf-models-official==2.13.1
```

---

## Architecture

DeepVariant has two distinct layers that matter for this port:

1. **C++ extensions** (`.so` modules) — pileup image generation, allele counting, BAM/VCF I/O, realignment. These are compiled via Bazel against TensorFlow C++ headers at build time.

2. **Python runtime** — ML inference/training via TensorFlow Python API. `tensorflow-metal` registers the Metal GPU device, though benchmarks show minimal benefit for the v1.9 inference workload.

The C++ layer only needs TF headers at build time. The Python layer uses pip-installed TensorFlow at runtime.

---

## What Was Changed

This fork modifies the following files from upstream DeepVariant v1.9.0. For the full list, see [the build fixes log](docs/macos-arm64-build-fixes.md).

### New Files

| File | Purpose |
|------|---------|
| `build-prereq-macos.sh` | Homebrew deps, Bazel 5.3.0, abseil-cpp, CLIF runtime, TF source |
| `run-prereq-macos.sh` | Python packages, `tensorflow-macos` + `tensorflow-metal` |
| `third_party/boost.BUILD` | Boost headers from Homebrew (`/opt/homebrew/include`) |

### Modified Files

| File | Change |
|------|--------|
| `.bazelrc` | `--config=macos_arm64`, `BOOST_PROCESS_VERSION=1`, `-Wno-unknown-warning-option` |
| `BUILD` | `py_runtime` interpreter path |
| `WORKSPACE` | `new_local_repository` entries for Boost and CLIF at `/opt/homebrew` |
| `settings.sh` | Darwin/ARM64 detection, Homebrew Python paths, macOS-compatible copt flags |
| `build_release_binaries.sh` | BSD `ln`/`sed` compat, `clang++` linking, zsh shebang fix, `PYTHON_BINARY` patch |
| `third_party/htslib.BUILD` | macOS `config.h` (no `fdatasync`, no SSE, `.dylib` plugin extension) |
| `third_party/libssw.BUILD` | `sse2neon.h` for ARM64 SSW (Smith-Waterman) |
| `third_party/sdsl_lite.BUILD` | Removed `"."` from includes (version file conflict) |
| `third_party/clif.BUILD` | Added abseil deps for CLIF C++ runtime |
| `third_party/gbwt.BUILD` | Added `@boost//:boost` dependency |
| `third_party/gbwtgraph.BUILD` | Added `@boost//:boost` dependency |
| `deepvariant/BUILD` | Added `@boost//:boost` to `fast_pipeline`, `stream_examples`, etc. |
| `deepvariant/realigner/BUILD` | Added `@boost//:boost` to `debruijn_graph` |
| `third_party/nucleus/io/BUILD` | Added `@boost//:boost` to `gbz_reader` |
| `deepvariant/fast_pipeline.h` | Boost 1.90 v1 process API includes |
| `deepvariant/fast_pipeline.cc` | Boost 1.90 v1 process API includes |
| `deepvariant/allelecounter.cc` | `int64_t`/`long` type mismatch fix |
| `deepvariant/alt_aligned_pileup_lib.cc` | `int64_t`/`long` type mismatch fixes |
| `deepvariant/make_examples_native.cc` | `int64_t`/`long` type mismatch fix |

### External Patches (Outside This Repo)

| Target | Change |
|--------|--------|
| `tensorflow/tensorflow.bzl` | Replaced GNU `ln -r -s` with Python `os.path.relpath` for macOS |
| `external/zlib/zutil.h` (Bazel cache) | Prevent `fdopen=NULL` on macOS (`TARGET_OS_MAC` guard) |

---

## Known Issues

1. **zlib `zutil.h` patch is not persistent.** It lives in the Bazel cache and must be reapplied after `bazel clean`. See step 3 above.

2. **zsh escapes `!` in strings.** The build script uses `bytes([0x23, 0x21])` to write `#!` shebangs. If you write custom scripts that generate shebangs, be aware of this.

3. **`int64_t` is `long long` on macOS ARM64**, while it is `long` on Linux x86_64. Both are 64-bit, but they are different types to the compiler, causing `std::max(int64_t, 0L)` template deduction failures. All instances in DeepVariant have been fixed.

4. **Boost 1.90+** split the `process` API into v1 and v2 namespaces. The build uses `-DBOOST_PROCESS_VERSION=1` to select v1.

5. **`tf-models-official`** depends on `tensorflow-text`, which has no ARM64 wheels for 2.13.x. It is installed with `--no-deps`. DeepVariant only uses `official.modeling.optimization`, which does not require `tensorflow-text`.

6. **Fast pipeline** (`fast_pipeline`) runs `make_examples` and `call_variants` simultaneously. `tensorflow-metal` registers the Metal GPU for `call_variants` inference, though benchmarks show minimal GPU speedup in v1.9 due to the "small model" optimization.

---

## Running DeepVariant

Once built, DeepVariant is run the same way as on Linux, but using the zip binaries directly instead of Docker:

```bash
# Activate venv with tensorflow-macos
source ~/dv-venv/bin/activate

INPUT_DIR="path/to/input"
OUTPUT_DIR="path/to/output"

# Step 1: make_examples
python3 bazel-bin/deepvariant/make_examples.zip \
  --mode calling \
  --ref "${INPUT_DIR}/reference.fasta" \
  --reads "${INPUT_DIR}/reads.bam" \
  --output "${OUTPUT_DIR}/examples.tfrecord.gz" \
  --examples "${OUTPUT_DIR}/examples.tfrecord.gz"

# Step 2: call_variants (Metal GPU available but minimal speedup in v1.9)
python3 bazel-bin/deepvariant/call_variants.zip \
  --outfile "${OUTPUT_DIR}/call_variants_output.tfrecord.gz" \
  --examples "${OUTPUT_DIR}/examples.tfrecord.gz" \
  --checkpoint "path/to/model.ckpt"

# Step 3: postprocess_variants
python3 bazel-bin/deepvariant/postprocess_variants.zip \
  --ref "${INPUT_DIR}/reference.fasta" \
  --infile "${OUTPUT_DIR}/call_variants_output.tfrecord.gz" \
  --outfile "${OUTPUT_DIR}/output.vcf.gz"
```

For the full command reference and model types (WGS, WES, PACBIO, ONT, etc.), see the [upstream documentation](https://github.com/google/deepvariant/blob/r1.9/docs/deepvariant-details.md).

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | macOS 11+ (Big Sur or later) |
| Architecture | Apple Silicon (M1, M2, M3, M4) |
| Python | 3.10 |
| Bazel | 5.3.0 (installed by `build-prereq-macos.sh`) |
| TensorFlow | 2.13.1 source (build time), `tensorflow-macos` 2.13.1 (runtime) |
| GPU | Metal via `tensorflow-metal` 1.0.0 (included, minimal speedup in v1.9) |
| Disk | ~30 GB (TF source + Bazel cache) |
| RAM | 16 GB minimum, 32 GB+ recommended |

---

## How to Cite

If you use DeepVariant in your work, please cite:

[A universal SNP and small-indel variant caller using deep neural networks. *Nature Biotechnology* 36, 983-987 (2018).](https://rdcu.be/7Dhl)
Ryan Poplin, Pi-Chuan Chang, David Alexander, Scott Schwartz, Thomas Colthurst, Alexander Ku, Dan Newburger, Jojo Dijamco, Nam Nguyen, Pegah T. Afshar, Sam S. Gross, Lizzie Dorfman, Cory Y. McLean, and Mark A. DePristo.
doi: https://doi.org/10.1038/nbt.4235

## License

[BSD-3-Clause license](LICENSE)

## Disclaimer

This is not an official Google product.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.
