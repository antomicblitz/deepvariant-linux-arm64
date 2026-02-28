# Homebrew Tap Implementation Guide

This document contains everything needed to create and publish a Homebrew tap for DeepVariant macOS ARM64. Create a new GitHub repo `antomicblitz/homebrew-deepvariant`, copy the files below into it, and follow the setup instructions.

---

## Repository Structure

```
homebrew-deepvariant/
├── Formula/
│   └── deepvariant.rb
└── README.md
```

---

## File 1: `Formula/deepvariant.rb`

```ruby
class Deepvariant < Formula
  desc "Deep learning variant caller for genomics — macOS ARM64 with Metal GPU"
  homepage "https://github.com/antomicblitz/deepvariant-macos-arm64-metal"
  url "https://github.com/antomicblitz/deepvariant-macos-arm64-metal/releases/download/v1.9.0/deepvariant-1.9.0-macos-arm64.tar.gz"
  sha256 "e75ab9bd7f324ce69210d89cb038e77f1537d599619c596c84220c424c08e2a3"
  license "BSD-3-Clause"
  version "1.9.0"

  # Pre-built ARM64 binaries — nothing for Homebrew to compile.
  bottle :unneeded

  depends_on :macos
  depends_on "python@3.10"
  depends_on "parallel"

  def install
    # ARM64 only — binaries are Mach-O arm64, not universal.
    if Hardware::CPU.intel?
      raise "DeepVariant macOS ARM64 requires Apple Silicon (M1/M2/M3/M4). " \
            "This formula does not support Intel Macs."
    end

    python = Formula["python@3.10"].opt_bin/"python3.10"

    # ── Install pre-built binaries and scripts into libexec ────────────────
    libexec.install "bin"
    libexec.install "scripts"
    libexec.install "licenses.zip" if (buildpath/"licenses.zip").exist?

    # ── Create Python virtualenv ──────────────────────────────────────────
    system python, "-m", "venv", libexec/"venv"
    pip = libexec/"venv/bin/pip"

    system pip, "install", "--upgrade", "pip"

    # Pin NumPy to 1.x FIRST — tensorflow-macos 2.13.1 crashes with NumPy 2.x
    system pip, "install", "numpy>=1.22,<=1.24.3"

    # TensorFlow with Metal GPU acceleration
    system pip, "install", "tensorflow-macos==2.13.1"
    system pip, "install", "tensorflow-metal==1.0.0"

    # Packages that pull in standard 'tensorflow' — install without deps
    system pip, "install", "--no-deps",
           "tensorflow-hub==0.14.0",
           "tensorflow-model-optimization==0.7.5",
           "tf-models-official==2.13.1"

    # tf-models-official runtime deps that DeepVariant actually needs
    system pip, "install",
           "tensorflow-datasets", "gin-config", "seqeval",
           "opencv-python-headless>=4.5", "tf-slim", "sacrebleu", "pyyaml"

    # Core DeepVariant Python dependencies
    system pip, "install",
           "absl-py", "parameterized", "contextlib2",
           "etils", "typing_extensions", "importlib_resources",
           "sortedcontainers==2.1.0", "intervaltree==3.1.0",
           "mock>=2.0.0", "ml_collections",
           "clu==0.0.9", "protobuf==4.21.9",
           "requests>=2.18", "joblib", "psutil", "ipython",
           "pandas==1.3.4", "Pillow==9.5.0", "pysam==0.20.0",
           "scikit-learn==1.0.2", "jax==0.4.35", "markupsafe==2.1.1"

    # Re-pin NumPy (jax / tensorflow-datasets can widen it to 2.x)
    system pip, "install", "--force-reinstall", "numpy>=1.22,<=1.24.3"

    # ── Create wrapper scripts in bin/ ────────────────────────────────────
    # Each wrapper sets DEEPVARIANT_HOME and activates the venv.

    # run_deepvariant — the main pipeline runner
    (bin/"run_deepvariant").write <<~BASH
      #!/bin/bash
      export DEEPVARIANT_HOME="#{libexec}"
      export PATH="#{libexec}/venv/bin:${PATH}"
      exec python3 "#{libexec}/scripts/run_deepvariant.py" "$@"
    BASH
    (bin/"run_deepvariant").chmod 0755

    # run_deeptrio
    (bin/"run_deeptrio").write <<~BASH
      #!/bin/bash
      export DEEPVARIANT_HOME="#{libexec}"
      export PATH="#{libexec}/venv/bin:${PATH}"
      exec python3 "#{libexec}/scripts/run_deeptrio.py" "$@"
    BASH
    (bin/"run_deeptrio").chmod 0755

    # deepvariant-download-model
    (bin/"deepvariant-download-model").write <<~BASH
      #!/bin/bash
      export DEEPVARIANT_HOME="#{libexec}"
      exec "#{libexec}/scripts/deepvariant-download-model" "$@"
    BASH
    (bin/"deepvariant-download-model").chmod 0755

    # deepvariant-quicktest
    (bin/"deepvariant-quicktest").write <<~BASH
      #!/bin/bash
      export DEEPVARIANT_HOME="#{libexec}"
      export PATH="#{libexec}/venv/bin:${PATH}"
      exec "#{libexec}/scripts/quicktest.sh" "$@"
    BASH
    (bin/"deepvariant-quicktest").chmod 0755

    # Individual tool wrappers (most commonly used directly)
    %w[make_examples call_variants postprocess_variants].each do |tool|
      (bin/tool).write <<~BASH
        #!/bin/bash
        export DEEPVARIANT_HOME="#{libexec}"
        export PATH="#{libexec}/venv/bin:${PATH}"
        exec python3 "#{libexec}/bin/#{tool}.zip" "$@"
      BASH
      (bin/tool).chmod 0755
    end
  end

  def caveats
    <<~EOS
      Models are NOT downloaded automatically (they are 500MB-2GB each).
      Download the WGS model to get started:

        deepvariant-download-model WGS

      Available model types: WGS, WES, PACBIO, ONT_R104, HYBRID, MASSEQ

      Run a quick end-to-end test:

        deepvariant-quicktest

      Usage:

        run_deepvariant \\
          --model_type=WGS \\
          --ref=reference.fasta \\
          --reads=input.bam \\
          --output_vcf=output.vcf \\
          --num_shards=$(sysctl -n hw.ncpu)
    EOS
  end

  test do
    # Verify TensorFlow imports correctly
    tf_version = shell_output(
      "#{libexec}/venv/bin/python3 -c 'import tensorflow; print(tensorflow.__version__)' 2>&1"
    ).strip
    assert_match "2.13", tf_version

    # Verify make_examples binary loads
    # absl-py exits 1 on --help (mark_flag_as_required), so allow exit code 1
    me_help = shell_output(
      "#{libexec}/venv/bin/python3 #{libexec}/bin/make_examples.zip --help 2>&1", 1
    )
    assert_match "creates tf.Example protos", me_help

    # Verify wrapper scripts exist and are executable
    assert_predicate bin/"run_deepvariant", :exist?
    assert_predicate bin/"run_deepvariant", :executable?
    assert_predicate bin/"run_deeptrio", :exist?
    assert_predicate bin/"deepvariant-download-model", :exist?
    assert_predicate bin/"make_examples", :exist?
    assert_predicate bin/"call_variants", :exist?
    assert_predicate bin/"postprocess_variants", :exist?
  end
end
```

---

## File 2: `README.md`

```markdown
# homebrew-deepvariant

Homebrew tap for [DeepVariant](https://github.com/antomicblitz/deepvariant-macos-arm64-metal) — a deep learning variant caller built natively for macOS ARM64 (Apple Silicon) with Metal GPU acceleration.

## Install

```bash
brew tap antomicblitz/deepvariant
brew install deepvariant
```

> **Note:** Installation takes several minutes — it creates a Python 3.10 virtualenv and installs TensorFlow + dependencies.

## Post-Install: Download Models

Models are not bundled (they are 500MB-2GB each). Download the model for your data type:

```bash
deepvariant-download-model WGS          # Whole genome sequencing
deepvariant-download-model WES          # Whole exome sequencing
deepvariant-download-model PACBIO       # PacBio HiFi
deepvariant-download-model ONT_R104     # Oxford Nanopore
deepvariant-download-model ALL          # All models
```

## Verify Installation

```bash
deepvariant-quicktest
```

This runs all three DeepVariant steps on a 10kb region of chr20, confirms Metal GPU detection, and produces a VCF output.

## Usage

```bash
run_deepvariant \
  --model_type=WGS \
  --ref=reference.fasta \
  --reads=input.bam \
  --output_vcf=output.vcf \
  --num_shards=$(sysctl -n hw.ncpu)
```

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Homebrew

## Uninstall

```bash
brew uninstall deepvariant
brew untap antomicblitz/deepvariant
```

Models are stored in `$(brew --prefix)/lib/deepvariant/models/`. Remove them manually if desired.
```

---

## Setup Instructions

### Step 1: Create the GitHub Repository

1. Go to https://github.com/new
2. Repository name: `homebrew-deepvariant`
3. Owner: `antomicblitz`
4. Public repository
5. No README, no .gitignore, no license (we'll push our own)

### Step 2: Clone and Populate

```bash
git clone https://github.com/antomicblitz/homebrew-deepvariant.git
cd homebrew-deepvariant

# Create the formula directory
mkdir -p Formula

# Copy the Formula/deepvariant.rb content from above into:
#   Formula/deepvariant.rb

# Copy the README.md content from above into:
#   README.md

# Commit and push
git add -A
git commit -m "Add DeepVariant v1.9.0 formula for macOS ARM64"
git push origin main
```

### Step 3: Test Locally

```bash
# Tap your new repo
brew tap antomicblitz/deepvariant

# Install (this takes several minutes for pip installs)
brew install deepvariant

# Verify
brew test deepvariant

# Download WGS model and run quicktest
deepvariant-download-model WGS
deepvariant-quicktest
```

### Step 4: Verify the Installation

After `brew install deepvariant`, confirm:

```bash
# Check wrapper scripts are on PATH
which run_deepvariant
which deepvariant-download-model
which make_examples

# Check TensorFlow + Metal GPU
$(brew --prefix deepvariant)/libexec/venv/bin/python3 -c "
import tensorflow as tf
print('TF version:', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
print('Metal GPUs:', len(gpus))
"

# Check make_examples loads
make_examples --help 2>&1 | head -5
```

---

## Updating the Formula for New Releases

When you release a new version of DeepVariant:

1. Build and package the new release in the main repo:
   ```bash
   cd deepvariant-macos-arm64-metal
   ./scripts/package_release.sh --release
   ```

2. Get the new SHA256:
   ```bash
   shasum -a 256 /tmp/deepvariant-*.tar.gz
   ```

3. Update the formula in the tap repo:
   ```bash
   cd homebrew-deepvariant
   # Edit Formula/deepvariant.rb:
   #   - Update version "X.Y.Z"
   #   - Update url with new version
   #   - Update sha256 with new hash
   git commit -am "Update DeepVariant to vX.Y.Z"
   git push
   ```

4. Users update via:
   ```bash
   brew update
   brew upgrade deepvariant
   ```

---

## How It Works

The formula:
1. Downloads the pre-built binary tarball (~192 MB) from GitHub Releases
2. Extracts binaries to `$(brew --prefix)/opt/deepvariant/libexec/bin/`
3. Creates a Python 3.10 venv at `libexec/venv/`
4. Pip-installs `tensorflow-macos==2.13.1` + `tensorflow-metal==1.0.0` + all dependencies
5. Creates shell wrapper scripts in `$(brew --prefix)/bin/` that set `DEEPVARIANT_HOME` and activate the venv
6. Models are downloaded post-install by the user via `deepvariant-download-model`

### Why Not `Language::Python::Virtualenv`?

Homebrew's standard Python formula pattern requires every pip dependency declared as a `resource` block with a PyPI URL and SHA256 hash. This doesn't work for DeepVariant because:
- `tensorflow-macos` requires specific install ordering (numpy 1.x must be pinned first)
- Several packages need `--no-deps` to avoid pulling in the standard `tensorflow` package
- The full transitive dependency tree is hundreds of packages

Instead, the formula uses manual `system pip, "install", ...` calls, which mirrors the install.sh logic exactly.

### Why `bottle :unneeded`?

The formula downloads a pre-built tarball and creates a venv — there's no compilation step. `bottle :unneeded` tells Homebrew not to try building or caching a bottle.

---

## Troubleshooting

### `brew install` fails during pip installs

The pip install step can fail if network connectivity is interrupted (it downloads ~500MB of Python packages). Re-run:
```bash
brew reinstall deepvariant
```

### "platform already registered" crash

This means both `tensorflow` and `tensorflow-metal` are installed. The formula uses `tensorflow-macos` (not `tensorflow`). If you have a conflicting global TF install, it won't affect the Homebrew venv since it's isolated.

### Models not found

Models are not installed automatically. Run:
```bash
deepvariant-download-model WGS
```

### Intel Mac

This formula only supports Apple Silicon. Intel Macs are not supported because the pre-built binaries are ARM64 Mach-O.
