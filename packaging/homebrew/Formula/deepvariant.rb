# Homebrew formula for DeepVariant — macOS ARM64 native build
#
# Installs pre-built DeepVariant binaries with tensorflow-macos + tensorflow-metal
# for Metal GPU-accelerated variant calling on Apple Silicon.
#
# This formula is intended for a custom tap (antomicblitz/homebrew-deepvariant),
# NOT homebrew-core, because it ships pre-built Bazel binaries.
#
# Usage:
#   brew tap antomicblitz/deepvariant
#   brew install deepvariant
#
class Deepvariant < Formula
  desc "Deep learning variant caller for genomic data (Apple Silicon native)"
  homepage "https://github.com/antomicblitz/deepvariant-macos-arm64-metal"
  url "https://github.com/antomicblitz/deepvariant-macos-arm64-metal/releases/download/v1.9.0/deepvariant-1.9.0-macos-arm64.tar.gz"
  sha256 "30b0bfc93634bd869ec624ed30d67a20c60a1861c833430eb24be3594fa89bd9"
  license "BSD-3-Clause"
  version "1.9.0"

  depends_on arch: :arm64
  depends_on :macos
  depends_on "python@3.10"
  depends_on "parallel"
  depends_on "samtools" => :recommended

  def install
    python = Formula["python@3.10"].opt_bin/"python3.10"
    venv = libexec/"venv"

    # Create a dedicated virtualenv
    system python, "-m", "venv", venv
    venv_pip = venv/"bin/pip"
    venv_python = venv/"bin/python3"

    # Upgrade pip and install setuptools (pysam needs pkg_resources,
    # which was removed in setuptools >= 72)
    system venv_pip, "install", "-q", "--upgrade", "pip", "setuptools<72", "wheel"

    # Pin NumPy to 1.x FIRST — tensorflow-macos 2.13.1 requires NumPy 1.x.
    # NumPy 2.x causes AttributeError (_ARRAY_API not found).
    system venv_pip, "install", "-q", "numpy>=1.22,<=1.24.3"

    # TensorFlow for macOS ARM64
    # tensorflow-metal provides ~4.25x speedup for call_variants via Metal GPU
    system venv_pip, "install", "-q", "tensorflow-macos==2.13.1"
    system venv_pip, "install", "-q", "tensorflow-metal==1.0.0"

    # Packages that would pull in regular 'tensorflow' — install without deps
    system venv_pip, "install", "-q", "--no-deps", "tensorflow-hub==0.14.0"
    system venv_pip, "install", "-q", "--no-deps", "tensorflow-model-optimization==0.7.5"
    system venv_pip, "install", "-q", "--no-deps", "tf-models-official==2.13.1"

    # tf-models-official runtime deps
    system venv_pip, "install", "-q",
      "tensorflow-datasets", "gin-config", "seqeval",
      "opencv-python-headless>=4.5", "tf-slim", "sacrebleu", "pyyaml"

    # DeepVariant Python dependencies
    system venv_pip, "install", "-q",
      "absl-py", "parameterized", "contextlib2", "etils",
      "typing_extensions", "importlib_resources",
      "sortedcontainers==2.1.0", "intervaltree==3.1.0",
      "mock>=2.0.0", "ml_collections", "clu==0.0.9",
      "protobuf==4.21.9", "requests>=2.18",
      "joblib", "psutil", "ipython",
      "pandas==1.3.4", "Pillow==9.5.0",
      "scikit-learn==1.0.2", "jax==0.4.35", "markupsafe==2.1.1"
    system venv_pip, "install", "-q", "--ignore-installed", "PyYAML"

    # pysam 0.20.0 requires:
    # - setuptools<72 (imports pkg_resources, removed in 72+)
    # - Cython<3 (Cython 3.x has breaking API changes)
    # Use --no-build-isolation so pip uses our pinned versions.
    system venv_pip, "install", "-q", "Cython<3"
    system venv_pip, "install", "-q", "--no-build-isolation", "pysam==0.20.0"

    # Re-pin NumPy (safety net)
    system venv_pip, "install", "-q", "--force-reinstall", "numpy>=1.22,<=1.24.3"

    # Fix tensorflow-metal rpath: the Metal plugin's dylib expects
    # _pywrap_tensorflow_internal.so at @rpath/ which resolves to a
    # _solib_darwin_arm64 directory that doesn't exist in pip installs.
    # Create the expected directory structure with a symlink.
    site_packages = venv/"lib/python3.10/site-packages"
    solib_dir = site_packages/"_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf"
    solib_dir.mkpath
    pywrap = site_packages/"tensorflow/python/_pywrap_tensorflow_internal.so"
    ln_sf pywrap, solib_dir/"_pywrap_tensorflow_internal.so" if pywrap.exist?

    # Install pre-built binaries
    (libexec/"bin").install Dir["bin/*"]
    (libexec/"scripts").install Dir["scripts/*"]

    # Make all binaries executable
    (libexec/"bin").glob("*").each do |f|
      f.chmod(0755) unless f.extname == ".zip"
    end

    # Create wrapper scripts for the main DeepVariant binaries
    dv_bins = %w[
      make_examples call_variants postprocess_variants
      vcf_stats_report show_examples fast_pipeline
    ]
    dv_bins.each do |name|
      bin_path = libexec/"bin"/name
      next unless bin_path.exist?

      if name == "fast_pipeline"
        # Native C++ binary — run directly
        bin.install_symlink bin_path
      else
        # Python zip binary — the zip's __main__.py re-execs python3 from
        # PATH, so we must ensure the venv's bin is first on PATH.
        (bin/name).write <<~BASH
          #!/bin/bash
          export PATH="#{venv}/bin:$PATH"
          exec "#{venv_python}" "#{bin_path}" "$@"
        BASH
        (bin/name).chmod(0755)
      end
    end

    # run_deepvariant wrapper (the main pipeline runner)
    (bin/"run_deepvariant").write <<~BASH
      #!/bin/bash
      export PATH="#{venv}/bin:$PATH"
      exec "#{venv_python}" "#{libexec}/scripts/run_deepvariant.py" "$@"
    BASH
    (bin/"run_deepvariant").chmod(0755)

    # run_deeptrio wrapper
    if (libexec/"scripts/run_deeptrio.py").exist?
      (bin/"run_deeptrio").write <<~BASH
        #!/bin/bash
        export PATH="#{venv}/bin:$PATH"
        exec "#{venv_python}" "#{libexec}/scripts/run_deeptrio.py" "$@"
      BASH
      (bin/"run_deeptrio").chmod(0755)
    end

    # Model download helper
    if (libexec/"scripts/deepvariant-download-model").exist?
      (bin/"deepvariant-download-model").write <<~BASH
        #!/bin/bash
        exec "#{libexec}/scripts/deepvariant-download-model" "$@"
      BASH
      (bin/"deepvariant-download-model").chmod(0755)
    end

    # Quicktest script — runs a small variant calling job to verify installation
    (bin/"deepvariant-quicktest").write <<~BASH
      #!/bin/bash
      # DeepVariant quicktest — verifies end-to-end pipeline via Homebrew install
      set -euo pipefail

      VENV_PYTHON="#{venv_python}"
      # Ensure venv python is first on PATH — the zip binaries' __main__.py
      # re-execs python3 from PATH, and it must find the venv's Python.
      export PATH="#{venv}/bin:$PATH"
      DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
      DATA_DIR="$HOME/deepvariant-quicktest"
      REGION="chr20:10000000-10010000"
      SHARDS=10

      GREEN='\\033[0;32m'; RED='\\033[0;31m'; NC='\\033[0m'
      pass() { echo -e "${GREEN}✓${NC}  $*"; }
      fail() { echo -e "${RED}✗${NC}  $*"; exit 1; }

      echo "================================================"
      echo "  DeepVariant v#{version} — Homebrew Quicktest"
      echo "================================================"
      echo ""

      # ── [ 1/5 ] Metal GPU detection ──────────────────────────────
      echo "[ 1/5 ]  Metal GPU detection"
      GPU_COUNT=$("$VENV_PYTHON" -c "
      import tensorflow as tf
      gpus = [d for d in tf.config.list_physical_devices() if 'GPU' in d.device_type]
      print(len(gpus))
      " 2>/dev/null || echo 0)

      if [[ "$GPU_COUNT" -ge 1 ]]; then
          pass "Metal GPU ENABLED ($GPU_COUNT device)"
      else
          fail "No Metal GPU detected. Check tensorflow-metal installation."
      fi
      echo ""

      # ── [ 2/5 ] Check model ───────────────────────────────────────
      echo "[ 2/5 ]  Checking WGS model"
      MODEL_DIR="$DV_HOME/models/wgs"
      if [[ -d "$MODEL_DIR" && -f "$MODEL_DIR/variables/variables.index" ]]; then
          pass "WGS model found at $MODEL_DIR"
      else
          echo "         WGS model not found. Downloading..."
          "$(dirname "$0")/deepvariant-download-model" WGS || fail "Model download failed"
          pass "WGS model downloaded"
      fi
      echo ""

      # ── [ 3/5 ] Download test data ──────────────────────────────
      echo "[ 3/5 ]  Downloading quickstart test data"
      mkdir -p "$DATA_DIR"

      BASE_URL="https://storage.googleapis.com/deepvariant/quickstart-testdata"
      FILES=(
          "ucsc.hg19.chr20.unittest.fasta"
          "ucsc.hg19.chr20.unittest.fasta.fai"
          "ucsc.hg19.chr20.unittest.fasta.gz"
          "ucsc.hg19.chr20.unittest.fasta.gz.fai"
          "ucsc.hg19.chr20.unittest.fasta.gz.gzi"
          "NA12878_S1.chr20.10_10p1mb.bam"
          "NA12878_S1.chr20.10_10p1mb.bam.bai"
      )

      for f in "${FILES[@]}"; do
          if [[ -f "$DATA_DIR/$f" ]]; then
              echo "         $f  (cached)"
          else
              echo "         $f  (downloading...)"
              curl -sSL "$BASE_URL/$f" -o "$DATA_DIR/$f" || fail "Failed to download $f"
          fi
      done
      echo ""

      # ── [ 4/5 ] Run DeepVariant pipeline ─────────────────────────
      echo "[ 4/5 ]  Running DeepVariant  (region: ${REGION}  |  shards: $SHARDS)"
      echo ""

      TMPDIR_RUN=$(mktemp -d)
      trap 'rm -rf "$TMPDIR_RUN"' EXIT

      REF="$DATA_DIR/ucsc.hg19.chr20.unittest.fasta"
      BAM="$DATA_DIR/NA12878_S1.chr20.10_10p1mb.bam"
      EXAMPLES="$TMPDIR_RUN/make_examples.tfrecord@${SHARDS}.gz"
      GVCF_RECORDS="$TMPDIR_RUN/gvcf.tfrecord@${SHARDS}.gz"
      CV_OUTPUT="$TMPDIR_RUN/callvariantsoutput.tfrecord.gz"
      OUT_VCF="$DATA_DIR/output.vcf.gz"
      OUT_GVCF="$DATA_DIR/output.g.vcf.gz"

      export TF_CPP_MIN_LOG_LEVEL=0
      export TF2_BEHAVIOR=1
      export TPU_ML_PLATFORM=Tensorflow

      # Use Homebrew-installed binaries via the venv python
      MAKE_EXAMPLES="$VENV_PYTHON #{libexec}/bin/make_examples"
      CALL_VARIANTS="$VENV_PYTHON #{libexec}/bin/call_variants"
      POSTPROCESS="$VENV_PYTHON #{libexec}/bin/postprocess_variants"

      echo "***** Running make_examples *****"
      time seq 0 $((SHARDS - 1)) | parallel -q --halt 2 --line-buffer \\
          $MAKE_EXAMPLES \\
              --mode calling \\
              --ref "$REF" \\
              --reads "$BAM" \\
              --examples "$EXAMPLES" \\
              --checkpoint "$MODEL_DIR" \\
              --gvcf "$GVCF_RECORDS" \\
              --regions "$REGION" \\
              --track_ref_reads \\
              --task {}

      echo ""
      echo "***** Running call_variants *****"
      time $CALL_VARIANTS \\
          --outfile "$CV_OUTPUT" \\
          --examples "$EXAMPLES" \\
          --checkpoint "$MODEL_DIR"

      echo ""
      echo "***** Running postprocess_variants (--cpus 1) *****"
      time $POSTPROCESS \\
          --ref "$REF" \\
          --infile "$CV_OUTPUT" \\
          --outfile "$OUT_VCF" \\
          --gvcf_outfile "$OUT_GVCF" \\
          --nonvariant_site_tfrecord_path "$GVCF_RECORDS" \\
          --cpus 1

      # ── [ 5/5 ] Results ──────────────────────────────────────────
      echo ""
      echo "[ 5/5 ]  Results"
      if [[ -f "$OUT_VCF" ]]; then
          VCF_CONTENT=$(zcat "$OUT_VCF" 2>/dev/null || true)
          VARIANT_COUNT=$(echo "$VCF_CONTENT" | grep -vc '^#' || true)
          pass "VCF confirmed — $VARIANT_COUNT variants called"
          echo "         Output: $OUT_VCF"
          echo "         gVCF:   $OUT_GVCF"
      else
          fail "Output VCF not found: $OUT_VCF"
      fi

      echo ""
      echo "================================================"
      echo "  PASSED — DeepVariant Homebrew quicktest complete"
      echo "================================================"
    BASH
    (bin/"deepvariant-quicktest").chmod(0755)
  end

  def caveats
    <<~EOS
      DeepVariant v#{version} for macOS ARM64 (Apple Silicon)

      Metal GPU acceleration is enabled (tensorflow-metal), providing
      ~4.25x speedup for call_variants inference.

      Get started:
        deepvariant-download-model WGS    # download model (~200 MB)
        deepvariant-quicktest              # verify everything works

      Available models: WGS, WES, PACBIO, ONT_R104, HYBRID, MASSEQ

      Usage:
        run_deepvariant \\
          --model_type WGS \\
          --ref reference.fasta \\
          --reads input.bam \\
          --output_vcf output.vcf \\
          --num_shards $(sysctl -n hw.perflevel0.logicalcpu)
    EOS
  end

  test do
    # Verify Python environment and TensorFlow import
    venv_python = libexec/"venv/bin/python3"
    system venv_python, "-c", "import tensorflow as tf; print(tf.__version__)"

    # Verify Metal GPU detection
    output = shell_output("#{venv_python} -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices(\"GPU\")))'")
    assert_match(/[0-9]+/, output.strip)

    # Verify binaries exist and are runnable
    assert_predicate bin/"make_examples", :executable?
    assert_predicate bin/"call_variants", :executable?
    assert_predicate bin/"postprocess_variants", :executable?
    assert_predicate bin/"run_deepvariant", :executable?
  end
end
