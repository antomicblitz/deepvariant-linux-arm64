#!/bin/bash
# DeepVariant macOS ARM64 Installer
#
# Installs pre-built DeepVariant binaries, sets up a Python environment
# with tensorflow-macos + tensorflow-metal for GPU acceleration, and downloads
# the requested model(s).
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | bash
#
# Environment variables:
#   DEEPVARIANT_HOME    Install directory (default: ~/.deepvariant)
#   MODEL_TYPES         Space-separated model types to download (default: WGS)
#                       Options: WGS WES PACBIO ONT_R104 HYBRID MASSEQ ALL NONE
#   USE_CONDA           Set to 1 to use conda instead of venv (auto-detected if
#                       Python 3.10 is not found but conda is available)
#   CONDA_ENV_NAME      Conda environment name (default: deepvariant)
#   SKIP_ENV            Set to 1 to skip Python environment creation entirely
#   SKIP_MODELS         Set to 1 to skip model downloads

set -euo pipefail

VERSION="1.9.0"
DEEPVARIANT_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
MODEL_TYPES="${MODEL_TYPES:-WGS}"
USE_CONDA="${USE_CONDA:-auto}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-deepvariant}"
SKIP_ENV="${SKIP_ENV:-0}"
SKIP_MODELS="${SKIP_MODELS:-0}"
TARBALL_URL="https://github.com/antomicblitz/deepvariant-macos-arm64-metal/releases/download/v${VERSION}/deepvariant-${VERSION}-macos-arm64.tar.gz"

# Backward compat: SKIP_VENV=1 still works
if [[ "${SKIP_VENV:-0}" == "1" ]]; then
  SKIP_ENV="1"
fi

echo "============================================"
echo " DeepVariant v${VERSION} macOS ARM64 Installer"
echo "============================================"
echo ""

################################################################################
# Preflight checks
################################################################################

echo "--- Checking system requirements..."

# Check macOS
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This installer is for macOS only."
  exit 1
fi

# Check ARM64
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: This installer is designed for Apple Silicon (arm64)."
  echo "         Running on $(uname -m) may have issues."
fi

# Check curl
if ! command -v curl &>/dev/null; then
  echo "ERROR: curl is required."
  exit 1
fi

# Detect conda
CONDA_CMD=""
for cmd in conda mamba micromamba; do
  if command -v "$cmd" &>/dev/null; then
    CONDA_CMD="$cmd"
    break
  fi
done

# Check if conda is x86_64 on ARM64 (reports osx-64 platform).
# `which conda` is a script wrapper, so `file` doesn't detect arch —
# use `conda info` to check the reported platform instead.
CONDA_IS_X86=false
if [[ -n "$CONDA_CMD" && "$(uname -m)" == "arm64" ]]; then
  CONDA_PLATFORM=$(${CONDA_CMD} info --json 2>/dev/null \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('platform',''))" 2>/dev/null || true)
  if [[ "$CONDA_PLATFORM" == "osx-64" ]]; then
    CONDA_IS_X86=true
  fi
fi

# Check Python 3.10 (needed for venv path)
PYTHON_CMD=""
for cmd in python3.10 python3; do
  if command -v "$cmd" &>/dev/null; then
    py_version=$("$cmd" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
    if [[ "$py_version" == "3.10" ]]; then
      PYTHON_CMD="$cmd"
      break
    fi
  fi
done

# Decide environment strategy
if [[ "${SKIP_ENV}" == "1" ]]; then
  ENV_TYPE="none"
elif [[ "${USE_CONDA}" == "1" ]]; then
  # User explicitly requested conda
  if [[ -z "$CONDA_CMD" ]]; then
    echo "ERROR: USE_CONDA=1 but no conda/mamba/micromamba found."
    echo "       Install Miniforge (native ARM64): https://github.com/conda-forge/miniforge"
    exit 1
  fi
  # Check if conda is x86_64 running under Rosetta on ARM64
  if [[ "$CONDA_IS_X86" == true ]]; then
    echo "  WARNING: Your conda is an x86_64 build running under Rosetta."
    echo "           tensorflow-macos requires native ARM64 Python."
    if [[ -n "$PYTHON_CMD" ]]; then
      echo "  Falling back to venv (your $PYTHON_CMD is native ARM64)."
      ENV_TYPE="venv"
    else
      echo ""
      echo "ERROR: Cannot create a working environment with x86_64 conda."
      echo "       Install native ARM64 conda (Miniforge):"
      echo "         curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh | bash"
      echo "       Or install Python 3.10: brew install python@3.10"
      exit 1
    fi
  else
    ENV_TYPE="conda"
  fi
elif [[ "${USE_CONDA}" == "0" ]]; then
  # User explicitly requested venv
  if [[ -z "$PYTHON_CMD" ]]; then
    echo "ERROR: Python 3.10 is required for venv mode."
    echo "       Install with: brew install python@3.10"
    echo "       Or use conda: USE_CONDA=1 curl -fsSL ... | bash"
    exit 1
  fi
  ENV_TYPE="venv"
else
  # Auto-detect: prefer venv if Python 3.10 exists, fall back to conda
  if [[ -n "$PYTHON_CMD" ]]; then
    ENV_TYPE="venv"
  elif [[ -n "$CONDA_CMD" ]]; then
    # Check if conda is x86_64 on ARM64 — can't install tensorflow-macos
    if [[ "$CONDA_IS_X86" == true ]]; then
      echo "ERROR: Python 3.10 not found and your conda is x86_64 (Rosetta)."
      echo "       tensorflow-macos requires native ARM64 Python."
      echo ""
      echo "  Option 1 (brew):  brew install python@3.10"
      echo "  Option 2 (conda): Install native ARM64 conda (Miniforge):"
      echo "                     https://github.com/conda-forge/miniforge"
      exit 1
    fi
    echo "  Python 3.10 not found, but ${CONDA_CMD} detected — using conda."
    ENV_TYPE="conda"
  else
    echo "ERROR: Python 3.10 is required (for tensorflow-macos 2.13.1 compatibility)."
    echo ""
    echo "  Option 1 (brew):  brew install python@3.10"
    echo "  Option 2 (conda): Install Miniforge (native ARM64 conda):"
    echo "                     https://github.com/conda-forge/miniforge"
    exit 1
  fi
fi

if [[ "$ENV_TYPE" == "venv" ]]; then
  echo "  Python: ${PYTHON_CMD} ($($PYTHON_CMD --version))"
  echo "  Environment: venv"
elif [[ "$ENV_TYPE" == "conda" ]]; then
  echo "  Conda: ${CONDA_CMD} ($(${CONDA_CMD} --version 2>&1 | head -1))"
  echo "  Environment: conda (${CONDA_ENV_NAME})"
else
  echo "  Environment: none (SKIP_ENV=1)"
fi

# Check for GNU parallel (needed by run_deepvariant.py for sharding)
# Conda will install it, so only warn for venv/none modes
if [[ "$ENV_TYPE" != "conda" ]] && ! command -v parallel &>/dev/null; then
  echo "  WARNING: GNU parallel not found. Multi-shard mode won't work."
  echo "           Install with: brew install parallel"
fi

echo "  Install directory: ${DEEPVARIANT_HOME}"
echo ""

################################################################################
# Download and extract binaries
################################################################################

echo "--- Downloading DeepVariant v${VERSION} binaries..."

DL_TMPDIR=$(mktemp -d)
TARBALL="${DL_TMPDIR}/deepvariant-${VERSION}-macos-arm64.tar.gz"

curl -fSL --progress-bar "${TARBALL_URL}" -o "${TARBALL}"

echo "--- Extracting..."
mkdir -p "${DEEPVARIANT_HOME}"
tar -xzf "${TARBALL}" -C "${DEEPVARIANT_HOME}" --strip-components=1

rm -rf "${DL_TMPDIR}"

# Make binaries executable
find "${DEEPVARIANT_HOME}/bin" -type f ! -name '*.zip' -exec chmod +x {} \;

echo "  Binaries installed to ${DEEPVARIANT_HOME}/bin/"

################################################################################
# Install pip packages (shared between venv and conda)
################################################################################

install_pip_packages() {
  echo "--- Installing Python packages (this may take a few minutes)..."

  # pip_q: install quietly and suppress harmless dependency-resolver warnings.
  # We intentionally use --no-deps for several packages (to avoid pulling in
  # the wrong tensorflow), which causes pip to print spurious "ERROR: pip's
  # dependency resolver" messages on every subsequent install. These are
  # cosmetic — real failures still abort via set -e (non-zero exit code).
  pip_q() {
    local _stderr
    _stderr=$(mktemp)
    if pip install -q "$@" 2>"$_stderr"; then
      rm -f "$_stderr"
    else
      cat "$_stderr" >&2
      rm -f "$_stderr"
      return 1
    fi
  }

  pip_q --upgrade pip

  # Pin NumPy to 1.x FIRST — tensorflow-macos 2.13.1 was compiled against
  # NumPy 1.x and will crash with AttributeError (_ARRAY_API not found) if
  # NumPy 2.x is present. This pin must come before any other install so that
  # pip's resolver never selects NumPy 2.x.
  pip_q "numpy>=1.22,<=1.24.3"

  # TensorFlow with Metal GPU
  pip_q "tensorflow-macos==2.13.1"
  pip_q "tensorflow-metal==1.0.0"

  # Packages that would pull in regular 'tensorflow' — install without deps
  pip_q --no-deps "tensorflow-hub==0.14.0"
  pip_q --no-deps "tensorflow-model-optimization==0.7.5"
  pip_q --no-deps "tf-models-official==2.13.1"

  # tf-models-official runtime deps that DeepVariant actually needs
  pip_q tensorflow-datasets gin-config seqeval \
    'opencv-python-headless>=4.5' tf-slim sacrebleu pyyaml

  # DeepVariant Python dependencies
  pip_q absl-py parameterized
  pip_q contextlib2
  pip_q etils typing_extensions importlib_resources
  pip_q 'sortedcontainers==2.1.0'
  pip_q 'intervaltree==3.1.0'
  pip_q 'mock>=2.0.0'
  pip_q ml_collections
  pip_q --ignore-installed PyYAML
  pip_q 'clu==0.0.9'
  pip_q 'protobuf==4.21.9'
  pip_q 'requests>=2.18'
  pip_q joblib psutil
  pip_q 'pandas==1.3.4'
  pip_q 'Pillow==9.5.0'
  pip_q 'pysam==0.20.0'
  pip_q 'scikit-learn==1.0.2'
  pip_q 'jax==0.4.35'
  pip_q 'markupsafe==2.1.1'

  # Re-pin NumPy as a safety net in case any package above upgraded it.
  # jax and tensorflow-datasets in particular are known to widen numpy bounds.
  pip_q --force-reinstall "numpy>=1.22,<=1.24.3"
}

################################################################################
# Create Python environment
################################################################################

if [[ "$ENV_TYPE" == "conda" ]]; then
  # ── Conda environment ──────────────────────────────────────────────────────
  echo ""

  # Check if env already exists
  if ${CONDA_CMD} env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME} "; then
    echo "--- Conda env '${CONDA_ENV_NAME}' already exists, skipping creation."
    echo "    To recreate: ${CONDA_CMD} env remove -n ${CONDA_ENV_NAME}"
  else
    echo "--- Creating conda environment '${CONDA_ENV_NAME}'..."

    # Force ARM64 platform — older conda installs (especially x86_64 builds
    # running under Rosetta) may default to osx-64 and fail to solve.
    export CONDA_SUBDIR=osx-arm64

    # Conda < 24.x has a broken solver that can't match __osx virtual package
    # versions on macOS 15+ (reports false "incompatible with your system" errors).
    # Update conda automatically if it's too old.
    CONDA_VER=$(${CONDA_CMD} --version 2>&1 | grep -oE '[0-9]+' | head -1)
    if [[ -n "$CONDA_VER" && "$CONDA_VER" -lt 24 ]]; then
      echo "--- Updating conda (${CONDA_CMD} $(${CONDA_CMD} --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+') is too old for macOS $(sw_vers -productVersion))..."
      ${CONDA_CMD} update -y -n base --override-channels -c conda-forge conda 2>&1 | tail -3
      echo ""
    fi

    # Create env with Python 3.10 and GNU parallel.
    # --override-channels ensures only conda-forge is used.
    ${CONDA_CMD} create -y -n "${CONDA_ENV_NAME}" python=3.10 parallel \
      --override-channels -c conda-forge

    echo "--- Activating conda environment..."

    # Activate the conda env for pip installs.
    # 'conda activate' requires shell init; use the env's Python directly.
    CONDA_PREFIX="$(${CONDA_CMD} env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"
    if [[ -z "$CONDA_PREFIX" ]]; then
      # micromamba uses a different list format
      CONDA_PREFIX="$(${CONDA_CMD} env list --json 2>/dev/null | python3 -c "
import sys, json
envs = json.load(sys.stdin).get('envs', [])
for e in envs:
    if e.endswith('/${CONDA_ENV_NAME}'):
        print(e)
        break
" 2>/dev/null || true)"
    fi

    if [[ -z "$CONDA_PREFIX" ]]; then
      echo "ERROR: Could not locate conda environment '${CONDA_ENV_NAME}'."
      exit 1
    fi

    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    export CONDA_PREFIX

    # Pin ARM64 subdir inside the env so future conda installs stay native
    ${CONDA_CMD} config --env --set subdir osx-arm64 2>/dev/null || true

    install_pip_packages

    echo "  Conda env '${CONDA_ENV_NAME}' created at ${CONDA_PREFIX}"
  fi

  # Record env type for wrapper scripts
  echo "conda:${CONDA_ENV_NAME}" > "${DEEPVARIANT_HOME}/.env_type"

elif [[ "$ENV_TYPE" == "venv" ]]; then
  # ── venv environment ────────────────────────────────────────────────────────
  VENV_DIR="${DEEPVARIANT_HOME}/venv"

  if [[ -d "${VENV_DIR}" ]]; then
    echo ""
    echo "--- Python venv already exists at ${VENV_DIR}, skipping."
    echo "    To recreate, delete it first: rm -rf ${VENV_DIR}"
  else
    echo ""
    echo "--- Creating Python virtual environment..."
    ${PYTHON_CMD} -m venv "${VENV_DIR}"

    # Activate venv for the rest of the script
    source "${VENV_DIR}/bin/activate"

    install_pip_packages

    echo "  Python venv created at ${VENV_DIR}"
  fi

  # Record env type for wrapper scripts
  echo "venv" > "${DEEPVARIANT_HOME}/.env_type"

else
  echo ""
  echo "--- Skipping Python environment creation (SKIP_ENV=1)."
fi

################################################################################
# Create wrapper scripts
################################################################################

echo ""
echo "--- Creating wrapper scripts..."

# Build activation snippet based on environment type
if [[ "$ENV_TYPE" == "conda" ]]; then
  # Wrapper activates the conda env by prepending its bin/ to PATH.
  # This avoids requiring 'conda init' in the user's shell.
  CONDA_PREFIX_RESOLVED="$(${CONDA_CMD} env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"
  if [[ -z "$CONDA_PREFIX_RESOLVED" ]]; then
    CONDA_PREFIX_RESOLVED="$(${CONDA_CMD} env list --json 2>/dev/null | python3 -c "
import sys, json
envs = json.load(sys.stdin).get('envs', [])
for e in envs:
    if e.endswith('/${CONDA_ENV_NAME}'):
        print(e)
        break
" 2>/dev/null || true)"
  fi
  ACTIVATE_SNIPPET="export PATH=\"${CONDA_PREFIX_RESOLVED}/bin:\${PATH}\""
else
  ACTIVATE_SNIPPET="source \"${DEEPVARIANT_HOME}/venv/bin/activate\""
fi

# run_deepvariant wrapper
cat > "${DEEPVARIANT_HOME}/bin/run_deepvariant" << WRAPPER
#!/bin/bash
export DEEPVARIANT_HOME="${DEEPVARIANT_HOME}"
${ACTIVATE_SNIPPET}
exec python3 "${DEEPVARIANT_HOME}/scripts/run_deepvariant.py" "\$@"
WRAPPER
chmod +x "${DEEPVARIANT_HOME}/bin/run_deepvariant"

# run_deeptrio wrapper
cat > "${DEEPVARIANT_HOME}/bin/run_deeptrio" << WRAPPER
#!/bin/bash
export DEEPVARIANT_HOME="${DEEPVARIANT_HOME}"
${ACTIVATE_SNIPPET}
exec python3 "${DEEPVARIANT_HOME}/scripts/run_deeptrio.py" "\$@"
WRAPPER
chmod +x "${DEEPVARIANT_HOME}/bin/run_deeptrio"

# Copy model download helper
cp "${DEEPVARIANT_HOME}/bin/run_deepvariant" /dev/null 2>/dev/null || true
DOWNLOAD_MODEL_SRC="$(cd "$(dirname "$0")" && pwd)/scripts/deepvariant-download-model"
if [[ -f "${DOWNLOAD_MODEL_SRC}" ]]; then
  cp "${DOWNLOAD_MODEL_SRC}" "${DEEPVARIANT_HOME}/bin/deepvariant-download-model"
else
  # Download from GitHub if running via curl pipe
  curl -fsSL "https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/scripts/deepvariant-download-model" \
    -o "${DEEPVARIANT_HOME}/bin/deepvariant-download-model"
fi
chmod +x "${DEEPVARIANT_HOME}/bin/deepvariant-download-model"

# Copy uninstall script
UNINSTALL_SRC="$(cd "$(dirname "$0")" && pwd)/scripts/uninstall.sh"
if [[ -f "${UNINSTALL_SRC}" ]]; then
  cp "${UNINSTALL_SRC}" "${DEEPVARIANT_HOME}/scripts/uninstall.sh"
else
  # Download from GitHub if running via curl pipe
  curl -fsSL "https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/scripts/uninstall.sh" \
    -o "${DEEPVARIANT_HOME}/scripts/uninstall.sh"
fi
chmod +x "${DEEPVARIANT_HOME}/scripts/uninstall.sh"

# Copy quicktest script
QUICKTEST_SRC="$(cd "$(dirname "$0")" && pwd)/scripts/quicktest.sh"
if [[ -f "${QUICKTEST_SRC}" ]]; then
  cp "${QUICKTEST_SRC}" "${DEEPVARIANT_HOME}/scripts/quicktest.sh"
else
  # Download from GitHub if running via curl pipe
  curl -fsSL "https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/scripts/quicktest.sh" \
    -o "${DEEPVARIANT_HOME}/scripts/quicktest.sh"
fi
chmod +x "${DEEPVARIANT_HOME}/scripts/quicktest.sh"

# Create uninstall wrapper in bin/
cat > "${DEEPVARIANT_HOME}/bin/deepvariant-uninstall" << 'WRAPPER'
#!/bin/bash
DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
exec "${DV_HOME}/scripts/uninstall.sh" "$@"
WRAPPER
chmod +x "${DEEPVARIANT_HOME}/bin/deepvariant-uninstall"

# Create quicktest wrapper in bin/ (with env activation, like run_deepvariant)
cat > "${DEEPVARIANT_HOME}/bin/deepvariant-quicktest" << WRAPPER
#!/bin/bash
export DEEPVARIANT_HOME="${DEEPVARIANT_HOME}"
${ACTIVATE_SNIPPET}
exec "${DEEPVARIANT_HOME}/scripts/quicktest.sh" "\$@"
WRAPPER
chmod +x "${DEEPVARIANT_HOME}/bin/deepvariant-quicktest"

echo "  Created: run_deepvariant, run_deeptrio, deepvariant-download-model, deepvariant-uninstall, deepvariant-quicktest"

################################################################################
# Download models
################################################################################

if [[ "${SKIP_MODELS}" != "1" && "${MODEL_TYPES}" != "NONE" ]]; then
  echo ""
  echo "--- Downloading models: ${MODEL_TYPES}"

  export DEEPVARIANT_HOME
  "${DEEPVARIANT_HOME}/bin/deepvariant-download-model" ${MODEL_TYPES}
else
  echo ""
  echo "--- Skipping model downloads."
  echo "    Download models later with: deepvariant-download-model WGS"
fi

################################################################################
# Update shell profile
################################################################################

echo ""
echo "--- Updating shell profile..."

SHELL_RC=""
if [[ -f "$HOME/.zshrc" ]] || [[ "$(basename "$SHELL")" == "zsh" ]]; then
  SHELL_RC="$HOME/.zshrc"
elif [[ -f "$HOME/.bash_profile" ]]; then
  SHELL_RC="$HOME/.bash_profile"
elif [[ -f "$HOME/.bashrc" ]]; then
  SHELL_RC="$HOME/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
  MARKER="# DeepVariant macOS ARM64"
  if ! grep -q "$MARKER" "$SHELL_RC" 2>/dev/null; then
    cat >> "$SHELL_RC" << PROFILE

${MARKER}
export DEEPVARIANT_HOME="${DEEPVARIANT_HOME}"
export PATH="\${DEEPVARIANT_HOME}/bin:\${PATH}"
PROFILE
    echo "  Added DEEPVARIANT_HOME and PATH to ${SHELL_RC}"
  else
    echo "  Shell profile already configured in ${SHELL_RC}"
  fi
else
  echo "  Could not detect shell profile. Add these to your profile manually:"
  echo "    export DEEPVARIANT_HOME=\"${DEEPVARIANT_HOME}\""
  echo "    export PATH=\"\${DEEPVARIANT_HOME}/bin:\${PATH}\""
fi

################################################################################
# Verification
################################################################################

echo ""
echo "--- Verifying installation..."

# Activate the right environment for verification
VERIFY_OK=false
if [[ "$ENV_TYPE" == "conda" ]]; then
  CONDA_PREFIX_RESOLVED="$(${CONDA_CMD} env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')"
  if [[ -z "$CONDA_PREFIX_RESOLVED" ]]; then
    CONDA_PREFIX_RESOLVED="$(${CONDA_CMD} env list --json 2>/dev/null | python3 -c "
import sys, json
envs = json.load(sys.stdin).get('envs', [])
for e in envs:
    if e.endswith('/${CONDA_ENV_NAME}'):
        print(e)
        break
" 2>/dev/null || true)"
  fi
  if [[ -n "$CONDA_PREFIX_RESOLVED" ]]; then
    export PATH="${CONDA_PREFIX_RESOLVED}/bin:${PATH}"
    VERIFY_OK=true
  fi
elif [[ "$ENV_TYPE" == "venv" && -d "${DEEPVARIANT_HOME}/venv" ]]; then
  source "${DEEPVARIANT_HOME}/venv/bin/activate"
  VERIFY_OK=true
fi

if [[ "$VERIFY_OK" == true ]]; then
  export DEEPVARIANT_HOME

  # Capture help output separately — absl-py exits non-zero on --help
  # (mark_flag_as_required warning) and pipefail would propagate that
  # exit code through the pipeline, making grep's success irrelevant.
  ME_HELP=$(python3 "${DEEPVARIANT_HOME}/bin/make_examples.zip" --help 2>&1 || true)
  if echo "$ME_HELP" | grep -q "creates tf.Example protos"; then
    echo "  make_examples: OK"
  else
    echo "  make_examples: FAILED (may need tensorflow-macos installed)"
  fi

  # Check Metal GPU
  GPU_STATUS=$(python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'ENABLED ({len(gpus)} device(s))' if gpus else 'not available (CPU only)')
" 2>/dev/null || echo "check failed")
  echo "  Metal GPU: ${GPU_STATUS}"
fi

################################################################################
# Done
################################################################################

echo ""
echo "============================================"
echo " DeepVariant v${VERSION} installed successfully"
echo "============================================"
echo ""
echo "To start using DeepVariant, either:"
echo "  1. Open a new terminal, or"
echo "  2. Run: source ${SHELL_RC:-~/.zshrc}"
if [[ "$ENV_TYPE" == "conda" ]]; then
  echo "  3. Activate conda: conda activate ${CONDA_ENV_NAME}"
fi
echo ""
echo "Quick start:"
echo "  run_deepvariant \\"
echo "    --model_type=WGS \\"
echo "    --ref=reference.fasta \\"
echo "    --reads=input.bam \\"
echo "    --output_vcf=output.vcf \\"
echo "    --num_shards=\$(sysctl -n hw.ncpu)"
echo ""
echo "Download additional models:"
echo "  deepvariant-download-model WES PACBIO ONT_R104"
echo "  deepvariant-download-model WGS --deeptrio"
echo ""
