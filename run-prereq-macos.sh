#!/bin/bash

# Copyright 2017 Google LLC.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# macOS ARM64 version of run-prereq.sh
# Installs runtime dependencies for DeepVariant on macOS with tensorflow-metal.

set -euo pipefail

echo "========== macOS ARM64 runtime prerequisites"
echo "========== Load config settings."

source settings.sh

################################################################################
# Verify macOS ARM64
################################################################################

note_build_stage "Verify macOS ARM64 environment"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This script is for macOS only. Use run-prereq.sh for Linux."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: This script is designed for Apple Silicon (arm64)."
  echo "         Running on $(uname -m) may have issues."
fi

################################################################################
# Homebrew system dependencies
################################################################################

note_build_stage "Install Homebrew system dependencies"

if ! command -v brew &>/dev/null; then
  echo "ERROR: Homebrew is required. Install from https://brew.sh"
  exit 1
fi

# For htslib: compression and SSL libraries
brew install openssl@3 xz bzip2 lzma || true
# For the de Bruijn graph
brew install boost || true

################################################################################
# Python setup
################################################################################

note_build_stage "Setup Python"

# Ensure Python 3.10 is available
if ! command -v python3.10 &>/dev/null; then
  echo "Python 3.10 not found. Installing via Homebrew..."
  brew install python@3.10 || true
fi

PYTHON_CMD="${PYTHON_BIN_PATH:-python3}"
echo "Using Python: ${PYTHON_CMD} ($(${PYTHON_CMD} --version))"

PIP_ARGS=("-q")

# Ensure pip is up to date
${PYTHON_CMD} -m pip install --upgrade pip

echo "$(${PYTHON_CMD} -m pip --version)"

################################################################################
# Python packages
################################################################################

note_build_stage "Install python3 packages"

${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" contextlib2
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" etils typing_extensions importlib_resources
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'enum34==1.1.8'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'sortedcontainers==2.1.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'intervaltree==3.1.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'mock>=2.0.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" ml_collections
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --ignore-installed PyYAML
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'clu==0.0.9'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'protobuf==4.21.9'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'argparse==1.4.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'pyasn1<0.5.0,>=0.4.6'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'requests>=2.18'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --ignore-installed 'oauth2client>=4.0.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'crcmod>=1.7'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'six>=1.11.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" joblib
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" psutil
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --upgrade google-api-python-client
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'pandas==1.3.4'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'jsonschema==3.2.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'altair==4.1.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'Pillow==9.5.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'ipython==8.22.2'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'pysam==0.20.0'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'scikit-learn==1.0.2'
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'setuptools==61.0.0'

# Install tf-models-official without deps to avoid pulling tensorflow-text,
# which has no ARM64 wheels for version 2.13.x. DeepVariant only uses
# official.modeling.optimization, which does not require tensorflow-text.
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --no-deps "tf-models-official==2.13.1"
# Install the tf-models-official dependencies that DeepVariant actually needs.
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" tensorflow-datasets gin-config seqeval \
  'opencv-python-headless>=4.5' tf-slim sacrebleu pyyaml

################################################################################
# TensorFlow + Metal GPU
################################################################################

note_build_stage "Install TensorFlow with Metal GPU support"

if [[ "${DV_USE_PREINSTALLED_TF}" = "1" ]]; then
  echo "Skipping TensorFlow installation at user request; will use pre-installed TensorFlow."
else
  echo "Installing TensorFlow ${DV_TENSORFLOW_STANDARD_CPU_WHL_VERSION} for macOS ARM64"
  # IMPORTANT: Use tensorflow-macos (Apple's build) + tensorflow-metal for GPU.
  # The standard 'tensorflow' pip package also includes Metal registration,
  # which causes a "platform already registered" crash when tensorflow-metal
  # tries to register it again. tensorflow-macos is designed to work with
  # tensorflow-metal and avoids this conflict.
  ${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --upgrade "tensorflow-macos==${DV_TENSORFLOW_STANDARD_CPU_WHL_VERSION}"
  ${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" "tensorflow-metal==1.0.0"

  # Install deps that would otherwise pull in regular 'tensorflow' and break things
  ${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --no-deps "tensorflow-hub==0.14.0"
  ${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --no-deps "tensorflow-model-optimization==0.7.5"

  # Verify TensorFlow and Metal GPU
  echo "Verifying TensorFlow setup..."
  ${PYTHON_CMD} -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('Available devices:')
for d in tf.config.list_physical_devices():
    print(f'  {d.device_type}: {d.name}')
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f'Metal GPU acceleration: ENABLED ({len(gpu_devices)} GPU(s))')
else:
    print('Metal GPU acceleration: NOT AVAILABLE (CPU only)')
" || echo "WARNING: TensorFlow verification failed (non-fatal)"
fi

# Temporary fix for markupsafe compatibility
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" --upgrade 'markupsafe==2.0.1'

################################################################################
# Misc dependencies (macOS equivalents)
################################################################################

note_build_stage "Install other packages via Homebrew"

# For htslib (SSL, compression)
brew install openssl@3 curl bzip2 xz || true

# For the de Bruijn graph
brew install boost || true

# Pin critical dependencies
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" 'protobuf==4.21.9'

# JAX
${PYTHON_CMD} -m pip install "${PIP_ARGS[@]}" "jax==0.4.35"

note_build_stage "run-prereq-macos.sh complete"

echo ""
echo "=========================================="
echo "macOS ARM64 runtime prerequisites installed."
echo "IMPORTANT: Use tensorflow-macos + tensorflow-metal for GPU."
echo "  Do NOT install the regular 'tensorflow' package alongside"
echo "  tensorflow-metal — it causes a 'platform already registered' crash."
echo "  A clean venv is recommended to avoid package conflicts."
echo "=========================================="
