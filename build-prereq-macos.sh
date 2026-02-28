#!/bin/bash
set -euo pipefail

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

# macOS ARM64 version of build-prereq.sh
# Sets up the build environment for DeepVariant on macOS with Apple Silicon.

echo "========== macOS ARM64 build prerequisites"
echo "========== Load config settings."

source settings.sh

################################################################################
# Verify macOS ARM64
################################################################################

note_build_stage "Verify macOS ARM64 environment"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: This script is for macOS only. Use build-prereq.sh for Linux."
  exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: This script is designed for Apple Silicon (arm64)."
fi

################################################################################
# Install runtime prerequisites
################################################################################

note_build_stage "Install the runtime packages"

./run-prereq-macos.sh

################################################################################
# Homebrew build dependencies
################################################################################

note_build_stage "Install Homebrew build dependencies"

brew install pkg-config zip unzip curl git wget cmake ninja coreutils || true
# coreutils provides GNU ln (gln), GNU sed (gsed), GNU realpath (grealpath)
brew install gnu-sed || true

################################################################################
# Bazel
################################################################################

note_build_stage "Install bazel"

function ensure_wanted_bazel_version {
  local wanted_bazel_version=$1
  rm -rf ~/bazel
  mkdir -p ~/bazel

  if
    v=$(bazel --bazelrc=/dev/null --ignore_all_rc_files version 2>/dev/null) &&
    echo "$v" | awk -v b="$wanted_bazel_version" '/Build label/ { exit ($3 != b)}'
  then
    echo "Bazel ${wanted_bazel_version} already installed on the machine, not reinstalling"
  else
    pushd ~/bazel
    # Bazel 5.3.0 has darwin-arm64 installers available
    curl -L -O "https://github.com/bazelbuild/bazel/releases/download/${wanted_bazel_version}/bazel-${wanted_bazel_version}-installer-darwin-arm64.sh"
    chmod +x "bazel-${wanted_bazel_version}-installer-darwin-arm64.sh"
    "./bazel-${wanted_bazel_version}-installer-darwin-arm64.sh" --user > /dev/null
    rm "bazel-${wanted_bazel_version}-installer-darwin-arm64.sh"
    popd
  fi
}

ensure_wanted_bazel_version "${DV_BAZEL_VERSION}"

################################################################################
# Abseil C++ (from source)
################################################################################

note_build_stage "Build and install abseil-cpp"

ABSL_PIN="${ABSL_PIN:-29bf8085f3bf17b84d30e34b3d7ff8248fda404e}"

if [[ -d /tmp/abseil-cpp-build ]]; then
  rm -rf /tmp/abseil-cpp-build
fi

git clone https://github.com/abseil/abseil-cpp.git /tmp/abseil-cpp-build
cd /tmp/abseil-cpp-build
git checkout "${ABSL_PIN}"
mkdir build && cd build
cmake .. \
  -DCMAKE_POSITION_INDEPENDENT_CODE=true \
  -DCMAKE_INSTALL_PREFIX=/opt/homebrew \
  -DCMAKE_CXX_STANDARD=17 \
  -DABSL_PROPAGATE_CXX_STD=ON
make -j"$(sysctl -n hw.ncpu)"
sudo make install
cd /tmp
rm -rf /tmp/abseil-cpp-build

# Install python runtime and test dependencies
pip3 install -q absl-py parameterized

################################################################################
# CLIF C++ runtime (headers and sources only)
################################################################################

note_build_stage "Install CLIF C++ runtime"

# DeepVariant only needs the CLIF C++ runtime library (@clif//:cpp_runtime),
# not the CLIF code generator (pyclif). The cpp_runtime is a set of .cc and .h
# files under clif/python/.

CLIF_INSTALL_DIR="/opt/homebrew"
CLIF_DIR="${CLIF_INSTALL_DIR}/clif"

if [[ ! -d "${CLIF_DIR}/python" ]]; then
  echo "Cloning CLIF repository for C++ runtime sources..."
  CLIF_TMP="/tmp/clif-build"
  rm -rf "${CLIF_TMP}"
  git clone https://github.com/google/clif.git "${CLIF_TMP}"

  # Copy only the C++ runtime files we need
  sudo mkdir -p "${CLIF_DIR}/python"
  sudo cp "${CLIF_TMP}"/clif/python/*.cc "${CLIF_DIR}/python/"
  sudo cp "${CLIF_TMP}"/clif/python/*.h "${CLIF_DIR}/python/"
  # Remove test files (not needed)
  sudo rm -f "${CLIF_DIR}/python/"*_test.cc

  # Also create dummy pyclif and pyclif_proto scripts so the sh_binary
  # targets in clif.BUILD don't fail. These are never actually called.
  sudo mkdir -p "${CLIF_DIR}/bin"
  echo '#!/bin/bash' | sudo tee "${CLIF_DIR}/bin/pyclif" > /dev/null
  echo 'echo "pyclif not available on macOS - using pybind11 instead"' | sudo tee -a "${CLIF_DIR}/bin/pyclif" > /dev/null
  sudo chmod +x "${CLIF_DIR}/bin/pyclif"
  echo '#!/bin/bash' | sudo tee "${CLIF_DIR}/bin/pyclif_proto" > /dev/null
  echo 'echo "pyclif_proto not available on macOS - using pybind11 instead"' | sudo tee -a "${CLIF_DIR}/bin/pyclif_proto" > /dev/null
  sudo chmod +x "${CLIF_DIR}/bin/pyclif_proto"

  rm -rf "${CLIF_TMP}"
  echo "CLIF C++ runtime installed to ${CLIF_DIR}"
else
  echo "CLIF C++ runtime already installed at ${CLIF_DIR}"
fi

################################################################################
# TensorFlow source (for C++ headers and Bazel build system)
################################################################################

note_build_stage "Download and configure TensorFlow sources"

# Getting the directory before switching out.
DV_DIR=$(pwd)

if [[ ! -d ../tensorflow ]]; then
  note_build_stage "Cloning TensorFlow from github as ../tensorflow doesn't exist"
  (cd .. && git clone https://github.com/tensorflow/tensorflow)
fi

# Configure TensorFlow for macOS ARM64.
# PYTHON_BIN_PATH and PYTHON_LIB_PATH are set in settings.sh.
(cd ../tensorflow &&
 git checkout "${DV_CPP_TENSORFLOW_TAG}" &&
 echo | ./configure)

# Update absl version to match what DeepVariant expects (same as Linux).
wget -q https://raw.githubusercontent.com/tensorflow/tensorflow/r2.13/third_party/absl/workspace.bzl \
  -O ../tensorflow/third_party/absl/workspace.bzl
rm -f ../tensorflow/third_party/absl/absl_designated_initializers.patch

# Use GNU sed (gsed) for macOS compatibility
gsed -i -e 's|b971ac5250ea8de900eae9f95e06548d14cd95fe|29bf8085f3bf17b84d30e34b3d7ff8248fda404e|g' ../tensorflow/third_party/absl/workspace.bzl
gsed -i -e 's|8eeec9382fc0338ef5c60053f3a4b0e0708361375fe51c9e65d0ce46ccfe55a7|affb64f374b16877e47009df966d0a9403dbf7fe613fe1f18e49802c84f6421e|g' ../tensorflow/third_party/absl/workspace.bzl
gsed -i -e 's|patch_file = \["//third_party/absl:absl_designated_initializers.patch"\],||g' ../tensorflow/third_party/absl/workspace.bzl

# Update tensorflow.bzl to use _message.so
patch ../tensorflow/tensorflow/tensorflow.bzl "${DV_DIR}"/third_party/tensorflow.bzl.patch

# Update pybind11 version (same as Linux)
gsed -i -e 's|v2.10.0.tar.gz|a7b91e33269ab6f3f90167291af2c4179fc878f5.zip|g' ../tensorflow/tensorflow/workspace2.bzl
gsed -i -e 's|eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec|09d2ab67e91457c966eb335b361bdc4d27ece2d4dea681d22e5d8307e0e0c023|g' ../tensorflow/tensorflow/workspace2.bzl
gsed -i -e 's|pybind11-2.10.0|pybind11-a7b91e33269ab6f3f90167291af2c4179fc878f5|g' ../tensorflow/tensorflow/workspace2.bzl

################################################################################
# Fix TensorFlow macOS-specific issues
################################################################################

note_build_stage "Apply macOS patches to TensorFlow"

# Fix 'realpath --relative-to' usage in TF genrules (GNU-only flag).
# macOS BSD realpath doesn't support --relative-to. We replace with grealpath
# from coreutils, or use Python as fallback.
GREALPATH="$(command -v grealpath || true)"
if [[ -n "${GREALPATH}" ]]; then
  echo "Using GNU realpath (grealpath) from coreutils"
  # Find and patch any TF files that use 'realpath --relative-to'
  find ../tensorflow -name "*.bzl" -o -name "*.py" -o -name "Makefile" -o -name "*.sh" | \
    xargs grep -l 'realpath --relative-to' 2>/dev/null | while read -r file; do
    echo "Patching realpath in: ${file}"
    gsed -i 's|realpath --relative-to|grealpath --relative-to|g' "${file}"
  done
else
  echo "WARNING: grealpath not found. Install coreutils: brew install coreutils"
fi

note_build_stage "Set pyparsing for CLIF."
export PATH="$HOME/.local/bin":$PATH
pip3 uninstall -y pyparsing 2>/dev/null || true
pip3 install -q -Iv 'pyparsing==2.2.2'

note_build_stage "build-prereq-macos.sh complete"

echo ""
echo "=========================================="
echo "macOS ARM64 build prerequisites installed."
echo "  - Bazel ${DV_BAZEL_VERSION}"
echo "  - TensorFlow ${DV_CPP_TENSORFLOW_TAG} (source)"
echo "  - Abseil C++ (${ABSL_PIN})"
echo "  - CLIF C++ runtime"
echo "=========================================="
