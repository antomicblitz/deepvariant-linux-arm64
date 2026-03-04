#!/bin/bash
set -euo pipefail

# Copyright 2017 Google LLC.
# Modifications Copyright 2024 deepvariant-linux-arm64 contributors.
#
# ARM64-specific build prerequisites for DeepVariant.
# Based on upstream build-prereq.sh with x86 references replaced.

echo ========== This script is maintained for Ubuntu 22.04 on ARM64.
echo ========== Load config settings.

source settings_arm64.sh

ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "ERROR: This script must be run on an aarch64 system. Detected: ${ARCH}"
  exit 1
fi

################################################################################
# Misc. setup
################################################################################

note_build_stage "Install the runtime packages"

./run-prereq.sh

note_build_stage "Update package list"

sudo -H apt-get -qq -y update

note_build_stage "build-prereq-arm64.sh: Install development packages"

wait_for_dpkg_lock
sudo -H NEEDRESTART_MODE=a apt-get -qq -y install pkg-config zip g++ zlib1g-dev unzip curl git wget > /dev/null

# ARM64-specific: install Boost from system packages (not Homebrew)
sudo -H apt-get -qq -y install libboost-dev libboost-graph-dev libboost-system-dev libboost-filesystem-dev libboost-math-dev > /dev/null

################################################################################
# bazel
################################################################################

note_build_stage "Install bazel for ARM64"

function ensure_wanted_bazel_version {
  local wanted_bazel_version=$1
  rm -rf ~/bazel
  mkdir ~/bazel

  if
    v=$(bazel --bazelrc=/dev/null --ignore_all_rc_files version) &&
    echo "$v" | awk -v b="$wanted_bazel_version" '/Build label/ { exit ($3 != b)}'
  then
    echo "Bazel ${wanted_bazel_version} already installed on the machine, not reinstalling"
  else
    pushd ~/bazel
    # Use ARM64 Bazel binary instead of x86
    curl -L -O https://github.com/bazelbuild/bazel/releases/download/"${wanted_bazel_version}"/bazel-"${wanted_bazel_version}"-linux-arm64
    chmod +x bazel-"${wanted_bazel_version}"-linux-arm64
    mkdir -p "${HOME}/bin"
    mv bazel-"${wanted_bazel_version}"-linux-arm64 "${HOME}/bin/bazel"
    popd
  fi
}

ensure_wanted_bazel_version "${DV_BAZEL_VERSION}"

# Build abseil for examples_from_stream.so
time sudo ./tools/build_absl.sh

################################################################################
# TensorFlow
################################################################################

note_build_stage "Download and configure TensorFlow sources"

DV_DIR=$(pwd)

if [[ ! -d ../tensorflow ]]; then
  note_build_stage "Cloning TensorFlow from github as ../tensorflow doesn't exist"
  (cd .. && git clone https://github.com/tensorflow/tensorflow)
fi

(cd ../tensorflow &&
 git checkout "${DV_CPP_TENSORFLOW_TAG}" &&
 echo | ./configure)

# Use newer absl version (same as upstream)
wget https://raw.githubusercontent.com/tensorflow/tensorflow/r2.13/third_party/absl/workspace.bzl -O ../tensorflow/third_party/absl/workspace.bzl
rm -f ../tensorflow/third_party/absl/absl_designated_initializers.patch
sed -i -e 's|b971ac5250ea8de900eae9f95e06548d14cd95fe|29bf8085f3bf17b84d30e34b3d7ff8248fda404e|g' ../tensorflow/third_party/absl/workspace.bzl
sed -i -e 's|8eeec9382fc0338ef5c60053f3a4b0e0708361375fe51c9e65d0ce46ccfe55a7|affb64f374b16877e47009df966d0a9403dbf7fe613fe1f18e49802c84f6421e|g' ../tensorflow/third_party/absl/workspace.bzl
sed -i -e 's|patch_file = \["//third_party/absl:absl_designated_initializers.patch"\],||g' ../tensorflow/third_party/absl/workspace.bzl

# Update tensorflow.bzl (same as upstream)
patch ../tensorflow/tensorflow/tensorflow.bzl "${DV_DIR}"/third_party/tensorflow.bzl.patch

# Update pybind11 (same as upstream)
sed -i -e 's|v2.10.0.tar.gz|a7b91e33269ab6f3f90167291af2c4179fc878f5.zip|g' ../tensorflow/tensorflow/workspace2.bzl
sed -i -e 's|eacf582fa8f696227988d08cfc46121770823839fe9e301a20fbce67e7cd70ec|09d2ab67e91457c966eb335b361bdc4d27ece2d4dea681d22e5d8307e0e0c023|g' ../tensorflow/tensorflow/workspace2.bzl
sed -i -e 's|pybind11-2.10.0|pybind11-a7b91e33269ab6f3f90167291af2c4179fc878f5|g' ../tensorflow/tensorflow/workspace2.bzl

note_build_stage "Set pyparsing to 2.2.2 for CLIF."
export PATH="$HOME/.local/bin":$PATH
pip3 uninstall -y pyparsing && pip3 install -Iv 'pyparsing==2.2.2'

note_build_stage "build-prereq-arm64.sh complete"
