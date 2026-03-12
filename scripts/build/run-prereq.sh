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

# This installs all the libraries (python, dso, etc) that are needed
# by DeepVariant at runtime (except for tensorflow, which is special).
# Some extra stuff may also be included.

set -euo pipefail

echo ========== This script is only maintained for Ubuntu 22.04.
echo ========== Load config settings.

source "$(dirname "$0")/settings.sh"

################################################################################
# misc setup
################################################################################

note_build_stage "Misc setup"

APT_ARGS=(
"-qq"
"-y"
)

if [[ "$EUID" = "0" ]]; then
  # Just in case:
  # https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
  rm -f /etc/apt/sources.list.d/cuda.list
  rm -f /etc/apt/sources.list.d/nvidia-ml.list
  apt-get update "${APT_ARGS[@]}" > /dev/null
  UV_ARGS=(
    "--quiet")
else
  UV_ARGS=(
    "--quiet")
fi

note_build_stage "Update package list"

$SUDO_H apt-get update "${APT_ARGS[@]}" > /dev/null

note_build_stage "run-prereq.sh: Install development packages"

# Need to wait for dpkg lock (see internal)
wait_for_dpkg_lock

# See https://askubuntu.com/questions/909277.
$SUDO_H DEBIAN_FRONTEND=noninteractive apt-get install "${APT_ARGS[@]}" pkg-config zip zlib1g-dev unzip curl git wget > /dev/null

note_build_stage "Install python3 packaging infrastructure"

# Fix this error:
# "error: command 'x86_64-linux-gnu-gcc' failed: No such file or directory"
$SUDO_H apt-get install "${APT_ARGS[@]}" "gcc"

# uv is installed via COPY --from in Dockerfiles; verify it's available
if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found. Install via Dockerfile: COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv"
  exit 1
fi

echo "$(python3 --version)"
echo "$(uv --version)"

################################################################################
# python packages
################################################################################

note_build_stage "Install python3 packages"

uv pip install "${UV_ARGS[@]}" contextlib2
uv pip install "${UV_ARGS[@]}" etils typing_extensions importlib_resources
uv pip install "${UV_ARGS[@]}" 'enum34==1.1.8'
uv pip install "${UV_ARGS[@]}" 'sortedcontainers==2.1.0'
uv pip install "${UV_ARGS[@]}" 'intervaltree==3.1.0'
uv pip install "${UV_ARGS[@]}" 'mock>=2.0.0'
uv pip install "${UV_ARGS[@]}" ml_collections
uv pip install "${UV_ARGS[@]}" --reinstall-package PyYAML PyYAML
uv pip install "${UV_ARGS[@]}" 'clu==0.0.9'
# Note that protobuf installed with pip needs to be 3.13 because of the pyclif
# version we're using. This is currently inconsistent with C++ protobuf version
# in WORKSPACE and protobuf.BUILD, but we can't update those, because those
# files need to be consistent with what TensorFlow needs, which is currently
# still 3.9.2.
# Ideally we want to make these protobuf versions all match, eventually.
uv pip install "${UV_ARGS[@]}" 'protobuf==4.21.9'
uv pip install "${UV_ARGS[@]}" 'argparse==1.4.0'

# Reason:
# ========== [Wed Dec 11 19:57:32 UTC 2019] Stage 'Install python3 packages' starting
# ERROR: pyasn1-modules 0.2.7 has requirement pyasn1<0.5.0,>=0.4.6, but you'll have pyasn1 0.1.9 which is incompatible.
uv pip install "${UV_ARGS[@]}" 'pyasn1<0.5.0,>=0.4.6'
uv pip install "${UV_ARGS[@]}" 'requests>=2.18'
uv pip install "${UV_ARGS[@]}" --reinstall-package oauth2client 'oauth2client>=4.0.0'
uv pip install "${UV_ARGS[@]}" 'crcmod>=1.7'
uv pip install "${UV_ARGS[@]}" 'six>=1.11.0'
uv pip install "${UV_ARGS[@]}" joblib
uv pip install "${UV_ARGS[@]}" psutil
uv pip install "${UV_ARGS[@]}" --upgrade google-api-python-client
uv pip install "${UV_ARGS[@]}" 'pandas==1.3.4'
# We manually install jsonschema here to pin it to v3.2.0, since
# the latest v4.0.1 has issues with Altair v4.1.0.
# See https://github.com/altair-viz/altair/issues/2496
# If Altair version is updated below, the jsonschema version
# should also be updated accordingly.
uv pip install "${UV_ARGS[@]}" --reinstall-package jsonschema 'jsonschema==3.2.0'
uv pip install "${UV_ARGS[@]}" 'altair==4.1.0'
uv pip install "${UV_ARGS[@]}" 'Pillow==9.5.0'
uv pip install "${UV_ARGS[@]}" 'ipython==8.22.2'
uv pip install "${UV_ARGS[@]}" 'pysam==0.20.0'
uv pip install "${UV_ARGS[@]}" 'scikit-learn==1.0.2'
uv pip install "${UV_ARGS[@]}" --reinstall-package setuptools 'setuptools==61.0.0'
# This is to avoid ERROR: No matching distribution found for opencv-python-headless==4.5.2.52.
# TODO: Make this the same as ${DV_GCP_OPTIMIZED_TF_WHL_VERSION}" later
# tf-models-official depends on tensorflow-text which has no aarch64 wheel.
# Only needed for training (train.py), not inference. Install without deps on ARM64.
if [ "$(uname -m)" = "aarch64" ]; then
  uv pip install "${UV_ARGS[@]}" --no-deps "tf-models-official==2.13.1" || true
else
  uv pip install "${UV_ARGS[@]}" "tf-models-official==2.13.1"
fi

################################################################################
# TensorFlow
################################################################################

note_build_stage "Install TensorFlow pip package"

if [[ "${DV_USE_PREINSTALLED_TF}" = "1" ]]; then
  echo "Skipping TensorFlow installation at user request; will use pre-installed TensorFlow."
else
  # Also pip install the latest TensorFlow with cpu support. We don't build the
  # full TF from source, but instead using prebuilt version. However, we still
  # need the full source version to build DeepVariant.

  # Gets the nightly TF build: https://pypi.python.org/pypi/tf-nightly which is
  # necessary right now if we aren't pinning the TF source. We have observed
  # runtime failures if there's too much skew between the released TF package and
  # the source.
  if [[ "${DV_TF_NIGHTLY_BUILD}" = "1" ]]; then
    if [[ "${DV_GPU_BUILD}" = "1" ]]; then
      echo "Installing GPU-enabled TensorFlow nightly wheel"
      uv pip install "${UV_ARGS[@]}" --upgrade tf_nightly_gpu
    else
      echo "Installing CPU-only TensorFlow nightly wheel"
      uv pip install "${UV_ARGS[@]}" --upgrade tf_nightly
    fi
  else
    # Use the official TF release pip package.
    if [[ "${DV_GPU_BUILD}" = "1" ]]; then
      echo "Installing GPU-enabled TensorFlow ${DV_TENSORFLOW_STANDARD_GPU_WHL_VERSION} wheel"
      uv pip install "${UV_ARGS[@]}" --upgrade "tensorflow==${DV_TENSORFLOW_STANDARD_GPU_WHL_VERSION}"
    else
      echo "Installing CPU TensorFlow ${DV_TENSORFLOW_STANDARD_CPU_WHL_VERSION} wheel"
      uv pip install "${UV_ARGS[@]}" --upgrade "tensorflow==${DV_TENSORFLOW_STANDARD_CPU_WHL_VERSION}"
    fi
  fi
fi

# A temporary fix.
# Context: intel-tensorflow 2.7.0 will end up updating markupsafe to 2.1.1,
# which caused the issue here: https://github.com/pallets/markupsafe/issues/286.
# Specifically:
# ImportError: cannot import name 'soft_unicode' from 'markupsafe'.
# So, forcing a downgrade. This isn't the best solution, but we need it to get
# our tests pass.
uv pip install "${UV_ARGS[@]}" --reinstall-package markupsafe 'markupsafe==2.0.1'

################################################################################
# CUDA
################################################################################

note_build_stage "Install CUDA"

# See https://www.tensorflow.org/install/source#gpu for versions required.
if [[ "${DV_GPU_BUILD}" = "1" ]]; then
  if [[ "${DV_INSTALL_GPU_DRIVERS}" = "1" ]]; then
    # This script is only maintained for Ubuntu 22.04.
    echo "Checking for CUDA..."
    if ! dpkg-query -W cuda-11-8; then
      echo "Installing CUDA..."
      UBUNTU_VERSION="2204"
      curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/cuda-ubuntu${UBUNTU_VERSION}.pin
      $SUDO mv cuda-ubuntu${UBUNTU_VERSION}.pin /etc/apt/preferences.d/cuda-repository-pin-600

      curl https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor | $SUDO tee /usr/share/keyrings/nvidia-cuda-archive-keyring.gpg > /dev/null
      echo \
        "deb [signed-by=/usr/share/keyrings/nvidia-cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | \
        $SUDO tee /etc/apt/sources.list.d/cuda.list > /dev/null
      $SUDO_H NEEDRESTART_MODE=a apt-get update "${APT_ARGS[@]}"
      $SUDO_H DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get full-upgrade "${APT_ARGS[@]}"
      $SUDO_H DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install "${APT_ARGS[@]}" cuda-11-8
    fi
    echo "Checking for CUDNN..."
    if [[ ! -e /usr/local/cuda-11/include/cudnn.h ]]; then
      echo "Installing CUDNN..."
      CUDNN_TAR_FILE="cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz"
      wget -q https://developer.download.nvidia.com/compute/redist/cudnn/v8.6.0/local_installers/11.8/${CUDNN_TAR_FILE}
      tar -xvf ${CUDNN_TAR_FILE}
      $SUDO cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/cudnn.h /usr/local/cuda-11/include
      $SUDO cp -P cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/libcudnn* /usr/local/cuda-11/lib64/
      $SUDO chmod a+r /usr/local/cuda-11/lib64/libcudnn*
      $SUDO ldconfig
    fi
    # Tensorflow says to do this.
    $SUDO_H NEEDRESTART_MODE=a apt-get install "${APT_ARGS[@]}" libcupti-dev > /dev/null
  fi

  # If we are doing a gpu-build, nvidia-smi should be install. Run it so we
  # can see what gpu is installed.
  nvidia-smi || :
fi

################################################################################
# TensorRT
################################################################################

note_build_stage "Install TensorRT"

# Address the issue:
# 'dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory'
# It's unclear whether we need this or not. Setting up to get rid of the errors.
if [[ "${DV_GPU_BUILD}" = "1" ]]; then
  uv pip install "${UV_ARGS[@]}" tensorrt==8.5.3.1
  echo "For debugging:"
  uv pip show tensorrt
  TENSORRT_PATH=$(python3 -c 'import tensorrt; print(tensorrt.__path__[0])')
  $SUDO ln -sf "${TENSORRT_PATH}/libnvinfer.so.8" "${TENSORRT_PATH}/libnvinfer.so.7"
  $SUDO ln -sf "${TENSORRT_PATH}/libnvinfer_plugin.so.8" "${TENSORRT_PATH}/libnvinfer_plugin.so.7"
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}:${TENSORRT_PATH}"
  $SUDO ldconfig
  # Just in case this still doesn't work, we link them.
  # This is a workaround that we might want to get rid of, if we can make sure
  # setting LD_LIBRARY_PATH and `$SUDO ldconfig`` works.
  if [[ ! -e /usr/local/nvidia/lib ]]; then
    $SUDO mkdir -p /usr/local/nvidia/lib
    $SUDO ln -sf "${TENSORRT_PATH}/libnvinfer.so.7" /usr/local/nvidia/lib/libnvinfer.so.7
    $SUDO ln -sf "${TENSORRT_PATH}/libnvinfer_plugin.so.7" /usr/local/nvidia/lib/libnvinfer_plugin.so.7
  fi
fi

################################################################################
# Misc dependencies
################################################################################

note_build_stage "Install other packages"

# for htslib
$SUDO_H NEEDRESTART_MODE=a apt-get install "${APT_ARGS[@]}" libssl-dev libcurl4-openssl-dev liblz-dev libbz2-dev liblzma-dev > /dev/null

# for the debruijn graph
$SUDO_H NEEDRESTART_MODE=a apt-get install "${APT_ARGS[@]}" libboost-graph-dev > /dev/null

# Just being safe, downgrade load-bearing dependencies at the end if needed.
uv pip install "${UV_ARGS[@]}" 'protobuf==4.21.9'

# internal#comment9
uv pip install "${UV_ARGS[@]}" "jax==0.4.35"

note_build_stage "run-prereq.sh complete"
