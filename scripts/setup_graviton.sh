#!/bin/bash
set -euo pipefail

# Setup script for ARM64 cloud instances (Graviton, Ampere Altra, etc.)
# Run this on a fresh Ubuntu 22.04/24.04 ARM64 instance before building.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-linux-arm64/main/scripts/setup_graviton.sh | bash

echo "========== DeepVariant ARM64 Instance Setup =========="

ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "ERROR: This script must be run on an aarch64 system. Detected: ${ARCH}"
  exit 1
fi

echo "========== Detected ARM64 architecture: ${ARCH}"

# System updates
echo "========== Updating system packages"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -qq -y

# Essential build tools
echo "========== Installing build essentials"
sudo apt-get install -qq -y \
  build-essential \
  pkg-config \
  zip \
  unzip \
  curl \
  wget \
  git \
  zlib1g-dev \
  libssl-dev \
  libcurl4-openssl-dev \
  liblz-dev \
  libbz2-dev \
  liblzma-dev \
  libboost-dev \
  libboost-graph-dev \
  python3 \
  python3-pip \
  python3-dev \
  python3-distutils

# Docker
echo "========== Installing Docker"
if ! command -v docker &> /dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "${USER}" || true
  echo "NOTE: Log out and back in for Docker group membership to take effect."
else
  echo "Docker already installed: $(docker --version)"
fi

# Verify Docker works on ARM64
echo "========== Verifying Docker ARM64 support"
sudo docker run --rm arm64v8/ubuntu:22.04 uname -m || echo "WARNING: Docker ARM64 test failed"

# CPU feature detection
echo "========== CPU Feature Detection"
echo "CPU model: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo 'N/A (check /proc/cpuinfo)')"
echo "CPU cores: $(nproc)"
echo "CPU flags: $(grep -m1 'Features' /proc/cpuinfo | cut -d: -f2 | tr ' ' '\n' | grep -E 'bf16|sve|neon|asimd|fp16' | tr '\n' ' ')"

if grep -q bf16 /proc/cpuinfo; then
  echo "BF16 SUPPORTED — OneDNN+ACL will use BF16 fast math for TF inference"
else
  echo "BF16 NOT supported — will use FP32 inference (still fast with OneDNN+ACL)"
fi

if grep -q sve /proc/cpuinfo; then
  echo "SVE SUPPORTED — Graviton3+ detected"
fi

# Memory and disk
echo "========== System Resources"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Disk: $(df -h / | awk 'NR==2 {print $4}') available"

# Recommended TF environment variables
echo "========== Writing ARM64 TF environment to /etc/profile.d/deepvariant-arm64.sh"
sudo tee /etc/profile.d/deepvariant-arm64.sh > /dev/null << 'ENVEOF'
# DeepVariant ARM64 TensorFlow optimizations
export TF_ENABLE_ONEDNN_OPTS=1
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=false
export OMP_PLACES=cores

# Enable BF16 on Graviton3+ (Neoverse V1/V2)
if grep -q bf16 /proc/cpuinfo 2>/dev/null; then
  export DNNL_DEFAULT_FPMATH_MODE=BF16
fi
ENVEOF

echo ""
echo "========== Setup complete! =========="
echo ""
echo "Next steps:"
echo "  1. Log out and back in (for Docker group)"
echo "  2. Clone the repo:  git clone https://github.com/antomicblitz/deepvariant-linux-arm64.git"
echo "  3. Build:           cd deepvariant-linux-arm64 && docker build -f docker/Dockerfile.arm64 -t deepvariant-arm64 ."
echo ""
