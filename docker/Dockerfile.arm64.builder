# Copyright 2019 Google LLC.
# Modifications Copyright 2024 deepvariant-linux-arm64 contributors.
#
# Builder-only Dockerfile: compiles DeepVariant C++ binaries via Bazel.
# Must be built on a native aarch64 host (~1hr on 8-core ARM64).
#
# Build and push the base builder image:
#   docker build -f docker/Dockerfile.arm64.builder \
#     -t ghcr.io/qte77/deepvariant-linux-arm64:base-builder-v1.9.0 .
#   docker push ghcr.io/qte77/deepvariant-linux-arm64:base-builder-v1.9.0
#
# Rebuild only when: C++ source, Bazel config, TF version, or build scripts change.
# The runtime Dockerfile (Dockerfile.arm64) references this image by default.

ARG FROM_IMAGE=arm64v8/ubuntu:24.04
ARG PYTHON_VERSION=3.10
ARG DV_GPU_BUILD=0
ARG VERSION=1.9.0

FROM ${FROM_IMAGE} AS builder
LABEL maintainer="https://github.com/antomicblitz/deepvariant-linux-arm64/issues"

ARG DV_GPU_BUILD
ARG PYTHON_VERSION
ENV DV_GPU_BUILD=${DV_GPU_BUILD}
ENV DV_BIN_PATH=/opt/deepvariant/bin
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Install Python 3.10 from deadsnakes PPA (Ubuntu 24.04 ships 3.12)
RUN apt-get update -qq && \
    apt-get install -qq -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update -qq && \
    apt-get install -qq -y python3.10 python3.10-dev python3.10-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 10 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copying DeepVariant source code
COPY . /opt/deepvariant

ARG VERSION
ENV VERSION=${VERSION}

WORKDIR /opt/deepvariant

# Bazel compilation of all C++ binaries (~1hr native ARM64, ~4hr+ under QEMU)
RUN chmod +x scripts/build/build-prereq-arm64.sh scripts/build/build_release_binaries_arm64.sh && \
    ./scripts/build/build-prereq-arm64.sh \
    && PATH="${HOME}/.local/bin:${HOME}/bin:${PATH}" pip3 install --ignore-installed cryptography cffi "httplib2<0.22" \
    && PATH="${HOME}/bin:${PATH}" ./scripts/build/build_release_binaries_arm64.sh
