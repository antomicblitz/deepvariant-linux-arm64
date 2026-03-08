# Copyright 2019 Google LLC.
# Modifications Copyright 2024 deepvariant-linux-arm64 contributors.
#
# Builder-only Dockerfile: compiles DeepVariant C++ binaries via Bazel.
# Must be built on a native aarch64 host (~1hr on 8-core ARM64).
#
# Build and push the base builder image:
#   docker build -f docker/Dockerfile.arm64.builder \
#     -t ghcr.io/qte77/deepvariant-linux-arm64:base-builder-v1.9.1 .
#   docker push ghcr.io/qte77/deepvariant-linux-arm64:base-builder-v1.9.1
#
# Rebuild only when: C++ source, Bazel config, TF version, or build scripts change.
# The runtime Dockerfile (Dockerfile.arm64) references this image by default.

ARG FROM_IMAGE=arm64v8/ubuntu:24.04
ARG PYTHON_VERSION=3.10
ARG DV_GPU_BUILD=0
ARG VERSION=1.9.1

FROM ${FROM_IMAGE} AS builder
LABEL maintainer="https://github.com/qte77/deepvariant-linux-arm64/issues" \
      org.opencontainers.image.source="https://github.com/qte77/deepvariant-linux-arm64"

ARG DV_GPU_BUILD
ARG PYTHON_VERSION
ENV DV_GPU_BUILD=${DV_GPU_BUILD}
ENV DV_BIN_PATH=/opt/deepvariant/bin
COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv

# Reason: uv manages Python 3.10 — no deadsnakes PPA needed
# Symlink to /usr/bin/python3 so Bazel and build scripts work unchanged
RUN uv venv /opt/venv --python 3.10 && \
    ln -sf /opt/venv/bin/python3 /usr/bin/python3 && \
    ln -sf /opt/venv/bin/python3 /usr/bin/python && \
    ln -sf /opt/venv/bin/python3 /usr/bin/python3.10
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copying DeepVariant source code
COPY . /opt/deepvariant

ARG VERSION
ENV VERSION=${VERSION}

WORKDIR /opt/deepvariant

# Bazel compilation of all C++ binaries (~1hr native ARM64, ~4hr+ under QEMU)
RUN chmod +x scripts/build/build-prereq-arm64.sh scripts/build/build_release_binaries_arm64.sh && \
    ./scripts/build/build-prereq-arm64.sh \
    && PATH="${HOME}/.local/bin:${HOME}/bin:${PATH}" uv pip install --ignore-installed cryptography cffi "httplib2<0.22" \
    && PATH="${HOME}/bin:${PATH}" ./scripts/build/build_release_binaries_arm64.sh
