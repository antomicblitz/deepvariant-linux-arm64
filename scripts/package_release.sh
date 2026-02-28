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

# Packages pre-built DeepVariant binaries into a release tarball and
# optionally creates a GitHub Release.
#
# Usage:
#   ./scripts/package_release.sh           # create tarball only
#   ./scripts/package_release.sh --release # create tarball + GitHub Release

set -euo pipefail

VERSION="1.9.0"
RELEASE_NAME="deepvariant-${VERSION}-macos-arm64"
STAGING_DIR="/tmp/${RELEASE_NAME}"
TARBALL="/tmp/${RELEASE_NAME}.tar.gz"

echo "=== Packaging DeepVariant v${VERSION} for macOS ARM64 ==="

# Verify we're in the repo root
if [[ ! -f "build_release_binaries.sh" ]]; then
  echo "ERROR: Run this script from the DeepVariant repo root."
  exit 1
fi

# Verify binaries exist
if [[ ! -f "bazel-bin/deepvariant/call_variants" ]]; then
  echo "ERROR: Binaries not found. Run build_release_binaries.sh first."
  exit 1
fi

# Clean staging area
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}/bin/deeptrio"
mkdir -p "${STAGING_DIR}/bin/labeler"
mkdir -p "${STAGING_DIR}/scripts"

echo "Copying DeepVariant binaries..."

# DeepVariant Python zip binaries (self-executing + .zip companion)
DV_BINS=(
  call_variants
  make_examples
  make_examples_pangenome_aware_dv
  make_examples_somatic
  postprocess_variants
  vcf_stats_report
  show_examples
  runtime_by_region_vis
  convert_to_saved_model
  train
  load_gbz_into_shared_memory
  multisample_make_examples
)

for bin in "${DV_BINS[@]}"; do
  if [[ -f "bazel-bin/deepvariant/${bin}" ]]; then
    cp "bazel-bin/deepvariant/${bin}" "${STAGING_DIR}/bin/"
    cp "bazel-bin/deepvariant/${bin}.zip" "${STAGING_DIR}/bin/"
    echo "  + ${bin}"
  else
    echo "  WARNING: bazel-bin/deepvariant/${bin} not found, skipping."
  fi
done

# fast_pipeline (native C++ binary, no .zip companion)
if [[ -f "bazel-bin/deepvariant/fast_pipeline" ]]; then
  cp "bazel-bin/deepvariant/fast_pipeline" "${STAGING_DIR}/bin/"
  echo "  + fast_pipeline (native)"
fi

# DeepTrio make_examples
if [[ -f "bazel-bin/deeptrio/make_examples" ]]; then
  cp "bazel-bin/deeptrio/make_examples" "${STAGING_DIR}/bin/deeptrio/"
  cp "bazel-bin/deeptrio/make_examples.zip" "${STAGING_DIR}/bin/deeptrio/"
  echo "  + deeptrio/make_examples"
fi

# Labeler
if [[ -f "bazel-bin/deepvariant/labeler/labeled_examples_to_vcf" ]]; then
  cp "bazel-bin/deepvariant/labeler/labeled_examples_to_vcf" "${STAGING_DIR}/bin/labeler/"
  cp "bazel-bin/deepvariant/labeler/labeled_examples_to_vcf.zip" "${STAGING_DIR}/bin/labeler/"
  echo "  + labeler/labeled_examples_to_vcf"
fi

echo "Copying runner scripts..."
cp scripts/run_deepvariant.py "${STAGING_DIR}/scripts/"
cp scripts/run_deeptrio.py "${STAGING_DIR}/scripts/"

echo "Copying licenses..."
if [[ -f "bazel-bin/licenses.zip" ]]; then
  cp "bazel-bin/licenses.zip" "${STAGING_DIR}/"
fi

# Create tarball
echo "Creating tarball..."
cd /tmp
tar -czf "${RELEASE_NAME}.tar.gz" "${RELEASE_NAME}/"
cd -

TARBALL_SIZE=$(du -h "${TARBALL}" | cut -f1)
echo ""
echo "=== Release tarball created ==="
echo "  Path: ${TARBALL}"
echo "  Size: ${TARBALL_SIZE}"

# Count binaries
BIN_COUNT=$(find "${STAGING_DIR}/bin" -type f ! -name '*.zip' | wc -l | tr -d ' ')
echo "  Binaries: ${BIN_COUNT}"

# Create GitHub Release if --release flag is passed
if [[ "${1:-}" == "--release" ]]; then
  echo ""
  echo "=== Creating GitHub Release ==="

  if ! command -v gh &>/dev/null; then
    echo "ERROR: GitHub CLI (gh) not found. Install with: brew install gh"
    exit 1
  fi

  NOTES_FILE=$(mktemp)
  cat > "${NOTES_FILE}" << 'NOTES'
Pre-built native binaries for Apple Silicon (M1/M2/M3/M4).

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/antomicblitz/deepvariant-macos-arm64-metal/r1.9/install.sh | bash
```

## What's Included

- 14 self-executing Python zip binaries (make_examples, call_variants, postprocess_variants, etc.)
- 1 native C++ binary (fast_pipeline)
- Runner scripts (run_deepvariant.py, run_deeptrio.py)

## Requirements

- macOS on Apple Silicon (M1/M2/M3/M4)
- Python 3.10
- Models are downloaded separately during installation from Google Cloud Storage

## Metal GPU

The install script sets up tensorflow-macos + tensorflow-metal for GPU-accelerated variant calling.
NOTES

  gh release create "v${VERSION}" \
    "${TARBALL}" \
    --repo antomicblitz/deepvariant-macos-arm64-metal \
    --title "DeepVariant v${VERSION} for macOS ARM64" \
    --notes-file "${NOTES_FILE}"

  rm -f "${NOTES_FILE}"

  echo ""
  echo "GitHub Release created: https://github.com/antomicblitz/deepvariant-macos-arm64-metal/releases/tag/v${VERSION}"
else
  echo ""
  echo "To create a GitHub Release, run:"
  echo "  ./scripts/package_release.sh --release"
fi
