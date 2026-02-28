#!/bin/bash
set -eu -o pipefail

# DeepVariant bioconda build script
# Handles both Linux x86_64 (existing behavior) and macOS ARM64 (pre-built binaries)

SHAREDIR="share/${PKG_NAME}-${PKG_VERSION}-${PKG_BUILDNUM}"
TGT="${PREFIX}/${SHAREDIR}"
mkdir -p "${TGT}" "${PREFIX}/bin"

OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  ############################################################################
  # macOS ARM64 — install pre-built native binaries from release tarball
  ############################################################################

  echo "Installing DeepVariant for macOS ARM64 (Apple Silicon)..."

  # The source tarball extracts to a directory with bin/ and scripts/
  LIBEXEC="${PREFIX}/libexec/deepvariant"
  mkdir -p "${LIBEXEC}"

  # Install pre-built binaries
  cp -R "${SRC_DIR}/bin" "${LIBEXEC}/"
  chmod +x "${LIBEXEC}"/bin/*

  # Install runner scripts
  if [[ -d "${SRC_DIR}/scripts" ]]; then
    cp -R "${SRC_DIR}/scripts" "${LIBEXEC}/"
    chmod +x "${LIBEXEC}"/scripts/* 2>/dev/null || true
  fi

  # Install pip packages that are not available via conda
  # tensorflow-metal provides ~4.25x speedup for call_variants via Metal GPU
  ${PYTHON} -m pip install --no-deps --no-build-isolation "tensorflow-metal==1.0.0"

  # Install tensorflow packages with --no-deps to avoid pulling in regular
  # 'tensorflow' which conflicts with tensorflow-macos
  ${PYTHON} -m pip install --no-deps --no-build-isolation "tensorflow-hub==0.14.0"
  ${PYTHON} -m pip install --no-deps --no-build-isolation "tensorflow-model-optimization==0.7.5"
  ${PYTHON} -m pip install --no-deps --no-build-isolation "tf-models-official==2.13.1"

  # tf-models-official runtime deps not in conda
  ${PYTHON} -m pip install --no-deps --no-build-isolation tf-slim

  # Additional pip-only DeepVariant deps
  ${PYTHON} -m pip install --no-deps --no-build-isolation "ml_collections"
  ${PYTHON} -m pip install --no-deps --no-build-isolation "clu==0.0.9"
  ${PYTHON} -m pip install --no-deps --no-build-isolation "etils"

  # Create wrapper scripts that use the conda environment's Python
  DV_BINS="make_examples call_variants postprocess_variants vcf_stats_report show_examples"
  for name in $DV_BINS; do
    if [[ -f "${LIBEXEC}/bin/${name}" ]]; then
      cat > "${PREFIX}/bin/${name}" << WRAPPER
#!/bin/bash
exec python3 "${LIBEXEC}/bin/${name}" "\$@"
WRAPPER
      chmod +x "${PREFIX}/bin/${name}"
    fi
  done

  # fast_pipeline is a native C++ binary — symlink directly
  if [[ -f "${LIBEXEC}/bin/fast_pipeline" ]]; then
    ln -sf "${LIBEXEC}/bin/fast_pipeline" "${PREFIX}/bin/fast_pipeline"
  fi

  # run_deepvariant pipeline wrapper
  if [[ -f "${LIBEXEC}/scripts/run_deepvariant.py" ]]; then
    cat > "${PREFIX}/bin/run_deepvariant" << WRAPPER
#!/bin/bash
exec python3 "${LIBEXEC}/scripts/run_deepvariant.py" "\$@"
WRAPPER
    chmod +x "${PREFIX}/bin/run_deepvariant"
  fi

  # run_deeptrio pipeline wrapper
  if [[ -f "${LIBEXEC}/scripts/run_deeptrio.py" ]]; then
    cat > "${PREFIX}/bin/run_deeptrio" << WRAPPER
#!/bin/bash
exec python3 "${LIBEXEC}/scripts/run_deeptrio.py" "\$@"
WRAPPER
    chmod +x "${PREFIX}/bin/run_deeptrio"
  fi

  # Model download helper
  if [[ -f "${LIBEXEC}/scripts/deepvariant-download-model" ]]; then
    ln -sf "${LIBEXEC}/scripts/deepvariant-download-model" "${PREFIX}/bin/deepvariant-download-model"
  fi

  echo "macOS ARM64 installation complete."
  echo "Metal GPU acceleration enabled (tensorflow-metal)."

else
  ############################################################################
  # Linux x86_64 — existing behavior (wrapper scripts + pip packages)
  ############################################################################

  echo "Installing DeepVariant for Linux x86_64..."

  # Install tf-slim via pip (conda-forge's tf-slim is a different package)
  ${PYTHON} -m pip install --no-deps --no-build-isolation --no-cache-dir \
    "git+https://github.com/google-research/tf-slim.git"

  # Copy wrapper scripts
  install -v -m 0755 ${RECIPE_DIR}/dv_make_examples.py "${PREFIX}/bin"
  install -v -m 0755 ${RECIPE_DIR}/dv_call_variants.py "${PREFIX}/bin"
  install -v -m 0755 ${RECIPE_DIR}/dv_postprocess_variants.py "${PREFIX}/bin"

  echo "Linux x86_64 installation complete."

fi
