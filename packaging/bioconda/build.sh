#!/bin/bash
set -eu -o pipefail

OS=$(uname -s)
ARCH=$(uname -m)

if [[ "$OS" == "Darwin" && "$ARCH" == "arm64" ]]; then
  ############################################################################
  # macOS ARM64 — install pre-built native binaries from release tarball
  ############################################################################

  echo "Installing DeepVariant for macOS ARM64 (Apple Silicon)..."

  # conda-build sets MACOSX_DEPLOYMENT_TARGET=11.0, but tensorflow-macos
  # wheels are built for macOS 12.0+.  The platform tag is baked into the
  # conda Python binary, so neither env-var overrides nor --platform flags
  # work inside the conda-build sandbox.
  # Solution: the release tarball bundles pre-downloaded wheels under wheels/.
  export MACOSX_DEPLOYMENT_TARGET=12.0

  LIBEXEC="${PREFIX}/libexec/deepvariant"
  mkdir -p "${LIBEXEC}" "${PREFIX}/bin"

  # Install pre-built binaries and scripts from tarball
  cp -R "${SRC_DIR}/bin" "${LIBEXEC}/"
  chmod +x "${LIBEXEC}"/bin/*

  if [[ -d "${SRC_DIR}/scripts" ]]; then
    cp -R "${SRC_DIR}/scripts" "${LIBEXEC}/"
    chmod +x "${LIBEXEC}"/scripts/* 2>/dev/null || true
  fi

  # Install pip packages from bundled wheels (included in release tarball).
  # numpy is installed via conda (run deps) — no pip install needed.
  # tensorflow-macos + tensorflow-metal first, then remaining TF ecosystem.
  ${PYTHON} -m pip install --no-deps --no-build-isolation \
    "${SRC_DIR}"/wheels/tensorflow_macos-*.whl
  ${PYTHON} -m pip install --no-deps --no-build-isolation \
    "${SRC_DIR}"/wheels/tensorflow_metal-*.whl
  ${PYTHON} -m pip install --no-deps --no-build-isolation \
    "${SRC_DIR}"/wheels/*.whl

  # Fix tensorflow-metal rpath: the Metal plugin's dylib expects
  # _pywrap_tensorflow_internal.so at a _solib_darwin_arm64 path that
  # doesn't exist in pip installs. Create the expected symlink.
  SITE_PKGS=$(${PYTHON} -c "import site; print(site.getsitepackages()[0])")
  SOLIB_DIR="${SITE_PKGS}/_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf"
  PYWRAP="${SITE_PKGS}/tensorflow/python/_pywrap_tensorflow_internal.so"
  if [[ -f "$PYWRAP" ]]; then
    mkdir -p "$SOLIB_DIR"
    ln -sf "$PYWRAP" "$SOLIB_DIR/_pywrap_tensorflow_internal.so"
  fi

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
  # Linux x86_64 — existing upstream behavior (unchanged)
  ############################################################################

  # ## Binary install with wrappers

  SHAREDIR="share/${PKG_NAME}-${PKG_VERSION}-${PKG_BUILDNUM}"
  TGT="${PREFIX}/${SHAREDIR}"
  [ -d "${TGT}" ] || mkdir -p "${TGT}"
  [ -d "${PREFIX}/bin" ] || mkdir -p "${PREFIX}/bin"

  cd ${PREFIX}
  cd ${SRC_DIR}

  # TF slim is difficult because there is an existing tf-slim package in conda-forge
  # https://github.com/conda-forge/tf-slim-feedstock
  # which is different than the google one: https://github.com/google-research/tf-slim
  # This appears to be a temporary situation: https://github.com/google-research/tf-slim/issues/6
  # so temporarily install via pip in the build.sh to avoid conflicts
  # https://github.com/google/deepvariant/blob/4b937f03a1336d1dc6fd4c0eef727e1f83d2152a/run-prereq.sh#L109
  ${PYTHON} -m pip install --no-deps --no-build-isolation --no-cache-dir -vvv \
    "git+https://github.com/google-research/tf-slim.git"

  # Copy wrapper scripts, pointing to internal binary and model directories
  install -v -m 0755 ${RECIPE_DIR}/dv_make_examples.py "${PREFIX}/bin"
  install -v -m 0755 ${RECIPE_DIR}/dv_call_variants.py "${PREFIX}/bin"
  install -v -m 0755 ${RECIPE_DIR}/dv_postprocess_variants.py "${PREFIX}/bin"

fi
