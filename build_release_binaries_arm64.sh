#!/bin/bash
# Copyright 2017 Google LLC.
# Modifications Copyright 2024 deepvariant-linux-arm64 contributors.
#
# Build release binaries for ARM64.
# Based on upstream build_release_binaries.sh with k8-opt replaced by aarch64-opt.

# NOLINT
source settings_arm64.sh

set -e

ARCH=$(uname -m)
if [[ "${ARCH}" != "aarch64" ]]; then
  echo "ERROR: This script must be run on an aarch64 system. Detected: ${ARCH}"
  exit 1
fi

# Bazel's --build_python_zip replaces our carefully engineered symbolic links
# with copies.  This function puts the symbolic links back.
function fix_zip_file {
  orig_zip_file=$1

  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXXX)
  cp "${orig_zip_file}.zip" "${TMPDIR}"

  pushd "${TMPDIR}" > /dev/null
  BN=$(basename "${orig_zip_file}")
  unzip -qq "${BN}.zip"

  find "runfiles/com_google_deepvariant" -name '*.so' ! -name 'examples_from_stream.so' -exec ln --force -s --relative "runfiles/com_google_protobuf/python/google/protobuf/pyext/_message.so" {} \;

  sed -i 's/  with zipfile.ZipFile(zip_path) as zf:/  if True:/' __main__.py
  sed -i 's/  for info in zf.infolist():/  if True:/' __main__.py
  sed -i 's/  zf.extract(info, dest_dir)/  os.system("unzip -qq " + zip_path + " -d " + dest_dir)/' __main__.py
  sed -i 's/  # UNC-prefixed paths must be absolute\/normalized. See/  return/' __main__.py

  rm -f "${BN}.zip"
  ZIP_OUT="/tmp/${BN}.zip"
  rm -f "${ZIP_OUT}"
  zip -q --symlinks -r "${ZIP_OUT}" *

  SELF_ZIP="/tmp/${BN}"
  echo '#!/usr/bin/env python3' | cat - "${ZIP_OUT}" > "${SELF_ZIP}"

  popd > /dev/null
  rm -f "${orig_zip_file}"
  mv "${SELF_ZIP}" "${orig_zip_file}"
  chmod +x "${orig_zip_file}"

  rm -f "${orig_zip_file}.zip"
  mv "${ZIP_OUT}" "${orig_zip_file}.zip"
}

# Build examples_from_stream.so C++ library
TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# shellcheck disable=SC2068
g++ -std=c++14 -shared \
        deepvariant/stream_examples_kernel.cc  \
        deepvariant/stream_examples_ops.cc \
        -o deepvariant/examples_from_stream.so \
        -fPIC \
        -l:libtensorflow_framework.so.2  \
        -I. \
        ${TF_CFLAGS[@]} \
        ${TF_LFLAGS[@]} \
        -D_GLIBCXX_USE_CXX11_ABI=1 \
        --std=c++17 \
        -DEIGEN_MAX_ALIGN_BYTES=64 \
        -O2

# Build fast_pipeline
# shellcheck disable=SC2086
bazel build -c opt \
  //deepvariant:fast_pipeline

# Build main binaries (no -march=corei7 — DV_COPT_FLAGS is clean for ARM64)
# shellcheck disable=SC2086
bazel build -c opt \
  --output_filter=DONT_MATCH_ANYTHING \
  --noshow_loading_progress \
  --show_result=0 \
  ${DV_COPT_FLAGS} \
  --build_python_zip \
  :binaries

# shellcheck disable=SC2086
bazel build -c opt \
  --output_filter=DONT_MATCH_ANYTHING \
  --noshow_loading_progress \
  --show_result=0 \
  ${DV_COPT_FLAGS} \
  --build_python_zip \
  //deepvariant/labeler:labeled_examples_to_vcf

# shellcheck disable=SC2086
bazel build -c opt \
  --output_filter=DONT_MATCH_ANYTHING \
  --noshow_loading_progress \
  --show_result=0 \
  ${DV_COPT_FLAGS} \
  --build_python_zip \
  //deepvariant:convert_to_saved_model

# shellcheck disable=SC2086
bazel build -c opt \
  --output_filter=DONT_MATCH_ANYTHING \
  --noshow_loading_progress \
  --show_result=0 \
  ${DV_COPT_FLAGS} \
  --build_python_zip \
  :binaries-deeptrio

# shellcheck disable=SC2086
bazel build  -c opt \
  --output_filter=DONT_MATCH_ANYTHING \
  --noshow_loading_progress \
  --show_result=0 \
  --noshow_progress \
  ${DV_COPT_FLAGS} \
  :licenses_zip

# Fix zip files — use aarch64-opt instead of k8-opt
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/train"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/call_variants"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/load_gbz_into_shared_memory"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/make_examples"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/make_examples_pangenome_aware_dv"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/make_examples_somatic"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deeptrio/make_examples"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/postprocess_variants"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/vcf_stats_report"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/show_examples"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/runtime_by_region_vis"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/convert_to_saved_model"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/multisample_make_examples"
fix_zip_file "bazel-out/${DV_BAZEL_OUTPUT_DIR}/bin/deepvariant/labeler/labeled_examples_to_vcf"
