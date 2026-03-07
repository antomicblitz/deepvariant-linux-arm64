#!/bin/bash
# autoconfig.sh — CPU-aware configuration advisor for DeepVariant ARM64.
#
# Inspects the current ARM64 CPU and recommends optimal backend, env vars,
# and thread counts. Enforces hard safety blocks for known failure modes
# (e.g., AmpereOne + OneDNN → SIGILL).
#
# Usage:
#   bash scripts/autoconfig.sh          # Human-readable banner on stderr
#   bash scripts/autoconfig.sh --json   # JSON config on stdout
#
# Exit codes:
#   0 — Config resolved successfully (may have warnings)
#   1 — Hard safety block triggered (e.g., AmpereOne + OneDNN=1 from user)
#   2 — Unknown CPU — safe fallback applied

set -euo pipefail

# --- Flags ---
JSON_OUTPUT=false
for arg in "$@"; do
  case "$arg" in
    --json) JSON_OUTPUT=true ;;
    --help|-h)
      echo "Usage: bash scripts/autoconfig.sh [--json]" >&2
      echo "  --json  Output machine-readable JSON to stdout" >&2
      exit 0
      ;;
  esac
done

# --- CPU Detection ---
IMPLEMENTER=""
PART=""
if [[ -f /proc/cpuinfo ]]; then
  IMPLEMENTER=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || true)
  PART=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || true)
fi

# Map implementer+part to CPU family
CPU_FAMILY="unknown-aarch64"
case "${IMPLEMENTER}/${PART}" in
  0x41/0xd0c) CPU_FAMILY="neoverse-n1" ;;
  0x41/0xd40) CPU_FAMILY="neoverse-v1-graviton3" ;;
  0x41/0xd4f) CPU_FAMILY="neoverse-v2-graviton4" ;;
  0x41/0xd49) CPU_FAMILY="neoverse-n2" ;;
  0xc0/0xac3) CPU_FAMILY="ampereone" ;;
esac

# Detect CPU flags (space-delimited to avoid partial matches)
_cpuinfo=""
[[ -f /proc/cpuinfo ]] && _cpuinfo=$(cat /proc/cpuinfo 2>/dev/null || true)
HAS_BF16=false; echo "$_cpuinfo" | grep -qw "bf16" 2>/dev/null && HAS_BF16=true
HAS_SVE=false;  echo "$_cpuinfo" | grep -qw "sve"  2>/dev/null && HAS_SVE=true
HAS_I8MM=false; echo "$_cpuinfo" | grep -qw "i8mm" 2>/dev/null && HAS_I8MM=true
HAS_ASIMD=false; echo "$_cpuinfo" | grep -qw "asimd" 2>/dev/null && HAS_ASIMD=true

# System info
VCPUS=$(nproc 2>/dev/null || echo 1)
RAM_GB=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
ORT_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)" 2>/dev/null || echo "not-installed")
TF_VERSION=$(python3 -c "import tensorflow; print(tensorflow.__version__)" 2>/dev/null || echo "not-installed")

# jemalloc detection
JEMALLOC_PATH="/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
JEMALLOC_AVAILABLE=false
[[ -f "${JEMALLOC_PATH}" ]] && JEMALLOC_AVAILABLE=true

# --- Configuration Resolution ---
BACKEND=""
TF_ONEDNN="1"
FPMATH=""
JEMALLOC_RECOMMENDED=true
RECOMMENDED_CV_THREADS="${VCPUS}"
RECOMMENDED_SHARDS="${VCPUS}"
EXIT_CODE=0
WARNINGS=()
NOTES=()

# ONNX model resolution
ONNX_MODEL=""
if [[ -f "/opt/models/wgs/model_int8_static.onnx" ]]; then
  ONNX_MODEL="/opt/models/wgs/model_int8_static.onnx"
elif [[ -f "/opt/models/wgs/model.onnx" ]]; then
  ONNX_MODEL="/opt/models/wgs/model.onnx"
  WARNINGS+=("FP32 ONNX model found — INT8 model not present. Accuracy OK, speed suboptimal.")
fi

case "${CPU_FAMILY}" in
  ampereone)
    BACKEND="onnx_int8"
    TF_ONEDNN="0"
    NOTES+=("AmpereOne: OneDNN+ACL DISABLED (SIGILL — ACL compiled for Neoverse-N1).")
    NOTES+=("INT8 ONNX is the optimal backend (\$2.32/genome vs \$2.49 TF Eigen FP32).")
    NOTES+=("See docs/onednn-ampereone.md for full diagnostic trail.")
    ;;
  neoverse-v1-graviton3|neoverse-v2-graviton4)
    if [[ "${HAS_BF16}" == "true" ]]; then
      BACKEND="tf_bf16"
      TF_ONEDNN="1"
      FPMATH="BF16"
      NOTES+=("Graviton3/4: BF16 BFMMLA active via OneDNN.")
    else
      BACKEND="onnx_int8"
      TF_ONEDNN="0"
      NOTES+=("Graviton3/4 without BF16 flag: using ONNX INT8, OneDNN OFF.")
    fi
    ;;
  neoverse-n1|neoverse-n2)
    BACKEND="onnx_int8"
    TF_ONEDNN="0"
    NOTES+=("Neoverse-N1/N2: no BF16, ONNX INT8 is 2.3x over FP32.")
    NOTES+=("OneDNN OFF: ACL FP32 adds 29% ME overhead on N1 without BF16 (332s vs 258s).")
    ;;
  *)
    BACKEND="tf_fp32"
    TF_ONEDNN="0"
    WARNINGS+=("Unknown ARM64 CPU (implementer=${IMPLEMENTER:-?} part=${PART:-?}). Using safe FP32 defaults.")
    EXIT_CODE=2
    ;;
esac

# Safety net: OneDNN is only beneficial with BF16 BFMMLA. Without BF16, ACL FP32
# adds overhead to make_examples small model (29% on N1). Force OFF when no BF16.
if [[ "${HAS_BF16}" == "false" && "${TF_ONEDNN}" == "1" ]]; then
  TF_ONEDNN="0"
  WARNINGS+=("Forced OneDNN OFF: no BF16 support detected. ACL FP32 adds ME overhead without benefit.")
fi

# Thread recommendation: INT8 ONNX CV saturates at 16 threads (verified benchmark)
if [[ "${BACKEND}" == "onnx_int8" && "${VCPUS}" -gt 16 ]]; then
  RECOMMENDED_CV_THREADS=16
fi

# Warn if no ONNX model found and backend requires it
if [[ "${BACKEND}" == "onnx_int8" && -z "${ONNX_MODEL}" ]]; then
  WARNINGS+=("No ONNX model found. Backend will fall back to TF SavedModel.")
fi

# --- Output ---
# Friendly CPU name for banner
_cpu_display=""
case "${CPU_FAMILY}" in
  neoverse-n1)             _cpu_display="Neoverse N1" ;;
  neoverse-v1-graviton3)   _cpu_display="Neoverse V1 (Graviton3)" ;;
  neoverse-v2-graviton4)   _cpu_display="Neoverse V2 (Graviton4)" ;;
  neoverse-n2)             _cpu_display="Neoverse N2" ;;
  ampereone)               _cpu_display="AmpereOne (Siryn)" ;;
  *)                       _cpu_display="Unknown ARM64" ;;
esac

_backend_display=""
case "${BACKEND}" in
  tf_bf16)   _backend_display="tf_bf16  (TF+OneDNN, BFMMLA active)" ;;
  onnx_int8) _backend_display="onnx_int8  (ONNX Runtime, INT8 static)" ;;
  tf_fp32)   _backend_display="tf_fp32  (TF Eigen, safe fallback)" ;;
esac

_onednn_display="ON"
[[ "${TF_ONEDNN}" == "0" ]] && _onednn_display="OFF"

_fpmath_display="${FPMATH:-—}"

_bf16_yn="no"; [[ "${HAS_BF16}" == "true" ]] && _bf16_yn="yes"
_sve_yn="no";  [[ "${HAS_SVE}" == "true" ]]  && _sve_yn="yes"
_i8mm_yn="no"; [[ "${HAS_I8MM}" == "true" ]] && _i8mm_yn="yes"

_jemalloc_display="not available"
if [[ "${JEMALLOC_AVAILABLE}" == "true" ]]; then
  _jemalloc_display="RECOMMENDED (enable: -e DV_USE_JEMALLOC=1)"
fi

# Banner (always stderr)
{
  echo "=== DeepVariant ARM64 autoconfig ==="
  printf "CPU:       %-30s [implementer=%s part=%s]\n" "${_cpu_display}" "${IMPLEMENTER:-?}" "${PART:-?}"
  printf "vCPUs:     %-5s |  RAM: %s GB\n" "${VCPUS}" "${RAM_GB}"
  printf "Features:  BF16=%-4s SVE=%-4s I8MM=%-4s\n" "${_bf16_yn}" "${_sve_yn}" "${_i8mm_yn}"
  printf "Backend:   %s\n" "${_backend_display}"
  printf "OneDNN:    %-4s |  FPMath: %s\n" "${_onednn_display}" "${_fpmath_display}"
  printf "jemalloc:  %s\n" "${_jemalloc_display}"
  printf "CV threads: %-4s |  Shards: %s\n" "${RECOMMENDED_CV_THREADS}" "${RECOMMENDED_SHARDS}"
  for w in "${WARNINGS[@]+"${WARNINGS[@]}"}"; do
    [[ -n "$w" ]] && echo "WARNING: $w"
  done
  for n in "${NOTES[@]+"${NOTES[@]}"}"; do
    [[ -n "$n" ]] && echo "NOTE: $n"
  done
  echo "===================================="
} >&2

# JSON output (stdout, only with --json)
if [[ "${JSON_OUTPUT}" == "true" ]]; then
  # Build warnings JSON array
  _warnings_json="[]"
  if [[ ${#WARNINGS[@]} -gt 0 ]]; then
    _warnings_json=$(printf '%s\n' "${WARNINGS[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
  fi
  _notes_json="[]"
  if [[ ${#NOTES[@]} -gt 0 ]]; then
    _notes_json=$(printf '%s\n' "${NOTES[@]}" | python3 -c "import sys,json; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))" 2>/dev/null || echo "[]")
  fi

  _onnx_model_json="null"
  [[ -n "${ONNX_MODEL}" ]] && _onnx_model_json="\"${ONNX_MODEL}\""

  _fpmath_json="null"
  [[ -n "${FPMATH}" ]] && _fpmath_json="\"${FPMATH}\""

  cat <<ENDJSON
{
  "cpu_family": "${CPU_FAMILY}",
  "implementer": "${IMPLEMENTER:-}",
  "part": "${PART:-}",
  "vcpus": ${VCPUS},
  "ram_gb": ${RAM_GB},
  "features": {"bf16": ${HAS_BF16}, "sve": ${HAS_SVE}, "i8mm": ${HAS_I8MM}, "asimd": ${HAS_ASIMD}},
  "backend": "${BACKEND}",
  "env": {
    "TF_ENABLE_ONEDNN_OPTS": "${TF_ONEDNN}",
    "ONEDNN_DEFAULT_FPMATH_MODE": ${_fpmath_json},
    "OMP_NUM_THREADS": "${VCPUS}",
    "OMP_PROC_BIND": "false",
    "OMP_PLACES": "cores"
  },
  "onnx_model": ${_onnx_model_json},
  "recommended_cv_threads": ${RECOMMENDED_CV_THREADS},
  "recommended_shards": ${RECOMMENDED_SHARDS},
  "jemalloc_recommended": ${JEMALLOC_RECOMMENDED},
  "jemalloc_available": ${JEMALLOC_AVAILABLE},
  "jemalloc_path": "${JEMALLOC_PATH}",
  "ort_version": "${ORT_VERSION}",
  "tf_version": "${TF_VERSION}",
  "warnings": ${_warnings_json},
  "notes": ${_notes_json}
}
ENDJSON
fi

exit "${EXIT_CODE}"
