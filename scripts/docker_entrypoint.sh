#!/bin/bash
# Docker entrypoint for DeepVariant ARM64.
# Detects CPU features and sets optimal backend/threading automatically.
# All variables can be overridden via docker run -e VAR=value.

set -e

# --- CPU feature detection ---
_cpuinfo=""
[[ -f /proc/cpuinfo ]] && _cpuinfo=$(cat /proc/cpuinfo 2>/dev/null || true)

_has_bf16=false
echo "$_cpuinfo" | grep -qw "bf16" 2>/dev/null && _has_bf16=true

_is_ampereone=false
if [[ -f /proc/cpuinfo ]]; then
  _impl=$(grep -m1 "CPU implementer" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || true)
  _part=$(grep -m1 "CPU part" /proc/cpuinfo 2>/dev/null | awk '{print $NF}' || true)
  [[ "${_impl}" == "0xc0" && "${_part}" == "0xac3" ]] && _is_ampereone=true
fi

# --- OneDNN configuration ---
# OneDNN+ACL is only beneficial when BF16 BFMMLA is available (Graviton3+).
# Without BF16, ACL FP32 GEMM adds 29% make_examples overhead vs Eigen on N1.
# AmpereOne: ACL (compiled for Neoverse-N1) triggers SIGILL. See docs/onednn-ampereone.md.
if [[ -z "${TF_ENABLE_ONEDNN_OPTS+x}" ]]; then
  # User did not set — apply auto-detection.
  if [[ "${_is_ampereone}" == "true" ]]; then
    export TF_ENABLE_ONEDNN_OPTS=0
    echo "deepvariant: AmpereOne detected — OneDNN OFF (ACL SIGILL, see docs/onednn-ampereone.md)" >&2
  elif [[ "${_has_bf16}" == "true" ]]; then
    export TF_ENABLE_ONEDNN_OPTS=1
    echo "deepvariant: BF16 detected — OneDNN ON (BFMMLA acceleration)" >&2
  else
    export TF_ENABLE_ONEDNN_OPTS=0
    echo "deepvariant: no BF16 — OneDNN OFF (Eigen fallback, avoids 29% ME overhead)" >&2
  fi
else
  # User explicitly set the variable. Warn if it's a known-bad combination.
  if [[ "${_is_ampereone}" == "true" && "${TF_ENABLE_ONEDNN_OPTS}" == "1" ]]; then
    echo "deepvariant: WARNING: TF_ENABLE_ONEDNN_OPTS=1 on AmpereOne will SIGILL. Forcing OFF." >&2
    export TF_ENABLE_ONEDNN_OPTS=0
  elif [[ "${_has_bf16}" == "false" && "${TF_ENABLE_ONEDNN_OPTS}" == "1" ]]; then
    echo "deepvariant: WARNING: TF_ENABLE_ONEDNN_OPTS=1 without BF16 adds 29% ME overhead. Consider setting to 0." >&2
  fi
fi

# BF16 fast math mode (Graviton3+, Neoverse V1/V2)
if [[ -z "${ONEDNN_DEFAULT_FPMATH_MODE:-}" && "${_has_bf16}" == "true" && "${TF_ENABLE_ONEDNN_OPTS}" == "1" ]]; then
  export ONEDNN_DEFAULT_FPMATH_MODE=BF16
  # Also set the legacy env var name for older OneDNN versions
  export DNNL_DEFAULT_FPMATH_MODE=BF16
fi

# --- ONNX INT8 auto-selection ---
# On non-BF16 platforms, ONNX INT8 is 2.3x faster than FP32.
# On BF16 platforms with enough RAM, TF+OneDNN BF16 is optimal.
# TF SavedModel uses ~26 GB RSS; forking postprocess can push past 32 GB → OOM.
# Fall back to ONNX INT8 (~3 GB RSS) on BF16 platforms with <48 GB available.
_ram_gb=$(awk '/MemTotal/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)

if [[ -z "${DV_USE_ONNX+x}" ]]; then
  if [[ "${_has_bf16}" == "true" && "${TF_ENABLE_ONEDNN_OPTS}" == "1" && "${_ram_gb}" -ge 48 ]]; then
    export DV_USE_ONNX=0
  elif [[ "${_has_bf16}" == "true" && "${TF_ENABLE_ONEDNN_OPTS}" == "1" && "${_ram_gb}" -lt 48 ]]; then
    if [[ -f /opt/models/wgs/model_int8_static.onnx ]]; then
      export DV_USE_ONNX=1
      echo "deepvariant: BF16 available but only ${_ram_gb} GB RAM — using INT8 ONNX (TF BF16 needs >=48 GB)" >&2
    else
      export DV_USE_ONNX=0
      echo "deepvariant: WARNING: BF16 with ${_ram_gb} GB RAM may OOM. Use --memory=48g+ or mount INT8 model." >&2
    fi
  elif [[ -f /opt/models/wgs/model_int8_static.onnx ]]; then
    export DV_USE_ONNX=1
    echo "deepvariant: INT8 ONNX auto-selected (non-BF16 platform, 2.3x over FP32)" >&2
  else
    export DV_USE_ONNX=0
  fi
fi

# --- Threading configuration ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$(nproc)}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"

# TF threading: all cores for intra-op (GEMM), single thread for inter-op
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-1}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-$(nproc)}"

# --- jemalloc ---
# Reduces malloc contention under many concurrent shards.
# Verified ~14-17% ME improvement on Graviton3 and AmpereOne.
# Enable: docker run -e DV_USE_JEMALLOC=1 ...
if [[ "${DV_USE_JEMALLOC:-0}" == "1" ]]; then
  _JEMALLOC="${DV_JEMALLOC_PATH:-/usr/lib/aarch64-linux-gnu/libjemalloc.so.2}"
  if [[ -f "${_JEMALLOC}" ]]; then
    export LD_PRELOAD="${_JEMALLOC}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "deepvariant: jemalloc enabled (${_JEMALLOC})" >&2
  else
    echo "deepvariant: WARNING: DV_USE_JEMALLOC=1 but ${_JEMALLOC} not found" >&2
  fi
fi

# --- Optional full autoconfig banner ---
# Enable: docker run -e DV_AUTOCONFIG=1 ...
if [[ "${DV_AUTOCONFIG:-0}" == "1" ]]; then
  _AUTOCONFIG_SCRIPT="/opt/deepvariant/scripts/autoconfig.sh"
  if [[ -x "${_AUTOCONFIG_SCRIPT}" ]]; then
    _ac_json=$("${_AUTOCONFIG_SCRIPT}" --json 2>/dev/null || true)
    if [[ -n "${_ac_json}" ]]; then
      while IFS='=' read -r _ac_key _ac_val; do
        [[ -z "${_ac_key}" ]] && continue
        if [[ -z "${!_ac_key+x}" ]]; then
          export "${_ac_key}=${_ac_val}"
          echo "deepvariant: autoconfig set ${_ac_key}=${_ac_val}" >&2
        fi
      done < <(echo "${_ac_json}" | python3 -c "
import sys, json
try:
    env = json.load(sys.stdin).get('env', {})
    for k, v in env.items():
        if v is not None:
            print(f'{k}={v}')
except Exception:
    pass
" 2>/dev/null || true)
      _ac_cpu=$(echo "${_ac_json}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('cpu_family',''))" 2>/dev/null || true)
      _ac_backend=$(echo "${_ac_json}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('backend',''))" 2>/dev/null || true)
      echo "deepvariant: autoconfig applied (${_ac_backend} on ${_ac_cpu})" >&2
    fi
  else
    echo "deepvariant: WARNING: DV_AUTOCONFIG=1 but autoconfig.sh not found" >&2
  fi
fi

exec "$@"
