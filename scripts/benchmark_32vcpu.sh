#!/bin/bash
# benchmark_32vcpu.sh — 32-vCPU scaling benchmark for DeepVariant ARM64.
#
# Tests whether BF16 TF+OneDNN scales beyond 16 threads (INT8 ONNX doesn't).
# Requires AWS vCPU quota increase for c7g.8xlarge or c8g.8xlarge instances.
#
# Usage:
#   bash scripts/benchmark_32vcpu.sh \
#     --platform graviton3 \
#     --usd-per-hr 1.15 \
#     --data-dir /data \
#     --image ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5
#
# What it runs (in order):
#   1. autoconfig.sh — verify CPU detected correctly
#   2. BF16 sequential, jemalloc OFF, N=4
#   3. BF16 sequential, jemalloc ON, N=4
#   4. INT8 ONNX sequential, jemalloc OFF, N=2 (reference comparison)
#   5. Outputs summary JSON + console table
#
# Key question: does CV rate improve at 32 threads for BF16?
#   At 16 vCPU: BF16 CV=185s (0.232 s/100)
#   Expected at 32 vCPU if scaling: ~110-130s
#   Expected at 32 vCPU if saturated: ~185s (same as 16 vCPU)

set -euo pipefail

# --- Defaults ---
PLATFORM=""
IMAGE="ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5"
DATA_DIR="/data"
USD_PER_HR=""
DOCKER_MEM="56g"
BATCH_SIZE=256
REGION="chr20"

usage() {
  cat >&2 <<'EOF'
Usage: bash scripts/benchmark_32vcpu.sh [OPTIONS]

Required:
  --platform PLATFORM    graviton3 or graviton4
  --usd-per-hr RATE     Hourly instance cost (e.g. 1.15 for c7g.8xlarge)

Optional:
  --data-dir DIR         Data directory (default: /data)
  --image IMAGE          Docker image (default: ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5)
  --docker-mem MEM       Docker memory limit (default: 56g)
  --batch-size N         call_variants batch size (default: 256)
  --help                 Show this help

Test matrix:
  1. BF16 sequential, jemalloc OFF, N=4
  2. BF16 sequential, jemalloc ON, N=4
  3. INT8 ONNX sequential, jemalloc OFF, N=2 (reference)

Platforms:
  graviton3   c7g.8xlarge (32 vCPU Neoverse V1, $1.15/hr)
  graviton4   c8g.8xlarge (32 vCPU Neoverse V2, $1.36/hr)
EOF
  exit 0
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --platform)    PLATFORM="$2"; shift 2 ;;
    --usd-per-hr)  USD_PER_HR="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --image)       IMAGE="$2"; shift 2 ;;
    --docker-mem)  DOCKER_MEM="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --help|-h)     usage ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PLATFORM}" ]]; then
  echo "ERROR: --platform required (graviton3 or graviton4)" >&2
  usage
fi

if [[ "${PLATFORM}" != "graviton3" && "${PLATFORM}" != "graviton4" ]]; then
  echo "ERROR: --platform must be graviton3 or graviton4" >&2
  exit 1
fi

if [[ -z "${USD_PER_HR}" ]]; then
  echo "ERROR: --usd-per-hr required" >&2
  exit 1
fi

NPROC=$(nproc)
NUM_SHARDS="${NPROC}"
RESULTS_DIR="${DATA_DIR}/benchmark_results/32vcpu_${PLATFORM}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# --- Machine metadata ---
CPU_MODEL=$(grep -m1 'model name\|CPU part' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo "unknown")
RAM_GB=$(awk '/^MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "unknown")
KERNEL=$(uname -r)
BF16_SUPPORT=$(grep -q bf16 /proc/cpuinfo 2>/dev/null && echo "true" || echo "false")

# --- Run autoconfig first ---
echo "============================================="
echo "  32-vCPU Scaling Benchmark (${PLATFORM})"
echo "============================================="
echo "vCPUs: ${NPROC}, RAM: ${RAM_GB} GB, CPU: ${CPU_MODEL}"
echo "BF16: ${BF16_SUPPORT}"
echo "Image: ${IMAGE}"
echo "Cost rate: \$${USD_PER_HR}/hr"
echo "Results: ${RESULTS_DIR}"
echo ""

echo "--- Running autoconfig ---"
docker run --rm "${IMAGE}" bash /opt/deepvariant/scripts/autoconfig.sh 2>&1 || true
echo ""

# --- Helper: convert time output to seconds ---
_real_to_sec() {
  local t="$1"
  if [[ "${t}" =~ ^([0-9]+)m([0-9.]+)s$ ]]; then
    python3 -c "print(round(${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}, 1))" 2>/dev/null || echo "null"
  else
    echo "null"
  fi
}

# --- Run a single benchmark ---
# Args: run_name backend jemalloc(on|off)
run_single() {
  local RUN_NAME="$1"
  local BACKEND="$2"   # "bf16" or "int8"
  local JEMALLOC="$3"  # "on" or "off"
  local OUT_DIR="${DATA_DIR}/output/${RUN_NAME}"
  local LOG="${RESULTS_DIR}/${RUN_NAME}.log"
  local CONTAINER_NAME="dv_32vcpu_${RUN_NAME}"

  echo ""
  echo ">>> [${RUN_NAME}] backend=${BACKEND} jemalloc=${JEMALLOC} ..."

  mkdir -p "${OUT_DIR}"
  # shellcheck disable=SC2046
  sudo chown -R $(id -u):$(id -g) "${OUT_DIR}" 2>/dev/null || true

  # Build env flags based on backend
  local ENV_FLAGS="-e CUDA_VISIBLE_DEVICES="
  local CV_EXTRA_ARGS="--batch_size=${BATCH_SIZE}"

  if [[ "${BACKEND}" == "bf16" ]]; then
    ENV_FLAGS="${ENV_FLAGS} -e TF_ENABLE_ONEDNN_OPTS=1"
    ENV_FLAGS="${ENV_FLAGS} -e ONEDNN_DEFAULT_FPMATH_MODE=BF16"
    ENV_FLAGS="${ENV_FLAGS} -e OMP_NUM_THREADS=${NPROC}"
    ENV_FLAGS="${ENV_FLAGS} -e OMP_PROC_BIND=false"
    ENV_FLAGS="${ENV_FLAGS} -e OMP_PLACES=cores"
  elif [[ "${BACKEND}" == "int8" ]]; then
    ENV_FLAGS="${ENV_FLAGS} -e TF_ENABLE_ONEDNN_OPTS=1"
    CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--use_onnx=true,--onnx_model=/opt/models/wgs/model_int8_static.onnx"
  fi

  if [[ "${JEMALLOC}" == "on" ]]; then
    ENV_FLAGS="${ENV_FLAGS} -e DV_USE_JEMALLOC=1"
  fi

  WALL_START=$(date +%s)

  # Run pipeline
  # shellcheck disable=SC2086
  docker run --rm \
    --name "${CONTAINER_NAME}" \
    --memory="${DOCKER_MEM}" \
    -v "${DATA_DIR}:/data" \
    ${ENV_FLAGS} \
    "${IMAGE}" \
    /opt/deepvariant/bin/run_deepvariant \
      --model_type=WGS \
      --ref=/data/reference/GRCh38_no_alt_analysis_set.fasta \
      --reads=/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
      --output_vcf="/data/output/${RUN_NAME}/output.vcf.gz" \
      --regions="${REGION}" \
      --num_shards="${NUM_SHARDS}" \
      --intermediate_results_dir="/data/output/${RUN_NAME}/intermediate" \
      --call_variants_extra_args="${CV_EXTRA_ARGS}" \
    2>&1 | tee "${LOG}"

  WALL_END=$(date +%s)
  WALL_TIME=$((WALL_END - WALL_START))

  # Extract per-stage timing
  local _reals
  mapfile -t _reals < <(grep -oP '(?<=^real\t)[0-9]+m[0-9.]+s' "${LOG}" 2>/dev/null || true)
  ME_TIME=$(_real_to_sec "${_reals[0]:-}")
  CV_TIME=$(_real_to_sec "${_reals[1]:-}")
  PP_TIME=$(_real_to_sec "${_reals[2]:-}")

  # Extract CV rate
  CV_RATE="null"
  if [[ "${CV_TIME}" != "null" ]]; then
    local last_rate
    last_rate=$(grep -oP '\[([0-9.]+) sec per 100\]' "${LOG}" 2>/dev/null | tail -1 | grep -oP '[0-9.]+' || echo "")
    if [[ -n "${last_rate}" ]]; then
      CV_RATE="${last_rate}"
    fi
  fi

  echo "  Wall: ${WALL_TIME}s  ME: ${ME_TIME}s  CV: ${CV_TIME}s  PP: ${PP_TIME}s"

  # Save per-run JSON
  cat > "${RESULTS_DIR}/${RUN_NAME}.json" <<JSONEOF
{
  "run_name": "${RUN_NAME}",
  "platform": "${PLATFORM}",
  "backend": "${BACKEND}",
  "jemalloc": "${JEMALLOC}",
  "wall_time_s": ${WALL_TIME},
  "make_examples_s": ${ME_TIME},
  "call_variants_s": ${CV_TIME},
  "postprocess_s": ${PP_TIME},
  "cv_rate_s_per_100": ${CV_RATE},
  "vcpus": ${NPROC},
  "num_shards": ${NUM_SHARDS},
  "machine": {
    "arch": "$(uname -m)",
    "cpu_model": "${CPU_MODEL}",
    "vcpus": ${NPROC},
    "ram_gb": ${RAM_GB},
    "kernel": "${KERNEL}",
    "bf16": ${BF16_SUPPORT}
  },
  "config": {
    "image": "${IMAGE}",
    "batch_size": ${BATCH_SIZE},
    "region": "${REGION}",
    "usd_per_hr": ${USD_PER_HR},
    "docker_mem": "${DOCKER_MEM}"
  }
}
JSONEOF
}

# --- Test matrix ---
echo "============================================="
echo "  Test 1/3: BF16, jemalloc OFF, N=4"
echo "============================================="
for i in 1 2 3 4; do
  run_single "bf16_off_${i}" "bf16" "off"
done

echo ""
echo "============================================="
echo "  Test 2/3: BF16, jemalloc ON, N=4"
echo "============================================="
for i in 1 2 3 4; do
  run_single "bf16_on_${i}" "bf16" "on"
done

echo ""
echo "============================================="
echo "  Test 3/3: INT8 ONNX, jemalloc OFF, N=2"
echo "============================================="
for i in 1 2; do
  run_single "int8_off_${i}" "int8" "off"
done

# --- Compute summary ---
echo ""
echo "============================================="
echo "  Computing Summary"
echo "============================================="

python3 - "${RESULTS_DIR}" "${USD_PER_HR}" "${PLATFORM}" <<'PYEOF'
import json, sys, os, math

results_dir = sys.argv[1]
usd_per_hr = float(sys.argv[2])
platform = sys.argv[3]

# 16-vCPU reference data
ref_16 = {
    'graviton3': {'bf16': {'wall': 487, 'cv': 185, 'me': 278},
                  'int8': {'wall': 507, 'cv': 194, 'me': 299}},
    'graviton4': {'int8': {'wall': 366, 'cv': 158, 'me': 194}},
}

def load_runs(backend, jemalloc):
    runs = []
    for f in sorted(os.listdir(results_dir)):
        if not f.endswith('.json') or f == 'benchmark_32vcpu_summary.json':
            continue
        with open(os.path.join(results_dir, f)) as fh:
            r = json.load(fh)
        if r.get('backend') == backend and r.get('jemalloc') == jemalloc:
            runs.append(r)
    return runs

def stats(values):
    values = [v for v in values if v is not None and v != 'null']
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return round(mean, 1), round(std, 1)

def safe_float(v):
    if v is None or v == 'null':
        return None
    return float(v)

def summarize(runs):
    wall_m, wall_s = stats([safe_float(r['wall_time_s']) for r in runs])
    me_m, me_s = stats([safe_float(r['make_examples_s']) for r in runs])
    cv_m, cv_s = stats([safe_float(r['call_variants_s']) for r in runs])
    pp_m, pp_s = stats([safe_float(r['postprocess_s']) for r in runs])
    cv_rate_m, cv_rate_s = stats([safe_float(r.get('cv_rate_s_per_100')) for r in runs])
    cost = round(wall_m * 48.1 / 3600 * usd_per_hr, 2) if wall_m else None
    return {
        'runs': len(runs),
        'wall_mean': wall_m, 'wall_std': wall_s,
        'me_mean': me_m, 'me_std': me_s,
        'cv_mean': cv_m, 'cv_std': cv_s,
        'pp_mean': pp_m, 'pp_std': pp_s,
        'cv_rate_mean': cv_rate_m, 'cv_rate_std': cv_rate_s,
        'cost_per_genome': cost,
    }

configs = [
    ('bf16', 'off', 'BF16 jemalloc OFF'),
    ('bf16', 'on',  'BF16 jemalloc ON'),
    ('int8', 'off', 'INT8 ONNX ref'),
]

results = {}
for backend, jemalloc, label in configs:
    runs = load_runs(backend, jemalloc)
    if runs:
        results[label] = summarize(runs)

# --- Console output ---
print()
print('=' * 70)
print(f'  32-vCPU SCALING BENCHMARK SUMMARY ({platform})')
print('=' * 70)
print(f'  Cost rate: ${usd_per_hr}/hr')
print()

header = f"{'Config':>22s}  {'N':>3s}  {'Wall':>8s}  {'ME':>8s}  {'CV':>8s}  {'CV rate':>10s}  {'$/genome':>10s}"
print(header)
print('-' * len(header))

for label, s in results.items():
    wall_str = f"{s['wall_mean']:>6.0f}s" if s['wall_mean'] else 'N/A'
    me_str = f"{s['me_mean']:>6.0f}s" if s['me_mean'] else 'N/A'
    cv_str = f"{s['cv_mean']:>6.0f}s" if s['cv_mean'] else 'N/A'
    rate_str = f"{s['cv_rate_mean']:.3f}" if s['cv_rate_mean'] else 'N/A'
    cost_str = f"${s['cost_per_genome']}" if s['cost_per_genome'] else 'N/A'
    print(f"  {label:>20s}  {s['runs']:>3d}  {wall_str:>8s}  {me_str:>8s}  {cv_str:>8s}  {rate_str:>10s}  {cost_str:>10s}")

# Compare against 16-vCPU reference
print()
print('  --- vs 16-vCPU reference ---')
for label, s in results.items():
    backend = 'bf16' if 'BF16' in label else 'int8'
    ref = ref_16.get(platform, {}).get(backend, {})
    if ref and s['wall_mean'] and s['cv_mean']:
        wall_delta = (s['wall_mean'] - ref['wall']) / ref['wall'] * 100
        cv_delta = (s['cv_mean'] - ref['cv']) / ref['cv'] * 100
        print(f"  {label:>20s}  wall: {wall_delta:>+.1f}%  CV: {cv_delta:>+.1f}%")

# Bottleneck analysis
print()
print('  --- Bottleneck analysis ---')
for label, s in results.items():
    if s['me_mean'] and s['cv_mean']:
        if s['me_mean'] > s['cv_mean'] * 0.9:
            print(f"  {label}: ME ({s['me_mean']:.0f}s) ≈ CV ({s['cv_mean']:.0f}s) — ME is bottleneck.")
            print(f"    → SVE Smith-Waterman optimization is the next target.")
        else:
            print(f"  {label}: CV ({s['cv_mean']:.0f}s) >> ME ({s['me_mean']:.0f}s) — CV still bottleneck.")

# Save summary JSON
summary = {
    'platform': platform,
    'vcpus': 32,
    'usd_per_hr': usd_per_hr,
    'results': results,
    'reference_16vcpu': ref_16.get(platform, {}),
    'formula': 'cost = wall_mean_s * 48.1 / 3600 * usd_per_hr',
}
summary_path = os.path.join(results_dir, 'benchmark_32vcpu_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print()
print(f"  Summary JSON: {summary_path}")
print()
PYEOF

echo "Done. Per-run JSONs and summary in: ${RESULTS_DIR}"
