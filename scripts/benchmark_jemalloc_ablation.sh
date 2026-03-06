#!/bin/bash
# Jemalloc ablation benchmark — measures the impact of jemalloc on DeepVariant
# pipeline performance with statistical rigor.
#
# Runs interleaved with/without jemalloc (off, on, off, on, ...) to eliminate
# filesystem cache ordering bias. Captures per-stage timing, peak RSS, and
# startup overhead. Produces per-run JSON + summary JSON with mean/std.
#
# Usage:
#   bash scripts/benchmark_jemalloc_ablation.sh \
#     --runs 4 \
#     --image deepvariant-arm64:onnx-fix-v3 \
#     --data-dir /data \
#     [--usd-per-hr 0.32] \
#     [--use-onnx] [--onnx-model /path/to/model.onnx] \
#     [--num-shards 16] [--batch-size 256] \
#     [--tf-onednn-opts 0]  # Set to 0 for AmpereOne (SIGILL with OneDNN+ACL)
set -euo pipefail

# --- Defaults ---
RUNS=4
IMAGE="deepvariant-arm64:onnx-fix-v3"
DATA_DIR="/data"
USD_PER_HR=""
USE_ONNX=""
ONNX_MODEL=""
NUM_SHARDS=""
BATCH_SIZE=256
DOCKER_MEM="28g"
REGION="chr20"
TF_ONEDNN_OPTS="1"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --runs)        RUNS="$2"; shift 2 ;;
    --image)       IMAGE="$2"; shift 2 ;;
    --data-dir)    DATA_DIR="$2"; shift 2 ;;
    --usd-per-hr)  USD_PER_HR="$2"; shift 2 ;;
    --use-onnx)    USE_ONNX="true"; shift ;;
    --onnx-model)  ONNX_MODEL="$2"; shift 2 ;;
    --num-shards)  NUM_SHARDS="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --docker-mem)  DOCKER_MEM="$2"; shift 2 ;;
    --tf-onednn-opts) TF_ONEDNN_OPTS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

NPROC=$(nproc)
NUM_SHARDS="${NUM_SHARDS:-${NPROC}}"
RESULTS_DIR="${DATA_DIR}/benchmark_results/jemalloc_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}"

# --- Machine metadata ---
CPU_MODEL=$(grep -m1 'model name\|CPU part' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo "unknown")
RAM_GB=$(awk '/^MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "unknown")
KERNEL=$(uname -r)
BF16_SUPPORT=$(grep -q bf16 /proc/cpuinfo 2>/dev/null && echo "true" || echo "false")

# Detect jemalloc path inside the Docker image
JEMALLOC_PATH=$(docker run --rm "${IMAGE}" bash -c \
  'ls /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 2>/dev/null || echo "not_found"')

# ORT version (best effort)
ORT_VERSION=$(docker run --rm "${IMAGE}" python3 -c \
  'try:
    import onnxruntime; print(onnxruntime.__version__)
except: print("unknown")' 2>/dev/null || echo "unknown")

echo "============================================="
echo "  Jemalloc Ablation Benchmark"
echo "============================================="
echo "Runs per condition: ${RUNS}"
echo "Image: ${IMAGE}"
echo "Platform: $(uname -m), ${NPROC} vCPUs, ${RAM_GB} GB RAM"
echo "CPU: ${CPU_MODEL}"
echo "BF16: ${BF16_SUPPORT}"
echo "jemalloc path: ${JEMALLOC_PATH}"
echo "ORT version: ${ORT_VERSION}"
echo "OneDNN: TF_ENABLE_ONEDNN_OPTS=${TF_ONEDNN_OPTS}"
echo "Shards: ${NUM_SHARDS}, Batch: ${BATCH_SIZE}"
echo "Region: ${REGION}"
if [[ -n "${USD_PER_HR}" ]]; then
  echo "Cost rate: \$${USD_PER_HR}/hr"
else
  echo "Cost rate: not provided (cost_per_genome will be null)"
fi
echo "Results: ${RESULTS_DIR}"
echo ""

# --- Build call_variants extra args ---
CV_EXTRA_ARGS="--batch_size=${BATCH_SIZE}"
if [[ -n "${USE_ONNX}" ]]; then
  CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--use_onnx=true"
  if [[ -n "${ONNX_MODEL}" ]]; then
    CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--onnx_model=${ONNX_MODEL}"
  fi
fi

# Common env vars for TF/ONNX optimization
COMMON_ENVS="-e TF_ENABLE_ONEDNN_OPTS=${TF_ONEDNN_OPTS} -e CUDA_VISIBLE_DEVICES="

# --- RSS monitoring helper ---
# Polls docker stats every 1s, records peak memory usage in MB.
# Usage: start_rss_monitor <container_name> <output_file>
#        stop_rss_monitor
start_rss_monitor() {
  local container="$1"
  local outfile="$2"
  (
    peak_mb=0
    while true; do
      # docker stats --no-stream outputs "X.XXMiB / Y.YYGiB" or "X.XXGiB / ..."
      mem_raw=$(docker stats --no-stream --format "{{.MemUsage}}" "${container}" 2>/dev/null | awk '{print $1}')
      if [[ -n "${mem_raw}" ]]; then
        # Convert to MB
        if [[ "${mem_raw}" == *GiB ]]; then
          mb=$(echo "${mem_raw%GiB}" | awk '{printf "%.0f", $1 * 1024}')
        elif [[ "${mem_raw}" == *MiB ]]; then
          mb=$(echo "${mem_raw%MiB}" | awk '{printf "%.0f", $1}')
        elif [[ "${mem_raw}" == *KiB ]]; then
          mb=$(echo "${mem_raw%KiB}" | awk '{printf "%.0f", $1 / 1024}')
        else
          mb=0
        fi
        if [[ "${mb}" -gt "${peak_mb}" ]]; then
          peak_mb="${mb}"
        fi
      fi
      sleep 1
    done
    # This line is reached when the subshell is killed
  ) &
  RSS_MONITOR_PID=$!
  RSS_OUTFILE="${outfile}"
}

stop_rss_monitor() {
  if [[ -n "${RSS_MONITOR_PID:-}" ]]; then
    kill "${RSS_MONITOR_PID}" 2>/dev/null || true
    wait "${RSS_MONITOR_PID}" 2>/dev/null || true
    RSS_MONITOR_PID=""
  fi
}

# --- Run a single benchmark ---
# Args: run_name jemalloc_flag(on|off)
run_single() {
  local RUN_NAME="$1"
  local JEMALLOC="$2"  # "on" or "off"
  local OUT_DIR="${DATA_DIR}/output/${RUN_NAME}"
  local LOG="${RESULTS_DIR}/${RUN_NAME}.log"
  local CONTAINER_NAME="dv_ablation_${RUN_NAME}"

  echo ""
  echo ">>> [${RUN_NAME}] jemalloc=${JEMALLOC} ..."

  mkdir -p "${OUT_DIR}"

  # Build jemalloc env flag
  local JEMALLOC_ENV=""
  if [[ "${JEMALLOC}" == "on" ]]; then
    JEMALLOC_ENV="-e DV_USE_JEMALLOC=1"
  fi

  # Start RSS monitor (in background, polls every 1s)
  local RSS_FILE="${RESULTS_DIR}/${RUN_NAME}_rss.txt"
  echo "0" > "${RSS_FILE}"

  WALL_START=$(date +%s)

  # Run pipeline — use --name for RSS monitoring
  docker run --rm \
    --name "${CONTAINER_NAME}" \
    --memory="${DOCKER_MEM}" \
    -v "${DATA_DIR}:/data" \
    ${COMMON_ENVS} ${JEMALLOC_ENV} \
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
    2>&1 | tee "${LOG}" &

  local DOCKER_PID=$!

  # Wait a moment for container to start, then begin RSS polling
  sleep 3
  local PEAK_RSS_MB=0
  (
    while kill -0 "${DOCKER_PID}" 2>/dev/null; do
      mem_raw=$(docker stats --no-stream --format "{{.MemUsage}}" "${CONTAINER_NAME}" 2>/dev/null | awk '{print $1}')
      if [[ -n "${mem_raw}" ]]; then
        if [[ "${mem_raw}" == *GiB ]]; then
          mb=$(echo "${mem_raw%GiB}" | awk '{printf "%.0f", $1 * 1024}')
        elif [[ "${mem_raw}" == *MiB ]]; then
          mb=$(echo "${mem_raw%MiB}" | awk '{printf "%.0f", $1}')
        else
          mb=0
        fi
        echo "${mb}"
      fi
      sleep 1
    done
  ) > "${RSS_FILE}" 2>/dev/null &
  local RSS_PID=$!

  # Wait for pipeline to finish
  wait "${DOCKER_PID}" || true
  WALL_END=$(date +%s)
  WALL_TIME=$((WALL_END - WALL_START))

  # Stop RSS monitor
  kill "${RSS_PID}" 2>/dev/null || true
  wait "${RSS_PID}" 2>/dev/null || true

  # Compute peak RSS from collected samples
  PEAK_RSS_MB=$(sort -n "${RSS_FILE}" 2>/dev/null | tail -1 || echo "0")
  [[ -z "${PEAK_RSS_MB}" ]] && PEAK_RSS_MB=0

  # Extract per-stage timing from log.
  # run_deepvariant.py wraps each step with `time`, producing `real XmY.YYYs` lines.
  # The three `real` lines appear in order: make_examples, call_variants, postprocess.
  # Convert "XmY.YYYs" to seconds.
  _real_to_sec() {
    local t="$1"
    if [[ "${t}" =~ ^([0-9]+)m([0-9.]+)s$ ]]; then
      python3 -c "print(round(${BASH_REMATCH[1]} * 60 + ${BASH_REMATCH[2]}, 1))" 2>/dev/null || echo "null"
    else
      echo "null"
    fi
  }
  local _reals
  _reals=($(grep -oP '(?<=^real\t)[0-9]+m[0-9.]+s' "${LOG}" 2>/dev/null || true))
  ME_TIME=$(_real_to_sec "${_reals[0]:-}")
  CV_TIME=$(_real_to_sec "${_reals[1]:-}")
  PP_TIME=$(_real_to_sec "${_reals[2]:-}")

  # Extract CV rate (s/100) from the last "Predicted N examples" log line
  CV_RATE="null"
  if [[ "${CV_TIME}" != "null" ]]; then
    local last_rate
    last_rate=$(grep -oP '\[([0-9.]+) sec per 100\]' "${LOG}" 2>/dev/null | tail -1 | grep -oP '[0-9.]+' || echo "")
    if [[ -n "${last_rate}" ]]; then
      CV_RATE="${last_rate}"
    fi
  fi

  # Startup overhead = wall - (ME + CV + PP)
  # Captures container init, model loading, and jemalloc arena setup
  STARTUP_OVERHEAD="null"
  if [[ "${ME_TIME}" != "null" && "${CV_TIME}" != "null" && "${PP_TIME}" != "null" ]]; then
    STARTUP_OVERHEAD=$(python3 -c "
me=${ME_TIME}; cv=${CV_TIME}; pp=${PP_TIME}; wall=${WALL_TIME}
overhead = wall - me - cv - pp
print(round(max(0, overhead), 1))
" 2>/dev/null || echo "null")
  fi

  echo "  Wall: ${WALL_TIME}s  ME: ${ME_TIME}s  CV: ${CV_TIME}s  PP: ${PP_TIME}s  RSS: ${PEAK_RSS_MB}MB  Startup: ${STARTUP_OVERHEAD}s"

  # Save per-run JSON
  cat > "${RESULTS_DIR}/${RUN_NAME}.json" <<JSONEOF
{
  "run_name": "${RUN_NAME}",
  "jemalloc": "${JEMALLOC}",
  "wall_time_s": ${WALL_TIME},
  "make_examples_s": ${ME_TIME},
  "call_variants_s": ${CV_TIME},
  "postprocess_s": ${PP_TIME},
  "cv_rate_s_per_100": ${CV_RATE},
  "peak_rss_mb": ${PEAK_RSS_MB},
  "startup_overhead_s": ${STARTUP_OVERHEAD},
  "machine": {
    "arch": "$(uname -m)",
    "cpu_model": "${CPU_MODEL}",
    "vcpus": ${NPROC},
    "ram_gb": ${RAM_GB},
    "kernel": "${KERNEL}",
    "bf16": ${BF16_SUPPORT},
    "ort_version": "${ORT_VERSION}",
    "jemalloc_path": "${JEMALLOC_PATH}"
  },
  "config": {
    "image": "${IMAGE}",
    "num_shards": ${NUM_SHARDS},
    "batch_size": ${BATCH_SIZE},
    "region": "${REGION}",
    "use_onnx": ${USE_ONNX:-false},
    "onnx_model": "${ONNX_MODEL}",
    "usd_per_hr": ${USD_PER_HR:-null},
    "docker_mem": "${DOCKER_MEM}",
    "tf_onednn_opts": ${TF_ONEDNN_OPTS}
  }
}
JSONEOF
}

# --- Interleaved run loop ---
echo ""
echo "Running ${RUNS} pairs (interleaved: off, on, off, on, ...):"
echo "============================================="

for i in $(seq 1 "${RUNS}"); do
  run_single "run_${i}_jemalloc_off" "off"
  run_single "run_${i}_jemalloc_on"  "on"
done

# --- Compute summary ---
echo ""
echo "============================================="
echo "  Computing Summary"
echo "============================================="

python3 - "${RESULTS_DIR}" "${USD_PER_HR}" <<'PYEOF'
import json, sys, os, math

results_dir = sys.argv[1]
usd_per_hr = float(sys.argv[2]) if sys.argv[2] else None

def load_runs(condition):
    """Load all run JSONs for a condition (on/off)."""
    runs = []
    for f in sorted(os.listdir(results_dir)):
        if f.endswith('.json') and f"jemalloc_{condition}" in f and f != 'jemalloc_ablation_summary.json':
            with open(os.path.join(results_dir, f)) as fh:
                runs.append(json.load(fh))
    return runs

def stats(values):
    """Compute mean and std for a list of numbers."""
    values = [v for v in values if v is not None and v != 'null']
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return round(mean, 2), round(std, 2)

def safe_float(v):
    if v is None or v == 'null':
        return None
    return float(v)

def summarize(runs):
    wall = [safe_float(r['wall_time_s']) for r in runs]
    me = [safe_float(r['make_examples_s']) for r in runs]
    cv = [safe_float(r['call_variants_s']) for r in runs]
    pp = [safe_float(r['postprocess_s']) for r in runs]
    cv_rate = [safe_float(r.get('cv_rate_s_per_100')) for r in runs]
    rss = [safe_float(r.get('peak_rss_mb', 0)) for r in runs]
    startup = [safe_float(r.get('startup_overhead_s')) for r in runs]

    wall_mean, wall_std = stats(wall)
    me_mean, me_std = stats(me)
    cv_mean, cv_std = stats(cv)
    pp_mean, pp_std = stats(pp)
    cv_rate_mean, cv_rate_std = stats(cv_rate)
    rss_mean, rss_std = stats(rss)
    startup_mean, startup_std = stats(startup)

    cost = None
    if usd_per_hr and wall_mean:
        cost = round(wall_mean * 48.1 / 3600 * usd_per_hr, 2)

    return {
        'runs': len(runs),
        'wall_mean': wall_mean, 'wall_std': wall_std,
        'me_mean': me_mean, 'me_std': me_std,
        'cv_mean': cv_mean, 'cv_std': cv_std,
        'pp_mean': pp_mean, 'pp_std': pp_std,
        'cv_rate_mean': cv_rate_mean, 'cv_rate_std': cv_rate_std,
        'peak_rss_mb_mean': rss_mean, 'peak_rss_mb_std': rss_std,
        'startup_overhead_mean': startup_mean, 'startup_overhead_std': startup_std,
        'cost_per_genome': cost,
    }

off_runs = load_runs('off')
on_runs = load_runs('on')
off_summary = summarize(off_runs)
on_summary = summarize(on_runs)

# Compute deltas
delta = {}
for key in ['wall', 'me', 'cv', 'pp', 'cv_rate']:
    off_val = off_summary.get(f'{key}_mean')
    on_val = on_summary.get(f'{key}_mean')
    if off_val and on_val and off_val > 0:
        delta[f'{key}_pct'] = round((on_val - off_val) / off_val * 100, 1)
if off_summary['cost_per_genome'] and on_summary['cost_per_genome']:
    delta['cost_per_genome_delta'] = round(
        on_summary['cost_per_genome'] - off_summary['cost_per_genome'], 2)

# Get machine/config from first run
machine = off_runs[0].get('machine', {}) if off_runs else {}
config = off_runs[0].get('config', {}) if off_runs else {}

summary = {
    'machine': machine,
    'config': config,
    'results': {
        'jemalloc_off': off_summary,
        'jemalloc_on': on_summary,
        'delta': delta,
    },
    'formula': 'cost = wall_mean_s * 48.1 / 3600 * usd_per_hr',
}

summary_path = os.path.join(results_dir, 'jemalloc_ablation_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

# --- Console output ---
print()
print('=' * 60)
print('  JEMALLOC ABLATION SUMMARY')
print('=' * 60)
print(f"  Runs per condition: {off_summary['runs']}")
print(f"  Instance: {machine.get('vcpus', '?')} vCPU, {machine.get('ram_gb', '?')} GB")
print(f"  CPU: {machine.get('cpu_model', '?')}")
if usd_per_hr:
    print(f"  Cost rate: ${usd_per_hr}/hr")
print()

header = f"{'':>18s}  {'jemalloc OFF':>18s}  {'jemalloc ON':>18s}  {'Delta':>10s}"
print(header)
print('-' * len(header))

def fmt_row(label, off_mean, off_std, on_mean, on_std, delta_pct):
    off_str = f"{off_mean:>7.1f} ±{off_std:>5.1f}" if off_mean is not None else "N/A"
    on_str = f"{on_mean:>7.1f} ±{on_std:>5.1f}" if on_mean is not None else "N/A"
    d_str = f"{delta_pct:>+6.1f}%" if delta_pct is not None else "N/A"
    print(f"  {label:>16s}  {off_str:>18s}  {on_str:>18s}  {d_str:>10s}")

for metric, label in [('wall', 'Wall (s)'), ('me', 'ME (s)'), ('cv', 'CV (s)'),
                       ('pp', 'PP (s)'), ('cv_rate', 'CV rate (s/100)'),
                       ('peak_rss_mb', 'Peak RSS (MB)'), ('startup_overhead', 'Startup (s)')]:
    fmt_row(label,
            off_summary.get(f'{metric}_mean'), off_summary.get(f'{metric}_std', 0),
            on_summary.get(f'{metric}_mean'), on_summary.get(f'{metric}_std', 0),
            delta.get(f'{metric}_pct'))

if off_summary['cost_per_genome'] or on_summary['cost_per_genome']:
    print()
    off_cost = off_summary['cost_per_genome']
    on_cost = on_summary['cost_per_genome']
    cost_delta = delta.get('cost_per_genome_delta', '')
    print(f"  {'$/genome':>16s}  {'$'+str(off_cost) if off_cost else 'N/A':>18s}  {'$'+str(on_cost) if on_cost else 'N/A':>18s}  {'$'+str(cost_delta) if cost_delta else 'N/A':>10s}")

print()
print(f"  Formula: cost = wall_mean × 48.1 / 3600 × $/hr")
print(f"  Summary JSON: {summary_path}")
print()
PYEOF

echo "Done. Per-run JSONs and summary in: ${RESULTS_DIR}"
