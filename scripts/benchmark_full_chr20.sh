#!/bin/bash
# Full chr20 benchmark — matches macOS benchmark protocol.
#
# Runs 2 × FP32 + 2 × BF16 with per-stage timing, then hap.py accuracy.
# Designed for AWS c7g.4xlarge (16 vCPU Graviton3, 32 GB RAM).
#
# Usage: bash scripts/benchmark_full_chr20.sh [--data-dir /data]
set -euo pipefail

IMAGE="ghcr.io/antomicblitz/deepvariant-arm64:optimized"
DATA_DIR="${DATA_DIR:-/data}"
DOCKER_MEM="28g"
BATCH_SIZE=256
REGION="chr20"
NPROC=$(nproc)
USE_ONNX=""
ONNX_MODEL=""

# BF16 env
BF16_ENVS="-e ONEDNN_DEFAULT_FPMATH_MODE=BF16"
COMMON_ENVS="-e TF_ENABLE_ONEDNN_OPTS=1 -e OMP_NUM_THREADS=${NPROC} -e OMP_PROC_BIND=false -e OMP_PLACES=cores -e CUDA_VISIBLE_DEVICES="

while [[ $# -gt 0 ]]; do
  case $1 in
    --image) IMAGE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --use_onnx) USE_ONNX="true"; shift ;;
    --onnx_model) ONNX_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RESULTS_DIR="${DATA_DIR}/benchmark_results"
mkdir -p "${RESULTS_DIR}"

echo "============================================="
echo "  DeepVariant Full chr20 Benchmark"
echo "============================================="
echo "Platform: $(uname -m), ${NPROC} vCPUs"
echo "Image: ${IMAGE}"
echo "Region: ${REGION} (full chromosome)"
echo "Batch size: ${BATCH_SIZE}"
echo "BF16: $(grep -q bf16 /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo "ONNX: ${USE_ONNX:-disabled}"
if [[ -n "${ONNX_MODEL}" ]]; then
  echo "ONNX model: ${ONNX_MODEL}"
fi
echo ""

# Build call_variants extra args
CV_EXTRA_ARGS="--batch_size=${BATCH_SIZE}"
if [[ -n "${USE_ONNX}" ]]; then
  CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--use_onnx=true"
  if [[ -n "${ONNX_MODEL}" ]]; then
    CV_EXTRA_ARGS="${CV_EXTRA_ARGS},--onnx_model=${ONNX_MODEL}"
  fi
fi

run_benchmark() {
  local RUN_NAME="$1"   # e.g. fp32_run1, bf16_run2
  local EXTRA_ENVS="$2" # additional env vars for BF16
  local OUT_DIR="${DATA_DIR}/output/${RUN_NAME}"

  echo ""
  echo ">>> Starting ${RUN_NAME} ..."
  echo "============================================="
  mkdir -p "${OUT_DIR}"

  WALL_START=$(date +%s)
  docker run --rm \
    --memory="${DOCKER_MEM}" \
    -v "${DATA_DIR}:/data" \
    ${COMMON_ENVS} ${EXTRA_ENVS} \
    "${IMAGE}" \
    /opt/deepvariant/bin/run_deepvariant \
      --model_type=WGS \
      --ref=/data/reference/GRCh38_no_alt_analysis_set.fasta \
      --reads=/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
      --output_vcf="/data/output/${RUN_NAME}/output.vcf.gz" \
      --regions="${REGION}" \
      --num_shards="${NPROC}" \
      --intermediate_results_dir="/data/output/${RUN_NAME}/intermediate" \
      --call_variants_extra_args="${CV_EXTRA_ARGS}" \
    2>&1 | tee "${RESULTS_DIR}/${RUN_NAME}.log"
  WALL_END=$(date +%s)
  WALL_TIME=$((WALL_END - WALL_START))

  # Extract per-stage timing from log using DeepVariant's timing output
  # Format: "Timer: making examples took X.XXXs."
  ME_TIME=$(grep -oP 'making examples.*?took \K[0-9.]+' "${RESULTS_DIR}/${RUN_NAME}.log" 2>/dev/null || echo "N/A")
  CV_TIME=$(grep -oP 'call_variants.*?took \K[0-9.]+' "${RESULTS_DIR}/${RUN_NAME}.log" 2>/dev/null || echo "N/A")
  PP_TIME=$(grep -oP 'postprocess_variants.*?took \K[0-9.]+' "${RESULTS_DIR}/${RUN_NAME}.log" 2>/dev/null || echo "N/A")

  echo ""
  echo "  ${RUN_NAME} WALL TIME: ${WALL_TIME}s ($(( WALL_TIME / 60 ))m$(( WALL_TIME % 60 ))s)"
  echo "    make_examples:        ${ME_TIME}s"
  echo "    call_variants:        ${CV_TIME}s"
  echo "    postprocess_variants: ${PP_TIME}s"

  # Save timing to file
  cat > "${RESULTS_DIR}/${RUN_NAME}_timing.json" <<JSONEOF
{
  "run": "${RUN_NAME}",
  "region": "${REGION}",
  "batch_size": ${BATCH_SIZE},
  "use_onnx": ${USE_ONNX:-false},
  "onnx_model": "${ONNX_MODEL}",
  "vcpus": ${NPROC},
  "wall_time_s": ${WALL_TIME},
  "make_examples_s": "${ME_TIME}",
  "call_variants_s": "${CV_TIME}",
  "postprocess_variants_s": "${PP_TIME}"
}
JSONEOF
}

# --- Run benchmarks ---
run_benchmark "fp32_run1" ""
run_benchmark "fp32_run2" ""
run_benchmark "bf16_run1" "${BF16_ENVS}"
run_benchmark "bf16_run2" "${BF16_ENVS}"

# --- hap.py accuracy validation (on run1 of each) ---
echo ""
echo ">>> Running hap.py accuracy validation..."
echo "============================================="

for CONFIG in fp32 bf16; do
  echo ""
  echo "  --- hap.py: ${CONFIG}_run1 ---"
  docker run --rm \
    -v "${DATA_DIR}:/data" \
    jmcdani20/hap.py:v0.3.12 \
    /opt/hap.py/bin/hap.py \
      /data/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
      "/data/output/${CONFIG}_run1/output.vcf.gz" \
      -f /data/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
      -r /data/reference/GRCh38_no_alt_analysis_set.fasta \
      -o "/data/output/${CONFIG}_run1/happy" \
      --engine=vcfeval \
      -l chr20 \
    2>&1 | tee "${RESULTS_DIR}/${CONFIG}_happy.log"

  if [[ -f "${DATA_DIR}/output/${CONFIG}_run1/happy.summary.csv" ]]; then
    echo ""
    echo "  ${CONFIG} accuracy (chr20):"
    head -1 "${DATA_DIR}/output/${CONFIG}_run1/happy.summary.csv"
    grep -E "^(SNP|INDEL)" "${DATA_DIR}/output/${CONFIG}_run1/happy.summary.csv" | \
      awk -F',' '{printf "    %-6s  F1=%s  Precision=%s  Recall=%s\n", $1, $7, $4, $5}'
  fi
done

# --- Final Summary ---
echo ""
echo ""
echo "============================================="
echo "       BENCHMARK SUMMARY"
echo "============================================="
echo ""
echo "Platform: $(uname -m), ${NPROC} vCPUs"
echo "BF16 hardware: $(grep -q bf16 /proc/cpuinfo 2>/dev/null && echo 'YES' || echo 'NO')"
echo "Region: ${REGION} (full chromosome)"
echo "Batch size: ${BATCH_SIZE}"
echo ""

printf "%-12s  %10s  %14s  %14s  %14s\n" "Run" "Wall(s)" "make_examples" "call_variants" "postprocess"
printf "%-12s  %10s  %14s  %14s  %14s\n" "---" "---" "---" "---" "---"
for RUN in fp32_run1 fp32_run2 bf16_run1 bf16_run2; do
  if [[ -f "${RESULTS_DIR}/${RUN}_timing.json" ]]; then
    WALL=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${RUN}_timing.json')); print(d['wall_time_s'])")
    ME=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${RUN}_timing.json')); print(d['make_examples_s'])")
    CV=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${RUN}_timing.json')); print(d['call_variants_s'])")
    PP=$(python3 -c "import json; d=json.load(open('${RESULTS_DIR}/${RUN}_timing.json')); print(d['postprocess_variants_s'])")
    printf "%-12s  %10ss  %14ss  %14ss  %14ss\n" "${RUN}" "${WALL}" "${ME}" "${CV}" "${PP}"
  fi
done

echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo "Done."
