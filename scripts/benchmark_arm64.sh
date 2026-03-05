#!/bin/bash
set -euo pipefail

# Benchmark DeepVariant ARM64 build using GIAB HG003 chr20 case study.
# This mirrors Google's published case study for direct comparison.
#
# Usage:
#   bash scripts/benchmark_arm64.sh [--accuracy] [--docker-image IMAGE]
#
# Requirements:
#   - Docker with deepvariant-arm64 image built
#   - ~50GB free disk for reference + BAM data
#   - Runs on ARM64 host (Graviton, Ampere, etc.)

DOCKER_IMAGE="deepvariant-arm64"
RUN_ACCURACY=false
USE_ONNX=false
DATA_DIR="${HOME}/deepvariant-benchmark"
OUTPUT_DIR="${DATA_DIR}/output"

while [[ $# -gt 0 ]]; do
  case $1 in
    --accuracy)
      RUN_ACCURACY=true
      shift
      ;;
    --onnx)
      USE_ONNX=true
      shift
      ;;
    --docker-image)
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      OUTPUT_DIR="${DATA_DIR}/output"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "========== DeepVariant ARM64 Benchmark =========="
echo "Docker image: ${DOCKER_IMAGE}"
echo "Data dir: ${DATA_DIR}"
echo "Accuracy validation: ${RUN_ACCURACY}"
echo "ONNX Runtime: ${USE_ONNX}"
echo ""

# System info
echo "========== System Info"
echo "Architecture: $(uname -m)"
echo "CPU: $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null || echo 'ARM64')"
echo "Cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
if grep -q bf16 /proc/cpuinfo 2>/dev/null; then
  echo "BF16: supported"
fi
echo ""

# TF/ONNX runtime config
echo "========== Runtime Config"
echo "TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-not set}"
echo "DNNL_DEFAULT_FPMATH_MODE=${DNNL_DEFAULT_FPMATH_MODE:-not set}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-not set}"
echo "TF_NUM_INTEROP_THREADS=${TF_NUM_INTEROP_THREADS:-not set}"
echo "TF_NUM_INTRAOP_THREADS=${TF_NUM_INTRAOP_THREADS:-not set}"
echo "KMP_BLOCKTIME=${KMP_BLOCKTIME:-not set}"
echo ""

# Download test data (GIAB HG003 chr20)
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

# Data sources (per upstream case study docs):
#   Reference: NCBI FTP (full genome .fna.gz, gunzip locally)
#   BAM: Google Cloud Storage (chr20 subset)
#   Truth: GIAB FTP (full genome VCF/BED)
NCBI_REF_DIR="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
GCS_DIR="https://storage.googleapis.com/deepvariant/case-study-testdata"
GIAB_DIR="https://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38"

REF="GRCh38_no_alt_analysis_set.fasta"
BAM="HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
TRUTH_VCF="HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TRUTH_BED="HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"

echo "========== Downloading test data (if not cached)"

# Reference FASTA (download compressed, decompress)
if [[ ! -f "${DATA_DIR}/${REF}" ]]; then
  echo "  Downloading reference FASTA (compressed)..."
  curl -s "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
    | gunzip > "${DATA_DIR}/${REF}"
fi
if [[ ! -f "${DATA_DIR}/${REF}.fai" ]]; then
  echo "  Downloading reference FASTA index..."
  curl -s -o "${DATA_DIR}/${REF}.fai" \
    "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai"
fi

# BAM (from GCS, chr20 subset)
for FILE in "${BAM}" "${BAM}.bai"; do
  if [[ ! -f "${DATA_DIR}/${FILE}" ]]; then
    echo "  Downloading ${FILE}..."
    curl -s -o "${DATA_DIR}/${FILE}" "${GCS_DIR}/${FILE}"
  else
    echo "  Cached: ${FILE}"
  fi
done

# Truth VCF and BED (from GIAB FTP)
for FILE in "${TRUTH_VCF}" "${TRUTH_VCF}.tbi" "${TRUTH_BED}"; do
  if [[ ! -f "${DATA_DIR}/${FILE}" ]]; then
    echo "  Downloading ${FILE}..."
    curl -s -o "${DATA_DIR}/${FILE}" "${GIAB_DIR}/${FILE}"
  else
    echo "  Cached: ${FILE}"
  fi
done

echo ""
echo "========== Running DeepVariant (chr20, WGS)"
START_TIME=$(date +%s)

ONNX_FLAG=""
if [[ "${USE_ONNX}" == "true" ]]; then
  ONNX_FLAG="--use_onnx"
  echo "Using ONNX Runtime for inference"
fi

docker run --rm \
  -v "${DATA_DIR}:/data" \
  -v "${OUTPUT_DIR}:/output" \
  "${DOCKER_IMAGE}" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref="/data/${REF}" \
    --reads="/data/${BAM}" \
    --output_vcf=/output/HG003_arm64.vcf.gz \
    --output_gvcf=/output/HG003_arm64.g.vcf.gz \
    --num_shards="$(nproc)" \
    --regions=chr20 \
    --intermediate_results_dir=/output/intermediate \
    ${ONNX_FLAG}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========== Benchmark Results"
echo "Total wall time: ${MINUTES}m ${SECONDS}s (${ELAPSED}s)"
echo "Cores used: $(nproc)"
echo ""

# Save benchmark result
cat > "${OUTPUT_DIR}/benchmark_result.json" << EOF
{
  "architecture": "$(uname -m)",
  "cpu_model": "$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo 'ARM64')",
  "cores": $(nproc),
  "ram_gb": $(free -g | awk '/^Mem:/ {print $2}'),
  "bf16_support": $(grep -q bf16 /proc/cpuinfo 2>/dev/null && echo true || echo false),
  "docker_image": "${DOCKER_IMAGE}",
  "use_onnx": ${USE_ONNX},
  "region": "chr20",
  "sample": "HG003",
  "wall_time_seconds": ${ELAPSED},
  "wall_time_formatted": "${MINUTES}m ${SECONDS}s",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
echo "Benchmark results saved to ${OUTPUT_DIR}/benchmark_result.json"

# Accuracy validation
if [[ "${RUN_ACCURACY}" == "true" ]]; then
  echo ""
  echo "========== Running Accuracy Validation (hap.py)"
  bash "$(dirname "$0")/validate_accuracy.sh" \
    --vcf "${OUTPUT_DIR}/HG003_arm64.vcf.gz" \
    --truth-vcf "${DATA_DIR}/${TRUTH_VCF}" \
    --truth-bed "${DATA_DIR}/${TRUTH_BED}" \
    --ref "${DATA_DIR}/${REF}" \
    --output-dir "${OUTPUT_DIR}/happy_results"
fi

echo ""
echo "========== Done =========="
