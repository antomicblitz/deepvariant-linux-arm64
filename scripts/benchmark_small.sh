#!/bin/bash
set -euo pipefail

# Small benchmark for DeepVariant ARM64 — runs a 5MB region of chr20.
# Designed for memory-constrained instances (16GB RAM).
#
# Usage:
#   bash scripts/benchmark_small.sh [--onnx] [--docker-image IMAGE]
#
# Results from Hetzner CAX31 (8 vCPU Neoverse-N1, 16GB RAM):
#   make_examples: ~1m (5MB region, 8 shards, ~2300 examples)
#   call_variants: TBD (needs TF_NUM_INTRAOP_THREADS=2 to avoid OOM)
#   postprocess_variants: ~5s

DOCKER_IMAGE="deepvariant-arm64"
USE_ONNX=false
DATA_DIR="${HOME}/benchmark-data"
OUTPUT_DIR="${DATA_DIR}/output"
REGION="chr20:10000000-15000000"
NUM_SHARDS=8
# Limit TF threads to avoid OOM on 16GB instances.
# TF SavedModel + InceptionV3 uses ~15GB with 8 threads.
TF_THREADS=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --onnx) USE_ONNX=true; shift ;;
    --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; OUTPUT_DIR="${DATA_DIR}/output"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --shards) NUM_SHARDS="$2"; shift 2 ;;
    --tf-threads) TF_THREADS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "========== DeepVariant ARM64 Small Benchmark =========="
echo "Docker image: ${DOCKER_IMAGE}"
echo "Region: ${REGION}"
echo "Shards: ${NUM_SHARDS}"
echo "TF intra-op threads: ${TF_THREADS}"
echo "ONNX: ${USE_ONNX}"
echo ""

# System info
echo "========== System Info"
echo "Architecture: $(uname -m)"
echo "Cores: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/ {print $2}')"
echo "Swap: $(free -h | awk '/^Swap:/ {print $2}')"
echo ""

# Data download (same as benchmark_arm64.sh)
mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

NCBI_REF_DIR="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
GCS_DIR="https://storage.googleapis.com/deepvariant/case-study-testdata"
REF="GRCh38_no_alt_analysis_set.fasta"
BAM="HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"

echo "========== Checking test data"
if [[ ! -f "${DATA_DIR}/${REF}" ]]; then
  echo "  Downloading reference FASTA (compressed, ~900MB)..."
  curl -s "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
    | gunzip > "${DATA_DIR}/${REF}"
fi
if [[ ! -f "${DATA_DIR}/${REF}.fai" ]]; then
  echo "  Downloading reference FASTA index..."
  curl -s -o "${DATA_DIR}/${REF}.fai" \
    "${NCBI_REF_DIR}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai"
fi
for FILE in "${BAM}" "${BAM}.bai"; do
  if [[ ! -f "${DATA_DIR}/${FILE}" ]]; then
    echo "  Downloading ${FILE}..."
    curl -s -o "${DATA_DIR}/${FILE}" "${GCS_DIR}/${FILE}"
  else
    echo "  Cached: ${FILE}"
  fi
done

# Clean previous output
rm -rf "${OUTPUT_DIR}/intermediate"
mkdir -p "${OUTPUT_DIR}"

ONNX_FLAG=""
if [[ "${USE_ONNX}" == "true" ]]; then
  ONNX_FLAG="--use_onnx"
  echo "Using ONNX Runtime for inference"
fi

echo ""
echo "========== Running DeepVariant (${REGION})"
START_TIME=$(date +%s)

docker run --rm \
  -v "${DATA_DIR}:/data" \
  -v "${OUTPUT_DIR}:/output" \
  -e TF_NUM_INTRAOP_THREADS="${TF_THREADS}" \
  -e TF_NUM_INTEROP_THREADS=1 \
  -e OMP_NUM_THREADS="${TF_THREADS}" \
  "${DOCKER_IMAGE}" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref="/data/${REF}" \
    --reads="/data/${BAM}" \
    --output_vcf=/output/HG003_arm64.vcf.gz \
    --output_gvcf=/output/HG003_arm64.g.vcf.gz \
    --num_shards="${NUM_SHARDS}" \
    --regions="${REGION}" \
    --intermediate_results_dir=/output/intermediate \
    ${ONNX_FLAG}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========== Results"
echo "Total wall time: ${MINUTES}m ${SECONDS}s (${ELAPSED}s)"
echo "Region: ${REGION}"
echo "TF threads: ${TF_THREADS}"
echo "ONNX: ${USE_ONNX}"

# Quick variant count
echo ""
echo "========== Output"
docker run --rm -v "${OUTPUT_DIR}:/output" "${DOCKER_IMAGE}" \
  bash -c "zcat /output/HG003_arm64.vcf.gz | grep -v '^#' | wc -l" \
  2>/dev/null && echo " variants called" || true

echo ""
echo "========== Done =========="
