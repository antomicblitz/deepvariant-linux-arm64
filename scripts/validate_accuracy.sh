#!/bin/bash
set -euo pipefail

# Validate DeepVariant ARM64 VCF accuracy against GIAB truth set using hap.py.
#
# Usage:
#   bash scripts/validate_accuracy.sh \
#     --vcf output.vcf.gz \
#     --truth-vcf HG003_truth.vcf.gz \
#     --truth-bed HG003_truth.bed \
#     --ref GRCh38.chr20.fasta \
#     --output-dir results/
#
# Targets (from Google's published DeepVariant v1.9.0 numbers):
#   SNP F1   >= 0.9995
#   INDEL F1 >= 0.9945

VCF=""
TRUTH_VCF=""
TRUTH_BED=""
REF=""
OUTPUT_DIR="./happy_results"

while [[ $# -gt 0 ]]; do
  case $1 in
    --vcf) VCF="$2"; shift 2 ;;
    --truth-vcf) TRUTH_VCF="$2"; shift 2 ;;
    --truth-bed) TRUTH_BED="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "${VCF}" || -z "${TRUTH_VCF}" || -z "${TRUTH_BED}" || -z "${REF}" ]]; then
  echo "ERROR: All of --vcf, --truth-vcf, --truth-bed, --ref are required."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========== Accuracy Validation =========="
echo "VCF:       ${VCF}"
echo "Truth VCF: ${TRUTH_VCF}"
echo "Truth BED: ${TRUTH_BED}"
echo "Reference: ${REF}"
echo "Output:    ${OUTPUT_DIR}"
echo ""

# Run hap.py via Docker (pkrusche/hap.py supports ARM64)
echo "========== Running hap.py"
docker run --rm \
  -v "$(dirname "${VCF}"):/vcf" \
  -v "$(dirname "${TRUTH_VCF}"):/truth" \
  -v "$(dirname "${REF}"):/ref" \
  -v "${OUTPUT_DIR}:/output" \
  jmcdani20/hap.py:v0.3.12 \
  /opt/hap.py/bin/hap.py \
    "/truth/$(basename "${TRUTH_VCF}")" \
    "/vcf/$(basename "${VCF}")" \
    -f "/truth/$(basename "${TRUTH_BED}")" \
    -r "/ref/$(basename "${REF}")" \
    -o /output/happy \
    --engine=vcfeval \
    --threads="$(nproc)"

echo ""
echo "========== Results =========="

# Parse hap.py summary CSV
if [[ -f "${OUTPUT_DIR}/happy.summary.csv" ]]; then
  echo ""
  echo "--- hap.py Summary ---"
  # Print header and data rows
  head -1 "${OUTPUT_DIR}/happy.summary.csv"
  grep -E "^(SNP|INDEL)" "${OUTPUT_DIR}/happy.summary.csv"

  echo ""
  echo "--- F1 Score Check ---"

  # Extract F1 scores for SNP and INDEL
  SNP_F1=$(grep "^SNP" "${OUTPUT_DIR}/happy.summary.csv" | grep "PASS" | awk -F',' '{print $NF}')
  INDEL_F1=$(grep "^INDEL" "${OUTPUT_DIR}/happy.summary.csv" | grep "PASS" | awk -F',' '{print $NF}')

  echo "SNP F1:   ${SNP_F1:-N/A}"
  echo "INDEL F1: ${INDEL_F1:-N/A}"

  # Check against targets
  PASS=true
  if [[ -n "${SNP_F1}" ]]; then
    if (( $(echo "${SNP_F1} < 0.9995" | bc -l) )); then
      echo "WARNING: SNP F1 ${SNP_F1} is below target 0.9995"
      PASS=false
    else
      echo "SNP F1 PASS (>= 0.9995)"
    fi
  fi

  if [[ -n "${INDEL_F1}" ]]; then
    if (( $(echo "${INDEL_F1} < 0.9945" | bc -l) )); then
      echo "WARNING: INDEL F1 ${INDEL_F1} is below target 0.9945"
      PASS=false
    else
      echo "INDEL F1 PASS (>= 0.9945)"
    fi
  fi

  if [[ "${PASS}" == "true" ]]; then
    echo ""
    echo "VALIDATION PASSED: ARM64 build meets accuracy targets."
  else
    echo ""
    echo "VALIDATION FAILED: ARM64 build does not meet accuracy targets."
    exit 1
  fi
else
  echo "ERROR: hap.py summary file not found at ${OUTPUT_DIR}/happy.summary.csv"
  exit 1
fi
