#!/usr/bin/env bash
# DeepVariant v1.9 — GPU Quicktest (macOS native, Apple Silicon)
#
# Verifies that the installation works end-to-end by running a small variant
# calling job on a 10kb region of chr20. Tests Metal GPU detection,
# make_examples, call_variants, and postprocess_variants.
#
# Usage:
#   ./scripts/quicktest.sh
#   # or after installation:
#   $DEEPVARIANT_HOME/scripts/quicktest.sh

set -euo pipefail

DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"

# ── Activate Python environment if not already active ─────────────────────
# When run directly (not via deepvariant-quicktest wrapper), the conda/venv
# environment won't be active. Detect and activate it automatically.
if [[ -f "$DV_HOME/.env_type" ]]; then
  ENV_INFO=$(cat "$DV_HOME/.env_type")
  if [[ "$ENV_INFO" == venv && -f "$DV_HOME/venv/bin/activate" ]]; then
    source "$DV_HOME/venv/bin/activate"
  elif [[ "$ENV_INFO" == conda:* ]]; then
    CONDA_ENV_NAME="${ENV_INFO#conda:}"
    # Find the conda env prefix and prepend to PATH (avoids needing conda init)
    for cmd in conda mamba micromamba; do
      if command -v "$cmd" &>/dev/null; then
        CONDA_PREFIX_QT=$($cmd env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')
        if [[ -n "$CONDA_PREFIX_QT" && -d "$CONDA_PREFIX_QT/bin" ]]; then
          export PATH="$CONDA_PREFIX_QT/bin:$PATH"
        fi
        break
      fi
    done
  fi
fi

DATA_DIR="$HOME/deepvariant-quicktest"
REGION="chr20:10000000-10010000"
SHARDS=10

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
pass() { echo -e "${GREEN}✓${NC}  $*"; }
fail() { echo -e "${RED}✗${NC}  $*"; exit 1; }

echo "================================================"
echo "  DeepVariant v1.9 — GPU Quicktest"
echo "================================================"
echo ""

# ── [ 1/4 ] Metal GPU detection ─────────────────────────────────────────────
echo "[ 1/4 ]  Metal GPU detection"
GPU_COUNT=$(python3 -c "
import tensorflow as tf
gpus = [d for d in tf.config.list_physical_devices() if 'GPU' in d.device_type]
print(len(gpus))
" 2>/dev/null || echo 0)

if [[ "$GPU_COUNT" -ge 1 ]]; then
    pass "Metal GPU ENABLED — METAL ($GPU_COUNT device)"
else
    fail "No Metal GPU detected. Check TF-Metal installation."
fi
echo ""

# ── [ 2/4 ] Download / cache test data ──────────────────────────────────────
echo "[ 2/4 ]  Downloading quickstart test data"
mkdir -p "$DATA_DIR"

BASE_URL="https://storage.googleapis.com/deepvariant/quickstart-testdata"
FILES=(
    "ucsc.hg19.chr20.unittest.fasta"
    "ucsc.hg19.chr20.unittest.fasta.fai"
    "ucsc.hg19.chr20.unittest.fasta.gz"
    "ucsc.hg19.chr20.unittest.fasta.gz.fai"
    "ucsc.hg19.chr20.unittest.fasta.gz.gzi"
    "NA12878_S1.chr20.10_10p1mb.bam"
    "NA12878_S1.chr20.10_10p1mb.bam.bai"
)

for f in "${FILES[@]}"; do
    if [[ -f "$DATA_DIR/$f" ]]; then
        echo "         $f  (cached)"
    else
        echo "         $f  (downloading...)"
        curl -sSL "$BASE_URL/$f" -o "$DATA_DIR/$f" || fail "Failed to download $f"
    fi
done
echo ""

# ── [ 3/4 ] Run DeepVariant ──────────────────────────────────────────────────
echo "[ 3/4 ]  Running DeepVariant  (region: ${REGION}  |  shards: $SHARDS)"
echo "         TF Metal messages enabled — look for 'Metal device set to:' below"
echo ""

TMPDIR_RUN=$(mktemp -d)
trap 'rm -rf "$TMPDIR_RUN"' EXIT

REF="$DATA_DIR/ucsc.hg19.chr20.unittest.fasta"
BAM="$DATA_DIR/NA12878_S1.chr20.10_10p1mb.bam"
EXAMPLES="$TMPDIR_RUN/make_examples.tfrecord@${SHARDS}.gz"
GVCF_RECORDS="$TMPDIR_RUN/gvcf.tfrecord@${SHARDS}.gz"
CV_OUTPUT="$TMPDIR_RUN/callvariantsoutput.tfrecord.gz"
OUT_VCF="$DATA_DIR/output.vcf.gz"
OUT_GVCF="$DATA_DIR/output.g.vcf.gz"

export TF_CPP_MIN_LOG_LEVEL=0
export TF2_BEHAVIOR=1
export TPU_ML_PLATFORM=Tensorflow

# ── Step 1: make_examples ────────────────────────────────────────────────────
echo "***** Running make_examples *****"
time seq 0 $((SHARDS - 1)) | parallel -q --halt 2 --line-buffer \
    "$DV_HOME/bin/make_examples" \
        --mode calling \
        --ref "$REF" \
        --reads "$BAM" \
        --examples "$EXAMPLES" \
        --checkpoint "$DV_HOME/models/wgs" \
        --gvcf "$GVCF_RECORDS" \
        --regions "$REGION" \
        --track_ref_reads \
        --task {}

# ── Verify make_examples output ───────────────────────────────────────────────
echo ""
echo "***** Verifying make_examples outputs *****"
EXAMPLES_COUNT=$(ls "$TMPDIR_RUN"/make_examples.tfrecord-* 2>/dev/null | wc -l | tr -d ' ')
echo "         make_examples shards written: $EXAMPLES_COUNT / $SHARDS"
[[ "$EXAMPLES_COUNT" -gt 0 ]] || fail "No make_examples shards found in $TMPDIR_RUN"

# ── Step 2: call_variants ────────────────────────────────────────────────────
echo ""
echo "***** Running call_variants *****"
time "$DV_HOME/bin/call_variants" \
    --outfile "$CV_OUTPUT" \
    --examples "$EXAMPLES" \
    --checkpoint "$DV_HOME/models/wgs"

# ── Step 3: postprocess_variants ─────────────────────────────────────────────
# --cpus 1 is REQUIRED for small regions. Without it, postprocess_variants
# auto-detects all CPU cores and creates that many genomic partitions.
# With only ~84 variants in a 10kb region, most partitions produce empty VCF
# files that bcftools naive_concat cannot parse, causing a segfault.
# --cpus 1 forces single-partition mode, bypassing _concat_vcf entirely.
echo ""
echo "***** Running postprocess_variants (--cpus 1) *****"
time "$DV_HOME/bin/postprocess_variants" \
    --ref "$REF" \
    --infile "$CV_OUTPUT" \
    --outfile "$OUT_VCF" \
    --gvcf_outfile "$OUT_GVCF" \
    --nonvariant_site_tfrecord_path "$GVCF_RECORDS" \
    --cpus 1

# ── [ 4/4 ] Results ───────────────────────────────────────────────────────────
echo ""
echo "[ 4/4 ]  Results"
if [[ -f "$OUT_VCF" ]]; then
    VARIANT_COUNT=$(zcat "$OUT_VCF" 2>/dev/null | grep -v '^#' | wc -l | tr -d ' ')
    pass "VCF confirmed — $VARIANT_COUNT variants called"
    echo "         Output: $OUT_VCF"
    echo "         gVCF:   $OUT_GVCF"
else
    fail "Output VCF not found: $OUT_VCF"
fi

echo ""
echo "================================================"
echo "  PASSED — DeepVariant GPU quicktest complete"
echo "================================================"
