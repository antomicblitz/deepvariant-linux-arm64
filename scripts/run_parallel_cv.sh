#!/bin/bash
# run_parallel_cv.sh — DeepVariant with parallel call_variants
#
# Drop-in replacement for run_deepvariant that splits call_variants into
# N parallel workers for 1.9-2.5x CV speedup on 32+ vCPU machines.
#
# Runs INSIDE the Docker container (not on the host). Use it as:
#
#   docker run ... ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.4 \
#     /opt/deepvariant/scripts/run_parallel_cv.sh \
#     --model_type=WGS \
#     --ref=/data/reference.fasta \
#     --reads=/data/input.bam \
#     --output_vcf=/data/output.vcf.gz \
#     --num_shards=32 \
#     --num_cv_workers=4
#
# Requirements:
#   - ONNX backend only (TF SavedModel uses ~26 GB per process — would OOM)
#   - num_shards must be divisible by num_cv_workers (auto mode handles this)
#   - Auto-scales: ~8 threads/worker, 1 worker on <16 vCPU, scales to nproc/8
#
# How it works:
#   1. Runs make_examples (same as run_deepvariant)
#   2. Splits ME output shards into N groups via symlinks
#   3. Launches N call_variants processes with scoped OMP_NUM_THREADS
#   4. Runs postprocess_variants on merged CVO outputs
#
# Critical: postprocess_variants MUST receive --regions and
# --small_model_cvo_records flags, or it silently produces wrong output.

set -euo pipefail

# ============================================================
# Argument parsing
# ============================================================
MODEL_TYPE=""
REF=""
READS=""
OUTPUT_VCF=""
NUM_SHARDS=""
NUM_CV_WORKERS="auto"
REGIONS=""
INTERMEDIATE_RESULTS_DIR=""
BATCH_SIZE=256
ONNX_MODEL=""
CUSTOMIZED_MODEL=""
SAMPLE_NAME=""
OUTPUT_GVCF=""
POSTPROCESS_CPUS=""
NOCOMPRESS=false

print_usage() {
  echo "Usage: run_parallel_cv.sh [options]"
  echo ""
  echo "Required:"
  echo "  --model_type=TYPE        WGS, WES, PACBIO, ONT_R104, etc."
  echo "  --ref=PATH               Reference FASTA"
  echo "  --reads=PATH             Input BAM/CRAM"
  echo "  --output_vcf=PATH        Output VCF"
  echo "  --num_shards=N           Number of make_examples shards"
  echo ""
  echo "Optional:"
  echo "  --num_cv_workers=N       Parallel CV workers (default: auto — nproc/8, min 1)"
  echo "  --regions=REGION         Genomic region (e.g., chr20)"
  echo "  --intermediate_results_dir=PATH"
  echo "  --batch_size=N           CV batch size (default: 256)"
  echo "  --onnx_model=PATH        Custom ONNX model path"
  echo "  --customized_model=PATH  Custom model checkpoint"
  echo "  --sample_name=NAME       Sample name for VCF header"
  echo "  --output_gvcf=PATH       Output gVCF (optional)"
  echo "  --postprocess_cpus=N     CPUs for postprocess (default: num_shards)"
  echo "  --nocompress             Write uncompressed TFRecords (saves ~8% ME CPU)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_type=*) MODEL_TYPE="${1#*=}" ;;
    --ref=*) REF="${1#*=}" ;;
    --reads=*) READS="${1#*=}" ;;
    --output_vcf=*) OUTPUT_VCF="${1#*=}" ;;
    --num_shards=*) NUM_SHARDS="${1#*=}" ;;
    --num_cv_workers=*) NUM_CV_WORKERS="${1#*=}" ;;
    --regions=*) REGIONS="${1#*=}" ;;
    --intermediate_results_dir=*) INTERMEDIATE_RESULTS_DIR="${1#*=}" ;;
    --batch_size=*) BATCH_SIZE="${1#*=}" ;;
    --onnx_model=*) ONNX_MODEL="${1#*=}" ;;
    --customized_model=*) CUSTOMIZED_MODEL="${1#*=}" ;;
    --sample_name=*) SAMPLE_NAME="${1#*=}" ;;
    --output_gvcf=*) OUTPUT_GVCF="${1#*=}" ;;
    --postprocess_cpus=*) POSTPROCESS_CPUS="${1#*=}" ;;
    --nocompress) NOCOMPRESS=true ;;
    --help|-h) print_usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; print_usage; exit 1 ;;
  esac
  shift
done

# Validate required args
for var_name in MODEL_TYPE REF READS OUTPUT_VCF NUM_SHARDS; do
  if [[ -z "${!var_name}" ]]; then
    echo "ERROR: --$(echo "$var_name" | tr 'A-Z' 'a-z') is required." >&2
    print_usage
    exit 1
  fi
done

# Detect available RAM (respects Docker --memory limit via cgroup).
# Each INT8 ONNX CV worker uses ~3 GB RSS. Reserve 4 GB for ME/PP/OS.
get_ram_gb() {
  local bytes
  # cgroup v2
  if [[ -f /sys/fs/cgroup/memory.max ]]; then
    bytes=$(cat /sys/fs/cgroup/memory.max)
    if [[ "$bytes" != "max" ]]; then
      echo $(( bytes / 1073741824 ))
      return
    fi
  fi
  # cgroup v1
  if [[ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]]; then
    bytes=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    if [[ $bytes -lt 9000000000000000000 ]]; then
      echo $(( bytes / 1073741824 ))
      return
    fi
  fi
  # Fallback: host total memory
  awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo
}

# Auto-scale CV workers if not explicitly set.
# Target ~8 threads per worker (GEMM sweet spot; saturates at ~16).
# Cap by available RAM (~3 GB per worker + 4 GB overhead).
# Finds the largest worker count that evenly divides NUM_SHARDS.
if [[ "$NUM_CV_WORKERS" == "auto" ]]; then
  NCPU=$(nproc)
  RAM_GB=$(get_ram_gb)
  MAX_BY_CPU=$(( NCPU / 8 ))
  MAX_BY_RAM=$(( (RAM_GB - 4) / 3 ))
  [[ $MAX_BY_CPU -lt 1 ]] && MAX_BY_CPU=1
  [[ $MAX_BY_RAM -lt 1 ]] && MAX_BY_RAM=1
  DESIRED=$MAX_BY_CPU
  [[ $MAX_BY_RAM -lt $DESIRED ]] && DESIRED=$MAX_BY_RAM
  NUM_CV_WORKERS=1
  for ((w=DESIRED; w>=2; w--)); do
    if [[ $((NUM_SHARDS % w)) -eq 0 ]]; then
      NUM_CV_WORKERS=$w
      break
    fi
  done
  if [[ $RAM_GB -lt 7 ]]; then
    echo "WARNING: Only ${RAM_GB} GB RAM detected. Minimum ~7 GB needed (4 GB overhead + 3 GB per CV worker)." >&2
    echo "  Consider increasing Docker --memory or using sequential mode (run_deepvariant)." >&2
  fi
  echo "Auto-scaled: ${NCPU} vCPU, ${RAM_GB} GB RAM → ${NUM_CV_WORKERS} CV workers (cpu-max=${MAX_BY_CPU}, ram-max=${MAX_BY_RAM})"
fi

# Validate num_cv_workers
if [[ "$NUM_CV_WORKERS" -lt 1 ]]; then
  echo "ERROR: --num_cv_workers must be >= 1" >&2
  exit 1
fi

if [[ $((NUM_SHARDS % NUM_CV_WORKERS)) -ne 0 ]]; then
  echo "ERROR: --num_shards ($NUM_SHARDS) must be divisible by --num_cv_workers ($NUM_CV_WORKERS)" >&2
  exit 1
fi

# Resolve model checkpoint path
if [[ -n "$CUSTOMIZED_MODEL" ]]; then
  MODEL_CKPT="$CUSTOMIZED_MODEL"
else
  case "$MODEL_TYPE" in
    WGS) MODEL_CKPT="/opt/models/wgs" ;;
    WES) MODEL_CKPT="/opt/models/wes" ;;
    PACBIO) MODEL_CKPT="/opt/models/pacbio" ;;
    ONT_R104) MODEL_CKPT="/opt/models/ont_r104" ;;
    HYBRID_PACBIO_ILLUMINA) MODEL_CKPT="/opt/models/hybrid_pacbio_illumina" ;;
    MASSEQ) MODEL_CKPT="/opt/models/masseq" ;;
    *) echo "ERROR: Unknown model_type: $MODEL_TYPE" >&2; exit 1 ;;
  esac
fi

# Resolve ONNX model path
if [[ -z "$ONNX_MODEL" ]]; then
  ONNX_MODEL="${MODEL_CKPT}/model.onnx"
fi
if [[ ! -f "$ONNX_MODEL" ]]; then
  echo "ERROR: ONNX model not found at $ONNX_MODEL" >&2
  echo "Parallel CV requires the ONNX backend (TF SavedModel uses ~26 GB per process)." >&2
  exit 1
fi

# Set up intermediate directory
if [[ -z "$INTERMEDIATE_RESULTS_DIR" ]]; then
  INTERMEDIATE_RESULTS_DIR=$(mktemp -d)
fi
mkdir -p "$INTERMEDIATE_RESULTS_DIR"

# Set file extension based on compression mode
if [[ "$NOCOMPRESS" == "true" ]]; then
  GZ_EXT=""
else
  GZ_EXT=".gz"
fi

THREADS_PER_WORKER=$(( $(nproc) / NUM_CV_WORKERS ))
SHARDS_PER_WORKER=$(( NUM_SHARDS / NUM_CV_WORKERS ))

echo "========================================"
echo "DeepVariant Parallel call_variants"
echo "  Model:       $MODEL_TYPE"
echo "  Shards:      $NUM_SHARDS"
echo "  CV workers:  $NUM_CV_WORKERS (${THREADS_PER_WORKER} threads each)"
echo "  Batch size:  $BATCH_SIZE"
echo "  ONNX model:  $ONNX_MODEL"
echo "  Intermediate: $INTERMEDIATE_RESULTS_DIR"
echo "========================================"

# ============================================================
# Step 1: make_examples
# ============================================================
echo ""
echo "=== Step 1/3: make_examples ==="

EXAMPLES="${INTERMEDIATE_RESULTS_DIR}/make_examples.tfrecord@${NUM_SHARDS}${GZ_EXT}"
SMALL_MODEL_CVO="${INTERMEDIATE_RESULTS_DIR}/make_examples_call_variant_outputs.tfrecord@${NUM_SHARDS}${GZ_EXT}"

# Build make_examples command. Uses GNU parallel to launch NUM_SHARDS workers,
# each handling one shard (--task N). This matches how run_deepvariant.py works.
ME_BASE_ARGS=(
  "--mode" "calling"
  "--ref" "$REF"
  "--reads" "$READS"
  "--examples" "$EXAMPLES"
  "--checkpoint" "$MODEL_CKPT"
)
if [[ -n "$REGIONS" ]]; then
  ME_BASE_ARGS+=("--regions" "$REGIONS")
fi
if [[ -n "$SAMPLE_NAME" ]]; then
  ME_BASE_ARGS+=("--sample_name" "$SAMPLE_NAME")
fi
if [[ -n "$OUTPUT_GVCF" ]]; then
  GVCF_TFRECORD="${INTERMEDIATE_RESULTS_DIR}/gvcf.tfrecord@${NUM_SHARDS}${GZ_EXT}"
  ME_BASE_ARGS+=("--gvcf" "$GVCF_TFRECORD")
fi

# Small model config — WGS, PACBIO, ONT_R104 use small models for filtering.
# This creates make_examples_call_variant_outputs.tfrecord@N.gz files that
# postprocess_variants needs via --small_model_cvo_records.
USE_SMALL_MODEL=false
case "$MODEL_TYPE" in
  WGS)
    USE_SMALL_MODEL=true
    ME_BASE_ARGS+=("--call_small_model_examples=true"
                   "--trained_small_model_path=/opt/smallmodels/wgs"
                   "--track_ref_reads=true"
                   "--small_model_snp_gq_threshold=20"
                   "--small_model_indel_gq_threshold=28"
                   "--small_model_vaf_context_window_size=51") ;;
  PACBIO)
    USE_SMALL_MODEL=true
    ME_BASE_ARGS+=("--call_small_model_examples=true"
                   "--trained_small_model_path=/opt/smallmodels/pacbio"
                   "--track_ref_reads=true"
                   "--small_model_snp_gq_threshold=15"
                   "--small_model_indel_gq_threshold=16"
                   "--small_model_vaf_context_window_size=51") ;;
  ONT_R104)
    USE_SMALL_MODEL=true
    ME_BASE_ARGS+=("--call_small_model_examples=true"
                   "--trained_small_model_path=/opt/smallmodels/ont_r104"
                   "--track_ref_reads=true"
                   "--small_model_snp_gq_threshold=9"
                   "--small_model_indel_gq_threshold=17"
                   "--small_model_vaf_context_window_size=51") ;;
esac

# Strip OMP vars for make_examples (they cause ~10% regression)
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES 2>/dev/null || true

ME_START=$(date +%s)
seq 0 $((NUM_SHARDS - 1)) | \
  parallel -q --halt 2 --line-buffer \
  /opt/deepvariant/bin/make_examples "${ME_BASE_ARGS[@]}" --task {}
ME_END=$(date +%s)
ME_TIME=$((ME_END - ME_START))
echo "make_examples: ${ME_TIME}s"

# ============================================================
# Step 2: parallel call_variants
# ============================================================
echo ""
echo "=== Step 2/3: call_variants (${NUM_CV_WORKERS}-way parallel) ==="

CV_START=$(date +%s)

# Create per-worker shard directories with renumbered symlinks
for ((w=0; w<NUM_CV_WORKERS; w++)); do
  WORKER_DIR="${INTERMEDIATE_RESULTS_DIR}/cv_worker_${w}"
  mkdir -p "$WORKER_DIR"

  SHARD_START=$((w * SHARDS_PER_WORKER))
  SHARD_END=$(( (w + 1) * SHARDS_PER_WORKER ))

  LOCAL_IDX=0
  for ((s=SHARD_START; s<SHARD_END; s++)); do
    SRC=$(printf "%s/make_examples.tfrecord-%05d-of-%05d%s" "$INTERMEDIATE_RESULTS_DIR" "$s" "$NUM_SHARDS" "$GZ_EXT")
    DST=$(printf "%s/examples.tfrecord-%05d-of-%05d%s" "$WORKER_DIR" "$LOCAL_IDX" "$SHARDS_PER_WORKER" "$GZ_EXT")
    ln -sf "$SRC" "$DST"

    # Copy example_info.json for shard 0 of each worker
    if [[ $LOCAL_IDX -eq 0 ]]; then
      SRC_INFO=$(printf "%s/make_examples.tfrecord-%05d-of-%05d%s.example_info.json" "$INTERMEDIATE_RESULTS_DIR" "$s" "$NUM_SHARDS" "$GZ_EXT")
      DST_INFO=$(printf "%s/examples.tfrecord-%05d-of-%05d%s.example_info.json" "$WORKER_DIR" 0 "$SHARDS_PER_WORKER" "$GZ_EXT")
      if [[ -f "$SRC_INFO" ]]; then
        ln -sf "$SRC_INFO" "$DST_INFO"
      fi
    fi
    LOCAL_IDX=$((LOCAL_IDX + 1))
  done
done

# Cap OMP_NUM_THREADS per worker to prevent thread oversubscription.
# Without this, each worker defaults to $(nproc) threads — on a 16-vCPU
# machine with 4 workers, that's 64 threads competing for 16 cores.
export OMP_NUM_THREADS=$THREADS_PER_WORKER
export OMP_PROC_BIND=false
export OMP_PLACES=cores

# Launch all workers in parallel
PIDS=()
for ((w=0; w<NUM_CV_WORKERS; w++)); do
  WORKER_DIR="${INTERMEDIATE_RESULTS_DIR}/cv_worker_${w}"
  CV_OUTFILE=$(printf "%s/call_variants_output-%05d-of-%05d.tfrecord%s" \
    "$INTERMEDIATE_RESULTS_DIR" "$w" "$NUM_CV_WORKERS" "$GZ_EXT")

  /opt/deepvariant/bin/call_variants \
    --outfile="$CV_OUTFILE" \
    --examples="${WORKER_DIR}/examples.tfrecord@${SHARDS_PER_WORKER}${GZ_EXT}" \
    --checkpoint="$MODEL_CKPT" \
    --use_onnx \
    --onnx_model="$ONNX_MODEL" \
    --batch_size="$BATCH_SIZE" \
    > "${INTERMEDIATE_RESULTS_DIR}/cv_worker_${w}.log" 2>&1 &
  PIDS+=($!)
  echo "  Worker $w: PID $!, shards $((w * SHARDS_PER_WORKER))-$(( (w+1) * SHARDS_PER_WORKER - 1 )), threads=$THREADS_PER_WORKER"
done

# Wait for all workers
FAILED=0
for ((w=0; w<NUM_CV_WORKERS; w++)); do
  if ! wait "${PIDS[$w]}"; then
    echo "ERROR: CV worker $w (PID ${PIDS[$w]}) failed:" >&2
    tail -20 "${INTERMEDIATE_RESULTS_DIR}/cv_worker_${w}.log" >&2
    FAILED=1
  fi
done
if [[ $FAILED -ne 0 ]]; then
  echo "ERROR: One or more call_variants workers failed. See logs above." >&2
  exit 1
fi

CV_END=$(date +%s)
CV_TIME=$((CV_END - CV_START))
echo "call_variants (${NUM_CV_WORKERS}-way): ${CV_TIME}s"

# ============================================================
# Step 3: postprocess_variants
# ============================================================
echo ""
echo "=== Step 3/3: postprocess_variants ==="

PP_CPUS="${POSTPROCESS_CPUS:-$NUM_SHARDS}"
CV_MERGED="${INTERMEDIATE_RESULTS_DIR}/call_variants_output@${NUM_CV_WORKERS}.tfrecord${GZ_EXT}"

PP_CMD=("/opt/deepvariant/bin/postprocess_variants"
  "--ref=$REF"
  "--infile=$CV_MERGED"
  "--outfile=$OUTPUT_VCF"
  "--cpus=$PP_CPUS"
)
if [[ "$USE_SMALL_MODEL" == "true" ]]; then
  PP_CMD+=("--small_model_cvo_records=$SMALL_MODEL_CVO")
fi
if [[ -n "$REGIONS" ]]; then
  PP_CMD+=("--regions=$REGIONS")
fi
if [[ -n "$OUTPUT_GVCF" ]]; then
  PP_CMD+=("--nonvariant_site_tfrecord_path=$GVCF_TFRECORD"
           "--gvcf_outfile=$OUTPUT_GVCF")
fi

# Strip OMP vars for postprocess
unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES 2>/dev/null || true

PP_START=$(date +%s)
"${PP_CMD[@]}"
PP_END=$(date +%s)
PP_TIME=$((PP_END - PP_START))
echo "postprocess_variants: ${PP_TIME}s"

# ============================================================
# Summary
# ============================================================
TOTAL=$((ME_TIME + CV_TIME + PP_TIME))
echo ""
echo "========================================"
echo "COMPLETE"
echo "  make_examples:      ${ME_TIME}s"
echo "  call_variants:      ${CV_TIME}s (${NUM_CV_WORKERS}-way parallel)"
echo "  postprocess:        ${PP_TIME}s"
echo "  Total:              ${TOTAL}s"
echo "  Output:             $OUTPUT_VCF"
echo "========================================"
