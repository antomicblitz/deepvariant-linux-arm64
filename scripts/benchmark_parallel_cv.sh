#!/bin/bash
# Parallel call_variants benchmark
#
# Splits call_variants into N parallel workers, each processing a subset
# of make_examples shards. Postprocess merges the CVO outputs using the
# standard sharded file pattern (@N notation).
#
# Usage:
#   bash scripts/benchmark_parallel_cv.sh
#
# Prerequisites:
#   - Sequential baseline must be run first (via run_deepvariant)
#   - INT8 ONNX model at /data/model_int8_static.onnx
#   - Reference and BAM at standard /data/ paths
#
# Key findings (2026-03-06):
#   - Graviton4 4-way: CV 61s (2.10x speedup over 128s sequential)
#   - Graviton3 4-way: CV 74s (1.90x speedup over 141s sequential)
#   - Oracle A2 4-way: CV 114s (2.47x speedup over 283s sequential)
#   - Variant count matches sequential baseline exactly (207,799)
#
# Critical postprocess flags (discovered via debugging):
#   --regions must match the region used in make_examples
#   --small_model_cvo_records must point to make_examples' small model outputs
#   Without these, postprocess partitions across the whole genome and a
#   pre-existing CVO sanity check bug silently kills the chr20 partition
#   worker in multiprocessing.Pool (producing only ~125 RefCall variants).

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2}"
REF="/data/reference/GRCh38_no_alt_analysis_set.fasta"
BAM="/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
REGION="chr20"
NUM_SHARDS=32
RESULTS_DIR="/data/benchmark_results/parallel_cv_$(date +%Y%m%d_%H%M%S)"
USD_PER_HR="${USD_PER_HR:-1.36}"
# TF_ONEDNN: set to 0 for AmpereOne (SIGILL), 1 for Graviton
TF_ONEDNN="${TF_ONEDNN:-1}"

mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "Parallel call_variants Benchmark"
echo "CPUs: $(nproc), RAM: $(free -h | awk '/Mem:/{print $2}')"
echo "Date: $(date -u)"
echo "Results: ${RESULTS_DIR}"
echo "========================================"

# ============================================================
# Step 1: Full sequential pipeline (baseline)
# ============================================================
echo ""
echo "### Step 1: Sequential baseline (run_deepvariant) ###"

SEQ_DIR="${RESULTS_DIR}/sequential"
SEQ_INTER="${SEQ_DIR}/intermediate"
mkdir -p "${SEQ_DIR}"

SEQ_START=$(date +%s)

docker run --rm \
  -e DV_USE_JEMALLOC=1 \
  -e "TF_ENABLE_ONEDNN_OPTS=${TF_ONEDNN}" \
  -e DNNL_DEFAULT_FPMATH_MODE=BF16 \
  --memory=56g \
  -v /data:/data \
  "${IMAGE}" \
  /opt/deepvariant/bin/run_deepvariant \
    --model_type=WGS \
    --ref="${REF}" \
    --reads="${BAM}" \
    --output_vcf="${SEQ_DIR}/output.vcf.gz" \
    --intermediate_results_dir="${SEQ_INTER}" \
    --regions="${REGION}" \
    --num_shards="${NUM_SHARDS}" \
    --call_variants_extra_args="--batch_size=256,--use_onnx=true,--onnx_model=/data/model_int8_static.onnx" \
  2>&1 | tee "${SEQ_DIR}/run.log"

SEQ_END=$(date +%s)
SEQ_WALL=$((SEQ_END - SEQ_START))
echo "${SEQ_WALL}" > "${SEQ_DIR}/wall_seconds.txt"
echo "Sequential wall: ${SEQ_WALL}s"

SEQ_VARIANTS=$(zgrep -c -v '^#' "${SEQ_DIR}/output.vcf.gz" 2>/dev/null || echo "0")
echo "Sequential variants: ${SEQ_VARIANTS}"

SMALL_MODEL_CVO="${SEQ_INTER}/make_examples_call_variant_outputs.tfrecord@${NUM_SHARDS}.gz"

echo "sequential | wall=${SEQ_WALL}s | variants=${SEQ_VARIANTS}" >> "${RESULTS_DIR}/summary.txt"

# ============================================================
# Step 2: Parallel CV tests
# ============================================================
run_parallel_cv() {
  local label="$1"
  local num_workers="$2"
  local threads_per_worker=$(($(nproc) / num_workers))
  local outdir="${RESULTS_DIR}/${label}"
  mkdir -p "${outdir}"

  echo ""
  echo "### ${label}: ${num_workers} CV workers x ${threads_per_worker} threads ###"

  # Create per-worker shard directories with renumbered symlinks
  local shards_per_worker=$((NUM_SHARDS / num_workers))
  for ((w=0; w<num_workers; w++)); do
    local worker_dir="${outdir}/worker_${w}_shards"
    mkdir -p "${worker_dir}"

    local shard_start=$((w * shards_per_worker))
    local shard_end=$(( (w + 1) * shards_per_worker ))
    if [[ $w -eq $((num_workers - 1)) ]]; then
      shard_end=${NUM_SHARDS}
    fi
    local worker_shard_count=$((shard_end - shard_start))

    # Renumber shards: original shard S -> local index 0..N-1
    local local_idx=0
    for ((s=shard_start; s<shard_end; s++)); do
      ln -sf "$(printf "${SEQ_INTER}/make_examples.tfrecord-%05d-of-%05d.gz" "$s" "${NUM_SHARDS}")" \
             "$(printf "${worker_dir}/examples.tfrecord-%05d-of-%05d.gz" "$local_idx" "$worker_shard_count")"
      if [[ $local_idx -eq 0 ]]; then
        ln -sf "$(printf "${SEQ_INTER}/make_examples.tfrecord-%05d-of-%05d.gz.example_info.json" "$s" "${NUM_SHARDS}")" \
               "$(printf "${worker_dir}/examples.tfrecord-%05d-of-%05d.gz.example_info.json" 0 "$worker_shard_count")"
      fi
      local_idx=$((local_idx + 1))
    done

    echo "  Worker ${w}: shards ${shard_start}-$((shard_end - 1)) (${worker_shard_count} shards, ${threads_per_worker} threads)"
  done

  local cv_start=$(date +%s)

  # Launch all CV workers in parallel Docker containers
  # Output files use sharded naming so postprocess can glob them
  local pids=()
  for ((w=0; w<num_workers; w++)); do
    local worker_dir="${outdir}/worker_${w}_shards"
    local shard_start=$((w * shards_per_worker))
    local shard_end=$(( (w + 1) * shards_per_worker ))
    if [[ $w -eq $((num_workers - 1)) ]]; then shard_end=${NUM_SHARDS}; fi
    local worker_shard_count=$((shard_end - shard_start))

    # call_variants auto-detects sharded outfile and writes directly to it
    local outfile
    outfile=$(printf "%s/call_variants_output-%05d-of-%05d.tfrecord.gz" "${outdir}" "$w" "$num_workers")

    docker run --rm \
      -e "DV_USE_JEMALLOC=1" \
      -e "OMP_NUM_THREADS=${threads_per_worker}" \
      -e "OMP_PROC_BIND=false" \
      -e "OMP_PLACES=cores" \
      -e "TF_ENABLE_ONEDNN_OPTS=${TF_ONEDNN}" \
      --memory=$((56 / num_workers))g \
      -v /data:/data \
      "${IMAGE}" \
      /opt/deepvariant/bin/call_variants \
        --outfile="${outfile}" \
        --examples="${worker_dir}/examples.tfrecord@${worker_shard_count}.gz" \
        --checkpoint=/opt/models/wgs \
        --use_onnx \
        --onnx_model=/data/model_int8_static.onnx \
        --batch_size=256 \
      > "${outdir}/cv_worker${w}.log" 2>&1 &
    pids+=($!)
  done

  # Wait for all workers
  local all_ok=true
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      echo "  ERROR: CV worker PID ${pid} failed" >&2
      all_ok=false
    fi
  done
  if [[ "${all_ok}" != "true" ]]; then
    echo "ERROR: One or more CV workers failed." >&2
    for ((w=0; w<num_workers; w++)); do
      echo "--- Worker ${w} (last 20 lines) ---" >&2
      tail -20 "${outdir}/cv_worker${w}.log" >&2
    done
    return 1
  fi

  local cv_end=$(date +%s)
  local cv_time=$((cv_end - cv_start))
  echo "  CV wall: ${cv_time}s"
  echo "${cv_time}" > "${outdir}/cv_seconds.txt"

  # Postprocess: merge CV outputs using @N sharded pattern
  # CRITICAL: --regions and --small_model_cvo_records must be passed
  local pp_start=$(date +%s)

  docker run --rm \
    --memory=56g \
    -v /data:/data \
    "${IMAGE}" \
    /opt/deepvariant/bin/postprocess_variants \
      --ref="${REF}" \
      --infile="${outdir}/call_variants_output@${num_workers}.tfrecord.gz" \
      --outfile="${outdir}/output.vcf.gz" \
      --cpus="${NUM_SHARDS}" \
      --small_model_cvo_records="${SMALL_MODEL_CVO}" \
      --regions="${REGION}" \
    2>&1 | tee "${outdir}/pp.log"

  local pp_end=$(date +%s)
  local pp_time=$((pp_end - pp_start))
  local total_cv_pp=$((cv_time + pp_time))

  local par_variants
  par_variants=$(zgrep -c -v '^#' "${outdir}/output.vcf.gz" 2>/dev/null || echo "0")
  echo "  PP=${pp_time}s  CV+PP=${total_cv_pp}s  variants=${par_variants}"

  if [[ "${par_variants}" -eq "${SEQ_VARIANTS}" ]]; then
    echo "  Variant count MATCHES sequential baseline"
  else
    echo "  WARNING: Variant count mismatch! (${par_variants} vs ${SEQ_VARIANTS})"
  fi

  echo "${label} | workers=${num_workers} | CV=${cv_time}s PP=${pp_time}s | CV+PP=${total_cv_pp}s | variants=${par_variants}" \
    | tee -a "${RESULTS_DIR}/summary.txt"
}

# Run parallel CV tests
run_parallel_cv "parallel_2w_run1" 2
run_parallel_cv "parallel_2w_run2" 2
run_parallel_cv "parallel_4w_run1" 4
run_parallel_cv "parallel_4w_run2" 4

echo ""
echo "========================================"
echo "ALL PARALLEL CV BENCHMARKS COMPLETE"
echo "========================================"
cat "${RESULTS_DIR}/summary.txt"
