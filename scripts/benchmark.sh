#!/usr/bin/env bash
# DeepVariant v1.9 — Performance Benchmark (macOS ARM64, Apple Silicon)
#
# Benchmarks DeepVariant on HG003 chr20, comparing your Apple Silicon Mac
# against published reference metrics from GCP instances. Optionally evaluates
# accuracy with hap.py against NIST truth sets.
#
# TensorFlow Metal GPU is available on all Apple Silicon Macs but provides
# minimal speedup for call_variants in v1.9 due to the "small model"
# optimization that pre-screens easy variants on CPU. The primary value of
# running on Apple Silicon is convenience and per-core efficiency, not GPU
# acceleration.
#
# Usage:
#   bash scripts/benchmark.sh [--runs N] [--skip-happy] [--shards N]
#                              [--batch-size N] [--output-dir DIR]

set -euo pipefail

# ── Prevent macOS sleep ──────────────────────────────────────────────────────
# caffeinate -i prevents idle sleep; -w $$ ties it to this script's lifetime.
# Runs in the background and auto-exits when the script finishes.
caffeinate -i -w $$ &

# ── Defaults ──────────────────────────────────────────────────────────────────
DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
NUM_RUNS=1
SKIP_HAPPY=false
SHARDS=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || sysctl -n hw.logicalcpu)
BATCH_SIZE=1024
OUTPUT_DIR="$HOME/deepvariant-benchmark"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; BOLD='\033[1m'
NC='\033[0m'
info()  { echo -e "${BLUE}==>${NC} $*"; }
pass()  { echo -e "${GREEN}✓${NC}  $*"; }
fail()  { echo -e "${RED}✗${NC}  $*"; exit 1; }
banner() { echo -e "\n${BOLD}$*${NC}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
  cat <<EOF
Usage: bash scripts/benchmark.sh [OPTIONS]

Benchmarks DeepVariant on HG003 chr20 (Apple Silicon, macOS ARM64)
and compares against published GCP reference metrics.

Options:
  --runs N            Number of runs for averaging (default: 1)
  --skip-happy        Skip hap.py accuracy evaluation (no Docker needed)
  --shards N          Parallel shards for make_examples (default: perf cores)
  --batch-size N      Batch size for call_variants (default: 1024)
  --output-dir DIR    Output directory (default: ~/deepvariant-benchmark)
  --help              Show this help
EOF
  exit 0
}

while (( "$#" )); do
  case "$1" in
    --runs)        NUM_RUNS="$2"; shift 2 ;;
    --skip-happy)  SKIP_HAPPY=true; shift ;;
    --shards)      SHARDS="$2"; shift 2 ;;
    --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
    --output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
    --help)        usage ;;
    *)             echo "Unknown option: $1"; usage ;;
  esac
done

# ── Activate Python environment ───────────────────────────────────────────────
if [[ -f "$DV_HOME/.env_type" ]]; then
  ENV_INFO=$(cat "$DV_HOME/.env_type")
  if [[ "$ENV_INFO" == venv && -f "$DV_HOME/venv/bin/activate" ]]; then
    source "$DV_HOME/venv/bin/activate"
  elif [[ "$ENV_INFO" == conda:* ]]; then
    CONDA_ENV_NAME="${ENV_INFO#conda:}"
    for cmd in conda mamba micromamba; do
      if command -v "$cmd" &>/dev/null; then
        CONDA_PREFIX_BM=$($cmd env list 2>/dev/null | grep "^${CONDA_ENV_NAME} " | awk '{print $NF}')
        if [[ -n "$CONDA_PREFIX_BM" && -d "$CONDA_PREFIX_BM/bin" ]]; then
          export PATH="$CONDA_PREFIX_BM/bin:$PATH"
        fi
        break
      fi
    done
  fi
fi

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR="$OUTPUT_DIR/data"
mkdir -p "$DATA_DIR/reference" "$DATA_DIR/input" "$DATA_DIR/benchmark"

REF="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta"
REF_FAI="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.fasta.fai"
BAM="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
BAM_BAI="$DATA_DIR/input/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai"
TRUTH_VCF="$DATA_DIR/benchmark/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TRUTH_VCF_TBI="$DATA_DIR/benchmark/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi"
TRUTH_BED="$DATA_DIR/benchmark/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"

RESULTS_JSONL="$OUTPUT_DIR/benchmark_runs.jsonl"
RESULTS_JSON="$OUTPUT_DIR/benchmark_results.json"

# ── Banner ────────────────────────────────────────────────────────────────────
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
echo ""
echo "================================================================"
echo "  DeepVariant v1.9 — Performance Benchmark"
echo "================================================================"
echo ""
echo "  Chip:         $CHIP"
echo "  Output dir:   $OUTPUT_DIR"
echo "  Shards:       $SHARDS"
echo "  Batch size:   $BATCH_SIZE"
echo "  Runs:         $NUM_RUNS"
echo "  hap.py:       $(if $SKIP_HAPPY; then echo SKIP; else echo YES; fi)"
echo ""

# ── [ 1 ] Download data ──────────────────────────────────────────────────────
banner "[ 1 ] Downloading benchmark data"

download_file() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  cached: $(basename "$dest")"
  else
    echo "  downloading: $(basename "$dest") ..."
    curl -sSL "$url" -o "$dest" || fail "Failed to download $(basename "$dest")"
  fi
}

# Reference genome (~800 MB compressed → ~3 GB uncompressed)
REF_FTP="ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids"
if [[ ! -f "$REF" ]]; then
  echo "  downloading: GRCh38 reference (gunzipping on the fly, ~3 GB) ..."
  curl -sSL "${REF_FTP}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz" \
    | gunzip > "$REF" || fail "Failed to download reference genome"
else
  echo "  cached: GRCh38_no_alt_analysis_set.fasta"
fi
download_file "${REF_FTP}/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.fai" "$REF_FAI"

# HG003 chr20 BAM (~1.6 GB)
BAM_URL="https://storage.googleapis.com/deepvariant/case-study-testdata"
download_file "${BAM_URL}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam" "$BAM"
download_file "${BAM_URL}/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam.bai" "$BAM_BAI"

# Truth sets (for hap.py)
if [[ "$SKIP_HAPPY" != "true" ]]; then
  GIAB_FTP="ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG003_NA24149_father/NISTv4.2.1/GRCh38"
  download_file "${GIAB_FTP}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz" "$TRUTH_VCF"
  download_file "${GIAB_FTP}/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz.tbi" "$TRUTH_VCF_TBI"
  download_file "${GIAB_FTP}/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed" "$TRUTH_BED"
fi

pass "Data ready"
echo ""

# ── [ 2 ] System info ────────────────────────────────────────────────────────
banner "[ 2 ] System info"
GPU_COUNT=$(python3 -c "
import tensorflow as tf
gpus = [d for d in tf.config.list_physical_devices() if 'GPU' in d.device_type]
print(len(gpus))
" 2>/dev/null || echo 0)

if [[ "$GPU_COUNT" -ge 1 ]]; then
  pass "Metal GPU available ($GPU_COUNT device) — used by TF for call_variants inference"
else
  echo "  Metal GPU: not detected (CPU-only inference)"
fi
echo "  Note: Metal GPU provides minimal speedup for call_variants in v1.9"
echo "        due to the 'small model' optimization that pre-screens on CPU."
echo ""

# ── Pipeline runner ───────────────────────────────────────────────────────────
run_pipeline() {
  local run_num="$1"

  local run_dir="$OUTPUT_DIR/runs/run_${run_num}"
  # Clean any stale output from interrupted prior runs to prevent corrupt data.
  # make_examples appends to existing tfrecords rather than overwriting, so
  # leftover partial files from a killed run will produce corrupt gzip streams.
  rm -rf "$run_dir"
  mkdir -p "$run_dir"

  local EXAMPLES="$run_dir/make_examples.tfrecord@${SHARDS}.gz"
  local CV_OUTPUT="$run_dir/call_variants_output.tfrecord.gz"
  local OUT_VCF="$run_dir/output.vcf.gz"

  export TF_CPP_MIN_LOG_LEVEL=0
  export TF2_BEHAVIOR=1
  export TPU_ML_PLATFORM=Tensorflow

  # ── make_examples ──
  info "make_examples (run $run_num)"
  SECONDS=0
  seq 0 $((SHARDS - 1)) | parallel -q --halt 2 --line-buffer \
    "$DV_HOME/bin/make_examples" \
      --mode calling \
      --ref "$REF" \
      --reads "$BAM" \
      --examples "$EXAMPLES" \
      --checkpoint "$DV_HOME/models/wgs" \
      --regions chr20 \
      --task {} \
    2>&1 | tee "$run_dir/make_examples.log"
  local me_seconds=$SECONDS

  # ── call_variants ──
  info "call_variants (run $run_num)"
  SECONDS=0
  "$DV_HOME/bin/call_variants" \
    --outfile "$CV_OUTPUT" \
    --examples "$EXAMPLES" \
    --checkpoint "$DV_HOME/models/wgs" \
    --batch_size "$BATCH_SIZE" \
    2>&1 | tee "$run_dir/call_variants.log"
  local cv_seconds=$SECONDS

  # ── postprocess_variants ──
  # --cpus 1 avoids a segfault in bcftools naive_concat when call_variants
  # dynamically shards its output and some shards are empty. With --cpus >1,
  # postprocess partitions the genome and some partitions produce empty VCFs
  # that bcftools cannot parse. --cpus 1 bypasses concat entirely.
  info "postprocess_variants (run $run_num)"
  SECONDS=0
  "$DV_HOME/bin/postprocess_variants" \
    --ref "$REF" \
    --infile "$CV_OUTPUT" \
    --outfile "$OUT_VCF" \
    --cpus 1 \
    2>&1 | tee "$run_dir/postprocess_variants.log"
  local pp_seconds=$SECONDS

  local total=$((me_seconds + cv_seconds + pp_seconds))

  pass "Run $run_num: make_examples=${me_seconds}s  call_variants=${cv_seconds}s  postprocess=${pp_seconds}s  total=${total}s"

  # Append JSON entry
  python3 -c "
import json
entry = {
    'run': $run_num,
    'stages': {
        'make_examples': $me_seconds,
        'call_variants': $cv_seconds,
        'postprocess_variants': $pp_seconds
    },
    'total': $total,
    'shards': $SHARDS,
    'batch_size': $BATCH_SIZE
}
print(json.dumps(entry))
" >> "$RESULTS_JSONL"
}

# ── hap.py runner ─────────────────────────────────────────────────────────────
run_happy() {
  banner "[ 4 ] hap.py accuracy evaluation"

  if ! command -v docker &>/dev/null; then
    echo "  WARNING: Docker not found. Skipping hap.py evaluation."
    echo "  Install Docker Desktop for Mac to enable accuracy benchmarking."
    return 0
  fi

  # Check if Docker daemon is running
  if ! docker info &>/dev/null; then
    echo "  WARNING: Docker daemon not running. Skipping hap.py evaluation."
    echo "  Start Docker Desktop and re-run with: bash scripts/benchmark.sh --skip-happy"
    return 0
  fi

  # Find a VCF to evaluate
  local vcf_to_eval=""
  if [[ -f "$OUTPUT_DIR/runs/run_1/output.vcf.gz" ]]; then
    vcf_to_eval="$OUTPUT_DIR/runs/run_1/output.vcf.gz"
  else
    echo "  WARNING: No output VCF found. Skipping hap.py."
    return 0
  fi

  local happy_dir="$OUTPUT_DIR/happy"
  mkdir -p "$happy_dir"

  info "Running hap.py on $(basename "$vcf_to_eval") ..."
  info "(hap.py runs under Rosetta 2 emulation — this is slow, be patient)"

  docker run --rm --platform linux/amd64 \
    -v "$DATA_DIR/reference:/reference" \
    -v "$DATA_DIR/benchmark:/benchmark" \
    -v "$(dirname "$vcf_to_eval"):/query" \
    -v "$happy_dir:/happy" \
    jmcdani20/hap.py:v0.3.12 /opt/hap.py/bin/hap.py \
      /benchmark/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
      "/query/$(basename "$vcf_to_eval")" \
      -f /benchmark/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
      -r /reference/GRCh38_no_alt_analysis_set.fasta \
      -o /happy/happy.output \
      --engine=vcfeval \
      --pass-only \
      -l chr20 \
    2>&1 | tee "$happy_dir/happy.log"

  if [[ -f "$happy_dir/happy.output.summary.csv" ]]; then
    pass "hap.py completed"
    # Parse summary.csv
    python3 -c "
import csv, json
results = {}
with open('$happy_dir/happy.output.summary.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Filter'] == 'PASS':
            vtype = row['Type']
            results[vtype] = {
                'TRUTH.TP': int(float(row['TRUTH.TP'])),
                'TRUTH.FN': int(float(row['TRUTH.FN'])),
                'QUERY.FP': int(float(row['QUERY.FP'])),
                'METRIC.Recall': float(row['METRIC.Recall']),
                'METRIC.Precision': float(row['METRIC.Precision']),
                'METRIC.F1_Score': float(row['METRIC.F1_Score'])
            }
print(json.dumps(results, indent=2))
" > "$happy_dir/happy_parsed.json"
  else
    echo "  WARNING: hap.py summary not found."
  fi
}

# ── Finalize results ──────────────────────────────────────────────────────────
finalize_results() {
  banner "Finalizing results"

  python3 << 'PYEOF'
import json, os, platform, subprocess, statistics

output_dir = os.environ['OUTPUT_DIR']
jsonl_path = os.path.join(output_dir, 'benchmark_runs.jsonl')
final_path = os.path.join(output_dir, 'benchmark_results.json')
happy_path = os.path.join(output_dir, 'happy', 'happy_parsed.json')

# Read run entries
runs = []
with open(jsonl_path) as f:
    for line in f:
        line = line.strip()
        if line:
            runs.append(json.loads(line))

# Summary statistics
summary = {}
for stage in ['make_examples', 'call_variants', 'postprocess_variants', 'total']:
    vals = [r['total'] if stage == 'total' else r['stages'][stage]
            for r in runs]
    summary[stage] = {
        'mean': round(statistics.mean(vals), 1),
        'std': round(statistics.stdev(vals), 1) if len(vals) > 1 else 0.0,
        'min': min(vals),
        'max': max(vals),
        'values': vals
    }

# System metadata
try:
    chip = subprocess.check_output(
        ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True
    ).strip()
except Exception:
    chip = platform.processor() or 'Unknown'

cores = int(subprocess.check_output(
    ['sysctl', '-n', 'hw.logicalcpu'], text=True
).strip())
perf_cores = int(subprocess.check_output(
    ['sysctl', '-n', 'hw.perflevel0.logicalcpu'], text=True
).strip())
ram = int(subprocess.check_output(
    ['sysctl', '-n', 'hw.memsize'], text=True
).strip()) // (1024**3)

# GPU info
try:
    gpu_cores = subprocess.check_output(
        ['system_profiler', 'SPDisplaysDataType'], text=True
    )
    # Extract GPU core count from system_profiler output
    import re
    gpu_match = re.search(r'Total Number of Cores:\s*(\d+)', gpu_cores)
    gpu_core_count = int(gpu_match.group(1)) if gpu_match else None
except Exception:
    gpu_core_count = None

result = {
    'metadata': {
        'hostname': platform.node(),
        'chip': chip,
        'total_cores': cores,
        'perf_cores': perf_cores,
        'gpu_cores': gpu_core_count,
        'ram_gb': ram,
        'deepvariant_version': '1.9.0',
        'tensorflow_version': '2.13.1',
        'timestamp': subprocess.check_output(
            ['date', '+%Y-%m-%dT%H:%M:%S'], text=True
        ).strip(),
        'region': 'chr20',
        'sample': 'HG003',
        'shards': runs[0]['shards'] if runs else 0,
        'batch_size': runs[0]['batch_size'] if runs else 0,
        'num_runs': len(runs)
    },
    'runs': runs,
    'summary': summary,
    # Published reference: 96-core GCP n2-standard-96, CPU-only, full genome
    # From docs/metrics.md
    'reference_full_genome': {
        'platform': 'GCP n2-standard-96 (96 vCPU, 384 GB RAM, CPU-only)',
        'make_examples': 2714,
        'call_variants': 986,
        'postprocess_variants': 411,
        'total': 4738
    },
    # Estimated reference: 16-vCPU GCP n2-standard-16 (8 physical cores)
    # Derived from DeepVariant-on-Spark paper (PMC7481958) scaling ratios
    # applied to v1.9 96-CPU metrics. See benchmark_viz.py for methodology.
    'reference_estimated_16vcpu': {
        'platform': 'GCP n2-standard-16 (8 phys. cores, estimated from scaling data)',
        'make_examples': int(2714 * 5.108),
        'call_variants': int(986 * 2.820),
        'postprocess_variants': int(411 * 1.167),
        'total': int(2714 * 5.108) + int(986 * 2.820) + int(411 * 1.167)
    }
}

# Add hap.py results if available
if os.path.exists(happy_path):
    with open(happy_path) as f:
        result['happy'] = json.loads(f.read())

with open(final_path, 'w') as f:
    json.dump(result, f, indent=2)

print(f"Results written to {final_path}")
PYEOF
}

# ── Main ──────────────────────────────────────────────────────────────────────

# Clear previous run data
: > "$RESULTS_JSONL"

# ── [ 3 ] Run pipeline ───────────────────────────────────────────────────────
banner "[ 3 ] Running DeepVariant pipeline"

for run in $(seq 1 "$NUM_RUNS"); do
  banner "=== Run $run/$NUM_RUNS ==="
  run_pipeline "$run"
  echo ""
done

# ── [ 4 ] hap.py ─────────────────────────────────────────────────────────────
if [[ "$SKIP_HAPPY" != "true" ]]; then
  run_happy
fi

# ── [ 5 ] Finalize ───────────────────────────────────────────────────────────
export OUTPUT_DIR
finalize_results

echo ""
echo "================================================================"
echo "  Benchmark complete!"
echo "================================================================"
echo ""
echo "  Results JSON:  $RESULTS_JSON"
echo ""
echo "  Visualize with:"
echo "    python3 scripts/benchmark_viz.py $RESULTS_JSON"
echo "    python3 scripts/benchmark_viz.py $RESULTS_JSON --show   # interactive"
echo ""
