#!/usr/bin/env bash
# DeepVariant v1.9 — Performance Benchmark (macOS ARM64, Apple Silicon)
#
# Benchmarks DeepVariant on HG003 chr20, comparing your Apple Silicon Mac
# against published reference metrics from GCP instances. Evaluates accuracy
# against NIST/GIAB truth sets using rtg-tools vcfeval (native) or hap.py (Docker).
#
# TensorFlow Metal GPU is available on all Apple Silicon Macs and provides a
# ~4.25x speedup for call_variants inference (measured: 224s GPU vs 950s CPU-only
# on M1 Max, HG003 chr20). The Metal GPU accelerates the CNN inference for
# variant classification.
#
# Usage:
#   bash scripts/benchmark.sh [--runs N] [--skip-accuracy] [--shards N]
#                              [--batch-size N] [--output-dir DIR]

set -euo pipefail

# ── Prevent macOS sleep ──────────────────────────────────────────────────────
# caffeinate -i prevents idle sleep; -w $$ ties it to this script's lifetime.
# Runs in the background and auto-exits when the script finishes.
caffeinate -i -w $$ &

# ── Defaults ──────────────────────────────────────────────────────────────────
DV_HOME="${DEEPVARIANT_HOME:-$HOME/.deepvariant}"
NUM_RUNS=1
SKIP_ACCURACY=false
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
  --skip-accuracy     Skip accuracy evaluation (no rtg-tools or Docker needed)
  --skip-happy        Alias for --skip-accuracy (backward compat)
  --shards N          Parallel shards for make_examples (default: perf cores)
  --batch-size N      Batch size for call_variants (default: 1024)
  --output-dir DIR    Output directory (default: ~/deepvariant-benchmark)
  --help              Show this help
EOF
  exit 0
}

while (( "$#" )); do
  case "$1" in
    --runs)           NUM_RUNS="$2"; shift 2 ;;
    --skip-accuracy)  SKIP_ACCURACY=true; shift ;;
    --skip-happy)     SKIP_ACCURACY=true; shift ;;
    --shards)         SHARDS="$2"; shift 2 ;;
    --batch-size)     BATCH_SIZE="$2"; shift 2 ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
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
echo "  Accuracy:     $(if $SKIP_ACCURACY; then echo SKIP; else echo YES; fi)"
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

# GIAB truth sets (for accuracy evaluation)
if [[ "$SKIP_ACCURACY" != "true" ]]; then
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
  pass "Metal GPU available ($GPU_COUNT device) — ~4.25x speedup for call_variants"
else
  echo "  WARNING: Metal GPU not detected. call_variants will be ~4x slower."
  echo "  Install tensorflow-metal for GPU acceleration."
fi
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

# ── Published reference accuracy (DeepVariant WGS case study, HG003 chr20) ──
# From docs/deepvariant-case-study.md — these are the targets to match.
REF_SNP_RECALL=0.999682; REF_SNP_PRECISION=0.999336; REF_SNP_F1=0.999509
REF_INDEL_RECALL=0.993437; REF_INDEL_PRECISION=0.995645; REF_INDEL_F1=0.994540

# ── Accuracy table printer ────────────────────────────────────────────────────
print_accuracy_table() {
  local json_file="$1"
  python3 << PYEOF
import json, sys

with open('$json_file') as f:
    data = json.load(f)

ref = {
    'SNP':   {'Recall': $REF_SNP_RECALL,   'Precision': $REF_SNP_PRECISION,   'F1': $REF_SNP_F1},
    'INDEL': {'Recall': $REF_INDEL_RECALL, 'Precision': $REF_INDEL_PRECISION, 'F1': $REF_INDEL_F1}
}

print()
print('  %-7s  %-10s  %-10s  %-10s  %-8s  %-8s  %-8s' % (
    'Type', 'TP', 'FP', 'FN', 'Recall', 'Precision', 'F1'))
print('  ' + '-' * 69)

any_deviation = False
for vtype in ['SNP', 'INDEL']:
    if vtype not in data:
        continue
    d = data[vtype]
    recall = d.get('METRIC.Recall', d.get('Recall', 0))
    precision = d.get('METRIC.Precision', d.get('Precision', 0))
    f1 = d.get('METRIC.F1_Score', d.get('F1', 0))
    tp = d.get('TRUTH.TP', d.get('TP', 0))
    fp = d.get('QUERY.FP', d.get('FP', 0))
    fn = d.get('TRUTH.FN', d.get('FN', 0))

    print('  %-7s  %-10d  %-10d  %-10d  %.6f  %.6f  %.6f' % (
        vtype, tp, fp, fn, recall, precision, f1))

    # Check for significant deviation from published reference
    r = ref.get(vtype, {})
    if r and abs(f1 - r['F1']) > 0.005:
        any_deviation = True

print()
print('  Published reference (GCP x86_64):')
for vtype in ['SNP', 'INDEL']:
    r = ref[vtype]
    print('  %-7s  %10s  %10s  %10s  %.6f  %.6f  %.6f' % (
        vtype, '', '', '', r['Recall'], r['Precision'], r['F1']))

if any_deviation:
    print()
    print('  WARNING: F1 deviates >0.5%% from published reference!')
    sys.exit(1)
else:
    print()
    print('  Accuracy matches published reference (within 0.5%%).')
PYEOF
}

# ── rtg vcfeval runner (native, no Docker) ────────────────────────────────────
patch_rtg_arm64() {
  # rtg-tools 3.11 launcher script rejects ARM64 (only checks for x86_64).
  # The tool is pure Java and works fine on ARM64. Patch the check if needed.
  local rtg_script
  rtg_script="$(command -v rtg)"
  if [[ -n "$rtg_script" ]] && grep -q '"$(uname -m)" != "x86_64"' "$rtg_script" 2>/dev/null; then
    if ! grep -q 'arm64' "$rtg_script" 2>/dev/null; then
      echo "  Patching rtg launcher for ARM64 compatibility ..."
      sed -i.bak 's|"$(uname -m)" != "x86_64"|"$(uname -m)" != "x86_64" \]\] \&\& [[ "$(uname -m)" != "arm64" \]\] \&\& [[ "$(uname -m)" != "aarch64"|' "$rtg_script"
    fi
  fi
}

run_vcfeval() {
  local vcf_to_eval="$1"
  local eval_dir="$OUTPUT_DIR/vcfeval"
  local ref_sdf="$DATA_DIR/reference/GRCh38_no_alt_analysis_set.sdf"

  # Ensure rtg works on ARM64
  patch_rtg_arm64

  # Build SDF from reference (one-time, cached)
  if [[ ! -d "$ref_sdf" ]]; then
    info "Building reference SDF for vcfeval (one-time) ..."
    rtg format -o "$ref_sdf" "$REF"
  else
    echo "  cached: reference SDF"
  fi

  # Clean previous output (rtg refuses to overwrite)
  rm -rf "$eval_dir"

  info "Running rtg vcfeval ..."
  rtg vcfeval \
    --baseline "$TRUTH_VCF" \
    --calls "$vcf_to_eval" \
    --template "$ref_sdf" \
    --output "$eval_dir" \
    --evaluation-regions "$TRUTH_BED" \
    --region chr20 \
    --output-mode split \
    2>&1 | tee "$eval_dir.log"

  if [[ -f "$eval_dir/summary.txt" ]]; then
    pass "rtg vcfeval completed"
    cat "$eval_dir/summary.txt"

    # Parse per-type metrics from ROC files (summary.txt is aggregated only)
    # ROC format: score  TP_baseline  FP  TP_call  FN  Precision  Sensitivity  F-measure
    # The last data row gives the all-pass (lowest threshold) metrics.
    python3 << PYEOF
import json, gzip, os

results = {}
roc_files = {
    'SNP': os.path.join('$eval_dir', 'snp_roc.tsv.gz'),
    'INDEL': os.path.join('$eval_dir', 'non_snp_roc.tsv.gz')
}

for vtype, roc_path in roc_files.items():
    if not os.path.exists(roc_path):
        continue
    last_line = None
    with gzip.open(roc_path, 'rt') as f:
        for line in f:
            if not line.startswith('#'):
                last_line = line.strip()
    if last_line:
        fields = last_line.split('\t')
        # fields: score, TP_baseline, FP, TP_call, FN, Precision, Sensitivity, F-measure
        tp_base = int(float(fields[1]))
        fp = int(float(fields[2]))
        fn = int(float(fields[4]))
        precision = float(fields[5])
        sensitivity = float(fields[6])
        f_measure = float(fields[7])

        results[vtype] = {
            'TRUTH.TP': tp_base,
            'QUERY.FP': fp,
            'TRUTH.FN': fn,
            'METRIC.Recall': sensitivity,
            'METRIC.Precision': precision,
            'METRIC.F1_Score': f_measure,
            'Recall': sensitivity,
            'Precision': precision,
            'F1': f_measure,
            'TP': tp_base,
            'FP': fp,
            'FN': fn
        }

with open('$eval_dir/vcfeval_parsed.json', 'w') as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
PYEOF
    return 0
  else
    echo "  WARNING: rtg vcfeval summary not found."
    return 1
  fi
}

# ── hap.py runner (Docker, Rosetta 2 emulation) ──────────────────────────────
run_happy() {
  local vcf_to_eval="$1"
  local happy_dir="$OUTPUT_DIR/happy"
  mkdir -p "$happy_dir"

  info "Running hap.py via Docker on $(basename "$vcf_to_eval") ..."
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
                'METRIC.F1_Score': float(row['METRIC.F1_Score']),
                'Recall': float(row['METRIC.Recall']),
                'Precision': float(row['METRIC.Precision']),
                'F1': float(row['METRIC.F1_Score']),
                'TP': int(float(row['TRUTH.TP'])),
                'FP': int(float(row['QUERY.FP'])),
                'FN': int(float(row['TRUTH.FN']))
            }
print(json.dumps(results, indent=2))
" > "$happy_dir/happy_parsed.json"
    return 0
  else
    echo "  WARNING: hap.py summary not found."
    return 1
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

# Add accuracy results if available (vcfeval or hap.py)
accuracy_json = os.environ.get('ACCURACY_JSON', '')
if accuracy_json and os.path.exists(accuracy_json):
    with open(accuracy_json) as f:
        result['accuracy'] = json.loads(f.read())
    result['accuracy_tool'] = 'vcfeval' if 'vcfeval' in accuracy_json else 'hap.py'
elif os.path.exists(happy_path):
    with open(happy_path) as f:
        result['accuracy'] = json.loads(f.read())
    result['accuracy_tool'] = 'hap.py'

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

# ── [ 4 ] Accuracy evaluation ─────────────────────────────────────────────────
ACCURACY_JSON=""
if [[ "$SKIP_ACCURACY" != "true" ]]; then
  banner "[ 4 ] Accuracy evaluation (vs GIAB HG003 truth set)"

  # Find output VCF from run 1
  VCF_TO_EVAL="$OUTPUT_DIR/runs/run_1/output.vcf.gz"
  if [[ ! -f "$VCF_TO_EVAL" ]]; then
    echo "  WARNING: No output VCF found. Skipping accuracy evaluation."
  else
    ACCURACY_DONE=false

    # Try rtg-tools vcfeval first (native, fast)
    if command -v rtg &>/dev/null; then
      info "Using rtg-tools vcfeval (native ARM64)"
      if run_vcfeval "$VCF_TO_EVAL"; then
        ACCURACY_JSON="$OUTPUT_DIR/vcfeval/vcfeval_parsed.json"
        ACCURACY_DONE=true
      fi
    fi

    # Fall back to Docker hap.py
    if [[ "$ACCURACY_DONE" != "true" ]]; then
      if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
        info "rtg-tools not found. Falling back to hap.py via Docker."
        if run_happy "$VCF_TO_EVAL"; then
          ACCURACY_JSON="$OUTPUT_DIR/happy/happy_parsed.json"
          ACCURACY_DONE=true
        fi
      fi
    fi

    # Neither available
    if [[ "$ACCURACY_DONE" != "true" ]]; then
      echo ""
      echo "  WARNING: No accuracy evaluation tool available."
      echo "  Install one of:"
      echo "    brew tap brewsci/bio && brew install rtg-tools   (recommended, native)"
      echo "    Install Docker Desktop for Mac                   (hap.py via Rosetta 2)"
    fi

    # Print accuracy comparison table
    if [[ -n "$ACCURACY_JSON" && -f "$ACCURACY_JSON" ]]; then
      banner "Accuracy comparison"
      print_accuracy_table "$ACCURACY_JSON"
    fi
  fi
fi

# ── [ 5 ] Finalize ───────────────────────────────────────────────────────────
export OUTPUT_DIR
export ACCURACY_JSON
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
