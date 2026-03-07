#!/bin/bash
# profile_make_examples.sh — perf profiling for make_examples on Linux ARM64
#
# Runs perf stat and perf record on the make_examples pipeline step to produce:
#   - Hardware counters: IPC, L1/LLC cache miss rates, branch mispredictions
#   - CPU profile: DSO (shared library) and function-level breakdowns
#   - Categorized summary comparing Linux ARM64 vs macOS ARM64 reference
#
# Profiles make_examples only (no call_variants or postprocess). Runs all shards
# in parallel via GNU parallel (same as run_deepvariant). Optionally runs twice:
# once with the default allocator, once with jemalloc, to isolate jemalloc's impact.
#
# Usage:
#   sudo bash scripts/profile_make_examples.sh [options]
#
# Options:
#   --data-dir DIR       Data directory (default: /data)
#   --image IMAGE        Docker image (default: ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2)
#   --region REGION      Region (default: chr20)
#   --num-shards N       Number of ME shards (default: nproc)
#   --skip-jemalloc      Skip the jemalloc comparison run
#   --output-dir DIR     Output directory (auto-generated if not specified)
#   --docker-mem MEM     Docker memory limit (default: 28g)
#   --perf-freq HZ       perf sampling frequency (default: 997)
#
# Requirements:
#   - Linux ARM64 host (Graviton3 c7g.4xlarge recommended)
#   - sudo / root access (for perf record -a)
#   - perf: sudo apt install linux-tools-generic linux-tools-$(uname -r)
#   - Test data at DATA_DIR: HG003 chr20 BAM + GRCh38 reference
#   - Docker with the DeepVariant ARM64 image pulled
#
# Output:
#   OUTPUT_DIR/
#     perf_stat_{default,jemalloc}.txt      Hardware counter reports
#     perf_record_{default,jemalloc}.data   Raw perf recording data
#     report_dso_{default,jemalloc}.txt     DSO (library) breakdown
#     report_symbols_{default,jemalloc}.txt Function-level hotspots
#     report_combined_{default,jemalloc}.txt DSO+symbol combined
#     summary.txt                           Categorized comparison table
#
# Profiling approach:
#   System-wide perf (perf stat -a, perf record -a) during the Docker run window.
#   On a clean benchmark instance, make_examples dominates all CPU cores, so
#   system-wide recording is representative. Docker client blocks until the
#   container exits, providing the timing window.

set -euo pipefail

# ============================================================
# Defaults
# ============================================================
IMAGE="ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.5"
DATA_DIR="/data"
REGION="chr20"
NUM_SHARDS=""
SKIP_JEMALLOC=false
OUTPUT_DIR=""
DOCKER_MEM="28g"
PERF_FREQ=997

# ============================================================
# Argument parsing
# ============================================================
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)       DATA_DIR="$2"; shift 2 ;;
    --image)          IMAGE="$2"; shift 2 ;;
    --region)         REGION="$2"; shift 2 ;;
    --num-shards)     NUM_SHARDS="$2"; shift 2 ;;
    --skip-jemalloc)  SKIP_JEMALLOC=true; shift ;;
    --output-dir)     OUTPUT_DIR="$2"; shift 2 ;;
    --docker-mem)     DOCKER_MEM="$2"; shift 2 ;;
    --perf-freq)      PERF_FREQ="$2"; shift 2 ;;
    --help|-h)
      sed -n '2,/^set -euo/{ /^#/s/^# \?//p }' "$0"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

NPROC=$(nproc)
NUM_SHARDS="${NUM_SHARDS:-${NPROC}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/output/perf_profile_$(date +%Y%m%d)}"
mkdir -p "${OUTPUT_DIR}"

# Paths inside the container (mounted at /data)
REF="/data/reference/GRCh38_no_alt_analysis_set.fasta"
BAM="/data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"

# ============================================================
# Prerequisites
# ============================================================
echo "============================================="
echo "  make_examples perf profiler"
echo "============================================="
echo "Platform: $(uname -m), ${NPROC} vCPUs"
CPU_MODEL=$(grep -m1 'model name\|CPU part' /proc/cpuinfo 2>/dev/null | sed 's/.*: //' || echo "unknown")
RAM_GB=$(awk '/^MemTotal/ {printf "%.0f", $2/1048576}' /proc/meminfo 2>/dev/null || echo "unknown")
echo "CPU: ${CPU_MODEL}"
echo "RAM: ${RAM_GB} GB"
echo "Image: ${IMAGE}"
echo "Region: ${REGION}"
echo "Shards: ${NUM_SHARDS}"
echo "Perf freq: ${PERF_FREQ} Hz"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check perf
if ! command -v perf &>/dev/null; then
  echo "ERROR: perf not found. Install with:"
  echo "  sudo apt install linux-tools-generic linux-tools-\$(uname -r)"
  exit 1
fi

# Check test data (host paths)
for F in "${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta" \
         "${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"; do
  if [[ ! -f "$F" ]]; then
    echo "ERROR: Test data not found: $F"
    echo "Download using the setup from scripts/benchmark_full_chr20.sh"
    exit 1
  fi
done

# Check Docker image
if ! docker image inspect "${IMAGE}" &>/dev/null; then
  echo "ERROR: Docker image not found: ${IMAGE}"
  echo "Pull it: docker pull ${IMAGE}"
  exit 1
fi

# Check perf permissions (system-wide recording requires root or paranoid=-1)
if ! perf stat -e cycles -a -- sleep 0.01 2>/dev/null; then
  echo "ERROR: Cannot run system-wide perf. Either:"
  echo "  1. Run this script with sudo"
  echo "  2. sudo sysctl -w kernel.perf_event_paranoid=-1"
  echo "  3. sudo sysctl -w kernel.kptr_restrict=0"
  exit 1
fi

echo "All prerequisites OK."
echo ""

# ============================================================
# Docker command builder
# ============================================================
# Writes a shell script that runs make_examples inside Docker.
# This script is then passed to `perf stat -- ./script.sh` or
# `perf record -- ./script.sh` so perf measures the full Docker window.
write_run_script() {
  local COND="$1"   # "default" or "jemalloc"
  local SCRIPT="$2" # output script path
  local TMP_DIR="/data/output/perf_me_${COND}"

  local ENVS="-e TF_ENABLE_ONEDNN_OPTS=1 -e CUDA_VISIBLE_DEVICES="
  if [[ "$COND" == "jemalloc" ]]; then
    ENVS="${ENVS} -e DV_USE_JEMALLOC=1"
  fi

  cat > "${SCRIPT}" <<RUNEOF
#!/bin/bash
set -euo pipefail
docker run --rm \\
  --memory=${DOCKER_MEM} \\
  -v ${DATA_DIR}:/data \\
  ${ENVS} \\
  ${IMAGE} \\
  bash -c '
    unset OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES
    mkdir -p ${TMP_DIR}
    seq 0 $((NUM_SHARDS - 1)) | parallel -q --halt 2 --line-buffer \\
      /opt/deepvariant/bin/make_examples \\
        --mode=calling \\
        --ref=${REF} \\
        --reads=${BAM} \\
        --examples=${TMP_DIR}/examples.tfrecord@${NUM_SHARDS}.gz \\
        --regions=${REGION} \\
        --channel_list=BASE_CHANNELS,insert_size \\
        --call_small_model_examples=true \\
        --trained_small_model_path=/opt/smallmodels/wgs \\
        --track_ref_reads=true \\
        --small_model_snp_gq_threshold=20 \\
        --small_model_indel_gq_threshold=28 \\
        --small_model_vaf_context_window_size=51 \\
        --task {}
  '
RUNEOF
  chmod +x "${SCRIPT}"
}

# ============================================================
# Profile one condition
# ============================================================
profile_condition() {
  local COND="$1"
  local TMP_DIR="${DATA_DIR}/output/perf_me_${COND}"
  local RUN_SCRIPT="${OUTPUT_DIR}/_run_${COND}.sh"

  write_run_script "${COND}" "${RUN_SCRIPT}"

  echo "============================================="
  echo "  Profiling: ${COND}"
  echo "============================================="

  # --- Phase 1: perf stat (hardware counters) ---
  # ARM64 PMU has ~6 counters. The -d flag causes "Failure to read #slots"
  # crash on Graviton — use explicit events instead. perf will multiplex
  # if there are more events than counters, giving approximate counts.
  echo ""
  echo "--- Phase 1: perf stat (${COND}) ---"
  echo ""

  local STAT_FILE="${OUTPUT_DIR}/perf_stat_${COND}.txt"
  local START_S END_S WALL

  START_S=$(date +%s)
  perf stat -a \
    -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses \
    -o "${STAT_FILE}" -- "${RUN_SCRIPT}" 2>&1 | tail -3
  END_S=$(date +%s)
  WALL=$((END_S - START_S))

  echo ""
  echo "perf stat done: ${WALL}s. Saved to ${STAT_FILE}"

  # Clean up ME output before next phase
  rm -rf "${TMP_DIR}" 2>/dev/null || true

  # --- Phase 2: perf record (CPU profile) ---
  echo ""
  echo "--- Phase 2: perf record (${COND}) ---"
  echo ""

  local RECORD_FILE="${OUTPUT_DIR}/perf_record_${COND}.data"

  START_S=$(date +%s)
  perf record -a -g -F "${PERF_FREQ}" -o "${RECORD_FILE}" -- "${RUN_SCRIPT}" 2>&1 | tail -3
  END_S=$(date +%s)
  WALL=$((END_S - START_S))

  echo ""
  echo "perf record done: ${WALL}s ($(du -h "${RECORD_FILE}" | cut -f1)). Saved to ${RECORD_FILE}"

  # --- Generate reports ---
  echo ""
  echo "--- Generating reports (${COND}) ---"

  # DSO (shared library) breakdown
  perf report -i "${RECORD_FILE}" --sort=dso --stdio --percent-limit=0.1 \
    > "${OUTPUT_DIR}/report_dso_${COND}.txt" 2>/dev/null || true

  # Function-level hotspots (>=0.5% overhead)
  perf report -i "${RECORD_FILE}" --sort=symbol --stdio --percent-limit=0.5 \
    > "${OUTPUT_DIR}/report_symbols_${COND}.txt" 2>/dev/null || true

  # Combined DSO+symbol for categorization
  perf report -i "${RECORD_FILE}" --sort=dso,symbol --stdio --percent-limit=0.3 \
    > "${OUTPUT_DIR}/report_combined_${COND}.txt" 2>/dev/null || true

  echo "Reports saved."

  # Clean up ME output and temp script
  rm -rf "${TMP_DIR}" 2>/dev/null || true
  rm -f "${RUN_SCRIPT}"
}

# ============================================================
# Run profiling
# ============================================================
profile_condition "default"
if [[ "${SKIP_JEMALLOC}" != "true" ]]; then
  profile_condition "jemalloc"
fi

# ============================================================
# Categorize and summarize
# ============================================================
echo ""
echo "============================================="
echo "  Generating summary"
echo "============================================="

python3 - "${OUTPUT_DIR}" "${SKIP_JEMALLOC}" <<'PYEOF'
import sys, os, re

output_dir = sys.argv[1]
skip_jemalloc = sys.argv[2] == "true"

# macOS reference profile (from CLAUDE.md)
MACOS_REF = {
    "Pileup image generation": 33.0,
    "Smith-Waterman realigner": 30.0,
    "malloc/free overhead":     18.0,
    "TFRecord I/O / gzip":     17.0,
    "Other":                     2.0,
}

# --- Parsers ---

def parse_perf_stat(filepath):
    """Parse perf stat output for key hardware counters.

    Handles both single-file and merged multi-group format.
    ARM64 perf stat lines look like:
      1,234,567,890      cycles           # comment  (XX.XX%)
      1,234,567,890      instructions     # 1.44 insn per cycle
    """
    result = {}
    if not os.path.exists(filepath):
        return result
    raw_counts = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Parse raw count lines: "1,234,567  event-name  # annotation"
            m = re.match(r'([\d,]+)\s+([\w-]+)', line)
            if m:
                val = int(m.group(1).replace(",", ""))
                event = m.group(2)
                raw_counts[event] = val
            # IPC annotation
            m = re.search(r'#\s+([\d.]+)\s+insn per cycle', line)
            if m:
                result["ipc"] = float(m.group(1))
            # L1 dcache miss rate annotation
            m = re.search(r'#\s+([\d.]+)\s*%\s+of all L1-dcache', line)
            if m:
                result["l1_dcache_miss_pct"] = float(m.group(1))
            # Cache miss rate (generic)
            m = re.search(r'#\s+([\d.]+)\s*%\s+of all cache', line)
            if m:
                result["cache_miss_pct"] = float(m.group(1))
            # Branch misprediction annotation
            m = re.search(r'#\s+([\d.]+)\s*%\s+of all branches', line)
            if m:
                result["branch_mispredict_pct"] = float(m.group(1))
            # Task clock
            m = re.search(r'([\d,.]+)\s+msec\s+task-clock', line)
            if m:
                result["task_clock_ms"] = float(m.group(1).replace(",", ""))
            # CPU utilization
            m = re.search(r'#\s+([\d.]+)\s+CPUs utilized', line)
            if m:
                result["cpus_utilized"] = float(m.group(1))

    # Compute derived metrics from raw counts if annotations weren't found
    if "ipc" not in result and "instructions" in raw_counts and "cycles" in raw_counts:
        result["ipc"] = round(raw_counts["instructions"] / max(raw_counts["cycles"], 1), 2)
    if "l1_dcache_miss_pct" not in result:
        loads = raw_counts.get("L1-dcache-loads", 0)
        misses = raw_counts.get("L1-dcache-load-misses", 0)
        if loads > 0:
            result["l1_dcache_miss_pct"] = round(misses / loads * 100, 2)
    if "branch_mispredict_pct" not in result:
        br = raw_counts.get("branches", 0)
        br_miss = raw_counts.get("branch-misses", 0)
        if br > 0:
            result["branch_mispredict_pct"] = round(br_miss / br * 100, 2)
    if "cache_miss_pct" not in result:
        refs = raw_counts.get("cache-references", 0)
        misses = raw_counts.get("cache-misses", 0)
        if refs > 0:
            result["cache_miss_pct"] = round(misses / refs * 100, 2)

    result["raw"] = raw_counts
    return result


def parse_dso_report(filepath):
    """Parse perf report --sort=dso output. Returns list of (overhead_pct, dso_name)."""
    result = []
    if not os.path.exists(filepath):
        return result
    with open(filepath) as f:
        for line in f:
            # "    XX.XX%  dso_name"
            m = re.match(r'\s+([\d.]+)%\s+(\S+)', line)
            if m:
                result.append((float(m.group(1)), m.group(2)))
    return result


def parse_symbol_report(filepath):
    """Parse perf report --sort=symbol output. Returns list of (overhead_pct, symbol)."""
    result = []
    if not os.path.exists(filepath):
        return result
    with open(filepath) as f:
        for line in f:
            # "    XX.XX%  [.] symbol_name"  or  "    XX.XX%  symbol_name"
            m = re.match(r'\s+([\d.]+)%\s+(?:\[.\]\s+)?(.+)', line)
            if m:
                sym = m.group(2).strip()
                if sym and not sym.startswith('#'):
                    result.append((float(m.group(1)), sym))
    return result


def parse_combined_report(filepath):
    """Parse perf report --sort=dso,symbol output. Returns list of (pct, dso, symbol)."""
    result = []
    if not os.path.exists(filepath):
        return result
    with open(filepath) as f:
        for line in f:
            # "    XX.XX%  dso_name  [.] symbol_name"
            m = re.match(r'\s+([\d.]+)%\s+(\S+)\s+(?:\[.\]\s+)?(.+)', line)
            if m:
                sym = m.group(3).strip()
                if sym and not sym.startswith('#'):
                    result.append((float(m.group(1)), m.group(2), sym))
    return result


# --- Categorization ---
# Maps DSO names and symbol names to categories.
# Order matters: first match wins.

DSO_PATTERNS = [
    # (category, dso_substring_list)
    ("malloc/free overhead",     ["libjemalloc", "libc-2.", "libc.so", "ld-linux"]),
    ("TFRecord I/O / gzip",     ["libz.so", "libdeflate"]),
    ("BAM reading (htslib)",     ["libhts", "htslib"]),
    ("TF / small model",        ["libtensorflow", "_pywrap_tensorflow", "libabsl", "libprotobuf"]),
    ("Python runtime",          ["libpython", "python3.10"]),
]

SYMBOL_PATTERNS = [
    # (category, regex_list)
    ("Smith-Waterman realigner", [
        r"ssw_", r"SmithWaterman", r"Realigner", r"DeBruijn", r"Haplotype",
        r"WindowAligner", r"SswAligner", r"AlignReads", r"Cigar",
        r"KmerIndex", r"CandidateHaplotype",
    ]),
    ("Pileup image generation", [
        r"PileupImage", r"ImageRow", r"EncodePileup", r"MakeExample",
        r"PileupImageCreator", r"EncodeRead", r"BaseColor", r"QualityColor",
        r"StrandColor", r"ReadMapping", r"BuildPileup", r"AddRead",
        r"pileup_image", r"channels",
    ]),
    ("malloc/free overhead", [
        r"\bmalloc\b", r"\bfree\b", r"_int_malloc", r"_int_free",
        r"\bcfree\b", r"\brealloc\b", r"\bcalloc\b",
        r"tcmalloc", r"je_malloc", r"je_free", r"arena_",
        r"__libc_malloc", r"__libc_free", r"__GI___libc_",
    ]),
    ("TFRecord I/O / gzip", [
        r"deflate", r"inflate", r"RecordWriter", r"TFRecord",
        r"Compress", r"Flush", r"adler32",
    ]),
    ("BAM reading (htslib)", [
        r"hts_", r"sam_", r"bam_", r"bgzf_", r"cram_",
        r"hfile_", r"kstring",
    ]),
    ("TF / small model", [
        r"tensorflow", r"Eigen", r"Session.*Run", r"OpKernel",
        r"DirectSession",
    ]),
]


def categorize_combined(entries):
    """Categorize combined DSO+symbol entries. Returns {category: total_pct}."""
    cats = {}
    for pct, dso, sym in entries:
        assigned = False
        # Try symbol-level categorization first (more specific)
        for cat, patterns in SYMBOL_PATTERNS:
            for pat in patterns:
                if re.search(pat, sym, re.IGNORECASE):
                    cats[cat] = cats.get(cat, 0) + pct
                    assigned = True
                    break
            if assigned:
                break
        if assigned:
            continue
        # Fall back to DSO-level
        for cat, substrings in DSO_PATTERNS:
            for sub in substrings:
                if sub.lower() in dso.lower():
                    cats[cat] = cats.get(cat, 0) + pct
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            cats["Other"] = cats.get("Other", 0) + pct
    return cats


def categorize_dso_only(entries):
    """Fallback categorization using DSO names only."""
    cats = {}
    for pct, dso in entries:
        assigned = False
        for cat, substrings in DSO_PATTERNS:
            for sub in substrings:
                if sub.lower() in dso.lower():
                    cats[cat] = cats.get(cat, 0) + pct
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            # DSOs from make_examples.zip are likely pileup/realigner/other ME code
            if "make_examples" in dso.lower() or "deepvariant" in dso.lower():
                cats["DeepVariant C++ (unresolved)"] = cats.get("DeepVariant C++ (unresolved)", 0) + pct
            else:
                cats["Other"] = cats.get("Other", 0) + pct
    return cats


# --- Process each condition ---
conditions = ["default"]
if not skip_jemalloc:
    conditions.append("jemalloc")

all_results = {}

for cond in conditions:
    hw = parse_perf_stat(os.path.join(output_dir, f"perf_stat_{cond}.txt"))
    dso_entries = parse_dso_report(os.path.join(output_dir, f"report_dso_{cond}.txt"))
    sym_entries = parse_symbol_report(os.path.join(output_dir, f"report_symbols_{cond}.txt"))
    combined = parse_combined_report(os.path.join(output_dir, f"report_combined_{cond}.txt"))

    # Prefer combined categorization; fall back to DSO-only
    if combined:
        cats = categorize_combined(combined)
    elif dso_entries:
        cats = categorize_dso_only(dso_entries)
    else:
        cats = {}

    all_results[cond] = {
        "hardware": hw,
        "dso": dso_entries,
        "symbols": sym_entries,
        "categories": cats,
    }


# --- Print summary ---
lines = []
def out(s=""):
    lines.append(s)
    print(s)

out()
out("=" * 72)
out("  make_examples CPU Profile — Linux ARM64")
out("=" * 72)
out()

# Hardware counters
out("--- Hardware Counters ---")
out()
header = f"  {'Metric':<35s}"
for cond in conditions:
    header += f"  {cond:>15s}"
out(header)
out("  " + "-" * (35 + 17 * len(conditions)))

hw_metrics = [
    ("ipc",                   "IPC (insn/cycle)"),
    ("cpus_utilized",         "CPUs utilized"),
    ("l1_dcache_miss_pct",    "L1-dcache miss rate (%)"),
    ("cache_miss_pct",        "Cache ref miss rate (%)"),
    ("branch_mispredict_pct", "Branch misprediction (%)"),
]
for key, label in hw_metrics:
    row = f"  {label:<35s}"
    for cond in conditions:
        val = all_results[cond]["hardware"].get(key)
        if val is not None:
            row += f"  {val:>15.2f}"
        else:
            row += f"  {'N/A':>15s}"
    out(row)

out()

# Category breakdown
out("--- CPU Profile by Category ---")
out()

# Collect all categories across conditions
all_cats = list(MACOS_REF.keys())
for cond in conditions:
    for cat in all_results[cond]["categories"]:
        if cat not in all_cats:
            all_cats.append(cat)

header = f"  {'Category':<35s}  {'macOS %':>8s}"
for cond in conditions:
    header += f"  {cond + ' %':>12s}"
if len(conditions) > 1:
    header += f"  {'Delta':>8s}"
out(header)
out("  " + "-" * (35 + 10 + 14 * len(conditions) + (10 if len(conditions) > 1 else 0)))

for cat in all_cats:
    macos_pct = MACOS_REF.get(cat, 0)
    macos_str = f"{macos_pct:>7.1f}%" if macos_pct > 0 else f"{'—':>8s}"
    row = f"  {cat:<35s}  {macos_str}"
    vals = []
    for cond in conditions:
        val = all_results[cond]["categories"].get(cat, 0)
        vals.append(val)
        row += f"  {val:>11.1f}%"
    if len(vals) > 1:
        delta = vals[1] - vals[0]
        row += f"  {delta:>+7.1f}%"
    out(row)

out()

# Top DSOs
for cond in conditions:
    out(f"--- Top Shared Libraries ({cond}) ---")
    out()
    for pct, dso in all_results[cond]["dso"][:15]:
        out(f"  {pct:>6.2f}%  {dso}")
    out()

# Top symbols
for cond in conditions:
    syms = all_results[cond]["symbols"]
    if syms:
        out(f"--- Top Functions ({cond}, >=0.5%) ---")
        out()
        for pct, sym in syms[:25]:
            out(f"  {pct:>6.2f}%  {sym}")
        out()
    else:
        out(f"--- Top Functions ({cond}) ---")
        out(f"  (No resolved symbols. See report_symbols_{cond}.txt)")
        out(f"  Symbol resolution may require debug symbols or running perf")
        out(f"  inside the container. DSO-level breakdown above is still valid.")
        out()

out(f"Full reports in: {output_dir}/")
out()

# Save summary to file
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Summary saved to: {summary_path}")
PYEOF

echo ""
echo "============================================="
echo "  Profiling complete"
echo "============================================="
echo "Output: ${OUTPUT_DIR}"
echo ""
echo "Files:"
ls -lh "${OUTPUT_DIR}/"
echo ""
echo "Interactive exploration:"
echo "  perf report -i ${OUTPUT_DIR}/perf_record_default.data"
echo "  perf report -i ${OUTPUT_DIR}/perf_record_default.data --sort=dso,symbol"
if [[ "${SKIP_JEMALLOC}" != "true" ]]; then
  echo "  perf diff ${OUTPUT_DIR}/perf_record_default.data ${OUTPUT_DIR}/perf_record_jemalloc.data"
fi
echo ""
echo "If symbol resolution is poor (many [unknown] entries), try:"
echo "  1. Mount host perf into container: -v \$(which perf):/usr/bin/perf:ro"
echo "     docker run --privileged ... perf record -g -o /data/perf.data -- make_examples ..."
echo "  2. Or install perf inside container: apt install linux-tools-generic"
