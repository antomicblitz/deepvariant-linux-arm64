# EfficientNet-B3 Training Pipeline — Resume Guide

## What Was Done

### 1. Environment Setup (COMPLETE)
- Created Python 3.10 venv at `~/dv-train-env`
- Installed TensorFlow 2.13.1 with GPU support
- Installed CUDA pip packages: cudnn-cu11, cublas-cu11, cufft-cu11, curand-cu11, cusolver-cu11, cusparse-cu11, cuda-runtime-cu11
- Added `LD_LIBRARY_PATH` to `~/dv-train-env/bin/activate` for NVIDIA libs
- Installed training deps: ml-collections, clu, tf-models-official==2.13.2, protobuf==3.20.3
- **GPU verified:** RTX 4090 (24GB VRAM) detected by TensorFlow

### 2. Data Download (COMPLETE)
- **Reference genome:** `~/dv-training-data/reference/GRCh38_no_alt_analysis_set.fasta` (3.0GB, from NCBI FTP)
- **Training BAM:** `~/dv-training-data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam` (46GB, from GCS)
  - NOTE: Original script used HG001 but that URL is 404. We use HG003 for training (chr1-19,21-22) and tuning (chr20)
- **Tuning BAM:** `~/dv-training-data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam` (1.1GB)
- **Truth VCF/BED:** `~/dv-training-data/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.*` (all present)
- Docker image `google/deepvariant:1.9.0` pulled

### 3. Training Script Fixes Applied
File: `scripts/train_efficientnet_b3.sh` was modified:
- Changed from HG001 to HG003 BAM/truth paths for training
- Changed regions from comma-separated to space-separated (DV 1.9.0 requirement)
- Added `--channel_list` flag (required in DV 1.9.0, was missing)

### 4. TFRecord Generation (IN PROGRESS or COMPLETE — check below)

**Training TFRecords** — 32 parallel Docker tasks generating shards:
```
~/dv-training-data/training_examples/training_examples.tfrecord-XXXXX-of-00032.gz
```

To check if this completed:
```bash
# Should show 32 shards, each non-empty
ls -lh ~/dv-training-data/training_examples/*.gz | wc -l  # should be 32
docker ps -q | wc -l  # should be 0 if done
```

**Tuning TFRecords** — NOT YET GENERATED. Need to run 32 parallel tasks for chr20:
```bash
DATA_DIR="$HOME/dv-training-data"
NPROC=32
DV_IMAGE="google/deepvariant:1.9.0"
CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref"

for i in $(seq 0 $((NPROC - 1))); do
  docker run --rm \
    -v "${DATA_DIR}/reference":/reference \
    -v "${DATA_DIR}/bam":/bam \
    -v "${DATA_DIR}/truth":/truth \
    -v "${DATA_DIR}/tuning_examples":/output \
    ${DV_IMAGE} \
    /opt/deepvariant/bin/make_examples \
      --mode training \
      --ref /reference/GRCh38_no_alt_analysis_set.fasta \
      --reads /bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
      --truth_variants /truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
      --confident_regions /truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
      --examples /output/tuning_examples.tfrecord@${NPROC}.gz \
      --channel_list "${CHANNEL_LIST}" \
      --regions "chr20" \
      --task ${i} > /dev/null 2>&1 &
done
wait
echo "Tuning TFRecords done. Shards: $(ls ~/dv-training-data/tuning_examples/*.gz | wc -l)"
```

## What Needs To Be Done Next

### Step A: Verify/Complete TFRecord Generation
```bash
# Check training shards exist (32 files, total ~4-5GB)
ls ~/dv-training-data/training_examples/*.gz | wc -l
du -sh ~/dv-training-data/training_examples/

# If less than 32, some tasks may have failed. Re-run missing tasks.
# If 0, re-run the full parallel generation (see section 4 above but for training_examples)
```

### Step B: Generate Tuning TFRecords (if not done)
Run the tuning command block above. Takes ~2-3 min with 32 parallel tasks.

### Step C: Count Examples and Create Dataset Configs
```bash
source ~/dv-train-env/bin/activate
DATA_DIR="$HOME/dv-training-data"
OUTPUT_DIR="${DATA_DIR}/output"
mkdir -p "${OUTPUT_DIR}"

# Count training examples
TRAIN_COUNT=$(python3 -c "
import tensorflow as tf
import glob
count = 0
for f in sorted(glob.glob('${DATA_DIR}/training_examples/training_examples.tfrecord-*-of-*.gz')):
    for _ in tf.data.TFRecordDataset(f, compression_type='GZIP'):
        count += 1
print(count)
")
echo "Training examples: ${TRAIN_COUNT}"

# Count tuning examples
TUNE_COUNT=$(python3 -c "
import tensorflow as tf
import glob
count = 0
for f in sorted(glob.glob('${DATA_DIR}/tuning_examples/tuning_examples.tfrecord-*-of-*.gz')):
    for _ in tf.data.TFRecordDataset(f, compression_type='GZIP'):
        count += 1
print(count)
")
echo "Tuning examples: ${TUNE_COUNT}"

# Create dataset config files
TRAIN_TFRECORDS=$(ls ${DATA_DIR}/training_examples/training_examples.tfrecord-*-of-*.gz | sort | tr '\n' ',' | sed 's/,$//')
TUNE_TFRECORDS=$(ls ${DATA_DIR}/tuning_examples/tuning_examples.tfrecord-*-of-*.gz | sort | tr '\n' ',' | sed 's/,$//')

cat > "${OUTPUT_DIR}/train.dataset_config.pbtxt" << EOF
name: "HG003_WGS_Training"
tfrecord_path: "${TRAIN_TFRECORDS}"
num_examples: ${TRAIN_COUNT}
EOF

cat > "${OUTPUT_DIR}/tune.dataset_config.pbtxt" << EOF
name: "HG003_chr20_Tuning"
tfrecord_path: "${TUNE_TFRECORDS}"
num_examples: ${TUNE_COUNT}
EOF
```

### Step D: Create Training Config
This is handled by `scripts/train_efficientnet_b3.sh` Step 4, which writes
`${OUTPUT_DIR}/efficientnet_b3_wgs.py`. But you can also run it manually — see the
script for the full config content. Key: update the `train_dataset_pbtxt` and
`tune_dataset_pbtxt` paths in the config.

### Step E: Run Training
```bash
source ~/dv-train-env/bin/activate
REPO_DIR="/home/antonio/deepvariant-linux-arm64"
OUTPUT_DIR="$HOME/dv-training-data/output"
EXPERIMENT_DIR="${OUTPUT_DIR}/efficientnet_b3_experiment"
mkdir -p "${EXPERIMENT_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

time python3 "${REPO_DIR}/deepvariant/train.py" \
  --config="${OUTPUT_DIR}/efficientnet_b3_wgs.py" \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --strategy=mirrored \
  --leader=local
```
Expected: ~1-2 hours on RTX 4090, 10 epochs, batch size 256.

### Step F: Export SavedModel
```bash
source ~/dv-train-env/bin/activate
python3 scripts/export_efficientnet_b3.py \
  --experiment_dir=$HOME/dv-training-data/output/efficientnet_b3_experiment
```

### Step G: Upload SavedModel
Upload `$HOME/dv-training-data/output/efficientnet_b3_experiment/saved_model/` to GitHub or GCS.

## Key Environment Details
- **Venv:** `source ~/dv-train-env/bin/activate` (has LD_LIBRARY_PATH for CUDA)
- **GPU:** RTX 4090, CUDA 12.6 driver, CUDA 11 pip libs (TF 2.13.1 compatible)
- **Data dir:** `~/dv-training-data/`
- **Repo:** `/home/antonio/deepvariant-linux-arm64/`
- **Docker image:** `google/deepvariant:1.9.0` (x86, for make_examples only)

## Gotchas
1. HG001 NovaSeq BAM URLs are 404 on GCS — we use HG003 for both training and tuning
2. DV 1.9.0 requires `--channel_list` flag for make_examples
3. DV 1.9.0 `--regions` expects space-separated, not comma-separated
4. The `--task 0` flag only processes 1/N_SHARDS of data — must run N parallel tasks
5. TF 2.13.1 pip install doesn't include CUDA libs — need separate nvidia-*-cu11 pip packages
6. LD_LIBRARY_PATH must include pip nvidia lib dirs (already in ~/dv-train-env/bin/activate)
