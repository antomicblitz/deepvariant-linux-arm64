#!/bin/bash
# Train EfficientNet-B3 model for DeepVariant WGS variant calling.
#
# Prerequisites:
#   1. bash scripts/setup_training_env.sh
#   2. bash scripts/download_training_data.sh
#
# This script:
#   1. Generates training TFRecords from HG001 using the official DeepVariant Docker
#   2. Generates tuning TFRecords from HG003 chr20
#   3. Trains EfficientNet-B3 using our modified keras_modeling.py
#   4. Exports the trained model as a SavedModel
#
# Usage: bash scripts/train_efficientnet_b3.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-$HOME/dv-training-data}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
NPROC=$(nproc)
DV_VERSION="1.9.0"
DV_IMAGE="google/deepvariant:${DV_VERSION}"

REF="${DATA_DIR}/reference/GRCh38_no_alt_analysis_set.fasta"
# Training: HG003 full BAM on chr1-19,21-22
TRAIN_BAM="${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam"
TRAIN_TRUTH="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TRAIN_BED="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"
# Tuning: HG003 chr20 only
TUNE_BAM="${DATA_DIR}/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam"
TUNE_TRUTH="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"
TUNE_BED="${DATA_DIR}/truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed"

TRAIN_DIR="${DATA_DIR}/training_examples"
TUNE_DIR="${DATA_DIR}/tuning_examples"
OUTPUT_DIR="${DATA_DIR}/output"

# ============================================
# Step 1: Generate training TFRecords from HG001
# ============================================
echo "=== Step 1: Generate training TFRecords from HG003 (chr1-19,21-22) ==="
echo "This uses the official DeepVariant x86 Docker for make_examples."
echo "Expected time: ~30-60 min depending on CPU count (${NPROC} cores)."

if [[ ! -f "${TRAIN_DIR}/training_examples.tfrecord-00000-of-$(printf '%05d' ${NPROC}).gz" ]]; then
  echo ">>> Pulling DeepVariant Docker image..."
  docker pull ${DV_IMAGE}

  echo ">>> Running make_examples in training mode..."
  # Use training regions: all autosomes except chr20 (held out for tuning)
  TRAINING_REGIONS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr21 chr22"

  CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref"

  time docker run --rm \
    -v "${DATA_DIR}/reference":/reference \
    -v "${DATA_DIR}/bam":/bam \
    -v "${DATA_DIR}/truth":/truth \
    -v "${TRAIN_DIR}":/output \
    ${DV_IMAGE} \
    /opt/deepvariant/bin/make_examples \
      --mode training \
      --ref /reference/GRCh38_no_alt_analysis_set.fasta \
      --reads /bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam \
      --truth_variants /truth/HG003_GRCh38_1_22_v4.2.1_benchmark.vcf.gz \
      --confident_regions /truth/HG003_GRCh38_1_22_v4.2.1_benchmark_noinconsistent.bed \
      --examples /output/training_examples.tfrecord@${NPROC}.gz \
      --channel_list "${CHANNEL_LIST}" \
      --regions "${TRAINING_REGIONS}" \
      --task 0

  echo ">>> Training TFRecords generated."
else
  echo ">>> Training TFRecords already exist, skipping."
fi

# Count training examples
TRAIN_COUNT=$(python3 -c "
import tensorflow as tf
import glob
count = 0
for f in sorted(glob.glob('${TRAIN_DIR}/training_examples.tfrecord-*-of-*.gz')):
    for _ in tf.data.TFRecordDataset(f, compression_type='GZIP'):
        count += 1
print(count)
" 2>/dev/null || echo "0")
echo ">>> Training examples: ${TRAIN_COUNT}"

# ============================================
# Step 2: Generate tuning TFRecords from HG003 chr20
# ============================================
echo ""
echo "=== Step 2: Generate tuning TFRecords from HG003 chr20 ==="

CHANNEL_LIST="read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref"
if [[ ! -f "${TUNE_DIR}/tuning_examples.tfrecord-00000-of-$(printf '%05d' ${NPROC}).gz" ]]; then
  time docker run --rm \
    -v "${DATA_DIR}/reference":/reference \
    -v "${DATA_DIR}/bam":/bam \
    -v "${DATA_DIR}/truth":/truth \
    -v "${TUNE_DIR}":/output \
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
      --task 0

  echo ">>> Tuning TFRecords generated."
else
  echo ">>> Tuning TFRecords already exist, skipping."
fi

TUNE_COUNT=$(python3 -c "
import tensorflow as tf
import glob
count = 0
for f in sorted(glob.glob('${TUNE_DIR}/tuning_examples.tfrecord-*-of-*.gz')):
    for _ in tf.data.TFRecordDataset(f, compression_type='GZIP'):
        count += 1
print(count)
" 2>/dev/null || echo "0")
echo ">>> Tuning examples: ${TUNE_COUNT}"

# ============================================
# Step 3: Create dataset config pbtxt files
# ============================================
echo ""
echo "=== Step 3: Creating dataset configs ==="

# Build comma-separated list of training TFRecord shards
TRAIN_TFRECORDS=$(ls ${TRAIN_DIR}/training_examples.tfrecord-*-of-*.gz 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')
TUNE_TFRECORDS=$(ls ${TUNE_DIR}/tuning_examples.tfrecord-*-of-*.gz 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')

cat > "${OUTPUT_DIR}/train.dataset_config.pbtxt" << EOF
name: "HG001_WGS_Training"
tfrecord_path: "${TRAIN_TFRECORDS}"
num_examples: ${TRAIN_COUNT}
EOF

cat > "${OUTPUT_DIR}/tune.dataset_config.pbtxt" << EOF
name: "HG003_chr20_Tuning"
tfrecord_path: "${TUNE_TFRECORDS}"
num_examples: ${TUNE_COUNT}
EOF

echo ">>> Dataset configs written to ${OUTPUT_DIR}/"

# ============================================
# Step 4: Create training config for EfficientNet-B3
# ============================================
echo ""
echo "=== Step 4: Creating EfficientNet-B3 training config ==="

cat > "${OUTPUT_DIR}/efficientnet_b3_wgs.py" << 'PYEOF'
"""Training config for EfficientNet-B3 WGS model."""
import ml_collections

def get_config():
  """EfficientNet-B3 WGS training config."""
  config = ml_collections.ConfigDict()

  # Model architecture
  config.model_type = 'efficientnet_b3'
  config.trial = 0

  # Paths (overridden by flags, but need defaults)
  config.train_dataset_pbtxt = ''
  config.tune_dataset_pbtxt = ''

  # Training hyperparameters (following ISPRAS recipe + DV WGS defaults)
  config.batch_size = 256  # Reduced for single GPU (vs 16384 for TPU)
  config.num_epochs = 10
  config.num_validation_examples = 150_000

  # SGD + momentum (same as upstream WGS)
  config.optimizer = 'sgd'
  config.momentum = 0.9
  config.use_ema = True
  config.ema_momentum = 0.99
  config.optimizer_weight_decay = 0.0

  config.weight_decay = 0.0001
  config.early_stopping_patience = 100
  config.learning_rate = 0.001  # Lower than InceptionV3 (0.01) — EfficientNet is more sensitive
  config.learning_rate_num_epochs_per_decay = 2.25
  config.learning_rate_decay_rate = 0.9999
  config.warmup_steps = 500  # Small warmup for EfficientNet
  config.label_smoothing = 0.01
  config.backbone_dropout_rate = 0.3  # EfficientNet-B3 default dropout

  config.use_mixed_precision = False  # Keep FP32 for accuracy validation
  config.init_checkpoint = ''
  config.init_backbone_with_imagenet = True  # Transfer learn from ImageNet
  config.best_checkpoint_metric = 'tune/f1_weighted'
  config.include_snp_indel_metrics = True  # Track SNP/INDEL F1 separately

  config.denovo_enabled = False
  config.class_weights = ''
  config.ablation_channels = ''
  config.alt_mode = None

  # Data pipeline
  config.steps_per_iter = 128
  config.log_every_steps = 100
  config.tune_every_steps = 0  # Tune at end of each epoch
  config.prefetch_buffer_bytes = 16 * 1000 * 1000
  config.shuffle_buffer_elements = 50_000
  config.input_read_threads = 8
  config.limit = 0

  return config
PYEOF

echo ">>> Config written to ${OUTPUT_DIR}/efficientnet_b3_wgs.py"

# ============================================
# Step 5: Train EfficientNet-B3
# ============================================
echo ""
echo "=== Step 5: Training EfficientNet-B3 ==="
echo "Batch size: 256 (single GPU)"
echo "Epochs: 10"

source ~/dv-train-env/bin/activate 2>/dev/null || true

# Set PYTHONPATH to include our modified DeepVariant source
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Update config paths
sed -i "s|config.train_dataset_pbtxt = ''|config.train_dataset_pbtxt = '${OUTPUT_DIR}/train.dataset_config.pbtxt'|" \
  "${OUTPUT_DIR}/efficientnet_b3_wgs.py"
sed -i "s|config.tune_dataset_pbtxt = ''|config.tune_dataset_pbtxt = '${OUTPUT_DIR}/tune.dataset_config.pbtxt'|" \
  "${OUTPUT_DIR}/efficientnet_b3_wgs.py"

EXPERIMENT_DIR="${OUTPUT_DIR}/efficientnet_b3_experiment"
mkdir -p "${EXPERIMENT_DIR}"

echo ">>> Starting training..."
time python3 "${REPO_DIR}/deepvariant/train.py" \
  --config="${OUTPUT_DIR}/efficientnet_b3_wgs.py" \
  --experiment_dir="${EXPERIMENT_DIR}" \
  --strategy=mirrored \
  --leader=local

echo ""
echo "=== Training complete ==="
echo "Checkpoints saved to: ${EXPERIMENT_DIR}/checkpoints/"
echo ""
echo "Next step: Export to SavedModel and convert for ARM64 inference."
echo "Run: python3 scripts/export_efficientnet_b3.py --experiment_dir=${EXPERIMENT_DIR}"
