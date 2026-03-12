## DeepVariant ARM64 v1.9.0-arm64.6

### What's New

**WES (Whole Exome Sequencing) INT8 support** — The Docker image now ships with pre-quantized INT8 ONNX models for both WGS and WES. No manual quantization or model mounting needed.

**Workflow integration** — New Nextflow DSL2 and Snakemake workflows for batch processing with platform-specific profiles (Graviton, Oracle A1/A2, Hetzner CAX).

### INT8 Models Included

| Model | Path | Size | Speedup |
|-------|------|------|---------|
| WGS INT8 | `/opt/models/wgs/model_int8_static.onnx` | 21 MB | 2.3x over FP32 ONNX |
| WES INT8 | `/opt/models/wes/model_int8_static.onnx` | 21 MB | 1.24x over FP32 ONNX |

Both models use static INT8 quantization (Percentile 99.99 calibration, QDQ format, 500 samples).

### WES INT8 Accuracy (HG003 IDT exome, Graviton4)

| Metric | INT8 | FP32 | Gate |
|--------|------|------|------|
| SNP F1 | 0.9931 | 0.9930 | >= 0.9920 |
| INDEL F1 | 0.9738 | 0.9776 | >= 0.9700 |

### Usage

```bash
docker pull ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6

# WGS (auto-selects INT8 on non-BF16 platforms)
docker run -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/scripts/run_parallel_cv.sh \
  --model_type=WGS \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz

# WES (same command, just change model_type)
docker run -v /path/to/data:/data \
  ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.6 \
  /opt/deepvariant/scripts/run_parallel_cv.sh \
  --model_type=WES \
  --ref=/data/reference.fasta \
  --reads=/data/input.bam \
  --output_vcf=/data/output.vcf.gz \
  --regions=/data/capture.bed
```

### Nextflow / Snakemake

```bash
# Nextflow
nextflow run workflows/nextflow/main.nf \
  --bam /data/sample.bam --ref /data/GRCh38.fasta \
  -profile arm64

# Snakemake
snakemake --cores 16 --configfile workflows/snakemake/config.yaml \
  --config bam=/data/sample.bam ref=/data/GRCh38.fasta
```

### Changes from v1.9.0-arm64.5

- Added WES INT8 static ONNX model (`/opt/models/wes/model_int8_static.onnx`)
- Added WGS INT8 static ONNX model (`/opt/models/wgs/model_int8_static.onnx`)
- Added Nextflow DSL2 workflow (`workflows/nextflow/`)
- Added Snakemake workflow (`workflows/snakemake/`)
- WES validation: INT8 accuracy matches FP32 (SNP F1 0.9931, INDEL F1 0.9738)
- Updated all image references to v1.9.0-arm64.6

### Base Image

Built as a layer on top of `v1.9.0-arm64.5` — all existing functionality unchanged.
