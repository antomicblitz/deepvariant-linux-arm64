# EfficientNet-B3 Training Experiment — Results & Findings

## Summary

**Result: EfficientNet-B3 is 3x SLOWER than InceptionV3 on CPU inference. This approach is a dead end for ARM64 acceleration.**

Despite having 3.2x fewer FLOPs (1.8G vs 5.7G) and 48% fewer parameters (12.3M vs 23.9M), EfficientNet-B3 runs approximately 3x slower than InceptionV3 on CPU. The FLOP count is misleading — depthwise separable convolutions and squeeze-and-excitation blocks have poor computational density on CPUs compared to InceptionV3's large dense convolutions that map perfectly to optimized GEMM kernels in OneDNN/BLAS.

## Benchmark Results

Measured on x86 CPU (TF 2.13.1, `CUDA_VISIBLE_DEVICES=""`, OneDNN enabled):

| Batch Size | InceptionV3 (img/s) | EfficientNet-B3 (img/s) | Speedup |
|-----------|---------------------|-------------------------|---------|
| 32 | 416 | 143 | **0.34x** |
| 64 | 530 | 172 | **0.32x** |
| 128 | 591 | 185 | **0.31x** |

The ratio will be similar on ARM64 since both architectures use the same TF/OneDNN GEMM kernels. The bottleneck is memory access patterns and kernel dispatch overhead, not raw FLOPs.

### Why FLOPs Don't Predict CPU Speed

- **InceptionV3** uses large, dense `Conv2D` operations (e.g., 3×3×192×192) that translate to single large GEMM calls — ideal for BLAS/OneDNN with high arithmetic intensity and good cache reuse.
- **EfficientNet-B3** uses depthwise separable convolutions (many small per-channel ops), squeeze-and-excitation blocks (global pooling + two FC layers per block), and more layers overall — resulting in many small kernel launches with poor data reuse.
- On GPUs or TPUs, the parallelism hides this overhead. On CPUs, kernel dispatch and memory latency dominate.

## Training Pipeline (What Was Built)

### Environment
- **Machine:** WSL2 with NVIDIA RTX 4090 (24GB VRAM), CUDA 12.6 driver
- **Python venv:** `~/dv-train-env` — TF 2.13.1 + CUDA 11.8 pip packages
- **GPU libs:** nvidia-cudnn-cu11, nvidia-cublas-cu11, nvidia-cufft-cu11, nvidia-curand-cu11, nvidia-cusolver-cu11, nvidia-cusparse-cu11, nvidia-cuda-runtime-cu11, nvidia-cuda-nvcc-cu11
- **LD_LIBRARY_PATH** set in `~/dv-train-env/bin/activate` to point to pip nvidia lib dirs
- **XLA_FLAGS** must point to `nvidia/cuda_nvcc` dir for `libdevice.10.bc`

### Data Preparation
- **Reference genome:** GRCh38_no_alt_analysis_set.fasta (3.0GB, NCBI FTP)
- **Training BAM:** HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.bam (46GB)
  - NOTE: Originally planned HG001 but those GCS URLs are 404. Used HG003 for both training and tuning.
- **Tuning BAM:** HG003 chr20 only (1.1GB, extracted via samtools)
- **Truth data:** HG003_GRCh38_1_22_v4.2.1_benchmark VCF + BED from GIAB

### TFRecord Generation
- Used `google/deepvariant:1.9.0` Docker image (x86) for `make_examples --mode training`
- **32 parallel Docker containers** (one per `--task`), required for DV 1.9.0 sharding
- **Training:** chr1-19,21-22 → 32 shards, 16GB total, 4,822,408 examples (~55 min)
- **Tuning:** chr20 only → 32 shards, 370MB, 108,794 examples (~3 min)
- DV 1.9.0 requires `--channel_list` flag and space-separated `--regions`

### Training Configuration
```python
# Key parameters from efficientnet_b3_wgs.py
config.model_type = 'efficientnet_b3'
config.batch_size = 128          # 256 OOMs on RTX 4090 with EfficientNet-B3
config.num_epochs = 1            # Reduced for speed test (originally 10)
config.learning_rate = 0.001     # Lower than InceptionV3's 0.01
config.optimizer = 'sgd'
config.momentum = 0.9
config.use_ema = True
config.init_backbone_with_imagenet = True
config.backbone_dropout_rate = 0.3
config.label_smoothing = 0.01
config.weight_decay = 0.0001
config.warmup_steps = 500
config.tune_every_steps = 5000   # 0 causes ZeroDivisionError in train.py
```

### Training Results (Partial — 1 epoch, stopped at step 5120)
- **Speed:** 4.5 steps/sec on RTX 4090 (128 images/step)
- **Convergence:** 98.7% train accuracy by step 2944 (< 8% of epoch 1)
- **Checkpoint:** EMA checkpoint saved at step 5120
- **SavedModel:** Exported to `~/dv-training-data/output/efficientnet_b3_experiment/saved_model/` (49MB)
- **Verification:** Output shape (1,3), softmax sum = 1.0

### Issues Encountered and Fixed

1. **Missing protobuf `_pb2.py` files** — Must compile protos before training:
   ```bash
   python3 -m grpc_tools.protoc -I. --python_out=. \
     deepvariant/protos/*.proto third_party/nucleus/protos/*.proto
   ```
   Requires `grpcio-tools` pip package. Note: grpcio-tools upgrades protobuf — must force `protobuf==3.20.3` back afterward for TF 2.13.1 compatibility.

2. **Weight shape mismatch in `load_weights_to_model_with_different_channels()`** — The original function zips layers by position, which breaks for EfficientNet-B3 because the 3-channel ImageNet model and 6-channel pileup model have different layer orderings. Fixed by matching layers by name (stripping `_N` suffixes) instead of position.

3. **ptxas not found** — TF 2.13.1 XLA compilation requires `ptxas` from CUDA 11.8. Install via `pip install nvidia-cuda-nvcc-cu11` and add to PATH. Also set `XLA_FLAGS=--xla_gpu_cuda_data_dir=<path_to_cuda_nvcc>` for `libdevice.10.bc`.

4. **OOM at batch_size=256** — EfficientNet-B3 has more memory-heavy intermediate activations than InceptionV3 (squeeze-and-excitation blocks, more layers). Reduced to batch_size=128.

5. **ZeroDivisionError with `tune_every_steps=0`** — The `roundup(0, steps_per_iter)` returns 0, then `step % 0` crashes. Set to a non-zero value.

## Data Quality Caveat

Training and tuning both used HG003 (same individual, different chromosomes). This means:
- Tuning metrics are inflated (model sees same person's sequencing artifacts in both sets)
- Checkpoint selection via `best_checkpoint_metric = 'tune/f1_weighted'` is biased
- **For speed benchmarking this doesn't matter** — the model architecture determines inference speed regardless of data quality
- For credible accuracy claims, would need HG002 validation and multi-sample training

## Conclusion

**The EfficientNet-B3 model swap is not viable for CPU inference acceleration.** The 3.2x FLOP reduction does not translate to speedup — it translates to a 3x slowdown due to EfficientNet's reliance on operations (depthwise convolutions, SE blocks) that are poorly optimized for CPU GEMM kernels.

For accelerating `call_variants` on ARM64 CPUs, the remaining viable approaches are:
1. **TF OneDNN tuning** — `get_concrete_function()` for Grappler optimizations, `KMP_AFFINITY`, system allocator
2. **BF16 fast math on Graviton3+** — via `DNNL_DEFAULT_FPMATH_MODE=BF16`
3. **INT8 quantization of InceptionV3** — preserve the architecture, reduce precision
4. **GPU offload** — CUDA on Jetson Orin, TFLite+OpenCL on Mali (hardware-specific)
