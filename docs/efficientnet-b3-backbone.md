# EfficientNet-B3 Backbone: Training Experiment (Not for Inference)

## Status: Training experiment only. Not for production inference.

EfficientNet-B3 is implemented in `deepvariant/keras_modeling.py` (`efficientnetb3()` and `efficientnetb3_with_imagenet()`) with a complete training pipeline (`scripts/train_efficientnet_b3.sh`, `scripts/export_efficientnet_b3.py`).

## Benchmark Result: 3x Slower Than InceptionV3 on ARM64 CPU

Despite having 3.2x fewer FLOPs, EfficientNet-B3 is **3x slower** than InceptionV3 for CPU inference:

| Model | FLOPs | Params | CPU img/s (batch=128) | Relative Speed |
|-------|-------|--------|-----------------------|----------------|
| InceptionV3 | 5.7G | 23.9M | 591 | **1.0x** |
| EfficientNet-B3 | 1.8G | 12.3M | 185 | **0.31x** |

## Root Cause

EfficientNet's depthwise separable convolutions and squeeze-and-excitation blocks produce many small kernel launches with poor data reuse. InceptionV3's dense `Conv2D` operations map to single large GEMM calls with high arithmetic intensity.

At DeepVariant's input size (100x221), EfficientNet's small spatial dimensions cause poor ACL/OneDNN tile utilization on ARM64. FLOPs do not predict CPU inference speed — kernel efficiency and memory access patterns dominate.

This generalizes to ALL "efficient" CNN architectures (MobileNetV2, MnasNet, etc.) that use depthwise separable convolutions.

## When EfficientNet-B3 Might Be Viable

- **GPU inference** (Jetson Orin, desktop GPUs): parallelism hides kernel dispatch overhead
- **NPU inference** (RK3588 RKNN): smaller model fits in limited memory budgets
- These are niche use cases not currently targeted

## Files

- `deepvariant/keras_modeling.py` — model architecture
- `scripts/train_efficientnet_b3.sh` — training pipeline
- `scripts/export_efficientnet_b3.py` — checkpoint → SavedModel export
- `docs/TRAINING_EXPERIMENT.md` — full experiment results and analysis
