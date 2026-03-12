# Agent Learnings — DeepVariant Linux ARM64

## Template

- **Context**: When/where this applies
- **Problem**: What issue this solves
- **Solution**: Implementation approach
- **References**: Related files

## Learned Patterns

### EfficientNet-B3 Is Not Faster on CPU

- **Context**: Model architecture selection for ARM64 inference
- **Problem**: EfficientNet-B3 has 3.2x fewer FLOPs than InceptionV3 but runs 3x slower on CPU
- **Solution**: FLOPs do not predict CPU speed. Depthwise separable convolutions produce many small kernel launches with poor data reuse. Dense Conv2D (InceptionV3) maps to single large GEMM calls. Stick with InceptionV3.
- **References**: `docs/TRAINING_EXPERIMENT.md`, `docs/architecture-arm64.md`

### AmpereOne OneDNN SIGILL

- **Context**: Running TF+OneDNN on Oracle A2 (AmpereOne/Siryn)
- **Problem**: OneDNN+ACL compiled for Neoverse-N1 triggers SIGILL on AmpereOne ISA
- **Solution**: Use `TF_ENABLE_ONEDNN_OPTS=0` (Eigen fallback) for all binaries on AmpereOne. Use INT8 ONNX for call_variants. `scripts/autoconfig.sh` enforces this automatically.
- **References**: `docs/onednn-ampereone.md`, `scripts/autoconfig.sh`

### ONNX Dynamic INT8 Unsupported on ARM64

- **Context**: Attempting dynamic INT8 quantization with ONNX Runtime on ARM64
- **Problem**: ConvInteger op is not implemented in ORT ARM64 CPUExecutionProvider
- **Solution**: Use static INT8 quantization instead (`scripts/quantize_static_onnx.py`). Static INT8 gives 2.3x speedup over ONNX FP32.
- **References**: `scripts/quantize_static_onnx.py`, `scripts/quantize_model_onnx.py`

### Jemalloc Benefits Are ME-Dominated

- **Context**: Using `DV_USE_JEMALLOC=1` for memory allocator optimization
- **Problem**: Expected uniform speedup across pipeline stages
- **Solution**: ME (make_examples) gets 14-17% speedup from jemalloc's per-thread arenas reducing malloc contention in C++ allocations. CV improvement is negligible — ONNX Runtime and TF use internal allocators that bypass glibc malloc.
- **References**: `scripts/benchmark_jemalloc_ablation.sh`, `docs/architecture-arm64.md`

### TF BF16 OOM on <48 GB Instances

- **Context**: Running TF SavedModel with BF16 on Graviton4
- **Problem**: TF SavedModel uses ~26 GB RSS; forking postprocess pushes total >32 GB, triggering OOM-kill
- **Solution**: Use ONNX backend on 32 GB instances (ONNX models ~3 GB/worker). BF16 TF requires 64+ GB (e.g., c8g.8xlarge).
- **References**: `docs/architecture-arm64.md` (cost projections table)
