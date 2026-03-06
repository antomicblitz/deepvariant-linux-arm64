# Graviton3 jemalloc ablation — PENDING (N=2, needs N=4)

Current data (2-run average):
- jemalloc OFF: wall=487s ME=278s CV=185s PP=24s ($/genome $3.77)
- jemalloc ON:  wall=443s ME=242s CV=188s PP=9s  ($/genome $3.43)
- ME delta: -13.8%  CV delta: +1.6%  Wall delta: -9.0%

## To complete

Restart c7g.4xlarge in us-east-1, then run:

```bash
bash scripts/benchmark_jemalloc_ablation.sh \
  --runs 4 \
  --usd-per-hr 0.58 \
  --num-shards 16 \
  --use-onnx \
  --onnx-model /opt/models/wgs/model_int8_static.onnx \
  --image ghcr.io/antomicblitz/deepvariant-arm64:v1.9.0-arm64.2
```

## After N=4 data arrives

1. Remove asterisks from Graviton3 jemalloc rows in README.md and CLAUDE.md
2. Update component timings if they change from 2-run averages
3. Delete this file
4. Update CHANGELOG.md with verification note
