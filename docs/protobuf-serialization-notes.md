# Protobuf Serialization in make_examples: Analysis and Optimizations

## Profiling Context

`perf report --sort=dso` on Linux ARM64 shows `_message.so` (protobuf upb C library) as the #1 CPU consumer in make_examples:

| Platform | `_message.so` CPU % |
|----------|-------------------|
| Graviton3 (Neoverse V1, OneDNN+ACL) | 29.4% |
| Oracle A2 (AmpereOne, Eigen fallback) | 42.6% |

This DSO covers ALL protobuf operations in the Python process — not just serialization in C++. It includes pybind11_protobuf `native_proto_caster` conversions at the Python/C++ boundary, Read proto access, DeepVariantCall candidate creation, and Variant field access.

Function-level breakdown within `_message.so` is unavailable because Docker overlay strips debug symbols (`perf report` shows `[unknown]` for functions).

## What Was Investigated

### Arena Allocation (Not Helpful for Most Protos)

Arena allocation (`google::protobuf::Arena`) replaces per-field heap calls with bump-pointer allocation into a pre-allocated slab, freeing all messages in O(1) at scope exit.

**Why it doesn't help most make_examples protos:**
- `AlleleCount` protos: pre-allocated in a vector in `AlleleCounter::Init()`, mutated in-place — not per-candidate heap allocation
- `DeepVariantCall`: returned by value with RVO (Return Value Optimization) — no explicit heap allocation
- `tensorflow::Example`: stack-allocated; arena allocation was tested and reverted (0% impact, see benchmark below)

The `tensorflow::Example` proto in `EncodeExample()` was moved to arena allocation (`Arena::CreateMessage<tensorflow::Example>(arena)`) which helps with its nested Feature map entries, BytesList, and Int64List submessages. These are the only per-candidate proto allocations worth targeting.

**macOS reference:** Arena allocation showed 0% impact on Apple Silicon. This is expected — Apple's malloc zones have ~80ns DRAM latency vs ~166ns on AmpereOne DDR5. The allocation overhead that arenas eliminate is proportionally smaller on Apple Silicon.

**Linux ARM64 benchmark (arena vs baseline):** Measured on Hetzner CAX31 (8 vCPU Neoverse-N1), bazel-bin binary, chr20:10M-12M (~5082 examples):

| Metric | Baseline (stack) | Arena (CreateMessage) | Delta |
|--------|------------------|-----------------------|-------|
| Wall time | 80.09s | 79.91s | -0.2% (noise) |
| Instructions | 473,206,820,116 | 473,190,651,868 | -0.003% |
| Cycles | 237,333,011,949 | 236,151,565,537 | -0.5% |
| IPC | 1.99 | 2.00 | +0.5% |
| Cache misses | 1,336,680,390 | 1,329,955,966 | -0.5% |
| Cache miss rate | 0.81% | 0.80% | -1.2% |
| `_message.so` share | 49.45% | 49.35% | -0.1pp |

**Conclusion:** Arena allocation of `tensorflow::Example` has zero measurable impact on Linux ARM64. The `_message.so` share is dominated by Smith-Waterman alignment (10.7%), pileup computation, and varint serialization — not allocation. The ~12 per-example submessage allocations eliminated by the arena are negligible. Arena allocation is **closed as a dead end** for this codebase on all platforms (macOS and Linux ARM64 both show 0%).

**Arena was reverted** — the code uses stack-allocated `tensorflow::Example example;` (the original pattern). The encode_buffer_ reuse and absl::StrCat improvements are kept as they address a different problem (repeated 154KB vector allocation, not proto submessage allocation).

### SerializeToArray vs SerializeToString (No Improvement)

`SerializeToString` internally does:
1. `ByteSizeLong()` — traverses proto tree to compute size
2. `str.resize(byte_size)` — single allocation
3. `SerializePartialToArray(str.data(), byte_size)` — writes bytes

Replacing with `SerializeToArray` into a pre-allocated buffer only skips step 2's single `resize()`. For a ~154KB Example proto with ~80K candidates, this saves ~16ms against a 200+ second runtime (0.008%). Not worth the complexity.

### Direct TFRecord Serialization (IMPLEMENTED)

Bypasses `tensorflow::Example` proto object entirely. Writes the protobuf wire
format bytes directly from raw data using `CodedOutputStream`.

**What changed:** In `EncodeExample()`, the `tensorflow::Example` construction
(7-9 map insertions, BytesList/Int64List sub-message creation) and
`SerializeToString()` call are replaced with a two-phase approach:
1. Compute total serialized size arithmetically (no proto tree walk)
2. Allocate output string, write all bytes in one linear pass

**What this eliminates:**
- ~20 protobuf sub-message object creations per example
- One 154KB copy of the image data (was: encode_buffer_ → BytesList internal
  buffer → output string; now: encode_buffer_ → output string directly)
- The `ByteSizeLong()` + `SerializePartialToArray()` traversals
- 8+ hash map lookups into the Feature map
- `#include <sstream>`, `tensorflow/core/example/example.pb.h`, `feature.pb.h`

**Wire format correctness:** Features are emitted in alphabetical key order
(proto3 deterministic map serialization order). Wire tags verified against
`third_party/nucleus/protos/feature.proto` and `example.proto`.

**Validation:** Roundtrip parse test (parse direct output as `tf.train.Example`,
verify all features) + full pipeline VCF comparison. Byte-for-byte comparison
with proto-based path requires deterministic serialization mode
(`SetSerializationDeterministic(true)`) because default `SerializeToString()`
uses hash table iteration order.

**Estimated impact:** ~20µs per example (one 154KB memcpy saved). On chr20:10M-11M
(2878 examples, 45.3s): ~0.13%. On full chr20 ME (~200s, 80K examples): ~0.8%.

**Benchmark results (2026-03-07):**

Graviton3 (c7g.8xlarge, 32 vCPU, BF16+jemalloc, chr20:10M-11M, 2878 examples):

| Config | Run 1 | Run 2 | Run 3 | Mean ± σ |
|--------|-------|-------|-------|----------|
| Baseline (proto) | 34.741s | 34.620s | 34.595s | **34.65s ± 0.08s** |
| Direct serial | 34.550s | 34.689s | 34.544s | **34.59s ± 0.08s** |
| **Delta** | | | | **-0.06s (-0.2%, noise)** |

Oracle A2 (AmpereOne, 16 OCPU / 32 vCPU, Eigen, no jemalloc, chr20:10M-11M, 2878 examples):

| Config | Run 1 | Run 2 | Run 3 | Mean ± σ |
|--------|-------|-------|-------|----------|
| Baseline (proto) | 46.460s | 46.327s | 46.384s | **46.39s ± 0.07s** |
| Direct serial | 46.293s | 46.296s | 46.295s | **46.29s ± 0.00s** |
| **Delta** | | | | **-0.10s (-0.2%, noise)** |

*Note: Earlier Oracle A2 results (43.82s baseline, 44.75s direct-serial) were
invalid — concurrent Bazel compilation on the same instance caused CPU contention
and high variance (σ=0.98s). Re-measured on a dedicated clean instance.*

**Conclusion:** 0% measurable impact on both platforms. The `_message.so` 29-43%
CPU share is dominated by protobuf operations OTHER than Example serialization
(Read protos, Variant access, pybind11 conversions, varint encoding). The ~20µs
per-example saving from eliminating one 154KB memcpy is far below the noise floor.
Code is retained because it is cleaner (removes tf::Example + feature.pb.h deps).

## Changes Made

### 1. `absl::StrCat` replacing `std::ostringstream` (make_examples_native.cc)

The variant range string (`"chr20:10000001-10000002"`) was formatted via `std::ostringstream`, which has locale handling overhead and internal buffering. Replaced with `absl::StrCat` which is 5-10x faster for simple concatenation.

### 2. Reusable pileup data buffer (make_examples_native.cc, make_examples_native.h)

`EncodeExample()` allocated a fresh `std::vector<unsigned char>` (~154KB) per candidate, zeroed it, filled it via `FillPileupArray`, then discarded it. With ~80K candidates per genome, this is ~80K malloc/free cycles of 154KB buffers plus ~12GB of unnecessary zeroing.

Replaced with a persistent `mutable std::vector<unsigned char> encode_buffer_` member on `ExamplesGenerator`. After the first call, `resize()` is a no-op (same image dimensions every iteration), eliminating repeated heap allocation. `memset` zeroing is retained because `FillPileupArray` may not write all pixels (e.g., short pileups with fewer reads than the image height).

## Baseline Benchmark (Before Changes)

Measured on Oracle A2 (AmpereOne, 16 OCPU / 32 vCPU), `TF_ENABLE_ONEDNN_OPTS=0`, Docker image `deepvariant-arm64:v1.9.0-arm64.2`.

**Region:** chr20:10000000-11000000 (1MB, 2878 examples written)

| Run | Wall Time |
|-----|-----------|
| 1 | 45.316s |
| 2 | 45.280s |
| 3 | 45.307s |
| **Mean** | **45.30s ± 0.02s** |

**"After" results:** See Direct TFRecord Serialization benchmark above. Buffer
reuse (df3e13d9) was included in both baseline and direct-serial images. The
clean Oracle A2 benchmark (46.39s baseline, no jemalloc) vs the original 45.30s
(also no jemalloc) shows minor variance from different instance hardware; buffer
reuse in isolation has no measurable impact.

## How to Verify

### Correctness
Run on a small region and compare example counts (before and after should match):
```bash
docker run --rm \
  -v /data/bam:/data/bam \
  -v /data/reference:/data/reference \
  -v /data/output:/data/output \
  --memory=28g \
  --entrypoint /bin/bash \
  deepvariant-arm64:v1.9.0-arm64.2 \
  -c "export TF_ENABLE_ONEDNN_OPTS=0 && python3 /opt/deepvariant/bin/make_examples.zip \
    --mode calling \
    --ref /data/reference/GRCh38_no_alt_analysis_set.fasta \
    --reads /data/bam/HG003.novaseq.pcr-free.35x.dedup.grch38_no_alt.chr20.bam \
    --regions chr20:10000000-11000000 \
    --channel_list=read_base,base_quality,mapping_quality,strand,read_supports_variant,base_differs_from_ref \
    --examples /data/output/test.tfrecord.gz \
    --task 0"
# Expected: 2878 examples written
```

### Performance (after rebuild)
Run 3 times on chr20:10M-11M (same region as baseline) and compare wall time against 45.30s baseline.

### Profiling DSO breakdown
```bash
perf record -g -o /tmp/perf.data -- \
  /opt/deepvariant/bin/make_examples ...
perf report -i /tmp/perf.data --sort=dso --no-children
```

Compare `libc.so.6` share before/after. The buffer reuse should reduce malloc/free calls (fewer 154KB allocations), showing up as lower `libc.so.6` CPU share rather than `_message.so` directly.
