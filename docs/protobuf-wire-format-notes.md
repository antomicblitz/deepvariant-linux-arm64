# Protobuf Wire Format Direct Serialization — Notes and Results

## Overview

This document covers the direct TFRecord serialization optimization for
`EncodeExample()` in `make_examples_native.cc`. The optimization bypasses
`tensorflow::Example` proto construction entirely, writing protobuf wire
format bytes directly via `CodedOutputStream`.

## Wire Format Reference

Verified from `third_party/nucleus/protos/feature.proto` and `example.proto`:

```
Example message:
  field 1 (features: Features) -> tag 0x0A (field 1, wire type 2)
    Features message:
      field 1 (feature: map<string, Feature>) -> repeated MapEntry, tag 0x0A
        MapEntry:
          field 1 (key: string) -> tag 0x0A
          field 2 (value: Feature) -> tag 0x12
            Feature (oneof kind):
              field 1 (bytes_list: BytesList) -> tag 0x0A
              field 2 (float_list: FloatList) -> tag 0x12
              field 3 (int64_list: Int64List) -> tag 0x1A
                BytesList: field 1 (value: repeated bytes) -> tag 0x0A
                Int64List: field 1 (value: packed repeated int64) -> tag 0x0A
```

For a BytesList feature (e.g., "image/encoded" with N bytes):
```
0x0A varint(mapentry_size)        // Features.feature MapEntry
  0x0A varint(key_len) key_bytes  // MapEntry.key
  0x12 varint(feature_size)       // MapEntry.value = Feature
    0x0A varint(byteslist_size)   // Feature.bytes_list
      0x0A varint(N) <N bytes>   // BytesList.value
```

For an Int64List feature with packed values:
```
0x0A varint(mapentry_size)
  0x0A varint(key_len) key_bytes
  0x12 varint(feature_size)
    0x1A varint(int64list_size)   // Feature.int64_list (field 3)
      0x0A varint(packed_len) varint(v1) [varint(v2) ...]
```

## Implementation

### Two-phase approach

1. **Size phase:** Compute total serialized size arithmetically (no proto tree
   walk). Helper functions `BytesFeatureEntryContentSize()` and
   `Int64FeatureEntryContentSize()` calculate exact wire sizes.
2. **Write phase:** Allocate output string of exact size, write all bytes in
   one linear pass using `CodedOutputStream`.

### Features emitted in alphabetical key order

Proto3 C++ serializes map entries sorted by key. Our direct serialization
matches this order:

1. `alt_allele_indices/encoded` (BytesList)
2. `denovo_label` (Int64List, optional)
3. `image/encoded` (BytesList, ~154KB)
4. `image/shape` (Int64List, 3 values: [H, W, C])
5. `label` (Int64List, optional)
6. `locus` (BytesList, ~20 bytes)
7. `sequencing_type` (Int64List, 1 value)
8. `variant/encoded` (BytesList, ~300-1000 bytes)
9. `variant_type` (Int64List, 1 value)

### Map ordering caveat

Proto3 C++ `SerializeToString()` serializes map entries in **hash table
iteration order** (NOT sorted alphabetically) by default. Our direct
serialization writes them in alphabetical order. Therefore:

- **Byte-for-byte comparison with proto-based output WILL NOT MATCH** using
  default `SerializeToString()`
- **Roundtrip parsing works fine** — protobuf parsers accept map entries in
  any order
- **To get byte-for-byte match:** Use deterministic serialization
  (`CodedOutputStream::SetSerializationDeterministic(true)`) on the reference
  proto output

### What was eliminated

- ~20 protobuf sub-message object creations per example
- One 154KB copy of the image data (was: encode_buffer_ -> BytesList internal
  buffer -> output string; now: encode_buffer_ -> output string directly)
- The `ByteSizeLong()` + `SerializePartialToArray()` traversals
- 8+ hash map lookups into the Feature map
- Dependencies on `tensorflow/core/example/{example,feature}.pb.h`

### Files modified

- `deepvariant/make_examples_native.cc` — wire format helpers + EncodeExample rewrite
- `deepvariant/make_examples_native.h` — no changes (encode_buffer_ already existed)

## Benchmark Results (2026-03-07)

### Graviton3 (c7g.8xlarge, 32 vCPU Neoverse V1, BF16+jemalloc)

Region: chr20:10M-11M, 2878 examples, 3 runs each.

| Config | Run 1 | Run 2 | Run 3 | Mean | sigma |
|--------|-------|-------|-------|------|-------|
| Baseline (proto) | 34.741s | 34.620s | 34.595s | **34.65s** | 0.08s |
| Direct serial | 34.550s | 34.689s | 34.544s | **34.59s** | 0.08s |
| **Delta** | | | | **-0.06s** | **-0.2% (noise)** |

### Oracle A2 (AmpereOne, 16 OCPU / 32 vCPU, Eigen fallback, no jemalloc)

Region: chr20:10M-11M, 2878 examples, 3 runs each.
Clean dedicated instance (no competing workloads).

| Config | Run 1 | Run 2 | Run 3 | Mean | sigma |
|--------|-------|-------|-------|------|-------|
| Baseline (proto) | 46.460s | 46.327s | 46.384s | **46.39s** | 0.07s |
| Direct serial | 46.293s | 46.296s | 46.295s | **46.29s** | 0.00s |
| **Delta** | | | | **-0.10s** | **-0.2% (noise)** |

### Oracle A2 variance note

An earlier Oracle A2 benchmark (43.82s +/- 0.98s baseline, 44.75s +/- 0.33s
direct-serial) was invalidated because concurrent Bazel compilation on the same
instance caused CPU contention. The high variance (sigma=0.98s vs sigma=0.07s on
the clean run) was the tell.

**Platform characteristic:** Even on a clean A2 instance, AmpereOne shows
slightly higher run-to-run variance than Graviton3 for this workload. The IPC
gap (1.72 vs 2.61) and higher L1 cache miss rate (1.16% vs 0.92%) on AmpereOne
contribute to this. **Future A2 benchmarks should plan for N>=5 runs** to get
tight confidence intervals — N=3 is sufficient on Graviton3 (sigma=0.08s) but
marginal on A2.

## Conclusion

**0% measurable impact on both platforms.** The ~20us per-example savings from
eliminating one 154KB memcpy is far below the noise floor on a 35-46 second
benchmark (0.13% theoretical maximum).

The `_message.so` 29-43% CPU share in make_examples is dominated by protobuf
operations OTHER than Example serialization: Read proto access, Variant field
access, pybind11 boundary crossings, and varint encoding throughout the pipeline.
The EncodeExample serialization path is a tiny fraction of that 29-43%.

**Code is retained** because it is cleaner: removes `tensorflow/core/example/
{example,feature}.pb.h` dependencies, eliminates tf::Example object construction,
and reduces the data copy chain from 3 copies to 2 copies of the 154KB image.

**The protobuf serialization optimization space is exhausted.** All three
approaches tested (arena allocation, SerializeToArray, direct wire format) show
0% impact. The remaining protobuf bottleneck cannot be addressed without
architectural changes (moving proto field access from Python/pybind11 into C++).

## Validation

### Roundtrip parse test

Run make_examples on a small region, then parse the output TFRecords:

```python
import tensorflow as tf
import sys

count = errors = 0
for path in sys.argv[1:]:
    for record in tf.data.TFRecordDataset(path, compression_type='GZIP'):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        features = example.features.feature
        for key in ['image/encoded', 'variant/encoded', 'alt_allele_indices/encoded',
                    'image/shape', 'variant_type', 'sequencing_type', 'locus']:
            if key not in features:
                print(f"ERROR: example {count} missing '{key}'")
                errors += 1
        shape = list(features['image/shape'].int64_list.value)
        img = features['image/encoded'].bytes_list.value[0]
        if len(shape) != 3 or len(img) != shape[0] * shape[1] * shape[2]:
            print(f"ERROR: example {count} shape/image mismatch")
            errors += 1
        count += 1
print(f"Validated {count} examples, {errors} errors")
sys.exit(1 if errors else 0)
```

### Full pipeline VCF comparison

Run ME -> CV -> PP on chr20:10M-11M with both old and new code. Compare:
```bash
diff <(bcftools view -H -f PASS output_direct.vcf.gz) \
     <(bcftools view -H -f PASS output_standard.vcf.gz)
```
Zero differences required.
