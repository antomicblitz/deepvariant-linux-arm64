# Agent Findings — v1.9.0-arm64.5 Work Session

_Shared communication document. Each agent appends findings here._

## Agent 1 (Direct TFRecord Serialization) — Pre-flight Understanding

### What I understand about the current serialization path

In `make_examples_native.cc`, `EncodeExample()` (lines 398-480):

1. `FillPileupArray()` fills `encode_buffer_` with raw pileup image pixels
   (100 rows × 221 cols × 7 channels = 154,700 bytes of uint8 data)
2. `EncodeAltAlleles()` serializes alt allele indices proto → small string (~10-50 bytes)
3. `absl::StrCat()` creates locus string like `"chr20:1000001-1000002"` (~20 bytes)
4. `EncodeVariant()` serializes the Variant proto → string (~300-1000 bytes)
5. Creates `tensorflow::Example example;` on the stack
6. Sets 7-9 features via repeated `example.mutable_features()->mutable_feature()["key"]`
   map lookups, each creating BytesList or Int64List sub-messages:
   - `locus` → BytesList (small string)
   - `variant/encoded` → BytesList (serialized Variant proto)
   - `variant_type` → Int64List (single int64 enum)
   - `alt_allele_indices/encoded` → BytesList (serialized proto)
   - `image/encoded` → BytesList (**154,700 bytes** — dominant feature)
   - `image/shape` → Int64List (3 values: [100, 221, 7])
   - `sequencing_type` → Int64List (single int64)
   - `label` → Int64List (optional, training only)
   - `denovo_label` → Int64List (optional)
7. `example.SerializeToString(&encoded_example)` — serializes the entire proto (~155KB)
8. Returns the serialized string

The returned string is passed to `ExampleWriter::Add()` in
`third_party/nucleus/io/example_writer.cc`, which calls
`tensorflow::io::RecordWriter::WriteRecord(value)` to write it to a
GZIP-compressed TFRecord file. The RecordWriter handles length prefixing,
CRC32C checksums, and GZIP compression.

**Data copies of the 154KB image:**
- Copy 1: `FillPileupArray()` writes pixels into `encode_buffer_`
- Copy 2: `add_value(encode_buffer_.data(), encode_buffer_.size())` copies
  into the BytesList's internal string buffer
- Copy 3: `SerializeToString()` copies from the proto's internal buffer into
  the output string

Direct serialization eliminates Copy 2.

### What "Direct TFRecord Serialization" means

Instead of constructing a `tensorflow::Example` protobuf object and calling
`SerializeToString()`, we write the protobuf wire format bytes directly.

The wire format for tf::Example (verified from `feature.proto` and `example.proto`):

```
Example message:
  field 1 (features: Features) → tag 0x0A (field 1, wire type 2)
    Features message:
      field 1 (feature: map<string, Feature>) → repeated MapEntry, tag 0x0A
        MapEntry:
          field 1 (key: string) → tag 0x0A
          field 2 (value: Feature) → tag 0x12
            Feature (oneof kind):
              field 1 (bytes_list: BytesList) → tag 0x0A
              field 2 (float_list: FloatList) → tag 0x12
              field 3 (int64_list: Int64List) → tag 0x1A
                BytesList: field 1 (value: repeated bytes) → tag 0x0A
                Int64List: field 1 (value: packed repeated int64) → tag 0x0A
```

For a BytesList feature "image/encoded" with N bytes of data:
```
0x0A varint(mapentry_size)        // Features.feature MapEntry
  0x0A varint(13) "image/encoded" // MapEntry.key
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

The entire structure is deterministic. The approach:
1. Compute total serialized size arithmetically (no proto tree walk)
2. Allocate output string of exact size
3. Write all bytes in one linear pass using CodedOutputStream

### The EncodeExample benchmark baseline

- **Region:** chr20:10000000-11000000 (1MB, 2878 examples)
- **Platform:** Oracle A2 (AmpereOne, 16 OCPU / 32 vCPU)
- **Config:** `TF_ENABLE_ONEDNN_OPTS=0`, jemalloc OFF
- **Docker image:** `deepvariant-arm64:v1.9.0-arm64.2` (pre-df3e13d9)
- **Results:** 45.316s, 45.280s, 45.307s → **45.30s ± 0.02s**

The df3e13d9 commit (buffer reuse + absl::StrCat) has NOT been benchmarked yet.
Step 0 will measure the "after" for that commit before implementing direct
serialization.

### My implementation plan (step by step)

1. Implement direct wire format serialization helpers using CodedOutputStream
2. Replace the proto construction in EncodeExample() with direct wire writes
3. Write byte-for-byte validation: serialize same data via both paths, compare
4. Write roundtrip test: parse direct-serialized bytes as tf::Example
5. Benchmark on Hetzner CAX31 (or Oracle A2 if available)
6. Document results regardless of outcome

### Risks I've identified

1. **Wire format correctness:** Any wrong tag or length silently corrupts data.
   Mitigated by byte-for-byte comparison against proto-based serialization.
2. **Map key ordering:** Proto3 C++ sorts map entries alphabetically, but this
   is implementation-specific. Must verify actual order empirically.
3. **Expected low impact:** ~0.13% on micro-benchmark (below noise floor),
   ~0.8% on full chr20. Most likely outcome: close as "not worth it."
4. **Future fragility:** If upstream adds features, direct path silently omits
   them. Keep old path as fallback.

### Files I will modify

- `deepvariant/make_examples_native.cc` — replace proto construction with direct wire format
- `docs/protobuf-serialization-notes.md` — update with benchmark results
- `docs/agent-findings.md` — this file, ongoing findings
- `CLAUDE.md` — update Phase 2G with final results

## Agent 1 — Implementation Notes

### Changes made to make_examples_native.cc

1. **Added includes:** `google/protobuf/io/coded_stream.h` and
   `google/protobuf/io/zero_copy_stream_impl_lite.h`

2. **Removed includes:** `<sstream>` (unused after absl::StrCat), and
   `tensorflow/core/example/{example,feature}.pb.h` (no longer constructing
   tf::Example proto).

3. **Added wire format helpers** (anonymous namespace, ~100 lines):
   - `VarintSize32()` / `VarintSize64()` — wrappers for CodedOutputStream
   - `BytesFeatureEntryContentSize()` — computes wire size of a BytesList MapEntry
   - `Int64FeatureEntryContentSize()` — computes wire size of an Int64List MapEntry
   - `WriteBytesFeatureEntry()` — writes a complete BytesList MapEntry
   - `WriteInt64FeatureEntry()` — writes a complete Int64List MapEntry

4. **Replaced EncodeExample body** (lines 534-661): Removed tf::Example proto
   construction (lines 432-477 of original). Replaced with:
   - Phase 1: Compute total serialized size arithmetically
   - Phase 2: Allocate output string, write wire format via CodedOutputStream
   - Features emitted in alphabetical key order

### Map ordering issue (IMPORTANT for validation)

Proto3 C++ `SerializeToString()` serializes map entries in **hash table
iteration order** (NOT sorted alphabetically). Our direct serialization writes
them in alphabetical order. Therefore:

- **Byte-for-byte comparison with proto-based output WILL NOT MATCH** using
  default `SerializeToString()`
- **Roundtrip parsing works fine** — protobuf parsers accept map entries in
  any order
- **To get byte-for-byte match:** Must use deterministic serialization
  (`CodedOutputStream::SetSerializationDeterministic(true)`) on the reference
  proto output

### Validation approach

Since adding a C++ unit test requires complex test fixtures (ExamplesGenerator
needs MakeExamplesOptions, reference genome, BAM reader, etc.), validation will
use:

1. **Roundtrip parse test (Python):** Read the TFRecord output, parse each
   record as `tf.train.Example`, verify all 7-9 features are present with
   correct types and non-empty values.

2. **Full pipeline VCF comparison:** Run ME → CV → PP with the new code on
   chr20:10M-11M and compare VCF output against a baseline VCF.

### Validation script (run on Hetzner after Docker rebuild)

```python
#!/usr/bin/env python3
"""Validate TFRecord examples written by direct serialization."""
import tensorflow as tf
import sys

EXPECTED_FEATURES = {
    'image/encoded': tf.io.FixedLenFeature((), tf.string),
    'variant/encoded': tf.io.FixedLenFeature((), tf.string),
    'alt_allele_indices/encoded': tf.io.FixedLenFeature((), tf.string),
    'image/shape': tf.io.VarLenFeature(tf.int64),
    'variant_type': tf.io.VarLenFeature(tf.int64),
    'sequencing_type': tf.io.VarLenFeature(tf.int64),
    'locus': tf.io.FixedLenFeature((), tf.string),
}

count = 0
errors = 0
for path in sys.argv[1:]:
    for record in tf.data.TFRecordDataset(path, compression_type='GZIP'):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        features = example.features.feature
        for key in EXPECTED_FEATURES:
            if key not in features:
                print(f"ERROR: example {count} missing feature '{key}'")
                errors += 1
        # Check image dimensions
        shape = list(features['image/shape'].int64_list.value)
        if len(shape) != 3 or shape[0] <= 0 or shape[1] <= 0 or shape[2] <= 0:
            print(f"ERROR: example {count} invalid shape {shape}")
            errors += 1
        img = features['image/encoded'].bytes_list.value[0]
        expected_size = shape[0] * shape[1] * shape[2]
        if len(img) != expected_size:
            print(f"ERROR: example {count} image size {len(img)} != {expected_size}")
            errors += 1
        count += 1

print(f"Validated {count} examples, {errors} errors")
sys.exit(1 if errors else 0)
```
