#!/usr/bin/env python3
# Copyright 2024 deepvariant-linux-arm64 contributors.
#
# BSD-3-Clause license (same as upstream DeepVariant).
"""Apply INT8 dynamic post-training quantization to an ONNX model.

Dynamic quantization quantizes weights only — no calibration data is needed
for the quantization itself. Activations remain FP32 at runtime.

Usage:
  # Basic quantization
  python scripts/quantize_model_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8.onnx

  # Quantize and validate against FP32 using real pileup images
  python scripts/quantize_model_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8.onnx \
    --validate \
    --tfrecord_dir /data/output/examples \
    --saved_model_dir /opt/models/wgs

NOTE: Run this AFTER validating FP32 ONNX accuracy with convert_model_onnx.py.
Dynamic INT8 quantization may introduce small accuracy differences — always
re-validate with hap.py before using in production.

Expected output diff (dynamic INT8 vs FP32): < 1e-3 on real pileup images.
"""

import argparse
import glob
import os


def _check_ort_version():
    """Verify ORT >= 1.17.0 for ARM64 INT8 MLAS kernel support."""
    import onnxruntime as ort
    from packaging.version import Version
    if Version(ort.__version__) < Version('1.17.0'):
        raise RuntimeError(
            f'ONNX Runtime >= 1.17.0 required for ARM64 INT8 MLAS kernels '
            f'(SMMLA support). Found: {ort.__version__}')


def quantize(input_path, output_path):
    """Apply dynamic INT8 quantization to an ONNX model."""
    _check_ort_version()
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f'Quantizing {input_path} -> {output_path}')
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100
    print(f'Original: {orig_size:.1f} MB')
    print(f'Quantized: {quant_size:.1f} MB ({reduction:.1f}% smaller)')
    return output_path


def load_real_images(tfrecord_dir, saved_model_dir, num_samples=500):
    """Extract real pileup images from DeepVariant TFRecords.

    This is for post-quantization validation, NOT calibration.
    Dynamic quantization does not use calibration data.

    Args:
        tfrecord_dir: Directory containing tfrecord-?????-of-?????.gz files.
        saved_model_dir: SavedModel dir containing example_info.json.
        num_samples: Number of images to extract.

    Returns:
        List of numpy arrays, each shape (1, H, W, C) in float32 range [-1, 1].
    """
    # Lazy imports — only needed when using real data validation.
    import json

    import numpy as np

    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    import tensorflow as tf  # noqa: E402

    # Get input shape from example_info.json
    input_shape = [100, 221, 7]
    if saved_model_dir:
        info_path = os.path.join(saved_model_dir, 'example_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            input_shape = info.get('shape', input_shape)

    # Find TFRecord files
    patterns = [
        os.path.join(tfrecord_dir, 'examples.tfrecord-?????-of-?????.gz'),
        os.path.join(tfrecord_dir, '*.tfrecord.gz'),
        os.path.join(tfrecord_dir, '*.tfrecord-?????-of-?????'),
    ]
    files = []
    for pattern in patterns:
        files = sorted(glob.glob(pattern))
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f'No TFRecord files found in {tfrecord_dir}. '
            f'Tried patterns: {patterns}')

    print(f'Found {len(files)} TFRecord files in {tfrecord_dir}')
    print(f'Extracting {num_samples} pileup images (shape {input_shape})...')

    # Parse TFRecords
    proto_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string),
    }

    images = []
    dataset = tf.data.TFRecordDataset(
        files, compression_type='GZIP', num_parallel_reads=4)
    for raw_record in dataset.take(num_samples):
        parsed = tf.io.parse_single_example(
            serialized=raw_record, features=proto_features)
        image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
        image = tf.reshape(image, input_shape)
        image = tf.cast(image, tf.float32)
        # preprocess_images: (x - 128) / 128 -> range [-1, 1]
        image = (image - 128.0) / 128.0
        images.append(image.numpy()[np.newaxis, ...])

    print(f'Extracted {len(images)} real pileup images.')
    return images, input_shape


def validate(onnx_fp32_path, onnx_int8_path, saved_model_dir=None,
             tfrecord_dir=None, num_samples=500):
    """Compare INT8 model outputs against FP32 ONNX.

    Uses real pileup images from TFRecords when available (preferred),
    falls back to random inputs in [-1, 1] range.

    For dynamic INT8 (weights-only quantization), expected max diff < 1e-3
    on real images. If it exceeds this, something is wrong.
    """
    import numpy as np
    import onnxruntime as ort

    use_real_data = tfrecord_dir is not None
    if use_real_data:
        images, input_shape = load_real_images(
            tfrecord_dir, saved_model_dir, num_samples)
    else:
        # Fall back to random inputs
        input_shape = [100, 221, 7]
        if saved_model_dir:
            import json
            info_path = os.path.join(saved_model_dir, 'example_info.json')
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                input_shape = info.get('shape', input_shape)
        images = [
            np.random.uniform(-1, 1, (1,) + tuple(input_shape)).astype(
                np.float32)
            for _ in range(num_samples)
        ]

    data_source = 'real TFRecord images' if use_real_data else 'random inputs'
    print(f'Validating INT8 vs FP32 with {len(images)} {data_source}, '
          f'shape {input_shape}')

    # Load both models
    sess_fp32 = ort.InferenceSession(
        onnx_fp32_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(
        onnx_int8_path, providers=['CPUExecutionProvider'])

    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8 = sess_int8.get_inputs()[0].name

    max_diff_overall = 0.0
    diffs = []
    for _i, img in enumerate(images):
        fp32_out = sess_fp32.run(None, {input_name_fp32: img})[0]
        int8_out = sess_int8.run(None, {input_name_int8: img})[0]

        max_diff = np.max(np.abs(fp32_out - int8_out))
        diffs.append(max_diff)
        max_diff_overall = max(max_diff_overall, max_diff)

    diffs = np.array(diffs)
    # Dynamic INT8: weights quantized, activations FP32 — tight threshold.
    threshold = 1e-3 if use_real_data else 1e-2
    status = 'PASSED' if max_diff_overall < threshold else 'FAILED'
    print(f'\nValidation {status}:')
    print(f'  Max diff:    {max_diff_overall:.4e} (threshold {threshold:.0e})')
    print(f'  Mean diff:   {np.mean(diffs):.4e}')
    print(f'  Median diff: {np.median(diffs):.4e}')
    print(f'  P99 diff:    {np.percentile(diffs, 99):.4e}')

    if max_diff_overall >= threshold:
        print('\nWARNING: INT8 quantization introduces significant differences.')
        print('Run hap.py accuracy validation before using in production.')
        if use_real_data:
            print('This exceeds the 1e-3 threshold for dynamic INT8 on real '
                  'data — investigate before proceeding.')

    return max_diff_overall


def main():
    parser = argparse.ArgumentParser(
        description='Apply dynamic INT8 quantization to ONNX model')
    parser.add_argument('--input', required=True,
                        help='Input FP32 ONNX model path')
    parser.add_argument('--output', required=True,
                        help='Output INT8 ONNX model path')
    parser.add_argument('--validate', action='store_true',
                        help='Validate INT8 output against FP32')
    parser.add_argument('--saved_model_dir',
                        help='TF SavedModel dir (for input shape from '
                             'example_info.json)')
    parser.add_argument('--tfrecord_dir',
                        help='Directory containing DeepVariant TFRecord files '
                             'for real-data validation (preferred over random '
                             'inputs)')
    parser.add_argument('--num_validation_samples', type=int, default=500,
                        help='Number of samples for validation (default: 500)')
    args = parser.parse_args()

    quantize(args.input, args.output)

    if args.validate:
        validate(args.input, args.output, args.saved_model_dir,
                 args.tfrecord_dir, args.num_validation_samples)


if __name__ == '__main__':
    main()
