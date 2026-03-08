#!/usr/bin/env python3
# Copyright 2024 deepvariant-linux-arm64 contributors.
#
# BSD-3-Clause license (same as upstream DeepVariant).
"""Apply static INT8 quantization to an ONNX model using real calibration data.

Static quantization quantizes BOTH weights AND activations to INT8, requiring
calibration data to determine activation ranges. This gives 2-4x speedup
(vs 1.5-2x for dynamic) but requires careful accuracy validation.

IMPORTANT: InceptionV3 is notably fragile to quantize. Published results show
ImageNet accuracy can drop from 74.6% to 21.0% with bad calibration (MinMax).
Use Percentile calibration (default) — NOT MinMax — for InceptionV3.

Usage:
  # Basic static quantization with Percentile calibration
  python scripts/quantize_static_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8_static.onnx \
    --tfrecord_dir /data/output/examples \
    --saved_model_dir /opt/models/wgs

  # With Entropy calibration (fallback if Percentile fails accuracy)
  python scripts/quantize_static_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8_static.onnx \
    --tfrecord_dir /data/output/examples \
    --saved_model_dir /opt/models/wgs \
    --calibration_method entropy

  # Custom number of calibration samples
  python scripts/quantize_static_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8_static.onnx \
    --tfrecord_dir /data/output/examples \
    --saved_model_dir /opt/models/wgs \
    --num_calibration_samples 1000

NOTE: Accuracy validation for static INT8 should use hap.py F1 scores, NOT
raw numerical thresholds. Static INT8 quantization error is inherently larger
than dynamic — tight numerical thresholds are not meaningful. Run the full
pipeline on HG002 and evaluate with hap.py.
"""

import argparse
import glob
import json
import os


def _check_ort_version():
    """Verify ORT >= 1.17.0 for ARM64 INT8 MLAS kernel support."""
    import onnxruntime as ort
    from packaging.version import Version
    if Version(ort.__version__) < Version('1.17.0'):
        raise RuntimeError(
            f'ONNX Runtime >= 1.17.0 required for ARM64 INT8 MLAS kernels '
            f'(SMMLA support). Found: {ort.__version__}')


class DeepVariantCalibrationDataReader:
    """Reads pileup images from DeepVariant TFRecords for ONNX calibration.

    Streams images lazily to avoid loading all calibration data into memory.
    Images are preprocessed identically to call_variants.py:
    uint8 -> float32 -> (x - 128) / 128 -> range [-1, 1].
    """

    def __init__(self, tfrecord_dir, saved_model_dir, num_samples=500):
        import numpy as np
        from onnxruntime.quantization import CalibrationDataReader  # noqa: F401

        self._np = np

        # Get input shape
        self._input_shape = [100, 221, 7]
        if saved_model_dir:
            info_path = os.path.join(saved_model_dir, 'example_info.json')
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                self._input_shape = info.get('shape', self._input_shape)

        # Find TFRecord files
        patterns = [
            os.path.join(tfrecord_dir,
                         'examples.tfrecord-?????-of-?????.gz'),
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

        print(f'Calibration: {len(files)} TFRecord files, '
              f'{num_samples} samples, shape {self._input_shape}')

        # Pre-load calibration images (required by ORT — CalibrationDataReader
        # must support get_next() returning dicts).
        self._images = self._load_images(files, num_samples)
        self._index = 0
        print(f'Loaded {len(self._images)} calibration images.')

    def _load_images(self, files, num_samples):
        """Extract preprocessed pileup images from TFRecords."""
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
        import numpy as np
        import tensorflow as tf  # noqa: E402

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
            image = tf.reshape(image, self._input_shape)
            image = tf.cast(image, tf.float32)
            # preprocess_images: (x - 128) / 128 -> range [-1, 1]
            image = (image - 128.0) / 128.0
            images.append(image.numpy()[np.newaxis, ...])

        return images

    def get_next(self):
        """Return next calibration sample as {input_name: ndarray} dict."""
        if self._index >= len(self._images):
            return None
        img = self._images[self._index]
        self._index += 1
        # Input name will be set by the quantizer from the model.
        return {'input': img}

    def set_input_name(self, input_name):
        """Called by the quantizer to set the actual model input name."""
        self._input_name = input_name

    def rewind(self):
        """Reset iterator to beginning."""
        self._index = 0


def get_input_name(model_path):
    """Get the input tensor name from an ONNX model."""
    import onnxruntime as ort
    sess = ort.InferenceSession(
        model_path, providers=['CPUExecutionProvider'])
    return sess.get_inputs()[0].name


def quantize_static(input_path, output_path, calibration_reader,
                    calibration_method='percentile',
                    percentile_value=99.99):
    """Apply static INT8 quantization with calibration data.

    Args:
        input_path: Path to FP32 ONNX model.
        output_path: Path for quantized INT8 ONNX model.
        calibration_reader: DeepVariantCalibrationDataReader instance.
        calibration_method: 'percentile' (default, recommended for InceptionV3)
            or 'entropy'. Do NOT use 'minmax' — InceptionV3's Inception modules
            with concatenated branches produce long-tailed activations that
            cause severe quantization error with MinMax.
        percentile_value: Percentile for clipping (default 99.99). Only used
            when calibration_method='percentile'.
    """
    from onnxruntime.quantization import CalibrationMethod, QuantType
    from onnxruntime.quantization import quantize_static as ort_quantize_static

    # Map method name to enum
    method_map = {
        'percentile': CalibrationMethod.Percentile,
        'entropy': CalibrationMethod.Entropy,
        'minmax': CalibrationMethod.MinMax,
    }
    if calibration_method not in method_map:
        raise ValueError(
            f'Unknown calibration method: {calibration_method}. '
            f'Use one of: {list(method_map.keys())}')

    if calibration_method == 'minmax':
        print('WARNING: MinMax calibration is NOT recommended for InceptionV3.')
        print('InceptionV3 Inception modules produce long-tailed activations.')
        print('Consider using --calibration_method=percentile instead.')

    cal_method = method_map[calibration_method]
    print(f'Static INT8 quantization: {input_path} -> {output_path}')
    print(f'  Calibration method: {calibration_method}')
    if calibration_method == 'percentile':
        print(f'  Percentile value: {percentile_value}')

    # Update calibration reader with actual input name
    input_name = get_input_name(input_path)

    # Patch get_next to use the correct input name
    original_get_next = calibration_reader.get_next

    def patched_get_next():
        result = original_get_next()
        if result is None:
            return None
        # Replace generic 'input' key with actual model input name
        img = list(result.values())[0]
        return {input_name: img}

    calibration_reader.get_next = patched_get_next

    # Build extra options for percentile calibration
    extra_options = {}
    if calibration_method == 'percentile':
        extra_options['CalibMovingAverage'] = True
        extra_options['CalibMovingAverageConstant'] = 0.01

    _check_ort_version()
    ort_quantize_static(
        input_path,
        output_path,
        calibration_reader,
        quant_format=None,  # Use default (QDQ for opset >= 13)
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=cal_method,
        extra_options=extra_options,
    )

    orig_size = os.path.getsize(input_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    reduction = (1 - quant_size / orig_size) * 100
    print(f'\nOriginal: {orig_size:.1f} MB')
    print(f'Quantized: {quant_size:.1f} MB ({reduction:.1f}% smaller)')
    print('\nIMPORTANT: Validate accuracy with hap.py on HG002 before use.')
    print('  SNP F1 >= 0.9995, INDEL F1 >= 0.9945')
    print('  Include GIAB stratified BED files (low-complexity, satellites,')
    print('  tandem repeats, homopolymers, segmental duplications).')


def main():
    parser = argparse.ArgumentParser(
        description='Apply static INT8 quantization to ONNX model')
    parser.add_argument('--input', required=True,
                        help='Input FP32 ONNX model path')
    parser.add_argument('--output', required=True,
                        help='Output INT8 ONNX model path')
    parser.add_argument('--tfrecord_dir', required=True,
                        help='Directory containing DeepVariant TFRecord files '
                             'for calibration')
    parser.add_argument('--saved_model_dir',
                        help='TF SavedModel dir (for input shape from '
                             'example_info.json)')
    parser.add_argument('--num_calibration_samples', type=int, default=500,
                        help='Number of calibration samples (default: 500)')
    parser.add_argument('--calibration_method', default='percentile',
                        choices=['percentile', 'entropy', 'minmax'],
                        help='Calibration method (default: percentile). '
                             'Do NOT use minmax for InceptionV3.')
    parser.add_argument('--percentile_value', type=float, default=99.99,
                        help='Percentile for clipping when using percentile '
                             'calibration (default: 99.99)')
    args = parser.parse_args()

    calibration_reader = DeepVariantCalibrationDataReader(
        args.tfrecord_dir, args.saved_model_dir,
        args.num_calibration_samples)

    quantize_static(
        args.input, args.output, calibration_reader,
        args.calibration_method, args.percentile_value)


if __name__ == '__main__':
    main()
