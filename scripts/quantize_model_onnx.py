#!/usr/bin/env python3
# Copyright 2024 deepvariant-linux-arm64 contributors.
#
# BSD-3-Clause license (same as upstream DeepVariant).
"""Apply INT8 post-training quantization to an ONNX model.

Dynamic quantization reduces model size and can improve inference speed
on ARM64 CPUs with dot-product instructions (Neoverse N1+).

Usage:
  python scripts/quantize_model_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8.onnx

  # Quantize and validate against original
  python scripts/quantize_model_onnx.py \
    --input /opt/models/wgs/model.onnx \
    --output /opt/models/wgs/model_int8.onnx \
    --validate --saved_model_dir /opt/models/wgs

NOTE: Run this AFTER validating FP32 ONNX accuracy with convert_model_onnx.py.
INT8 quantization may introduce small accuracy differences — always re-validate
with hap.py before using in production.
"""

import argparse
import os


def quantize(input_path, output_path):
    """Apply dynamic INT8 quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

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


def validate(onnx_fp32_path, onnx_int8_path, saved_model_dir=None,
             num_samples=100):
    """Compare INT8 model outputs against FP32 ONNX (and optionally TF)."""
    import numpy as np
    import onnxruntime as ort

    # Determine input shape
    input_shape = [100, 221, 7]
    if saved_model_dir:
        import json
        info_path = os.path.join(saved_model_dir, 'example_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            input_shape = info.get('shape', input_shape)

    print(f'Validating INT8 vs FP32 with {num_samples} samples, '
          f'shape {input_shape}')

    # Load both models
    sess_fp32 = ort.InferenceSession(
        onnx_fp32_path, providers=['CPUExecutionProvider'])
    sess_int8 = ort.InferenceSession(
        onnx_int8_path, providers=['CPUExecutionProvider'])

    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8 = sess_int8.get_inputs()[0].name

    max_diff_overall = 0.0
    for i in range(num_samples):
        dummy = np.random.uniform(-1, 1,
                                  (1,) + tuple(input_shape)).astype(np.float32)

        fp32_out = sess_fp32.run(None, {input_name_fp32: dummy})[0]
        int8_out = sess_int8.run(None, {input_name_int8: dummy})[0]

        max_diff = np.max(np.abs(fp32_out - int8_out))
        max_diff_overall = max(max_diff_overall, max_diff)

    # INT8 has looser tolerance than FP32 conversion
    threshold = 1e-2
    status = 'PASSED' if max_diff_overall < threshold else 'WARNING'
    print(f'Validation {status}: max diff {max_diff_overall:.4e} '
          f'(threshold {threshold:.0e})')
    if max_diff_overall >= threshold:
        print('WARNING: INT8 quantization introduces significant differences.')
        print('Run hap.py accuracy validation before using in production.')
    return max_diff_overall


def main():
    parser = argparse.ArgumentParser(
        description='Apply INT8 quantization to ONNX model')
    parser.add_argument('--input', required=True,
                        help='Input FP32 ONNX model path')
    parser.add_argument('--output', required=True,
                        help='Output INT8 ONNX model path')
    parser.add_argument('--validate', action='store_true',
                        help='Validate INT8 output against FP32')
    parser.add_argument('--saved_model_dir',
                        help='TF SavedModel dir (for input shape detection)')
    parser.add_argument('--num_validation_samples', type=int, default=100,
                        help='Number of samples for validation')
    args = parser.parse_args()

    quantize(args.input, args.output)

    if args.validate:
        validate(args.input, args.output, args.saved_model_dir,
                 args.num_validation_samples)


if __name__ == '__main__':
    main()
