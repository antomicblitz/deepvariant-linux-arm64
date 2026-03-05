#!/usr/bin/env python3
# Copyright 2024 deepvariant-linux-arm64 contributors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Convert DeepVariant TF SavedModel to ONNX format.

Usage:
  # Convert a model
  python scripts/convert_model_onnx.py \
    --saved_model_dir /opt/models/wgs \
    --output /opt/models/wgs/model.onnx

  # Convert and validate against TF
  python scripts/convert_model_onnx.py \
    --saved_model_dir /opt/models/wgs \
    --output /opt/models/wgs/model.onnx \
    --validate --num_validation_samples 100

  # Convert all standard models
  python scripts/convert_model_onnx.py --convert_all --models_dir /opt/models
"""

import argparse
import json
import os
import subprocess
import sys
import time


def convert(saved_model_dir, output_path, opset=17):
    """Convert a TF SavedModel to ONNX format using tf2onnx."""
    print(f'Converting {saved_model_dir} -> {output_path} (opset {opset})')
    cmd = [
        sys.executable, '-m', 'tf2onnx.convert',
        '--saved-model', saved_model_dir,
        '--output', output_path,
        '--opset', str(opset),
    ]
    start = time.time()
    subprocess.run(cmd, check=True)
    elapsed = time.time() - start
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f'Conversion complete: {size_mb:.1f} MB in {elapsed:.1f}s')
    return output_path


def get_input_shape(saved_model_dir):
    """Read input shape from example_info.json alongside the SavedModel."""
    info_path = os.path.join(saved_model_dir, 'example_info.json')
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        return info.get('shape', [100, 221, 7])
    return [100, 221, 7]


def validate(saved_model_dir, onnx_path, num_samples=100):
    """Validate ONNX model produces identical logits to TF SavedModel."""
    import numpy as np

    # Lazy imports — only needed for validation
    import tensorflow as tf
    import onnxruntime as ort

    input_shape = get_input_shape(saved_model_dir)
    print(f'Validating with input shape {input_shape}, {num_samples} samples')

    # Load TF model
    tf_model = tf.saved_model.load(saved_model_dir)
    infer = tf_model.signatures['serving_default']

    # Load ONNX model
    sess = ort.InferenceSession(onnx_path,
                                providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    print(f'ONNX input name: {input_name}')

    max_diff_overall = 0.0
    for i in range(num_samples):
        # Random inputs in DeepVariant's preprocessed range [-1, 1]
        dummy = np.random.uniform(-1, 1,
                                  (1,) + tuple(input_shape)).astype(np.float32)

        # TF inference
        tf_out = infer(tf.constant(dummy))['classification'].numpy()

        # ONNX inference
        onnx_out = sess.run(None, {input_name: dummy})[0]

        max_diff = np.max(np.abs(tf_out - onnx_out))
        max_diff_overall = max(max_diff_overall, max_diff)

        if max_diff > 1e-5:
            print(f'FAIL: Sample {i}: max diff {max_diff:.2e} > 1e-5')
            print(f'  TF output:   {tf_out[0]}')
            print(f'  ONNX output: {onnx_out[0]}')
            raise ValueError(
                f'Validation failed at sample {i}: max diff {max_diff:.2e}')

    print(f'Validation PASSED: {num_samples} samples, '
          f'max diff {max_diff_overall:.2e} < 1e-5')
    return max_diff_overall


def convert_all(models_dir, opset=17, do_validate=False,
                num_validation_samples=100):
    """Convert all standard DeepVariant models in a directory."""
    model_types = ['wgs', 'wes', 'pacbio', 'hybrid_pacbio_illumina',
                   'ont_r104', 'masseq']
    results = {}
    for model_type in model_types:
        saved_model_dir = os.path.join(models_dir, model_type)
        if not os.path.exists(os.path.join(saved_model_dir, 'saved_model.pb')):
            print(f'Skipping {model_type}: no saved_model.pb found')
            continue

        output_path = os.path.join(saved_model_dir, 'model.onnx')
        convert(saved_model_dir, output_path, opset)

        if do_validate:
            max_diff = validate(saved_model_dir, output_path,
                                num_validation_samples)
            results[model_type] = max_diff
        else:
            results[model_type] = 'converted (not validated)'

    print('\n=== Conversion Summary ===')
    for model_type, result in results.items():
        if isinstance(result, float):
            print(f'  {model_type}: max diff {result:.2e}')
        else:
            print(f'  {model_type}: {result}')
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Convert DeepVariant SavedModel to ONNX format')
    parser.add_argument('--saved_model_dir',
                        help='Path to TF SavedModel directory')
    parser.add_argument('--output',
                        help='Output .onnx file path')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--validate', action='store_true',
                        help='Validate ONNX output matches TF')
    parser.add_argument('--num_validation_samples', type=int, default=100,
                        help='Number of random samples for validation')
    parser.add_argument('--convert_all', action='store_true',
                        help='Convert all standard models')
    parser.add_argument('--models_dir', default='/opt/models',
                        help='Models root directory (for --convert_all)')
    args = parser.parse_args()

    if args.convert_all:
        convert_all(args.models_dir, args.opset, args.validate,
                    args.num_validation_samples)
    elif args.saved_model_dir and args.output:
        convert(args.saved_model_dir, args.output, args.opset)
        if args.validate:
            validate(args.saved_model_dir, args.output,
                     args.num_validation_samples)
    else:
        parser.error('Either --convert_all or both --saved_model_dir and '
                     '--output are required')


if __name__ == '__main__':
    main()
