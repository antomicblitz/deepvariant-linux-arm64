#!/usr/bin/env python3
"""Benchmark FP32 vs INT8 ONNX inference on ARM64.

Usage (inside Docker container):
  python3 /data/scripts/benchmark_int8_onnx.py \
    --fp32_model /opt/models/wgs/model.onnx \
    --int8_model /data/model_int8_static.onnx \
    --tfrecord_dir /data/output/fp32_run1/intermediate \
    --saved_model_dir /opt/models/wgs
"""
import argparse
import glob
import json
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np


def load_images(tfrecord_dir, saved_model_dir, num_samples=1024):
    """Extract real pileup images from TFRecords."""
    import tensorflow as tf

    input_shape = [100, 221, 7]
    if saved_model_dir:
        info_path = os.path.join(saved_model_dir, 'example_info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            input_shape = info.get('shape', input_shape)

    files = sorted(glob.glob(
        os.path.join(tfrecord_dir, 'make_examples.tfrecord-*-of-*.gz')))
    if not files:
        raise FileNotFoundError(f'No TFRecord files in {tfrecord_dir}')

    print(f'Loading {num_samples} images from {len(files)} TFRecord files...')
    proto_features = {'image/encoded': tf.io.FixedLenFeature((), tf.string)}
    images = []
    dataset = tf.data.TFRecordDataset(
        files, compression_type='GZIP', num_parallel_reads=4)
    for raw in dataset.take(num_samples):
        parsed = tf.io.parse_single_example(
            serialized=raw, features=proto_features)
        image = tf.io.decode_raw(parsed['image/encoded'], tf.uint8)
        image = tf.reshape(image, input_shape)
        image = tf.cast(image, tf.float32)
        image = (image - 128.0) / 128.0
        images.append(image.numpy())
    print(f'Loaded {len(images)} images, shape {input_shape}')
    return images, input_shape


def benchmark_model(model_path, model_name, images, batch_sizes, num_threads):
    """Benchmark a single model across batch sizes."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(
        model_path, so, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name

    results = {}
    for bs in batch_sizes:
        batches = []
        for i in range(0, len(images), bs):
            batch = np.stack(images[i:i + bs])
            if len(batch) == bs:
                batches.append(batch)
        if not batches:
            continue

        # Warmup
        sess.run(None, {input_name: batches[0]})

        # Benchmark: 2 passes
        total_imgs = 0
        start = time.time()
        for _ in range(2):
            for batch in batches:
                sess.run(None, {input_name: batch})
                total_imgs += len(batch)
        elapsed = time.time() - start

        rate = elapsed / total_imgs * 100
        ips = total_imgs / elapsed
        print(f'  {model_name:<16} bs={bs:<4} {rate:.3f} s/100  '
              f'{ips:.1f} img/s  ({total_imgs} imgs in {elapsed:.1f}s)')
        results[bs] = {'rate': rate, 'ips': ips, 'elapsed': elapsed,
                       'total_imgs': total_imgs}

    del sess
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fp32_model', required=True)
    parser.add_argument('--int8_model', required=True)
    parser.add_argument('--tfrecord_dir', required=True)
    parser.add_argument('--saved_model_dir', default=None)
    parser.add_argument('--num_images', type=int, default=1024)
    parser.add_argument('--num_threads', type=int, default=0)
    args = parser.parse_args()

    if args.num_threads == 0:
        args.num_threads = os.cpu_count() or 16

    import onnxruntime as ort
    print(f'ORT: {ort.__version__}')
    print(f'Threads: {args.num_threads}')

    images, _ = load_images(
        args.tfrecord_dir, args.saved_model_dir, args.num_images)

    batch_sizes = [64, 128, 256, 512]

    print(f'\n{"="*70}')
    print('FP32 ONNX:')
    fp32_results = benchmark_model(
        args.fp32_model, 'FP32', images, batch_sizes, args.num_threads)

    print('\nINT8 static:')
    int8_results = benchmark_model(
        args.int8_model, 'INT8', images, batch_sizes, args.num_threads)

    print(f'\n{"="*70}')
    print('Speedup (INT8 / FP32):')
    for bs in batch_sizes:
        if bs in fp32_results and bs in int8_results:
            speedup = fp32_results[bs]['rate'] / int8_results[bs]['rate']
            print(f'  bs={bs}: {speedup:.2f}x  '
                  f'(FP32: {fp32_results[bs]["rate"]:.3f}, '
                  f'INT8: {int8_results[bs]["rate"]:.3f} s/100)')

    # Also compare to TF+OneDNN BF16 baseline (0.232 s/100 measured)
    print('\nReference: TF+OneDNN BF16 = 0.232 s/100 (measured)')
    print('Reference: TF+OneDNN FP32 = 0.379 s/100 (measured)')
    best_int8 = min(int8_results.values(), key=lambda x: x['rate'])
    best_bs = [k for k, v in int8_results.items()
               if v['rate'] == best_int8['rate']][0]
    print(f'\nBest INT8: {best_int8["rate"]:.3f} s/100 (bs={best_bs})')
    print(f'  vs FP32 ONNX: {fp32_results[best_bs]["rate"] / best_int8["rate"]:.2f}x')
    print(f'  vs TF BF16:   {"FASTER" if best_int8["rate"] < 0.232 else "SLOWER"} '
          f'({best_int8["rate"] / 0.232:.2f}x of BF16 rate)')
    print(f'  vs TF FP32:   {"FASTER" if best_int8["rate"] < 0.379 else "SLOWER"} '
          f'({best_int8["rate"] / 0.379:.2f}x of FP32 rate)')


if __name__ == '__main__':
    main()
