#!/usr/bin/env python3
"""DeepVariant benchmark visualization.

Reads benchmark_results.json produced by benchmark.sh and generates
matplotlib charts comparing your Apple Silicon Mac against GCP instances:
  1. A comparable-core GCP instance (n2-standard-16, 8 physical cores, CPU-only)
  2. The published 96-core GCP reference (n2-standard-96, CPU-only)

Key findings from benchmarking on M1 Max:
  - make_examples: Apple Silicon matches or beats an equivalent-core GCP
    instance (~1.06x), showing strong per-core efficiency for this
    embarrassingly parallel stage.
  - call_variants: Metal GPU provides a 4.25x speedup over CPU-only on the
    same hardware (224s vs 950s). Despite the v1.9 "small model" optimization
    that pre-screens easy variants on CPU, GPU still accelerates the full CNN
    inference for hard sites significantly.
  - Overall: Apple Silicon is competitive on a per-core basis for the
    CPU-bound stages, but cannot match cloud instances with many more cores.

Usage:
    python3 scripts/benchmark_viz.py ~/deepvariant-benchmark/benchmark_results.json
    python3 scripts/benchmark_viz.py results.json --show          # interactive
    python3 scripts/benchmark_viz.py results.json -o ./charts/    # custom output dir
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# Published reference numbers from docs/deepvariant-case-study.md (chr20)
# and docs/metrics.md (full genome, 96-CPU GCP instance)
PUBLISHED_CHR20 = {
    'SNP':   {'Recall': 0.996252, 'Precision': 0.999257, 'F1': 0.997752},
    'INDEL': {'Recall': 0.994919, 'Precision': 0.998280, 'F1': 0.996597},
}
PUBLISHED_WGS_FULL = {
    'SNP':   {'Recall': 0.993756, 'Precision': 0.998521, 'F1': 0.996133},
    'INDEL': {'Recall': 0.994105, 'Precision': 0.997591, 'F1': 0.995845},
}

# Published full-genome timings from docs/metrics.md (n2-standard-96, CPU-only)
# in seconds
PUBLISHED_FULL_GENOME_96 = {
    'make_examples': 45 * 60 + 14,       # 45m13.77s
    'call_variants': 16 * 60 + 26,       # 16m25.61s
    'postprocess_variants': 6 * 60 + 51, # 6m51.14s
    'total': 78 * 60 + 58,               # 78m57.99s
}

# Estimated full-genome timings for n2-standard-16 (16 vCPU = 8 physical cores)
#
# Derived from the DeepVariant-on-Spark paper (PMC7481958) which benchmarked
# DeepVariant at 16/32/64/96 CPU counts. We use the ratio of 16-CPU to 96-CPU
# times from that paper, applied to the v1.9 96-CPU official metrics:
#
#   Paper scaling ratios (16-CPU / 96-CPU):
#     make_examples:  6.13h / 1.20h = 5.108x  (nearly linear — embarrassingly parallel)
#     call_variants: 10.80h / 3.83h = 2.820x  (sub-linear — TF inference bottleneck)
#     postprocess:    0.56h / 0.48h = 1.167x  (nearly constant — mostly single-threaded)
#
# Why 16 vCPUs is the fair comparison:
#   n2-standard-16 = 8 physical Intel Cascade Lake cores with hyperthreading (16 vCPUs)
#   Apple M1 Max   = 8 physical high-performance ARM cores (no HT) + 2 efficiency cores
#   Both have 8 physical cores, making this the closest apples-to-apples comparison.
#   The M1 Max additionally has a 32-core Metal GPU for call_variants inference.
ESTIMATED_FULL_GENOME_16 = {
    'make_examples': int(PUBLISHED_FULL_GENOME_96['make_examples'] * 5.108),      # ~3.85h
    'call_variants': int(PUBLISHED_FULL_GENOME_96['call_variants'] * 2.820),      # ~46m
    'postprocess_variants': int(PUBLISHED_FULL_GENOME_96['postprocess_variants'] * 1.167),  # ~8m
}
ESTIMATED_FULL_GENOME_16['total'] = sum(
    ESTIMATED_FULL_GENOME_16[s] for s in ['make_examples', 'call_variants', 'postprocess_variants']
)

STAGE_ORDER = ['make_examples', 'call_variants', 'postprocess_variants', 'total']
STAGE_LABELS = ['make_examples', 'call_variants', 'postprocess\nvariants', 'Total']

# chr20 fraction of the total genome
CHR20_SCALE = 64_444_167 / 3_088_286_401


def format_time(seconds):
    """Format seconds as a human-readable string."""
    if seconds < 60:
        return f'{seconds:.0f}s'
    m, s = divmod(seconds, 60)
    if m < 60:
        return f'{m:.0f}m{s:02.0f}s'
    h, m = divmod(m, 60)
    return f'{h:.0f}h{m:02.0f}m'


def plot_vs_reference(summary, metadata, output_path, show=False, dpi=150):
    """Grouped bar chart: Mac vs 16-vCPU (fair) and 96-vCPU GCP references."""
    mac_means = [summary[s]['mean'] / 60 for s in STAGE_ORDER]
    mac_stds = [summary[s]['std'] / 60 for s in STAGE_ORDER]

    ref16_times = [ESTIMATED_FULL_GENOME_16[s] * CHR20_SCALE / 60 for s in STAGE_ORDER]
    ref96_times = [PUBLISHED_FULL_GENOME_96[s] * CHR20_SCALE / 60 for s in STAGE_ORDER]

    x = np.arange(len(STAGE_ORDER))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_mac = ax.bar(x - width, mac_means, width, yerr=mac_stds,
                      label=f'{metadata["chip"]} (Metal GPU)',
                      color='#2196F3', capsize=3,
                      edgecolor='white', linewidth=0.5)
    bars_16 = ax.bar(x, ref16_times, width,
                     label='GCP n2-standard-16\n(8 phys. cores, CPU-only)\n[estimated]',
                     color='#FF9800', capsize=3,
                     edgecolor='white', linewidth=0.5)
    bars_96 = ax.bar(x + width, ref96_times, width,
                     label='GCP n2-standard-96\n(48 phys. cores, CPU-only)',
                     color='#9E9E9E', capsize=3,
                     edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title(
        f'DeepVariant v{metadata.get("deepvariant_version", "1.9")} — '
        f'{metadata["chip"]} vs GCP Instances\n'
        f'{metadata["sample"]} {metadata["region"]} | '
        f'{metadata["shards"]} shards | '
        f'{metadata["num_runs"]} run(s)',
        fontsize=13
    )
    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    for bars in [bars_mac, bars_16, bars_96]:
        for bar in bars:
            raw_secs = bar.get_height() * 60
            label = format_time(raw_secs)
            ax.annotate(label,
                        xy=(bar.get_x() + bar.get_width() / 2,
                            bar.get_height()),
                        xytext=(0, 4), textcoords='offset points',
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    plt.close()


def plot_speedup_fair(summary, metadata, output_path, show=False, dpi=150):
    """Horizontal bar chart: speedup vs comparable 16-vCPU GCP instance."""
    speedups_16 = []
    speedups_96 = []
    for s in STAGE_ORDER:
        mac_mean = summary[s]['mean']
        ref16_scaled = ESTIMATED_FULL_GENOME_16[s] * CHR20_SCALE
        ref96_scaled = PUBLISHED_FULL_GENOME_96[s] * CHR20_SCALE
        speedups_16.append(ref16_scaled / mac_mean if mac_mean > 0 else 0)
        speedups_96.append(ref96_scaled / mac_mean if mac_mean > 0 else 0)

    colors = ['#4CAF50' if s > 1.0 else '#F44336' for s in speedups_16]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = np.arange(len(STAGE_ORDER))

    bars = ax.barh(y, speedups_16, color=colors, height=0.5,
                   edgecolor='white', linewidth=0.5)

    # Show 96-vCPU reference as small markers
    ax.scatter(speedups_96, y, marker='|', color='#616161', s=200,
               zorder=5, label='vs 96-vCPU GCP')

    ax.set_xlabel('Relative Speed (GCP time / Mac time)', fontsize=12)
    ax.set_title(
        f'{metadata["chip"]} vs GCP n2-standard-16 (8 phys. cores, CPU-only)\n'
        f'{metadata["sample"]} {metadata["region"]} '
        f'(estimated from published scaling data)',
        fontsize=13
    )
    ax.set_yticks(y)
    ax.set_yticklabels(STAGE_LABELS, fontsize=10)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=0.8,
               label='Same speed (1.0x)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    for bar, val_16, val_96 in zip(bars, speedups_16, speedups_96):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val_16:.2f}x', va='center', fontsize=11, fontweight='bold')

    # Time annotations below each bar
    for i, s in enumerate(STAGE_ORDER):
        mac_t = format_time(summary[s]['mean'])
        ref16_t = format_time(ESTIMATED_FULL_GENOME_16[s] * CHR20_SCALE)
        ref96_t = format_time(PUBLISHED_FULL_GENOME_96[s] * CHR20_SCALE)
        ax.text(0.01, i - 0.3,
                f'Mac {mac_t} / 16-vCPU {ref16_t} / 96-vCPU {ref96_t}',
                va='top', ha='left', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_accuracy(happy_results, metadata, output_path, show=False, dpi=150):
    """Table figure: accuracy metrics vs published reference."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    headers = [
        'Type', 'Metric',
        f'This Run\n({metadata["chip"].split()[-1]})',
        'Published chr20\n(96-CPU GCP)',
        'Published WGS\n(96-CPU, all chr)',
        'Delta vs\nchr20 ref'
    ]

    cell_data = []
    cell_colors = []
    metric_map = {
        'Recall': 'METRIC.Recall',
        'Precision': 'METRIC.Precision',
        'F1 Score': 'METRIC.F1_Score',
    }

    for vtype in ['SNP', 'INDEL']:
        for metric_label, metric_key in metric_map.items():
            this_val = happy_results[vtype][metric_key]
            pub_chr20 = PUBLISHED_CHR20[vtype][metric_label.split()[0]]
            pub_wgs = PUBLISHED_WGS_FULL[vtype][metric_label.split()[0]]
            delta = this_val - pub_chr20

            delta_str = f'{delta:+.6f}'
            delta_color = '#E8F5E9' if delta >= 0 else '#FFEBEE'

            row = [
                vtype, metric_label,
                f'{this_val:.6f}',
                f'{pub_chr20:.6f}',
                f'{pub_wgs:.6f}',
                delta_str
            ]
            cell_data.append(row)
            cell_colors.append(
                ['white', 'white', '#E3F2FD', 'white', 'white', delta_color]
            )

    table = ax.table(
        cellText=cell_data,
        colLabels=headers,
        cellColours=cell_colors,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#1565C0')
        table[0, j].set_text_props(color='white', fontweight='bold')

    for i in range(1, len(cell_data) + 1):
        table[i, 0].set_text_props(fontweight='bold')

    ax.set_title(
        f'Accuracy: {metadata["sample"]} {metadata["region"]} — '
        f'DeepVariant v{metadata["deepvariant_version"]} on {metadata["chip"]}\n'
        f'vs. published benchmarks (NIST v4.2.1 truth)',
        fontsize=12, pad=20
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize DeepVariant benchmark results'
    )
    parser.add_argument('results_json',
                        help='Path to benchmark_results.json')
    parser.add_argument('--output-dir', '-o', default=None,
                        help='Directory for PNG output (default: same as JSON)')
    parser.add_argument('--show', action='store_true',
                        help='Display charts interactively')
    parser.add_argument('--dpi', type=int, default=150,
                        help='DPI for saved images (default: 150)')
    args = parser.parse_args()

    with open(args.results_json) as f:
        data = json.load(f)

    output_dir = args.output_dir or os.path.dirname(
        os.path.abspath(args.results_json))
    os.makedirs(output_dir, exist_ok=True)

    metadata = data['metadata']
    summary = data.get('summary', {})
    happy = data.get('happy')

    charts_made = 0

    if summary:
        path1 = os.path.join(output_dir, 'benchmark_vs_reference.png')
        plot_vs_reference(summary, metadata, path1,
                          show=args.show, dpi=args.dpi)
        print(f'Saved: {path1}')
        charts_made += 1

        path2 = os.path.join(output_dir, 'benchmark_speedup.png')
        plot_speedup_fair(summary, metadata, path2,
                          show=args.show, dpi=args.dpi)
        print(f'Saved: {path2}')
        charts_made += 1

    if happy:
        path3 = os.path.join(output_dir, 'benchmark_accuracy.png')
        plot_accuracy(happy, metadata, path3,
                      show=args.show, dpi=args.dpi)
        print(f'Saved: {path3}')
        charts_made += 1

    if charts_made == 0:
        print('No data to visualize.', file=sys.stderr)
        sys.exit(1)

    print(f'\nDone — {charts_made} chart(s) saved to {output_dir}/')


if __name__ == '__main__':
    main()
