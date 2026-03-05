#!/usr/bin/env python3
"""Generate benchmark visualization charts for README."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette
C_FP32 = '#4A90D9'
C_BF16 = '#E85D75'
C_ONNX = '#7B8794'
C_ME = '#5B9BD5'
C_CV = '#ED7D31'
C_PP = '#A5A5A5'
C_GRAVITON = '#FF9900'  # AWS orange
C_GCP = '#4285F4'       # GCP blue

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


def chart1_call_variants_rate():
    """Bar chart: call_variants inference rate across platforms."""
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = [
        'GCP t2a-std-8\nNeoverse-N1\n8 vCPU, FP32',
        'GCP t2a-std-16\nNeoverse-N1\n16 vCPU, FP32',
        'Graviton3\nc7g.4xlarge\n16 vCPU, FP32',
        'Graviton3\nc7g.4xlarge\n16 vCPU, BF16',
    ]
    rates = [0.880, 0.512, 0.379, 0.232]
    colors = [C_GCP, C_GCP, C_GRAVITON, C_BF16]

    bars = ax.bar(configs, rates, color=colors, width=0.6, edgecolor='white', linewidth=0.5)

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add speedup annotation
    ax.annotate('1.61x faster', xy=(3, 0.232), xytext=(2.5, 0.55),
                fontsize=12, fontweight='bold', color=C_BF16,
                arrowprops=dict(arrowstyle='->', color=C_BF16, lw=1.5),
                ha='center')

    ax.set_ylabel('Seconds per 100 examples (lower is better)', fontsize=12)
    ax.set_title('DeepVariant call_variants Inference Rate', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'call_variants_rate.png'), bbox_inches='tight')
    plt.close()
    print('Generated call_variants_rate.png')


def chart2_wall_time_breakdown():
    """Stacked bar chart: per-stage wall time for Graviton3 FP32 vs BF16."""
    fig, ax = plt.subplots(figsize=(7, 5))

    labels = ['FP32', 'BF16']
    me = [255, 278]
    cv = [298, 185]
    pp = [29, 24]

    x = np.arange(len(labels))
    w = 0.45

    bars_me = ax.bar(x, me, w, label='make_examples', color=C_ME)
    bars_cv = ax.bar(x, cv, w, bottom=me, label='call_variants', color=C_CV)
    bars_pp = ax.bar(x, pp, w, bottom=[m+c for m,c in zip(me,cv)], label='postprocess_variants', color=C_PP)

    # Add time labels inside bars
    for i, (m, c, p) in enumerate(zip(me, cv, pp)):
        ax.text(i, m/2, f'{m}s', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
        ax.text(i, m + c/2, f'{c}s', ha='center', va='center', fontweight='bold', color='white', fontsize=11)
        ax.text(i, m + c + p/2, f'{p}s', ha='center', va='center', fontsize=9, color='#333')

    # Total labels
    totals = [sum(x) for x in zip(me, cv, pp)]
    offsets = [-0.15, 0.15]  # offset to avoid overlap
    for i, total in enumerate(totals):
        ax.text(i, total + 10, f'{total}s total\n({total//60}m{total%60}s)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Speedup arrow between CV bars
    ax.annotate('', xy=(1, 278+185), xytext=(0, 255+298),
                arrowprops=dict(arrowstyle='->', color=C_BF16, lw=2))
    ax.text(0.5, 430, '38% faster\ncall_variants', ha='center', fontsize=10,
            fontweight='bold', color=C_BF16)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.set_ylabel('Wall time (seconds)', fontsize=12)
    ax.set_title('AWS Graviton3 (c7g.4xlarge, 16 vCPU)\nFull chr20 Pipeline Breakdown', fontsize=13, fontweight='bold', pad=15)
    ax.set_ylim(0, 650)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'graviton3_wall_time.png'), bbox_inches='tight')
    plt.close()
    print('Generated graviton3_wall_time.png')


def chart3_accuracy():
    """Grouped bar chart: SNP and INDEL F1 for FP32 vs BF16."""
    fig, ax = plt.subplots(figsize=(6, 4.5))

    labels = ['SNP F1', 'INDEL F1']
    fp32 = [0.9974, 0.9940]
    bf16 = [0.9974, 0.9940]

    x = np.arange(len(labels))
    w = 0.3

    bars1 = ax.bar(x - w/2, fp32, w, label='FP32', color=C_FP32)
    bars2 = ax.bar(x + w/2, bf16, w, label='BF16', color=C_BF16)

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                    f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('Variant Calling Accuracy (chr20, GIAB HG003)\nBF16 = FP32 (zero degradation)', fontsize=12, fontweight='bold', pad=15)
    ax.set_ylim(0.990, 1.001)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'accuracy_fp32_vs_bf16.png'), bbox_inches='tight')
    plt.close()
    print('Generated accuracy_fp32_vs_bf16.png')


def chart4_cost_comparison():
    """Horizontal bar chart: estimated cost per genome across platforms."""
    fig, ax = plt.subplots(figsize=(9, 4))

    # Only include: Google official reference, and our measured Graviton3 data
    # Google n2-standard-96: 79 min official benchmark, $3.8074/hr = $5.01/genome
    # Graviton3 FP32: 581s/chr20 * 48.1 = 27945s = 7.76hr * $0.58/hr = $4.50/genome
    # Graviton3 BF16: 486s/chr20 * 48.1 = 23374s = 6.49hr * $0.58/hr = $3.76/genome
    platforms = [
        'GCP n2-std-96 (x86)\n96 vCPU, ~79 min *',
        'Graviton3 FP32\n16 vCPU, ~7.8 hr **',
        'Graviton3 BF16\n16 vCPU, ~6.5 hr **',
    ]
    costs = [5.01, 4.50, 3.76]
    colors_list = [C_GCP, C_GRAVITON, C_BF16]

    bars = ax.barh(platforms, costs, color=colors_list, height=0.55, edgecolor='white')

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'${cost:.2f}', va='center', fontweight='bold', fontsize=11)

    ax.set_xlabel('Estimated cost per 30x human genome (USD, on-demand)', fontsize=11)
    ax.set_title('Cost per Genome Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 7)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

    # Footnotes
    fig.text(0.12, -0.02, '* Google official benchmark (n2-standard-96, $3.81/hr)',
             fontsize=8, color='#666')
    fig.text(0.12, -0.07, '** Extrapolated from measured chr20 wall time x 48.1 (c7g.4xlarge, $0.58/hr)',
             fontsize=8, color='#666')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'cost_per_genome.png'), bbox_inches='tight')
    plt.close()
    print('Generated cost_per_genome.png')


if __name__ == '__main__':
    chart1_call_variants_rate()
    chart2_wall_time_breakdown()
    chart3_accuracy()
    chart4_cost_comparison()
    print(f'\nAll charts saved to {os.path.abspath(OUT_DIR)}/')
