---
title: 'DeepVariant-ARM64: INT8-quantized variant calling on Linux AArch64'
tags:
  - bioinformatics
  - variant calling
  - ARM64
  - deep learning
  - genomics
  - quantization
authors:
  - name: Antonio Lamb
    orcid: 0000-0003-0885-2585
    affiliation: 1
affiliations:
  - name: Lamb Consulting, Switzerland
    index: 1
date: 8 March 2026
bibliography: paper.bib
archive_doi: "https://doi.org/10.5281/zenodo.18909055"
---

# Summary

DeepVariant [@Poplin2018] is the most accurate open-source variant caller for germline short-read sequencing, but it has no official Linux ARM64 build. This fork provides the first open-source ARM64 port of DeepVariant v1.9.0, including a post-training INT8-quantized ONNX model that is 74% smaller (21 MB vs. 84 MB) and 2.3$\times$ faster at inference than the FP32 baseline — while preserving WGS accuracy (SNP F1 $\Delta$<0.0001 vs. same-platform FP32 on chr20). A parallel `call_variants` wrapper breaks the GEMM saturation ceiling at 16+ threads, yielding an additional 1.7–2.5$\times$ CV speedup. Together, these optimizations enable full 30$\times$ WGS variant calling at \$0.80/genome on Oracle A1 (dedicated ARM64 vCPUs) or \$0.33/genome on Hetzner CAX41 (shared vCPU), compared to ~\$5/genome on the standard x86 configuration.

# Statement of need

ARM64 server adoption is accelerating in genomics, driven by platforms such as AWS Graviton, Oracle Ampere, and Hetzner Cloud ARM. Yet DeepVariant's build system assumes x86 (SSE4/AVX) throughout, and an upstream request for ARM64 support [@DeepVariantIssue834] has remained unresolved since 2024. The only existing ARM64 option is Sentieon DNAscope [@Freed2022] — a proprietary tool requiring a commercial license, which limits access for academic and resource-constrained laboratories.

This fork addresses the gap by patching the Bazel build system, fixing eight upstream compilation errors for ARM64/GCC 13, and shipping a validated Docker image. The enabling technical contribution is INT8 static quantization of DeepVariant's InceptionV3 model using ONNX Runtime with calibration data drawn from real genomic TFRecords. Post-training INT8 quantization typically degrades accuracy by 0.6–3% on vision CNNs [@Nagel2021]; that it does not here — not even in difficult genomic regions such as segmental duplications and homopolymer runs — is the central finding. The explanation is that InceptionV3's large, dense convolutions are unusually quantization-friendly, lacking the depthwise separable layers that cause INT8 degradation in architectures like EfficientNet and MobileNet [@Jacob2018].

An important engineering detail: INT8 quantization can produce probability vectors that do not sum to 1.0. Without correction, `postprocess_variants` crashes with `ValueError: Invalid genotype likelihoods`. The ONNX inference path in this fork renormalizes outputs before passing them to `round_gls()`. This fix is mandatory for INT8 deployment and is included in the Docker image.

# Key results

Accuracy was validated with `rtg vcfeval` [@Cleary2015] against the Genome in a Bottle HG003 v4.2.1 truth set [@Zook2019].

**INT8 quantization preserves WGS accuracy on the same ARM64 platform:**

| Metric | FP32 (chr20) | BF16 (chr20) | INT8 ONNX (chr20) | INT8 Full Genome |
|--------|:------:|:------:|:---------:|:----------------:|
| SNP F1 | 0.9977 | 0.9977 | 0.9978 | 0.9961 |
| INDEL F1 | 0.9961 | 0.9961 | 0.9962 | 0.9956 |

The chr20-to-full-genome drop is expected (harder chromosomes contribute more errors). The gap vs. x86 (~0.9996 SNP F1) likely reflects differences in the ARM64 build environment (TensorFlow version, OneDNN backend, numerical precision paths) rather than INT8 quantization, since the chr20 INT8 result matches or exceeds FP32 and BF16 on the same ARM64 platform.

For WES (HG003 IDT exome 100x), INT8 matches the FP32 baseline (SNP F1 0.9931 vs. 0.9930, Overall F1 0.9923 vs. 0.9924), with a small INDEL F1 reduction ($-0.0038$) within expected post-training quantization variance.

**Five-platform benchmarking corpus:**

| Platform | Best config | \$/genome | Wall time (30x WGS est.) |
|----------|------------|:---------:|:------------------------:|
| AWS Graviton4 (c8g) | INT8 + 4-way CV | \$3.13 | ~2.4 hr |
| AWS Graviton3 (c7g) | BF16 + jemalloc | \$3.43 | ~5.9 hr |
| Oracle A2 (AmpereOne) | INT8 + 4-way CV | \$2.14 | ~4.3 hr |
| Oracle A1 (Altra) | INT8 + 4-way CV | \$0.80 | ~5.0 hr |
| Hetzner CAX41 (Altra) | INT8 + jemalloc | \$0.33 | ~7.7 hr |

# Documented dead ends

An EfficientNet-B3 model swap was built end-to-end but ran 3$\times$ slower than InceptionV3 on ARM64 CPUs despite 3.2$\times$ fewer FLOPs. Depthwise separable convolutions and squeeze-and-excitation blocks have poor GEMM density on CPU NEON hardware, negating their theoretical FLOP advantage. Full details are provided in `docs/TRAINING_EXPERIMENT.md`.

# Software availability and reproducibility

The software is BSD-3 licensed and available at [https://github.com/antomicblitz/deepvariant-linux-arm64](https://github.com/antomicblitz/deepvariant-linux-arm64). A Docker image is published on GitHub Container Registry. Nextflow and Snakemake workflows are provided in `workflows/`. Reproduction scripts (`scripts/benchmark_full_chr20.sh`, `scripts/validate_accuracy.sh`) allow independent verification of all reported accuracy and performance numbers on any ARM64 Linux machine with Docker installed.

# AI usage disclosure

Claude Opus 4 (Anthropic, 2025–2026) was used for code generation assistance, build system debugging, documentation drafting, and test scaffolding throughout this project. All AI-assisted outputs were reviewed, validated, and modified by the author. Core design decisions — including the INT8 renormalization fix for `postprocess_variants`, platform-specific backend selection logic (OneDNN/SIGILL avoidance on AmpereOne), the parallel `call_variants` architecture, and the GEMM saturation analysis — were made by the human author. Benchmark data was collected and validated independently by the author on physical cloud instances across five ARM64 platforms. The AI tool was not used to generate or fabricate any experimental results.

# Acknowledgements

The author thanks Google for releasing DeepVariant as open-source software, the Genome in a Bottle Consortium for the HG003 truth set, and the ONNX Runtime team for ARM64 INT8 quantization support.

# References
