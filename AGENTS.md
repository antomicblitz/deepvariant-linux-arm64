# Agent Instructions for DeepVariant Linux ARM64

**Behavioral rules and decision frameworks for AI coding agents.**
For technical workflows and coding standards, see [CONTRIBUTING.md](CONTRIBUTING.md).
For ARM64 architecture details, see [docs/architecture-arm64.md](docs/architecture-arm64.md).

## Project Overview

ARM64 Linux port of [google/deepvariant](https://github.com/google/deepvariant) v1.9.0
with hardware-accelerated inference (OneDNN+ACL, BF16, INT8 ONNX). Targets Graviton,
Ampere, Jetson, and RK3588 platforms. Build system: Bazel 5.3.0 + TF 2.13.1.

## Core Rules

- Follow SDLC principles: maintainability, modularity, reusability
- **Never assume missing context** — ask if uncertain about requirements
- **Never hallucinate libraries** — verify in `pyproject.toml`
- **Always confirm file paths exist** before referencing
- **Document new patterns** in AGENT_LEARNINGS.md

## Decision Framework

**Priority:** User instructions > AGENTS.md > CONTRIBUTING.md > Project patterns

**Anti-Scope-Creep:**
- Do NOT implement features without explicit requirements
- Architecture docs (`docs/architecture-arm64.md`) are reference, not task lists
- Dead ends are documented — do NOT revisit them without new evidence

## Architecture Quick Reference

**Build:** Bazel 5.3.0 (`BUILD`, `WORKSPACE`, `.bazelrc`)
**Docker:** `docker/Dockerfile.arm64` (build), `docker/Dockerfile.arm64.runtime` (deploy)
**Scripts:** `scripts/` — benchmarks, model conversion, run wrappers (Python + Bash)
**Core:** `deepvariant/` — upstream source with ARM64 modifications
**Settings:** `scripts/build/settings_arm64.sh` — ARM64 build flags, OneDNN+ACL, BF16

### Inference Backends (Priority)

1. **TF+OneDNN+ACL** — production default, best on Neoverse-N1/N2
2. **TF+OneDNN BF16** — 38% CV speedup on Graviton3+ (Neoverse V1/V2)
3. **INT8 ONNX (static)** — 2.3x over ONNX FP32, matches BF16, low memory for parallel CV
4. **TF Eigen** — fallback for AmpereOne (OneDNN causes SIGILL)

### Known Dead Ends (Do NOT Revisit)

- EfficientNet-B3: 3x slower than InceptionV3 on CPU (depthwise conv penalty)
- ONNX dynamic INT8: ConvInteger op missing on ARM64
- ONNX ACL ExecutionProvider: 16 ops, fragile builds
- KMP_AFFINITY tuning: 30% regression
- fast_pipeline at 16 vCPU: 42% slower (CPU contention)

## Quality Thresholds

Before starting any task:
- **Context**: 8/10 — understand requirements, codebase, dependencies
- **Clarity**: 7/10 — clear implementation path
- **Alignment**: 8/10 — follows project patterns
- **Success**: 7/10 — confident in completing correctly

Below threshold: gather more context or ask the user.

## Mandatory Compliance

1. **Use `make` recipes** — see [CONTRIBUTING.md](CONTRIBUTING.md) command reference
2. **Run `make validate`** before task completion
3. **Fix ALL lint/type errors** before proceeding
4. **Write tests** for new Python functionality
5. **Update AGENT_LEARNINGS.md** when discovering new patterns

## Agent Quick Reference

**Pre-Task:** Read AGENTS.md > CONTRIBUTING.md. Verify quality thresholds.
**During Task:** Use make commands. Follow existing patterns.
**Post-Task:** Run `make validate`. Document new patterns. Review: did we forget anything?

## Claude Code Infrastructure

**Rules** (`.claude/rules/`): Core principles and context management
**Skills** (`.claude/skills/`): reviewing-code, researching-codebase, committing-staged-with-message
