# OneDNN ACL Dispatch Bug on AmpereOne — Diagnostic Trail

## Summary

The TF 2.13.1 aarch64 wheel bundles OneDNN with ACL (Arm Compute Library)
statically linked, compiled for Neoverse-N1. On AmpereOne (CPU part `0xac3`,
Armv8.6-A), ACL's advanced ISA dispatch path triggers SIGILL after a
non-deterministic number of `predict_on_batch` calls on the small dense model
(70-dim input, `small_model/inference.py:141`).

**Fix:** `TF_ENABLE_ONEDNN_OPTS=0` for ALL binaries (Eigen fallback).
call_variants also crashes with OneDNN+BF16 on AmpereOne — the SIGILL affects
both the small dense model in make_examples AND the InceptionV3 CNN in
call_variants. Use INT8 ONNX for call_variants instead.

**Why not `ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD`?** This ISA cap works for 2-way
concurrency but still triggers SIGILL under 16-way parallel shards. The bug
persists even in ASIMD-only mode under high concurrency, suggesting the root
cause is deeper than ISA dispatch — possibly memory corruption in ACL's thread
pool or allocator under many concurrent TF sessions.

**Impact:** The small model is a 70-dimensional dense network that runs in
microseconds per call. Disabling OneDNN for it has negligible performance
impact on make_examples. All BF16 speedup (1.61x call_variants) comes from the
large CNN in call_variants, which is unaffected.

## Root Cause: CONFIRMED via GDB (2026-03-07)

**Faulting instruction:** `cntb x0` at `0xffffe48c5a9c` — "Count Bytes in SVE
vector", an **SVE instruction**. AmpereOne has **NO SVE** (no `sve` flag in
`/proc/cpuinfo`).

**Call chain:**
```
TF Conv2D (MklConvOp::Compute)
  -> OneDNN acl_indirect_gemm_convolution
    -> ACL CpuGemmAssemblyDispatch::validate()
      -> has_opt_gemm<float, float, Nothing>
        -> find_implementation()
          -> gemm_fp32_methods lambda #16   <-- contains cntb
            -> cntb x0                      <-- SIGILL
```

ACL v22.11's GEMM method selection iterates through ALL candidates, including
SVE ones. Lambda #16 unconditionally executes `cntb` to check SVE vector length
**before** the candidate is rejected by feature detection. This is a bug in ACL
v22.11.

**BF16-specific:** The SIGILL only occurs when `ONEDNN_DEFAULT_FPMATH_MODE=BF16`
is active. FP32 OneDNN works fine on AmpereOne (tested: small model 16-way pass,
InceptionV3 single batch pass). BF16 mode causes ACL to probe SVE-BF16 GEMM
candidates, which execute `cntb` unconditionally.

**Why non-deterministic for make_examples:** The small model in make_examples
uses dense layers, not Conv2D. The `cntb` probe only triggers for certain
operator types under certain conditions. Under high concurrency (16-way), thread
scheduling variations trigger different ACL code paths, explaining the variable
crash point (20K-58K candidates).

ACL is statically linked into `libtensorflow_cc.so.2` (569 MB). There is no
separate `libarm_compute.so` to replace. The only options are:
1. Disable OneDNN entirely (`TF_ENABLE_ONEDNN_OPTS=0`) — **production fix**
2. Rebuild TF from source with SVE removed from ACL BUILD file — **in progress**

See [docs/onednn-bf16-sigill-analysis.md](onednn-bf16-sigill-analysis.md) for
the full GDB analysis and TF rebuild plan.

## Diagnostic Tests (2026-03-07)

All tests on Oracle A2 VM.Standard.A2.Flex (16 OCPU / 32 vCPU, 64 GB RAM),
AmpereOne (Siryn, CPU part 0xac3), Docker image `deepvariant-arm64:v1.9.0-arm64.2`.

Test data: HG003 chr20, GRCh38 reference, 35x WGS PCR-free BAM.

### Test Matrix

| Test | Shards | Concurrency | ISA Setting | OMP | Result | Crash Point |
|------|--------|-------------|-------------|-----|--------|-------------|
| Single shard | 1 | 1 process | Full (default) | 8 | **SIGILL** | ~40K candidates |
| 2-way no stagger | 2 | 2 concurrent | Full (default) | 16 | **SIGILL** | Task 1: ~20K, Task 0: ~58K |
| 2-way sequential | 2 | 1 at a time | Full (default) | 16 | **SIGILL** | Task 0: ~34K candidates |
| 2-way stagger 3s | 2 | 2 concurrent | Full (default) | 16 | Completed* | *lucky timing — not reliable* |
| 2-way NEON cap | 2 | 2 concurrent | ADVANCED_SIMD | 16 | **PASS** | — |
| 4-way NEON cap | 4 | 4 concurrent | ADVANCED_SIMD | 8 | Partial* | 2/4 shards incomplete |
| **16-way NEON cap** | 16 | 16 concurrent | ADVANCED_SIMD | 2 | **SIGILL** | 4/16 shards crashed |
| **16-way OneDNN OFF** | 16 | 16 concurrent | N/A (Eigen) | 2 | **PASS** | — |

*The staggered test completed because both shards finished their half of chr20
before the buggy dispatch path triggered. This is not a reliable fix — the
crash point is non-deterministic and varies by 3x (20K–58K candidates).*

*`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD` works at low concurrency (2-way) but fails
under 16-way parallel shards. The bug is deeper than ISA dispatch. Only
`TF_ENABLE_ONEDNN_OPTS=0` (complete Eigen fallback) is reliable.*

### Key Observations

1. **Not a concurrency bug.** Single-process and sequential tests both SIGILL.
   Concurrency changes the timing but is not the root cause.

2. **Non-deterministic crash point.** The SIGILL occurs after processing
   20K–58K candidates (thousands of `predict_on_batch` calls). The variation
   suggests a lazy dispatch or warmup-triggered kernel switch in ACL.

3. **Both models affected.** make_examples crashes in the small model
   (`small_model/inference.py:141` → `predict_on_batch`). call_variants also
   crashes with `TF_ENABLE_ONEDNN_OPTS=1` — SIGILL during the first inference
   batch on InceptionV3 (confirmed 2026-03-07). The ACL dispatch bug affects
   all OneDNN code paths on AmpereOne, not just the small dense model.

4. **`ONEDNN_MAX_CPU_ISA=ADVANCED_SIMD` is NOT reliable.** Works at 2-way
   concurrency but fails under 16-way parallel shards (4/16 crashed). The bug
   is deeper than ISA dispatch.

5. **AmpereOne CPU flags include `bf16` and `i8mm`** but NOT `sve` or `sve2`.
   The buggy dispatch path IS SVE-specific — GDB confirms `cntb x0` (SVE
   instruction) is the faulting instruction. ACL v22.11 unconditionally probes
   SVE candidates during BF16 GEMM method selection.

### objdump Analysis (2026-03-06)

```
# Zero BFMMLA/SMMLA/UMMLA/FMLAL instructions in TF binaries
objdump -d libtensorflow_framework.so.2 | grep -cE "(bfmmla|smmla|ummla|fmlal)" → 0
objdump -d _pywrap_tensorflow_internal.so | grep -cE "(bfmmla|smmla|ummla|fmlal)" → 0

# No separate ACL shared objects — statically linked
find / -name "libarm_compute*.so" → (empty)
```

The matrix multiply instructions (BFMMLA etc.) are not present as static code.
The SIGILL is caused by `cntb` (SVE vector length probe), not matrix multiply
instructions. ACL's GEMM candidate selection code contains `cntb` in SVE
candidate lambdas that execute unconditionally before feature checking.

## Production Configuration

```bash
# docker_entrypoint.sh — AmpereOne section
# OneDNN OFF for ALL binaries (ACL SIGILL affects both ME and CV)
if [[ "${_part}" == "0xac3" ]]; then
  export TF_ENABLE_ONEDNN_OPTS=0
fi
```

**call_variants also crashes** with `TF_ENABLE_ONEDNN_OPTS=1` on AmpereOne.
Tested 2026-03-07: CV with BF16 produces SIGILL during the first inference
batch on the InceptionV3 model (different ACL dispatch path than the small
model, but same root cause — N1-targeted ACL on AmpereOne ISA).

Best working backend for AmpereOne call_variants: INT8 ONNX (`--use_onnx`,
0.358 s/100, $2.32/genome with jemalloc).

## Fix: TF Source Rebuild with SVE Removed (CONCLUDED — NO BENEFIT)

Built TF v2.13.1 from source on AmpereOne with SVE/SVE2 **removed** from the
ACL BUILD file (`third_party/compute_library/BUILD`). Build succeeded (44 min,
18,998 Bazel actions). SIGILL eliminated — BF16 mode runs without crashing.

**However, ACL v22.11 has NO NEON-only BF16 GEMM kernels.** All BF16 GEMM in
ACL v22.11 is SVE-only. Removing SVE eliminates the crash but provides no
performance benefit. Worse, it also removes the FP32 SVE GEMM kernels that
were working on the stock wheel.

### Benchmark (2026-03-07, rebuilt wheel, AmpereOne 32 vCPU)

| Config | Rate (s/100) | vs Stock FP32 |
|--------|-------------|---------------|
| Stock wheel, FP32 OneDNN | 0.390 | baseline |
| Stock wheel, BF16 OneDNN | SIGILL | — |
| **Rebuilt wheel (no SVE), BF16** | **0.770** | **2.0x slower** |
| **Rebuilt wheel (no SVE), FP32** | **0.786** | **2.0x slower** |
| Stock wheel, Eigen (OneDNN OFF) | 0.390 | same as FP32 OneDNN |
| INT8 ONNX (current best) | 0.358 | 1.09x faster |

**Conclusion:** BF16 on AmpereOne via ACL v22.11 is not achievable. INT8 ONNX
($2.32/genome) remains the best backend.

## Fix Attempt: ACL v23.08 Upgrade (CONCLUDED — SIGILL PERSISTS)

Upgraded ACL from v22.11 to v23.08 within the TF 2.13.1 build (2026-03-07).
Kept SVE enabled (hypothesis: v23.08 fixed the `cntb` probe). Required three
BUILD file fixes:

1. **Subpackage conflicts:** ACL v23.08 ships `BUILD.bazel` in `src/`, `arm_compute/`,
   `support/`, `utils/`, `tests/`, `examples/`, `scripts/` — all deleted via
   `ctx.delete()` in `third_party/repo.bzl` to prevent Bazel subpackage errors.

2. **Include path ordering:** ACL v23.08 arm_conv code uses `arm_gemm::roundup`
   and `arm_gemm::VLType` from `arm_gemm/utils.hpp`. The BUILD file's `includes`
   had `convolution/common` before `arm_gemm`, causing the wrong `utils.hpp` to
   shadow the arm_gemm one. Fixed by reordering: `arm_gemm` and `arm_conv` first.

3. **OpenCL link error:** v23.08 added `src/runtime/heuristics/` with OpenCL
   references (`clReleaseCommandQueue`). Excluded from build glob.

**Build succeeded** (18,998 actions, ~50 min). Wheel: 193 MB, imports correctly.

### Results

| Test | Result |
|------|--------|
| FP32 OneDNN Conv2D | **PASS** — ACL v23.08 SVE FP32 works on AmpereOne |
| BF16 OneDNN Conv2D | **SIGILL** — same `cntb` crash as v22.11 |
| BF16 + ISA cap (ADVANCED_SIMD) | **SIGILL** — ISA cap doesn't prevent ACL probe |

**ACL v23.08 did NOT fix the `cntb` SVE probe bug.** The BF16 kernel dispatch
path still unconditionally executes `cntb` before feature checking. This is an
ACL architectural issue, not a version-specific bug.

### Final Assessment

BF16 on AmpereOne is **blocked at the ACL level** across both v22.11 and v23.08.
The root cause is ACL's GEMM method selection design: SVE candidate lambdas
execute `cntb` unconditionally during iteration, before the feature-detection
guard can reject them. AmpereOne supports BF16 (BFMMLA works — verified via
standalone test) but NOT SVE. ACL has no code path for BF16-without-SVE dispatch.

**Remaining options (all high-effort, diminishing returns):**
1. ~~Patch ACL's `gemm_fp32.cpp` to skip SVE lambdas when `!has_sve()`~~ — **ATTEMPTED,
   see Fix Attempt 3 below. Fixed SIGILL but uncovered 3 more cascading bugs.**
2. Wait for ACL v24.x+ that may properly guard SVE probes and fix indirect_gemm
3. Use ONNX INT8 ($2.32/genome) — **recommended production path**

## Fix Attempt 3: ACL v23.08 SVE Filter Patch + OneDNN Bypass (2026-03-08)

### Approach

Two patches applied to ACL v23.08 + OneDNN v2.7.3:

1. **ACL `gemm_fp32.cpp`:** Changed two SVE BF16 kernel filter lambdas from
   `has_bf16()` to `has_svebf16()`. These kernels (`sve_hybrid_fp32bf16fp32_mmla_6x4VL`,
   `sve_hybrid_fp32bf16fp32_mmla_4x6VL`) checked `has_bf16()` but NOT `has_sve()`,
   so they passed the filter on AmpereOne (which has BF16 but no SVE). Their
   estimate functions call `svcntb()` (SVE instruction) causing SIGILL.

2. **OneDNN `acl_convolution_utils.cpp`:** Added early return in
   `init_conf_indirect_gemm()` for BF16 fast_math mode. The indirect GEMM path
   (`NEGEMMConv2d` / `CpuGemmDirectConv2d`) crashes because it permutes weights
   with `PermutationVector{3,0,1,2}` before passing to `CpuGemmAssemblyDispatch`,
   which re-validates and fails for BF16 NEON kernels, leaving `_arm_gemm=nullptr`.
   The subsequent `workspace()` call dereferences the null pointer (SIGSEGV).

### Results: 4 Cascading Bugs Found

| Bug | Status | Description |
|-----|--------|-------------|
| 1. SVE filter bug | **FIXED** | `has_bf16()` → `has_svebf16()` for 2 SVE BF16 kernels |
| 2. indirect_gemm null pointer | **FIXED** | Skip indirect_gemm for BF16, fall through to GEMM conv |
| 3. Conv2D falls to `gemm:ref` | **UNFIXED** | `acl_gemm_convolution` also rejects BF16 — Conv2D uses unaccelerated reference GEMM |
| 4. Graph mode crash (remapping) | **UNFIXED** | Grappler remapping (op fusion) + BF16 inner_product causes SIGSEGV in full InceptionV3 model.predict |

### Detailed Bug Analysis

**Bug 1 (FIXED):** In `gemm_fp32.cpp`, the `gemm_fp32_methods` table has two SVE
BF16 hybrid kernels with incorrect filter conditions:
```cpp
// BEFORE (buggy): passes on AmpereOne because has_bf16()=true
"sve_hybrid_fp32bf16fp32_mmla_6x4VL"
  filter: args._fast_mode && args._ci->has_bf16()

// AFTER (fixed): requires both SVE and BF16
  filter: args._fast_mode && args._ci->has_svebf16()
```
Compare with correctly-filtered SVE BF16 kernels that already use `has_svebf16()`.

**Bug 2 (FIXED):** After bug 1 fix, Conv2D with BF16 hits a SIGSEGV (exit 139)
in `CpuGemmAssemblyDispatch::workspace()`. Root cause:
- `CpuGemmDirectConv2d::configure()` permutes conv weights (NHWC→NCHW-like)
- Passes permuted weights to `CpuGemmAssemblyDispatch::configure()`
- Assembly dispatch's `validate()` rejects the permuted dimensions for BF16
- `configure()` silently returns without setting `_arm_gemm`
- `workspace()` dereferences `_arm_gemm` (null) → SIGSEGV
- `ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr)` is compiled out in release builds

**Bug 3 (UNFIXED):** After bypassing indirect_gemm, Conv2D should use
`acl_gemm_convolution` (im2col + standard GEMM). But `acl_init_conf()` calls
`NEGEMMConvolutionLayer::has_opt_impl()` which traces through to
`CpuGemmAssemblyDispatch::has_opt_impl()` → `has_opt_gemm<float,float>()`.
This should find the NEON BF16 fixed-format kernels
(`a64_ffinterleaved_bf16fp32_mmla_8x12`, `a64_ffhybrid_fp32bf16fp32_mmla_4x24`),
which ARE compiled in (symbols verified in `libtensorflow_cc.so.2`) and have
correct filters (`fast_mode && has_bf16()`). Yet Conv2D falls all the way to
`gemm:ref` (reference implementation), which ignores the `attr-fpmath:bf16`
hint and runs plain FP32 math. Performance: **identical to FP32** (5.3ms vs 5.4ms).

**Bug 4 (UNFIXED):** In graph mode (`model.predict`), Grappler's **remapping**
optimization fuses ops (e.g., Conv2D+BiasAdd+Relu → `_FusedConv2D`). These fused
ops go through a different OneDNN code path. The final Dense(3) layer's
inner_product execution with BF16-formatted weights (`AB4a4b` format, reordered
from FP32) crashes with SIGSEGV. Reproduces consistently with InceptionV3 but
NOT with:
- Eager mode (`model(x)` — works)
- Graph mode with remapping disabled (works)
- Standalone Dense(3) in isolation (works)
- Small models with few layers (works)

### Smoke Test Results (after patches, before graph mode crash discovery)

| Test | Result |
|------|--------|
| FP32 OneDNN Conv2D (32x32, ic=3, oc=16) | **PASS** — `indirect_gemm:acl` |
| BF16 OneDNN Conv2D (32x32, ic=3, oc=16) | **PASS** — `gemm:ref` (no BF16 benefit) |
| BF16 OneDNN Conv2D (149x149, ic=32, oc=64) | **PASS** — `gemm:ref` |
| BF16 Conv2D × 100 iterations | **PASS** |
| BF16 MatMul (64x256 × 256x128, eager) | **PASS** — `gemm:jit` with BF16 |
| BF16 Dense(3) (standalone, eager) | **PASS** |
| BF16 InceptionV3 eager `model(x)` | **PASS** |
| BF16 InceptionV3 `model.predict` (graph mode) | **SIGSEGV** — bug 4 |

### Conclusion

BF16 on AmpereOne via TF 2.13.1 + ACL v23.08 + OneDNN v2.7.3 is **not viable**.
The ACL NEON BF16 GEMM kernels exist and are compiled, but the integration between
OneDNN's convolution dispatch, ACL's fixed-format kernel selection, and TF's
Grappler graph optimizer has too many interacting bugs. Each fix reveals the next
layer of failure.

**The patches are preserved for reference but should NOT be deployed:**
- `third_party/compute_library/acl_ampereone_bf16_no_sve.patch`
- `third_party/mkl_dnn/onednn_acl_indirect_gemm_bf16.patch`

**Production recommendation unchanged:** INT8 ONNX ($2.32/genome) for AmpereOne.

See [docs/onednn-bf16-sigill-analysis.md](onednn-bf16-sigill-analysis.md) and
[docs/ampereone-bf16-rebuild.md](ampereone-bf16-rebuild.md) for full details.

## References

- [docs/oracle-a2-sigill.md](oracle-a2-sigill.md) — Original SIGILL investigation
- [docs/oracle-a2-wheel-test.md](oracle-a2-wheel-test.md) — Wheel swap test procedure
- CLAUDE.md § Phase 2.2d — Oracle A2 benchmark results
- AmpereOne CPU flags (corrected 2026-03-07 — NO SVE):
  `fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid
  asimdrdm jscvt fcma lrcpc dcpop sha3 asimddp sha512 asimdfhm dit uscat
  ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 rng bti ecv`
