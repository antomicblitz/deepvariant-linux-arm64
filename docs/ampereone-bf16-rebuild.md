# AmpereOne BF16 Rebuild — Root Cause Analysis and Fix Plan

## Root Cause: CONFIRMED via GDB (2026-03-07)

### The faulting instruction

```
=> 0xffffe48c5a9c: cntb x0
```

`cntb` = "Count Bytes in SVE vector" — an **SVE instruction**.
AmpereOne does **NOT** have SVE (confirmed: no `sve` flag in `/proc/cpuinfo`).

### Call chain

```
TF Conv2D (MklConvOp::Compute)
  -> OneDNN acl_indirect_gemm_convolution
    -> ACL CpuGemmAssemblyDispatch::validate()
      -> has_opt_gemm<float, float, Nothing>
        -> find_implementation()
          -> gemm_fp32_methods lambda #16   <-- contains cntb
            -> cntb x0                      <-- SIGILL
```

ACL v22.11's GEMM method selection iterates through ALL candidates, including SVE
ones. Lambda #16 unconditionally executes `cntb` to check SVE vector length
**before** being rejected as a candidate. This is a bug in ACL v22.11 — the probe
should be guarded by SVE feature detection.

### AmpereOne CPU flags (corrected)

Previous docs (`oracle-a2-sigill.md`) incorrectly listed `sve` in the flags.
Actual flags from this instance:

```
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid
asimdrdm jscvt fcma lrcpc dcpop sha3 asimddp sha512 asimdfhm dit uscat
ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 rng bti ecv
```

**NO `sve`. NO `sve2`.** Has `i8mm` and `bf16` (NEON extensions, not SVE).

### Test matrix (Oracle A2, AmpereOne 0xac3, 32 vCPU, 64 GB)

| Test | OneDNN | FPMATH | Result |
|------|--------|--------|--------|
| Small model 16-way x 64K preds | ON | FP32 | **PASS** |
| InceptionV3 single batch | ON | FP32 | **PASS** |
| InceptionV3 single batch | ON | **BF16** | **SIGILL** (`cntb x0`) |
| InceptionV3 single batch | OFF | N/A | **PASS** (Eigen) |

### Benchmark: FP32 OneDNN vs Eigen (single shard, ~20K examples)

| Backend | CV time | Rate |
|---------|---------|------|
| Eigen (TF_ENABLE_ONEDNN_OPTS=0) | 307s | ~0.39 s/100 |
| FP32 OneDNN (TF_ENABLE_ONEDNN_OPTS=1) | 307s | ~0.39 s/100 |

**FP32 OneDNN = Eigen** on AmpereOne. No speedup without BF16.

## Why BF16 would help (if the SVE probe is fixed)

AmpereOne has `bf16` and `i8mm` CPU flags. These enable **NEON** BF16 instructions
(BFDOT, BFMMLA) — NOT SVE BF16. BFMMLA is an Armv8.6-A Advanced SIMD instruction
that works without SVE.

On Graviton3 (which has both SVE and BF16), BF16 gives 1.61x call_variants speedup.
AmpereOne should see similar gains from NEON BFMMLA if the SVE probe is eliminated.

## Fix: TF Source Build with SVE Disabled in ACL

### Why this works

ACL's BUILD file (`third_party/compute_library/BUILD`) compiles 3 libraries:
- `arm_compute` (base): `-march=armv8-a` + NEON defines
- `arm_compute_sve`: `-march=armv8.2-a+sve`
- `arm_compute_sve2`: `-march=armv8.6-a+sve2`

The base library links both SVE libraries as deps. SVE GEMM candidates (including
the one with `cntb`) are always present in the binary.

Fix: Remove SVE/SVE2 from the BUILD file:
1. Remove `arm_compute_sve` and `arm_compute_sve2` library targets
2. Remove `ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE2` defines
3. Remove SVE deps from `arm_compute` base library
4. Keep `ARM_COMPUTE_ENABLE_BF16` and `ARM_COMPUTE_ENABLE_I8MM` — these enable
   NEON BF16/I8MM kernels compiled with `-march=armv8.6-a+bf16+i8mm`
5. Change base library copts from `-march=armv8-a` to `-march=armv8.6-a+bf16+i8mm`
   (or `-mcpu=ampere1`) so the compiler generates NEON BF16 instructions

### Actual result (2026-03-07)

Build succeeded: 44 minutes, 18,998 Bazel actions, 192 MB wheel. SIGILL
eliminated — BF16 mode runs without crashing on all smoke tests (Conv2D,
InceptionV3).

**However, Risk #1 was confirmed:** ACL v22.11 does NOT have NEON-only BF16
GEMM kernels. All BF16 GEMM in ACL v22.11 is SVE-only. With SVE removed,
BF16 mode falls back to basic NEON FP32 GEMM (no BF16 acceleration).

Additionally, removing SVE also removes the FP32 SVE GEMM kernels that WERE
working on the stock wheel (0.390 s/100). The rebuilt wheel's FP32 rate
degrades to 0.786 s/100 (basic A64 NEON GEMM without SVE).

### Benchmark (rebuilt wheel, AmpereOne 32 vCPU)

| Config | Rate (s/100) | vs Stock FP32 |
|--------|-------------|---------------|
| Stock wheel, FP32 OneDNN | 0.390 | baseline |
| Stock wheel, BF16 OneDNN | SIGILL | — |
| **Rebuilt wheel (no SVE), BF16** | **0.770** | **2.0x slower** |
| **Rebuilt wheel (no SVE), FP32** | **0.786** | **2.0x slower** |
| Stock wheel, Eigen (OneDNN OFF) | 0.390 | same |
| INT8 ONNX (current best) | 0.358 | 1.09x faster |

### Build steps (for reference)

See `build_tf_ampereone.sh`. Key changes to `third_party/compute_library/BUILD`:
- Remove `arm_compute_sve` and `arm_compute_sve2` cc_library targets
- Remove SVE deps from `arm_compute` target
- Remove `ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE2`,
  `ARM_COMPUTE_ENABLE_SVEF32MM` from defines list
- Change base copts to `-march=armv8.6-a+bf16+i8mm`

Build issues: Python 3.10 via deadsnakes PPA (Ubuntu 24.04 PEP 668),
`cache.h` missing `#include <cstdint>` (GCC 13), NumPy 2.x incompatibility
(need `numpy<2.0`), Bazel numpy header cache (`bazel clean --expunge`).

### Conclusion (v22.11)

**BF16 on AmpereOne via ACL v22.11 is not achievable.** The SVE removal
approach eliminates the crash but provides no performance benefit. INT8 ONNX
($2.32/genome) remains the best backend for AmpereOne.

---

## Fix Attempt 2: ACL v23.08 Upgrade (NO BENEFIT)

Upgraded ACL from v22.11 to v23.08 within the TF 2.13.1 build. Kept SVE
enabled (hypothesis: v23.08 fixed the `cntb` probe). Build succeeded after
three fixes (subpackage conflicts, include path ordering, OpenCL link error).

**Result:** FP32 OneDNN works, but BF16 still SIGILLs — same `cntb` crash.
ACL v23.08 did NOT fix the SVE probe bug. See `docs/onednn-ampereone.md`
for full details.

---

## Fix Attempt 3: ACL v23.08 SVE Filter Patch + OneDNN Bypass (2026-03-08)

### Approach

Two targeted patches applied to ACL v23.08 + OneDNN v2.7.3 source within
the TF 2.13.1 build tree:

**Patch 1 — ACL `gemm_fp32.cpp` SVE filter fix:**
`third_party/compute_library/acl_ampereone_bf16_no_sve.patch`

Two SVE BF16 hybrid kernels have incorrect filter conditions — they check
`has_bf16()` but NOT `has_sve()`, so they pass the filter on AmpereOne
(which has BF16 but no SVE). Their estimate functions call `svcntb()` (SVE
instruction), causing SIGILL.

```cpp
// BEFORE (buggy): passes on AmpereOne because has_bf16()=true
"sve_hybrid_fp32bf16fp32_mmla_6x4VL"
  filter: args._fast_mode && args._ci->has_bf16()

// AFTER (fixed): requires both SVE and BF16
  filter: args._fast_mode && args._ci->has_svebf16()
```

Compare with correctly-filtered SVE BF16 kernels that already use
`has_svebf16()` (e.g., `sve_interleaved_bf16fp32_mmla_8x3VL`).

**Patch 2 — OneDNN `acl_convolution_utils.cpp` indirect GEMM bypass:**
`third_party/mkl_dnn/onednn_acl_indirect_gemm_bf16.patch`

After patch 1 fixes the SIGILL, Conv2D with BF16 hits a SIGSEGV (exit 139)
in `CpuGemmAssemblyDispatch::workspace()`. The indirect GEMM path
(`NEGEMMConv2d` / `CpuGemmDirectConv2d`) permutes conv weights before
passing to assembly dispatch. The dispatch's `validate()` rejects the
permuted dimensions for BF16 NEON kernels, `configure()` silently returns
without setting `_arm_gemm`, and the subsequent `workspace()` call
dereferences null. The `ARM_COMPUTE_ERROR_ON` guard is compiled out in
release builds.

Fix: early return `status::unimplemented` from `init_conf_indirect_gemm()`
when fpmath mode is BF16 or ANY. Conv2D falls through to
`acl_gemm_convolution` (im2col + standard GEMM).

### Results: 4 Cascading Bugs Found

| # | Bug | Status | Patch |
|---|-----|--------|-------|
| 1 | SVE filter: `has_bf16()` → `has_svebf16()` for 2 SVE BF16 kernels | **FIXED** | `acl_ampereone_bf16_no_sve.patch` |
| 2 | indirect_gemm null pointer dereference on BF16 weight permutation | **FIXED** | `onednn_acl_indirect_gemm_bf16.patch` |
| 3 | Conv2D falls to `gemm:ref` — ACL `has_opt_impl`/`validate` rejects BF16 for `acl_gemm_convolution` | NOT FIXED — upstream ACL issue |
| 4 | Graph mode + Grappler remapping crash on fused ops (`_FusedConv2D` + BF16 inner_product) | NOT FIXED — TF/OneDNN integration issue |

### Bug Details

**Bug 1 (FIXED — SVE filter):** In ACL v23.08 `gemm_fp32.cpp`, the
`gemm_fp32_methods` dispatch table has two SVE BF16 hybrid kernels with
wrong filter conditions. On AmpereOne: `has_bf16()=true`, `has_sve()=false`,
`has_svebf16()=false`. The buggy filter passes → estimate lambda runs →
`svcntb()` → SIGILL. Five other SVE BF16 kernels in the same file correctly
use `has_svebf16()`.

**Bug 2 (FIXED — null pointer):** After bug 1 fix, the OneDNN convolution
dispatcher tries `acl_indirect_gemm` first (highest priority). This path
uses `CpuGemmDirectConv2d`, which permutes weights with
`PermutationVector{3,0,1,2}` before passing to `CpuGemmAssemblyDispatch`.
The assembly dispatch's `validate()` rejects the permuted dimensions for
BF16 NEON kernels. `configure()` returns without setting `_arm_gemm` (the
internal GEMM handle). The `ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr)`
assertion is compiled out in release builds (`-DNDEBUG`). The subsequent
`workspace()` call dereferences `_arm_gemm` → SIGSEGV.

**Bug 3 (UNFIXED — gemm:ref fallback):** After bypassing indirect_gemm,
Conv2D should use `acl_gemm_convolution` (im2col + standard GEMM). OneDNN
calls `NEGEMMConvolutionLayer::has_opt_impl()` which traces through to
`CpuGemmAssemblyDispatch::has_opt_impl()` → `has_opt_gemm<float,float>()`.
This should find the NEON BF16 fixed-format kernels
(`a64_ffinterleaved_bf16fp32_mmla_8x12`, `a64_ffhybrid_fp32bf16fp32_mmla_4x24`),
which ARE compiled in (symbols verified via `nm libtensorflow_cc.so.2`) and
have correct filters (`fast_mode && has_bf16()`). Yet Conv2D falls all the
way to `gemm:ref` (reference implementation), which ignores the
`attr-fpmath:bf16` hint and runs plain FP32 math. **Performance: identical
to FP32** (5.3ms vs 5.4ms per Conv2D — zero BF16 benefit).

**Bug 4 (UNFIXED — Grappler crash):** In graph mode (`model.predict`),
TF's Grappler **remapping** optimization fuses ops (e.g.,
Conv2D+BiasAdd+Relu → `_FusedConv2D`). These fused ops go through a
different OneDNN code path. The final Dense(3) layer's inner_product
execution with BF16-formatted weights (`AB4a4b` format, reordered from
FP32) crashes with SIGSEGV. Reproduces consistently with InceptionV3 but
NOT with: eager mode (`model(x)`), graph mode with remapping disabled,
standalone Dense(3) in isolation, or small models.

### Smoke Test Results

| Test | Result |
|------|--------|
| FP32 OneDNN Conv2D (32x32, ic=3, oc=16) | **PASS** — `indirect_gemm:acl` |
| BF16 OneDNN Conv2D (32x32, ic=3, oc=16) | **PASS** — `gemm:ref` (no BF16 benefit) |
| BF16 OneDNN Conv2D (149x149, ic=32, oc=64) | **PASS** — `gemm:ref` |
| BF16 Conv2D x 100 iterations | **PASS** |
| BF16 MatMul (64x256 x 256x128, eager) | **PASS** — `gemm:jit` with BF16 |
| BF16 Dense(3) (standalone, eager) | **PASS** |
| BF16 InceptionV3 eager `model(x)` | **PASS** |
| BF16 InceptionV3 `model.predict` (graph mode) | **SIGSEGV** — bug 4 |

### Benchmark (Conv2D microbenchmark, AmpereOne 32 vCPU)

| Config | Time (ms) | OneDNN impl | BF16 benefit? |
|--------|-----------|-------------|---------------|
| FP32 OneDNN (stock) | 5.4 | `indirect_gemm:acl` | — |
| BF16 OneDNN (patched) | 5.3 | `gemm:ref` | **No** — ref ignores BF16 |
| INT8 ONNX | ~3.6 | N/A (ORT) | N/A |

---

## Final Conclusion

**BF16 on AmpereOne via TF 2.13.1 + ACL v23.08 + OneDNN v2.7.3 is not
viable.** The two committed patches fix genuine bugs (SVE filter and null
pointer dereference), but two more unfixed bugs remain:

1. ACL's NEON BF16 fixed-format GEMM kernels exist and are compiled, but
   the convolution integration path (`NEGEMMConvolutionLayer::has_opt_impl`)
   does not select them — Conv2D falls to the unaccelerated `gemm:ref`
   reference implementation. Even if the routing were fixed, `gemm:ref`
   provides zero BF16 speedup (5.3ms vs 5.4ms FP32).

2. TF's Grappler remapping + OneDNN BF16 inner_product crashes in graph
   mode on large models (InceptionV3). This is deep in the TF/OneDNN
   integration layer and not patchable without upstream changes.

**Production recommendation: INT8 ONNX ($2.32/genome on Oracle A2).**

### Next Step: Upstream Issue

The two committed patches + this bug chain documentation provide sufficient
evidence to file an issue with ARM-software/ComputeLibrary:
- `third_party/compute_library/acl_ampereone_bf16_no_sve.patch` (bug 1 fix)
- `third_party/mkl_dnn/onednn_acl_indirect_gemm_bf16.patch` (bug 2 fix)
- Bug 3 requires upstream ACL changes to route Conv2D to NEON BF16 kernels
- Bug 4 requires upstream TF/OneDNN changes for Grappler + BF16 interaction
