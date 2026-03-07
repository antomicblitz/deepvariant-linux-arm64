# OneDNN BF16 SIGILL on AmpereOne — GDB Root Cause Analysis

## Summary

AmpereOne (CPU part `0xac3`, implementer `0xc0`) crashes with SIGILL when
TF+OneDNN uses BF16 fpmath mode. Root cause: ACL v22.11's GEMM method
selection executes `cntb x0` (SVE instruction) unconditionally during BF16
candidate probing. AmpereOne has **no SVE** support.

Fix: Rebuild TF 2.13.1 from source with SVE/SVE2 removed from ACL BUILD file.

## GDB Analysis (2026-03-07)

### Faulting instruction

```
=> 0xffffe48c5a9c: cntb x0
```

`cntb` = "Count Bytes in SVE vector" — an SVE instruction.

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

### Why BF16-specific

FP32 OneDNN works fine on AmpereOne:

| Test | OneDNN | FPMATH | Result |
|------|--------|--------|--------|
| Small model 16-way x 64K preds | ON | FP32 | **PASS** |
| InceptionV3 single batch | ON | FP32 | **PASS** |
| InceptionV3 single batch | ON | **BF16** | **SIGILL** (`cntb x0`) |
| InceptionV3 single batch | OFF | N/A | **PASS** (Eigen) |

BF16 fpmath mode causes OneDNN to request BF16-capable GEMM kernels from ACL.
ACL's `find_implementation()` iterates ALL candidates including SVE-BF16 ones.
Lambda #16 contains `cntb x0` to check SVE vector length — this executes
**before** the candidate is rejected by feature detection.

### FP32 OneDNN vs Eigen benchmark

| Backend | CV time | Rate |
|---------|---------|------|
| Eigen (TF_ENABLE_ONEDNN_OPTS=0) | 307s | ~0.39 s/100 |
| FP32 OneDNN (TF_ENABLE_ONEDNN_OPTS=1) | 307s | ~0.39 s/100 |

FP32 OneDNN = Eigen on AmpereOne. No speedup without BF16.

### AmpereOne CPU flags (corrected)

Previous docs incorrectly listed `sve` in the flags. Actual flags from
`/proc/cpuinfo` on Oracle A2 (VM.Standard.A2.Flex):

```
fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid
asimdrdm jscvt fcma lrcpc dcpop sha3 asimddp sha512 asimdfhm dit uscat
ilrcpc flagm ssbs sb paca pacg dcpodp flagm2 frint i8mm bf16 rng bti ecv
```

**NO `sve`. NO `sve2`.** Has `i8mm` and `bf16` (NEON extensions, not SVE).

## Fix: Remove SVE from ACL BUILD

### ACL BUILD file structure

`third_party/compute_library/BUILD` compiles 3 libraries:
- `arm_compute` (base): `-march=armv8-a` + NEON defines
- `arm_compute_sve`: `-march=armv8.2-a+sve`
- `arm_compute_sve2`: `-march=armv8.6-a+sve2`

The base library links both SVE libraries as deps. SVE GEMM candidates
(including the one with `cntb`) are always present in the binary.

### Changes applied

1. **Remove** `arm_compute_sve` and `arm_compute_sve2` cc_library targets
2. **Remove** `ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE`, `ARM_COMPUTE_ENABLE_SVE2`,
   `ARM_COMPUTE_ENABLE_SVEF32MM` from defines
3. **Remove** SVE deps from `arm_compute` base library
4. **Change** base copts: `-march=armv8-a` -> `-march=armv8.6-a+bf16+i8mm`
5. **Keep** `ARM_COMPUTE_ENABLE_BF16` and `ARM_COMPUTE_ENABLE_I8MM`

### Result: SIGILL eliminated, but NO BF16 speedup

The rebuilt wheel (SVE removed) eliminates the SIGILL — BF16 mode runs
without crashing. However, **Risk #1 was confirmed**: ACL v22.11 does NOT
have NEON-only BF16 GEMM kernels. All BF16 GEMM in ACL v22.11 is SVE-only.

### Benchmark (2026-03-07, rebuilt wheel, AmpereOne 32 vCPU)

| Config | Rate (s/100) | vs Stock FP32 |
|--------|-------------|---------------|
| Stock wheel, FP32 OneDNN | 0.390 | baseline |
| Stock wheel, BF16 OneDNN | SIGILL | — |
| **Rebuilt wheel (no SVE), BF16** | **0.770** | **2.0x slower** |
| **Rebuilt wheel (no SVE), FP32** | **0.786** | **2.0x slower** |
| Stock wheel, Eigen (OneDNN OFF) | 0.390 | same as FP32 OneDNN |
| INT8 ONNX (current best) | 0.358 | 1.09x faster |

**Key findings:**
1. Removing SVE eliminates the SIGILL (BF16 mode works)
2. But removing SVE also removes the FP32 SVE GEMM kernels that WERE working
3. ACL v22.11's BF16 GEMM is SVE-only — no NEON BF16 GEMM fallback
4. The rebuilt wheel's FP32 rate (0.786) = basic A64 NEON GEMM (no SVE)
5. Stock wheel FP32 (0.390) uses SVE FP32 GEMM which happens to work on
   AmpereOne (the `cntb` probe only executes for BF16 candidates)

**Why removing SVE hurts FP32 too:** The stock wheel has SVE FP32 GEMM
kernels compiled with `-march=armv8.2-a+sve`. Despite AmpereOne not having
SVE in `/proc/cpuinfo`, these FP32 SVE kernels DO execute successfully at
0.390 s/100. This suggests either: (a) AmpereOne has partial SVE support
not reflected in cpuinfo, or (b) the FP32 SVE GEMM code path doesn't
actually use SVE instructions (just compiled with SVE flags but using only
NEON). Either way, removing SVE removes these kernels, degrading FP32.

### Conclusion

**BF16 on AmpereOne via ACL v22.11 is not achievable.** The SVE removal
approach eliminates the crash but provides no performance benefit. The
practical options are:

1. **Keep INT8 ONNX ($2.32/genome)** — current best, no changes needed
2. **Upgrade ACL to v23.x+** — may have NEON BF16 GEMM, but requires
   upgrading OneDNN+TF dependency chain (high effort)
3. **Patch the `cntb` probe** — surgically guard the SVE vector length
   check with `__builtin_cpu_supports("sve")` to keep SVE FP32 GEMM
   while blocking SVE BF16 GEMM (medium effort, fragile)

Option 1 is the pragmatic choice. The $2.32/genome cost is already the
cheapest tested platform.

## Build details

Build script: `build_tf_ampereone.sh`
Instance: Oracle A2 VM.Standard.A2.Flex (16 OCPU / 32 vCPU, 64 GB RAM)
Disk: Expanded from 46 GB to 200 GB for TF source build
Python: 3.10.20 (via deadsnakes PPA, in venv)
Bazel: 5.3.0 (aarch64)
Build time: ~44 minutes (18,998 Bazel actions)

Build issues encountered:
- Ubuntu 24.04 PEP 668: need Python 3.10 venv via deadsnakes PPA
- `cache.h` missing `#include <cstdint>`: GCC 13 compatibility
- NumPy 2.x API incompatibility: need `numpy<2.0` (1.26.4)
- Bazel numpy header cache: `bazel clean --expunge` + reconfigure required

Bazel flags:
```
--config=opt
--config=mkl_aarch64_threadpool
--copt=-march=armv8.6-a+bf16+i8mm
--jobs=28
--local_ram_resources=55000
--cxxopt=-include --cxxopt=cstdint
--host_cxxopt=-include --host_cxxopt=cstdint
```

## References

- [docs/onednn-ampereone.md](onednn-ampereone.md) — Diagnostic trail and production config
- [docs/ampereone-bf16-rebuild.md](ampereone-bf16-rebuild.md) — Build plan
- [docs/oracle-a2-sigill.md](oracle-a2-sigill.md) — Original investigation
