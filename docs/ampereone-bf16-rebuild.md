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

### Conclusion

**BF16 on AmpereOne via ACL v22.11 is not achievable.** The SVE removal
approach eliminates the crash but provides no performance benefit. INT8 ONNX
($2.32/genome) remains the best backend for AmpereOne.

Future options:
1. **Keep INT8 ONNX** — pragmatic, no changes needed
2. **Upgrade ACL to v23.x+** — may have NEON BF16 GEMM (high effort)
3. **Patch `cntb` probe** — guard with `__builtin_cpu_supports("sve")`
   to keep SVE FP32 GEMM while blocking SVE BF16 GEMM (medium effort, fragile)
