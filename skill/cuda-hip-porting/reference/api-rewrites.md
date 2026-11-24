# Manual API Rewrites

Things hipify can't translate, with the patterns we've used in this project.

## 1. WMMA → rocWMMA (tensor cores)

NVIDIA `nvcuda::wmma` is left **completely untouched** by hipify. You need a full manual rewrite to `rocwmma`. See `05_wmmaGemm/wmmaGemm.hip` and `wmmaGemm_rewrite.hip.txt` for a complete worked example (16 manual changes).

### Header & namespace
```cpp
// CUDA
#include <mma.h>
using namespace nvcuda;

// HIP
#include <rocwmma/rocwmma.hpp>
// no using-namespace needed; reference rocwmma:: directly
```

### Fragment declarations
```cpp
// CUDA
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

// HIP — note: half → _Float16
rocwmma::fragment<rocwmma::matrix_a, 16, 16, 16, _Float16, rocwmma::row_major> a_frag;
rocwmma::fragment<rocwmma::matrix_b, 16, 16, 16, _Float16, rocwmma::row_major> b_frag;
rocwmma::fragment<rocwmma::accumulator, 16, 16, 16, float> c_frag;
```

### Operations
```cpp
// CUDA → HIP (drop the wmma::, prefix with rocwmma::)
wmma::fill_fragment(c, 0.0f);            → rocwmma::fill_fragment(c_frag, 0.0f);
wmma::load_matrix_sync(a, ptr, ldm);     → rocwmma::load_matrix_sync(a_frag, ptr, ldm);
wmma::store_matrix_sync(ptr, c, ldm,
                        wmma::mem_row_major);
                                         → rocwmma::store_matrix_sync(ptr, c_frag, ldm,
                                                                       rocwmma::mem_row_major);
wmma::mma_sync(c, a, b, c);              → rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
```

### Block size differs!
- CUDA WMMA: warp = 32 threads, so a block of `dim3(128,1)` = 4 warps.
- AMD MFMA (rocWMMA): wavefront = 64 threads on CDNA, so the same logical mapping needs `dim3(256,1)` = 4 wavefronts.

Check `warpSize` (provided as a built-in by both compilers — 32 on NVIDIA, 64 on AMD CDNA, 32 on RDNA). Use it when computing thread→tile mappings instead of hardcoding 32.

## 2. cuBLASLt → hipBLASLt

Hipify warns: `unsupported HIP identifier: CUBLASLT_MATMUL_DESC_*`

| CUDA | HIP |
|---|---|
| `cublasLtHandle_t` | `hipblasLtHandle_t` |
| `cublasLtCreate` | `hipblasLtCreate` |
| `cublasLtMatmulDesc_t` | `hipblasLtMatmulDesc_t` |
| `CUBLASLT_MATMUL_DESC_TRANSA` | `HIPBLASLT_MATMUL_DESC_TRANSA` |
| `CUBLASLT_MATMUL_DESC_SCALE_TYPE` | `HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE` (semantics differ slightly) |
| `cublasLtMatmul` | `hipblasLtMatmul` |
| `-lcublasLt` | `-lhipblaslt` |

The API surface mostly matches but enum values may be renamed — check `<hipblaslt/hipblaslt.h>`. For complex matmul descriptor setup (scale modes, epilogues, bias), test each case individually.

## 3. Cache-control intrinsics: `__stcs`, `__stcg`, `__ldcs`, `__ldcg`

These are NVIDIA-specific cache-bypass hints. **HIP has no equivalent.** Replace with plain memory access:

```cpp
// CUDA
__stcs(out + idx, val);          → *(out + idx) = val;
__ldcs(in + idx);                → *(in + idx);
__stcg(reinterpret_cast<int4*>(p), v);
                                 → *reinterpret_cast<int4*>(p) = v;
```

Functionally equivalent, but the AMD GPU will use its default cache policy. Performance impact is usually <5%.

If the project has a header that defines a polyfill (like `llmc/cuda_common.h` does), patch the polyfill once and let it propagate.

## 4. PTX inline assembly

Anything in `asm("...")` is NVIDIA-only and needs a manual replacement.

### Example: `dp4a` (4×8-bit dot product)
```cpp
// CUDA — uses NVIDIA's dp4a PTX instruction
int result;
asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));

// HIP — manual implementation, slower but correct
signed char *ab = (signed char*)&a;
signed char *bb = (signed char*)&b;
int result = c;
for (int i = 0; i < 4; i++) result += (int)ab[i] * (int)bb[i];
```

### Other common PTX → HIP rewrites
| PTX | HIP equivalent |
|---|---|
| `prmt.b32` (byte permute) | Bit shifts and masks, or `__builtin_amdgcn_perm` |
| `bfind`, `bfe` (bit-field extract) | `__builtin_clz`, manual shifts |
| `vote.ballot` | `__ballot(pred)` (HIP intrinsic, returns 64-bit on CDNA) |
| `shfl.sync.*` | `__shfl_*` family — hipify does this automatically |
| `mma.sync.*` (tensor core PTX) | rocWMMA or `__builtin_amdgcn_mfma_*` |
| `nanosleep.u32` (`__nanosleep`) | **No equivalent.** Remove or replace with a busy-wait if absolutely needed. |
| `cp.async.*` (async copy) | `__builtin_amdgcn_global_load_lds` on gfx90a+ |

## 5. Texture / surface objects

Most texture object code translates with hipify (`cudaTextureObject_t` → `hipTextureObject_t`, `cudaCreateTextureObject` → `hipCreateTextureObject`). What doesn't:

- **`cudaResourceTypeArray` with non-trivial channel formats.** `hipChannelFormatDesc` exists but some channel descriptors (e.g. cudaChannelFormatKindNV12) have no HIP equivalent.
- **Surface stores with format conversion** (`surf2Dwrite<float>`) sometimes need explicit casting on HIP.
- **Texture references** (the deprecated pre-CUDA-5 API) — port to texture objects first, then hipify.

See `06_texturePTX/texturePTX.hip` for a working example.

## 6. bfloat16 conversions

```cpp
// CUDA
__nv_bfloat16 b = __float2bfloat16_rn(f);    // _rn = round-to-nearest

// HIP
__hip_bfloat16 b = __float2bfloat16(f);      // round-to-nearest is default
```

If the explicit `_rn` matters (it usually doesn't because RN is the default IEEE rounding), use a bit-twiddle. Hipify warns:
```
warning: unsupported device function "__float2bfloat16_rn"
```

## 7. Managed memory fallback

The pattern from `07_llmc/cuda_utils.cuh`:

```cpp
// Try managed memory; fall back gracefully if device doesn't support it
int cudaMallocConditionallyManaged(void** out, size_t bytes, ...) {
    err = hipMallocManaged(out, bytes);
    if (err == hipSuccess) {
        hipMemAdvise(*out, bytes, hipMemAdviseSetPreferredLocation, hipCpuDeviceId);
    } else {
        err = hipMalloc(out, bytes);  // fallback
    }
    return err;
}
```

Works on both CUDA and HIP after hipify — HIP supports `hipMallocManaged` on most modern GPUs but coverage is thinner than CUDA's. Always have a `hipMalloc` fallback.

## 8. Stream priorities

`cudaStreamCreateWithPriority` → `hipStreamCreateWithPriority` translates fine, but priority *ranges* differ. Always query:
```cpp
int low, high;
hipDeviceGetStreamPriorityRange(&low, &high);
```
On NVIDIA: typically `[-1, 0]`. On AMD: typically `[0, ...]`. Don't hardcode priority constants.

## 9. Cooperative groups

`cooperative_groups::*` is supported in HIP via `<hip/hip_cooperative_groups.h>` and the namespace is the same. Hipify handles the include rewrite. Caveat: `cooperative_groups::wait()` and grid-wide sync require `--offload-arch=gfx90a` or newer plus `-fgpu-rdc`.

## 10. cuRAND → hipRAND

Mostly mechanical rename; hipify does it. The few gotchas:
- `curandStatePhilox4_32_10_t` → `hiprandStatePhilox4_32_10_t`
- Generated values *should* be bit-identical for same seed/state, but verify if randomness affects test pass/fail.
