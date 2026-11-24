# Troubleshooting Common Build Errors

Symptoms and fixes for compile/link errors that come up during CUDA→HIP ports.

## 1. `fatal error: 'hip/hip_runtime.h' file not found`

You ran `hipcc` outside the ROCm container, or `/opt/rocm/include` isn't in your include path.

```bash
# Verify
which hipcc                    # should be /opt/rocm/bin/hipcc
hipconfig --cpp_config         # should print -I/opt/rocm/include ...
```

If you're inside the container and still see this, the image is broken — try `docker pull rocm/rocm-terminal:latest` to refresh.

## 2. Hipify left `cudaCheck(...)` macro names unchanged

This is **expected and correct**. Hipify substitutes the macro body but not the user-defined macro identifier. The grep result is misleading:

```
$ grep cudaCheck foo.hip
foo.hip:42: cudaCheck(hipMalloc(&p, sz));     // body translated, name kept
```

The `cudaCheck` macro now wraps `hipError_t`, not `cudaError_t`. Either:
- Leave it (functionally fine, just cosmetically inconsistent), or
- Rename the macro globally: `sed -i 's/cudaCheck/hipCheck/g' foo.hip`

## 3. `error: use of undeclared identifier '__stcs'` (or `__stcg`, `__ldcs`)

Hipify warned about these but couldn't fix them. See `api-rewrites.md` §3 — replace with plain `*ptr = val`.

## 4. `error: no member named 'CUBLASLT_MATMUL_DESC_SCALE_TYPE'`

cuBLASLt enum was passed through but doesn't exist in hipBLASLt. Map manually:

| cuBLASLt | hipBLASLt |
|---|---|
| `CUBLASLT_MATMUL_DESC_SCALE_TYPE` | `HIPBLASLT_MATMUL_DESC_COMPUTE_TYPE` |
| `CUBLASLT_EPILOGUE_BIAS` | `HIPBLASLT_EPILOGUE_BIAS` |
| `CUBLASLT_EPILOGUE_RELU_BIAS` | `HIPBLASLT_EPILOGUE_RELU_BIAS` |

When in doubt: `grep -r CUBLASLT /opt/rocm/include/hipblaslt/` — the renamed equivalents are usually one-to-one with the prefix swapped.

## 5. `error: 'nvcuda::wmma::*' undeclared` after hipify

Hipify did not translate WMMA. Do the manual rewrite — see `api-rewrites.md` §1. Symptom is hundreds of errors all referring to `nvcuda::wmma::fragment` or `wmma::load_matrix_sync`.

## 6. `linker: undefined reference to 'hipblasLtMatmul'`

You forgot `-lhipblaslt`. Full library map:

| You used | Add link flag |
|---|---|
| `hipBLAS` | `-lhipblas` |
| `hipBLASLt` | `-lhipblaslt` |
| `hipFFT` | `-lhipfft` |
| `MIOpen` (cuDNN equiv) | `-lmiopen` |
| `rocBLAS` | `-lrocblas` |
| `hipRAND` | `-lhiprand` |
| `rocWMMA` | nothing — header-only |
| `rocPRIM` / `hipCUB` | nothing — header-only |

## 7. `hipify-clang` errors with "fatal error: ..." on the first include

`hipify-clang` runs the file through clang, which needs every `#include` to resolve. Add the missing include path after `--`:

```bash
hipify-clang foo.cu -o foo.hip -- \
    -I./include \
    -I/usr/local/cuda/include \
    -I/path/to/third_party \
    -std=c++17
```

If a header is *truly* unavailable (e.g. it's NVIDIA-only like `<cudnn.h>`), use `hipify-perl` instead — it doesn't need to parse the file.

## 8. Kernel launches succeed but produce garbage on AMD GPU

Almost always a wavefront-size assumption:

```cpp
// CUDA: assumes warp = 32
int warpId = threadIdx.x / 32;
int laneId = threadIdx.x % 32;

// HIP-portable
int warpId = threadIdx.x / warpSize;   // warpSize is built-in, 32 on NV/RDNA, 64 on CDNA
int laneId = threadIdx.x % warpSize;
```

Other suspects:
- Hardcoded `__shfl_*` masks (`0xffffffff` is wrong for 64-thread waves; use `__activemask()` or `~0ULL`).
- Block size: `dim3(128)` = 4 warps on NV, only 2 wavefronts on CDNA.
- Shared memory size — HIP and CUDA expose the same per-block static shared memory (typically 48-64 KB), but if you used `cudaFuncSetAttribute` to bump it past 48 KB, the equivalent on HIP is `hipFuncSetAttribute(MaxDynamicSharedMemorySize, ...)`.

## 9. `clang: error: unsupported option '--use_fast_math'` (in hipcc)

Hipify-perl sometimes carries this through. The hipcc equivalent is `-ffast-math` (no `--use_`). Just edit the build script.

## 10. `error: '_Float16' was not declared in this scope` (in CPU code)

`_Float16` is a clang extension; gcc-built host code can't see it. Options:
- Compile the offending file with `hipcc` (which uses clang) instead of `gcc`.
- Use `__fp16` inside `#ifdef __HIPCC__` blocks.
- Wrap the `_Float16` storage in a `uint16_t` and only convert inside device code.

## 11. Image runs locally but not in CI: "no kernel image is available for execution on the device"

Architecture mismatch. The binary was built for an arch the runtime GPU doesn't support.
- CUDA: rebuild with `-arch=sm_<your-gpu-cc>` or use a multi-arch fat binary.
- HIP: rebuild with `--offload-arch=gfx<your-gpu-isa>`. Find ISA with `rocminfo | grep gfx`.

## 12. `hipDeviceSynchronize()` returns `hipErrorNoDevice` when host has no GPU

Expected when running compile-test in a non-GPU container. Your code should handle this gracefully — both `hello_cuda.cu` and `hello_hip.cpp` in this project show the pattern:

```cpp
hipError_t err = hipDeviceSynchronize();
if (err != hipSuccess) {
    printf("GPU kernel launch failed (expected without GPU): %s\n", hipGetErrorString(err));
}
```

## 13. ROCm container: "Permission denied" writing build outputs

The `rocm/rocm-terminal` image runs as `rocm-user` (UID 1000). If your host user has a different UID, the build artifacts will be owned by 1000 on the host. Either:
- `chown -R $(id -u):$(id -g) .` after the build, or
- Build into a `/tmp/build` subdirectory inside the container, then `cp` outputs out, or
- Pass `--user $(id -u):$(id -g)` (works for compile-only, may break runtime tools).

## 14. Hipify output has stray Windows-style `\r\n` line endings

`hipify-perl` on Windows-formatted source files preserves CRLF. If gcc/clang complains:
```bash
dos2unix foo.hip
```
Or filter the hipify output: `hipify-perl foo.cu | tr -d '\r' > foo.hip`.

## 15. Double-translation when re-running hipify on already-hipified code

Don't run hipify on `.hip` files. Always re-run from the original `.cu` or you'll get gibberish like `hiphip*` token concatenations. Keep the original `.cu` (some projects rename to `_original.cu`) so you can always restart from a known baseline.
