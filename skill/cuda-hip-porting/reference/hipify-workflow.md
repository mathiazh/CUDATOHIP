# Hipify Workflow

`hipify` is AMD's automated CUDA→HIP source translator. Two implementations ship with ROCm:

- **`hipify-perl`** — regex-based, fast, no compile required, works on any text. Best for one-shot files and scripts. Translates ~70-90% of typical CUDA mechanically.
- **`hipify-clang`** — clang-AST-based, needs the file to actually parse with all its includes. Catches things `hipify-perl` misses (template specializations, macro expansions). Required for projects with complex headers.

Both are in `/opt/rocm/bin/` inside `rocm/rocm-terminal`.

## Default invocations

### hipify-perl (preferred for single files)
```bash
# In the rocm container
hipify-perl foo.cu > foo.hip 2> hipify_stderr.txt
hipify-perl --print-stats foo.cu > foo.hip 2> hipify_stats.txt
```

`--print-stats` writes a summary of how many tokens were converted, unconverted, etc. The `01_vectorAdd/hipify_stats.txt` file is an example — keep these alongside the source so you can audit the translation later.

### hipify-clang (preferred for projects)
```bash
hipify-clang foo.cu -o foo.hip -- -I./include -I/usr/local/cuda/include -std=c++17
```

Notes:
- Everything after `--` is forwarded to clang as compile args. You almost always need `-I/usr/local/cuda/include` so it can find `cuda_runtime.h`.
- `hipify-clang` will refuse to translate a file that doesn't fully parse. If you get errors, fix the include path first; the translation comes out clean afterwards.
- For multi-file projects, run it per-file in a loop or use the `hipify-clang -p <build-dir>` mode if you have a `compile_commands.json`.

## What hipify silently translates well

- All `cuda*` runtime API calls → `hip*` (`cudaMalloc` → `hipMalloc`, etc.)
- Most `cuda*_t` types → `hip*_t`
- Headers: `<cuda_runtime.h>` → `<hip/hip_runtime.h>`, `<cuda_fp16.h>` → `<hip/hip_fp16.h>`
- Error constants: `cudaSuccess` → `hipSuccess`
- Memory copy directions: `cudaMemcpyHostToDevice` → `hipMemcpyHostToDevice`
- `__shfl_*_sync` warp intrinsics
- cuBLAS handles and most basic cuBLAS calls

## What hipify gets WRONG (silently, no warning)

These compile but are broken or surprising — watch for them:

1. **`cudaCheck`-style user macros.** Hipify substitutes the *body* of the macro (so `cudaCheck(cudaMalloc(...))` becomes `cudaCheck(hipMalloc(...))`) but leaves the macro *name* as `cudaCheck`. The code still works because the macro now wraps a HIP call, but `grep cuda` will still match. Look for warnings like:
   ```
   warning: llmc/cuda_common.h:#52 : inline void cudaCheck_(hipError_t error, ...
   ```
   The function takes `hipError_t` but is named `cudaCheck_`. Rename for cleanliness; it's correct as-is.

2. **`cudaError_t` inside templates.** Sometimes missed; grep the output for any remaining `cuda` tokens.

3. **Constant memory layout.** `__constant__ float foo[N]` translates fine, but if you do `cudaMemcpyToSymbol(foo, ...)`, double-check — the HIP equivalent has slightly different argument semantics in older ROCm versions.

4. **Stream creation flags.** `cudaStreamNonBlocking` → `hipStreamNonBlocking` — works. But `cudaStreamPerThread` semantics differ; stream-per-thread is not a first-class HIP concept.

## What hipify CANNOT translate (will warn or pass through)

These are the entries you'll see in `hipify_stderr.txt`:

| Warning pattern | What it means | Action |
|---|---|---|
| `unsupported HIP identifier: CUBLASLT_MATMUL_DESC_*` | cuBLASLt enum/handle | Rewrite to hipBLASLt; see `api-rewrites.md`. |
| `unsupported device function "__stcs"` (also `__stcg`, `__ldcs`) | Cache-control intrinsic | Replace with plain `*ptr = val` / `*ptr`. Not in HIP. |
| `unsupported device function "__nanosleep"` | GPU-side sleep | Remove. AMD has no equivalent. |
| `unsupported device function "__float2bfloat16_rn"` | NVIDIA-specific bf16 conversion | Use `__float2bfloat16` (round-to-nearest is default in HIP) or bit-cast. |
| `nvcuda::wmma::*` (no warning, just left untouched) | Tensor-core API | Full rewrite to `rocwmma::*`. |
| `asm("...")` blocks | Inline PTX | Manual rewrite — usually portable C, occasionally MFMA. |
| `__nv_bfloat16` types passed through | bf16 type | `__hip_bfloat16` is the equivalent; sometimes hipify maps, sometimes not. |

## Always save the warnings

```bash
hipify-perl --print-stats foo.cu > foo.hip 2> hipify_stderr.txt
# or
hipify-clang foo.cu -o foo.hip -- -I... 2> hipify_stderr.txt
```

The stderr file is the punch list for manual fixups. Without it, you have no way to know what needs attention. Every project in `/scratch/rozum/cuda/` keeps these files.

## Reading a stats file

Sample from `01_vectorAdd/hipify_stats.txt` would show:
```
[HIPIFY] info: file 'vectorAdd.cu' statistics:
  CONVERTED refs count: 23
  UNCONVERTED refs count: 0
  CONVERSION %: 100.0
```

100% means a clean mechanical translation — but doesn't guarantee a clean compile. Always try to compile.

## When hipify is wrong, check both versions

Compare the original `.cu` with the produced `.hip` for sanity:
```bash
diff -u foo.cu foo.hip | less
```

This will catch dropped lines, mis-translated identifiers, and anywhere hipify "lost" a reference.
