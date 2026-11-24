---
name: cuda-hip-porting
description: Port CUDA code to AMD HIP/ROCm. Covers Docker container setup (nvidia/cuda + rocm/rocm-terminal), building with nvcc/hipcc, automated translation with hipify-perl/hipify-clang, interpreting hipify warnings, and manually rewriting unsupported APIs (WMMA, PTX intrinsics, cuBLASLt, __ldcs/__stcs, texture objects, etc.). Triggers on tasks involving .cu/.hip/.cpp HIP code, hipify, ROCm, rocm-terminal, hipcc, nvcc, or CUDA-to-ROCm migration.
---

# CUDA ↔ HIP Porting Skill

Port CUDA codebases to AMD HIP/ROCm. The workflow is always the same:

1. **Establish a working CUDA build** in an `nvidia/cuda` container so you have a known-good baseline.
2. **Run hipify** (perl for one-shot, clang for projects with headers/macros) to do the mechanical 90%.
3. **Read the warnings** — they are the punch list. Every `warning:` is something hipify could not translate.
4. **Compile in a `rocm/rocm-terminal` container** with `hipcc`. Fix errors one API at a time.
5. **Rewrite the unsupported pieces by hand** — WMMA, PTX inline asm, cuBLASLt, `__stcs`/`__ldcs`, etc. Most have HIP equivalents; a few need manual implementations or different libraries (rocWMMA, hipBLASLt, MFMA).

The reference files in `reference/` hold the deep details. Read them when you hit the relevant problem — don't preload them.

## Reference files

- `reference/docker-setup.md` — Pulling, running, and using the CUDA and ROCm containers; volume mounts; GPU passthrough; non-GPU compile-only flows.
- `reference/build-commands.md` — `nvcc`/`hipcc` invocations, arch flags (`sm_XX` ↔ `gfx9XX`), linking cuBLAS/hipBLAS, common Makefile patterns.
- `reference/hipify-workflow.md` — When to use `hipify-perl` vs `hipify-clang`, how to capture stats/warnings, interpreting per-line warnings, and what hipify silently changes.
- `reference/api-rewrites.md` — Manual rewrites for things hipify can't handle: WMMA→rocWMMA, cuBLASLt→hipBLASLt, `__stcs`/`__ldcs`/`__stcg`, `__nanosleep`, PTX `dp4a`, texture objects, `__float2bfloat16_rn`.
- `reference/troubleshooting.md` — Common compile/link errors and fixes (header-only libs, missing `hip::device`, gencode mismatches, double-translation of `cudaCheck` macros, etc.).

## Quick start

If the user says "port this CUDA program to HIP":

```bash
# 1) Get CUDA building (in CUDA container)
docker run --rm -v /scratch/rozum/cuda:/work -w /work nvidia/cuda:13.2.1-devel-ubuntu22.04 \
    nvcc -O2 -o foo_cuda foo.cu

# 2) Translate (hipify-perl is fine for single files; use hipify-clang for projects)
hipify-perl foo.cu > foo.hip 2> hipify_stderr.txt

# 3) Build HIP version (in ROCm container)
docker run --rm -v /scratch/rozum/cuda:/work -w /work rocm/rocm-terminal:latest \
    bash -c "hipcc -O2 -o foo_hip foo.hip"
```

Keep both binaries (`foo_cuda` and `foo_hip`) so you can diff outputs later.

## Project layout convention

The existing work in `/scratch/rozum/cuda/` follows this pattern — match it for new ports:

```
NN_projectName/
├── projectName.cu              # original CUDA (or _original.cu if renamed)
├── projectName_cuda            # nvcc-built binary
├── projectName.hip             # hipified + manually fixed
├── projectName_hip             # hipcc-built binary
├── hipify_stats.txt            # stdout from hipify (counts)
└── hipify_stderr.txt           # stderr from hipify (warnings — the real punch list)
```

For non-trivial rewrites, save a `projectName_rewrite.hip.txt` describing what was rewritten by hand and why (see `05_wmmaGemm/wmmaGemm_rewrite.hip.txt` for the pattern).

## Key principles (learned the hard way)

- **Always keep the warnings file.** `2> hipify_stderr.txt` — hipify's warnings *are* the gap analysis. Don't discard them.
- **Don't trust a clean hipify run.** A warning-free output can still fail to compile. Hipify silently translates many things that don't have direct equivalents (e.g. `cudaCheck` macros get token-substituted to call `hip*` functions but the macro name stays `cudaCheck`).
- **Compile-only is fine without a GPU.** Both containers compile without `--gpus all`. You only need GPU passthrough for `cudaDeviceSynchronize`/`hipDeviceSynchronize` to actually run kernels. Most porting work is compile-iterate-fix.
- **PTX inline asm has no hipify path.** Anything in `asm("...")` blocks needs manual replacement — usually a portable C implementation or AMD MFMA equivalent. See `reference/api-rewrites.md`.
- **WMMA is a complete rewrite, not a translation.** NVIDIA `nvcuda::wmma::*` → AMD `rocwmma::*`. Hipify translates **zero** WMMA code. Plan for this upfront if the project uses tensor cores.
- **`__stcs`/`__stcg`/`__ldcs` (cache hints) are not in HIP.** Replace with plain `*ptr = value` / `*ptr` reads — HIP/ROCm does not expose those cache control intrinsics. Performance impact is real but functionally equivalent.
- **The 7.05GB `nvidia/cuda` image is dev-tools only.** It has `nvcc` but no GPU runtime check at build time, which is what you want for compile-only ports.

## When the user provides only CUDA source

If the original CUDA code is missing context (e.g. a header is referenced but not provided, or a kernel is incomplete), reconstruct it minimally to get a compile, then port. Note any reconstructions in a comment so the user can verify.
