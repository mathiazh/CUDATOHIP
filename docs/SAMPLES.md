# Sample Projects - What Each One Exercises

The samples are intentionally ordered from trivial to production-shaped. Each
row lists what hipify handles automatically and what requires manual rewrite.

| # | Project | NVIDIA libs | CUDA features | hipify auto? | Manual rewrite? | Tech-plan gap |
|---|---|---|---|---|---|---|
| 01 | vectorAdd | none | runtime API only | 100% | none | none |
| 02 | matrixMul | none | shared memory | 100% | none | none |
| 03 | blasReduction | cuBLAS | `__shfl_down_sync(0xFFFFFFFF, ...)` | mostly | wave-size hardcode | §3.3 latent, §3.7 |
| 04 | warpBallotScan | none | `__ballot_sync`, `__shfl_*_sync`, `cg::tiled_partition<32>`, `asm("vote.sync...")`, `__half` | partial | warp-size + cooperative-groups tile + inline PTX | §3.1, §3.3 |
| 05 | wmmaGemm | none | `nvcuda::wmma::fragment`, `mma_sync`, fp16 | **zero** for WMMA | full rewrite to `rocwmma::` | §3.2 |
| 06 | texturePTX | none | `cudaTextureObject_t`, `asm("dp4a.s32.s32 ...")` | partial | texture pass + dp4a lift | §3.1 |
| 07 | llmc (GPT-2) | cuBLASLt, cuBLAS, cuDNN, NCCL, NVTX, NVML, MPI | warp-sync, cooperative groups, Unified Memory, `__ldcs`/`__ldg`, fp16/bf16, shared memory | varies per file | many | §3.2, §3.3, §3.6, §3.7, §3.8, §3.10 |
| 08 | cudaSamples (NVIDIA) | every NVIDIA library across ~190 individual samples | every CUDA feature class (NVRTC, Driver API, CUDA Graphs, Dynamic Parallelism, multi-GPU, Vulkan/D3D/EGL interop, etc.) | varies per sample | varies per sample | all §3.x gaps + interop blind spots beyond the tech plan |

## Why the order matters

- **01-02** confirm the toolchain works before any complications.
- **03** introduces the first library (cuBLAS &rarr; hipBLAS) and the first
  hidden gotcha: the `0xFFFFFFFF` warp mask is a wave32 assumption that
  breaks silently on CDNA wave64.
- **04** stacks three independent wave-size-32 assumptions on top of each
  other plus an inline-PTX block. Most instructive for understanding §3.3.
- **05** is the first sample hipify cannot meaningfully translate. Read the
  `wmmaGemm_rewrite.hip.txt` notes next to it - the comment at the top
  says "HIPIFY translated ZERO WMMA code. 16 manual changes required."
- **06** is the simplest inline-PTX case (`dp4a` has a clean AMD intrinsic).
  Use this to understand the PTX lifting problem before tackling harder PTX
  ops (cp.async, mma.sync, ld.global.nc) that live in real codebases.
- **07** is the "real workload" shape: ~18K LOC of CUDA across 28 files,
  pulling in five NVIDIA libraries, multi-GPU, mixed precision. Run the
  profiler on it (`python3 tools/profile_cuda_project.py samples/07_llmc`)
  to see the full feature inventory before deciding which files to port
  first.

## What the samples do NOT cover

Known blind spots in this corpus, worth supplementing later:

- **CUDA Graphs** (§3.5) - none of the samples use them. Real inference
  servers do.
- **Driver API** beyond NVML (§3.10) - llmc uses NVML for FLOPS reporting
  but no `cuCtx*` / `cuMemMap` / VMM patterns.
- **Atomics with explicit scope** (§3.4) - no sample exercises
  `cuda::atomic` or `__threadfence_system`.
- **CUDA Dynamic Parallelism** - rare in production; documented as a
  restructuring task in the tech plan, not a translation.

If you need to teach or test against those gaps, add a sample.

## How to use sample 08

The NVIDIA CUDA Samples repo is included whole (~190 individual samples
under `samples/08_cudaSamples/Samples/`, grouped into categories 0-8). It is
not meant to be read end-to-end. Use it as a feature-coverage corpus:

- **Pick by feature.** Need a CUDA Graphs example? Look in
  `Samples/3_CUDA_Features/`. Need NVRTC? Check `clock_nvrtc`,
  `matrixMul_nvrtc`. Need Driver API? `matrixMulDrv`. Need cooperative
  groups beyond what 04 covers? `Samples/3_CUDA_Features/cdpSimpleQuicksort`,
  `simpleCooperativeGroups`.
- **Use the profiler to triage.** Run
  `python3 tools/profile_cuda_project.py samples/08_cudaSamples/Samples/<name>`
  on any individual sample to see what it exercises before porting.
- **It fills the documented blind spots** in samples 01-07 (CUDA Graphs,
  Driver API patterns beyond NVML, atomic scopes, Dynamic Parallelism,
  graphics interop). When you need to teach or test a §3.5 / §3.10 case,
  grab a sample from here instead of writing one from scratch.
- **It is not a port target.** Hipify-ing the whole tree at once is a
  mistake; the samples have very different difficulty and many depend on
  NVIDIA-only APIs (NVENC, NVDEC, OptiX) that have no AMD equivalent.
