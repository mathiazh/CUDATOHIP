# Build Commands: nvcc and hipcc

## nvcc (CUDA)

Single-file build:
```bash
nvcc -O2 -std=c++17 -o foo_cuda foo.cu
```

With architecture targeting (`sm_XX` = SM major.minor):
```bash
nvcc -O2 -arch=sm_80 -o foo_cuda foo.cu          # A100
nvcc -O2 -arch=sm_86 -o foo_cuda foo.cu          # RTX 3090, A40
nvcc -O2 -arch=sm_89 -o foo_cuda foo.cu          # H100/L40 (Hopper/Ada)
nvcc -O2 -arch=sm_90 -o foo_cuda foo.cu          # H100 (compute_90)
```

For multi-arch fat binaries (slower compile, runs anywhere):
```bash
nvcc -O2 \
    -gencode arch=compute_70,code=sm_70 \
    -gencode arch=compute_80,code=sm_80 \
    -gencode arch=compute_90,code=sm_90 \
    -o foo_cuda foo.cu
```

Common library links:
```bash
-lcublas -lcublasLt   # cuBLAS / cuBLASLt
-lcurand              # random
-lcufft               # FFT
-lcudnn               # cuDNN (only with USE_CUDNN=1)
-lnvrtc               # runtime compilation
```

Useful flags from `07_llmc/Makefile`:
```
--threads=0 -t=0       # parallel device compile, 0 = nproc
--use_fast_math        # relaxed FP, big speedup
-std=c++17             # required for many modern projects
-O3                    # default for production
-G                     # device-side debug (much slower)
-lineinfo              # source mapping for nsight without -G slowdown
```

## hipcc (HIP)

Single-file build:
```bash
hipcc -O2 -std=c++17 -o foo_hip foo.hip
```

With architecture targeting (`gfx9XX` = AMD ISA):
```bash
hipcc -O2 --offload-arch=gfx908  -o foo_hip foo.hip   # MI100
hipcc -O2 --offload-arch=gfx90a  -o foo_hip foo.hip   # MI200 / MI210 / MI250
hipcc -O2 --offload-arch=gfx940  -o foo_hip foo.hip   # MI300A
hipcc -O2 --offload-arch=gfx942  -o foo_hip foo.hip   # MI300X
hipcc -O2 --offload-arch=gfx1100 -o foo_hip foo.hip   # RX 7900 XTX
```

Multi-arch:
```bash
hipcc -O2 \
    --offload-arch=gfx90a \
    --offload-arch=gfx942 \
    -o foo_hip foo.hip
```

Common library links:
```bash
-lhipblas -lhipblaslt   # hipBLAS / hipBLASLt (cuBLAS / cuBLASLt equivalents)
-lhiprand               # random
-lhipfft                # FFT
-lmiopen                # cuDNN equivalent
# rocWMMA is header-only — just include <rocwmma/rocwmma.hpp>
```

Useful flags:
```
-fgpu-rdc                # relocatable device code (needed for some multi-TU projects)
-fno-gpu-rdc             # default, faster
-ggdb                    # host debug
-O3                      # production
-Rpass=...               # pass-specific opt remarks
```

## Architecture cheat sheet (CUDA → HIP)

| NVIDIA arch | AMD equivalent (closest) | Notes |
|---|---|---|
| `sm_70` (V100)  | `gfx906` (MI50)  | First-gen tensor cores ↔ no MFMA |
| `sm_75` (T4)    | `gfx906`         | INT8 tensor cores |
| `sm_80` (A100)  | `gfx90a` (MI200) | Both have BF16 + FP64 tensor units |
| `sm_86` (A40, RTX 3090) | `gfx1100` (RX 7900) | Consumer-class |
| `sm_89` (L40, RTX 4090) | `gfx1100`        | |
| `sm_90` (H100)  | `gfx942` (MI300X)| FP8, much wider matrix ops |

When in doubt for HIP: `gfx90a` is the safest "modern datacenter" target.

## Querying the host

```bash
# CUDA
nvidia-smi
nvcc --version
nvcc --list-gpu-arch

# HIP
rocm-smi
hipcc --version
rocminfo | grep -A1 "ISA Info"
```

## Makefile patterns from this project

The `07_llmc` `Makefile` is a good reference for a real project:
- Uses `?=` for overridable defaults (`CC ?= clang`, `FORCE_NVCC_O ?= 3`)
- Builds `.o` files into a `build/` subdir
- Conditional `USE_CUDNN=1`, `USE_MULTI_GPU=1` flags
- Detects host with `$(shell uname)` for cross-platform paths

For a pure HIP equivalent, swap:
- `nvcc` → `hipcc`
- `-lcublas -lcublasLt` → `-lhipblas -lhipblaslt`
- `-arch=sm_80` → `--offload-arch=gfx90a`
- `--use_fast_math` → `-ffast-math`
- `-Xcompiler -fopenmp` → `-fopenmp` (hipcc passes through directly)
