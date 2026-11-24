# Docker Setup for CUDA and HIP/ROCm

The porting workflow uses two containers: one for CUDA compilation, one for HIP/ROCm compilation. Neither needs an actual GPU for the compile-iterate-fix loop — only for runtime testing.

## Images used in this project

| Purpose | Image | Size | Notes |
|---|---|---|---|
| CUDA compile | `nvidia/cuda:13.2.1-devel-ubuntu22.04` | 7.05 GB | Has `nvcc`, cuBLAS, cuBLASLt, cuDNN headers. Use this as the baseline. |
| Older CUDA | `nvidia/cuda:12.3.2-devel-ubuntu22.04` | 7.07 GB | Use only if porting a project that requires CUDA 12.x APIs. |
| ROCm/HIP | `rocm/rocm-terminal:latest` | 2.99 GB | Has `hipcc`, `hipify-perl`, `hipify-clang`, hipBLAS, rocWMMA, rocBLAS headers. |

The `-devel-` tag is required — the `-runtime-` and `-base-` tags do **not** include `nvcc`.

## Pull

```bash
docker pull nvidia/cuda:13.2.1-devel-ubuntu22.04
docker pull rocm/rocm-terminal:latest
```

## Compile-only run (no GPU needed)

This is the workhorse pattern. Mount the source dir, run a single compile command, and exit.

```bash
# CUDA
docker run --rm \
    -v /scratch/rozum/cuda:/work -w /work \
    nvidia/cuda:13.2.1-devel-ubuntu22.04 \
    nvcc -O2 -arch=sm_80 -o myapp_cuda myapp.cu

# HIP
docker run --rm \
    -v /scratch/rozum/cuda:/work -w /work \
    rocm/rocm-terminal:latest \
    bash -lc "hipcc -O2 -o myapp_hip myapp.hip"
```

Why `bash -lc` for the ROCm container: `rocm/rocm-terminal` runs as the unprivileged `rocm-user`; the `-l` (login shell) ensures `PATH` picks up `/opt/rocm/bin/hipcc`.

## Interactive sessions for iterating

When you're going to run many builds, start a long-lived container and `docker exec` into it:

```bash
# CUDA
docker run -d --name cuda \
    -v /scratch/rozum/cuda:/work -w /work \
    nvidia/cuda:13.2.1-devel-ubuntu22.04 sleep infinity

docker exec -it cuda bash

# ROCm
docker run -d --name rocm \
    -v /scratch/rozum/cuda:/work -w /work \
    rocm/rocm-terminal:latest sleep infinity

docker exec -it rocm bash
```

Stop and remove when done: `docker stop cuda rocm && docker rm cuda rocm`.

The existing `cuda` container at `nvidia/cuda:12.3.2-devel-ubuntu22.04` (container ID `4d621494dcbc`) is the one used for the `01_*` through `08_*` builds — restart it with `docker start -ai cuda` rather than re-creating if you want the same exact environment.

## Running with a real GPU

Only required if the user wants to *execute* the binary, not just compile it.

```bash
# NVIDIA — needs nvidia-container-toolkit installed on host
docker run --rm --gpus all \
    -v /scratch/rozum/cuda:/work -w /work \
    nvidia/cuda:13.2.1-devel-ubuntu22.04 \
    ./myapp_cuda

# AMD — needs /dev/kfd and /dev/dri passthrough; user must be in video/render groups
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --security-opt seccomp=unconfined \
    -v /scratch/rozum/cuda:/work -w /work \
    rocm/rocm-terminal:latest \
    ./myapp_hip
```

If the host has no GPU at all, kernel launches will fail at `cudaDeviceSynchronize`/`hipDeviceSynchronize` with an error — the binary still loads and runs the host code. The `hello_cuda.cu` and `hello_hip.cpp` reference programs at the repo root demonstrate this — they print runtime version info and gracefully report the missing GPU.

## Troubleshooting

- **`nvcc: command not found`** — you pulled a non-`-devel-` image. Re-pull with `-devel-`.
- **`hipcc: not found` after `docker exec`** — use `bash -lc` instead of `bash -c`, or source `/etc/profile.d/rocm.sh`.
- **Permission denied writing build artifacts** — `rocm/rocm-terminal` runs as UID 1000 (`rocm-user`). Either mount a directory you own as UID 1000, or `chown` the output afterwards. Some users add `--user $(id -u):$(id -g)` to the `docker run` line, but that breaks ROCm tools that expect to find user `rocm-user`.
- **Image keeps re-downloading** — your container runtime may be configured with a per-job ephemeral cache. Pre-pull on the build host.
