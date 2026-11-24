# Example AI Commands

Copy-paste these into a `claude` session running in the project directory.
The `cuda-hip-porting` skill must be installed (see top-level README).

---

## A. First port - "do the whole thing"

Use this on any of the seven samples. Best first run: `samples/01_vectorAdd`.

> Use the cuda-hip-porting skill to port this project from CUDA to HIP.
> Steps I want you to follow:
>   1. Run hipify on the .cu files and save the output as .hip files.
>   2. Read every hipify warning. For any warning that hipify did not
>      auto-rewrite, apply the manual rewrite recipe from the skill's
>      reference files and explain what you changed and why.
>   3. Build the original CUDA version with nvcc and the new HIP version
>      with hipcc. Fix build errors as they appear.
>   4. Run both binaries on small input and diff the output. Report any
>      numerical differences.
>   5. Summarize: which APIs translated cleanly, which required manual
>      rewrites, and any unresolved gaps.

---

## B. Just profile this codebase - no port yet

Use this on an unfamiliar CUDA codebase before deciding to port.

> Run tools/profile_cuda_project.py on this directory. Then read the
> output and tell me:
>   - Which NVIDIA libraries are in use, and what their ROCm equivalents are.
>   - Which CUDA features (inline PTX, WMMA, warp-sync, cooperative groups,
>     Unified Memory, texture objects, Driver API, NVML) are present.
>   - Which tech-plan gaps (3.1 through 3.10) this codebase exercises.
>   - Based on that, estimate the difficulty: trivial / moderate / hard /
>     research-grade. Justify the rating.

---

## C. Port only what hipify cannot handle

Useful when hipify has already been run and you want focused help on the
warnings.

> Read hipify_stats.txt (or stderr.txt) in this directory. For each warning
> that hipify did not auto-rewrite, locate the relevant code in the .cu
> source, apply the manual rewrite recipe from the cuda-hip-porting skill,
> and produce the corresponding .hip code. Do not touch anything hipify
> already translated cleanly.

---

## D. Verify a hand-written port against the CUDA baseline

Useful when you (or a previous engineer) already produced a .hip file and
want a second opinion.

> Compare the CUDA source (.cu) and the HIP port (.hip) in this directory.
> Identify:
>   - Any CUDA construct in the .cu that has no corresponding rewrite in the
>     .hip (silent omission).
>   - Any rewrite in the .hip that does not preserve the semantics of the
>     original (wrong translation).
>   - Any place where the port hard-codes a 32-thread warp assumption that
>     will fail on CDNA wave64.
>   - Any place where the port silently changes numerical behavior
>     (atomic scope, fence type, rounding mode, mixed precision).
> For each finding, cite the line numbers in both files.

---

## E. Demonstrate a specific gap (teaching mode)

Useful for onboarding a new colleague.

> I want to see why the cuda-hip-porting skill calls out §3.2 (Tensor Core /
> MMA) as a Critical gap. Run hipify on samples/05_wmmaGemm/wmmaGemm.cu.
> Show me the warnings hipify emits. Then walk through the rocWMMA rewrite
> step by step, pointing out each place where the AMD fragment layout
> differs from the NVIDIA WMMA fragment layout.

---

## F. Apply to a real customer project (sanitized)

Use this when picking up a customer codebase. Replace `<PATH>`.

> Treat `<PATH>` as a customer CUDA codebase I want to evaluate for
> porting to ROCm. Do not modify any files in that tree.
>   1. Run the profiler on it.
>   2. Open the top three most CUDA-feature-dense files and summarize what
>      each does and which cuda-hip-porting recipes will apply.
>   3. Produce a one-page migration scoping note: what hipify will handle,
>      what needs manual work, which libraries lack ROCm equivalents, and
>      a rough difficulty rating.
> Output the scoping note as Markdown.

---

## Tips

- The skill assumes you are running from inside the project directory so
  it can resolve relative paths to `.cu` files. `cd` first.
- If hipify is not on PATH, the skill will try the Docker image in
  `skill/cuda-hip-porting/reference/docker-setup.md`. Bring that up first.
- For library-heavy ports (anything pulling in cuBLASLt, cuDNN, NCCL),
  pass `USE_CUDNN=1` or equivalent flags explicitly - the skill will not
  guess your build configuration.
- If the model gets confused, the single best recovery prompt is:
  "Stop. Re-read the cuda-hip-porting skill's reference/api-rewrites.md
  and tell me which recipe applies here before changing any code."
