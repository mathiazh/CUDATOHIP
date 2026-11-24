# Quickstart - Your First Port in 10 Minutes

This walks through porting `samples/01_vectorAdd` from CUDA to HIP using the
`cuda-hip-porting` Claude Code skill, then verifies on both an NVIDIA and an
AMD GPU.

## Prerequisites

- `claude` (Claude Code) installed and authenticated.
- `nvcc` available (or a CUDA Docker image - see `skill/cuda-hip-porting/reference/docker-setup.md`).
- `hipcc` available (or a ROCm Docker image - same file).
- The `cuda-hip-porting` skill installed to `~/.claude/skills/` (see the top-level README).

## 1. Profile first

```bash
python3 tools/profile_cuda_project.py samples/01_vectorAdd
```

You'll see: 1 CUDA file, no NVIDIA libraries, no special CUDA features, zero
tech-plan gaps. Good - this is the sanity baseline.

## 2. Ask Claude Code to do the port

From the `samples/01_vectorAdd` directory, start `claude` and paste the prompt
from `docs/EXAMPLE_AI_COMMAND.md` (the "First port" prompt). The skill will:

1. Run `hipify-perl` (or `hipify-clang` if a compile-commands DB exists).
2. Read the warnings and apply manual rewrites for anything hipify flagged.
3. Build with `hipcc`.
4. Run, compare output against the CUDA baseline.

For `01_vectorAdd` no manual rewrites are needed - it's pure hipify.

## 3. Verify

```bash
cd samples/01_vectorAdd
nvcc vectorAdd.cu -o vectorAdd_cuda    # builds on NVIDIA host
hipcc vectorAdd.hip -o vectorAdd_hip   # builds on AMD host
./vectorAdd_cuda > /tmp/out_cuda.txt
./vectorAdd_hip  > /tmp/out_hip.txt
diff /tmp/out_cuda.txt /tmp/out_hip.txt && echo OK
```

## 4. Move up the difficulty curve

Repeat the prompt on `samples/03_blasReduction` next (first library port). Then
**05_wmmaGemm** - this is the most instructive sample; hipify will warn but
emit no rewrites for the WMMA fragments and the skill will walk through the
rocWMMA replacement.

## 5. Read what the skill knows

Open `skill/cuda-hip-porting/SKILL.md` and the files under `reference/`. The
recipes there are the actual knowledge you're transferring - the AI prompts
just drive them.
