# CUDA-to-HIP Porting Handoff Package

A self-contained handoff of the AI-assisted CUDA-to-HIP / ROCm porting skill, the
sample projects used to develop and validate it, and a small profiler tool for
characterizing new CUDA codebases.

## What's inside

```
.
├── README.md                     <-- you are here
├── docs/
│   ├── QUICKSTART.md             <-- 10-minute walkthrough; start here
│   ├── EXAMPLE_AI_COMMAND.md     <-- copy/paste prompts for Claude Code
│   └── SAMPLES.md                <-- what each sample exercises and why
├── skill/
│   └── cuda-hip-porting/         <-- drop this in ~/.claude/skills/
│       ├── SKILL.md
│       └── reference/
│           ├── api-rewrites.md
│           ├── build-commands.md
│           ├── docker-setup.md
│           ├── hipify-workflow.md
│           └── troubleshooting.md
├── samples/                      <-- 7 CUDA projects, ordered by difficulty
│   ├── 01_vectorAdd/             trivial; sanity baseline
│   ├── 02_matrixMul/             shared-memory tile GEMM
│   ├── 03_blasReduction/         cuBLAS + warp shuffle
│   ├── 04_warpBallotScan/        ballot/shfl + cooperative groups + inline PTX
│   ├── 05_wmmaGemm/              Tensor Core WMMA (hipify translates ZERO)
│   ├── 06_texturePTX/            texture objects + dp4a inline PTX
│   ├── 07_llmc/                  Karpathy GPT-2: cuBLASLt + cuDNN + NCCL + NVTX
│   └── 08_cudaSamples/           NVIDIA CUDA Samples repo (~190 individual samples)
└── tools/
    └── profile_cuda_project.py   static profiler; answers the discovery questionnaire
```

## Install in 60 seconds

```bash
# 1. Install the skill so Claude Code picks it up.
mkdir -p ~/.claude/skills
cp -r skill/cuda-hip-porting ~/.claude/skills/

# 2. Verify the profiler runs.
python3 tools/profile_cuda_project.py samples/01_vectorAdd

# 3. Read the quickstart and then the example prompts.
$EDITOR docs/QUICKSTART.md
$EDITOR docs/EXAMPLE_AI_COMMAND.md
```

You will also need a working ROCm (`hipcc`) and CUDA (`nvcc`) toolchain for the
build/verify step. See `skill/cuda-hip-porting/reference/docker-setup.md` for the
Docker images we use.

## Recommended path through the samples

The samples are intentionally ordered. If you only have time for one, do **05**
&mdash; it shows the most interesting failure mode of `hipify` (it translates
zero WMMA code) and the manual rewrite recipe.

1. **01 vectorAdd** &mdash; prove the build + hipify + run loop works end to end.
2. **03 blasReduction** &mdash; first library port (cuBLAS &rarr; hipBLAS).
3. **05 wmmaGemm** &mdash; Tensor Cores; this is the moment hipify stops doing the work for you.
4. **06 texturePTX** &mdash; inline PTX; needs manual AMDGCN equivalents.
5. **07 llmc** &mdash; the &ldquo;real codebase&rdquo; shape: cuBLASLt + cuDNN + NCCL + multi-GPU.

Samples 02 and 04 are short and worth doing for completeness; they are not on
the critical path.

## What the profiler is for

`tools/profile_cuda_project.py <dir>` walks a CUDA project and reports which
NVIDIA libraries, CUDA features (inline PTX, WMMA, warp-sync, cooperative
groups, Unified Memory, texture objects, etc.) and tech-plan gaps are present.
Use it before starting a port to know what you're walking into, and as a
discovery aid in customer conversations.

```bash
python3 tools/profile_cuda_project.py samples/07_llmc \
  --json /tmp/llmc.json \
  --md   /tmp/llmc.md
```
