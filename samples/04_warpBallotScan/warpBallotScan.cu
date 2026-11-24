/* warpBallotScan.cu
 *
 * Warp-level ballot scan with mixed precision, cooperative groups,
 * inline PTX, and device attribute queries.
 *
 * Tests: __ballot_sync, __popc, __shfl_up_sync, cooperative_groups,
 *        __half precision, inline PTX, cudaFuncGetAttributes,
 *        cudaDeviceGetAttribute, multiple warp-size-dependent patterns.
 *
 * Source: Custom example for CUDA-to-ROCm conversion evaluation (Score D target).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Kernel 1: Warp ballot scan - counts active threads with predicates
// Uses __ballot_sync and __popc (warp-size-dependent intrinsics)
// ============================================================
__global__ void warpBallotScan(const float *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;

    // Each thread checks if its value exceeds threshold
    float val = (idx < n) ? input[idx] : 0.0f;
    int pred = (val > 0.5f) ? 1 : 0;

    // Ballot: collect predicate across entire warp
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, pred);

    // Count active threads in warp using __popc (population count)
    int activeCount = __popc(ballot);

    // Exclusive prefix sum within warp using ballot + mask
    unsigned int mask = (1u << lane) - 1;  // Mask for lanes below current
    int prefixSum = __popc(ballot & mask);

    if (idx < n) {
        output[idx] = prefixSum;  // Each thread gets its prefix sum
    }

    // First thread of each warp reports the total
    if (lane == 0 && idx < n) {
        printf("Warp %d: %d of 32 threads active, ballot=0x%08X\n",
               idx / 32, activeCount, ballot);
    }
}

// ============================================================
// Kernel 2: Inclusive scan using __shfl_up_sync (warp-level)
// ============================================================
__global__ void warpInclusiveScan(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];

    // Warp-level inclusive scan using shuffle-up
    for (int offset = 1; offset < 32; offset *= 2) {
        float neighbor = __shfl_up_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x % 32 >= offset) {
            val += neighbor;
        }
    }

    data[idx] = val;
}

// ============================================================
// Kernel 3: Mixed precision - convert float to half and back
// Uses __half type and __float2half / __half2float intrinsics
// ============================================================
__global__ void mixedPrecisionKernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Convert to half precision
    __half h_val = __float2half(input[idx]);

    // Do some computation in half precision
    __half h_two = __float2half(2.0f);
    __half h_result = __hmul(h_val, h_two);

    // Convert back to float
    output[idx] = __half2float(h_result);
}

// ============================================================
// Kernel 4: Cooperative groups - thread block tile operations
// ============================================================
__global__ void cooperativeGroupsKernel(float *data, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = data[idx];

    // Use cooperative groups shuffle instead of direct intrinsics
    float sum = val;
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum += warp.shfl_down(sum, offset);
    }

    // Broadcast result from lane 0 to all threads in warp
    sum = warp.shfl(sum, 0);

    data[idx] = sum;
}

// ============================================================
// Kernel 5: Inline PTX for warp-level vote
// ============================================================
__device__ int ptxVoteAll(int pred) {
    // Updated for CUDA 9.0+/sm_70+ PTX ISA: vote requires explicit .sync with mask.
    // Original 'vote.all.pred' (pre-sync) fails to assemble on CUDA 12.6 ptxas
    // ("Instruction 'vote' without '.sync' is not supported on .target sm_70 and higher").
    int result;
    asm volatile("{ \n\t"
                 "  .reg .pred p, q; \n\t"
                 "  setp.ne.s32 p, %1, 0; \n\t"
                 "  vote.sync.all.pred q, p, 0xffffffff; \n\t"
                 "  selp.s32 %0, 1, 0, q; \n\t"
                 "}" : "=r"(result) : "r"(pred));
    return result;
}

__global__ void ptxVoteKernel(const int *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    int allPositive = ptxVoteAll(val > 0);
    output[idx] = allPositive;
}

// ============================================================
// Host: Device attribute queries and kernel launch
// ============================================================
void printDeviceInfo() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));

    int warpSize, maxThreads, smCount, sharedMemPerBlock;
    CHECK_CUDA(cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&maxThreads, cudaDevAttrMaxThreadsPerBlock, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, device));

    printf("Device %d: warpSize=%d, maxThreads=%d, SMs=%d, sharedMem=%d\n",
           device, warpSize, maxThreads, smCount, sharedMemPerBlock);

    // Get kernel attributes
    cudaFuncAttributes attrs;
    CHECK_CUDA(cudaFuncGetAttributes(&attrs, warpBallotScan));
    printf("warpBallotScan: regs=%d, sharedMem=%zu, maxThreads=%d\n",
           attrs.numRegs, attrs.sharedSizeBytes, attrs.maxThreadsPerBlock);
}

int main(void) {
    printf("[Warp Ballot Scan + Mixed Precision + Cooperative Groups + PTX] - Starting...\n");

    const int N = 256;
    size_t size_f = N * sizeof(float);
    size_t size_i = N * sizeof(int);

    // Allocate
    float *h_input = (float *)malloc(size_f);
    float *h_output = (float *)malloc(size_f);
    int *h_ioutput = (int *)malloc(size_i);

    // Initialize with values 0.0 - 1.0
    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    float *d_input, *d_output;
    int *d_iinput, *d_ioutput;
    CHECK_CUDA(cudaMalloc(&d_input, size_f));
    CHECK_CUDA(cudaMalloc(&d_output, size_f));
    CHECK_CUDA(cudaMalloc(&d_iinput, size_i));
    CHECK_CUDA(cudaMalloc(&d_ioutput, size_i));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size_f, cudaMemcpyHostToDevice));

    // Print device info
    printDeviceInfo();

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernels
    printf("\nLaunching warpBallotScan...\n");
    warpBallotScan<<<gridSize, blockSize>>>(d_input, d_ioutput, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\nLaunching warpInclusiveScan...\n");
    CHECK_CUDA(cudaMemcpy(d_output, d_input, size_f, cudaMemcpyDeviceToDevice));
    warpInclusiveScan<<<gridSize, blockSize>>>(d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("\nLaunching mixedPrecisionKernel...\n");
    mixedPrecisionKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Verify mixed precision result BEFORE cooperativeGroupsKernel overwrites d_output
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size_f, cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < N; i++) {
        float expected = h_input[i] * 2.0f;
        if (fabs(h_output[i] - expected) > 0.01f) {
            printf("Mismatch at %d: got %.4f, expected %.4f\n", i, h_output[i], expected);
            passed = false;
        }
    }
    printf("\nMixed precision test: %s\n", passed ? "PASS" : "FAIL");

    printf("\nLaunching cooperativeGroupsKernel...\n");
    cooperativeGroupsKernel<<<gridSize, blockSize>>>(d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_iinput));
    CHECK_CUDA(cudaFree(d_ioutput));
    free(h_input);
    free(h_output);
    free(h_ioutput);

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
