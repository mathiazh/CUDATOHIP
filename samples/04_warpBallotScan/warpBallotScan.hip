#include "hip/hip_runtime.h"
/* warpBallotScan.cu
 *
 * Warp-level ballot scan with mixed precision, cooperative groups,
 * inline PTX, and device attribute queries.
 *
 * Tests: __ballot_sync, __popc, __shfl_up_sync, cooperative_groups,
 *        __half precision, inline PTX, hipFuncGetAttributes,
 *        hipDeviceGetAttribute, multiple warp-size-dependent patterns.
 *
 * Source: Custom example for CUDA-to-ROCm conversion evaluation (Score D target).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Kernel 1: Warp ballot scan - counts active threads with predicates
// Uses __ballot_sync and __popc (warp-size-dependent intrinsics)
// ============================================================
__global__ void warpBallotScan(const float *input, int *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % warpSize;  // Fixed: was hardcoded 32

    // Each thread checks if its value exceeds threshold
    float val = (idx < n) ? input[idx] : 0.0f;
    int pred = (val > 0.5f) ? 1 : 0;

    // Ballot: collect predicate across entire warp
    unsigned long long ballot = __ballot(pred);

    // Count active threads in warp using __popc (population count)
    int activeCount = __popcll(ballot);

    // Exclusive prefix sum within warp using ballot + mask
    unsigned long long mask = (1ULL << lane) - 1;  // Mask for lanes below current
    int prefixSum = __popcll(ballot & mask);

    if (idx < n) {
        output[idx] = prefixSum;  // Each thread gets its prefix sum
    }

    // First thread of each warp reports the total
    if (lane == 0 && idx < n) {
        printf("Warp %d: %d of %d threads active, ballot=0x%016llX\n",
               idx / warpSize, activeCount, (int)warpSize, ballot);
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
    for (int offset = 1; offset < warpSize; offset *= 2) {
        float neighbor = __shfl_up(val, offset);
        if (threadIdx.x % warpSize >= offset) {
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
    // NOTE: tile<32> works on RDNA (warp=32) but may need tile<64> on CDNA (wavefront=64)
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
    // Rewritten from inline PTX to HIP intrinsic
    // Original used NVIDIA PTX: vote.all.pred
    // HIP equivalent: __all() intrinsic
    return __all(pred);
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
    CHECK_CUDA(hipGetDevice(&device));

    int warpSize, maxThreads, smCount, sharedMemPerBlock;
    CHECK_CUDA(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, device));
    CHECK_CUDA(hipDeviceGetAttribute(&maxThreads, hipDeviceAttributeMaxThreadsPerBlock, device));
    CHECK_CUDA(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
    CHECK_CUDA(hipDeviceGetAttribute(&sharedMemPerBlock, hipDeviceAttributeMaxSharedMemoryPerBlock, device));

    printf("Device %d: warpSize=%d, maxThreads=%d, SMs=%d, sharedMem=%d\n",
           device, warpSize, maxThreads, smCount, sharedMemPerBlock);

    // Get kernel attributes
    hipFuncAttributes attrs;
    CHECK_CUDA(hipFuncGetAttributes(&attrs, reinterpret_cast<const void*>(warpBallotScan)));
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
    CHECK_CUDA(hipMalloc(&d_input, size_f));
    CHECK_CUDA(hipMalloc(&d_output, size_f));
    CHECK_CUDA(hipMalloc(&d_iinput, size_i));
    CHECK_CUDA(hipMalloc(&d_ioutput, size_i));

    CHECK_CUDA(hipMemcpy(d_input, h_input, size_f, hipMemcpyHostToDevice));

    // Print device info
    printDeviceInfo();

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Launch kernels
    printf("\nLaunching warpBallotScan...\n");
    warpBallotScan<<<gridSize, blockSize>>>(d_input, d_ioutput, N);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());

    printf("\nLaunching warpInclusiveScan...\n");
    CHECK_CUDA(hipMemcpy(d_output, d_input, size_f, hipMemcpyDeviceToDevice));
    warpInclusiveScan<<<gridSize, blockSize>>>(d_output, N);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());

    printf("\nLaunching mixedPrecisionKernel...\n");
    mixedPrecisionKernel<<<gridSize, blockSize>>>(d_input, d_output, N);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());

    // Verify mixed precision result BEFORE cooperativeGroupsKernel overwrites d_output
    CHECK_CUDA(hipMemcpy(h_output, d_output, size_f, hipMemcpyDeviceToHost));

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
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());

    // Cleanup
    CHECK_CUDA(hipFree(d_input));
    CHECK_CUDA(hipFree(d_output));
    CHECK_CUDA(hipFree(d_iinput));
    CHECK_CUDA(hipFree(d_ioutput));
    free(h_input);
    free(h_output);
    free(h_ioutput);

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
