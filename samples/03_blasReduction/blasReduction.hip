#include "hip/hip_runtime.h"
/* blasReduction.cu
 *
 * Combines cuBLAS SGEMM with a custom warp-level reduction kernel.
 * Tests: cuBLAS library calls, warp shuffle intrinsics, hipOccupancyMaxPotentialBlockSize,
 *        pinned memory, streams, error handling with cuBLAS status codes.
 *
 * Source: Custom example for CUDA-to-ROCm conversion evaluation.
 *         Based on patterns from NVIDIA cuda-samples (simpleCUBLAS + reduction).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// Error checking macros
#define CHECK_CUDA(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    hipblasStatus_t stat = call; \
    if (stat != HIPBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)stat); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Warp-level reduction using shuffle intrinsics
__device__ float warpReduceSum(float val) {
    // Uses __shfl_down_sync with full warp mask
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
    return val;
}

// Block-level reduction kernel using shared memory + warp shuffle
__global__ void blockReduceSum(const float *input, float *output, int n) {
    __shared__ float shared[64]; // One element per warp (assuming 32 warps max)

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load from global memory
    float val = (idx < n) ? input[idx] : 0.0f;

    // Warp-level reduction
    int lane = tid % warpSize;       // Lane within warp
    int wid  = tid / warpSize;       // Warp ID within block
    val = warpReduceSum(val);

    // Write reduced value of each warp to shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Read from shared memory only if that warp existed
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (tid < numWarps) ? shared[tid] : 0.0f;

    // Final reduction within first warp
    if (wid == 0) {
        val = warpReduceSum(val);
    }

    // Write result for this block
    if (tid == 0) output[blockIdx.x] = val;
}

// Simple verification kernel: element-wise multiply
__global__ void elementMul(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] * B[idx];
    }
}

void initMatrix(float *data, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = val;
    }
}

int main(void) {
    printf("[cuBLAS + Warp Reduction Test] - Starting...\n");

    // Matrix dimensions for SGEMM: C(MxN) = A(MxK) * B(KxN)
    const int M = 256, K = 128, N = 256;
    const float alpha = 1.0f, beta = 0.0f;

    printf("Matrix multiply: C(%d,%d) = A(%d,%d) * B(%d,%d)\n", M, N, M, K, K, N);

    // Allocate host memory (pinned for async transfers)
    float *h_A, *h_B, *h_C;
    CHECK_CUDA(hipHostMalloc(&h_A, M * K * sizeof(float)));
    CHECK_CUDA(hipHostMalloc(&h_B, K * N * sizeof(float)));
    CHECK_CUDA(hipHostMalloc(&h_C, M * N * sizeof(float)));

    initMatrix(h_A, M, K, 1.0f);
    initMatrix(h_B, K, N, 0.01f);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_reduce_out;
    CHECK_CUDA(hipMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA(hipMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(hipMalloc(&d_C, M * N * sizeof(float)));

    // Create stream
    hipStream_t stream;
    CHECK_CUDA(hipStreamCreate(&stream));

    // Copy data to device
    CHECK_CUDA(hipMemcpyAsync(d_A, h_A, M * K * sizeof(float), hipMemcpyHostToDevice, stream));
    CHECK_CUDA(hipMemcpyAsync(d_B, h_B, K * N * sizeof(float), hipMemcpyHostToDevice, stream));

    // ========== cuBLAS SGEMM ==========
    printf("Performing cuBLAS SGEMM...\n");
    hipblasHandle_t handle;
    CHECK_CUBLAS(hipblasCreate(&handle));
    CHECK_CUBLAS(hipblasSetStream(handle, stream));

    // cuBLAS uses column-major, so we compute C = A * B by calling
    // hipblasSgemm with transposed arguments
    CHECK_CUBLAS(hipblasSgemm(handle,
        HIPBLAS_OP_N, HIPBLAS_OP_N,
        N, M, K,
        &alpha,
        d_B, N,
        d_A, K,
        &beta,
        d_C, N));

    CHECK_CUDA(hipStreamSynchronize(stream));
    printf("SGEMM complete.\n");

    // ========== Warp Reduction ==========
    printf("Performing warp-level reduction on SGEMM result...\n");
    int totalElements = M * N;

    // Use hipOccupancyMaxPotentialBlockSize to find optimal block size
    int minGridSize, blockSize;
    CHECK_CUDA(hipOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, blockReduceSum, 0, 0));
    printf("Optimal block size for reduction: %d\n", blockSize);

    int gridSize = (totalElements + blockSize - 1) / blockSize;
    CHECK_CUDA(hipMalloc(&d_reduce_out, gridSize * sizeof(float)));

    blockReduceSum<<<gridSize, blockSize, 0, stream>>>(d_C, d_reduce_out, totalElements);
    CHECK_CUDA(hipGetLastError());

    // Second pass: reduce the partial sums
    float *d_final_sum;
    CHECK_CUDA(hipMalloc(&d_final_sum, sizeof(float)));

    if (gridSize > 1) {
        blockReduceSum<<<1, blockSize, 0, stream>>>(d_reduce_out, d_final_sum, gridSize);
        CHECK_CUDA(hipGetLastError());
    } else {
        CHECK_CUDA(hipMemcpyAsync(d_final_sum, d_reduce_out, sizeof(float), hipMemcpyDeviceToDevice, stream));
    }

    // Copy result back
    float h_sum = 0.0f;
    CHECK_CUDA(hipMemcpyAsync(&h_sum, d_final_sum, sizeof(float), hipMemcpyDeviceToHost, stream));
    CHECK_CUDA(hipStreamSynchronize(stream));

    // Verify: each element of C should be K * 1.0 * 0.01 = 1.28
    // Total sum should be M * N * 1.28 = 256 * 256 * 1.28 = 83886.08
    float expected = (float)M * N * K * 1.0f * 0.01f;
    float rel_err = fabs(h_sum - expected) / fabs(expected);
    printf("Reduction sum: %.2f, Expected: %.2f, Relative error: %e\n", h_sum, expected, rel_err);

    bool passed = (rel_err < 1e-5);
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUBLAS(hipblasDestroy(handle));
    CHECK_CUDA(hipFree(d_A));
    CHECK_CUDA(hipFree(d_B));
    CHECK_CUDA(hipFree(d_C));
    CHECK_CUDA(hipFree(d_reduce_out));
    CHECK_CUDA(hipFree(d_final_sum));
    CHECK_CUDA(hipHostFree(h_A));
    CHECK_CUDA(hipHostFree(h_B));
    CHECK_CUDA(hipHostFree(h_C));
    CHECK_CUDA(hipStreamDestroy(stream));

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
