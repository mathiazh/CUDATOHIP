/* wmmaGemm.cu
 *
 * Matrix multiplication using NVIDIA Tensor Cores via WMMA API.
 * C = alpha * A * B + beta * C (half-precision inputs, float accumulator)
 *
 * Tests: nvcuda::wmma namespace, wmma::fragment, wmma::load_matrix_sync,
 *        wmma::mma_sync, wmma::store_matrix_sync, __half, cooperative_groups,
 *        warp-level Tensor Core operations, cudaOccupancy.
 *
 * Based on: NVIDIA cuda-samples/cudaTensorCoreGemm
 * Source: Custom example for CUDA-to-ROCm conversion evaluation (Score E target).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// WMMA tile dimensions (fixed by hardware)
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Matrix dimensions (must be multiples of WMMA tile size)
const int M_GLOBAL = 64;
const int N_GLOBAL = 64;
const int K_GLOBAL = 64;

// ============================================================
// WMMA GEMM Kernel: C = alpha * A * B + beta * C
// A is MxK (row-major, half), B is KxN (row-major, half), C is MxN (row-major, float)
// ============================================================
__global__ void wmmaGemmKernel(
    const __half *A, const __half *B, float *C,
    int M, int N, int K,
    float alpha, float beta)
{
    // Each warp computes one WMMA_M x WMMA_N tile of C
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpN = blockIdx.y;

    if (warpM >= M / WMMA_M || warpN >= N / WMMA_N) return;

    // Declare WMMA fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop over K dimension in tiles
    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        // Bounds check
        if (aRow + WMMA_M <= M && aCol + WMMA_K <= K &&
            bRow + WMMA_K <= K && bCol + WMMA_N <= N) {

            // Load tiles from global memory into fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);

            // Perform the matrix multiply-accumulate: C += A * B
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Scale by alpha and add beta * C_old
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        // Load existing C values
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old_frag;
        wmma::load_matrix_sync(c_old_frag, C + cRow * N + cCol, N, wmma::mem_row_major);

        // Apply alpha and beta scaling
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * c_frag.x[i] + beta * c_old_frag.x[i];
        }

        // Store result
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

// ============================================================
// Helper: Initialize half-precision matrix
// ============================================================
void initHalfMatrix(__half *data, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = __float2half(val);
    }
}

void initFloatMatrix(float *data, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; i++) {
        data[i] = val;
    }
}

// ============================================================
// Host main
// ============================================================
int main(void) {
    printf("[WMMA Tensor Core GEMM] - Starting...\n");
    printf("Matrix: C(%d,%d) = alpha * A(%d,%d) * B(%d,%d) + beta * C\n",
           M_GLOBAL, N_GLOBAL, M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL);
    printf("WMMA tile: %dx%dx%d, Data: A,B=half, C=float\n", WMMA_M, WMMA_N, WMMA_K);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t sizeA = M_GLOBAL * K_GLOBAL * sizeof(__half);
    size_t sizeB = K_GLOBAL * N_GLOBAL * sizeof(__half);
    size_t sizeC = M_GLOBAL * N_GLOBAL * sizeof(float);

    // Allocate host memory
    __half *h_A = (__half *)malloc(sizeA);
    __half *h_B = (__half *)malloc(sizeB);
    float  *h_C = (float *)malloc(sizeC);

    initHalfMatrix(h_A, M_GLOBAL, K_GLOBAL, 1.0f);
    initHalfMatrix(h_B, K_GLOBAL, N_GLOBAL, 0.01f);
    initFloatMatrix(h_C, M_GLOBAL, N_GLOBAL, 0.0f);

    // Allocate device memory
    __half *d_A, *d_B;
    float  *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, sizeA));
    CHECK_CUDA(cudaMalloc(&d_B, sizeB));
    CHECK_CUDA(cudaMalloc(&d_C, sizeC));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice));

    // Launch WMMA kernel
    // Each warp processes one WMMA_M x WMMA_N tile
    // Grid: (M/WMMA_M warps in X, N/WMMA_N in Y)
    int warpsPerBlock = 4;  // 4 warps = 128 threads
    dim3 block(warpsPerBlock * 32, 1);
    dim3 grid((M_GLOBAL / WMMA_M + warpsPerBlock - 1) / warpsPerBlock, N_GLOBAL / WMMA_N);

    printf("Launching WMMA kernel: grid(%d,%d), block(%d,%d)\n",
           grid.x, grid.y, block.x, block.y);

    wmmaGemmKernel<<<grid, block>>>(d_A, d_B, d_C,
                                     M_GLOBAL, N_GLOBAL, K_GLOBAL,
                                     alpha, beta);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify: C = A * B where A=1.0, B=0.01, K=64
    // Each element of C should be K * 1.0 * 0.01 = 0.64
    float expected = K_GLOBAL * 1.0f * 0.01f;
    bool passed = true;
    for (int i = 0; i < M_GLOBAL * N_GLOBAL; i++) {
        float rel_err = fabs(h_C[i] - expected) / fabs(expected);
        if (rel_err > 1e-2) {  // Half precision tolerance
            printf("Mismatch at [%d]: got %.4f, expected %.4f, err=%.4e\n",
                   i, h_C[i], expected, rel_err);
            passed = false;
            if (i > 5) { printf("...(truncated)\n"); break; }
        }
    }
    printf("Result: %s\n", passed ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
