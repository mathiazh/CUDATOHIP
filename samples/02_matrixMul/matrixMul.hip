#include "hip/hip_runtime.h"
/* Based on NVIDIA cuda-samples/Samples/0_Introduction/matrixMul
 * Modified to remove helper_cuda.h and helper_functions.h dependencies
 * for standalone compilation.
 */

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

// Error checking macro (replaces checkCudaErrors from helper_cuda.h)
#define CHECK_CUDA(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    hipGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int aBegin = wA * BLOCK_SIZE * by;
    int aEnd   = aBegin + wA - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * bx;
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}

void ConstantInit(float *data, int size, float val)
{
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

int MatrixMultiply(int block_size, const dim3 &dimsA, const dim3 &dimsB)
{
    unsigned int size_A     = dimsA.x * dimsA.y;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A;
    CHECK_CUDA(hipHostMalloc(&h_A, mem_size_A));

    unsigned int size_B     = dimsB.x * dimsB.y;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B;
    CHECK_CUDA(hipHostMalloc(&h_B, mem_size_B));

    hipStream_t stream;

    const float valB = 0.01f;
    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, valB);

    float *d_A, *d_B, *d_C;

    dim3 dimsC(dimsB.x, dimsA.y, 1);
    unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
    float *h_C;
    CHECK_CUDA(hipHostMalloc(&h_C, mem_size_C));

    CHECK_CUDA(hipMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    CHECK_CUDA(hipMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    CHECK_CUDA(hipMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

    hipEvent_t start, stop;
    CHECK_CUDA(hipEventCreate(&start));
    CHECK_CUDA(hipEventCreate(&stop));
    CHECK_CUDA(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    CHECK_CUDA(hipMemcpyAsync(d_A, h_A, mem_size_A, hipMemcpyHostToDevice, stream));
    CHECK_CUDA(hipMemcpyAsync(d_B, h_B, mem_size_B, hipMemcpyHostToDevice, stream));

    dim3 threads(block_size, block_size);
    dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

    printf("Computing result using CUDA Kernel...\n");

    if (block_size == 16) {
        MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    } else {
        MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
    }

    CHECK_CUDA(hipStreamSynchronize(stream));

    CHECK_CUDA(hipEventRecord(start, stream));

    int nIter = 300;
    for (int j = 0; j < nIter; j++) {
        if (block_size == 16) {
            MatrixMulCUDA<16><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        } else {
            MatrixMulCUDA<32><<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);
        }
    }

    CHECK_CUDA(hipEventRecord(stop, stream));
    CHECK_CUDA(hipEventSynchronize(stop));

    float msecTotal = 0.0f;
    CHECK_CUDA(hipEventElapsedTime(&msecTotal, start, stop));

    float msecPerMatrixMul = msecTotal / nIter;
    double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops, WorkgroupSize= %u threads/block\n",
           gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

    CHECK_CUDA(hipMemcpyAsync(h_C, d_C, mem_size_C, hipMemcpyDeviceToHost, stream));
    CHECK_CUDA(hipStreamSynchronize(stream));

    printf("Checking computed result for correctness: ");
    bool correct = true;
    double eps = 1.e-6;

    for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
        double abs_err = fabs(h_C[i] - (dimsA.x * valB));
        double dot_length = dimsA.x;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                   i, h_C[i], dimsA.x * valB, eps);
            correct = false;
        }
    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    CHECK_CUDA(hipHostFree(h_A));
    CHECK_CUDA(hipHostFree(h_B));
    CHECK_CUDA(hipHostFree(h_C));
    CHECK_CUDA(hipFree(d_A));
    CHECK_CUDA(hipFree(d_B));
    CHECK_CUDA(hipFree(d_C));
    CHECK_CUDA(hipEventDestroy(start));
    CHECK_CUDA(hipEventDestroy(stop));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    printf("[Matrix Multiply Using CUDA] - Starting...\n");

    int block_size = 32;
    dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

    printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);

    // hipProfilerStart() removed - deprecated in ROCm, use roctracer/rocTX instead
    int result = MatrixMultiply(block_size, dimsA, dimsB);
    // hipProfilerStop() removed - deprecated in ROCm, use roctracer/rocTX instead

    return result;
}
