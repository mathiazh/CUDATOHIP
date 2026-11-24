/* wmmaGemm.hip - COMPLETE REWRITE from CUDA WMMA to AMD rocWMMA
 * HIPIFY translated ZERO WMMA code. 16 manual changes required.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#define CHECK_HIP(call) do {     hipError_t err = call;     if (err != hipSuccess) {         fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err));         exit(EXIT_FAILURE);     } } while(0)

const int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
const int M_GLOBAL = 64, N_GLOBAL = 64, K_GLOBAL = 64;

__global__ void wmmaGemmKernel(const _Float16 *A, const _Float16 *B, float *C,
    int M, int N, int K, float alpha, float beta) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y;
    if (warpM >= M / WMMA_M || warpN >= N / WMMA_N) return;

    rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major> a_frag;
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, _Float16, rocwmma::row_major> b_frag;
    rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    rocwmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        int aRow = warpM * WMMA_M, aCol = k, bRow = k, bCol = warpN * WMMA_N;
        if (aRow + WMMA_M <= M && aCol + WMMA_K <= K && bRow + WMMA_K <= K && bCol + WMMA_N <= N) {
            rocwmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            rocwmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            rocwmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M, cCol = warpN * WMMA_N;
    if (cRow + WMMA_M <= M && cCol + WMMA_N <= N) {
        rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_old;
        rocwmma::load_matrix_sync(c_old, C + cRow * N + cCol, N, rocwmma::mem_row_major);
        
        // rocwmma fragment access: cast mStorage to float* for element-wise operations
        float* c_ptr = reinterpret_cast<float*>(&c_frag);
        float* co_ptr = reinterpret_cast<float*>(&c_old);
        for (int i = 0; i < c_frag.num_elements; i++) c_ptr[i] = alpha * c_ptr[i] + beta * co_ptr[i];
        rocwmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, rocwmma::mem_row_major);
    }
}

void initHalf(_Float16 *d, int n, float v) { for (int i=0;i<n;i++) d[i]=(_Float16)v; }
void initFloat(float *d, int n, float v) { for (int i=0;i<n;i++) d[i]=v; }

int main(void) {
    printf("[rocWMMA GEMM (converted from CUDA WMMA)] - Starting...\n");
    printf("Matrix: C(%d,%d) = A(%d,%d) * B(%d,%d), tile=%dx%dx%d\n",
           M_GLOBAL, N_GLOBAL, M_GLOBAL, K_GLOBAL, K_GLOBAL, N_GLOBAL, WMMA_M, WMMA_N, WMMA_K);

    size_t sA=M_GLOBAL*K_GLOBAL*sizeof(_Float16), sB=K_GLOBAL*N_GLOBAL*sizeof(_Float16), sC=M_GLOBAL*N_GLOBAL*sizeof(float);
    _Float16 *hA=(_Float16*)malloc(sA), *hB=(_Float16*)malloc(sB); float *hC=(float*)malloc(sC);
    initHalf(hA, M_GLOBAL*K_GLOBAL, 1.0f); initHalf(hB, K_GLOBAL*N_GLOBAL, 0.01f); initFloat(hC, M_GLOBAL*N_GLOBAL, 0.0f);

    _Float16 *dA, *dB; float *dC;
    CHECK_HIP(hipMalloc(&dA, sA)); CHECK_HIP(hipMalloc(&dB, sB)); CHECK_HIP(hipMalloc(&dC, sC));
    CHECK_HIP(hipMemcpy(dA, hA, sA, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dB, hB, sB, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(dC, hC, sC, hipMemcpyHostToDevice));

    dim3 block(4*64,1), grid((M_GLOBAL/WMMA_M+3)/4, N_GLOBAL/WMMA_N);
    printf("Launching: grid(%d,%d), block(%d,%d)\n", grid.x, grid.y, block.x, block.y);
    wmmaGemmKernel<<<grid, block>>>(dA, dB, dC, M_GLOBAL, N_GLOBAL, K_GLOBAL, 1.0f, 0.0f);
    CHECK_HIP(hipGetLastError()); CHECK_HIP(hipDeviceSynchronize());
    CHECK_HIP(hipMemcpy(hC, dC, sC, hipMemcpyDeviceToHost));

    float expected = K_GLOBAL * 1.0f * 0.01f; bool ok = true;
    for (int i=0; i<M_GLOBAL*N_GLOBAL; i++) {
        if (fabs(hC[i]-expected)/fabs(expected) > 1e-2) {
            printf("Mismatch[%d]: %.4f vs %.4f\n", i, hC[i], expected); ok=false; if(i>5){printf("...\n");break;}
        }
    }
    printf("Result: %s\n", ok ? "PASS" : "FAIL");
    CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
    free(hA); free(hB); free(hC);
    printf("Done.\n"); return ok ? 0 : 1;
}
