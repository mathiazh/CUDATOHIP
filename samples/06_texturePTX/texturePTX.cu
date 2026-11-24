/* texturePTX.cu
 *
 * Image processing pipeline using NVIDIA-only CUDA features:
 *   1. Texture objects with hardware bilinear interpolation
 *   2. Surface objects for read-write texture access
 *   3. Inline PTX dp4a instruction (integer dot-product)
 *   4. __nanosleep() for throttling (NVIDIA-specific intrinsic)
 *   5. cudaTextureDesc with non-default addressing/filtering modes
 *   6. cudaResourceDesc with cudaArray (CUDA-specific memory layout)
 *
 * These features have NO direct HIP/ROCm equivalent.
 * Score F target: HIPIFY + AI cannot produce usable output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Kernel 1: Texture-based bilinear interpolation
// Uses CUDA texture objects with hardware filtering
// HIP texture support is LIMITED and doesn't support all modes
// ============================================================
__global__ void textureInterpolateKernel(
    cudaTextureObject_t texObj,
    float *output,
    int width, int height,
    float scaleX, float scaleY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Hardware bilinear interpolation at fractional coordinates
        // This uses CUDA texture hardware - specific filtering modes
        float u = (x + 0.5f) * scaleX;
        float v = (y + 0.5f) * scaleY;

        // tex2D with texture object - CUDA-specific
        float val = tex2D<float>(texObj, u, v);
        output[y * width + x] = val;
    }
}

// ============================================================
// Kernel 2: Surface read-write operations
// CUDA surface objects allow read AND write to texture memory
// HIP does NOT support surface objects
// ============================================================
__global__ void surfaceProcessKernel(
    cudaSurfaceObject_t surfObj,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        // Read from surface
        float val;
        surf2Dread(&val, surfObj, x * sizeof(float), y);

        // Process (simple gamma correction)
        val = powf(val, 0.4545f);

        // Write back to same surface
        surf2Dwrite(val, surfObj, x * sizeof(float), y);
    }
}

// ============================================================
// Kernel 3: Inline PTX dp4a (4-element dot product of 8-bit integers)
// dp4a is an NVIDIA-specific instruction (Volta+)
// There is NO direct AMD equivalent instruction
// ============================================================
__device__ int ptx_dp4a(int a, int b, int c) {
    int result;
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(result) : "r"(a), "r"(b), "r"(c));
    return result;
}

__global__ void dp4aKernel(const int *A, const int *B, int *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Packed 4x8-bit dot product: dp4a(a, b, c) = c + dot(a_bytes, b_bytes)
    // This performs 4 multiply-adds on packed 8-bit integers in a single instruction
    C[idx] = ptx_dp4a(A[idx], B[idx], 0);
}

// ============================================================
// Kernel 4: __nanosleep - NVIDIA-specific sleep intrinsic
// Used for power management / throttling in real-time systems
// NO HIP equivalent exists
// ============================================================
__global__ void nanosleepKernel(float *data, int n, unsigned int ns) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx] *= 2.0f;

    // Sleep for specified nanoseconds (NVIDIA Volta+ only)
    __nanosleep(ns);

    data[idx] += 1.0f;
}

// ============================================================
// Host: Setup texture/surface objects with CUDA arrays
// ============================================================
int main(void) {
    printf("[Texture + Surface + PTX dp4a + nanosleep] - Starting...\n");
    printf("This project uses NVIDIA-only features with NO HIP equivalent.\n\n");

    const int WIDTH = 64, HEIGHT = 64;
    const int N = WIDTH * HEIGHT;

    // ========== Texture Object Setup (CUDA-specific) ==========
    printf("Setting up CUDA texture object with bilinear interpolation...\n");

    // Create CUDA array (special memory layout for texture hardware)
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray_t cuArray;
    CHECK_CUDA(cudaMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));

    // Initialize with test data
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i / N;

    CHECK_CUDA(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, WIDTH * sizeof(float),
                                    WIDTH * sizeof(float), HEIGHT, cudaMemcpyHostToDevice));

    // Create texture descriptor with specific filtering
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode     = cudaFilterModeLinear;  // Hardware bilinear interpolation
    texDesc.readMode       = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    // Create resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Create texture object
    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    printf("Texture object created with bilinear filtering.\n");

    // ========== Surface Object Setup (CUDA-specific) ==========
    printf("Setting up CUDA surface object...\n");
    cudaSurfaceObject_t surfObj = 0;
    CHECK_CUDA(cudaCreateSurfaceObject(&surfObj, &resDesc));
    printf("Surface object created.\n");

    // ========== Allocate output and integer arrays ==========
    float *d_output;
    int *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_A, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C, N * sizeof(int)));

    // Initialize integer arrays for dp4a
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_A[i] = 0x01020304;  // packed 4x8-bit: [1, 2, 3, 4]
        h_B[i] = 0x01010101;  // packed 4x8-bit: [1, 1, 1, 1]
    }
    CHECK_CUDA(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // ========== Launch Kernels ==========
    printf("\n1. Texture interpolation kernel...\n");
    textureInterpolateKernel<<<grid, block>>>(texObj, d_output, WIDTH, HEIGHT, 0.5f, 0.5f);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Done.\n");

    printf("2. Surface read-write kernel...\n");
    surfaceProcessKernel<<<grid, block>>>(surfObj, WIDTH, HEIGHT);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Done.\n");

    printf("3. PTX dp4a kernel...\n");
    dp4aKernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Done.\n");

    printf("4. Nanosleep kernel...\n");
    float *d_sleep_data;
    CHECK_CUDA(cudaMalloc(&d_sleep_data, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_sleep_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));
    nanosleepKernel<<<(N + 255) / 256, 256>>>(d_sleep_data, N, 1000);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   Done.\n");

    // Verify dp4a result: dot([1,2,3,4], [1,1,1,1]) = 1+2+3+4 = 10
    int *h_C = (int *)malloc(N * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != 10) {
            printf("dp4a mismatch at %d: got %d, expected 10\n", i, h_C[i]);
            passed = false;
            break;
        }
    }
    printf("\ndp4a result: %s\n", passed ? "PASS" : "FAIL");

    // Cleanup
    CHECK_CUDA(cudaDestroyTextureObject(texObj));
    CHECK_CUDA(cudaDestroySurfaceObject(surfObj));
    CHECK_CUDA(cudaFreeArray(cuArray));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_sleep_data));
    free(h_data);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
