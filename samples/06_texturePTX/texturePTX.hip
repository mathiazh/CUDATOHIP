#include "hip/hip_runtime.h"
/* texturePTX.cu
 *
 * Image processing pipeline using NVIDIA-only CUDA features:
 *   1. Texture objects with hardware bilinear interpolation
 *   2. Surface objects for read-write texture access
 *   3. Inline PTX dp4a instruction (integer dot-product)
 *   4. __nanosleep() for throttling (NVIDIA-specific intrinsic)
 *   5. hipTextureDesc with non-default addressing/filtering modes
 *   6. hipResourceDesc with hipArray (CUDA-specific memory layout)
 *
 * These features have NO direct HIP/ROCm equivalent.
 * Score F target: HIPIFY + AI cannot produce usable output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>

#define CHECK_CUDA(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================
// Kernel 1: Texture-based bilinear interpolation
// Uses CUDA texture objects with hardware filtering
// HIP texture support is LIMITED and doesn't support all modes
// ============================================================
__global__ void textureInterpolateKernel(
    hipTextureObject_t texObj,
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
    hipSurfaceObject_t surfObj,
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
    // REWRITTEN: NVIDIA PTX dp4a has NO HIP equivalent.
    // Manual implementation of 4x8-bit dot product.
    // This is SLOWER than hardware dp4a but functionally equivalent.
    signed char *a_bytes = (signed char *)&a;
    signed char *b_bytes = (signed char *)&b;
    int result = c;
    for (int i = 0; i < 4; i++) {
        result += (int)a_bytes[i] * (int)b_bytes[i];
    }
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
    // __nanosleep - NO HIP EQUIVALENT. Removed. AMD has no GPU-side sleep intrinsic.

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
    hipChannelFormatDesc channelDesc = hipCreateChannelDesc<float>();
    hipArray_t cuArray;
    CHECK_CUDA(hipMallocArray(&cuArray, &channelDesc, WIDTH, HEIGHT));

    // Initialize with test data
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i / N;

    CHECK_CUDA(hipMemcpy2DToArray(cuArray, 0, 0, h_data, WIDTH * sizeof(float),
                                    WIDTH * sizeof(float), HEIGHT, hipMemcpyHostToDevice));

    // Create texture descriptor with specific filtering
    hipTextureDesc texDesc = {};
    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.filterMode     = hipFilterModeLinear;  // Hardware bilinear interpolation
    texDesc.readMode       = hipReadModeElementType;
    texDesc.normalizedCoords = true;

    // Create resource descriptor
    hipResourceDesc resDesc = {};
    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // Create texture object
    hipTextureObject_t texObj = 0;
    CHECK_CUDA(hipCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    printf("Texture object created with bilinear filtering.\n");

    // ========== Surface Object Setup (CUDA-specific) ==========
    printf("Setting up CUDA surface object...\n");
    hipSurfaceObject_t surfObj = 0;
    CHECK_CUDA(hipCreateSurfaceObject(&surfObj, &resDesc));
    printf("Surface object created.\n");

    // ========== Allocate output and integer arrays ==========
    float *d_output;
    int *d_A, *d_B, *d_C;
    CHECK_CUDA(hipMalloc(&d_output, N * sizeof(float)));
    CHECK_CUDA(hipMalloc(&d_A, N * sizeof(int)));
    CHECK_CUDA(hipMalloc(&d_B, N * sizeof(int)));
    CHECK_CUDA(hipMalloc(&d_C, N * sizeof(int)));

    // Initialize integer arrays for dp4a
    int *h_A = (int *)malloc(N * sizeof(int));
    int *h_B = (int *)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_A[i] = 0x01020304;  // packed 4x8-bit: [1, 2, 3, 4]
        h_B[i] = 0x01010101;  // packed 4x8-bit: [1, 1, 1, 1]
    }
    CHECK_CUDA(hipMemcpy(d_A, h_A, N * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(d_B, h_B, N * sizeof(int), hipMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // ========== Launch Kernels ==========
    printf("\n1. Texture interpolation kernel...\n");
    textureInterpolateKernel<<<grid, block>>>(texObj, d_output, WIDTH, HEIGHT, 0.5f, 0.5f);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());
    printf("   Done.\n");

    printf("2. Surface read-write kernel...\n");
    surfaceProcessKernel<<<grid, block>>>(surfObj, WIDTH, HEIGHT);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());
    printf("   Done.\n");

    printf("3. PTX dp4a kernel...\n");
    dp4aKernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());
    printf("   Done.\n");

    printf("4. Nanosleep kernel...\n");
    float *d_sleep_data;
    CHECK_CUDA(hipMalloc(&d_sleep_data, N * sizeof(float)));
    CHECK_CUDA(hipMemcpy(d_sleep_data, h_data, N * sizeof(float), hipMemcpyHostToDevice));
    nanosleepKernel<<<(N + 255) / 256, 256>>>(d_sleep_data, N, 1000);
    CHECK_CUDA(hipGetLastError());
    CHECK_CUDA(hipDeviceSynchronize());
    printf("   Done.\n");

    // Verify dp4a result: dot([1,2,3,4], [1,1,1,1]) = 1+2+3+4 = 10
    int *h_C = (int *)malloc(N * sizeof(int));
    CHECK_CUDA(hipMemcpy(h_C, d_C, N * sizeof(int), hipMemcpyDeviceToHost));

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
    CHECK_CUDA(hipDestroyTextureObject(texObj));
    CHECK_CUDA(hipDestroySurfaceObject(surfObj));
    CHECK_CUDA(hipFreeArray(cuArray));
    CHECK_CUDA(hipFree(d_output));
    CHECK_CUDA(hipFree(d_A));
    CHECK_CUDA(hipFree(d_B));
    CHECK_CUDA(hipFree(d_C));
    CHECK_CUDA(hipFree(d_sleep_data));
    free(h_data);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done.\n");
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
