#include "gpu.h"
#include <cstdint>
#include <curand.h>
#include <curand_kernel.h>

uint32_t *gpuAlloc(void) {
    uint32_t *gpu_mem;

    cudaError_t err = cudaMalloc(&gpu_mem, SCREEN_SIZE * 4);
    if (err != cudaSuccess)
        return NULL;

    return gpu_mem;
};

void gpuFree(void *gpu_mem) { cudaFree(gpu_mem); }

int gpuBlit(void *src, void *dst) {
    cudaError_t err =
        cudaMemcpy(dst, src, SCREEN_SIZE * 4, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        return 1;
    return 0;
}
