#include "particles.h"

ParticleSystem::ParticleSystem(int width, int height) {
    this->width = width;
    this->height = height;

    ParticleType *particles = new ParticleType[width * height];

    for (int i = 0; i < width * height; i++) {
        particles[i] = ParticleType::Empty;
    }

    // a cross
    particles[5 * width + 5] = ParticleType::Sand;
    particles[5 * width + 6] = ParticleType::Sand;
    particles[5 * width + 7] = ParticleType::Sand;
    particles[5 * width + 8] = ParticleType::Sand;
    particles[5 * width + 9] = ParticleType::Sand;
    particles[5 * width + 10] = ParticleType::Sand;
    particles[5 * width + 11] = ParticleType::Sand;

    particles[6 * width + 8] = ParticleType::Sand;
    particles[7 * width + 8] = ParticleType::Sand;
    particles[8 * width + 8] = ParticleType::Sand;
    particles[9 * width + 8] = ParticleType::Sand;
    particles[10 * width + 8] = ParticleType::Sand;

    cudaMalloc(&this->particles, width * height * sizeof(ParticleType));
    cudaMemcpy(this->particles, particles,
               width * height * sizeof(ParticleType), cudaMemcpyHostToDevice);

    delete[] particles;
}

ParticleSystem::~ParticleSystem() { cudaFree(this->particles); }

__global__ void updateParticles(ParticleType *particles, int width,
                                int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int index = y * width + x;
    ParticleType particle = particles[index];

    if (particle == ParticleType::Sand) {
        particles[index] = ParticleType::Empty;
    } else {
        particles[index] = ParticleType::Sand;
    }
}

__host__ __device__ uint32_t static color(uint8_t r, uint8_t g, uint8_t b) {
    return (r << 16) | (g << 8) | b;
}

__global__ void renderParticles(int width, int height, int cellSize,
                                ParticleType *particles, uint32_t *buf) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int pos = SCREEN_WIDTH * y + x;

    // calculate the index of the pixel (using cellSize for multiple pixels to
    // same index)
    // eg. x = 0, y = 0, cellSize = 2, index = 0
    // eg. x = 1, y = 0, cellSize = 2, index = 0
    // eg. x = 0, y = 1, cellSize = 2, index = 0
    // eg. x = 1, y = 1, cellSize = 2, index = 0
    // eg. x = 2, y = 0, cellSize = 2, index = 1
    int index = (y / cellSize) * (width) + (x / cellSize);

    if (index >= width * height) {
        return;
    }

    ParticleType particle = particles[index];

    if (particle == ParticleType::Sand) {
        buf[pos] = color(255, 255, 0);
    } else {
        buf[pos] = color(0, 0, 0);
    }
}

void ParticleSystem::update() {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    updateParticles<<<dimGrid, dimBlock>>>(this->particles, width, height);
    cudaDeviceSynchronize();
}

void ParticleSystem::render(uint32_t *buf) {
    const dim3 blocksPerGrid(H_TILES, V_TILES);
    const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);

    renderParticles<<<blocksPerGrid, threadsPerBlock>>>(width, height, 10,
                                                        this->particles, buf);

    cudaDeviceSynchronize();
}