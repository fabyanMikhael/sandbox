#include "particles.h"

ParticleSystem::ParticleSystem(int width, int height) {
    this->width = width;
    this->height = height;

    ParticleType *particles = new ParticleType[width * height];

    for (int i = 0; i < width * height; i++) {
        bool is_sand = rand() % 2 == 0;

        particles[i] = is_sand ? ParticleType::Sand : ParticleType::Empty;
    }

    cudaMalloc(&this->particles, width * height * sizeof(ParticleType));
    cudaMalloc(&this->nextParticles, width * height * sizeof(ParticleType));
    cudaMemcpy(this->particles, particles,
               width * height * sizeof(ParticleType), cudaMemcpyHostToDevice);
    cudaMemcpy(this->nextParticles, particles,
               width * height * sizeof(ParticleType), cudaMemcpyHostToDevice);

    delete[] particles;
}

ParticleSystem::~ParticleSystem() { cudaFree(this->particles); }

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

    if ((y / cellSize) >= height || (x / cellSize) >= width) {
        return;
    }

    ParticleType particle = particles[index];

    if (particle == ParticleType::Sand) {
        buf[pos] = color(255, 255, 0);
    } else {
        buf[pos] = color(0, 0, 0);
    }
}

__global__ void updateParticles(ParticleType *particles,
                                ParticleType *nextParticles, int width,
                                int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const auto getIndex = [&](int x, int y) { return y * width + x; };
    const auto getParticle = [&](int x, int y) {
        if (x < 0 || x >= width || y < 0 || y >= height) {
            return ParticleType::Empty;
        }

        return particles[getIndex(x, y)];
    };

    int index = getIndex(x, y);
    ParticleType particle = particles[index];

    ParticleType bottomParticle = getParticle(x, y + 1);
    ParticleType topParticle = getParticle(x, y - 1);
    ParticleType leftParticle = getParticle(x - 1, y);
    ParticleType rightParticle = getParticle(x + 1, y);
    ParticleType bottomLeftParticle = getParticle(x - 1, y + 1);
    ParticleType bottomRightParticle = getParticle(x + 1, y + 1);
    ParticleType topLeftParticle = getParticle(x - 1, y - 1);
    ParticleType topRightParticle = getParticle(x + 1, y - 1);

    // game of life rules:

    const auto getSurroundingCount = [&](int x, int y) {
        int count = 0;

        if (getParticle(x, y + 1) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x, y - 1) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x - 1, y) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x + 1, y) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x - 1, y + 1) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x + 1, y + 1) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x - 1, y - 1) == ParticleType::Sand) {
            count++;
        }

        if (getParticle(x + 1, y - 1) == ParticleType::Sand) {
            count++;
        }

        return count;
    };

    // 1. Any live cell with fewer than two live neighbours dies, as if by
    // underpopulation.
    // 2. Any live cell with two or three live neighbours lives on to the next
    // generation.
    // 3. Any live cell with more than three live neighbours dies, as if by
    // overpopulation.
    // 4. Any dead cell with exactly three live neighbours becomes a live cell,
    // as if by reproduction.

    int count = getSurroundingCount(x, y);
    // 1:
    if (count < 2 || count > 3) {
        nextParticles[index] = ParticleType::Empty;
    } else if (count == 3) {
        nextParticles[index] = ParticleType::Sand;
    } else {
        nextParticles[index] = particle;
    }
}

void ParticleSystem::update() {
    dim3 dimBlock(32, 32);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    updateParticles<<<dimGrid, dimBlock>>>(this->particles, this->nextParticles,
                                           width, height);
    cudaDeviceSynchronize();

    std::swap(this->particles, this->nextParticles);
}

void ParticleSystem::render(uint32_t *buf) {
    const dim3 blocksPerGrid(H_TILES, V_TILES);
    const dim3 threadsPerBlock(TILE_WIDTH, TILE_HEIGHT);

    renderParticles<<<blocksPerGrid, threadsPerBlock>>>(width, height, 2,
                                                        this->particles, buf);

    cudaDeviceSynchronize();
}