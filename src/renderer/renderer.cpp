#include "renderer.h"
#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include <string.h>

#include "gpu.h"

Renderer::Renderer(ParticleSystem *particleSystem, uint32_t fps) {
    this->time_step = 1000. / 60.;
    this->next_time_step = SDL_GetTicks();
    this->particleSystem = particleSystem;

    if (SDL_Init(SDL_INIT_EVERYTHING) < 0) {
        std::cerr << "could not initialize sdl2: " << SDL_GetError()
                  << std::endl;
        exit(1);
    }

    window = SDL_CreateWindow("main", SDL_WINDOWPOS_UNDEFINED,
                              SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH,
                              SCREEN_HEIGHT, SDL_WINDOW_SHOWN);

    default_screen =
        SDL_CreateRGBSurface(0, SCREEN_WIDTH, SCREEN_HEIGHT, 32, 0x00FF0000,
                             0x0000FF00, 0x000000FF, 0xFF000000);

    if (default_screen == NULL) {
        SDL_Log("SDL_CreateRGBSurface() failed: %s", SDL_GetError());
        exit(1);
    }

    sdlRenderer = SDL_CreateRenderer(
        window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    sdlTexture = SDL_CreateTexture(sdlRenderer, SDL_PIXELFORMAT_ARGB8888,
                                   SDL_TEXTUREACCESS_STREAMING |
                                       SDL_TEXTUREACCESS_TARGET,
                                   SCREEN_WIDTH, SCREEN_HEIGHT);

    if (sdlTexture == NULL) {
        SDL_Log("SDL_Error failed: %s", SDL_GetError());
        exit(1);
    }

    gpu_mem = gpuAlloc();
    if (gpu_mem == NULL) {
        std::cerr << "failed to alloc gpu memory" << std::endl;
    }
}

Renderer::~Renderer() {
    SDL_DestroyTexture(sdlTexture);
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(window);

    SDL_Quit();
}

bool Renderer::draw() {

    SDL_Event e;
    if (SDL_PollEvent(&e)) {
        if (e.type == SDL_QUIT) {
            return false;
        }
    }

    uint32_t now = SDL_GetTicks();
    if (next_time_step <= now) {

        SDL_LockSurface(default_screen);
        particleSystem->render(gpu_mem);
        if (gpuBlit(gpu_mem, default_screen->pixels) != 0) {
            std::cerr << "cuda error" << std::endl;
        };
        SDL_UnlockSurface(default_screen);

        SDL_UpdateTexture(sdlTexture, NULL, default_screen->pixels,
                          default_screen->pitch);
        SDL_RenderClear(sdlRenderer);
        SDL_RenderCopy(sdlRenderer, sdlTexture, NULL, NULL);
        SDL_RenderPresent(sdlRenderer);

        next_time_step += time_step;
    } else {
        SDL_Delay(next_time_step - now);
    }

    return true;
}
