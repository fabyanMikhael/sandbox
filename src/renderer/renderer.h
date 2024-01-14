#include "../particles/particles.h"
#include <SDL2/SDL.h>

class Renderer {
  private:
    SDL_Window *window;
    SDL_Renderer *sdlRenderer;
    SDL_Texture *sdlTexture;
    SDL_Surface *default_screen;
    uint32_t next_time_step;
    uint32_t time_step;
    uint32_t *gpu_mem;
    ParticleSystem *particleSystem;

  public:
    Renderer(ParticleSystem *particleSystem, uint32_t fps = 1);
    ~Renderer();
    bool draw();
};