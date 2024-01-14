#pragma once

#include "../const.h"
#include <cstdint>

enum ParticleType { Empty, Sand };

class ParticleSystem {
  public:
    ParticleSystem(int width, int height);
    ~ParticleSystem();
    void update();
    void render(uint32_t *buf);

  private:
    int width;
    int height;
    ParticleType *particles;
};