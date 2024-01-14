#include "particles/particles.h"

#include "renderer/renderer.h"

int main() {
    ParticleSystem particleSystem(32 * 5, 32 * 5);
    Renderer renderer(&particleSystem);

    while (renderer.draw())
        ;

    return 0;
}