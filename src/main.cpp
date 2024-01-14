#include "particles/particles.h"

#include "renderer/renderer.h"

int main() {
    ParticleSystem particleSystem(100, 100);
    Renderer renderer(&particleSystem);

    while (renderer.draw()) {
        particleSystem.update();
    };

    return 0;
}