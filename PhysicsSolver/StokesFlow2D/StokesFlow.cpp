#include "StokesFlow.h"

int main() {
    StokesFlow<2> stokes_flow(10, {32, 32}, {1.0, 1.0}, 1.0, 1.0, 0.01);

    stokes_flow.test_main();
    return 0;
}