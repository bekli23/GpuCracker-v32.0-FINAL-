#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Includem runner-ul
#include "runner.h"

// =============================================================
// MAIN ENTRY POINT
// =============================================================
int main(int argc, char* argv[]) {
    // 1. Parsare Argumente
    ProgramConfig cfg = parseArgs(argc, argv);

    // 2. Afisare Help
    if (cfg.help) {
        printHelp();
        return 0;
    }

    // 3. Initializare Random
    srand((unsigned int)time(NULL));

    // 4. Pornire Runner
    Runner runner(cfg);
    runner.start();

    return 0;
}