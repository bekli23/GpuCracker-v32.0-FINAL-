#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Includem runner-ul care conține logica CPU
#include "runner.h"

// =============================================================
// MAIN ENTRY POINT (CPU ONLY)
// =============================================================
int main(int argc, char* argv[]) {
    // 1. Parsare Argumente
    ProgramConfig cfg = parseArgs(argc, argv);

    // 2. Afișare Help
    if (cfg.help) {
        printHelp();
        return 0;
    }

    // 3. Inițializare Random
    srand((unsigned int)time(NULL));

    // 4. Pornire Runner
    // Acum compilatorul C++ vede clasa Runner corect
    Runner runner(cfg);
    runner.start();

    return 0;
}