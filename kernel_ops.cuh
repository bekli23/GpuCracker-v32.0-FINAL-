#pragma once
#include <cuda_runtime.h>

void selectGpuDevice(int deviceId);
void initGpuRng(void** d_state, int totalThreads, int blocks, int threads);

// Modificat: primeste entropyBytes
void launchGpuGeneration(void* d_state, unsigned char* d_output, unsigned long long seed, int totalThreads, int points, int blocks, int threads, bool useSequential, int entropyBytes);