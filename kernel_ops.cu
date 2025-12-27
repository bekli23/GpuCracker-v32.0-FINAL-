#include "kernel_ops.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>

void selectGpuDevice(int deviceId) {
    cudaSetDevice(deviceId);
}

__global__ void init_rng(curandState *state, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__device__ unsigned int swap_endian_gpu(unsigned int val) {
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
}

__global__ void generate_entropy_kernel(curandState *state, unsigned char *output, unsigned long long baseSeed, int points, bool useSequential, int entropyBytes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    curandState localState;
    if (!useSequential) {
        localState = state[idx];
    }

    // Determinam unde trebuie sa scriem contorul in functie de lungimea frazei
    // Daca sunt 12 cuvinte (16 bytes), scriem in ultimii 8 bytes ai blocului de 16.
    // Daca sunt 24 cuvinte (32 bytes), scriem in ultimii 8 bytes ai blocului de 32.
    int counterOffsetWords = (entropyBytes / 4) - 2; 
    if (counterOffsetWords < 0) counterOffsetWords = 0;

    for (int p = 0; p < points; ++p) {
        int outputIdx = (idx * points) + p;
        int offset = outputIdx * 32; // Stride fix de 32

        unsigned int r[8] = {0}; // Initializam tot cu 0

        if (useSequential) {
            // --- MOD SECVENTIAL ODOMETRU ---
            unsigned long long uniqueId = baseSeed + outputIdx;
            
            // Scriem ID-ul exact la finalul entropiei utile
            // Astfel, ultimele cuvinte se vor schimba primele
            r[counterOffsetWords]     = swap_endian_gpu((unsigned int)(uniqueId >> 32));
            r[counterOffsetWords + 1] = swap_endian_gpu((unsigned int)(uniqueId & 0xFFFFFFFF));
            
        } else {
            // --- MOD RANDOM ---
            r[0] = curand(&localState); r[1] = curand(&localState);
            r[2] = curand(&localState); r[3] = curand(&localState);
            r[4] = curand(&localState); r[5] = curand(&localState);
            r[6] = curand(&localState); r[7] = curand(&localState);
        }

        // Scriere in memorie
        unsigned int* outPtr = reinterpret_cast<unsigned int*>(output + offset);
        #pragma unroll
        for(int k=0; k<8; k++) outPtr[k] = r[k];
    }

    if (!useSequential) {
        state[idx] = localState;
    }
}

void launchGpuGeneration(void* d_state, unsigned char* d_output, unsigned long long seed, int totalThreads, int points, int blocks, int threads, bool useSequential, int entropyBytes) {
    generate_entropy_kernel<<<blocks, threads>>>((curandState*)d_state, d_output, seed, points, useSequential, entropyBytes);
    cudaDeviceSynchronize();
}

void initGpuRng(void** d_state, int totalThreads, int blocks, int threads) {
    cudaMalloc(d_state, totalThreads * sizeof(curandState));
    init_rng<<<blocks, threads>>>((curandState*)*d_state, time(NULL));
    cudaDeviceSynchronize();
}