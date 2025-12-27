#include "cuda_provider.h"
#include "kernel_ops.cuh" 
#include <cuda_runtime.h>
#include <iostream>

// --- AICI ESTE CODUL EFECTIV (IMPLEMENTAREA) ---

CudaProvider::CudaProvider(int devId, int blk, int th, int pts, bool seq) 
    : deviceId(devId), blocks(blk), threads(th), points(pts), sequentialMode(seq) 
{
    totalThreads = blocks * threads;
    batchSize = totalThreads * points;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    deviceName = "[CUDA] " + std::string(prop.name);
}

CudaProvider::~CudaProvider() {
    if (d_rngState) cudaFree(d_rngState);
    if (d_output) cudaFree(d_output);
}

void CudaProvider::init() {
    selectGpuDevice(deviceId);
    cudaMalloc(&d_output, (size_t)batchSize * 32);
    // Aceasta functie vine din kernel_ops.cu
    initGpuRng(&d_rngState, totalThreads, blocks, threads);
}

void CudaProvider::generate(unsigned char* hostBuffer, unsigned long long seed, int entropyBytes) {
    // Aceasta functie vine din kernel_ops.cu
    launchGpuGeneration(d_rngState, d_output, seed, totalThreads, points, blocks, threads, sequentialMode, entropyBytes);
    cudaMemcpy(hostBuffer, d_output, (size_t)batchSize * 32, cudaMemcpyDeviceToHost);
}

int CudaProvider::getBatchSize() const { 
    return batchSize; 
}

std::string CudaProvider::getName() const { 
    return deviceName; 
}

std::string CudaProvider::getConfig() const { 
    return "B:" + std::to_string(blocks) + " T:" + std::to_string(threads) + " P:" + std::to_string(points); 
}