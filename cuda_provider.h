#pragma once
#include "gpu_interface.h"
#include <cuda_runtime.h>
#include <string>

class CudaProvider : public IGpuProvider {
private:
    int deviceId;
    int blocks;
    int threads;
    int points;
    int totalThreads;
    int batchSize;
    void* d_rngState = nullptr;
    unsigned char* d_output = nullptr;
    std::string deviceName;
    bool sequentialMode; 

public:
    // Constructor (DOAR Declaratie - fara cod)
    CudaProvider(int devId, int blk, int th, int pts, bool seq);
    
    // Destructor
    ~CudaProvider();

    // Metode (DOAR Declaratii)
    void init() override;
    
    // Semnatura trebuie sa fie identica cu cea din gpu_interface.h
    void generate(unsigned char* hostBuffer, unsigned long long seed, int entropyBytes) override;
    
    int getBatchSize() const override;
    std::string getName() const override;
    std::string getConfig() const override;
};