#pragma once
#include <string>

class IGpuProvider {
public:
    virtual ~IGpuProvider() {}
    virtual void init() = 0;
    // Modificat: primeste entropyBytes
    virtual void generate(unsigned char* hostBuffer, unsigned long long seed, int entropyBytes) = 0;
    virtual int getBatchSize() const = 0;
    virtual std::string getName() const = 0;
    virtual std::string getConfig() const = 0;
};