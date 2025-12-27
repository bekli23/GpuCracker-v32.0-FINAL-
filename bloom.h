#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <cstring>

typedef unsigned int uint;

class BloomFilter {
private:
    std::vector<uint8_t> data;
    uint64_t bitSize = 0; 
    uint32_t numHashes = 0; 
    bool loaded = false;

    // Helper Endianness
    uint64_t read_be64(const uint8_t* ptr) const {
        uint64_t val = 0; for(int i=0; i<8; ++i) val = (val << 8) | ptr[i]; return val;
    }
    uint32_t read_be32(const uint8_t* ptr) const {
        uint32_t val = 0; for(int i=0; i<4; ++i) val = (val << 8) | ptr[i]; return val;
    }

    // --- MURMURHASH3 IDENTIC CU BUILD_BLOOM.CPP ---
    inline uint32_t rotl32(uint32_t x, int8_t r) const { return (x << r) | (x >> (32 - r)); }
    
    uint32_t MurmurHash3(const void* key, int len, uint32_t seed) const {
        const uint8_t* dataPtr = (const uint8_t*)key;
        const int nblocks = len / 4;
        uint32_t h1 = seed;
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;
        
        // LOGICA EXACTA DIN BUILDER (Pointer Arithmetic)
        const uint32_t* blocks = (const uint32_t*)(dataPtr + nblocks * 4);

        for (int i = -nblocks; i; i++) {
            uint32_t k1;
            // Folosim memcpy pentru a emula citirea directa uint32 (safe alignment)
            // dar pastrand offset-ul negativ specific implementarii tale
            memcpy(&k1, (const uint8_t*)blocks + (i * 4), 4);

            k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2;
            h1 ^= k1; h1 = rotl32(h1, 13); h1 = h1 * 5 + 0xe6546b64;
        }

        const uint8_t* tail = (const uint8_t*)(dataPtr + nblocks * 4);
        uint32_t k1 = 0;
        switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
            k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2; h1 ^= k1;
        };

        h1 ^= len; h1 ^= h1 >> 16; h1 *= 0x85ebca6b; h1 ^= h1 >> 13; h1 *= 0xc2b2ae35; h1 ^= h1 >> 16;
        return h1;
    }

public:
    BloomFilter() {}

    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        std::streamsize fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        if (fileSize < 25) return false;

        std::vector<uint8_t> buffer(fileSize);
        if (!file.read((char*)buffer.data(), fileSize)) return false;

        if (buffer[0]!='B' || buffer[1]!='L' || buffer[2]!='M' || buffer[3]!='3') return false;

        bitSize = read_be64(&buffer[5]);
        numHashes = read_be32(&buffer[13]);
        uint64_t arrayLen = read_be64(&buffer[17]);
        size_t headerSize = 25;

        if (fileSize < (headerSize + arrayLen)) return false;
        data.assign(buffer.begin() + headerSize, buffer.begin() + headerSize + arrayLen);
        loaded = true;
        std::cout << "[Bloom] Loaded! M=" << bitSize << " bits, K=" << numHashes << "\n";
        return true;
    }

    bool isLoaded() const { return loaded; }
    const uint8_t* getRawData() const { return data.data(); }
    size_t getSize() const { return data.size(); }

    bool check_hash160(const std::vector<uint8_t>& hash160) const {
        if (!loaded || hash160.empty()) return false;
        
        // Seed-uri critice
        uint32_t h1 = MurmurHash3(hash160.data(), (int)hash160.size(), 0xFBA4C795);
        uint32_t h2 = MurmurHash3(hash160.data(), (int)hash160.size(), 0x43876932);
        
        for (uint32_t i = 0; i < numHashes; i++) {
            uint64_t idx = ((uint64_t)h1 + (uint64_t)i * h2) % bitSize;
            if (!(data[idx/8] & (1 << (idx%8)))) return false;
        }
        return true;
    }
};