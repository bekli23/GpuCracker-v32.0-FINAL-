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
    // [MODIFICAT] Structura interna pentru a tine minte datele unui singur fisier .blf
    struct BloomLayer {
        std::vector<uint8_t> data;
        uint64_t bitSize = 0;
        uint32_t numHashes = 0;
    };

    // [MODIFICAT] Lista de straturi (fisiere) incarcate
    std::vector<BloomLayer> layers; 
    bool loaded = false;

    // --- HELPER FUNCTIONS ---
    uint64_t read_be64(const uint8_t* ptr) const {
        uint64_t val = 0; for(int i=0; i<8; ++i) val = (val << 8) | ptr[i]; return val;
    }
    uint32_t read_be32(const uint8_t* ptr) const {
        uint32_t val = 0; for(int i=0; i<4; ++i) val = (val << 8) | ptr[i]; return val;
    }

    // --- SINCRONIZARE CU BUILD_BLOOM.CPP ---
    inline uint32_t rotl32(uint32_t x, int8_t r) const { return (x << r) | (x >> (32 - r)); }

    uint32_t MurmurHash3(const void* key, int len, uint32_t seed) const {
        const uint8_t* data = (const uint8_t*)key;
        const int nblocks = len / 4;

        uint32_t h1 = seed;
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;

        const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);

        for (int i = -nblocks; i; i++) {
            uint32_t k1;
            std::memcpy(&k1, (const uint8_t*)blocks + (i * 4), 4);

            k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2;
            h1 ^= k1; h1 = rotl32(h1, 13); h1 = h1 * 5 + 0xe6546b64;
        }

        const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
        uint32_t k1 = 0;

        switch (len & 3) {
        case 3: k1 ^= tail[2] << 16;
        case 2: k1 ^= tail[1] << 8;
        case 1: k1 ^= tail[0];
            k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2; h1 ^= k1;
        };

        h1 ^= len;
        h1 ^= h1 >> 16; h1 *= 0x85ebca6b; h1 ^= h1 >> 13; h1 *= 0xc2b2ae35; h1 ^= h1 >> 16;

        return h1;
    }

    // [NOU] Functie privata pentru incarcarea unui singur fisier
    bool loadSingleFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Bloom] Error: Cannot open " << filename << "\n";
            return false;
        }

        file.seekg(0, std::ios::end);
        size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);
        
        // Header minim BLM3 = 25 bytes
        if (fileSize < 25) return false;

        std::vector<uint8_t> buffer(fileSize);
        if (!file.read((char*)buffer.data(), fileSize)) return false;

        // Verificam Magic Number "BLM3"
        if (buffer[0]!='B' || buffer[1]!='L' || buffer[2]!='M' || buffer[3]!='3') {
            std::cerr << "[Bloom] Invalid format (Magic mismatch) in " << filename << "\n";
            return false;
        }

        BloomLayer layer;
        layer.bitSize = read_be64(&buffer[5]);
        layer.numHashes = read_be32(&buffer[13]);
        uint64_t arrayLen = read_be64(&buffer[17]);
        size_t headerSize = 25;

        if (fileSize < (headerSize + arrayLen)) {
            std::cerr << "[Bloom] File truncated: " << filename << "\n";
            return false;
        }
        
        // Copiem datele efective in noul layer
        layer.data.assign(buffer.begin() + headerSize, buffer.begin() + headerSize + arrayLen);
        layers.push_back(layer);
        
        std::cout << "[Bloom] Loaded " << filename << "! M=" << layer.bitSize << " bits, K=" << layer.numHashes << "\n";
        return true;
    }

public:
    // [MODIFICAT] Incarca o lista de fisiere (vector de string-uri)
    bool load(const std::vector<std::string>& filenames) {
        layers.clear();
        int loadedCount = 0;
        for(const auto& f : filenames) {
            if(loadSingleFile(f)) {
                loadedCount++;
            }
        }
        loaded = (loadedCount > 0);
        if(!loaded) {
            std::cerr << "[Bloom] No valid Bloom Filter files loaded!\n";
        }
        return loaded;
    }

    // [LEGACY] Pentru compatibilitate cu apeluri vechi care trimit un singur string
    bool load(const std::string& filename) {
        std::vector<std::string> v = { filename };
        return load(v);
    }

    bool isLoaded() const { return loaded; }

    // [NOU] Accesoare pentru GPU Runner (Multi-file support)
    size_t getLayerCount() const { return layers.size(); }
    
    const uint8_t* getLayerData(size_t index) const {
        if(index < layers.size()) return layers[index].data.data();
        return nullptr;
    }
    
    size_t getLayerSize(size_t index) const {
        if(index < layers.size()) return layers[index].data.size();
        return 0;
    }

    // [LEGACY] Returneaza primul layer (pentru compatibilitate)
    const uint8_t* getRawData() const { return getLayerData(0); }
    size_t getSize() const { return getLayerSize(0); }

    // [MODIFICAT] Verificare CPU pe TOATE fisierele incarcate
    bool check_hash160(const std::vector<uint8_t>& hash160) const {
        if (!loaded || hash160.empty()) return false;
        
        // Calculam hash-urile de baza o singura data
        uint32_t h1_base = MurmurHash3(hash160.data(), (int)hash160.size(), 0xFBA4C795);
        uint32_t h2_base = MurmurHash3(hash160.data(), (int)hash160.size(), 0x43876932);

        // Iteram prin fiecare fisier .blf incarcat
        for (const auto& layer : layers) {
            bool match = true;
            for (uint32_t i = 0; i < layer.numHashes; ++i) {
                // Atentie: folosim bitSize specific layer-ului curent
                uint64_t idx = ((uint64_t)h1_base + (uint64_t)i * h2_base) % layer.bitSize;
                
                // Verificam bitul in vectorul de date al layer-ului curent
                if (!(layer.data[idx / 8] & (1 << (idx % 8)))) {
                    match = false;
                    break; // Trecem la urmatorul fisier daca bitul lipseste aici? NU, daca bitul lipseste, acest filtru zice NU.
                }
            }
            // Daca match a ramas true, inseamna ca hash160 exista in ACEST layer
            if (match) return true; 
        }

        // Daca am trecut prin toate layerele si nu am gasit match
        return false;
    }
};