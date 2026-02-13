#pragma once

#include <stdint.h>
#include <stddef.h>

// --- DEPENDINȚE C++ PENTRU XPRV GENERATOR ---
#ifdef __cplusplus
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstring>
#include <algorithm>
// ATENTIE: Necesita linkare cu OpenSSL (-lssl -lcrypto)
#include <openssl/sha.h>
#endif

// =============================================================
//  1. INTERFAȚA CUDA (C Linkage)
//  Aceste funcții sunt implementate în mnemonic_gpu.cu
// =============================================================
#ifdef __cplusplus
extern "C" {
#endif

    /**
     * Inițializează memoria GPU cu lista de cuvinte (BIP39).
     * Se apelează o singură dată la start.
     */
    void launch_gpu_init(const char* host_wordlist_raw, int length, int wordCount);

    /**
     * Lansează kernelul GPU pentru a căuta chei.
     */
    void launch_gpu_mnemonic_search(
        unsigned long long start_seed, 
        unsigned long long count, 
        int blocks, 
        int threads, 
        const void* bloom_filter, 
        size_t bloom_size, 
        unsigned long long* out_seeds, 
        int* out_count
    );

#ifdef __cplusplus
}
#endif

// =============================================================
//  2. GENERATOR XPRV (C++ Class)
//  Implementare inline pentru simplitate
// =============================================================
#ifdef __cplusplus

class XprvGenerator {
private:
    const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Helper: Base58 Encode (Standard Bitcoin)
    std::string EncodeBase58(const unsigned char* data, size_t len) {
        std::vector<unsigned char> digits((len * 138 / 100) + 1);
        size_t digitslen = 1;
        for (size_t i = 0; i < len; i++) {
            unsigned int carry = data[i];
            for (size_t j = 0; j < digitslen; j++) {
                carry += (unsigned int)(digits[j]) << 8;
                digits[j] = (unsigned char)(carry % 58);
                carry /= 58;
            }
            while (carry > 0) {
                digits[digitslen++] = (unsigned char)(carry % 58);
                carry /= 58;
            }
        }
        std::string result = "";
        for (size_t i = 0; i < len && data[i] == 0; i++) result += '1';
        for (size_t i = 0; i < digitslen; i++) result += BASE58_ALPHABET[digits[digitslen - 1 - i]];
        return result;
    }

public:
    XprvGenerator() {}

    /**
     * Genereaza un string XPRV valid (Mainnet).
     * @param seedIndex - Indexul curent (folosit pentru modul schematic).
     * @param mode - "random" (ignora indexul) sau "schematic" (deterministic).
     */
    std::string generate(uint64_t seedIndex, const std::string& mode) {
        unsigned char xprvData[78];
        // Version bytes pentru Mainnet XPRV: 0x0488ADE4
        unsigned char version[4] = {0x04, 0x88, 0xAD, 0xE4}; 
        
        // 1. Construct Header
        memcpy(xprvData, version, 4);
        xprvData[4] = 0x00;             // Depth: 0 (Master Node)
        memset(xprvData + 5, 0, 4);     // Parent Fingerprint: 0x00000000
        memset(xprvData + 9, 0, 4);     // Child Number: 0x00000000

        // 2. Generate Chain Code (32 bytes) & Key Material (32 bytes)
        unsigned char buffer[64]; // [0..31] = ChainCode, [32..63] = Key

        if (mode == "random") {
            // Mod Random: Folosim entropie hardware
            std::random_device rd;
            std::mt19937_64 gen(rd());
            for(int i = 0; i < 8; i++) {
                uint64_t r = gen();
                memcpy(buffer + i*8, &r, 8);
            }
        } else {
            // Mod Schematic: Determinist bazat pe seedIndex
            // Creeaza un pattern repetitiv modificat de index
            for(int i = 0; i < 64; i++) {
                buffer[i] = (seedIndex >> ((i % 8) * 8)) & 0xFF;
                // Adaugam o constanta pentru a evita zero-uri simple
                buffer[i] ^= (0xAA + i); 
            }
            
            // Injectam indexul direct in primii 8 bytes pentru unicitate garantata
            uint64_t* ptr = (uint64_t*)buffer;
            ptr[0] ^= seedIndex;
            ptr[1] ^= (seedIndex * 6364136223846793005ULL + 1442695040888963407ULL);
        }

        // Copiem Chain Code
        memcpy(xprvData + 13, buffer, 32); 

        // 3. Construct Private Key (33 bytes)
        // 0x00 prefix este obligatoriu pentru chei private in BIP32
        xprvData[45] = 0x00; 
        memcpy(xprvData + 46, buffer + 32, 32); 

        // 4. Calculate Checksum (Double SHA256)
        unsigned char hash1[SHA256_DIGEST_LENGTH];
        unsigned char hash2[SHA256_DIGEST_LENGTH];
        
        SHA256(xprvData, 78, hash1);
        SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);

        // 5. Append Checksum and Encode
        unsigned char finalBuffer[82];
        memcpy(finalBuffer, xprvData, 78);
        memcpy(finalBuffer + 78, hash2, 4); // Primii 4 bytes din hash2

        return EncodeBase58(finalBuffer, 82);
    }
};

#endif // __cplusplus