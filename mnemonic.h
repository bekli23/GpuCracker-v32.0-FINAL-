#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <map>
#include <cstdint>
#include <random>

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <secp256k1.h>

#include "utils.h"
#include "bloom.h"

extern "C" {
    void launch_gpu_init(const char* host_wordlist, int length);
    void launch_gpu_mnemonic_search(unsigned long long startSeed, unsigned long long count, int blocks, int threads, const void* bloomFilterData, size_t bloomFilterSize, unsigned long long* outFoundSeeds, int* outFoundCount);
}

struct MnemonicResult {
    struct AddrRow { std::string type, path, addr; bool isHit; };
    std::string mnemonic;
    std::vector<AddrRow> rows;
};

class Bip32 {
public:
    static void Derive(secp256k1_context* ctx, const uint8_t* seed, int seedLen, const std::string& path, std::vector<uint8_t>& outPrivKey) {
        uint8_t I[64], key[32], chain[32];
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, seedLen, I, NULL);
        memcpy(key, I, 32); memcpy(chain, I + 32, 32);
        std::stringstream ss(path); std::string segment;
        while (std::getline(ss, segment, '/')) {
            if (segment == "m") continue;
            bool hardened = (!segment.empty() && segment.back() == '\''); 
            if (hardened) segment.pop_back();
            uint32_t index = 0;
            try { index = std::stoul(segment); } catch(...) { continue; }
            if (hardened) index |= 0x80000000;
            uint8_t data[37];
            if (index & 0x80000000) { data[0] = 0x00; memcpy(data + 1, key, 32); }
            else { 
                secp256k1_pubkey pub; 
                if(!secp256k1_ec_pubkey_create(ctx, &pub, key)) return;
                size_t len = 33; secp256k1_ec_pubkey_serialize(ctx, data, &len, &pub, SECP256K1_EC_COMPRESSED); 
            }
            data[33] = (index >> 24) & 0xFF; data[34] = (index >> 16) & 0xFF;
            data[35] = (index >> 8) & 0xFF;  data[36] = index & 0xFF;
            HMAC(EVP_sha512(), chain, 32, data, 37, I, NULL);
            uint8_t tweak[32]; memcpy(tweak, I, 32); memcpy(chain, I + 32, 32);
            secp256k1_ec_privkey_tweak_add(ctx, key, tweak);
        }
        outPrivKey.assign(key, key + 32);
    }
};

class MnemonicTool {
private:
    std::vector<std::string> wordlist;
    secp256k1_context* ctx;

public:
    MnemonicTool() { ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY); }
    ~MnemonicTool() { if (ctx) secp256k1_context_destroy(ctx); }

    // Suport pentru orice fisier de limba din folderul bip39/
    bool loadWordlist(const std::string& lang) {
        std::vector<std::string> search_paths = { lang, "bip39/" + lang + ".txt", "bip39/" + lang };
        std::ifstream file;
        for (const auto& p : search_paths) {
            file.open(p);
            if (file.is_open()) break;
        }
        if (!file.is_open()) return false;

        wordlist.clear();
        std::string line;
        while (std::getline(file, line)) {
            line.erase(line.find_last_not_of(" \n\r\t") + 1);
            if (!line.empty()) wordlist.push_back(line);
        }
        return (wordlist.size() == 2048);
    }

    MnemonicResult processSeed(unsigned long long seed, const std::string& order, int wordCount, const BloomFilter& bloom) {
        MnemonicResult result;
        
        // Calculam entropia necesara: 12 cuvinte = 16 bytes, 24 cuvinte = 32 bytes
        int entropyBytes = (wordCount * 11 - wordCount / 3) / 8;
        uint8_t ent[32] = {0}; 

        if (order == "random") {
            std::mt19937_64 r(seed);
            for (int k = 0; k < entropyBytes; k++) ent[k] = r() & 0xFF;
        } else {
            // Sequential: Seed-ul de 64 biti este pus la sfarsitul entropiei
            for (int k = 0; k < 8 && k < entropyBytes; k++) 
                ent[entropyBytes - 1 - k] = (uint8_t)((seed >> (k * 8)) & 0xFF);
        }

        // Generare Checksum (primii ENT/32 biti din SHA256)
        uint8_t h[32];
        SHA256(ent, entropyBytes, h);
        int checksumBits = entropyBytes * 8 / 32;

        std::string phr;
        for (int w = 0; w < wordCount; w++) {
            int bitStart = w * 11;
            int v = 0;
            for (int b = 0; b < 11; b++) {
                int currBit = bitStart + b;
                int bit;
                if (currBit < entropyBytes * 8) {
                    bit = (ent[currBit / 8] >> (7 - (currBit % 8))) & 1;
                } else {
                    // Extragere din checksum
                    int csBitIdx = currBit - (entropyBytes * 8);
                    bit = (h[0] >> (7 - csBitIdx)) & 1;
                }
                v = (v << 1) | bit;
            }
            phr += wordlist[v] + (w < (wordCount - 1) ? " " : "");
        }
        result.mnemonic = phr;

        uint8_t mSeed[64];
        if (PKCS5_PBKDF2_HMAC(phr.c_str(), (int)phr.length(), (const unsigned char*)"mnemonic", 8, 2048, EVP_sha512(), 64, mSeed)) {
            struct Path { std::string label, path; };
            std::vector<Path> paths = { 
                {"BIP32", "m/0/0"}, 
                {"BIP44", "m/44'/0'/0'/0/0"}, 
                {"BIP49", "m/49'/0'/0'/0/0"}, 
                {"BIP84", "m/84'/0'/0'/0/0"} 
            };
            
            for (auto& p : paths) {
                std::vector<uint8_t> pk;
                Bip32::Derive(ctx, mSeed, 64, p.path, pk);
                if (pk.empty()) continue;

                secp256k1_pubkey pub;
                if (secp256k1_ec_pubkey_create(ctx, &pub, pk.data())) {
                    uint8_t cPub[33]; size_t clen = 33;
                    secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
                    std::vector<uint8_t> vPub(cPub, cPub + 33), pubHash, checkPayload;
                    Hash160(vPub, pubHash);
                    
                    std::string addr;
                    if (p.label == "BIP84") { 
                        addr = PubKeyToNativeSegwit(vPub); 
                        checkPayload = pubHash; 
                    }
                    else if (p.label == "BIP49") {
                        addr = PubKeyToNestedSegwit(vPub);
                        std::vector<uint8_t> redeem = { 0x00, 0x14 };
                        redeem.insert(redeem.end(), pubHash.begin(), pubHash.end());
                        Hash160(redeem, checkPayload);
                    } else { 
                        addr = PubKeyToLegacy(vPub); 
                        checkPayload = pubHash; 
                    }
                    
                    bool hit = bloom.isLoaded() ? bloom.check_hash160(checkPayload) : false;
                    result.rows.push_back({p.label, p.path, addr, hit});
                }
            }
        }
        return result;
    }
};