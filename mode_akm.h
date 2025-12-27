#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdint>
#include <random>

#include <openssl/sha.h>
#include <secp256k1.h>

#include "utils.h"
#include "bloom.h"
#include "akm.h"

// Bridge către codul CUDA (Implementat în mnemonic_gpu.cu sau GpuCore.cu)
extern "C" void launch_gpu_akm_search(
    unsigned long long startSeed, 
    unsigned long long count, 
    int blocks, 
    int threads, 
    const void* bloomFilterData, 
    size_t bloomFilterSize,
    unsigned long long* outFoundSeeds, 
    int* outFoundCount,
    int targetBits 
);

// Structura rezultat pentru UI
struct AkmResult {
    struct AddrRow { std::string type, path, addr; std::string status; bool isHit; };
    std::string phrase;
    std::string hexKey;
    std::vector<AddrRow> rows;
};

// Helper pentru calculul cuvintelor din cheie (BigInt)
inline int BigIntDivMod(std::vector<uint8_t>& n, int divisor) {
    int remainder = 0;
    for (size_t i = 0; i < n.size(); i++) {
        unsigned int val = (remainder << 8) + n[i];
        n[i] = val / divisor;
        remainder = val % divisor;
    }
    return remainder;
}

class AkmTool {
private:
    AkmLogic akm;
    secp256k1_context* ctx;

    void applyBitMask(std::vector<uint8_t>& key, int bits) {
        if (bits <= 0 || bits > 256) return;
        int topBitIndex = bits - 1;
        int byteIndex = 31 - (topBitIndex / 8);
        int bitInByte = topBitIndex % 8;
        for (int i = 0; i < byteIndex; ++i) key[i] = 0;
        unsigned char mask = (1 << (bitInByte + 1)) - 1;
        key[byteIndex] &= mask;
        key[byteIndex] |= (1 << bitInByte);
    }

public:
    AkmTool() {
        ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    }

    ~AkmTool() {
        if (ctx) secp256k1_context_destroy(ctx);
    }

    void init(const std::string& profile, const std::string& wordlistPath) {
        akm.init(profile, wordlistPath);
    }

    AkmResult processAkmSeed(unsigned long long seed, const std::string& order, int targetBits, int phraseLen, const BloomFilter& bloom) {
        AkmResult result;
        std::vector<uint8_t> directKey(32, 0);
        const std::vector<std::string>& akmWords = akm.getWordList();
        size_t akmSize = akmWords.size();

        if (akmSize == 0) return result;

        if (order == "random") {
            std::mt19937_64 r(seed);
            for (auto& b : directKey) b = (uint8_t)(r() % 256);
        } else {
            for (int b = 0; b < 8; ++b) directKey[31 - b] = (seed >> (b * 8)) & 0xFF;
        }

        if (targetBits > 0) applyBitMask(directKey, targetBits);
        result.hexKey = toHex(directKey);

        std::vector<uint8_t> tempForWords = directKey;
        std::vector<std::string> phraseWords(phraseLen);
        for (int w = phraseLen - 1; w >= 0; --w) {
            int idx = BigIntDivMod(tempForWords, (int)akmSize);
            phraseWords[w] = akmWords[idx];
        }
        for (const auto& w : phraseWords) result.phrase += w + " ";

        secp256k1_pubkey pubkey;
        if (secp256k1_ec_pubkey_create(ctx, &pubkey, directKey.data())) {
            uint8_t cPub[33], uPub[65];
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(ctx, cPub, &len, &pubkey, SECP256K1_EC_COMPRESSED);
            len = 65;
            secp256k1_ec_pubkey_serialize(ctx, uPub, &len, &pubkey, SECP256K1_EC_UNCOMPRESSED);

            std::vector<uint8_t> vPubC(cPub, cPub + 33), vPubU(uPub, uPub + 65);

            struct Check { std::string label, addr; std::vector<uint8_t> payloadData; };
            std::vector<Check> checks = {
                {"Comp P2PKH",   PubKeyToLegacy(vPubC),        vPubC},
                {"Uncomp P2PKH", PubKeyToLegacy(vPubU),        vPubU},
                {"Comp P2SH",     PubKeyToNestedSegwit(vPubC), vPubC},
                {"Bech32",        PubKeyToNativeSegwit(vPubC), vPubC}
            };

            for (const auto& c : checks) {
                std::vector<uint8_t> payload;
                if (c.label.find("P2SH") != std::string::npos) {
                    std::vector<uint8_t> kH; Hash160(c.payloadData, kH);
                    std::vector<uint8_t> r = { 0x00, 0x14 };
                    r.insert(r.end(), kH.begin(), kH.end());
                    Hash160(r, payload);
                } else {
                    Hash160(c.payloadData, payload);
                }

                bool isHit = bloom.isLoaded() ? bloom.check_hash160(payload) : false;
                result.rows.push_back({"AKM", c.label, c.addr, (isHit ? "HIT" : "-"), isHit});
            }
        }
        return result;
    }

    size_t getWordCount() { return akm.getWordCount(); }
    void listProfiles() { akm.listProfiles(); }
};