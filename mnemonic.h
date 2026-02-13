#pragma once
#define _CRT_SECURE_NO_WARNINGS

// --- FIX OPENSSL 3.0 DEPRECATION ---
#define OPENSSL_SUPPRESS_DEPRECATED
#define OPENSSL_API_COMPAT 0x10100000L
#pragma warning(disable : 4996) 

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
#include <cctype> 
#include <bitset> 

// ============================================================================
// 1. STRUCTURI GLOBALE
// ============================================================================

// Include MultiCoin pentru generarea adreselor specifice
#include "multicoin.h" 
#include "bloom.h"

struct RowInfo { 
    std::string type; 
    std::string path; 
    std::string addr; 
    std::string secret; 
    bool isHit; 
};

struct MnemonicResult {
    std::string mnemonic;
    std::string entropyHex; 
    std::vector<RowInfo> rows;
};

struct PathInfo { 
    std::string label; 
    std::string path; 
    std::string coin; 

    PathInfo(std::string l, std::string p, std::string c = "BTC") 
        : label(l), path(p), coin(c) {}
};

// ============================================================================
// OPENSSL & SECP256K1
// ============================================================================
#ifndef __CUDACC__
    #include <openssl/sha.h>
    #include <openssl/ripemd.h>
    #include <openssl/hmac.h>
    #include <openssl/evp.h>
    #include <secp256k1.h>
#endif

#include "utils.h"

// ============================================================================
// CLASELE DE CPU
// ============================================================================
#ifndef __CUDACC__

extern "C" {
    void launch_gpu_init(const char* host_wordlist, int length, int wordCount);
    void launch_gpu_mnemonic_search(unsigned long long start, unsigned long long count, int b, int t, const void* bf, size_t sz, unsigned long long* res, int* cnt);
}

class Bip32 {
public:
    // Derivare clasica din Seed
    static void Derive(secp256k1_context* ctx, const uint8_t* seed, int seedLen, const std::string& path, std::vector<uint8_t>& outPrivKey) {
        uint8_t I[64], key[32], chain[32];
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, seedLen, I, NULL);
        memcpy(key, I, 32); memcpy(chain, I + 32, 32);
        ApplyPath(ctx, key, chain, path, outPrivKey);
    }

    // Derivare din XPRV (Key + ChainCode)
    static void DeriveFromXprv(secp256k1_context* ctx, const std::vector<uint8_t>& parentKey, const std::vector<uint8_t>& parentChain, const std::string& path, std::vector<uint8_t>& outPrivKey) {
        uint8_t key[32], chain[32];
        memcpy(key, parentKey.data(), 32);
        memcpy(chain, parentChain.data(), 32);
        ApplyPath(ctx, key, chain, path, outPrivKey);
    }

private:
    static void ApplyPath(secp256k1_context* ctx, uint8_t* key, uint8_t* chain, const std::string& path, std::vector<uint8_t>& outPrivKey) {
        std::stringstream ss(path); std::string segment;
        while (std::getline(ss, segment, '/')) {
            if (segment == "m") continue;
            bool hardened = (!segment.empty() && segment.back() == '\''); 
            if (hardened) segment.pop_back();
            
            uint32_t index = 0;
            try { index = std::stoul(segment); } catch(...) { continue; }
            if (hardened) index |= 0x80000000;
            
            uint8_t data[37];
            uint8_t I[64];

            if (index & 0x80000000) { 
                // Hardened: 0x00 || Key || Index
                data[0] = 0x00; memcpy(data + 1, key, 32); 
            } else { 
                // Normal: Point(Key) || Index
                secp256k1_pubkey pub; 
                if(!secp256k1_ec_pubkey_create(ctx, &pub, key)) return; 
                size_t len = 33; 
                secp256k1_ec_pubkey_serialize(ctx, data, &len, &pub, SECP256K1_EC_COMPRESSED); 
            }
            
            data[33] = (index >> 24) & 0xFF; data[34] = (index >> 16) & 0xFF; 
            data[35] = (index >> 8) & 0xFF;  data[36] = index & 0xFF;
            
            HMAC(EVP_sha512(), chain, 32, data, 37, I, NULL);
            uint8_t tweak[32]; memcpy(tweak, I, 32); memcpy(chain, I + 32, 32);
            
            if(!secp256k1_ec_privkey_tweak_add(ctx, key, tweak)) return;
        }
        outPrivKey.assign(key, key + 32);
    }
};

class MnemonicTool {
private:
    std::vector<std::string> wordlist;
    secp256k1_context* ctx;
    std::vector<PathInfo> activePaths; 

    static constexpr const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    bool isHexChar(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
    
    bool isHexString(const std::string& s) {
        if (s.length() != 64) return false;
        for(char c : s) if(!isHexChar(c)) return false;
        return true;
    }

    std::string BytesToHex(const std::vector<uint8_t>& data) {
        std::stringstream ss; ss << std::hex << std::setfill('0');
        for(auto b : data) ss << std::setw(2) << (int)b;
        return ss.str();
    }

    std::vector<uint8_t> HexToBytes(const std::string& hex) {
        std::vector<uint8_t> bytes;
        for (unsigned int i = 0; i < hex.length(); i += 2) {
            std::string byteString = hex.substr(i, 2);
            try { bytes.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16)); } catch(...) {}
        }
        return bytes;
    }

    std::string BytesToBinaryString(const std::vector<uint8_t>& bytes) {
        std::string bin = "";
        for (auto b : bytes) bin += std::bitset<8>(b).to_string();
        return bin;
    }

    // Helper WIF pentru export (simplificat, default BTC mainnet 0x80)
    std::string PrivKeyToWIF(const std::vector<uint8_t>& privKey, bool compressed) {
        std::vector<uint8_t> data;
        data.push_back(0x80); // Mainnet
        data.insert(data.end(), privKey.begin(), privKey.end());
        if (compressed) data.push_back(0x01);
        
        // Manual Base58Check Encode pt independenta
        uint8_t hash1[32], hash2[32];
        SHA256(data.data(), data.size(), hash1);
        SHA256(hash1, 32, hash2);
        data.push_back(hash2[0]); data.push_back(hash2[1]); data.push_back(hash2[2]); data.push_back(hash2[3]);
        
        int zeros = 0;
        while (zeros < data.size() && data[zeros] == 0) zeros++;
        std::vector<unsigned char> b58((data.size() * 138 / 100) + 1);
        memset(b58.data(), 0, b58.size());
        
        for (size_t i = zeros; i < data.size(); i++) {
            int carry = data[i];
            for (auto it = b58.rbegin(); it != b58.rend(); it++) {
                carry += 256 * (*it);
                *it = carry % 58;
                carry /= 58;
            }
        }
        auto it = b58.begin();
        while (it != b58.end() && *it == 0) it++;
        
        std::string str(zeros, '1');
        while (it != b58.end()) str += pszBase58[*it++];
        return str;
    }

    std::vector<uint8_t> DecodeBase58(const std::string& str) {
        std::vector<uint8_t> vch;
        const char* pbegin = str.c_str();
        const char* pend = pbegin + str.length();
        while (pbegin != pend && isspace(*pbegin)) pbegin++;
        int zeros = 0;
        while (pbegin != pend && *pbegin == '1') { zeros++; pbegin++; }
        int size = (pend - pbegin) * 733 / 1000 + 1;
        std::vector<unsigned char> b256(size);
        while (pbegin != pend && !isspace(*pbegin)) {
            const char* ch = strchr(pszBase58, *pbegin);
            if (ch == NULL) break;
            int carry = ch - pszBase58;
            for (int i = b256.size() - 1; i >= 0; i--) {
                carry += 58 * b256[i];
                b256[i] = carry % 256;
                carry /= 256;
            }
            pbegin++;
        }
        std::vector<unsigned char>::iterator it = b256.begin();
        while (it != b256.end() && *it == 0) it++;
        vch.assign(zeros, 0x00);
        vch.insert(vch.end(), it, b256.end());
        return vch;
    }

    // --- DECODOR XPRV (Extended Private Key) ---
    bool decodeXprv(const std::string& xprv, std::vector<uint8_t>& outKey, std::vector<uint8_t>& outChain) {
        if (xprv.substr(0, 4) != "xprv") return false;
        std::vector<uint8_t> decoded = DecodeBase58(xprv);
        if (decoded.size() < 82) return false;
        
        outChain.assign(decoded.begin() + 13, decoded.begin() + 45);
        outKey.assign(decoded.begin() + 46, decoded.begin() + 78);
        return true;
    }

public:
    MnemonicTool() { ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY); }
    ~MnemonicTool() { if (ctx) secp256k1_context_destroy(ctx); }
    void setPaths(const std::vector<PathInfo>& paths) { activePaths = paths; }
    
    // Verificare Bloom
    bool checkAddressInBloom(const std::string& addr, const BloomFilter& bloom) {
        if (!bloom.isLoaded()) return false;
        // Hex (ETH)
        if (addr.rfind("0x", 0) == 0 && addr.length() == 42) {
             return bloom.check_hash160(HexToBytes(addr.substr(2)));
        }
        // Base58 (BTC, LTC, etc.)
        if (addr.length() >= 26 && addr.length() <= 35) {
            std::vector<uint8_t> decoded = DecodeBase58(addr);
            if(decoded.size() == 25) {
                // Hash160 se afla la offset 1, lungime 20
                std::vector<uint8_t> h160(decoded.begin() + 1, decoded.begin() + 21);
                return bloom.check_hash160(h160);
            }
        }
        return false; 
    }

    bool loadWordlist(const std::string& lang) {
        std::vector<std::string> search_paths = { lang, "bip39/" + lang + ".txt", "bip39/" + lang };
        std::ifstream file;
        for (const auto& p : search_paths) { file.open(p); if (file.is_open()) break; }
        if (!file.is_open()) return false;
        wordlist.clear();
        std::string line;
        while (std::getline(file, line)) {
            line.erase(line.find_last_not_of(" \n\r\t") + 1);
            if (!line.empty()) wordlist.push_back(line);
        }
        return (wordlist.size() > 0);
    }

    std::string phraseFromEntropyString(const std::string& entropyStr, const std::string& mode) {
        if (wordlist.empty()) return "ERROR_NO_WL";
        std::vector<uint8_t> entropyBytes;
        if (mode == "hex" || mode == "default") entropyBytes = HexToBytes(entropyStr);
        else if (mode == "bin") {
            for(size_t i=0; i<entropyStr.length(); i+=8) {
                if(i+8 <= entropyStr.length()) entropyBytes.push_back((uint8_t)std::bitset<8>(entropyStr.substr(i, 8)).to_ulong());
            }
        }
        if (entropyBytes.empty()) return "ERROR_EMPTY_ENTROPY";
        
        uint8_t hash[SHA256_DIGEST_LENGTH]; SHA256(entropyBytes.data(), entropyBytes.size(), hash);
        int checksumLenBits = (int)entropyBytes.size() * 8 / 32;
        std::string bits = BytesToBinaryString(entropyBytes);
        std::string hashBits = BytesToBinaryString(std::vector<uint8_t>(hash, hash + 32));
        bits += hashBits.substr(0, checksumLenBits);
        
        std::string phrase = "";
        for (int i = 0; i < (int)bits.length() / 11; i++) {
            phrase += wordlist[std::stoi(bits.substr(i * 11, 11), nullptr, 2)] + (i < (int)bits.length() / 11 - 1 ? " " : "");
        }
        return phrase;
    }

    // --- GENERARE RAW KEY (BRAINWALLET) ---
    void addRawKeyRows(const std::vector<uint8_t>& privKey, MnemonicResult& result, const BloomFilter& bloom) {
        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, privKey.data())) return;

        // 1. ETH Address (Uncompressed)
        uint8_t uPub[65]; size_t len = 65;
        secp256k1_ec_pubkey_serialize(ctx, uPub, &len, &pub, SECP256K1_EC_UNCOMPRESSED);
        std::vector<uint8_t> uPubVec(uPub, uPub + 65);
        
        std::string addrEth = MultiCoin::GenEth(uPubVec);
        result.rows.push_back({"Brainwallet [ETH]", "Raw (SHA256)", addrEth, "0x" + BytesToHex(privKey), checkAddressInBloom(addrEth, bloom)});

        // 2. BTC/MultiCoin Compressed
        uint8_t cPub[33]; size_t clen = 33;
        secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
        std::vector<uint8_t> vPubC(cPub, cPub + 33);
        
        std::string wifComp = PrivKeyToWIF(privKey, true);

        // BTC Legacy Compressed
        std::string legC = MultiCoin::GenLegacy(vPubC, "BTC");
        result.rows.push_back({"Brainwallet [BTC Legacy]", "Raw", legC, wifComp, checkAddressInBloom(legC, bloom)});

        // BTC Segwit (P2SH)
        std::string p2sh = MultiCoin::GenP2SH(vPubC, "BTC");
        result.rows.push_back({"Brainwallet [BTC P2SH]", "Raw", p2sh, wifComp, checkAddressInBloom(p2sh, bloom)});

        // BTC Bech32
        std::string bech = MultiCoin::GenBech32(vPubC, "BTC");
        result.rows.push_back({"Brainwallet [BTC Bech32]", "Raw", bech, wifComp, checkAddressInBloom(bech, bloom)});
    }

    // --- LOGICA CENTRALA DE DERIVARE MULTI-COIN ---
    void checkPathsAndAdd(const uint8_t* seed, MnemonicResult& result, const BloomFilter& bloom, int seedLen = 64) {
        // Daca nu avem cai, folosim un default BTC
        std::vector<PathInfo> pathsToCheck = activePaths.empty() ? std::vector<PathInfo>{ PathInfo("BIP32", "m/0/0", "BTC") } : activePaths;
        
        for (auto& p : pathsToCheck) {
            std::vector<uint8_t> pk; 
            // Derivare BIP32 (Master -> Child Key)
            Bip32::Derive(ctx, seed, seedLen, p.path, pk);
            
            // Generare PubKey din PrivKey
            secp256k1_pubkey pub;
            if (!secp256k1_ec_pubkey_create(ctx, &pub, pk.data())) continue;

            // Compressed Public Key
            uint8_t cPub[33]; size_t clen = 33;
            secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
            std::vector<uint8_t> vPubC(cPub, cPub + 33);

            // Uncompressed Public Key
            uint8_t uPub[65]; size_t ulen = 65;
            secp256k1_ec_pubkey_serialize(ctx, uPub, &ulen, &pub, SECP256K1_EC_UNCOMPRESSED);
            std::vector<uint8_t> vPubU(uPub, uPub + 65);

            // GENERARE ADRESE BAZAT PE MONEDA (MULTI-COIN)
            struct AddrGen { std::string type; std::string val; };
            std::vector<AddrGen> addrs;

            if (p.coin == "ETH") {
                addrs.push_back({ "ETH", MultiCoin::GenEth(vPubU) });
            } 
            else {
                // Monede gen BTC, LTC, DOGE, DASH, BTG, ZEC...
                // 1. Legacy
                addrs.push_back({ p.coin + "-" + p.label + " [Legacy]", MultiCoin::GenLegacy(vPubC, p.coin) });
                addrs.push_back({ p.coin + "-" + p.label + " [Legacy Uncomp]", MultiCoin::GenLegacy(vPubU, p.coin) });
                
                // 2. P2SH (Segwit)
                addrs.push_back({ p.coin + "-" + p.label + " [P2SH]", MultiCoin::GenP2SH(vPubC, p.coin) });

                // 3. Bech32 (Daca moneda suporta, functia returneaza string gol daca nu)
                std::string b32 = MultiCoin::GenBech32(vPubC, p.coin);
                if (!b32.empty()) {
                     addrs.push_back({ p.coin + "-" + p.label + " [Bech32]", b32 });
                }
            }

            // Verificare Bloom & Adaugare in Rezultat
            for (const auto& ag : addrs) {
                bool hit = checkAddressInBloom(ag.val, bloom);
                result.rows.push_back({ ag.type, p.path, ag.val, "", hit });
            }
        }
    }

    // --- Procesare Raw Line (Brainwallet / Import) ---
    MnemonicResult processRawLine(std::string line, const BloomFilter& bf) {
        MnemonicResult r;
        r.mnemonic = line;
        
        // Hash simplu SHA256 pe input (Brainwallet style)
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256((const unsigned char*)line.c_str(), line.length(), hash);
        
        // Pentru brainwallet, folosim cheia privata derivata din hash
        std::vector<uint8_t> pk(hash, hash + 32);
        addRawKeyRows(pk, r, bf);
        
        return r;
    }

    // --- Functia Principala ---
    MnemonicResult processSeed(unsigned long long seedVal, std::string order, int wordCount, const BloomFilter& bloom, const std::string& filter = "ALL", const std::string& entropyMode = "default", std::string explicitPhrase = "") {
        MnemonicResult result;
        if (!explicitPhrase.empty()) return processRawLine(explicitPhrase, bloom);

        int entropyBytesLen = (wordCount * 11 - wordCount / 3) / 8;
        if(entropyBytesLen <= 0) entropyBytesLen = 32;

        std::vector<uint8_t> ent;
        std::string visualInfo = "";

        if (entropyMode == "schematic") {
            ent = generateSchematicEntropy(seedVal, entropyBytesLen);
            result.entropyHex = BytesToHex(ent);
        }
        else if (entropyMode != "default" && entropyMode != "" && entropyMode != "random") {
            auto p = generateEntropyWithVisual(seedVal, entropyMode, entropyBytesLen);
            ent = p.first; visualInfo = p.second; result.entropyHex = visualInfo;
        } else {
            std::mt19937_64 r(seedVal);
            ent.resize(entropyBytesLen);
            for (int k = 0; k < entropyBytesLen; k++) ent[k] = r() & 0xFF;
            result.entropyHex = BytesToHex(ent);
        }

        if (wordlist.empty()) return result;

        // Generare Mnemonic
        uint8_t h[32]; SHA256(ent.data(), ent.size(), h);
        std::string bits = BytesToBinaryString(ent);
        std::string hashBits = BytesToBinaryString(std::vector<uint8_t>(h, h + 32));
        bits += hashBits.substr(0, wordCount / 3);
        
        std::string phr = "";
        for (int w = 0; w < wordCount; w++) {
            int idx = std::stoi(bits.substr(w * 11, 11), nullptr, 2);
            if(idx < wordlist.size()) phr += wordlist[idx] + (w < wordCount - 1 ? " " : "");
        }
        result.mnemonic = phr;

        // Derivare BIP39 Seed (PBKDF2)
        uint8_t mSeed[64];
        if (PKCS5_PBKDF2_HMAC(phr.c_str(), (int)phr.length(), (const unsigned char*)"mnemonic", 8, 2048, EVP_sha512(), 64, mSeed)) {
            checkPathsAndAdd(mSeed, result, bloom, 64);
        }
        return result;
    }

    // Helpers Entropie
    std::pair<std::vector<uint8_t>, std::string> generateEntropyWithVisual(unsigned long long seed, const std::string& mode, int entropyBytes) {
        std::mt19937_64 rng(seed);
        std::string visual = "";
        std::vector<uint8_t> bytes;
        // ... (Logica generare entropie pastrata din codul anterior sau simplificata aici) ...
        // Implementare simpla pt compilare
        bytes.resize(entropyBytes);
        for(int i=0; i<entropyBytes; i++) bytes[i] = rng() & 0xFF;
        return {bytes, "EntropyGen"}; 
    }
    
    std::vector<uint8_t> generateSchematicEntropy(unsigned long long seed, int entropyBytes) {
        return std::vector<uint8_t>(entropyBytes, (uint8_t)seed);
    }
};
#endif