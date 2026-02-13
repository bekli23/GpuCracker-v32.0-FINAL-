#define _CRT_SECURE_NO_WARNINGS

// SUPRIMARE ERORI OPENSSL DEPRECATED (Critic pentru VS2022+)
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
#include <cctype>

// Librarii externe
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <secp256k1.h>

// Includem DOAR multicoin.h pentru Keccak (ETH). 
#include "multicoin.h"

// ============================================================================
//  HELPERE LOCALE
// ============================================================================

const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

std::string EncodeBase58(const unsigned char* pbegin, const unsigned char* pend) {
    int zeros = 0;
    while (pbegin != pend && *pbegin == 0) { pbegin++; zeros++; }
    std::vector<unsigned char> b58((pend - pbegin) * 138 / 100 + 1);
    std::vector<unsigned char>::iterator it = b58.begin();
    *it = 0; 
    while (pbegin != pend) {
        int carry = *pbegin;
        for (std::vector<unsigned char>::iterator i = b58.begin(); i != it + 1; ++i) {
            carry += 256 * (*i);
            *i = carry % 58;
            carry /= 58;
        }
        while (carry != 0) {
            it++; *it = carry % 58; carry /= 58;
        }
        pbegin++;
    }
    std::string str;
    str.reserve(zeros + (it - b58.begin() + 1));
    str.assign(zeros, '1');
    for (std::vector<unsigned char>::reverse_iterator i = b58.rbegin() + (b58.size() - 1 - (it - b58.begin())); i != b58.rend(); ++i)
        str += pszBase58[*i];
    return str;
}

std::string EncodeBase58Check(const std::vector<uint8_t>& vchIn) {
    std::vector<uint8_t> vch(vchIn);
    uint8_t hash[SHA256_DIGEST_LENGTH], hash2[SHA256_DIGEST_LENGTH];
    SHA256(vch.data(), vch.size(), hash);
    SHA256(hash, SHA256_DIGEST_LENGTH, hash2);
    vch.push_back(hash2[0]); vch.push_back(hash2[1]); vch.push_back(hash2[2]); vch.push_back(hash2[3]);
    return EncodeBase58(vch.data(), vch.data() + vch.size());
}

void Hash160(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
    uint8_t sha[SHA256_DIGEST_LENGTH];
    SHA256(in.data(), in.size(), sha);
    uint8_t rip[RIPEMD160_DIGEST_LENGTH];
    RIPEMD160(sha, SHA256_DIGEST_LENGTH, rip);
    out.assign(rip, rip + 20);
}

std::string PubKeyToLegacy(const std::vector<uint8_t>& pubKey) {
    std::vector<uint8_t> hash; Hash160(pubKey, hash);
    std::vector<uint8_t> data = {0x00}; 
    data.insert(data.end(), hash.begin(), hash.end());
    return EncodeBase58Check(data);
}

std::string PubKeyToNestedSegwit(const std::vector<uint8_t>& pubKey) {
    std::vector<uint8_t> pubHash; Hash160(pubKey, pubHash);
    std::vector<uint8_t> script = { 0x00, 0x14 };
    script.insert(script.end(), pubHash.begin(), pubHash.end());
    std::vector<uint8_t> scriptHash; Hash160(script, scriptHash);
    std::vector<uint8_t> data = {0x05}; 
    data.insert(data.end(), scriptHash.begin(), scriptHash.end());
    return EncodeBase58Check(data);
}

namespace Bech32 {
    const char* charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    std::string encode(const std::vector<int>& values) {
        uint32_t chk = 1;
        std::string ret = "bc1"; 
        uint32_t gen[] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
        auto polymod_step = [&](uint8_t v) {
            uint8_t b = chk >> 25;
            chk = ((chk & 0x1ffffff) << 5) ^ v;
            for(int i=0; i<5; ++i) if((b>>i)&1) chk ^= gen[i];
        };
        polymod_step(3); polymod_step(3); polymod_step(0); polymod_step(2); polymod_step(3); 
        for(int v : values) { polymod_step(v); ret += charset[v]; }
        for(int i=0; i<6; ++i) polymod_step(0);
        chk ^= 1;
        for(int i=0; i<6; ++i) ret += charset[(chk >> ((5-i)*5)) & 31];
        return ret;
    }
}

std::string PubKeyToNativeSegwit(const std::vector<uint8_t>& pubKey) {
    std::vector<uint8_t> h160; Hash160(pubKey, h160);
    std::vector<int> data5; data5.push_back(0); 
    int acc = 0, bits = 0;
    for(uint8_t b : h160) {
        acc = (acc << 8) | b; bits += 8;
        while(bits >= 5) { data5.push_back((acc >> (bits - 5)) & 31); bits -= 5; }
    }
    if(bits > 0) data5.push_back((acc << (5 - bits)) & 31);
    return Bech32::encode(data5);
}

std::string PrivKeyToWIF(const std::vector<uint8_t>& privKey, bool compressed) {
    std::vector<uint8_t> data;
    data.push_back(0x80); 
    data.insert(data.end(), privKey.begin(), privKey.end());
    if (compressed) data.push_back(0x01);
    return EncodeBase58Check(data);
}

std::string ToHex(const std::vector<uint8_t>& data) {
    std::stringstream ss; ss << "0x" << std::hex << std::setfill('0');
    for (auto b : data) ss << std::setw(2) << (int)b;
    return ss.str();
}

// ============================================================================
//  DERIVARE & PATH PARSING
// ============================================================================

void DerivePath(secp256k1_context* ctx, const uint8_t* seed, int seedLen, const std::string& path, std::vector<uint8_t>& outPrivKey) {
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
        if (index & 0x80000000) {
            data[0] = 0x00; memcpy(data + 1, key, 32);
        } else {
            secp256k1_pubkey pub;
            if(!secp256k1_ec_pubkey_create(ctx, &pub, key)) return;
            size_t len = 33; secp256k1_ec_pubkey_serialize(ctx, data, &len, &pub, SECP256K1_EC_COMPRESSED);
        }
        data[33] = (index >> 24) & 0xFF; data[34] = (index >> 16) & 0xFF;
        data[35] = (index >> 8) & 0xFF;  data[36] = index & 0xFF;
        HMAC(EVP_sha512(), chain, 32, data, 37, I, NULL);
        uint8_t tweak[32]; memcpy(tweak, I, 32); memcpy(chain, I + 32, 32);
        if(!secp256k1_ec_privkey_tweak_add(ctx, key, tweak)) return;
    }
    outPrivKey.assign(key, key + 32);
}

void expandPathRecursive(const std::vector<std::string>& segments, int index, std::string currentPath, std::vector<std::string>& results) {
    if (index >= segments.size()) { results.push_back(currentPath); return; }
    std::string seg = segments[index];
    size_t dash = seg.find('-');
    if (dash != std::string::npos) {
        bool hardened = (seg.back() == '\'');
        std::string sStart = seg.substr(0, dash);
        std::string sEnd = seg.substr(dash + 1);
        if (hardened && !sEnd.empty() && sEnd.back() == '\'') sEnd.pop_back();
        int start = 0, end = 0;
        try { start = std::stoi(sStart); end = std::stoi(sEnd); } catch(...) { return; }
        for (int i = start; i <= end; i++) {
            std::string nextSeg = std::to_string(i) + (hardened ? "'" : "");
            std::string nextPath = (currentPath.empty() ? "" : currentPath + "/") + nextSeg;
            expandPathRecursive(segments, index + 1, nextPath, results);
        }
    } else {
        std::string nextPath = (currentPath.empty() ? "" : currentPath + "/") + seg;
        expandPathRecursive(segments, index + 1, nextPath, results);
    }
}

std::vector<std::string> load_wordlist(const std::string& filepath) {
    std::vector<std::string> wl;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::string alt = "bip39/" + filepath;
        if (filepath.find(".txt") == std::string::npos) alt += ".txt";
        file.open(alt);
    }
    if (!file.is_open()) {
        std::cerr << "[ERROR] Could not open wordlist: " << filepath << "\n";
        return wl;
    }
    std::string line;
    while (std::getline(file, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) wl.push_back(line);
    }
    return wl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== SeedRecover v2.4 (Independent & Robust) ===\n";

    std::string langFile = "english.txt";
    std::string seedPhrase = "";
    std::string pathFile = "";
    bool hasBTC = true;
    bool hasETH = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--langs" && i + 1 < argc) langFile = argv[++i];
        else if (arg == "--seed" && i + 1 < argc) seedPhrase = argv[++i];
        else if (arg == "--path-file" && i + 1 < argc) pathFile = argv[++i];
        else if (arg == "--multi-coin" && i + 1 < argc) {
            std::string coins = argv[++i];
            std::transform(coins.begin(), coins.end(), coins.begin(), [](unsigned char c){ return std::toupper(c); });
            hasBTC = (coins.find("BTC") != std::string::npos);
            hasETH = (coins.find("ETH") != std::string::npos);
        }
    }

    if (seedPhrase.empty()) {
        std::cout << "Usage: SeedRecover.exe --langs <file> --seed \"...\" [--path-file list.txt]\n";
        return 1;
    }

    auto wl = load_wordlist(langFile);
    if (wl.empty()) return 1;
    std::cout << "[INFO] Loaded dictionary: " << wl.size() << " words.\n";

    uint8_t mSeed[64];
    if (!PKCS5_PBKDF2_HMAC(seedPhrase.c_str(), (int)seedPhrase.length(), (const unsigned char*)"mnemonic", 8, 2048, EVP_sha512(), 64, mSeed)) {
        std::cerr << "[ERROR] PBKDF2 failed.\n"; return 1;
    }

    struct PathDef { std::string label; std::string path; std::string coin; };
    std::vector<PathDef> paths;

    if (!pathFile.empty()) {
        std::ifstream pf(pathFile);
        if (pf.is_open()) {
            std::string line;
            while (std::getline(pf, line)) {
                line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                if (line.empty()) continue;
                std::vector<std::string> segments, expanded;
                std::stringstream ss(line); std::string seg;
                while (std::getline(ss, seg, '/')) segments.push_back(seg);
                expandPathRecursive(segments, 0, "", expanded);

                for (const auto& p : expanded) {
                    bool isEth = (p.find("60'") != std::string::npos);
                    bool isBtc = (p.find("44'/0'") != std::string::npos || p.find("49'/0'") != std::string::npos || p.find("84'/0'") != std::string::npos);
                    bool isGeneric = !isEth && !isBtc;

                    if (isEth && hasETH) paths.push_back({"ETH-Custom", p, "ETH"});
                    else if (isBtc && hasBTC) {
                        std::string lbl = "CUSTOM";
                        if (p.find("44'") != std::string::npos) lbl = "BIP44";
                        else if (p.find("49'") != std::string::npos) lbl = "BIP49";
                        else if (p.find("84'") != std::string::npos) lbl = "BIP84";
                        paths.push_back({lbl, p, "BTC"});
                    }
                    else if (isGeneric) {
                        if (hasBTC) {
                            paths.push_back({"BIP32", p, "BTC"});
                            paths.push_back({"BIP141", p, "BTC"}); 
                        }
                        if (hasETH) {
                            paths.push_back({"BIP32-ETH", p, "ETH"});
                        }
                    }
                }
            }
            pf.close();
        }
    } else {
        if (hasBTC) {
            paths.push_back({"BIP32", "m/0/0", "BTC"});
            paths.push_back({"BIP44", "m/44'/0'/0'/0/0", "BTC"});
            paths.push_back({"BIP49", "m/49'/0'/0'/0/0", "BTC"});
            paths.push_back({"BIP84", "m/84'/0'/0'/0/0", "BTC"});
            paths.push_back({"BIP141", "m/0/0", "BTC"});
        }
        if (hasETH) {
            paths.push_back({"BIP44-ETH", "m/44'/60'/0'/0/0", "ETH"});
            paths.push_back({"BIP32-ETH", "m/0/0", "ETH"});
        }
    }

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    std::cout << "\nTYPE                PATH                        ADDRESS                                      PRIVATE KEY (WIF/Hex)\n";
    std::cout << "----------------------------------------------------------------------------------------------------------------------------\n";

    for (const auto& p : paths) {
        std::vector<uint8_t> privKey;
        DerivePath(ctx, mSeed, 64, p.path, privKey);

        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, privKey.data())) continue;

        if (p.coin == "ETH") {
            uint8_t uPub[65]; size_t len = 65;
            secp256k1_ec_pubkey_serialize(ctx, uPub, &len, &pub, SECP256K1_EC_UNCOMPRESSED);
            std::vector<uint8_t> vPub(uPub, uPub + 65);
            std::string addr = EthUtils::PubKeyToEthAddress(vPub);
            std::string pk = ToHex(privKey);
            std::cout << std::left << std::setw(20) << (p.label + " [ETH]") << std::setw(28) << p.path << std::setw(45) << addr << pk << "\n";
        } 
        else {
            uint8_t cPub[33]; size_t clen = 33;
            secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
            std::vector<uint8_t> vPubC(cPub, cPub + 33);

            uint8_t uPub[65]; size_t ulen = 65;
            secp256k1_ec_pubkey_serialize(ctx, uPub, &ulen, &pub, SECP256K1_EC_UNCOMPRESSED);
            std::vector<uint8_t> vPubU(uPub, uPub + 65);

            struct BtcType { std::string name; std::string addr; std::string wif; };
            std::vector<BtcType> types;

            types.push_back({ "Bech32", PubKeyToNativeSegwit(vPubC), PrivKeyToWIF(privKey, true) });
            types.push_back({ "Comp P2PKH", PubKeyToLegacy(vPubC), PrivKeyToWIF(privKey, true) });
            types.push_back({ "Comp P2SH", PubKeyToNestedSegwit(vPubC), PrivKeyToWIF(privKey, true) });
            types.push_back({ "Uncomp P2PKH", PubKeyToLegacy(vPubU), PrivKeyToWIF(privKey, false) });

            for (const auto& t : types) {
                std::cout << std::left << std::setw(20) << (p.label + " [" + t.name + "]") 
                          << std::setw(28) << p.path 
                          << std::setw(45) << t.addr 
                          << t.wif << "\n";
            }
        }
    }

    std::cout << "----------------------------------------------------------------------------------------------------------------------------\n";
    secp256k1_context_destroy(ctx);
    return 0;
}