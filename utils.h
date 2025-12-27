#pragma once
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <cstdint>

// --- TIPURI DE DATE ---
typedef unsigned long long ulong;
typedef unsigned int uint;
typedef unsigned char uchar;

// Librării externe
#include <secp256k1.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>

// Networking Windows
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
    // Add any Linux-specific headers if needed, otherwise leave empty
    #include <unistd.h>
#include <wininet.h>
#pragma comment(lib, "wininet.lib")

// =============================================================
// SECTIUNEA 1: IMPLEMENTARE BECH32 (SegWit Native)
// =============================================================
namespace Bech32 {
    inline constexpr char const* charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
    
    inline uint32_t polymod(const std::vector<uint8_t>& values) {
        static const uint32_t GEN[5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
        uint32_t chk = 1;
        for (uint8_t v : values) {
            uint8_t b = chk >> 25;
            chk = ((chk & 0x1ffffff) << 5) ^ v;
            for (int i = 0; i < 5; ++i) {
                if ((b >> i) & 1) chk ^= GEN[i];
            }
        }
        return chk;
    }

    inline std::vector<uint8_t> expand_hrp(const std::string& hrp) {
        std::vector<uint8_t> ret;
        ret.reserve(hrp.size() * 2 + 1);
        for (char c : hrp) ret.push_back(static_cast<uint8_t>(c) >> 5);
        ret.push_back(0);
        for (char c : hrp) ret.push_back(static_cast<uint8_t>(c) & 31);
        return ret;
    }

    inline std::string encode(const std::string& hrp, const std::vector<uint8_t>& values) {
        std::vector<uint8_t> checksum_input = expand_hrp(hrp);
        checksum_input.insert(checksum_input.end(), values.begin(), values.end());
        checksum_input.resize(checksum_input.size() + 6, 0);
        uint32_t mod = polymod(checksum_input) ^ 1;
        std::string ret = hrp + "1";
        for (uint8_t v : values) ret += charset[v];
        for (int i = 0; i < 6; ++i) ret += charset[(mod >> (5 * (5 - i))) & 31];
        return ret;
    }

    inline bool convert_bits(const std::vector<uint8_t>& in, int fromBits, int toBits, bool pad, std::vector<uint8_t>& out) {
        int acc = 0; int bits = 0; 
        int maxv = (1 << toBits) - 1; 
        int max_acc = (1 << (fromBits + toBits - 1)) - 1;
        for (uint8_t value : in) {
            acc = ((acc << fromBits) | value) & max_acc;
            bits += fromBits;
            while (bits >= toBits) { 
                bits -= toBits; 
                out.push_back(static_cast<uint8_t>((acc >> bits) & maxv)); 
            }
        }
        if (pad) { 
            if (bits > 0) out.push_back(static_cast<uint8_t>((acc << (toBits - bits)) & maxv)); 
        }
        else if (bits >= fromBits || ((acc << (toBits - bits)) & maxv)) return false;
        return true;
    }

    inline std::string encode_segwit(const std::string& hrp, int witver, const std::vector<uint8_t>& program) {
        std::vector<uint8_t> data; 
        data.push_back(static_cast<uint8_t>(witver));
        convert_bits(program, 8, 5, true, data);
        return encode(hrp, data);
    }
}

// =============================================================
// SECTIUNEA 2: UTILITARE HASH, BASE58 & HEX
// =============================================================

inline std::string toHex(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : data) ss << std::setw(2) << (int)b;
    return ss.str();
}

inline void Hash160(const std::vector<uint8_t>& input, std::vector<uint8_t>& output) {
    unsigned char shaResult[SHA256_DIGEST_LENGTH];
    SHA256(input.data(), input.size(), shaResult);
    output.resize(20);
    RIPEMD160(shaResult, SHA256_DIGEST_LENGTH, output.data());
}

inline void DoubleSHA256(const std::vector<uint8_t>& input, std::vector<uint8_t>& output) {
    unsigned char hash1[SHA256_DIGEST_LENGTH];
    SHA256(input.data(), input.size(), hash1);
    output.resize(SHA256_DIGEST_LENGTH);
    SHA256(hash1, SHA256_DIGEST_LENGTH, output.data());
}

inline std::string EncodeBase58Check(const std::vector<uint8_t>& input) {
    static const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::vector<uint8_t> vch(input);
    std::vector<uint8_t> hash;
    DoubleSHA256(vch, hash);
    vch.insert(vch.end(), hash.begin(), hash.begin() + 4);
    
    std::vector<uint8_t> digits((vch.size() * 138 / 100) + 1, 0);
    size_t digitslen = 1;
    for (size_t i = 0; i < vch.size(); i++) {
        uint32_t carry = static_cast<uint32_t>(vch[i]);
        for (size_t j = 0; j < digitslen; j++) {
            carry += static_cast<uint32_t>(digits[j]) << 8;
            digits[j] = static_cast<uint8_t>(carry % 58);
            carry /= 58;
        }
        while (carry > 0) { 
            digits[digitslen++] = static_cast<uint8_t>(carry % 58); 
            carry /= 58; 
        }
    }
    std::string result = "";
    for (size_t i = 0; i < vch.size() && vch[i] == 0; i++) result += '1';
    for (size_t i = 0; i < digitslen; i++) result += pszBase58[digits[digitslen - 1 - i]];
    return result;
}

// =============================================================
// SECTIUNEA 3: CRYPTO CORE (BIP32)
// =============================================================

struct ExtendedKey {
    std::vector<uint8_t> key; 
    std::vector<uint8_t> chainCode; 
    int depth = 0; 
    uint32_t childIndex = 0;
};

inline void HmacSha512(const std::vector<uint8_t>& key, const std::vector<uint8_t>& data, std::vector<uint8_t>& output) {
    output.resize(64); 
    unsigned int len = 64;
    HMAC(EVP_sha512(), key.data(), static_cast<int>(key.size()), data.data(), data.size(), output.data(), &len);
}

inline ExtendedKey GetMasterKey(const std::vector<uint8_t>& seed) {
    std::vector<uint8_t> output; 
    const std::string salt = "Bitcoin seed";
    std::vector<uint8_t> saltBytes(salt.begin(), salt.end());
    HmacSha512(saltBytes, seed, output);
    ExtendedKey k; 
    k.key.assign(output.begin(), output.begin() + 32); 
    k.chainCode.assign(output.begin() + 32, output.end());
    k.depth = 0; 
    k.childIndex = 0;
    return k;
}

inline ExtendedKey Derive(const ExtendedKey& parent, uint32_t index, secp256k1_context* ctx) {
    std::vector<uint8_t> data;
    if (index >= 0x80000000) {
        data.push_back(0x00); 
        data.insert(data.end(), parent.key.begin(), parent.key.end());
    } else {
        secp256k1_pubkey pub;
        if (secp256k1_ec_pubkey_create(ctx, &pub, parent.key.data())) {
            uint8_t comp[33]; 
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(ctx, comp, &len, &pub, SECP256K1_EC_COMPRESSED);
            data.insert(data.end(), comp, comp + 33);
        }
    }
    data.push_back(static_cast<uint8_t>((index >> 24) & 0xFF)); 
    data.push_back(static_cast<uint8_t>((index >> 16) & 0xFF));
    data.push_back(static_cast<uint8_t>((index >> 8) & 0xFF)); 
    data.push_back(static_cast<uint8_t>(index & 0xFF));
    
    std::vector<uint8_t> i_out; 
    HmacSha512(parent.chainCode, data, i_out);
    ExtendedKey child; 
    child.key.assign(i_out.begin(), i_out.begin() + 32);
    secp256k1_ec_privkey_tweak_add(ctx, child.key.data(), parent.key.data());
    child.chainCode.assign(i_out.begin() + 32, i_out.end()); 
    child.depth = parent.depth + 1; 
    child.childIndex = index;
    return child;
}

// =============================================================
// SECTIUNEA 4: CONVERTOARE ADRESE
// =============================================================

inline std::string PubKeyToLegacy(const std::vector<uint8_t>& pubKeyBytes) {
    std::vector<uint8_t> keyHash; Hash160(pubKeyBytes, keyHash);
    std::vector<uint8_t> addr; 
    addr.push_back(0x00); 
    addr.insert(addr.end(), keyHash.begin(), keyHash.end());
    return EncodeBase58Check(addr);
}

inline std::string PubKeyToNestedSegwit(const std::vector<uint8_t>& pubKeyBytes) {
    std::vector<uint8_t> keyHash; Hash160(pubKeyBytes, keyHash);
    std::vector<uint8_t> redeem; 
    redeem.push_back(0x00); 
    redeem.push_back(0x14); 
    redeem.insert(redeem.end(), keyHash.begin(), keyHash.end());
    std::vector<uint8_t> scriptHash; Hash160(redeem, scriptHash);
    std::vector<uint8_t> addr; 
    addr.push_back(0x05); 
    addr.insert(addr.end(), scriptHash.begin(), scriptHash.end());
    return EncodeBase58Check(addr);
}

inline std::string PubKeyToNativeSegwit(const std::vector<uint8_t>& pubKeyBytes) {
    std::vector<uint8_t> keyHash; Hash160(pubKeyBytes, keyHash);
    return Bech32::encode_segwit("bc", 0, keyHash);
}

// =============================================================
// SECTIUNEA 5: ONLINE API CHECKER
// =============================================================

struct OnlineInfo {
    std::string totalReceived = "0";
    std::string txCount = "0";
    bool success = false;
    std::string source = "Blockchain.info"; 
};

inline OnlineInfo CheckAddressOnline(const std::string& address) {
    OnlineInfo info;
    HINTERNET hInt = InternetOpenA("GpuCracker/4.0", INTERNET_OPEN_TYPE_DIRECT, NULL, NULL, 0);
    if (!hInt) return info;

    std::string url = "https://blockchain.info/rawaddr/" + address;
    HINTERNET hFile = InternetOpenUrlA(hInt, url.c_str(), NULL, 0, INTERNET_FLAG_RELOAD | INTERNET_FLAG_SECURE, 0);
    if (!hFile) {
        InternetCloseHandle(hInt);
        return info;
    }

    std::string response;
    char buffer[4096];
    DWORD bytesRead;
    while (InternetReadFile(hFile, buffer, sizeof(buffer), &bytesRead) && bytesRead > 0) {
        response.append(buffer, bytesRead);
    }
    
    InternetCloseHandle(hFile);
    InternetCloseHandle(hInt);
    if (response.empty()) return info;

    auto extract = [&](const std::string& key) -> std::string {
        size_t pos = response.find(key);
        if (pos == std::string::npos) return "0";
        size_t start = response.find(":", pos) + 1;
        size_t end = response.find_first_of(",}", start);
        std::string val = response.substr(start, end - start);
        val.erase(std::remove(val.begin(), val.end(), ' '), val.end());
        return val;
    };

    info.totalReceived = extract("\"total_received\"");
    info.txCount = extract("\"n_tx\"");
    info.success = true;
    return info;
}