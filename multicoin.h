#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <map>

// Librarii externe
#include <openssl/sha.h>
#include <openssl/ripemd.h>

class MultiCoin {
private:
    // --- KECCAK-256 (ETH) ---
    static void keccakf(uint64_t st[25]) {
        const int rhoOffsets[25] = { 0, 1, 62, 28, 27, 36, 44, 6, 55, 20, 3, 10, 43, 25, 39, 41, 45, 15, 21, 8, 18, 2, 61, 56, 14 };
        const uint64_t RNDC[24] = {
            0x0000000000000001, 0x0000000000008082, 0x800000000000808a, 0x8000000080008000,
            0x000000000000808b, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
            0x000000000000008a, 0x0000000000000088, 0x0000000080008009, 0x000000008000000a,
            0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
            0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
            0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
        };
        for (int round = 0; round < 24; round++) {
            uint64_t bc[5];
            for (int i = 0; i < 5; ++i) bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
            for (int i = 0; i < 5; ++i) {
                uint64_t t = bc[(i + 4) % 5] ^ ((bc[(i + 1) % 5] << 1) | (bc[(i + 1) % 5] >> 63));
                for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
            }
            uint64_t t = st[1], bc0;
            for (int i = 0; i < 24; ++i) {
                int j = rhoOffsets[i + 1];
                bc0 = st[j];
                st[j] = (t << rhoOffsets[i + 1]) | (t >> (64 - rhoOffsets[i + 1]));
                t = bc0;
            }
            for (int j = 0; j < 25; j += 5) {
                for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
                for (int i = 0; i < 5; ++i) st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
            st[0] ^= RNDC[round];
        }
    }

    static void Keccak256(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
        uint64_t st[25]; memset(st, 0, sizeof(st));
        size_t rsiz = 136; size_t offset = 0; size_t len = in.size();
        while (len >= rsiz) {
            for (size_t i = 0; i < rsiz / 8; i++) {
                uint64_t word = 0;
                for(int k=0; k<8; k++) word |= ((uint64_t)in[offset + i * 8 + k]) << (8*k);
                st[i] ^= word;
            }
            keccakf(st); offset += rsiz; len -= rsiz;
        }
        uint64_t lastWord = 0;
        for (size_t i = 0; i < len; i++) lastWord |= ((uint64_t)in[offset + i]) << (8 * i);
        st[len / 8] ^= lastWord;
        st[len / 8] ^= ((uint64_t)0x01) << (8 * (len % 8)); 
        st[(rsiz - 1) / 8] ^= ((uint64_t)0x80) << 56;       
        keccakf(st);
        out.resize(32);
        for (int i = 0; i < 4; i++) {
            for (int k = 0; k < 8; k++) out[i * 8 + k] = (st[i] >> (8 * k)) & 0xFF;
        }
    }

    static void Hash160(const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
        uint8_t sha[SHA256_DIGEST_LENGTH];
        SHA256(in.data(), in.size(), sha);
        out.resize(RIPEMD160_DIGEST_LENGTH);
        RIPEMD160(sha, SHA256_DIGEST_LENGTH, out.data());
    }

    static std::string EncodeBase58Check(const std::vector<uint8_t>& payload, const std::vector<uint8_t>& prefix) {
        std::vector<uint8_t> vch = prefix;
        vch.insert(vch.end(), payload.begin(), payload.end());
        uint8_t hash1[SHA256_DIGEST_LENGTH], hash2[SHA256_DIGEST_LENGTH];
        SHA256(vch.data(), vch.size(), hash1); SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
        vch.push_back(hash2[0]); vch.push_back(hash2[1]); vch.push_back(hash2[2]); vch.push_back(hash2[3]);
        static const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        int zeros = 0;
        const unsigned char* pbegin = vch.data();
        const unsigned char* pend = vch.data() + vch.size();
        while (pbegin != pend && *pbegin == 0) { pbegin++; zeros++; }
        int size = (pend - pbegin) * 138 / 100 + 1;
        std::vector<unsigned char> b58(size);
        std::vector<unsigned char>::iterator it = b58.begin(); *it = 0;
        while (pbegin != pend) {
            int carry = *pbegin;
            for (std::vector<unsigned char>::iterator i = b58.begin(); i != it + 1; ++i) {
                carry += 256 * (*i); *i = carry % 58; carry /= 58;
            }
            while (carry != 0) { it++; *it = carry % 58; carry /= 58; }
            pbegin++;
        }
        std::string str; str.reserve(zeros + (it - b58.begin() + 1));
        str.assign(zeros, '1');
        for (std::vector<unsigned char>::reverse_iterator i = b58.rbegin() + (b58.size() - 1 - (it - b58.begin())); i != b58.rend(); ++i)
            str += pszBase58[*i];
        return str;
    }

    // --- BECH32 IMPLEMENTATION (GENERIC) ---
    static std::string Bech32Encode(const std::string& hrp, const std::vector<uint8_t>& witprog) {
        static const char* charset = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";
        std::vector<int> values;
        
        // Convert 8-bit to 5-bit
        int acc = 0, bits = 0;
        for (uint8_t v : witprog) {
            acc = (acc << 8) | v;
            bits += 8;
            while (bits >= 5) {
                values.push_back((acc >> (bits - 5)) & 31);
                bits -= 5;
            }
        }
        if (bits > 0) values.push_back((acc << (5 - bits)) & 31);

        // Witness Version 0 (Insert at start)
        values.insert(values.begin(), 0);

        // Checksum calculation
        uint32_t chk = 1;
        auto polymod_step = [&](uint32_t v) {
            uint32_t b = chk >> 25;
            chk = ((chk & 0x1ffffff) << 5) ^ v;
            static const uint32_t gen[] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
            for (int i = 0; i < 5; ++i) if ((b >> i) & 1) chk ^= gen[i];
        };

        // HRP Expand
        for (char c : hrp) polymod_step(c >> 5);
        polymod_step(0);
        for (char c : hrp) polymod_step(c & 31);

        // Data part
        for (int v : values) polymod_step(v);

        // Finalize
        for (int i = 0; i < 6; ++i) polymod_step(0);
        chk ^= 1;

        std::string ret = hrp + "1";
        for (int v : values) ret += charset[v];
        for (int i = 0; i < 6; ++i) ret += charset[(chk >> ((5 - i) * 5)) & 31];
        return ret;
    }

public:
    // ========================================================================
    // GENERARE ADRESE PENTRU DIVERSE MONEDE
    // ========================================================================

    // 1. LEGACY (P2PKH) - Starts with 1 (BTC), L (LTC), D (DOGE), etc.
    static std::string GenLegacy(const std::vector<uint8_t>& pubKey, const std::string& coin) {
        std::vector<uint8_t> h160; Hash160(pubKey, h160);
        std::vector<uint8_t> prefix = {0x00}; // Default BTC

        std::string c = coin;
        std::transform(c.begin(), c.end(), c.begin(), ::tolower);

        if (c == "btc" || c == "bch" || c == "bsv") prefix = {0x00};
        else if (c == "ltc") prefix = {0x30};  // 'L'
        else if (c == "doge") prefix = {0x1E}; // 'D'
        else if (c == "dash") prefix = {0x4C}; // 'X'
        else if (c == "btg") prefix = {38};    // 'G'
        else if (c == "zec" || c == "zcash") prefix = {0x1C, 0xB8}; // 't1'

        return EncodeBase58Check(h160, prefix);
    }

    // 2. P2SH (Segwit Compatible) - Starts with 3 (BTC), M (LTC), etc.
    static std::string GenP2SH(const std::vector<uint8_t>& pubKey, const std::string& coin) {
        std::vector<uint8_t> pubHash; Hash160(pubKey, pubHash);
        std::vector<uint8_t> script = { 0x00, 0x14 }; // 0x0014 = P2WPKH PUSH
        script.insert(script.end(), pubHash.begin(), pubHash.end());
        std::vector<uint8_t> scriptHash; Hash160(script, scriptHash);

        std::vector<uint8_t> prefix = {0x05}; // Default BTC

        std::string c = coin;
        std::transform(c.begin(), c.end(), c.begin(), ::tolower);

        if (c == "btc" || c == "bch" || c == "bsv" || c == "btg") prefix = {0x05}; // '3'
        else if (c == "ltc") prefix = {0x32};  // 'M' (Standard modern LTC P2SH)
        else if (c == "doge") prefix = {0x16}; // '9' or 'A' (Multisig)
        else if (c == "dash") prefix = {0x10}; // '7'
        else if (c == "zec" || c == "zcash") prefix = {0x1C, 0xBD}; // 't3'

        return EncodeBase58Check(scriptHash, prefix);
    }

    // 3. BECH32 (Native Segwit) - Starts with bc1 (BTC), ltc1 (LTC)
    static std::string GenBech32(const std::vector<uint8_t>& pubKey, const std::string& coin) {
        std::string c = coin;
        std::transform(c.begin(), c.end(), c.begin(), ::tolower);

        // DOGE, DASH, ZCASH usually don't use standard BIP173 Bech32 or have different implementation
        // We support BTC, LTC, BTG, BCH (uses CashAddr - similar but diff constants), BSV.
        // For simplicity/compatibility we primarily implement BTC/LTC/BTG here.
        
        std::string hrp = "bc";
        if (c == "ltc") hrp = "ltc";
        else if (c == "btg") hrp = "btg";
        else if (c == "bch") return "BCH_CASHADDR_REQ"; // BCH uses CashAddr
        else if (c != "btc") return ""; // Not supported or standard

        std::vector<uint8_t> h160; Hash160(pubKey, h160);
        return Bech32Encode(hrp, h160);
    }

    // 4. ETHEREUM (Address)
    static std::string GenEth(const std::vector<uint8_t>& pubKey) {
        std::vector<uint8_t> toHash = pubKey;
        if (toHash.size() == 65 && toHash[0] == 0x04) toHash.erase(toHash.begin()); // Remove uncomp prefix
        std::vector<uint8_t> hash;
        Keccak256(toHash, hash);
        std::stringstream ss;
        ss << "0x" << std::hex << std::setfill('0');
        for (size_t i = 12; i < 32; i++) ss << std::setw(2) << (int)hash[i];
        return ss.str();
    }
};