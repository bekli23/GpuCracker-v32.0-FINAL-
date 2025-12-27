#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <map>
#include <cmath>

// Includem headerele proiectului
#include "akm.h"
#include "utils.h" 
#include <secp256k1.h>

// --- HELPER FUNCTIONS ---

std::string BytesToHex(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < data.size(); ++i) {
        ss << std::setw(2) << (int)data[i];
    }
    return ss.str();
}

std::string ToHexStr(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << (int)data[i];
    }
    return ss.str();
}

// === LOGICA PENTRU RANGE ===
void applyBitMask(std::vector<uint8_t>& key, int bits) {
    if (bits <= 0 || bits >= 256) return;
    int topBitIndex = bits - 1; 
    int byteIndex = 31 - (topBitIndex / 8); 
    int bitInByte = topBitIndex % 8;

    for (int i = 0; i < byteIndex; ++i) key[i] = 0;
    unsigned char mask = (1 << (bitInByte + 1)) - 1;
    key[byteIndex] &= mask;
    key[byteIndex] |= (1 << bitInByte);
}

// === LOGICA: SCHEMATIC DECODE (Words -> BigInt Key) ===
std::vector<uint8_t> DecodeSchematic(const std::vector<std::string>& phraseWords, AkmLogic& akm) {
    std::vector<uint8_t> resultKey(32, 0);
    
    // 1. Obtinem Wordlist-ul si construim harta inversa (Cuvant -> Index)
    const std::vector<std::string>& wordList = akm.getWordList();
    std::map<std::string, int> wordMap;
    for(size_t i = 0; i < wordList.size(); ++i) {
        wordMap[wordList[i]] = (int)i;
    }
    
    size_t base = wordList.size();
    if (base == 0) return resultKey; 

    // Calculam valoarea: Val = w0 * base^0 + w1 * base^1 ... (Big Int logic)
    for (size_t i = 0; i < phraseWords.size(); ++i) {
        std::string w = phraseWords[i];
        if (wordMap.find(w) == wordMap.end()) {
            std::cerr << "[ERR] Word not found in list: " << w << "\n";
            continue;
        }
        int idx = wordMap[w];

        // Multiply Result by Base
        unsigned long long carry = 0;
        for (int k = 31; k >= 0; --k) {
            unsigned long long val = (unsigned long long)resultKey[k] * base + carry;
            resultKey[k] = (uint8_t)(val & 0xFF);
            carry = val >> 8;
        }

        // Add Index
        carry = idx;
        for (int k = 31; k >= 0; --k) {
            unsigned long long val = (unsigned long long)resultKey[k] + carry;
            resultKey[k] = (uint8_t)(val & 0xFF);
            carry = val >> 8;
            if (carry == 0) break;
        }
    }
    
    return resultKey;
}

// Helper pentru a alege wordlist-ul default in functie de profil
std::string GetDefaultWordlist(const std::string& profile) {
    // Profilurile clasice AKM folosesc wordlist_512_ascii
    if (profile.find("akm") != std::string::npos || profile == "auto-linear") {
        return "akm\\wordlist_512_ascii.txt";
    }
    // Daca e un profil BIP39 standard (exemplu)
    if (profile == "bip39") {
        return "akm\\wordlist_2048_english.txt";
    }
    // Default fallback
    return "akm\\wordlist_512_ascii.txt";
}

void printUsage() {
    std::cout << "AKM Seed to Private Key Converter v3.5 (Profile Support)\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << "Usage:\n";
    std::cout << "  akm_seed2priv.exe --phrase \"word1 ...\" --profile <name> [options]\n";
    std::cout << "\nRequired:\n";
    std::cout << "  --phrase \"...\" : The seed phrase to convert.\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --profile <name> : Set AKM Profile Logic. Default: akm3-puzzle71\n";
    std::cout << "                     Available: auto-linear, akm2-core, akm2-lab-v1,\n";
    std::cout << "                                akm2-fixed123-pack-v1, akm3-puzzle71\n";
    std::cout << "  --wordlist <file>: Override the default wordlist file.\n";
    std::cout << "  --mode <type>    : 'schematic' (default) or 'hash'.\n";
    std::cout << "  --akm-bit <n>    : Apply range mask (2^n-1 .. 2^n).\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(); return 1; }

    std::string phraseStr = "";
    std::string filename = "";
    std::string mode = "schematic"; 
    std::string profile = "akm3-puzzle71"; // Profil default
    std::string wordlistPath = ""; // Daca e gol, se decide automat
    int targetBits = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") { printUsage(); return 0; }
        else if (arg == "--phrase" && i + 1 < argc) phraseStr = argv[++i];
        else if (arg == "--file" && i + 1 < argc) filename = argv[++i];
        else if (arg == "--akm-bit" && i + 1 < argc) targetBits = std::stoi(argv[++i]);
        else if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--profile" && i + 1 < argc) profile = argv[++i];
        else if (arg == "--wordlist" && i + 1 < argc) wordlistPath = argv[++i];
    }

    // 1. Citire Fraza
    if (!filename.empty()) {
        std::ifstream f(filename);
        if (f.is_open()) {
            std::getline(f, phraseStr);
            f.close();
            // Trim whitespace
            phraseStr.erase(0, phraseStr.find_first_not_of(" \t\r\n"));
            phraseStr.erase(phraseStr.find_last_not_of(" \t\r\n") + 1);
        } else {
            std::cerr << "Error: Could not open file " << filename << "\n";
            return 1;
        }
    }
    if (phraseStr.empty()) { std::cerr << "Error: No phrase provided.\n"; return 1; }

    // 2. Initializare AKM cu Profilul specific
    if (wordlistPath.empty()) {
        wordlistPath = GetDefaultWordlist(profile);
    }

    std::cout << "[INIT] Loading Profile : " << profile << "\n";
    std::cout << "[INIT] Loading Wordlist: " << wordlistPath << "\n";

    AkmLogic akm;
    
    // --- FIX AICI ---
    // Am eliminat 'if (!akm.init...)' deoarece init returneaza void
    akm.init(profile, wordlistPath);
    // ----------------

    // 3. Parsare cuvinte
    std::stringstream ss(phraseStr);
    std::string segment;
    std::vector<std::string> words;
    while (std::getline(ss, segment, ' ')) {
        if (!segment.empty()) words.push_back(segment);
    }

    // 4. Decodare in Private Key
    std::vector<uint8_t> privKey;

    if (mode == "schematic") {
        std::cout << "[INFO] Mode: Schematic (Base-N Calculation)\n";
        // Schematic foloseste matematica pura pe indecsi
        privKey = DecodeSchematic(words, akm);
    } else {
        std::cout << "[INFO] Mode: Hash (Profile internal logic)\n";
        // Aici AKM foloseste logica specifica profilului
        privKey = akm.phrase_to_key(words);
    }

    // 5. Masca de biti (Range)
    if (targetBits > 0) {
        std::cout << "[INFO] Applying Range Mask: 2^" << (targetBits - 1) << " .. 2^" << targetBits << "\n";
        applyBitMask(privKey, targetBits);
    }

    // 6. Generare PubKey si Adrese
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pubkey;

    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, privKey.data())) {
        std::cerr << "Error creating public key (Invalid Private Key generated).\n";
        secp256k1_context_destroy(ctx);
        return 1;
    }

    uint8_t cPub[33]; size_t len = 33;
    secp256k1_ec_pubkey_serialize(ctx, cPub, &len, &pubkey, SECP256K1_EC_COMPRESSED);
    std::vector<uint8_t> vPubComp(cPub, cPub + 33);

    uint8_t uPub[65]; size_t lenU = 65;
    secp256k1_ec_pubkey_serialize(ctx, uPub, &lenU, &pubkey, SECP256K1_EC_UNCOMPRESSED);
    std::vector<uint8_t> vPubUncomp(uPub, uPub + 65);

    std::string p2pkh_c = PubKeyToLegacy(vPubComp);
    std::string p2pkh_u = PubKeyToLegacy(vPubUncomp);
    std::string p2sh = PubKeyToNestedSegwit(vPubComp);
    std::string bech32 = PubKeyToNativeSegwit(vPubComp);

    std::cout << "\n================================================================\n";
    std::cout << "PROFILE: " << profile << "\n";
    std::cout << "PHRASE : " << phraseStr << "\n";
    std::cout << "PRIV   : " << ToHexStr(privKey.data(), privKey.size()) << "\n";
    std::cout << "PUB (c): " << BytesToHex(vPubComp) << "\n";
    std::cout << "\n-- ADDRESSES (MAINNET) --\n";
    std::cout << std::left << std::setw(20) << "P2PKH (Comp)" << ": " << p2pkh_c << "\n";
    std::cout << std::left << std::setw(20) << "P2PKH (Uncomp)" << ": " << p2pkh_u << "\n";
    std::cout << std::left << std::setw(20) << "P2SH (Segwit)" << ": " << p2sh << "\n";
    std::cout << std::left << std::setw(20) << "Bech32" << ": " << bech32 << "\n";
    std::cout << "================================================================\n";

    secp256k1_context_destroy(ctx);
    return 0;
}