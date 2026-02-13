#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <map>
#include <bitset>
#include <sstream>
#include <cstring>
#include <random>
#include <csignal> 
#include <mutex>

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <secp256k1.h>

// --- FIX FOR ETHUTILS REDEFINITION ---
#define EthUtils EthUtils_Renamed_To_Avoid_Conflict
#include "multicoin.h" 
#undef EthUtils
// -------------------------------------

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <afunix.h> 
    #pragma comment(lib, "ws2_32.lib")
    typedef SOCKET SocketHandle;
    #define INVALID_SOCKET_VAL INVALID_SOCKET
    #define CLOSE_SOCKET(s) closesocket(s)
    #define POPFEN _popen
    #define PCLOSE _pclose
#else
    #include <sys/socket.h>
    #include <sys/un.h>
    #include <unistd.h>
    #include <fcntl.h>
    typedef int SocketHandle;
    #define INVALID_SOCKET_VAL -1
    #define CLOSE_SOCKET(s) close(s)
    #define POPFEN popen
    #define PCLOSE pclose
#endif

#include "mnemonic.h"
#include "args.h"
#include "bloom.h"
#include "utils.h"

// --- FIX FOR BytesToHex IDENTIFIER NOT FOUND ---
inline std::string BytesToHex(const std::vector<uint8_t>& bytes) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) {
        ss << std::setw(2) << (int)b;
    }
    return ss.str();
}
// -----------------------------------------------

#ifndef __CUDACC__

static std::atomic<bool>* globalRunningPtr = nullptr;

void handleSigInt(int s) {
    if (globalRunningPtr) *globalRunningPtr = false;
}

// ============================================================================
// CLEVER AI - MARKOV CHAIN LEARNING
// ============================================================================
#define NGRAM_SIZE 3 

class CleverAI {
private:
    std::map<std::string, std::vector<char>> chain;
    std::vector<std::string> startGrams; 
    std::mt19937 rng;
    std::mutex aiMutex;
    bool isTrained = false;

    std::string memoryFile = "ai_memory.dat";
    std::string inputFile = "word_ai.txt";

public:
    CleverAI() {
        rng.seed((unsigned int)std::chrono::steady_clock::now().time_since_epoch().count());
        loadMemory();     // Incarca "creierul" existent
        trainFromInput(); // Invata din textul nou
    }

    void train(const std::string& text) {
        if (text.length() < NGRAM_SIZE) return;
        startGrams.push_back(text.substr(0, NGRAM_SIZE));
        for (size_t i = 0; i < text.length() - NGRAM_SIZE; i++) {
            std::string key = text.substr(i, NGRAM_SIZE);
            char nextChar = text[i + NGRAM_SIZE];
            chain[key].push_back(nextChar);
        }
        isTrained = true;
    }

    void trainFromInput() {
        std::ifstream f(inputFile);
        if (!f.is_open()) return;
        std::string line;
        int count = 0;
        while (std::getline(f, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if (line.length() > NGRAM_SIZE) {
                train(line);
                count++;
            }
        }
        f.close();
        if (count > 0) saveMemory();
    }

    void saveMemory() {
        std::lock_guard<std::mutex> lock(aiMutex);
        std::ofstream f(memoryFile, std::ios::binary);
        if (!f.is_open()) return;
        size_t mapSize = chain.size();
        f.write((char*)&mapSize, sizeof(mapSize));
        for (const auto& pair : chain) {
            size_t keyLen = pair.first.length();
            f.write((char*)&keyLen, sizeof(keyLen));
            f.write(pair.first.c_str(), keyLen);
            size_t vecSize = pair.second.size();
            f.write((char*)&vecSize, sizeof(vecSize));
            f.write(pair.second.data(), vecSize);
        }
        size_t startSize = startGrams.size();
        f.write((char*)&startSize, sizeof(startSize));
        for(const auto& s : startGrams) {
            size_t sLen = s.length();
            f.write((char*)&sLen, sizeof(sLen));
            f.write(s.c_str(), sLen);
        }
        f.close();
    }

    void loadMemory() {
        std::ifstream f(memoryFile, std::ios::binary);
        if (!f.is_open()) return;
        size_t mapSize;
        if(f.read((char*)&mapSize, sizeof(mapSize))) {
            for (size_t i = 0; i < mapSize; i++) {
                size_t keyLen; f.read((char*)&keyLen, sizeof(keyLen));
                std::string key(keyLen, ' '); f.read(&key[0], keyLen);
                size_t vecSize; f.read((char*)&vecSize, sizeof(vecSize));
                std::vector<char> vals(vecSize); f.read(vals.data(), vecSize);
                chain[key] = vals;
            }
        }
        size_t startSize;
        if(f.read((char*)&startSize, sizeof(startSize))) {
            for(size_t i=0; i<startSize; i++) {
                size_t sLen; f.read((char*)&sLen, sizeof(sLen));
                std::string s(sLen, ' '); f.read(&s[0], sLen);
                startGrams.push_back(s);
            }
        }
        if (!chain.empty()) isTrained = true;
        f.close();
    }

    std::string generate() {
        // Fallback daca nu a invatat nimic
        if (!isTrained || startGrams.empty()) {
            const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            std::string res; 
            for(int i=0; i<12; i++) res += charset[rng() % 62];
            return res;
        }

        std::string currentGram = startGrams[rng() % startGrams.size()];
        std::string result = currentGram;

        // Generam pana la 32 caractere
        for (int i = 0; i < 32; i++) {
            
            // 5% sansa sa adauge un spatiu, daca ultimul caracter nu e deja spatiu
            if ((rng() % 100 < 5) && result.back() != ' ') {
                result += ' ';
                // Resetam gram-ul curent alegand un nou inceput aleatoriu
                currentGram = startGrams[rng() % startGrams.size()];
                result += currentGram;
                continue;
            }

            if (chain.find(currentGram) == chain.end()) break; 
            const std::vector<char>& possibilities = chain[currentGram];
            if (possibilities.empty()) break;
            
            char nextChar = possibilities[rng() % possibilities.size()];
            result += nextChar;
            currentGram = currentGram.substr(1) + nextChar;
        }
        return result;
    }
};

// ============================================================================
// BRAINWALLET GENERATOR (STANDARD / RANDOM)
// ============================================================================
class BrainwalletGen {
private:
    std::mt19937 rng;
public:
    BrainwalletGen() {
        rng.seed((unsigned int)std::chrono::steady_clock::now().time_since_epoch().count());
    }

    std::string generate(const std::string& mode) {
        if (mode == "alpha") {
            const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            int len = 8 + (rng() % 9); 
            std::string res; res.reserve(len);
            for (int i = 0; i < len; ++i) res += charset[rng() % (sizeof(charset) - 1)];
            return res;
        }
        else if (mode == "num") {
            const char charset[] = "0123456789";
            int len = 4 + (rng() % 5);
            std::string res; res.reserve(len);
            for (int i = 0; i < len; ++i) res += charset[rng() % (sizeof(charset) - 1)];
            return res;
        }
        else if (mode == "hex") {
            const char charset[] = "0123456789abcdef";
            std::string res; res.reserve(32);
            for (int i = 0; i < 32; ++i) res += charset[rng() % 16];
            return res;
        }
        else if (mode == "schematic") {
            std::string base;
            if (rng() % 2 == 0) { 
                char c1 = 'a' + (rng() % 26);
                char c2 = 'a' + (rng() % 26);
                base += c1; base += c2; 
            } else { 
                base += std::to_string(rng() % 10);
                base += std::to_string(rng() % 10); 
            }
            std::string res = "";
            int repeats = 3 + (rng() % 4); 
            for(int i=0; i<repeats; ++i) res += base;
            return res;
        }
        else { 
            const std::string charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ";
            int len = 8 + (rng() % 24); 
            std::string res; res.reserve(len);
            for (int i = 0; i < len; ++i) {
                res += charset[rng() % charset.length()];
            }
            return res;
        }
    }
};

class EntropyConverter {
private:
    static int hexCharToInt(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return 0;
    }
    static char intToHexChar(int v) {
        if (v >= 0 && v <= 9) return v + '0';
        if (v >= 10 && v <= 15) return v + 'A' - 10;
        return '0';
    }
    static int divModHex(std::string &hexNum, int divisor) {
        std::string quotient = "";
        int remainder = 0;
        bool leadingZeros = true;
        for (char c : hexNum) {
            int currentVal = hexCharToInt(c);
            int value = remainder * 16 + currentVal;
            int digit = value / divisor;
            remainder = value % divisor;
            if (digit > 0 || !leadingZeros) {
                quotient += intToHexChar(digit);
                leadingZeros = false;
            }
        }
        if (quotient == "") quotient = "0";
        hexNum = quotient;
        return remainder;
    }
    static bool isZero(const std::string& hexNum) {
        for (char c : hexNum) if (c != '0') return false;
        return true;
    }

public:
    static std::string toBinary(std::string hex) {
        std::string bin = "";
        for (char c : hex) {
            switch (toupper(c)) {
                case '0': bin+="0000"; break; case '1': bin+="0001"; break;
                case '2': bin+="0010"; break; case '3': bin+="0011"; break;
                case '4': bin+="0100"; break; case '5': bin+="0101"; break;
                case '6': bin+="0110"; break; case '7': bin+="0111"; break;
                case '8': bin+="1000"; break; case '9': bin+="1001"; break;
                case 'A': bin+="1010"; break; case 'B': bin+="1011"; break;
                case 'C': bin+="1100"; break; case 'D': bin+="1101"; break;
                case 'E': bin+="1110"; break; case 'F': bin+="1111"; break;
            }
        }
        return bin;
    }
    static std::string toBase10(std::string hex) {
        std::string res = "";
        if (isZero(hex)) return "0";
        while (!isZero(hex)) {
            int rem = divModHex(hex, 10);
            res += std::to_string(rem);
        }
        std::reverse(res.begin(), res.end());
        return res;
    }
    static std::string toDice(std::string hex) {
        std::string res = "";
        if (isZero(hex)) return "";
        while (!isZero(hex)) {
            int rem = divModHex(hex, 6);
            res += std::to_string(rem + 1);
        }
        std::reverse(res.begin(), res.end());
        return res;
    }
    static std::string toBase6(std::string hex) {
        std::string res = "";
        if (isZero(hex)) return "0";
        while (!isZero(hex)) {
            int rem = divModHex(hex, 6);
            res += std::to_string(rem);
        }
        std::reverse(res.begin(), res.end());
        return res;
    }
    static std::string toCards(std::string hex) {
        const char* RANKS[] = {"A","2","3","4","5","6","7","8","9","T","J","Q","K"};
        const char* SUITS[] = {"C","D","H","S"}; 
        std::string res = "";
        if (isZero(hex)) return "";
        while (!isZero(hex)) {
            int rem = divModHex(hex, 52);
            int suitIdx = rem / 13;
            int rankIdx = rem % 13;
            std::string card = std::string(RANKS[rankIdx]) + std::string(SUITS[suitIdx]);
            if (!res.empty()) res += " ";
            res += card;
        }
        std::stringstream ss(res);
        std::string segment;
        std::vector<std::string> seglist;
        while(std::getline(ss, segment, ' ')) seglist.push_back(segment);
        std::reverse(seglist.begin(), seglist.end());
        std::string finalRes = "";
        for(size_t i=0; i<seglist.size(); ++i) {
            finalRes += seglist[i];
            if(i != seglist.size()-1) finalRes += " ";
        }
        return finalRes;
    }
};

class CheckMode {
private:
    MnemonicTool& mnemonicTool;
    const BloomFilter& bloom;
    ProgramConfig cfg;
    std::atomic<bool>& running;
    
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    std::map<std::string, int> wordMap;
    
    BrainwalletGen bwGen;
    CleverAI cleverAI; // Instanta AI actualizata
    secp256k1_context* ctx;

    std::string formatUnits(double num, const std::string& unit) {
        const char* suffixes[] = { "", " K", " M", " B", " T", " Q" };
        int i = 0; while (num >= 1000 && i < 5) { num /= 1000; i++; }
        std::stringstream ss; ss << std::fixed << std::setprecision(2) << num << suffixes[i] << " " << unit;
        return ss.str();
    }

    void logHit(const std::string& info, const std::string& addr, const std::string& source) {
        std::string fname = cfg.winFile.empty() ? "win.txt" : cfg.winFile;
        std::ofstream file(fname, std::ios::out | std::ios::app);
        if (file.is_open()) {
            file << "[HIT] " << source << " | Addr: " << addr << " | Info: " << info << "\n";
            file.flush();
            file.close();
        }
    }

    std::vector<uint8_t> localHexToBytes(const std::string& hex) {
        std::vector<uint8_t> bytes;
        for (unsigned int i = 0; i < hex.length(); i += 2) {
            std::string byteString = hex.substr(i, 2);
            try { bytes.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16)); } catch(...) {}
        }
        return bytes;
    }

    std::vector<uint8_t> localDecodeBase58(const std::string& str) {
        static const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
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

    std::string localPubKeyToP2SH(const std::vector<uint8_t>& pubKey) {
        unsigned char sha256_res[SHA256_DIGEST_LENGTH];
        SHA256(pubKey.data(), pubKey.size(), sha256_res);
        unsigned char ripemd160_res[RIPEMD160_DIGEST_LENGTH];
        RIPEMD160(sha256_res, SHA256_DIGEST_LENGTH, ripemd160_res);

        std::vector<uint8_t> redeemScript;
        redeemScript.push_back(0x00);
        redeemScript.push_back(0x14); 
        redeemScript.insert(redeemScript.end(), ripemd160_res, ripemd160_res + 20);

        SHA256(redeemScript.data(), redeemScript.size(), sha256_res);
        RIPEMD160(sha256_res, SHA256_DIGEST_LENGTH, ripemd160_res);

        std::vector<uint8_t> data;
        data.push_back(0x05);
        data.insert(data.end(), ripemd160_res, ripemd160_res + 20);
        
        unsigned char hash1[SHA256_DIGEST_LENGTH];
        SHA256(data.data(), data.size(), hash1);
        unsigned char hash2[SHA256_DIGEST_LENGTH];
        SHA256(hash1, SHA256_DIGEST_LENGTH, hash2);
        
        data.insert(data.end(), hash2, hash2 + 4);

        const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        std::string ret = "";
        
        std::vector<unsigned char> vch = data;
        std::vector<unsigned char>::const_iterator pbegin = vch.begin();
        while (pbegin != vch.end() && *pbegin == 0) pbegin++;
        int zeros = pbegin - vch.begin();

        int size = vch.size();
        std::vector<unsigned char> b58((size * 138 / 100) + 1);
        memset(b58.data(), 0, b58.size());
        
        int length = 0;
        for (; pbegin != vch.end(); pbegin++) {
            int carry = *pbegin;
            int i = 0;
            for (std::vector<unsigned char>::reverse_iterator it = b58.rbegin(); (it != b58.rend() || carry != 0); it++, i++) {
                if (it == b58.rend()) break; 
                carry += 256 * (*it);
                *it = carry % 58;
                carry /= 58;
            }
            length = i;
        }
        
        std::vector<unsigned char>::const_iterator it = b58.end() - length;
        while (it != b58.end() && *it == 0) it++;
        
        ret.reserve(zeros + (b58.end() - it));
        ret.assign(zeros, '1');
        while (it != b58.end()) ret += pszBase58[*(it++)];
        
        return ret;
    }

    bool localCheckAddress(const std::string& addr) {
        if (!bloom.isLoaded()) return false;
        
        if (addr.rfind("0x", 0) == 0 && addr.length() == 42) {
             return bloom.check_hash160(localHexToBytes(addr.substr(2)));
        }
        
        if (addr.length() >= 26 && addr.length() <= 35 && (addr[0] == '1' || addr[0] == '3')) {
            std::vector<uint8_t> decoded = localDecodeBase58(addr);
            if (decoded.size() == 25) {
                std::vector<uint8_t> hash160(decoded.begin() + 1, decoded.begin() + 21);
                return bloom.check_hash160(hash160);
            }
        }
        return false; 
    }

    bool parseExternalLogLine(const std::string& line, MnemonicResult& res) {
        if (line.find("BIP") == std::string::npos || line.find("m/") == std::string::npos) return false;

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        while (ss >> token) tokens.push_back(token);

        if (tokens.size() < 3) return false;

        std::string addr = "";
        std::string path = "";
        std::string type = "";

        for (size_t i = 0; i < tokens.size(); i++) {
            if (tokens[i].rfind("m/", 0) == 0) { 
                path = tokens[i];
                for(size_t j=0; j<i; j++) type += tokens[j] + " ";
                if(!type.empty()) type.pop_back();
                if (i + 1 < tokens.size()) addr = tokens[i+1];
                break;
            }
        }

        if (addr.empty() || path.empty()) return false;
        if (addr == "-" || addr.length() < 10) return false;

        RowInfo row;
        row.type = type;
        row.path = path;
        row.addr = addr;
        row.isHit = localCheckAddress(addr);

        res.mnemonic = "External Log (No Secret)";
        res.entropyHex = "Input: Log Line";
        res.rows.clear();
        res.rows.push_back(row);
        
        return true; 
    }

    bool isAddressTypeMatch(const std::string& typeStr, const std::string& pathStr, std::string filter) {
        if (filter.empty() || filter == "ALL") return true;
        std::string f = filter; std::transform(f.begin(), f.end(), f.begin(), ::toupper);
        std::string t = typeStr; std::transform(t.begin(), t.end(), t.begin(), ::toupper);
        std::string p = pathStr; std::transform(p.begin(), p.end(), p.begin(), ::toupper);

        if (f.find("ETH") != std::string::npos && t.find("ETH") == std::string::npos) return false;
        if (f.find("BTC") != std::string::npos && t.find("ETH") != std::string::npos) return false;
        if (t.find("ETH") == std::string::npos) { 
            if (f.find("BIP") != std::string::npos) {
                if (f.find("BIP32") != std::string::npos) {
                    if (p.find("44'") != std::string::npos || p.find("49'") != std::string::npos || p.find("84'") != std::string::npos) return false;
                }
                else if (f.find("BIP44") != std::string::npos && p.find("44'") == std::string::npos) return false;
                else if (f.find("BIP49") != std::string::npos && p.find("49'") == std::string::npos) return false;
                else if (f.find("BIP84") != std::string::npos && p.find("84'") == std::string::npos) return false;
                else if (f.find("BIP141") != std::string::npos) {
                    if (t.find("BECH32") == std::string::npos && t.find("P2SH") == std::string::npos && t.find("SEGWIT") == std::string::npos) return false;
                }
            }
            if (f.find("UNCOMP") != std::string::npos) { if (t.find("UNCOMP") == std::string::npos) return false; } 
            else if (f.find("COMP") != std::string::npos) { if (t.find("UNCOMP") != std::string::npos) return false; }
            if (f.find("BECH32") != std::string::npos && t.find("BECH32") == std::string::npos) return false;
            if (f.find("P2SH") != std::string::npos && t.find("P2SH") == std::string::npos) return false;
            if ((f.find("P2PKH") != std::string::npos || f.find("LEGACY") != std::string::npos) && t.find("P2PKH") == std::string::npos) return false;
        }
        return true;
    }

    void loadWordMap() {
        if (!wordMap.empty()) return;
        std::string path = cfg.wordlistPath.empty() ? "bip39/english.txt" : cfg.wordlistPath;
        std::ifstream file(path);
        if (!file.is_open()) file.open("english.txt");
        if (file.is_open()) {
            std::string w; int idx = 0;
            while (file >> w) wordMap[w] = idx++;
        }
    }

    std::string mnemonicToEntropyHex(const std::string& phrase) {
        loadWordMap();
        if (wordMap.empty()) return "Error: Wordlist not found";
        std::stringstream ss(phrase);
        std::string word;
        std::vector<int> indices;
        while (ss >> word) {
            if (wordMap.find(word) == wordMap.end()) return "Error: Invalid Word";
            indices.push_back(wordMap[word]);
        }
        std::string bits = "";
        for (int idx : indices) bits += std::bitset<11>(idx).to_string();
        int totalBits = (int)bits.length();
        int entBits = (totalBits * 32) / 33;
        bits = bits.substr(0, entBits);
        std::stringstream hexss;
        hexss << std::hex << std::setfill('0');
        for (size_t i = 0; i < bits.length(); i += 8) {
            if (i + 8 > bits.length()) break;
            std::string byteStr = bits.substr(i, 8);
            unsigned long val = std::bitset<8>(byteStr).to_ulong();
            hexss << std::setw(2) << val;
        }
        return hexss.str();
    }

    void drawTableHeader() {
        std::cout << "TYPE                PATH                        ADDRESS                                      STATUS\033[K\n";
        std::cout << "--------------------------------------------------------------------------------------\033[K\n";
    }

public:
    CheckMode(MnemonicTool& mt, const BloomFilter& bf, ProgramConfig c, std::atomic<bool>& r) 
        : mnemonicTool(mt), bloom(bf), cfg(c), running(r) {
        ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    }
    
    ~CheckMode() {
        if (ctx) secp256k1_context_destroy(ctx);
    }

    void run() {
        globalRunningPtr = &running;
        signal(SIGINT, handleSigInt); 

        // [MODIFICAT IMPORTANT] Am eliminat codul care suprascria caile cu cele default.
        // Acum folosim ce a incarcat Runner (din fisier).

        FILE* fpInput = nullptr;
        FILE* pipeInput = nullptr;
        SocketHandle sockFD = INVALID_SOCKET_VAL;

        bool usingPipe = false;
        bool usingSocket = false;
        bool singleShot = false;
        bool isBrainwallet = !cfg.brainwalletMode.empty();
        
        char* lineBuffer = new char[8192];
        std::string socketBuffer = "";

        if (!cfg.inputFile.empty()) {
            fpInput = fopen(cfg.inputFile.c_str(), "rb"); 
            if (!fpInput) { std::cerr << "[ERROR] Cannot open file: " << cfg.inputFile << "\n"; delete[] lineBuffer; return; }
        } 
        else if (!cfg.execCommand.empty()) {
            pipeInput = POPFEN(cfg.execCommand.c_str(), "r");
            if (!pipeInput) { std::cerr << "[ERROR] Failed to start process.\n"; delete[] lineBuffer; return; }
            usingPipe = true;
        }
        else if (!cfg.socketPath.empty()) {
            #ifdef _WIN32
                WSADATA wsaData; WSAStartup(MAKEWORD(2, 2), &wsaData);
            #endif
            sockFD = socket(AF_UNIX, SOCK_STREAM, 0);
            if (sockFD == INVALID_SOCKET_VAL) { std::cerr << "[ERROR] Could not create socket.\n"; delete[] lineBuffer; return; }
            struct sockaddr_un addr;
            memset(&addr, 0, sizeof(addr));
            addr.sun_family = AF_UNIX;
            strncpy(addr.sun_path, cfg.socketPath.c_str(), sizeof(addr.sun_path) - 1);
            if (connect(sockFD, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
                std::cerr << "[ERROR] Could not connect to socket.\n";
                CLOSE_SOCKET(sockFD); delete[] lineBuffer; return;
            }
            usingSocket = true;
        }
        else if (!cfg.entropyStr.empty()) {
            singleShot = true;
        }
        else if (isBrainwallet) {
            // Internal generator
        }
        else {
            std::cerr << "[ERROR] Check mode requires source.\n";
            delete[] lineBuffer;
            return;
        }

        startTime = std::chrono::high_resolution_clock::now();
        std::cout << "\033[?25l"; 
        std::cout << "\033[2J";   

        std::string line;
        unsigned long long count = 0;
        unsigned long long totalAddrChecked = 0;

        while (running) {
            if (cfg.count > 0 && count >= cfg.count) {
                running = false;
                break;
            }

            bool hasLine = false;

            if (singleShot) { line = cfg.entropyStr; hasLine = true; running = false; } 
            else if (fpInput) {
                if (fgets(lineBuffer, 8192, fpInput)) {
                    line = lineBuffer;
                    hasLine = true;
                } else {
                    running = false; 
                }
            }
            else if (usingPipe) {
                if (fgets(lineBuffer, 8192, pipeInput)) { line = lineBuffer; hasLine = true; } 
                else running = false;
            } 
            else if (usingSocket) {
                size_t newlinePos = socketBuffer.find('\n');
                if (newlinePos != std::string::npos) {
                    line = socketBuffer.substr(0, newlinePos);
                    socketBuffer.erase(0, newlinePos + 1);
                    hasLine = true;
                } else {
                    int bytesRead = recv(sockFD, lineBuffer, 8191, 0);
                    if (bytesRead > 0) {
                        lineBuffer[bytesRead] = '\0';
                        socketBuffer += lineBuffer;
                        newlinePos = socketBuffer.find('\n');
                        if (newlinePos != std::string::npos) {
                            line = socketBuffer.substr(0, newlinePos);
                            socketBuffer.erase(0, newlinePos + 1);
                            hasLine = true;
                        }
                    } else running = false;
                }
            }
            else if (isBrainwallet) { 
                if (cfg.brainwalletMode == "clever") {
                    line = cleverAI.generate(); // Foloseste noul AI
                } else {
                    line = bwGen.generate(cfg.brainwalletMode);
                }
                hasLine = true;
            }

            if (hasLine) {
                line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());
                line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
                
                if (!isBrainwallet || fpInput || usingPipe) {
                    size_t first = line.find_first_not_of(" \t");
                    if (first == std::string::npos) continue; 
                    line = line.substr(first, line.find_last_not_of(" \t") - first + 1);
                    if(line.length() < 1) continue;
                }

                count++;
                MnemonicResult res;
                bool isExternalLog = false;
                bool foundHit = false;

                if (isBrainwallet) {
                    uint8_t privKey[32];
                    SHA256((const uint8_t*)line.c_str(), line.length(), privKey);
                    secp256k1_pubkey pub;
                    if (secp256k1_ec_pubkey_create(ctx, &pub, privKey)) {
                        uint8_t cPub[33]; size_t clen = 33;
                        secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
                        std::vector<uint8_t> vPubC(cPub, cPub + 33);

                        uint8_t uPub[65]; size_t ulen = 65;
                        secp256k1_ec_pubkey_serialize(ctx, uPub, &ulen, &pub, SECP256K1_EC_UNCOMPRESSED);
                        std::vector<uint8_t> vPubU(uPub, uPub + 65);

                        res.mnemonic = line; 
                        res.entropyHex = BytesToHex(std::vector<uint8_t>(privKey, privKey+32));

                        std::string bech = PubKeyToNativeSegwit(vPubC);
                        bool h1 = localCheckAddress(bech);
                        res.rows.push_back({"Bech32", "SHA256", bech, "", h1});

                        std::string p2sh = localPubKeyToP2SH(vPubC);
                        bool h2 = localCheckAddress(p2sh);
                        res.rows.push_back({"P2SH-Segwit", "SHA256", p2sh, "", h2});

                        std::string legC = PubKeyToLegacy(vPubC);
                        bool h3 = localCheckAddress(legC);
                        res.rows.push_back({"Legacy (C)", "SHA256", legC, "", h3});

                        std::string legU = PubKeyToLegacy(vPubU);
                        bool h4 = localCheckAddress(legU);
                        res.rows.push_back({"Legacy (U)", "SHA256", legU, "", h4});

                        if (h1 || h2 || h3 || h4) foundHit = true;
                    }
                }
                else if (parseExternalLogLine(line, res)) {
                    isExternalLog = true;
                } 
                else {
                    std::string processedMnemonic = "";
                    std::string displayEntropy = line;

                    if (cfg.entropyMode != "default" && cfg.entropyMode != "hex" && 
                        cfg.entropyMode != "card" && cfg.entropyMode != "dice" && 
                        cfg.entropyMode != "bin" && cfg.entropyMode != "base6" && 
                        cfg.entropyMode != "base10") {
                         std::string phrase = mnemonicTool.phraseFromEntropyString(line, cfg.entropyMode);
                         if (phrase.rfind("ERROR", 0) != 0) {
                             processedMnemonic = phrase;
                             displayEntropy = line + " [" + cfg.entropyMode + "]";
                         }
                    }

                    if (!processedMnemonic.empty()) {
                        res = mnemonicTool.processRawLine(processedMnemonic, bloom);
                        res.entropyHex = displayEntropy; 
                    } else {
                        res = mnemonicTool.processRawLine(line, bloom);
                        if (res.entropyHex.find("Seed (") != std::string::npos) {
                            std::string hexEnt = mnemonicToEntropyHex(res.mnemonic);
                            if (hexEnt.find("Error") == std::string::npos) {
                                std::string convertedEntropy;
                                std::string label;
                                if (cfg.entropyMode == "card") { convertedEntropy = EntropyConverter::toCards(hexEnt); label = " (Cards)"; }
                                else if (cfg.entropyMode == "dice") { convertedEntropy = EntropyConverter::toDice(hexEnt); label = " (Dice)"; }
                                else if (cfg.entropyMode == "bin") { convertedEntropy = EntropyConverter::toBinary(hexEnt); label = " (Binary)"; }
                                else if (cfg.entropyMode == "base10") { convertedEntropy = EntropyConverter::toBase10(hexEnt); label = " (Base10)"; }
                                else if (cfg.entropyMode == "base6") { convertedEntropy = EntropyConverter::toBase6(hexEnt); label = " (Base6)"; }
                                else { convertedEntropy = hexEnt; label = " (Entropy)"; }
                                res.entropyHex = convertedEntropy + label;
                            }
                        }
                    }
                }

                totalAddrChecked += res.rows.size();
                double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count();
                if(elapsed < 0.001) elapsed = 0.001;

                bool shouldPrint = foundHit || (count <= 50) || (count % 1000 == 0);

                if (shouldPrint) {
                    std::cout << "\033[H=== Check Mode (CPU) ===\n";
                    if (isBrainwallet) std::cout << "Input: BRAINWALLET (" << (fpInput ? "FILE" : cfg.brainwalletMode) << ")\033[K\n";
                    else if (singleShot) std::cout << "Input: Single Value\033[K\n";
                    else if (usingPipe) std::cout << "Input: EXEC [" << cfg.execCommand << "]\033[K\n";
                    else if (usingSocket) std::cout << "Input: SOCKET [" << cfg.socketPath << "]\033[K\n";
                    else std::cout << "Input: FILE [" << cfg.inputFile << "]\033[K\n";
                    
                    std::cout << "Mode: " << (isBrainwallet ? "SHA256 Pass" : cfg.entropyMode) << " | Filter: " << cfg.setAddress << "\033[K\n";
                    std::cout << "Bloom: " << (bloom.isLoaded() ? "Loaded" : "Not Loaded") << "\033[K\n\n";

                    if (!isExternalLog) {
                        bool isError = !isBrainwallet && ((res.mnemonic.find("Error") != std::string::npos) || 
                                       (res.mnemonic.find("Invalid") != std::string::npos) ||
                                       (res.rows.empty()));

                        if (isError) {
                            std::cout << "Input:  " << line << "\033[K\n";
                            std::cout << "Status: \033[1;31mInvalid Mnemonic (Checksum or Word Error)\033[0m\033[K\n";
                            if (res.mnemonic != line && res.mnemonic.length() > 0) {
                                std::cout << "Detail: " << res.mnemonic << "\033[K\n";
                            }
                            std::cout << "\n";
                        } 
                        else {
                            std::cout << (isBrainwallet ? "Pass:   " : "Phrase: ") << res.mnemonic << "\033[K\n";
                            std::cout << "PrivKey: " << res.entropyHex << "\033[K\n\n";
                        }
                    } else {
                        std::cout << "Reading Pre-Formatted Log Stream...\033[K\n\n\n";
                    }

                    drawTableHeader();

                    int rowsPrinted = 0;
                    for(const auto& row : res.rows) {
                        if (!isAddressTypeMatch(row.type, row.path, cfg.setAddress)) continue;
                        // [MODIFICAT AICI]: Marim limita de afisare la 500
                        if(rowsPrinted >= 500) break;
                        rowsPrinted++;
                        std::cout << std::left << std::setw(20) << row.type << std::setw(28) << row.path << std::setw(45) << row.addr;
                        if(row.isHit) {
                            std::cout << "\033[1;32mHIT\033[0m";
                            logHit((isExternalLog ? "Log Stream" : res.mnemonic), row.addr, row.path);
                        } else std::cout << "-";
                        std::cout << "\033[K\n"; 
                    }
                    // Nu mai umplem cu linii goale daca am depasit ecranul
                    if (rowsPrinted < 22) {
                        for(int i = rowsPrinted; i < 22; i++) std::cout << "\033[K\n";
                    }
                    std::cout << "--------------------------------------------------------------------------------------\033[K\n";
                    std::cout << "Checked: " << count << " | Speed: " << formatUnits((double)totalAddrChecked / elapsed, "addr/s") << "    \033[K\n";
                }
            }
        }

        if (fpInput) fclose(fpInput);
        if (usingPipe) PCLOSE(pipeInput); 
        if (usingSocket && sockFD != INVALID_SOCKET_VAL) { CLOSE_SOCKET(sockFD); }
        #ifdef _WIN32
            if(usingSocket) WSACleanup();
        #endif
        delete[] lineBuffer;

        std::cout << "\033[30B"; 
        std::cout << "\033[?25h"; 
        std::cout << "\n[CHECK] Job Finished.\n";
    }
};
#endif