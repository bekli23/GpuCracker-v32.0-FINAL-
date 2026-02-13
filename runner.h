#pragma once
#define _CRT_SECURE_NO_WARNINGS

#ifdef _WIN32
    #ifndef NOMINMAX
    #define NOMINMAX
    #endif
#endif

#include <iostream>
#include <vector>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <sstream>
#include <map>
#include <random>
#include <cstring>

// Dependinte Proiect
#include "args.h"
#include "gpu_interface.h"
#include "bloom.h"
#include "cuda_provider.h"
#include "opencl_provider.h"
#include "utils.h"
#include "mnemonic.h"
#include "mode_akm.h"
#include "check.h"
#include "scan.h" 
#include "multicoin.h" // [CRITIC] Pentru suportul extins de adrese

#ifdef ENABLE_VULKAN
#include "vulkan_provider.h"
#include <vulkan/vulkan.h>
#endif
#include <CL/cl.h>

#ifndef __CUDACC__
    #include <openssl/sha.h>
    #include <openssl/ripemd.h>
    #include <openssl/hmac.h>
    #include <openssl/evp.h>
    #include <secp256k1.h>
#endif

extern "C" {
    void launch_gpu_akm_search(
        unsigned long long startSeed,
        unsigned long long count,
        int blocks,
        int threads,
        int points,
        const void* bloomFilterData,
        size_t bloomFilterSize,
        unsigned long long* outFoundSeeds,
        int* outFoundCount,
        int targetBits,
        bool sequential,
        const void* prefix,
        int prefixLen
    );
}

#ifdef _WIN32
#include <windows.h>
inline void setupConsole() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0; GetConsoleMode(hOut, &dwMode); dwMode |= 0x0004; 
    SetConsoleMode(hOut, dwMode); SetConsoleOutputCP(65001); 
    CONSOLE_CURSOR_INFO ci; GetConsoleCursorInfo(hOut, &ci); ci.bVisible = false; SetConsoleCursorInfo(hOut, &ci);
}
inline void restoreConsole() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO ci; GetConsoleCursorInfo(hOut, &ci); ci.bVisible = true; SetConsoleCursorInfo(hOut, &ci);
}
#else
#include <unistd.h>
inline void setupConsole() {} 
inline void restoreConsole() {}
#endif

inline void moveTo(int r, int c) { std::cout << "\033[" << r << ";" << c << "H"; }

struct Scheme { std::string name; std::string path; int type; };

struct DisplayState {
    unsigned long long countId = 0;
    std::string mnemonic = "-";
    std::string entropyHex = "-"; 
    std::string hexKey = "-";
    struct AddrInfo { std::string type, path, addr; std::string status; bool isHit; };
    std::vector<AddrInfo> rows;
    int currentBit = 0;
};

struct ActiveGpuContext { IGpuProvider* provider; std::string backend; int deviceId; int globalId; };

#ifndef __CUDACC__
class Runner {
private:
    ProgramConfig cfg;
    BloomFilter bloom;
    MnemonicTool mnemonicTool;
    AkmTool akmTool;
    XprvGenerator xprvGen; 
    
    std::atomic<bool> running{ true };
    std::atomic<unsigned long long> totalSeedsChecked{ 0 }; 
    std::atomic<unsigned long long> totalAddressesChecked{ 0 }; 
    std::atomic<unsigned long long> realSeedsProcessed{ 0 }; 
    
    std::atomic<int> currentActiveBit{ 0 };

    std::mutex displayMutex, fileMutex;
    DisplayState currentDisplay;
    std::vector<ActiveGpuContext> activeGpus;
    std::vector<unsigned char*> hostBuffers;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    
    std::string startHexFullDisplay = ""; 

    int uiBaseLine = 22; int totalCores = 0; int workerCores = 0;
    bool isInputMode = false;
    int loadedPathsCount = 0;

    std::vector<std::string> wordlist;
    std::map<std::string, int> wordMap;
    int entropyBytes = 16;

    unsigned long long convertHexToULL(std::string hexStr) {
        try {
            if (hexStr.size() > 2 && (hexStr.substr(0, 2) == "0x" || hexStr.substr(0, 2) == "0X")) {
                hexStr = hexStr.substr(2);
            }
            if (hexStr.length() > 16) {
                hexStr = hexStr.substr(hexStr.length() - 16); 
            }
            return std::stoull(hexStr, nullptr, 16);
        } catch (...) {
            return 0;
        }
    }

    void expandPathRecursive(const std::vector<std::string>& segments, int index, std::string currentPath, std::vector<std::string>& results) {
        if (index >= (int)segments.size()) {
            results.push_back(currentPath);
            return;
        }
        
        std::string seg = segments[index];
        size_t dash = seg.find('-');
        
        if (dash != std::string::npos) {
            bool hardened = (seg.back() == '\'');
            std::string sStart = seg.substr(0, dash);
            std::string sEnd = seg.substr(dash + 1);
            if (hardened && !sEnd.empty() && sEnd.back() == '\'') sEnd.pop_back();

            int start = 0, end = 0;
            try { 
                start = std::stoi(sStart); 
                end = std::stoi(sEnd); 
            } catch(...) { 
                std::string nextPath = (currentPath.empty() ? "" : currentPath + "/") + seg;
                expandPathRecursive(segments, index + 1, nextPath, results);
                return;
            }

            for (int i = start; i <= end; i++) {
                std::string nextSeg = std::to_string(i) + (hardened ? "'" : "");
                std::string nextPath = (currentPath.empty() ? "m" : currentPath) + "/" + nextSeg;
                if (index == 0 && currentPath.empty()) nextPath = "m/" + nextSeg;
                else if (!currentPath.empty()) nextPath = currentPath + "/" + nextSeg;
                
                expandPathRecursive(segments, index + 1, nextPath, results);
            }
        } else {
            std::string nextPath;
            if (index == 0 && currentPath.empty()) nextPath = seg;
            else nextPath = currentPath + "/" + seg;
            
            expandPathRecursive(segments, index + 1, nextPath, results);
        }
    }

    void loadPathFile() {
        std::vector<PathInfo> finalPaths;
        std::string coinList = cfg.multiCoin; // ex: "btc,ltc,doge"
        std::transform(coinList.begin(), coinList.end(), coinList.begin(), ::tolower);
        
        // Detectare monede active din argumentul --multi-coin
        bool hasBTC = (coinList.find("btc") != std::string::npos);
        bool hasETH = (coinList.find("eth") != std::string::npos);
        bool hasLTC = (coinList.find("ltc") != std::string::npos);
        bool hasDOGE = (coinList.find("doge") != std::string::npos);
        bool hasDASH = (coinList.find("dash") != std::string::npos);
        bool hasBCH = (coinList.find("bch") != std::string::npos);
        bool hasBSV = (coinList.find("bsv") != std::string::npos);
        bool hasBTG = (coinList.find("btg") != std::string::npos); // Bitcoin Gold
        bool hasZEC = (coinList.find("zec") != std::string::npos) || (coinList.find("zcash") != std::string::npos);

        // Default daca nu e nimic specificat
        if (!hasBTC && !hasETH && !hasLTC && !hasDOGE && !hasDASH && !hasBCH && !hasBSV && !hasBTG && !hasZEC) {
            hasBTC = true; 
        }

        if (!cfg.pathFile.empty()) {
            std::ifstream file(cfg.pathFile);
            if (file.is_open()) {
                std::cout << "[INFO] Loading paths from file: " << cfg.pathFile << "\n";
                std::string line;
                int countLoaded = 0;
                while (std::getline(file, line)) {
                    size_t commentPos = line.find('#');
                    if (commentPos != std::string::npos) line = line.substr(0, commentPos);
                    line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
                    if (line.empty()) continue;

                    std::vector<std::string> segments;
                    std::stringstream ss(line);
                    std::string segment;
                    while (std::getline(ss, segment, '/')) {
                        if(!segment.empty()) segments.push_back(segment);
                    }

                    std::vector<std::string> expanded;
                    expandPathRecursive(segments, 0, "", expanded);

                    for (const auto& p : expanded) {
                        // Logica de detectie pentru fisiere custom
                        // Daca userul pune m/44'/2'/... stim ca e LTC
                        std::string label = "CUSTOM";
                        std::string coin = "BTC"; // Default

                        if (p.find("m/44'/0'") == 0 || p.find("m/49'/0'") == 0 || p.find("m/84'/0'") == 0) { coin = "BTC"; label = "BTC-PATH"; }
                        else if (p.find("m/44'/2'") == 0) { coin = "LTC"; label = "LTC-PATH"; }
                        else if (p.find("m/44'/3'") == 0) { coin = "DOGE"; label = "DOGE-PATH"; }
                        else if (p.find("m/44'/5'") == 0) { coin = "DASH"; label = "DASH-PATH"; }
                        else if (p.find("m/44'/60'") == 0) { coin = "ETH"; label = "ETH-PATH"; }
                        else if (p.find("m/44'/145'") == 0) { coin = "BCH"; label = "BCH-PATH"; }
                        else if (p.find("m/44'/236'") == 0) { coin = "BSV"; label = "BSV-PATH"; }
                        else if (p.find("m/44'/156'") == 0) { coin = "BTG"; label = "BTG-PATH"; }
                        else if (p.find("m/44'/133'") == 0) { coin = "ZEC"; label = "ZEC-PATH"; }

                        // Adaugam doar daca moneda a fost ceruta sau e BTC/Generic
                        bool add = false;
                        if (coin == "BTC" && hasBTC) add = true;
                        else if (coin == "ETH" && hasETH) add = true;
                        else if (coin == "LTC" && hasLTC) add = true;
                        else if (coin == "DOGE" && hasDOGE) add = true;
                        else if (coin == "DASH" && hasDASH) add = true;
                        else if (coin == "BCH" && hasBCH) add = true;
                        else if (coin == "BSV" && hasBSV) add = true;
                        else if (coin == "BTG" && hasBTG) add = true;
                        else if (coin == "ZEC" && hasZEC) add = true;
                        // Daca e o cale generica (ex: m/0/0) si nu am detectat moneda specifica,
                        // o adaugam la toate monedele active pentru siguranta
                        else if (label == "CUSTOM") {
                            if (hasBTC) finalPaths.push_back({"GENERIC-BTC", p, "BTC"});
                            if (hasLTC) finalPaths.push_back({"GENERIC-LTC", p, "LTC"});
                            if (hasDOGE) finalPaths.push_back({"GENERIC-DOGE", p, "DOGE"});
                            if (hasETH) finalPaths.push_back({"GENERIC-ETH", p, "ETH"});
                            if (hasDASH) finalPaths.push_back({"GENERIC-DASH", p, "DASH"});
                            if (hasBCH) finalPaths.push_back({"GENERIC-BCH", p, "BCH"});
                            if (hasBSV) finalPaths.push_back({"GENERIC-BSV", p, "BSV"});
                            if (hasBTG) finalPaths.push_back({"GENERIC-BTG", p, "BTG"});
                            if (hasZEC) finalPaths.push_back({"GENERIC-ZEC", p, "ZEC"});
                            add = false; // Deja adaugat multiplu
                        }

                        if (add) {
                            finalPaths.push_back({label, p, coin});
                            countLoaded++;
                        }
                    }
                }
                file.close();
                std::cout << "[INFO] Total Expanded Paths: " << countLoaded << "\n";
                
                if (countLoaded == 0) {
                    std::cout << "[WARN] File was empty. Loading EXTENDED defaults.\n";
                    goto load_defaults;
                }

            } else {
                std::cout << "[WARN] Cannot open path file. Using EXTENDED defaults.\n";
                goto load_defaults;
            }
        }
        else {
        load_defaults:
            // --- BITCOIN (BTC) ---
            if (hasBTC) {
                // BIP32 / Legacy Misc
                finalPaths.push_back({"BIP32", "m/0", "BTC"});
                finalPaths.push_back({"BIP32", "m/0/0", "BTC"});
                finalPaths.push_back({"BIP32", "m/0/0'", "BTC"});
                finalPaths.push_back({"BIP32", "m/0'/0", "BTC"});
                finalPaths.push_back({"BIP32", "m/0'/0/0", "BTC"});
                finalPaths.push_back({"BIP32", "m/1/10", "BTC"});

                // Standard BIPs
                finalPaths.push_back({"BIP44", "m/44'/0'/0'/0/0", "BTC"});
                finalPaths.push_back({"BIP49", "m/49'/0'/0'/0/0", "BTC"});
                finalPaths.push_back({"BIP84", "m/84'/0'/0'/0/0", "BTC"});
                finalPaths.push_back({"BIP141", "m/0/0", "BTC"});
            }

            // --- LITECOIN (LTC) ---
            if (hasLTC) {
                finalPaths.push_back({"LTC-BIP44", "m/44'/2'/0'/0/0", "LTC"});
                finalPaths.push_back({"LTC-BIP49", "m/49'/2'/0'/0/0", "LTC"}); // Segwit LTC
                finalPaths.push_back({"LTC-BIP84", "m/84'/2'/0'/0/0", "LTC"}); // Native Segwit LTC
            }

            // --- DOGECOIN (DOGE) ---
            if (hasDOGE) {
                finalPaths.push_back({"DOGE-BIP44", "m/44'/3'/0'/0/0", "DOGE"});
            }

            // --- DASH ---
            if (hasDASH) {
                finalPaths.push_back({"DASH-BIP44", "m/44'/5'/0'/0/0", "DASH"});
            }

            // --- BITCOIN GOLD (BTG) ---
            if (hasBTG) {
                finalPaths.push_back({"BTG-BIP44", "m/44'/156'/0'/0/0", "BTG"});
                finalPaths.push_back({"BTG-BIP49", "m/49'/156'/0'/0/0", "BTG"});
                finalPaths.push_back({"BTG-BIP84", "m/84'/156'/0'/0/0", "BTG"});
            }

            // --- BITCOIN CASH (BCH) ---
            if (hasBCH) {
                finalPaths.push_back({"BCH-BIP44", "m/44'/145'/0'/0/0", "BCH"});
            }

            // --- BITCOIN SV (BSV) ---
            if (hasBSV) {
                finalPaths.push_back({"BSV-BIP44", "m/44'/236'/0'/0/0", "BSV"});
            }

            // --- ZCASH (ZEC) ---
            if (hasZEC) {
                finalPaths.push_back({"ZEC-BIP44", "m/44'/133'/0'/0/0", "ZEC"});
            }

            // --- ETHEREUM (ETH) ---
            if (hasETH) {
                finalPaths.push_back({"ETH-BIP44", "m/44'/60'/0'/0/0", "ETH"});
                finalPaths.push_back({"ETH-BIP32", "m/0", "ETH"}); 
            }
        }
        mnemonicTool.setPaths(finalPaths);
        loadedPathsCount = (int)finalPaths.size();
    }

    void loadWords() {
        if (cfg.runMode != "mnemonic") return; 
        std::string target = cfg.language;
        if (target.find(".txt") == std::string::npos) target += ".txt";
        std::vector<std::string> paths = { "bip39/" + target, "bip39\\" + target, target, "bip39/english.txt" };
        std::ifstream file;
        for (const auto& p : paths) { file.open(p); if (file.is_open()) break; }
        if (!file.is_open()) {
            std::cout << "\n\033[1;31m[CRITICAL] Wordlist not found!\033[0m\n";
            return; 
        }
        wordlist.clear(); wordMap.clear();
        std::string line;
        while (std::getline(file, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c){ return std::isspace(c); }), line.end());
            if (!line.empty()) { 
                wordlist.push_back(line); 
                wordMap[line] = (int)wordlist.size() - 1; 
            }
        }
        file.close();
    }

    void uploadWordsToGPU() {
        if (wordlist.empty()) return;
        std::string flatWL = "";
        for (const auto& w : wordlist) flatWL += w + " ";
        launch_gpu_init(flatWL.c_str(), (int)flatWL.length(), (int)wordlist.size());
        std::cout << "[GPU] Wordlist uploaded (" << wordlist.size() << " words).\n";
    }

    void calculateMnemonicParams() {
        entropyBytes = (cfg.words * 11 - cfg.words / 3) / 8;
    }

    bool isAddressTypeMatch(const std::string& typeStr, const std::string& pathStr, const std::string& filter) {
    if (filter.empty() || filter == "ALL") return true;

    std::string t = typeStr;
    std::string p = pathStr;
    std::transform(t.begin(), t.end(), t.begin(), ::toupper);
    std::transform(p.begin(), p.end(), p.begin(), ::toupper);

    std::stringstream ss(filter);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t\r\n"));
        item.erase(item.find_last_not_of(" \t\r\n") + 1);
        if (item.empty()) continue;

        std::string f = item;
        std::transform(f.begin(), f.end(), f.begin(), ::toupper);

        if (f == "LEGACY" || f == "P2PKH") {
            if (t.find("LEGACY") != std::string::npos || t.find("P2PKH") != std::string::npos)
                return true;
        }
        else if (f == "P2SH") {
            if (t.find("P2SH") != std::string::npos) return true;
        }
        else if (f == "BECH32") {
            if (t.find("BECH32") != std::string::npos) return true;
        }
        else if (f == "SEGWIT") {
            if (t.find("P2SH") != std::string::npos || t.find("BECH32") != std::string::npos || t.find("SEGWIT") != std::string::npos)
                return true;
        }
        else if (f == "ETH") {
            if (t.find("ETH") != std::string::npos) return true;
        }
        else if (f == "COMP") {
            // Comprimat: orice tip care NU conține "UNCOMP"
            if (t.find("UNCOMP") == std::string::npos) return true;
        }
        else if (f == "UNCOMP") {
            if (t.find("UNCOMP") != std::string::npos) return true;
        }
        else {
            // Căutare generală în tip sau cale
            if (t.find(f) != std::string::npos || p.find(f) != std::string::npos)
                return true;
        }
    }
    return false;
}
    std::string formatUnits(double num, const std::string& unit) {
        const char* suffixes[] = { "", " K", " M", " B", " T", " Q" };
        int i = 0; while (num >= 1000 && i < 5) { num /= 1000; i++; }
        std::stringstream ss; ss << std::fixed << std::setprecision(2) << num << suffixes[i] << " " << unit;
        return ss.str();
    }

    void logDetailedHit(const std::string& mode, const std::string& info, const std::string& secret, const std::string& addr, const std::string& pk, const std::string& path) {
        std::lock_guard<std::mutex> lock(fileMutex);
        std::string fname = cfg.winFile.empty() ? "win.txt" : cfg.winFile;
        std::ofstream file(fname, std::ios::out | std::ios::app);
        if (file.is_open()) {
            file << "\n======================================================\n";
            file << "HIT FOUND (" << mode << ") | " << __DATE__ << " " << __TIME__ << "\n";
            file << "------------------------------------------------------\n";
            file << "Seed/Entropy: " << info << "\n";
            file << "Secret:       " << secret << "\n";
            file << "Address:      " << addr << "\n";
            file << "Path:         " << path << "\n";
            file << "PrivKey:      " << pk << "\n";
            file << "======================================================\n\n";
            file.flush();
            file.close();
        }
    }

    void detectHardware() {
        int gIdx = 0; bool useAll = (cfg.deviceId == -1);
        int cudaCount = 0;
        if (cudaGetDeviceCount(&cudaCount) == cudaSuccess) {
            for (int i = 0; i < cudaCount; i++) {
                if (useAll || cfg.deviceId == gIdx) {
                    try {
                        auto* p = new CudaProvider(i, cfg.cudaBlocks, cfg.cudaThreads, cfg.pointsPerThread, true);
                        p->init(); activeGpus.push_back({ p, "CUDA", i, gIdx });
                    } catch (...) {}
                }
                gIdx++;
            }
        }
        for(auto& gpu : activeGpus) hostBuffers.push_back(new unsigned char[(size_t)gpu.provider->getBatchSize() * 32]);
    }

public:
    Runner(ProgramConfig c) : cfg(c) {
        totalCores = std::thread::hardware_concurrency();
        workerCores = (cfg.cpuCores > 0) ? cfg.cpuCores : totalCores;
        setupConsole();
        
        if (!cfg.startFrom.empty()) {
            std::string s = cfg.startFrom;
            if (s.length() >= 2 && (s.substr(0,2) == "0x" || s.substr(0,2) == "0X")) s = s.substr(2);
            s.erase(0, s.find_first_not_of('0'));
            if (s.empty()) s = "0";
            startHexFullDisplay = "0x" + s;

            if (s.length() > 16) {
                std::string lowPart = s.substr(s.length() - 16);
                std::string highPart = s.substr(0, s.length() - 16);
                try {
                    totalSeedsChecked = std::stoull(lowPart, nullptr, 16);
                } catch(...) { totalSeedsChecked = 0; }

                if (cfg.runMode == "akm") {
                    akmTool.setHighHexPrefix(highPart);
                    std::cout << "[INFO] Huge Start Detected. Low: " << std::hex << totalSeedsChecked 
                              << " High: " << highPart << std::dec << "\n";
                }
            } else {
                try {
                    totalSeedsChecked = std::stoull(s, nullptr, 16);
                } catch(...) { totalSeedsChecked = 0; }
            }
            std::cout << "[INFO] Resuming from offset: " << startHexFullDisplay << "\n";
        }

        loadPathFile();
        if (cfg.runMode == "akm") {
            if (cfg.akmListProfiles) { akmTool.listProfiles(); exit(0); }
            akmTool.init(cfg.akmProfile, "akm/wordlist_512_ascii.txt");
        } else {
            calculateMnemonicParams();
            loadWords();
            mnemonicTool.loadWordlist(cfg.language);
            if (!wordlist.empty()) uploadWordsToGPU();
        }
        
        if (!cfg.bloomFiles.empty()) bloom.load(cfg.bloomFiles);
        detectHardware();
    }

    ~Runner() { restoreConsole(); for (auto& g : activeGpus) if(g.provider) delete g.provider; for (auto* b : hostBuffers) delete[] b; }

    void drawInterface() { std::cout << "\033[2J\033[H"; }

    void updateStats() {
        DisplayState s; { std::lock_guard<std::mutex> lock(displayMutex); s = currentDisplay; }
        double secs = std::max(0.001, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count());
        static int uiFrame = 0; uiFrame++;

        std::cout << "\033[H"; 
        std::cout << "=== GpuCracker v42.3 (Multi-Coin + XPRV) ===\n";
        std::cout << "Class C: 100M h/s | Class D: 1B+ h/s\n"; 
        
        for (const auto& gpu : activeGpus) {
            std::string memInfo = "Shared/Global";
#ifdef _WIN32
            if (gpu.backend == "CUDA") {
                size_t f = 0, t = 0; cudaSetDevice(gpu.deviceId); 
                if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
                    std::stringstream ss; ss << (t-f)/(1024*1024) << "/" << t/(1024*1024) << " MB";
                    memInfo = ss.str();
                }
            }
#endif
            std::string gpuName = gpu.provider->getName();
            std::cout << "GPU " << gpu.globalId << ": \033[1;33m" << gpuName << "\033[0m (" << gpu.backend << ")\n";
            std::cout << "Conf:    \033[1;36m" << gpu.provider->getConfig() << "\033[0m | VRAM: " << memInfo << "\n";
            
            std::string recommendation = "";
            bool isLowSettings = (cfg.cudaBlocks < 1024 || cfg.cudaThreads < 512);

            if (gpuName.find("1060") != std::string::npos) {
                 if (cfg.cudaBlocks < 2048) {
                      recommendation = " [REC: GTX 1060 likes High VRAM. Try: --blocks 2048 --threads 2048 --points 128 for >14 T/s]";
                 }
            }
            else if (gpuName.find("RTX") != std::string::npos || gpuName.find("3060") != std::string::npos || gpuName.find("4090") != std::string::npos) {
                if (cfg.cudaBlocks < 4096) {
                    recommendation = " [REC: RTX Card detected. Try --blocks 4096 --threads 512 for speed boost]";
                }
            }
            else if (isLowSettings) {
                 recommendation = " [REC: High Load (100%) != Max Speed. Increase VRAM usage (--blocks 2048 --threads 1024) for optimal throughput]";
            }

            if (!recommendation.empty()) {
                std::cout << "\033[1;32m" << recommendation << "\033[0m\n";
            }
        }

        std::cout << "CPU:    Using " << workerCores << "/" << totalCores << " Cores\n";
        std::cout << "Filter: " << (cfg.setAddress.empty() ? "ALL" : cfg.setAddress) << "\n";
        
        size_t totalBloomSize = 0;
        for(size_t i=0; i<bloom.getLayerCount(); i++) totalBloomSize += bloom.getLayerSize(i);
        
        std::cout << "Bloom Info: " << bloom.getLayerCount() << " File(s) | Total Size: " 
                  << (totalBloomSize / 1024 / 1024) << " MB\n";

        std::cout << "Paths:  Loaded " << loadedPathsCount << " variations (" << cfg.multiCoin << ")\n";
        std::cout << "\033[1;37mClasses of Attack Strategy:\033[0m\n";
        std::cout << "Class A: 10K h/s | \033[1;32mClass B: 1M h/s (GPU Experimental)\033[0m\n";
        std::cout << "Class C: 100M h/s | Class D: 1B+ h/s\n";
        
        std::string entDisplay = "Standard Random";
        
        if (cfg.runMode == "xprv-mode") {
             entDisplay = "XPRV SCANNER (Auto: " + cfg.entropyMode + ")";
        }
        else if (!cfg.xprv.empty()) {
            entDisplay = "Extended Private Key (XPRV Derivation - Single)";
        }
        else if (cfg.runMode == "akm") {
             // [MODIFICAT] Afișăm profilul curent
             std::string currentProfile = cfg.akmProfile.empty() ? "Default" : cfg.akmProfile;

             if (cfg.akmBits.size() > 1) {
                 int liveBit = currentActiveBit.load();
                 entDisplay = "AKM Mode [" + currentProfile + "] [POOL: " + std::to_string(cfg.akmBits.size()) + "] Current: " + std::to_string(liveBit) + " bits";
             } else {
                 entDisplay = "AKM Mode [" + currentProfile + "] (Bit Target: " + std::to_string((cfg.akmBits.empty() ? 0 : cfg.akmBits[0])) + ")";
             }
        }
        else if (!cfg.entropyStr.empty()) {
            if (cfg.entropyStr == "random") entDisplay = "Generator: RANDOM (" + cfg.entropyMode + ")";
            else if (cfg.entropyStr == "schematic") entDisplay = "Generator: SCHEMATIC Patterns";
            else if (cfg.entropyStr == "brainwallet") entDisplay = "Generator: BRAINWALLET (Sequential)";
            else entDisplay = "PRESET VALUE (Checking single entropy)";
        }
        else if (cfg.mnemonicOrder == "schematic") {
            entDisplay = "Generator: SCHEMATIC Patterns (via Order)";
        }

        std::cout << "Entropy: \033[1;36m" << entDisplay << "\033[0m\n\n";

        if (cfg.count > 0) std::cout << "Target: Stop after " << cfg.count << " seeds.\n";
        else std::cout << "\n";

        std::cout << "Seed # " << s.countId << "\033[K\n";
        std::cout << "Filtered Entropy: \033[1;35m" << s.entropyHex << "\033[0m\033[K\n";
        std::cout << "Phrase/Key: " << s.mnemonic << "\033[K\n";
        std::cout << "PrivKey:    " << s.hexKey << "\033[K\n\n";
        
        std::cout << "TYPE                PATH                                               ADDRESS                                                               STATUS\n";
        std::cout << "--------------------------------------------------------------------------------------\n";

        std::map<std::string, std::vector<DisplayState::AddrInfo>> grouped;
        for (const auto& row : s.rows) grouped[row.type].push_back(row);
        for (const auto& kv : grouped) {
            const std::vector<DisplayState::AddrInfo>& items = kv.second;
            if (items.empty()) continue;
            int idx = uiFrame % items.size();
            const auto& currentItem = items[idx];
            std::cout << std::left << std::setw(20) << currentItem.type << std::setw(28) << currentItem.path << std::setw(45) << currentItem.addr;
            if (currentItem.isHit) std::cout << "\033[1;32mHIT\033[0m"; else std::cout << "-";
            std::cout << "\033[K\n"; 
        }
        for(int i = (int)grouped.size(); i < 20; i++) std::cout << "\033[K\n"; 
        std::cout << "--------------------------------------------------------------------------------------\n";

        std::cout << "Total: " << formatUnits((double)totalAddressesChecked.load(), "addr") << " | Speed: " << formatUnits((double)totalAddressesChecked.load() / secs, "addr/s") << "   \033[K\n";
    }

    void workerClassBGPU(int gpuIdx) {
        bool hasEntropyStr = !cfg.entropyStr.empty();
        
        if (cfg.runMode == "xprv-mode") {
             std::mt19937_64 rng(std::random_device{}() + gpuIdx);
             
             while(running) {
                 if (cfg.count > 0 && realSeedsProcessed.load() >= cfg.count) { running = false; break; }
                 
                 unsigned long long currentId;
                 if (cfg.entropyMode == "schematic") currentId = totalSeedsChecked.fetch_add(1);
                 else currentId = rng(); 

                 std::string generatedXprv = xprvGen.generate(currentId, cfg.entropyMode);
                 MnemonicResult res = mnemonicTool.processSeed(0, "random", cfg.words, bloom, cfg.setAddress, "xprv", generatedXprv);
                 
                 bool anyHit = false;
                 for(auto& r : res.rows) {
                     if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                         if(r.isHit) {
                             logDetailedHit("XPRV_SCAN_HIT", std::to_string(currentId), generatedXprv, r.addr, "DERIVED", r.path);
                             anyHit = true;
                         }
                     }
                 }

                 if (gpuIdx == 0 && (currentId % 50 == 0)) {
                    std::lock_guard<std::mutex> lock(displayMutex); 
                    currentDisplay.countId = currentId; 
                    currentDisplay.mnemonic = generatedXprv; 
                    currentDisplay.entropyHex = "XPRV-" + cfg.entropyMode; 
                    currentDisplay.hexKey = "AUTO-GEN"; 
                    currentDisplay.rows.clear();
                    for(auto& r : res.rows) {
                        if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                             currentDisplay.rows.push_back({r.type, r.path, r.addr, (r.isHit ? "HIT" : "-"), r.isHit});
                        }
                    }
                 }

                 realSeedsProcessed.fetch_add(1);
                 totalAddressesChecked.fetch_add(res.rows.size());
             }
             return;
        }

        bool isKeyword = (cfg.entropyStr == "random" || cfg.entropyStr == "schematic" || 
                          cfg.entropyStr == "bip32" || cfg.entropyStr == "raw" ||
                          cfg.entropyStr == "brainwallet" || cfg.entropyStr == "password" ||
                          cfg.entropyStr == "alpha" || cfg.entropyStr == "num");
        
        bool isXprv = !cfg.xprv.empty();
        bool isPresetValue = (hasEntropyStr && !isKeyword) || isXprv;
        
        if (isPresetValue) {
            if (gpuIdx == 0) { 
                std::string phrase;
                if (isXprv) {
                    phrase = cfg.xprv;
                } else {
                    phrase = mnemonicTool.phraseFromEntropyString(cfg.entropyStr, cfg.entropyMode);
                }

                if (phrase.substr(0, 5) == "ERROR") { running = false; return; }
                
                MnemonicResult res = mnemonicTool.processSeed(0, "random", cfg.words, bloom, cfg.setAddress, "default", phrase);
                std::vector<DisplayState::AddrInfo> fRows;
                for(auto& r : res.rows) {
                    if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                        fRows.push_back({r.type, r.path, r.addr, (r.isHit ? "HIT" : "-"), r.isHit});
                        if(r.isHit) logDetailedHit("PRESET_HIT", cfg.entropyStr, res.mnemonic, r.addr, "DERIVED", r.path);
                    }
                }
                { 
                    std::lock_guard<std::mutex> lock(displayMutex); 
                    currentDisplay.countId = 1; 
                    currentDisplay.mnemonic = res.mnemonic; 
                    currentDisplay.entropyHex = isXprv ? "XPRV" : cfg.entropyStr; 
                    currentDisplay.hexKey = "PRESET"; 
                    currentDisplay.rows = fRows; 
                }
                running = false; 
            }
            return;
        }

        ActiveGpuContext& gpu = activeGpus[gpuIdx];
        if (gpu.backend == "CUDA") cudaSetDevice(gpu.deviceId);
        unsigned long long bSz = gpu.provider->getBatchSize();
        unsigned long long* fSeeds = new unsigned long long[1024]; int fCnt = 0;
        std::mt19937_64 rng(std::random_device{}() + gpuIdx);
        
        std::string genMode = "default";
        if (isKeyword) {
            if (cfg.entropyStr == "random") genMode = cfg.entropyMode;
            else genMode = cfg.entropyStr; 
        }

        std::vector<int> targetBits = cfg.akmBits;
        if (targetBits.empty()) targetBits.push_back(0); 
        int akmLen = (!cfg.akmLengths.empty()) ? cfg.akmLengths[0] : 10;
        bool isAkmSchematic = (cfg.runMode == "akm" && cfg.akmGenMode == "schematic");
        bool forceSequential = !cfg.startFrom.empty() || isAkmSchematic;

        while (running) {
            if (cfg.count > 0 && realSeedsProcessed.load() >= cfg.count) { running = false; break; }
            unsigned long long base = 0;
            
            if (cfg.mnemonicOrder == "sequential" || cfg.mnemonicOrder == "schematic" || forceSequential) {
                unsigned long long current = totalSeedsChecked.load();
                if (!forceSequential && cfg.count > 0 && current >= cfg.count) { running = false; break; }
                base = totalSeedsChecked.fetch_add(bSz);
            } else { 
                base = rng(); 
            }

            int currentTargetBit = 0;
            if (targetBits.size() > 1) {
                int rIdx = rng() % targetBits.size();
                currentTargetBit = targetBits[rIdx];
            } else {
                currentTargetBit = targetBits[0];
            }

            if (gpuIdx == 0) {
                currentActiveBit.store(currentTargetBit);
            }

            if (bloom.isLoaded() && gpu.backend == "CUDA") {
                for (size_t k = 0; k < bloom.getLayerCount(); k++) {
                    const uint8_t* blfData = bloom.getLayerData(k);
                    size_t blfSize = bloom.getLayerSize(k);

                    if (cfg.runMode == "akm") {
						bool sequential = (cfg.akmGenMode == "schematic" || !cfg.startFrom.empty());
                        const std::vector<uint8_t>& highBytes = akmTool.getHighBytes();
                        const void* prefixPtr = highBytes.empty() ? nullptr : highBytes.data();
                        int prefixLen = (int)highBytes.size();
						
                         launch_gpu_akm_search(base, bSz, cfg.cudaBlocks, cfg.cudaThreads, cfg.pointsPerThread, 
                                               blfData, blfSize, 
                                               fSeeds, &fCnt, currentTargetBit,
											   sequential, prefixPtr, prefixLen);
                         
                         if (fCnt > 0) {
                            std::string akmOrder = isAkmSchematic ? "schematic" : cfg.mnemonicOrder;
                            for (int i = 0; i < fCnt; i++) {
                                 AkmResult res = akmTool.processAkmSeed(fSeeds[i], akmOrder, currentTargetBit, akmLen, bloom);
                                 for (auto& r : res.rows) {
                                     if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                                         if (r.isHit) logDetailedHit("AKM_HIT_BIT_" + std::to_string(currentTargetBit), std::to_string(fSeeds[i]), res.phrase, r.addr, res.hexKey, "AKM_Derivation");
                                     }
                                 }
                            }
                            fCnt = 0;
                         }
                    } 
                    else {
                         launch_gpu_mnemonic_search(base, bSz, cfg.cudaBlocks, cfg.cudaThreads, 
                                                    blfData, blfSize, 
                                                    fSeeds, &fCnt);
                         if (fCnt > 0) {
                            for (int i = 0; i < fCnt; i++) {
                                 MnemonicResult res = mnemonicTool.processSeed(fSeeds[i], cfg.mnemonicOrder, cfg.words, bloom, cfg.setAddress, genMode);
                                 for (auto& r : res.rows) {
                                     if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                                         if (r.isHit) logDetailedHit("MNEM_GPU_HIT", std::to_string(fSeeds[i]), res.mnemonic, r.addr, "BIP39", r.path);
                                     }
                                 }
                            }
                            fCnt = 0;
                         }
                    }
                }
            }

            if (gpuIdx == 0) {
                unsigned long long visualSeed = base;
                if (cfg.mnemonicOrder == "random" && !forceSequential) visualSeed += (rng() % bSz);
                
                if (cfg.runMode == "akm") {
                    std::string akmOrder = isAkmSchematic ? "schematic" : cfg.mnemonicOrder;
                    AkmResult res = akmTool.processAkmSeed(visualSeed, akmOrder, currentTargetBit, akmLen, bloom);
                    { 
                        std::lock_guard<std::mutex> lock(displayMutex); 
                        currentDisplay.countId = visualSeed; 
                        currentDisplay.mnemonic = res.phrase;
                        currentDisplay.entropyHex = "AKM (Bit " + std::to_string(currentTargetBit) + ")";
                        currentDisplay.hexKey = res.hexKey;
                        currentDisplay.rows.clear();
                        for(auto& r : res.rows) {
                            if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                                currentDisplay.rows.push_back({r.type, r.path, r.addr, (r.isHit ? "HIT" : "-"), r.isHit});
                                if (r.isHit) logDetailedHit("AKM_UI_HIT", std::to_string(visualSeed), res.phrase, r.addr, res.hexKey, "AKM_Display");
                            }
                        }
                    }
                } else {
                    MnemonicResult res = mnemonicTool.processSeed(visualSeed, cfg.mnemonicOrder, cfg.words, bloom, cfg.setAddress, genMode);
                    { 
                        std::lock_guard<std::mutex> lock(displayMutex); 
                        currentDisplay.countId = visualSeed; 
                        currentDisplay.mnemonic = res.mnemonic;
                        currentDisplay.entropyHex = res.entropyHex; 
                        currentDisplay.hexKey = (genMode == "default") ? "BIP39" : genMode;
                        currentDisplay.rows.clear();
                        for(auto& r : res.rows) {
                            if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                                currentDisplay.rows.push_back({r.type, r.path, r.addr, (r.isHit ? "HIT" : "-"), r.isHit});
                                if (r.isHit) logDetailedHit("MNEM_UI_HIT", std::to_string(visualSeed), res.mnemonic, r.addr, "BIP39", r.path);
                            }
                        }
                    }
                }
            }
            
            realSeedsProcessed.fetch_add(bSz);
            totalAddressesChecked.fetch_add(bSz * 4);
        }
        delete[] fSeeds;
    }

    void start() {
        if (cfg.runMode == "check") {
            CheckMode checker(mnemonicTool, bloom, cfg, running);
            checker.run();
            return;
        }

        startTime = std::chrono::high_resolution_clock::now(); drawInterface();
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)activeGpus.size(); i++) {
            threads.emplace_back(&Runner::workerClassBGPU, this, i);
        }
        while (running) { 
            std::this_thread::sleep_for(std::chrono::milliseconds(50)); 
            updateStats();
            if (cfg.count > 0 && realSeedsProcessed.load() >= cfg.count) running = false;
        }
        for (auto& th : threads) if (th.joinable()) th.join();
        updateStats(); 
        restoreConsole();
        std::cout << "\n\n[INFO] Job Finished. Total seeds checked: " << realSeedsProcessed.load() << "\n";
    }
};
#endif