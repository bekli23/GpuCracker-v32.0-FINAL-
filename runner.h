#pragma once
#define _CRT_SECURE_NO_WARNINGS

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

#include "args.h"
#include "gpu_interface.h"
#include "bloom.h"
#include "cuda_provider.h"
#include "opencl_provider.h"
#include "utils.h"
#include "mnemonic.h"
#include "mode_akm.h"

#ifdef ENABLE_VULKAN
#include "vulkan_provider.h"
#include <vulkan/vulkan.h>
#endif
#include <CL/cl.h>

#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <secp256k1.h>

// Bridge către kernel-urile GPU (CUDA)
extern "C" void launch_gpu_akm_search(unsigned long long start, unsigned long long count, int b, int t, const void* bf, size_t sz, unsigned long long* res, int* cnt, int bits);
extern "C" void launch_gpu_mnemonic_search(unsigned long long start, unsigned long long count, int b, int t, const void* bf, size_t sz, unsigned long long* res, int* cnt);

// =============================================================
// GĂRZI CONSOLĂ ȘI PLATFORMĂ
// =============================================================
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
    std::string hexKey = "-";
    struct AddrInfo { std::string type, path, addr; std::string status; bool isHit; };
    std::vector<AddrInfo> rows;
    int currentBit = 0;
};

struct ActiveGpuContext { IGpuProvider* provider; std::string backend; int deviceId; int globalId; };

class Runner {
private:
    ProgramConfig cfg;
    BloomFilter bloom;
    MnemonicTool mnemonicTool;
    AkmTool akmTool;
    
    std::atomic<bool> running{ true };
    std::atomic<unsigned long long> totalSeedsChecked{ 0 };
    std::atomic<unsigned long long> totalAddressesChecked{ 0 };
    
    std::mutex displayMutex, fileMutex;
    DisplayState currentDisplay;
    std::vector<ActiveGpuContext> activeGpus;
    std::vector<unsigned char*> hostBuffers;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    
    int uiBaseLine = 22; int totalCores = 0; int workerCores = 0;
    bool isInputMode = false;

    bool isAddressTypeMatch(const std::string& typeStr, const std::string& pathStr, std::string filter) {
        if (filter.empty() || filter == "ALL") return true;
        std::string f = filter; std::transform(f.begin(), f.end(), f.begin(), ::toupper);
        std::string sZ = typeStr + " " + pathStr; std::transform(sZ.begin(), sZ.end(), sZ.begin(), ::toupper);
        if (f == "LEGACY" || f == "P2PKH") return (sZ.find("P2PKH") != std::string::npos || sZ.find("LEGACY") != std::string::npos);
        if (f == "P2SH") return (sZ.find("P2SH") != std::string::npos || sZ.find("NESTED") != std::string::npos);
        if (f == "SEGWIT") return (sZ.find("BECH32") != std::string::npos || sZ.find("NATIVE") != std::string::npos);
        return (sZ.find(f) != std::string::npos);
    }

    std::string formatUnits(double num, const std::string& unit) {
        const char* suffixes[] = { "", " K", " M", " B", " T", " Q" };
        int i = 0; while (num >= 1000 && i < 5) { num /= 1000; i++; }
        std::stringstream ss; ss << std::fixed << std::setprecision(2) << num << suffixes[i] << " " << unit;
        return ss.str();
    }

    void logDetailedHit(const std::string& mode, const std::string& info, const std::string& secret, const std::string& addr, const std::string& pk, const std::string& path) {
        std::lock_guard<std::mutex> lock(fileMutex);
        std::ofstream file(cfg.winFile, std::ios::app);
        if (!file.is_open()) return;
        file << "\n=== HIT " << mode << " ===\nSeed: " << info << "\nPhrase: " << secret << "\nAddress: " << addr << "\nPrivKey: " << pk << "\nPath: " << path << "\n================\n";
        file.close();
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
        if (cfg.runMode == "akm") {
            if (cfg.akmListProfiles) { akmTool.listProfiles(); exit(0); }
            akmTool.init(cfg.akmProfile, "akm/wordlist_512_ascii.txt");
        } else mnemonicTool.loadWordlist(cfg.language);
        if (!cfg.bloomFile.empty()) bloom.load(cfg.bloomFile);
        detectHardware();
    }

    ~Runner() { restoreConsole(); for (auto& g : activeGpus) if(g.provider) delete g.provider; for (auto* b : hostBuffers) delete[] b; }

    void drawInterface() {
        std::cout << "\033[2J\033[H=== GpuCracker v32.0 (FINAL) ===\n";
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
            std::cout << "GPU " << gpu.globalId << ": \033[1;33m" << gpu.provider->getName() << "\033[0m (" << gpu.backend << ")\n";
            std::cout << "Conf:   \033[1;36m" << gpu.provider->getConfig() << "\033[0m | VRAM: " << memInfo << "\n";
        }
        std::cout << "CPU:    Using " << workerCores << "/" << totalCores << " Cores (" << (workerCores>1?"Active":"Disabled") << ")\n";
        if (cfg.runMode == "akm") {
            int wordsNum = (!cfg.akmLengths.empty()) ? cfg.akmLengths[0] : 10;
            std::cout << "Mode:   AKM | Profile: " << cfg.akmProfile << "\nGen:    " << cfg.akmGenMode << " | Words: " << wordsNum << "\n";
            if (!cfg.akmBits.empty()) {
                std::cout << "Ranges: "; for(int b : cfg.akmBits) std::cout << "2^" << b << " "; std::cout << "\n";
            }
        }
        std::cout << "Filter: " << (cfg.setAddress.empty() ? "ALL" : cfg.setAddress) << "\n\n";

        // Adăugare Classes of Attack Strategy
        std::cout << "\033[1;37mClasses of Attack Strategy:\033[0m\n";
        std::cout << "Class A: 10K h/s | \033[1;32mClass B: 1M h/s (GPU Experimental)\033[0m\n";
        std::cout << "Class C: 100M h/s | Class D: 1B+ h/s\n";
        std::cout << "Status: \033[1;33mClass B GPU logic experimental active" << (cfg.showRealSpeed ? " [REAL]" : "") << "\033[0m\n";

        uiBaseLine = 22; moveTo(uiBaseLine, 1);
        std::cout << "Seed #\nPhrase:\nPrivKey:\n\nTYPE      PATH                      ADDRESS                                     STATUS\n--------------------------------------------------------------------------------------\n";
    }

    void updateStats() {
        DisplayState s; { std::lock_guard<std::mutex> lock(displayMutex); s = currentDisplay; }
        double secs = std::max(0.001, std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - startTime).count());
        moveTo(uiBaseLine, 8); std::cout << s.countId << (cfg.runMode == "akm" ? " [Bit: " + std::to_string(s.currentBit) + "]" : "") << "\033[K";
        moveTo(uiBaseLine + 1, 9); std::cout << s.mnemonic << "\033[K";
        moveTo(uiBaseLine + 2, 11); std::cout << s.hexKey << "\033[K";
        int r = uiBaseLine + 6;
        for (const auto& row : s.rows) {
            moveTo(r++, 1); std::cout << std::left << std::setw(10) << row.type << std::setw(26) << row.path << std::setw(45) << row.addr;
            if (row.isHit) std::cout << "\033[1;32m" << row.status << "\033[0m"; else std::cout << "-";
            std::cout << "\033[K";
        }
        moveTo(r + 1, 1); std::cout << "Total: " << formatUnits((double)totalAddressesChecked.load(), "addr") << " | Speed: " << formatUnits((double)totalAddressesChecked.load() / secs, "addr/s") << "\033[K" << std::flush;
    }

    void workerClassBGPU(int gpuIdx) {
        ActiveGpuContext& gpu = activeGpus[gpuIdx];
        if (gpu.backend == "CUDA") cudaSetDevice(gpu.deviceId);
        unsigned long long bSz = gpu.provider->getBatchSize();
        unsigned long long* fSeeds = new unsigned long long[1024]; int fCnt = 0;
        std::mt19937_64 rng(std::random_device{}() + gpuIdx); size_t bRotIdx = 0;
        int pLen = (!cfg.akmLengths.empty()) ? cfg.akmLengths[0] : 10;
        auto lastUi = std::chrono::high_resolution_clock::now();

        while (running) {
            unsigned long long base;
            int bit = (cfg.runMode == "akm" && !cfg.akmBits.empty()) ? cfg.akmBits[bRotIdx++ % cfg.akmBits.size()] : 0;
            base = (cfg.mnemonicOrder == "random") ? rng() : totalSeedsChecked.fetch_add(bSz);

            if (bloom.isLoaded() && gpu.backend == "CUDA") {
                if (cfg.runMode == "akm") launch_gpu_akm_search(base, bSz, cfg.cudaBlocks, cfg.cudaThreads, bloom.getRawData(), bloom.getSize(), fSeeds, &fCnt, bit);
                else launch_gpu_mnemonic_search(base, bSz, cfg.cudaBlocks, cfg.cudaThreads, bloom.getRawData(), bloom.getSize(), fSeeds, &fCnt);
            }

            if (fCnt > 0) {
                for (int i = 0; i < fCnt; i++) {
                    AkmResult res = akmTool.processAkmSeed(fSeeds[i], cfg.mnemonicOrder, bit, pLen, bloom);
                    for (auto& r : res.rows) if (r.isHit) logDetailedHit("GPU_HIT", std::to_string(fSeeds[i]), res.phrase, r.addr, res.hexKey, r.path);
                }
                fCnt = 0;
            }

            if (gpuIdx == 0 && std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - lastUi).count() > 1000) {
                lastUi = std::chrono::high_resolution_clock::now();
                AkmResult res = akmTool.processAkmSeed(base, cfg.mnemonicOrder, bit, pLen, bloom);
                std::vector<DisplayState::AddrInfo> fRows;
                for(auto& r : res.rows) {
                    if (isAddressTypeMatch(r.type, r.path, cfg.setAddress)) {
                        fRows.push_back({r.type, r.path, r.addr, r.status, r.isHit});
                        if (r.isHit) logDetailedHit("UI_HIT", std::to_string(base), res.phrase, r.addr, res.hexKey, r.path);
                    }
                }
                { std::lock_guard<std::mutex> lock(displayMutex); currentDisplay.countId = base; currentDisplay.currentBit = bit; currentDisplay.mnemonic = res.phrase; currentDisplay.hexKey = res.hexKey; currentDisplay.rows = fRows; }
            }
            totalAddressesChecked.fetch_add(cfg.showRealSpeed ? bSz : bSz * 4);
        }
        delete[] fSeeds;
    }

    void workerLegacyHybrid(int gpuIdx) {
        ActiveGpuContext& gpu = activeGpus[gpuIdx];
        unsigned char* buffer = hostBuffers[gpuIdx]; int bSz = gpu.provider->getBatchSize();
        std::vector<Scheme> schemes = {{"LEGACY", "m/0/0", 0}, {"SEGWIT", "m/84'/0'/0'/0/0", 2}};
        while(running) {
            unsigned long long base = totalSeedsChecked.fetch_add(bSz);
            gpu.provider->generate(buffer, base, 32);
            #pragma omp parallel num_threads(workerCores)
            {
                secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
                #pragma omp for schedule(guided)
                for (int i = 0; i < bSz; ++i) {
                    std::vector<uint8_t> priv(buffer + (i * 32), buffer + (i * 32) + 32);
                    for (const auto& s : schemes) {
                        secp256k1_pubkey pub; if (!secp256k1_ec_pubkey_create(ctx, &pub, priv.data())) continue;
                        uint8_t cPub[33]; size_t len = 33; secp256k1_ec_pubkey_serialize(ctx, cPub, &len, &pub, SECP256K1_EC_COMPRESSED);
                        std::vector<uint8_t> vPub(cPub, cPub + 33), payload; Hash160(vPub, payload);
                        if (bloom.isLoaded() && bloom.check_hash160(payload)) logDetailedHit("HYBRID", "HIT", "-", PubKeyToNativeSegwit(vPub), toHex(priv), s.path);
                    }
                }
                secp256k1_context_destroy(ctx);
            }
            totalAddressesChecked.fetch_add(bSz * schemes.size());
        }
    }

    void start() {
        startTime = std::chrono::high_resolution_clock::now(); drawInterface();
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)activeGpus.size(); i++) {
            if (activeGpus[i].backend == "CUDA" && bloom.isLoaded() && !isInputMode) threads.emplace_back(&Runner::workerClassBGPU, this, i);
            else threads.emplace_back(&Runner::workerLegacyHybrid, this, i);
        }
        while (running) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); updateStats(); }
        for (auto& th : threads) if (th.joinable()) th.join();
        restoreConsole();
    }
};