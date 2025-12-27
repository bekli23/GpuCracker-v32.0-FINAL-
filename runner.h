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

extern "C" void launch_gpu_akm_search(unsigned long long start, unsigned long long count, int b, int t, const void* bf, size_t sz, unsigned long long* res, int* cnt, int bits);
extern "C" void launch_gpu_mnemonic_search(unsigned long long start, unsigned long long count, int b, int t, const void* bf, size_t sz, unsigned long long* res, int* cnt);

#ifdef _WIN32
#include <windows.h>
inline void setupConsole() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0; GetConsoleMode(hOut, &dwMode); dwMode |= 0x0004; 
    SetConsoleMode(hOut, dwMode); SetConsoleOutputCP(65001); 
    CONSOLE_CURSOR_INFO ci; GetConsoleCursorInfo(hOut, &ci); ci.bVisible = false; SetConsoleCursorInfo(hOut, &ci);
}
#else
    // Add any Linux-specific headers if needed, otherwise leave empty
    #include <unistd.h>
inline void restoreConsole() {
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO ci; GetConsoleCursorInfo(hOut, &ci); ci.bVisible = true; SetConsoleCursorInfo(hOut, &ci);
}
#else
inline void setupConsole() {} inline void restoreConsole() {}
#endif

inline void moveTo(int r, int c) { std::cout << "\033[" << r << ";" << c << "H"; }

struct DisplayState {
    unsigned long long countId = 0;
    std::string mnemonic = "-";
    std::string hexKey = "-";
    struct AddrInfo { std::string type, path, addr; bool isHit; };
    std::vector<AddrInfo> rows;
    int currentBit = 0;
};

struct ActiveGpuContext { 
    IGpuProvider* provider; 
    std::string backend; 
    int deviceId; 
    int globalId;
};

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
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    
    int uiBaseLine = 22; 
    int totalCores = 0;
    int workerCores = 0;

    bool isAddressTypeMatch(const std::string& typeStr, const std::string& pathStr, std::string filter) {
        if (filter.empty() || filter == "ALL") return true;
        std::string f = filter; std::transform(f.begin(), f.end(), f.begin(), ::toupper);
        std::string searchZone = typeStr + " " + pathStr;
        std::transform(searchZone.begin(), searchZone.end(), searchZone.begin(), ::toupper);
        
        if (f == "LEGACY" || f == "P2PKH") {
            return (searchZone.find("P2PKH") != std::string::npos || searchZone.find("LEGACY") != std::string::npos || 
                    searchZone.find("BIP44") != std::string::npos || searchZone.find("M/44'") != std::string::npos);
        }
        if (f == "P2SH") {
            return (searchZone.find("P2SH") != std::string::npos || searchZone.find("NESTED") != std::string::npos || 
                    searchZone.find("BIP49") != std::string::npos || searchZone.find("M/49'") != std::string::npos);
        }
        if (f == "SEGWIT" || f == "P2WPKH") {
            return (searchZone.find("BECH32") != std::string::npos || searchZone.find("P2WPKH") != std::string::npos || 
                    searchZone.find("NATIVE") != std::string::npos || searchZone.find("BIP84") != std::string::npos ||
                    searchZone.find("M/84'") != std::string::npos);
        }
        return (searchZone.find(f) != std::string::npos);
    }

    std::string formatUnits(double num, const std::string& unit) {
        const char* suffixes[] = { "", " K", " M", " B", " T", " Q" };
        int i = 0;
        while (num >= 1000 && i < 5) { num /= 1000; i++; }
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << num << suffixes[i] << " " << unit;
        return ss.str();
    }

    void logDetailedHit(const std::string& mode, const std::string& info, const std::string& secret, const std::string& addr, const std::string& pk, const std::string& path) {
        std::lock_guard<std::mutex> lock(fileMutex);
        std::ofstream file(cfg.winFile, std::ios::app);
        if (!file.is_open()) return;
        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        file << "\n=== HIT " << mode << " [" << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "] ===\n"
             << "Seed: " << info << "\nPhrase: " << secret << "\nAddress: " << addr << "\nPrivKey: " << pk << "\nPath: " << path << "\n================\n";
    }

    void detectHardware() {
        int globalIndex = 0;
        bool useAll = (cfg.deviceId == -1);
        std::vector<std::string> addedDeviceNames;

        if (cfg.gpuType == "auto" || cfg.gpuType == "cuda") {
            int cudaCount = 0;
            if (cudaGetDeviceCount(&cudaCount) == cudaSuccess) {
                for (int i = 0; i < cudaCount; i++) {
                    if (useAll || cfg.deviceId == globalIndex) {
                        try {
                            auto* p = new CudaProvider(i, cfg.cudaBlocks, cfg.cudaThreads, cfg.pointsPerThread, true);
                            p->init();
                            activeGpus.push_back({ p, "CUDA", i, globalIndex });
                            addedDeviceNames.push_back(p->getName());
                        } catch (...) {}
                    }
                    globalIndex++;
                }
            }
        }

        if (cfg.gpuType == "auto" || cfg.gpuType == "opencl") {
            cl_uint nPlat = 0; clGetPlatformIDs(0, nullptr, &nPlat);
            if (nPlat > 0) {
                std::vector<cl_platform_id> platforms(nPlat); clGetPlatformIDs(nPlat, platforms.data(), nullptr);
                for (auto& platform : platforms) {
                    cl_uint nDev = 0; clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &nDev);
                    if (nDev > 0) {
                        std::vector<cl_device_id> devices(nDev); clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, nDev, devices.data(), nullptr);
                        for (int i = 0; i < (int)nDev; i++) {
                            char name[128]; clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 128, name, nullptr);
                            std::string devName(name);
                            bool isDup = false;
                            for(const auto& a : addedDeviceNames) if(devName.find(a) != std::string::npos) isDup = true;
                            if (cfg.gpuType == "auto" && isDup) { globalIndex++; continue; } 

                            if (useAll || cfg.deviceId == globalIndex) {
                                try {
                                    int totalThreads = cfg.cudaBlocks * cfg.cudaThreads;
                                    auto* p = new OpenClProvider(0, i, totalThreads, cfg.pointsPerThread, true);
                                    p->init();
                                    activeGpus.push_back({ p, "OpenCL", i, globalIndex });
                                } catch (...) {}
                            }
                            globalIndex++;
                        }
                    }
                }
            }
        }
    }

public:
    Runner(ProgramConfig c) : cfg(c) {
        totalCores = std::thread::hardware_concurrency();
        workerCores = (cfg.cpuCores > 0) ? cfg.cpuCores : totalCores;
        setupConsole();
        if (cfg.runMode == "mnemonic") mnemonicTool.loadWordlist("english.txt");
        else if (cfg.runMode == "akm") akmTool.init(cfg.akmProfile, "akm\\wordlist_512_ascii.txt");
        if (!cfg.bloomFile.empty()) bloom.load(cfg.bloomFile);
        detectHardware();
    }

    ~Runner() { restoreConsole(); for (auto& g : activeGpus) if(g.provider) delete g.provider; }

    void drawInterface() {
        std::cout << "\033[2J\033[H=== GpuCracker v32.0 (FINAL) ===\n";
        for (const auto& gpu : activeGpus) {
            std::string memInfo = "Shared/Global";
            if (gpu.backend == "CUDA") {
                size_t f = 0, t = 0; cudaSetDevice(gpu.deviceId); 
                if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
                    std::stringstream ss; ss << (t-f)/(1024*1024) << "/" << t/(1024*1024) << " MB";
                    memInfo = ss.str();
                }
            }
            std::cout << "GPU " << gpu.globalId << ": \033[1;33m" << gpu.provider->getName() << "\033[0m (" << gpu.backend << ")\n";
            std::cout << "Conf:   \033[1;36m" << gpu.provider->getConfig() << "\033[0m | VRAM: " << memInfo << "\n";
        }
        std::string ompStatus = (workerCores > 1) ? "\033[1;32mActive (OpenMP)\033[0m" : "\033[1;31mDisabled\033[0m";
        std::cout << "CPU:    Using " << workerCores << "/" << totalCores << " Cores (" << ompStatus << ")\n";

        if (cfg.runMode == "akm") {
            int wordsNum = (!cfg.akmLengths.empty()) ? cfg.akmLengths[0] : cfg.words;
            std::cout << "Mode:   AKM | Profile: " << cfg.akmProfile << "\n";
            std::cout << "Gen:    " << cfg.akmGenMode << " | Words: " << wordsNum << "\n";
            if (!cfg.akmBits.empty()) {
                std::cout << "Ranges: "; for(int b : cfg.akmBits) std::cout << "2^" << b << " "; std::cout << "\n";
            }
        } else {
            std::cout << "Lang:   " << cfg.language << " | Words: " << cfg.words << " | Order: " << cfg.mnemonicOrder << "\n";
        }
        std::cout << "Filter: " << (cfg.setAddress.empty() ? "ALL" : cfg.setAddress) << "\n";

        std::cout << "\n\033[1;37mClasses of Attack Strategy:\033[0m\n";
        std::cout << "Class A: 10K h/s | \033[1;32mClass B: 1M h/s (GPU Experimental)\033[0m\n";
        std::cout << "Class C: 100M h/s | Class D: 1B+ h/s\n";
        std::cout << "Status: \033[1;33mClass B GPU logic experimental active" 
                  << (cfg.showRealSpeed ? " [REAL SPEED MODE]" : "") << "\033[0m\n";

        uiBaseLine = 22; 
        moveTo(uiBaseLine, 1);
        std::cout << "Seed #\nPhrase:\nPrivKey:\n\nTYPE      PATH                      ADDRESS                                     STATUS\n--------------------------------------------------------------------------------------\n";
    }

    void updateStats() {
        DisplayState s; { std::lock_guard<std::mutex> lock(displayMutex); s = currentDisplay; }
        auto now = std::chrono::high_resolution_clock::now();
        double secs = std::chrono::duration<double>(now - startTime).count();
        if (secs <= 0) secs = 0.001;

        moveTo(uiBaseLine, 8); 
        std::cout << s.countId << (cfg.runMode == "akm" ? " [Bit: " + std::to_string(s.currentBit) + "]" : "") << "\033[K";
        moveTo(uiBaseLine + 1, 9); std::cout << s.mnemonic << "\033[K";
        moveTo(uiBaseLine + 2, 11); std::cout << s.hexKey << "\033[K";
        
        int r = uiBaseLine + 6;
        for (const auto& row : s.rows) {
            moveTo(r++, 1);
            std::cout << std::left << std::setw(10) << row.type << std::setw(26) << row.path << std::setw(45) << row.addr;
            if (row.isHit) std::cout << "\033[1;32mHIT\033[0m"; else std::cout << "-";
            std::cout << "\033[K";
        }
        
        std::string speedLabel = cfg.showRealSpeed ? "Real Speed: " : "Speed: ";
        moveTo(r + 1, 1);
        std::cout << "Total: " << formatUnits((double)totalAddressesChecked.load(), "addr") 
                  << " | " << speedLabel << "\033[1;32m" << formatUnits((double)totalAddressesChecked.load() / secs, "addr/s") << "\033[0m\033[K" << std::flush;
    }

    void workerClassBGPU(int gpuIdx) {
        ActiveGpuContext& gpu = activeGpus[gpuIdx];
        if (gpu.backend == "CUDA") cudaSetDevice(gpu.deviceId);
        unsigned long long batchSize = gpu.provider->getBatchSize();
        unsigned long long* foundSeeds = new unsigned long long[1024];
        int foundCount = 0;
        std::mt19937_64 rng(std::random_device{}() + gpuIdx);
        size_t bitRotationIdx = 0;
        int phraseLen = (!cfg.akmLengths.empty()) ? cfg.akmLengths[0] : cfg.words;

        auto lastUiUpdate = std::chrono::high_resolution_clock::now();

        while (running) {
            unsigned long long currentTaskSize = batchSize;
            unsigned long long base;
            int bit = 0;
            if (cfg.runMode == "akm" && !cfg.akmBits.empty()) {
                bit = cfg.akmBits[bitRotationIdx % cfg.akmBits.size()]; bitRotationIdx++;
            }

            if (!cfg.infinite && cfg.count > 0) {
                base = totalSeedsChecked.fetch_add(batchSize);
                if (base >= (unsigned long long)cfg.count) { running = false; break; }
                if (base + batchSize > (unsigned long long)cfg.count) currentTaskSize = (unsigned long long)cfg.count - base;
            } else {
                base = (cfg.mnemonicOrder == "random") ? rng() : totalSeedsChecked.fetch_add(batchSize);
            }

            if (bloom.isLoaded()) {
                if (gpu.backend == "CUDA") {
                    if (cfg.runMode == "akm") launch_gpu_akm_search(base, currentTaskSize, cfg.cudaBlocks, cfg.cudaThreads, bloom.getRawData(), bloom.getSize(), foundSeeds, &foundCount, bit);
                    else launch_gpu_mnemonic_search(base, currentTaskSize, cfg.cudaBlocks, cfg.cudaThreads, bloom.getRawData(), bloom.getSize(), foundSeeds, &foundCount);
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            if (foundCount > 0) {
                for (int i = 0; i < foundCount; i++) {
                    if (cfg.runMode == "akm") {
                         AkmResult res = akmTool.processAkmSeed(foundSeeds[i], cfg.mnemonicOrder, bit, phraseLen, bloom);
                         for (auto& r : res.rows) if (r.isHit) logDetailedHit("HIT AKM", std::to_string(foundSeeds[i]), res.phrase, r.addr, res.hexKey, r.path);
                    } else {
                         MnemonicResult res = mnemonicTool.processSeed(foundSeeds[i], cfg.mnemonicOrder, cfg.words, bloom);
                         for (auto& r : res.rows) if (r.isHit) logDetailedHit("HIT MNEM", std::to_string(foundSeeds[i]), res.mnemonic, r.addr, "-", r.path);
                    }
                }
            }

            auto now = std::chrono::high_resolution_clock::now();
            if (gpuIdx == 0 && std::chrono::duration_cast<std::chrono::milliseconds>(now - lastUiUpdate).count() > 1000) {
                lastUiUpdate = now;
                std::vector<DisplayState::AddrInfo> fRows;
                if (cfg.runMode == "akm") {
                    AkmResult res = akmTool.processAkmSeed(base, cfg.mnemonicOrder, bit, phraseLen, bloom);
                    for(auto& row : res.rows) if (isAddressTypeMatch(row.type, row.path, cfg.setAddress)) fRows.push_back({row.type, row.path, row.addr, row.isHit});
                    { std::lock_guard<std::mutex> lock(displayMutex); currentDisplay.countId = base; currentDisplay.currentBit = bit; currentDisplay.mnemonic = res.phrase; currentDisplay.hexKey = res.hexKey; currentDisplay.rows = fRows; }
                } else {
                    MnemonicResult res = mnemonicTool.processSeed(base, cfg.mnemonicOrder, cfg.words, bloom);
                    for(auto& row : res.rows) if (isAddressTypeMatch(row.type, row.path, cfg.setAddress)) fRows.push_back({row.type, row.path, row.addr, row.isHit});
                    { std::lock_guard<std::mutex> lock(displayMutex); currentDisplay.countId = base; currentDisplay.mnemonic = res.mnemonic; currentDisplay.hexKey = "BIP39"; currentDisplay.rows = fRows; }
                }
            }

            // LOGICA SELECTIVA PENTRU VITEZA
            unsigned long long increment = cfg.showRealSpeed ? currentTaskSize : (currentTaskSize * 4);
            totalAddressesChecked.fetch_add(increment);
        }
        delete[] foundSeeds;
    }

    void start() {
        drawInterface(); startTime = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;
        for (int i = 0; i < (int)activeGpus.size(); i++) threads.emplace_back(&Runner::workerClassBGPU, this, i);
        while (running) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); updateStats(); }
        for (auto& th : threads) if (th.joinable()) th.join();
        updateStats(); restoreConsole();
    }
};