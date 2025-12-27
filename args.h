#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>

struct ProgramConfig {
    bool infinite = false;
    unsigned long long count = 0; // 0 = infinit, sau limita setata de --count
    
    // --- MAIN MODES ---
    std::string runMode = "mnemonic"; // default: mnemonic

    // --- AKM SPECIFIC ---
    std::string akmProfile = "auto-linear"; 
    std::string akmPhrase = "";        
    std::string akmFile = "";          
    std::vector<int> akmLengths;       
    std::string akmGenMode = "random"; // random, schematic, wave
    bool akmListProfiles = false;      
    std::vector<int> akmBits;          

    // --- GPU SETTINGS ---
    std::string gpuType = "auto";      // cuda, opencl, vulkan, auto
    int platformId = 0;            
    int deviceId = -1;                 // -1 reprezinta "toate dispozitivele" (auto)
    
    // --- PERFORMANCE ---
    int cudaBlocks = 128;
    int cudaThreads = 128;
    int pointsPerThread = 1; 
    int cpuCores = 0;                  // 0 = Auto detect
    
    // --- FILES & DATA ---
    std::string bloomFile;
    std::string winFile = "win.txt";
    std::string wordlistPath = "bip39\\english.txt";
    
    // --- MNEMONIC SETTINGS ---
    std::string language = "english";  
    int words = 0;                     // 0 default pentru detectie automata
    
    // --- STRATEGIES ---
    std::string mnemonicOrder = "random"; // random sau sequential
    std::string mnemonicPrefix = "";      
    std::string mnemonicBrute = "";       
    
    // --- ADDRESS SETTINGS ---
    // Suporta: ALL, LEGACY, P2PKH, P2SH, SEGWIT, P2WPKH, P2WSH, TAPROOT, BARE
    std::string setAddress = "ALL";

    // --- OPTIUNI VITEZA ---
    bool showRealSpeed = false; // Flag pentru --speed (viteza reala fara multiplicatori)
    
    // --- FLAGS ---
    bool help = false; 
    bool verbose = true; 
};

// Helper: transforma string-ul "71,72" intr-un vector de intregi
inline std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try { 
            item.erase(std::remove_if(item.begin(), item.end(), ::isspace), item.end());
            if(!item.empty()) res.push_back(std::stoi(item)); 
        } catch(...) {}
    }
    return res;
}

inline ProgramConfig parseArgs(int argc, char* argv[]) {
    ProgramConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // --- MODES & PROFILES ---
        if (arg == "--mode" && i + 1 < argc) cfg.runMode = argv[++i];
        else if (arg == "--profile" && i + 1 < argc) cfg.akmProfile = argv[++i];

        // --- AKM SPECIFIC COMMANDS ---
        else if (arg == "--akm-phrase" && i + 1 < argc) cfg.akmPhrase = argv[++i];
        else if ((arg == "--akm-file" || arg == "--input") && i + 1 < argc) cfg.akmFile = argv[++i];
        else if (arg == "--akm-word" && i + 1 < argc) cfg.akmLengths = parseIntList(argv[++i]);
        else if (arg == "--akm-mode" && i + 1 < argc) cfg.akmGenMode = argv[++i];
        else if (arg == "--akm-list-profiles") cfg.akmListProfiles = true;
        else if (arg == "--akm-bit" && i + 1 < argc) cfg.akmBits = parseIntList(argv[++i]);

        // --- GENERAL ---
        else if (arg == "--infinite") cfg.infinite = true;
        else if (arg == "--count" && i + 1 < argc) cfg.count = std::stoull(argv[++i]);
        else if (arg == "--speed") cfg.showRealSpeed = true;
        else if (arg == "--help" || arg == "-h") cfg.help = true;
        else if (arg == "--quiet") cfg.verbose = false;
        
        // --- GPU CONFIG ---
        else if (arg == "--type" && i + 1 < argc) cfg.gpuType = argv[++i];
        else if (arg == "--platform" && i + 1 < argc) cfg.platformId = std::stoi(argv[++i]);
        else if ((arg == "--device" || arg == "-d") && i + 1 < argc) cfg.deviceId = std::stoi(argv[++i]);
        else if ((arg == "--blocks" || arg == "-b") && i + 1 < argc) cfg.cudaBlocks = std::stoi(argv[++i]);
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) cfg.cudaThreads = std::stoi(argv[++i]);
        else if ((arg == "--points" || arg == "-p") && i + 1 < argc) cfg.pointsPerThread = std::stoi(argv[++i]);
        
        // --- CPU CONTROL ---
        else if (arg == "--cores" && i + 1 < argc) cfg.cpuCores = std::stoi(argv[++i]);
        
        // --- FILES ---
        else if (arg == "--bloom-keys" && i + 1 < argc) cfg.bloomFile = argv[++i];
        else if (arg == "--win" && i + 1 < argc) cfg.winFile = argv[++i];

        // --- MNEMONIC SETTINGS ---
        else if (arg == "--langs" && i + 1 < argc) cfg.language = argv[++i];
        else if (arg == "--words" && i + 1 < argc) cfg.words = std::stoi(argv[++i]);
        else if (arg == "--mnemonic-order" && i + 1 < argc) cfg.mnemonicOrder = argv[++i];

        // --- STRATEGIES ---
        else if (arg == "--mnemonic-prefix" && i + 1 < argc) cfg.mnemonicPrefix = argv[++i];
        else if (arg == "--mnemonic-brute" && i + 1 < argc) cfg.mnemonicBrute = argv[++i];
        else if (arg == "--setaddress" && i + 1 < argc) cfg.setAddress = argv[++i];
    }

    // Logica de stabilire a lungimii implicite
    if (cfg.runMode == "akm") {
        if (cfg.akmLengths.empty()) {
            if (cfg.words > 0) cfg.akmLengths = { cfg.words };
            else cfg.akmLengths = { 10 }; 
        }
    } else {
        if (cfg.words == 0) cfg.words = 12;
    }

    return cfg;
}

inline void printHelp() {
    std::cout << "==============================================================\n";
    std::cout << "   GpuCracker v32.0 (FINAL) - Modular Architecture\n";
    std::cout << "==============================================================\n\n";
    
    std::cout << "Usage: GpuCracker.exe [OPTIONS]\n\n";
    
    std::cout << "--- OPERATING MODES ---\n";
    std::cout << "  --mode NAME         Select operation mode (Default: mnemonic)\n";
    std::cout << "                      Available: mnemonic, akm\n";
    
    std::cout << "\n--- AKM MODE SETTINGS ---\n";
    std::cout << "  --profile NAME      Select AKM Profile (e.g., akm3-puzzle71)\n";
    std::cout << "  --input FILE        Input text file with phrases (Alias for --akm-file)\n";
    std::cout << "  --akm-phrase STR    Check a single specific phrase\n";
    std::cout << "  --words N           Set FIXED length for AKM gen (overrides default list)\n";
    std::cout << "  --akm-word LIST     Set multiple lengths (e.g. '3,5,10').\n";
    std::cout << "  --akm-mode MODE     Gen mode: random | schematic | wave\n";
    std::cout << "  --akm-bit LIST      Limit bits (e.g. '71,72')\n";
    std::cout << "  --akm-list-profiles List available profiles\n";

    std::cout << "\n--- ESSENTIAL ---\n";
    std::cout << "  --bloom-keys FILE   Path to the Bloom Filter file (.blf)\n";
    std::cout << "  --infinite          Run continuously without stopping.\n";
    std::cout << "  --count N           Stop after checking N seeds.\n";
    std::cout << "  --speed             Show real GPU seed generation speed (no multipliers).\n";
    std::cout << "  --win FILE          File path to save successful hits (Default: win.txt)\n";
    
    std::cout << "\n--- CPU PERFORMANCE ---\n";
    std::cout << "  --cores N           Limit the number of CPU cores used.\n";

    std::cout << "\n--- GPU CONFIGURATION ---\n";
    std::cout << "  --type TYPE         Backend: 'cuda', 'opencl', 'vulkan', or 'auto'\n";
    std::cout << "  --device N          GPU device ID.\n";
    std::cout << "  --blocks N, --threads N, --points N  (Performance tuning)\n";

    std::cout << "\n--- MNEMONIC MODE SETTINGS ---\n";
    std::cout << "  --langs NAME        Language file (english, french, etc.)\n";
    std::cout << "  --words N           12, 15, 18, 21, 24 words.\n";
    std::cout << "  --mnemonic-order    'random' or 'sequential'\n";

    std::cout << "\n--- ADDRESS SETTINGS ---\n";
    std::cout << "  --setaddress TYPE   ALL, LEGACY, P2PKH, P2SH, SEGWIT,\n";
    std::cout << "                      P2WPKH, P2WSH, TAPROOT, BARE\n";
    std::cout << "==============================================================\n";
}