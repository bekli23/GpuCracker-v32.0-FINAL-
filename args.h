#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cctype>

// =============================================================
// CONFIGURATION STRUCTURE
// =============================================================
struct ProgramConfig {
    bool infinite = false;
    unsigned long long count = 0; // 0 = infinite (default), or limit set by --count
    
    // --- MAIN MODES ---
    // Options: "mnemonic", "akm", "check", "scan", "xprv-mode"
    std::string runMode = "mnemonic"; 

    // --- CHECK MODE CONFIG ---
    std::string inputFile = "";       // --file
    std::string execCommand = "";     // --exec
    std::string socketPath = "";      // --socket

    // --- BRAINWALLET MODE ---
    // Options: "random", "alpha", "num", "schematic", "hex", "clever"
    std::string brainwalletMode = ""; 

    // --- XPRV MODE ---
    std::string xprv = ""; // --xprv (single key check)

    // --- AKM SPECIFIC ---
    std::string akmProfile = "auto-linear"; 
    std::string akmPhrase = "";        
    std::string akmFile = "";            
    std::vector<int> akmLengths;        
    std::string akmGenMode = "random"; // random, schematic, wave
    bool akmListProfiles = false;        
    std::vector<int> akmBits;            

    // --- RESUME / CONTINUE ---
    std::string startFrom = "";       // --continue <HEX>

    // --- GPU SETTINGS ---
    std::string gpuType = "auto";      // cuda, opencl, vulkan, auto
    int platformId = 0;                
    int deviceId = -1;                 // -1 = all devices
    
    // --- PERFORMANCE ---
    int cudaBlocks = 128;
    int cudaThreads = 128;
    int pointsPerThread = 1; 
    int cpuCores = 0;                  // 0 = Auto
    
    // --- FILES & DATA ---
    // [MODIFICAT] Suport pentru multiple fisiere BLF
    std::vector<std::string> bloomFiles; 
    std::string winFile = "win.txt";
    std::string wordlistPath = "bip39/english.txt";
    
    // --- CUSTOM PATHS & ENTROPY ---
    std::string pathFile = ""; 
    std::string entropyMode = "default"; // hex, bin, dice, random, schematic
    std::string entropyStr = "";         // Raw entropy or Seed
    
    // --- MULTI COIN ---
    std::string multiCoin = "btc";       // ex: "btc,eth"

    // --- MNEMONIC SETTINGS ---
    std::string language = "english";  
    int words = 0;                     
    
    // --- STRATEGIES ---
    std::string mnemonicOrder = "random"; // random, sequential, schematic
    std::string mnemonicPrefix = "";      
    std::string mnemonicBrute = "";       
    
    // --- ADDRESS SETTINGS ---
    std::string setAddress = "ALL";

    // --- FLAGS ---
    bool showRealSpeed = false; 
    bool help = false; 
    bool verbose = true; 
};

// =============================================================
// ARGUMENT PARSER HELPERS
// =============================================================

// Helper: Parse comma-separated integers (e.g., "12,15,24")
inline std::vector<int> parseIntList(const std::string& s) {
    std::vector<int> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try { 
            // Remove whitespace
            item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c){ return std::isspace(c); }), item.end());
            if(!item.empty()) res.push_back(std::stoi(item)); 
        } catch(...) {}
    }
    return res;
}

// [MODIFICAT] Helper: Parse comma-separated strings (e.g., "file1.blf,file2.blf")
inline std::vector<std::string> parseStringList(const std::string& s) {
    std::vector<std::string> res;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // Remove whitespace (optional, but good for file paths cleanup)
        item.erase(std::remove_if(item.begin(), item.end(), [](unsigned char c){ return std::isspace(c); }), item.end());
        if(!item.empty()) res.push_back(item);
    }
    return res;
}

// =============================================================
// ARGUMENT PARSER LOGIC
// =============================================================
inline ProgramConfig parseArgs(int argc, char* argv[]) {
    ProgramConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        // --- MODES ---
        if (arg == "--mode" && i + 1 < argc) cfg.runMode = argv[++i];
        else if (arg == "--profile" && i + 1 < argc) cfg.akmProfile = argv[++i];
        else if (arg == "--brainwallet" && i + 1 < argc) cfg.brainwalletMode = argv[++i];

        // --- GENERATION STRATEGY (CRITICAL FOR XPRV/AUTO MODE) ---
        else if (arg == "--random") cfg.entropyMode = "random";        
        else if (arg == "--schematic") cfg.entropyMode = "schematic"; 

        // --- CHECK INPUTS ---
        else if (arg == "--file" && i + 1 < argc) cfg.inputFile = argv[++i];
        else if (arg == "--exec" && i + 1 < argc) cfg.execCommand = argv[++i];
        else if (arg == "--socket" && i + 1 < argc) cfg.socketPath = argv[++i];

        // --- XPRV ---
        else if (arg == "--xprv" && i + 1 < argc) cfg.xprv = argv[++i];

        // --- AKM ---
        else if (arg == "--akm-phrase" && i + 1 < argc) cfg.akmPhrase = argv[++i];
        else if ((arg == "--akm-file" || arg == "--input") && i + 1 < argc) cfg.akmFile = argv[++i];
        else if (arg == "--akm-word" && i + 1 < argc) cfg.akmLengths = parseIntList(argv[++i]);
        else if (arg == "--akm-mode" && i + 1 < argc) cfg.akmGenMode = argv[++i];
        else if (arg == "--akm-list-profiles") cfg.akmListProfiles = true;
        else if (arg == "--akm-bit" && i + 1 < argc) cfg.akmBits = parseIntList(argv[++i]);

        // --- COMMON ---
        else if (arg == "--continue" && i + 1 < argc) cfg.startFrom = argv[++i];
        else if (arg == "--infinite") { cfg.infinite = true; cfg.count = 0; }
        else if (arg == "--count" && i + 1 < argc) cfg.count = std::stoull(argv[++i]);
        else if (arg == "--speed") cfg.showRealSpeed = true;
        else if (arg == "--help" || arg == "-h") cfg.help = true;
        else if (arg == "--quiet") cfg.verbose = false;
        
        // --- GPU/CPU CONFIG ---
        else if (arg == "--type" && i + 1 < argc) cfg.gpuType = argv[++i];
        else if (arg == "--platform" && i + 1 < argc) cfg.platformId = std::stoi(argv[++i]);
        else if ((arg == "--device" || arg == "-d") && i + 1 < argc) cfg.deviceId = std::stoi(argv[++i]);
        else if ((arg == "--blocks" || arg == "-b") && i + 1 < argc) cfg.cudaBlocks = std::stoi(argv[++i]);
        else if ((arg == "--threads" || arg == "-t") && i + 1 < argc) cfg.cudaThreads = std::stoi(argv[++i]);
        else if ((arg == "--points" || arg == "-p") && i + 1 < argc) cfg.pointsPerThread = std::stoi(argv[++i]);
        else if (arg == "--cores" && i + 1 < argc) cfg.cpuCores = std::stoi(argv[++i]);
        
        // --- DATA ---
        // [MODIFICAT] Folosim parseStringList pentru a suporta multiple fisiere
        else if (arg == "--bloom-keys" && i + 1 < argc) cfg.bloomFiles = parseStringList(argv[++i]);
        else if (arg == "--win" && i + 1 < argc) cfg.winFile = argv[++i];
        
        // --- MNEMONIC ---
        else if (arg == "--langs" && i + 1 < argc) cfg.language = argv[++i];
        else if (arg == "--words" && i + 1 < argc) cfg.words = std::stoi(argv[++i]);
        else if (arg == "--mnemonic-order" && i + 1 < argc) cfg.mnemonicOrder = argv[++i];
        else if (arg == "--mnemonic-prefix" && i + 1 < argc) cfg.mnemonicPrefix = argv[++i];
        else if (arg == "--mnemonic-brute" && i + 1 < argc) cfg.mnemonicBrute = argv[++i];

        // --- PATHS ---
        else if (arg == "--setaddress" && i + 1 < argc) cfg.setAddress = argv[++i];
        else if (arg == "--path-file" && i + 1 < argc) cfg.pathFile = argv[++i];
        else if (arg == "--multi-coin" && i + 1 < argc) cfg.multiCoin = argv[++i];
        else if (arg == "--coin" && i + 1 < argc) cfg.multiCoin = argv[++i];

        // --- ENTROPY (BIP39) ---
        else if (arg == "--entropy-str" && i + 1 < argc) cfg.entropyStr = argv[++i];
        else if (arg == "--hex") cfg.entropyMode = "hex";
        else if (arg == "--bin") cfg.entropyMode = "bin";
        else if (arg == "--dice") cfg.entropyMode = "dice";
        else if (arg == "--card") cfg.entropyMode = "card";
        else if (arg == "--base6") cfg.entropyMode = "base6";
        else if (arg == "--base10") cfg.entropyMode = "base10";
    }

    // Default settings logic
    // Daca modul este AKM si nu s-a specificat lungimea, se seteaza default
    if (cfg.runMode == "akm" && cfg.akmLengths.empty()) {
        if (cfg.words > 0) cfg.akmLengths = { cfg.words }; else cfg.akmLengths = { 10 }; 
    } 
    // Daca runMode este xprv-mode si nu s-a setat entropyMode, il setam default pe random
    else if (cfg.runMode == "xprv-mode" && (cfg.entropyMode == "default" || cfg.entropyMode.empty())) {
        cfg.entropyMode = "random";
    }
    else if (cfg.runMode != "check" && cfg.runMode != "scan" && cfg.runMode != "xprv-mode" && cfg.words == 0) {
        // Pentru mnemonic standard, default la 12 cuvinte
        cfg.words = 12;
    }
    return cfg;
}

// =============================================================
// HELP MENU (MATCHING OLD VERSION)
// =============================================================
inline void printHelp() {
    std::cout << "==============================================================\n";
    std::cout << "   GpuCracker v42.2 (Full Command List) - Modular Arch\n";
    std::cout << "==============================================================\n\n";

    std::cout << "Usage: GpuCracker.exe [OPTIONS]\n\n";

    std::cout << "--- OPERATING MODES ---\n";
    std::cout << "  --mode NAME              Select operation mode (Default: mnemonic)\n";
    std::cout << "                            Available: mnemonic, akm, check, scan, xprv-mode\n";

    std::cout << "\n--- SCAN MODE (XPRV/Path Derivation) ---\n";
    std::cout << "  --mode scan              Derive addresses from XPRV using path lists.\n";
    std::cout << "  --xprv KEY               Extended Private Key (BIP32) root.\n";
    std::cout << "  --path-file FILE         File containing derivation paths.\n";
    std::cout << "  --random         Random mode for xprv generation exe --xprv-mode --random .\n";
    std::cout << "  --schematic         schematic mode for xprv generation exe --xprv-mode --schematic.\n";

    std::cout << "\n--- BRAINWALLET MODE (Infinite Password Gen) ---\n";
    std::cout << "  --brainwallet MODE       Enable password generator strategy:\n";
    std::cout << "                            - 'random'    : Random words from internal dict\n";
    std::cout << "                            - 'alpha'      : Alphanumeric strings (a-z0-9)\n";
    std::cout << "                            - 'num'        : Numeric strings / PINs\n";
    std::cout << "                            - 'hex'        : Random Hex strings (32 chars)\n";
    std::cout << "                            - 'schematic' : Recurring patterns (e.g. 'ababab')\n";
    std::cout << "                            - 'clever'      : AI-like Markov Chain generator\n";

    std::cout << "\n--- CHECK MODE (Scanner/Verifier) ---\n";
    std::cout << "  --file FILE              Read seeds/keys/passwords from a text file.\n";
    std::cout << "  --exec CMD               Execute external app (Pipe input from stdout).\n";
    std::cout << "  --socket PATH            Read from Unix Domain Socket (local IPC).\n";

    std::cout << "\n--- RESUME / OFFSET ---\n";
    std::cout << "  --continue HEX           Start search from specific HEX offset.\n";
    std::cout << "                            Useful for sequential scans or AKM ranges.\n";

    std::cout << "\n--- MNEMONIC SETTINGS (BIP39) ---\n";
    std::cout << "  --langs NAME             Language file (english, french, etc.)\n";
    std::cout << "  --words N                12, 15, 18, 21, 24 words.\n";
    std::cout << "  --mnemonic-order VAL     Iteration strategy:\n";
    std::cout << "                            - 'random'    : Random seeds (Default)\n";
    std::cout << "                            - 'sequential': Incremental (+1)\n";
    std::cout << "                            - 'schematic' : Pattern based seeds\n";
    std::cout << "  --mnemonic-prefix VAL    Fixed prefix for mnemonics (optional).\n";
    std::cout << "  --mnemonic-brute VAL     Specific brute-force strategy (if implemented).\n";

    std::cout << "\n--- ADDRESS & PATH SETTINGS ---\n";
    std::cout << "  --multi-coin LIST        Select coins (e.g. \"BTC, LTC, DOGE, DASH, BTG, ZCASH and ETH\"). Default: btc\n";
    std::cout << "  --path-file FILE         Load custom derivation paths from file.\n";
    std::cout << "                            Format: m/44'/0'/0-5'/0/0 (supports ranges)\n";
    std::cout << "  --setaddress FILTER      Filter output/check generation.\n";
    std::cout << "                            - Types: ALL, LEGACY, P2SH, SEGWIT, ETH\n";
    std::cout << "                            - Formats: COMP, UNCOMP, BECH32, P2PKH\n";
	std::cout << "                            - Exemple : \"Comp P2PKH\", \"Uncomp P2PKH\", \"Bech32\", \"Comp P2SH\", etc. \n";

    std::cout << "\n--- AKM MODE SETTINGS (Puzzle) ---\n";
    std::cout << "  --profile NAME           Select AKM Profile (e.g., akm3-puzzle71)\n";
    std::cout << "  --akm-bit LIST           Limit search bits (e.g. '66,67,71')\n";
    std::cout << "  --akm-word LIST          Set accepted phrase lengths (e.g. '12')\n";
    std::cout << "  --akm-mode MODE          Gen mode: random | schematic | wave\n";
    std::cout << "  --akm-list-profiles      List available profiles and exit.\n";

    std::cout << "\n--- ESSENTIAL ---\n";
    std::cout << "  --bloom-keys FILE(s)     Path to the Bloom Filter file(s) (.blf)\n";
    std::cout << "                            Supports multiple files: file1.blf,file2.blf\n";
    std::cout << "  --infinite               Run continuously without stopping.\n";
    std::cout << "  --count N                Stop after checking N seeds/keys.\n";
    std::cout << "  --speed                  Show real GPU seed generation speed.\n";
    std::cout << "  --win FILE               File path to save successful hits.\n";
    
    std::cout << "\n--- HARDWARE CONFIGURATION ---\n";
    std::cout << "  --type TYPE              Backend: 'cuda', 'opencl', 'vulkan', or 'auto'\n";
    std::cout << "  --device N               GPU device ID (-1 for all).\n";
    std::cout << "  --platform N             OpenCL platform ID.\n";
    std::cout << "  --blocks N               CUDA Blocks count (e.g. 1024, 16384).\n";
    std::cout << "  --threads N              Threads per Block (e.g. 256, 512).\n";
    std::cout << "  --points N               Hashes per Thread (Loop Unrolling).\n";
    std::cout << "  --cores N                CPU threads for address derivation (0=Auto).\n";

    std::cout << "\n--- ENTROPY MODES (BIP39 Seed Only) ---\n";
    std::cout << "  --hex                    [0-9A-F] Hexadecimal entropy\n";
    std::cout << "  --base10                 [0-9] Decimal entropy\n";
    std::cout << "  --bin                    [0-1] Binary entropy\n";
    std::cout << "  --dice                   [1-6] Dice rolls\n";
    std::cout << "  --base6                  [0-5] Base 6\n";
    std::cout << "  --card                   [A2-9TJQK][CDHS] Card decks\n";
    std::cout << "  --entropy-str VAL        Input strategy for Mnemonic generation:\n";
    std::cout << "                            - 'random'    : Standard BIP39\n";
    std::cout << "                            - 'bip32'      : Raw Hex Seed (No Mnemonic)\n";
    std::cout << "                            - 'schematic' : Pattern based Seed\n";
    std::cout << "                            - <value>      : Check specific single value\n";

    std::cout << "==============================================================\n";
}