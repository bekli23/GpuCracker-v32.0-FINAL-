#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <algorithm> // necesar pentru std::sort
#include <openssl/sha.h>

#include "akm_extra.h" 

class AkmLogic {
private:
    std::unordered_map<std::string, std::string> customHex;
    std::unordered_map<std::string, AkmRule> specialRules;
    std::vector<std::string> baseWords;

    void load_wordlist_from_file(const std::string& filePath) {
        baseWords.clear();
        std::ifstream file(filePath);
        
        if (!file.is_open()) {
            std::cerr << "[AKM] Warning: Could not open wordlist " << filePath << "\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(std::remove_if(line.begin(), line.end(), 
                [](unsigned char c){ return std::isspace(c); }), line.end());
            
            if (!line.empty()) {
                baseWords.push_back(line);
            }
        }
        file.close();
    }

    // Incarca profil custom din fisier .txt
    void load_external_profile(const std::string& filename) {
        std::string targetFile = filename;
        std::ifstream file(targetFile);
        
        if (!file.is_open()) {
            targetFile = filename + ".txt";
            file.open(targetFile);
        }

        if (!file.is_open()) return;

        std::cout << "[AKM] Loading EXTERNAL profile from: " << targetFile << "\n";
        
        std::string line;
        while (std::getline(file, line)) {
            size_t commentPos = line.find('#');
            if (commentPos != std::string::npos) line = line.substr(0, commentPos);
            line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c){ return std::isspace(c); }), line.end());
            
            if (line.empty()) continue;

            size_t delimPos = line.find('=');
            if (delimPos != std::string::npos) {
                std::string word = line.substr(0, delimPos);
                std::string hexVal = line.substr(delimPos + 1);
                if (!word.empty() && !hexVal.empty()) customHex[word] = hexVal;
            }
        }
        file.close();
        std::cout << "[AKM] External profile loaded. Rules count: " << customHex.size() << "\n";
    }

    std::string extend_token(const std::string& base, const std::string& word) {
        if (base.length() >= 8) return base.substr(0, 8);
        
        uint8_t hash[SHA256_DIGEST_LENGTH];
        SHA256((const uint8_t*)word.c_str(), word.length(), hash);
        char buf[9];
        sprintf(buf, "%02x%02x%02x%02x", hash[0], hash[1], hash[2], hash[3]);
        
        std::string extended = base + std::string(buf);
        return extended.substr(0, 8);
    }

public:
    const std::vector<std::string>& getWordList() const { return baseWords; }
    size_t getWordCount() const { return baseWords.size(); }

    void listProfiles() {
        std::cout << "[AKM] Available Profiles:\n";
        std::cout << "  - auto-linear (Index 0..511 -> Hex)\n";
        std::cout << "  - akm2-core\n";
        std::cout << "  - akm2-lab-v1\n";
        std::cout << "  - akm2-fixed123-pack-v1\n";
        std::cout << "  - akm3-puzzle71\n";
        std::cout << "  - [CUSTOM] (file.txt)\n";
    }

    void init(const std::string& profileName, const std::string& wordlistPath) {
        // 1. Incarca Wordlist
        load_wordlist_from_file(wordlistPath);
        
        // 2. SORTEAZA (Important pentru ordinea 1,2,3...)
        std::sort(baseWords.begin(), baseWords.end());

        customHex.clear();
        specialRules.clear();

        // 3. LOGICA PROFILE
        
        // === PROFILUL AUTOMAT LINIAR ===
        if (profileName == "auto-linear") {
            std::cout << "[AKM] Generating AUTO-LINEAR profile (Mapping all words to index)...\n";
            for (size_t i = 0; i < baseWords.size(); ++i) {
                char buf[16];
                // Converteste indexul (0, 1, 2...) in string HEX ("0", "1", "a", "1ff")
                sprintf(buf, "%x", (unsigned int)i);
                customHex[baseWords[i]] = std::string(buf);
            }
            std::cout << "[AKM] Mapped " << customHex.size() << " words sequentially.\n";
            return; // Gata, nu mai incarcam altceva
        }

        // --- Profile Interne ---
        load_akm_extra_profile(profileName, customHex, specialRules);
        
        // --- Profile Externe ---
        if (customHex.empty() && specialRules.empty()) {
            load_external_profile(profileName);
        }
        
        // Fallback
        if (customHex.empty() && specialRules.empty()) {
            customHex["mare"] = "0000"; 
        }
    }

    std::string get_token8(const std::string& word) {
        if (customHex.count(word)) return extend_token(customHex[word], word);
        
        if (specialRules.count(word)) {
            const auto& r = specialRules[word];
            if (!r.fixed8.empty()) return r.fixed8;
            if (!r.fixed3.empty()) return extend_token(r.fixed3, word);
            if (!r.fixed2.empty()) return extend_token(r.fixed2, word);
            if (!r.fixed1.empty()) return extend_token(r.fixed1, word);
            if (!r.pad_nibble.empty()) return extend_token(r.pad_nibble, word);
        }
        return extend_token("", word);
    }

    std::vector<uint8_t> phrase_to_key(const std::vector<std::string>& words) {
        bool allMare = true;
        for(const auto& w : words) if (w != "mare") allMare = false;
        if (allMare && !words.empty()) return std::vector<uint8_t>(32, 0xFF);

        std::string bigHex = "";
        for (const auto& w : words) {
            bigHex += get_token8(w);
            if (bigHex.length() >= 64) break;
        }
        
        std::vector<uint8_t> key;
        for (size_t i = 0; i < 64; i += 2) {
            if (i+1 < bigHex.length()) {
                std::string byteString = bigHex.substr(i, 2);
                key.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16));
            } else {
                key.push_back(0); 
            }
        }
        while (key.size() < 32) key.push_back(0);
        return key;
    }
};