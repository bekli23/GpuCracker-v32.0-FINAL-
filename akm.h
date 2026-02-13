#pragma once
#define _CRT_SECURE_NO_WARNINGS

#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <openssl/sha.h>

#include "akm_extra.h" 

class AkmLogic {
private:
    std::unordered_map<std::string, std::string> customHex;
    std::unordered_map<std::string, AkmRule> specialRules;
    std::vector<std::string> baseWords;
    bool isStrict = false;

    // Verificam daca e HEX valid (0-9, a-f)
    bool isValidHex(const std::string& s) {
        if (s.empty()) return false;
        for (char c : s) {
            if (!isxdigit((unsigned char)c)) return false;
        }
        return true;
    }

    // Functie helper pentru curatarea stringurilor
    std::string cleanString(const std::string& input) {
        std::string s = input;
        s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c){ 
            return std::isspace(c) || c == '\r' || c == '\n'; 
        }), s.end());
        return s;
    }

    // Incarcare Inteligenta cu detectie automata #STRICT_HEX
    void load_smart_file(const std::string& filePath) {
        std::ifstream file(filePath);
        if (!file.is_open()) {
            std::cerr << "[AKM] CRITICAL ERROR: Cannot open file " << filePath << "\n";
            return;
        }

        std::cout << "[AKM] FORCE READING PROFILE: " << filePath << " ...\n";

        // Resetam totul inainte de incarcare
        baseWords.clear();
        customHex.clear();
        isStrict = false;

        std::vector<std::string> lines;
        std::string line;
        
        // 1. Verificare Header
        // Citim prima linie validă (sărim peste linii goale de la început)
        while (std::getline(file, line)) {
            std::string cleanHeader = cleanString(line);
            if (cleanHeader.empty()) continue;

            if (cleanHeader == "#STRICT_HEX" || cleanHeader == "STRICT_HEX") {
                isStrict = true;
                std::cout << "[AKM] HEADER DETECTED (#STRICT_HEX) -> ACTIVATING STRICT MODE!\n";
            } else {
                // Prima linie nu e header, o pastram ca data
                lines.push_back(line);
            }
            break; // Am procesat prima linie relevanta
        }

        // 2. Citim restul
        while (std::getline(file, line)) {
            lines.push_back(line);
        }
        file.close();

        // 3. Procesam
        int loaded = 0;
        int ignored = 0;

        for (auto& l : lines) {
            std::string cleanL = cleanString(l);
            if (cleanL.empty() || cleanL[0] == '#') continue;

            size_t eqPos = cleanL.find('=');
            
            if (isStrict) {
                // --- LOGICA STRICTA ---
                if (eqPos != std::string::npos) {
                    std::string w = cleanL.substr(0, eqPos);
                    std::string v = cleanL.substr(eqPos + 1);
                    
                    if (isValidHex(v)) {
                        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
                        customHex[w] = v;
                        baseWords.push_back(w);
                        loaded++;
                    } else {
                        ignored++; 
                    }
                } else {
                    // Ignoram orice linie fara egal in mod strict
                    ignored++;
                }
            } 
            else {
                // --- LOGICA STANDARD ---
                if (eqPos != std::string::npos) {
                    std::string w = cleanL.substr(0, eqPos);
                    std::string v = cleanL.substr(eqPos + 1);
                    customHex[w] = v;
                    baseWords.push_back(w);
                } else {
                    baseWords.push_back(cleanL);
                }
                loaded++;
            }
        }

        if (isStrict) {
            std::cout << "[AKM] RESULT: Loaded " << loaded << " VALID words. Ignored " << ignored << " junk words.\n";
            std::cout << "[AKM] DICTIONARY SIZE: " << baseWords.size() << " (Should be 16 for hex)\n";
        } else {
            std::cout << "[AKM] Loaded " << loaded << " words (Standard Mode).\n";
        }
    }

    std::string extend_token(const std::string& base, const std::string& word) {
        if (base.length() >= 1) return base;
        uint8_t hash[SHA256_DIGEST_LENGTH];
        SHA256((const uint8_t*)word.c_str(), word.length(), hash);
        char buf[9]; sprintf(buf, "%02x%02x%02x%02x", hash[0], hash[1], hash[2], hash[3]);
        return std::string(buf).substr(0,8);
    }

public:
    const std::vector<std::string>& getWordList() const { return baseWords; }
	//const std::vector<uint8_t>& getHighBytes() const { return highBytes; }
    size_t getWordCount() const { return baseWords.size(); }

    int get_word_hex_len(const std::string& word) {
        if (customHex.count(word)) return (int)customHex[word].length();
        return 0;
    }

    void listProfiles() {
        std::cout << "  - auto-linear\n  - specialx.txt (requires #STRICT_HEX header)\n";
    }

    void init(const std::string& profileName, const std::string& defaultWordlistPath) {
        customHex.clear();
        specialRules.clear();
        isStrict = false;

        std::string finalFileToLoad = defaultWordlistPath;

        // [FIX CRITIC DE PRIORITATE]
        // Daca numele profilului se termina in .txt, IGNORAM path-ul default si folosim profilul ca fisier.
        if (profileName.size() >= 4 && profileName.substr(profileName.size() - 4) == ".txt") {
            finalFileToLoad = profileName;
            std::cout << "[AKM] Profile name ends in .txt -> Overriding wordlist to: " << finalFileToLoad << "\n";
        }

        // Incarcam fisierul decis mai sus
        if (!finalFileToLoad.empty()) {
            load_smart_file(finalFileToLoad);
        } else {
            // Fallback doar daca nu avem niciun fisier
            if (profileName == "auto-linear") {
               // ... logica veche ...
            }
        }

        // Daca fisierul a activat STRICT MODE, ne oprim aici.
        if (isStrict) {
            std::sort(baseWords.begin(), baseWords.end());
            return; 
        }

        // --- Logica Veche pentru alte profile ---
        if (profileName == "auto-linear" && !baseWords.empty()) {
            std::cout << "[AKM] Auto-Linear Mapping...\n";
            for (size_t i = 0; i < baseWords.size(); ++i) {
                char buf[16]; sprintf(buf, "%x", (unsigned int)i);
                customHex[baseWords[i]] = std::string(buf);
            }
        } else {
            load_akm_extra_profile(profileName, customHex, specialRules);
        }
    }

    std::string get_token8(const std::string& word) {
        if (customHex.count(word)) return customHex[word];
        if (isStrict) return ""; // In mod strict nu returnam nimic pt cuvinte straine
        return extend_token("", word);
    }

    std::vector<uint8_t> phrase_to_key(const std::vector<std::string>& words) {
        std::string bigHex = "";
        for (const auto& w : words) {
            bigHex += get_token8(w);
            if (bigHex.length() >= 64) break; 
        }
        while (bigHex.length() < 64) bigHex = "0" + bigHex;
        if (bigHex.length() > 64) bigHex = bigHex.substr(bigHex.length() - 64);
        
        std::vector<uint8_t> key;
        for (size_t i = 0; i < 64; i += 2) {
            std::string byteString = bigHex.substr(i, 2);
            key.push_back((uint8_t)strtol(byteString.c_str(), NULL, 16));
        }
        return key;
    }
};