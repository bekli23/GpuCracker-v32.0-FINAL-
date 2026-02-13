#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <map>
#include <algorithm>
#include <mutex>
#include <sstream>
#include <cctype>

// Nivelul de "inteligenta" (3 = analizeaza grupuri de 3 caractere)
// E.g. [RO] -> Mas...
#define NGRAM_SIZE 3 

class CleverAI {
private:
    // Harta mentala: "abc" -> ['d', 'e', 'f']
    std::map<std::string, std::vector<char>> chain;
    std::vector<std::string> startGrams; 
    std::mt19937 rng;
    std::mutex aiMutex;
    bool isTrained = false;

    std::string memoryFile = "ai_memory.dat";
    std::string inputFile = "word_ai.txt";

    // --- DATA SPECIALIST LOGIC ---
    
    // 1. Detectie Simpla a Limbii (Heuristic bazat pe sufixe)
    std::string detectLanguage(const std::string& word) {
        std::string w = word;
        // Convertim la lowercase pt verificare
        std::transform(w.begin(), w.end(), w.begin(), ::tolower);
        
        // Reguli simple RO
        if (w.length() > 4) {
            std::string end2 = w.substr(w.length()-2);
            std::string end3 = w.substr(w.length()-3);
            if (end2 == "ul" || end2 == "ea" || end2 == "ua" || end2 == "ii" || end2 == "le") return "[RO]";
            if (end3 == "lui" || end3 == "lor" || end3 == "esc" || end3 == "are") return "[RO]";
        }

        // Reguli simple EN
        if (w.length() > 4) {
            std::string end2 = w.substr(w.length()-2);
            std::string end3 = w.substr(w.length()-3);
            if (end2 == "ed" || end2 == "ly" || end2 == "er" || end2 == "on") return "[EN]";
            if (end3 == "ing" || end3 == "the" || end3 == "ion" || end3 == "ity") return "[EN]";
        }

        // Default daca nu suntem siguri (sau cuvant scurt)
        return "[EN]"; 
    }

    // 2. Curatare si Formatare CamelCase
    std::string cleanAndFormat(const std::string& raw) {
        std::string cleaned = "";
        
        // A. Eliminare punctuatie/cifre
        for (char c : raw) {
            if (std::isalpha(static_cast<unsigned char>(c))) {
                cleaned += c;
            }
        }

        // B. Verificare lungime minima (3 litere)
        if (cleaned.length() < 3) return "";

        // C. Aplicare CamelCase (Prima mare, restul mici)
        for (size_t i = 0; i < cleaned.length(); i++) {
            if (i == 0) cleaned[i] = std::toupper(static_cast<unsigned char>(cleaned[i]));
            else cleaned[i] = std::tolower(static_cast<unsigned char>(cleaned[i]));
        }

        // D. Adaugare Tag Limba
        std::string tag = detectLanguage(cleaned);
        return tag + cleaned; // Ex: "[RO]Masina"
    }

public:
    CleverAI() {
        rng.seed((unsigned int)std::chrono::steady_clock::now().time_since_epoch().count());
        loadMemory(); // Incarca memoria anterioara
        trainFromInput(); // Invata din fisierul text curent
    }

    // Functia de invatare Markov
    void train(const std::string& text) {
        if (text.length() < NGRAM_SIZE) return;

        // Invatam cum incepe cuvantul (ex: "[RO")
        startGrams.push_back(text.substr(0, NGRAM_SIZE));

        // Invatam tranzitiile
        for (size_t i = 0; i < text.length() - NGRAM_SIZE; i++) {
            std::string key = text.substr(i, NGRAM_SIZE);
            char nextChar = text[i + NGRAM_SIZE];
            chain[key].push_back(nextChar);
        }
        isTrained = true;
    }

    // Citeste, Proceseaza si Invata din word_ai.txt
    void trainFromInput() {
        std::ifstream f(inputFile);
        if (!f.is_open()) return;

        std::string line;
        int count = 0;
        // Citim linie cu linie
        while (std::getline(f, line)) {
            std::stringstream ss(line);
            std::string word;
            // Citim cuvant cu cuvant de pe linie (pentru a elimina spatiile extra)
            while (ss >> word) {
                // APLICAM LOGICA DE DATA SPECIALIST
                std::string processed = cleanAndFormat(word);
                
                if (!processed.empty() && processed.length() >= NGRAM_SIZE) {
                    train(processed);
                    count++;
                }
            }
        }
        f.close();
        if (count > 0) saveMemory();
    }

    // Salveaza "creierul"
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

    // Incarca "creierul"
    void loadMemory() {
        std::ifstream f(memoryFile, std::ios::binary);
        if (!f.is_open()) return;

        size_t mapSize;
        if(f.read((char*)&mapSize, sizeof(mapSize))) {
            for (size_t i = 0; i < mapSize; i++) {
                size_t keyLen;
                f.read((char*)&keyLen, sizeof(keyLen));
                std::string key(keyLen, ' ');
                f.read(&key[0], keyLen);

                size_t vecSize;
                f.read((char*)&vecSize, sizeof(vecSize));
                std::vector<char> vals(vecSize);
                f.read(vals.data(), vecSize);

                chain[key] = vals;
            }
        }
        
        size_t startSize;
        if(f.read((char*)&startSize, sizeof(startSize))) {
            for(size_t i=0; i<startSize; i++) {
                size_t sLen;
                f.read((char*)&sLen, sizeof(sLen));
                std::string s(sLen, ' ');
                f.read(&s[0], sLen);
                startGrams.push_back(s);
            }
        }
        
        if (!chain.empty()) isTrained = true;
        f.close();
    }

    // Genereaza o parola noua (Phrase Style)
    std::string generate() {
        // Fallback daca nu exista date antrenate
        if (!isTrained || startGrams.empty()) {
            const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
            std::string res; 
            for(int i=0; i<14; i++) res += charset[rng() % 62];
            return res;
        }

        // 1. Alege un inceput (ex: "[RO")
        std::string currentGram = startGrams[rng() % startGrams.size()];
        std::string result = currentGram;

        // 2. Generare (max 40 caractere pt fraze)
        for (int i = 0; i < 40; i++) {
            
            // --- LOGICA SPAȚIU ALEATORIU (PHRASE GENERATOR) ---
            // 8% sansa sa bage spatiu daca:
            // - Nu e deja spatiu la final
            // - Lungimea curenta < 30 (sa nu taie brusc la final)
            // - Am generat deja macar 5 caractere (sa nu puna spatiu imediat dupa tag)
            if ((rng() % 100 < 8) && result.back() != ' ' && result.length() < 30 && result.length() > 5) {
                result += ' ';
                
                // Cand incepem un cuvant nou, alegem un nou StartGram
                // Astfel se pastreaza logica: [Tag]Cuvant [Tag]Cuvant
                // Sau pur si simplu Cuvant (daca invata ca dupa spatiu urmeaza majuscula)
                // In cazul nostru, alegem un nou startGram aleatoriu din lista invatata
                // pentru a combina cuvinte din limbi/structuri diferite.
                currentGram = startGrams[rng() % startGrams.size()];
                result += currentGram;
                
                // Sarim peste procesarea NGRAM pentru bucata adaugata
                continue; 
            }
            // --------------------------------------------------

            if (chain.find(currentGram) == chain.end()) break; 

            const std::vector<char>& possibilities = chain[currentGram];
            if (possibilities.empty()) break;

            char nextChar = possibilities[rng() % possibilities.size()];
            result += nextChar;

            // Shiftam fereastra: "abc" + 'd' -> "bcd"
            currentGram = currentGram.substr(1) + nextChar;
        }

        return result;
    }
};