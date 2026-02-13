#define _CRT_SECURE_NO_WARNINGS
#define OPENSSL_SUPPRESS_DEPRECATED
#define OPENSSL_API_COMPAT 0x10100000L
#pragma warning(disable : 4996)

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cstdint>

#include <secp256k1.h>
#include "multicoin.h"

void FillPrivateKey(uint64_t counter, std::vector<uint8_t>& outKey) {
    std::fill(outKey.begin(), outKey.end(), 0);
    for (int i = 0; i < 8; i++) {
        outKey[31 - i] = (counter >> (i * 8)) & 0xFF;
    }
}

int main(int argc, char* argv[]) {
    // Configurare default
    uint64_t startFrom = 1;
    uint64_t count = 50;
    std::string filename = "schematic_output.txt";

    if (argc > 1) count = std::stoull(argv[1]);
    if (argc > 2) startFrom = std::stoull(argv[2]);

    // Mesaj doar in consola, nu in fisier
    std::cout << "Generare " << count << " adrese incepand de la indexul " << startFrom << "...\n";

    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    if (!ctx) {
        std::cerr << "Eroare critica: Lipsa DLL-uri sau memorie!\n";
        return -1;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Nu pot scrie in " << filename << "\n";
        return 1;
    }

    std::vector<uint8_t> privKey(32);
    uint64_t current = startFrom;

    for (uint64_t i = 0; i < count; i++) {
        FillPrivateKey(current, privKey);

        // Verificam validitatea cheii
        if (secp256k1_ec_seckey_verify(ctx, privKey.data())) {
            secp256k1_pubkey pub;
            if (secp256k1_ec_pubkey_create(ctx, &pub, privKey.data())) {
                
                // 1. Compressed Address
                uint8_t cPub[33]; size_t clen = 33;
                secp256k1_ec_pubkey_serialize(ctx, cPub, &clen, &pub, SECP256K1_EC_COMPRESSED);
                std::vector<uint8_t> vPubComp(cPub, cPub + 33);
                std::string btcCompressed = MultiCoin::GenLegacy(vPubComp, "BTC");
                
                // Scriem adresa Compressed
                file << btcCompressed << "\n";

                // 2. Uncompressed Address
                uint8_t uPub[65]; size_t ulen = 65;
                secp256k1_ec_pubkey_serialize(ctx, uPub, &ulen, &pub, SECP256K1_EC_UNCOMPRESSED);
                std::vector<uint8_t> vPubUncomp(uPub, uPub + 65);
                std::string btcUncompressed = MultiCoin::GenLegacy(vPubUncomp, "BTC");

                // Scriem adresa Uncompressed
                file << btcUncompressed << "\n";
            }
        }
        
        current++;
        
        // Feedback vizual in consola la fiecare 100 de adrese
        if (i % 100 == 0) std::cout << "\rProgres: " << i << "/" << count << std::flush;
    }

    std::cout << "\nGata. Fisierul " << filename << " contine doar adresele.\n";
    
    secp256k1_context_destroy(ctx);
    file.close();
    return 0;
}