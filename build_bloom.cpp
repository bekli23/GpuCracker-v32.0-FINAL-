#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <cstring>
#include <sstream>

// OpenSSL - Necesar pentru validarea checksum-ului adreselor
#include <openssl/sha.h>
#include <openssl/ripemd.h>

// --- CONFIGURARE BLOOM ---
#define DEFAULT_FP_RATE 0.000000001 // 1e-9 (Foarte precis, 1 la un miliard sanse de eroare)
#define MIN_BLOOM_SIZE_BITS (8 * 1024 * 1024) // Minim 8 milioane biti (1MB)

// Limita de siguranta pentru hash-ul de 32-bit (aprox 3.5 miliarde biti / ~420MB)
// Daca filtrul depaseste aceasta marime, coliziunile devin probabile pe 32-bit,
// asa ca trecem automat pe hash-ul de 64-bit (MurmurHash3_x64_128).
#define BLOOM_32BIT_LIMIT 3500000000ULL 

#ifdef _WIN32
    typedef unsigned long long ulong;
    typedef unsigned int uint;
#else
    #include <sys/types.h>
#endif

// ============================================================================
//  1. ALGORITMI DE HASHING (32-BIT si 64-BIT)
// ============================================================================

inline uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }
inline uint64_t rotl64(uint64_t x, int8_t r) { return (x << r) | (x >> (64 - r)); }

// --- 32-BIT Hash (Murmur3_x86_32) ---
// Folosit pentru fisiere mici/medii. Este rapid si compatibil cu versiuni vechi.
inline uint32_t MurmurHash3_x86_32(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    const uint32_t* blocks = (const uint32_t*)(data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1;
        memcpy(&k1, (const uint8_t*)blocks + (i * 4), 4);
        k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2;
        h1 ^= k1; h1 = rotl32(h1, 13); h1 = h1 * 5 + 0xe6546b64;
    }
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 4);
    uint32_t k1 = 0;
    switch (len & 3) {
    case 3: k1 ^= tail[2] << 16;
    case 2: k1 ^= tail[1] << 8;
    case 1: k1 ^= tail[0];
            k1 *= c1; k1 = rotl32(k1, 15); k1 *= c2; h1 ^= k1;
    };
    h1 ^= len; h1 ^= h1 >> 16; h1 *= 0x85ebca6b; h1 ^= h1 >> 13; h1 *= 0xc2b2ae35; h1 ^= h1 >> 16;
    return h1;
}

// --- 64-BIT Hash (Murmur3_x64_128) ---
// Folosit automat pentru fisiere MARI (> 100M adrese). Elimina coliziunile pe filtre gigant.
#define BIG_CONSTANT(x) (x##LLU)
void MurmurHash3_x64_128(const void* key, const int len, const uint32_t seed, void* out) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 16;
    uint64_t h1 = seed;
    uint64_t h2 = seed;
    const uint64_t c1 = BIG_CONSTANT(0x87c37b91114253d5);
    const uint64_t c2 = BIG_CONSTANT(0x4cf5ad432745937f);
    const uint64_t* blocks = (const uint64_t*)(data);

    for (int i = 0; i < nblocks; i++) {
        uint64_t k1 = blocks[i * 2 + 0];
        uint64_t k2 = blocks[i * 2 + 1];
        k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
        h1 = rotl64(h1, 27); h1 += h2; h1 = h1 * 5 + 0x52dce729;
        k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
        h2 = rotl64(h2, 31); h2 += h1; h2 = h2 * 5 + 0x38495ab5;
    }
    const uint8_t* tail = (const uint8_t*)(data + nblocks * 16);
    uint64_t k1 = 0;
    uint64_t k2 = 0;
    switch (len & 15) {
    case 15: k2 ^= ((uint64_t)tail[14]) << 48;
    case 14: k2 ^= ((uint64_t)tail[13]) << 40;
    case 13: k2 ^= ((uint64_t)tail[12]) << 32;
    case 12: k2 ^= ((uint64_t)tail[11]) << 24;
    case 11: k2 ^= ((uint64_t)tail[10]) << 16;
    case 10: k2 ^= ((uint64_t)tail[9]) << 8;
    case  9: k2 ^= ((uint64_t)tail[8]) << 0;
             k2 *= c2; k2 = rotl64(k2, 33); k2 *= c1; h2 ^= k2;
    case  8: k1 ^= ((uint64_t)tail[7]) << 56;
    case  7: k1 ^= ((uint64_t)tail[6]) << 48;
    case  6: k1 ^= ((uint64_t)tail[5]) << 40;
    case  5: k1 ^= ((uint64_t)tail[4]) << 32;
    case  4: k1 ^= ((uint64_t)tail[3]) << 24;
    case  3: k1 ^= ((uint64_t)tail[2]) << 16;
    case  2: k1 ^= ((uint64_t)tail[1]) << 8;
    case  1: k1 ^= ((uint64_t)tail[0]) << 0;
             k1 *= c1; k1 = rotl64(k1, 31); k1 *= c2; h1 ^= k1;
    };
    h1 ^= len; h2 ^= len;
    h1 += h2; h2 += h1;
    auto fmix64 = [](uint64_t k) {
        k ^= k >> 33; k *= BIG_CONSTANT(0xff51afd7ed558ccd);
        k ^= k >> 33; k *= BIG_CONSTANT(0xc4ceb9fe1a85ec53);
        k ^= k >> 33; return k;
    };
    h1 = fmix64(h1); h2 = fmix64(h2);
    h1 += h2; h2 += h1;
    ((uint64_t*)out)[0] = h1;
    ((uint64_t*)out)[1] = h2;
}

// Helpers scriere fisier binar
void write_be64(std::ofstream& f, uint64_t val) {
    uint8_t bytes[8]; for (int i=0; i<8; ++i) bytes[7-i] = (val >> (i*8)) & 0xFF;
    f.write((char*)bytes, 8);
}
void write_be32(std::ofstream& f, uint32_t val) {
    uint8_t bytes[4]; for (int i=0; i<4; ++i) bytes[3-i] = (val >> (i*8)) & 0xFF;
    f.write((char*)bytes, 4);
}
std::string ToHex(const std::vector<uint8_t>& data) {
    std::stringstream ss; ss << std::hex << std::setfill('0');
    for (auto b : data) ss << std::setw(2) << (int)b;
    return ss.str();
}

// ============================================================================
//  LOGICA DE PARSARE ADRESE (ETH, BTC Legacy, BTC Bech32)
// ============================================================================

int hexCharToInt(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

// Parsare Ethereum (0x...)
std::vector<uint8_t> ParseEthAddress(const std::string& str) {
    if (str.length() != 42 || str.substr(0, 2) != "0x") return {};
    std::string hexPart = str.substr(2);
    std::vector<uint8_t> bytes; bytes.reserve(20);
    for (size_t i = 0; i < hexPart.length(); i += 2) {
        int h = hexCharToInt(hexPart[i]);
        int l = hexCharToInt(hexPart[i+1]);
        if (h == -1 || l == -1) return {}; 
        bytes.push_back((uint8_t)(h * 16 + l));
    }
    return bytes;
}

// Helper SHA256 dublu
void DoubleSHA256(const void* data, size_t len, uint8_t* output) {
    uint8_t hash1[SHA256_DIGEST_LENGTH]; 
    SHA256((const uint8_t*)data, len, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, output);
}

// Parsare BTC Base58 (1... sau 3...)
const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
int8_t mapBase58[256]; bool base58Initialized = false;

void InitBase58() { 
    memset(mapBase58, -1, sizeof(mapBase58)); 
    for (int i = 0; pszBase58[i]; i++) mapBase58[(uint8_t)pszBase58[i]] = i; 
    base58Initialized = true;
}

bool DecodeBase58Check(const std::string& str, std::vector<uint8_t>& vchRet) {
    if (!base58Initialized) InitBase58();
    int zeros = 0; while (zeros < (int)str.size() && str[zeros] == '1') zeros++;
    std::vector<unsigned char> b58(str.size() * 733 / 1000 + 10, 0);
    int length = 0;
    for (char c : str) {
        if (mapBase58[(uint8_t)c] == -1) return false;
        int carry = mapBase58[(uint8_t)c];
        for (int i = 0; i < length || carry != 0; i++) {
            int val = carry + 58 * b58[i];
            b58[i] = val % 256; carry = val / 256;
            if (i >= length && carry == 0) length = i + 1; else if (i >= length) length++;
        }
    }
    std::vector<uint8_t> result; result.reserve(zeros + length);
    result.assign(zeros, 0x00); for (int i = length - 1; i >= 0; i--) result.push_back(b58[i]);
    if (result.size() < 4) return false;
    std::vector<uint8_t> data(result.begin(), result.end() - 4);
    std::vector<uint8_t> checksum(result.end() - 4, result.end());
    uint8_t hash[32]; DoubleSHA256(data.data(), data.size(), hash);
    if (memcmp(hash, checksum.data(), 4) != 0) return false;
    vchRet = data; 
    return true;
}

// Parsare BTC Bech32 (bc1...)
namespace Bech32 {
    uint32_t polymod(const std::vector<uint8_t>& values) {
        static const uint32_t GEN[5] = {0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3};
        uint32_t chk = 1;
        for (uint8_t v : values) {
            uint8_t b = chk >> 25; chk = ((chk & 0x1ffffff) << 5) ^ v;
            for (int i = 0; i < 5; ++i) if ((b >> i) & 1) chk ^= GEN[i];
        } return chk;
    }
    std::vector<uint8_t> expand_hrp(const std::string& hrp) {
        std::vector<uint8_t> ret; ret.reserve(hrp.size()*2+1);
        for(char c:hrp) ret.push_back(c>>5); ret.push_back(0); for(char c:hrp) ret.push_back(c&31); return ret;
    }
    bool convert_bits(const std::vector<uint8_t>& in, int from, int to, bool pad, std::vector<uint8_t>& out) {
        int acc=0, bits=0, maxv=(1<<to)-1, max_acc=(1<<(from+to-1))-1;
        for(uint8_t v:in) {
            acc=((acc<<from)|v)&max_acc; bits+=from;
            while(bits>=to) { bits-=to; out.push_back((acc>>bits)&maxv); }
        }
        if(pad && bits>0) out.push_back((acc<<(to-bits))&maxv);
        else if(bits>=from || ((acc<<(to-bits))&maxv)) return false;
        return true;
    }
}

bool DecodeBech32(const std::string& str, std::vector<uint8_t>& vchRet) {
    if(str.size()>90 || str.size()<8) return false;
    std::string bech=str; for(char& c:bech) c=std::tolower(c);
    size_t pos=bech.rfind('1');
    if(pos==std::string::npos || pos==0 || pos+7>bech.size()) return false;
    std::string hrp=bech.substr(0, pos);
    if(hrp!="bc" && hrp!="tb") return false;
    std::vector<uint8_t> data;
    for(size_t i=pos+1; i<bech.size(); ++i) {
        const char* p = strchr("qpzry9x8gf2tvdw0s3jn54khce6mua7l", bech[i]);
        if(!p) return false; data.push_back(p-"qpzry9x8gf2tvdw0s3jn54khce6mua7l");
    }
    std::vector<uint8_t> hrpExp=Bech32::expand_hrp(hrp);
    hrpExp.insert(hrpExp.end(), data.begin(), data.end());
    uint32_t poly = Bech32::polymod(hrpExp);
    if(poly != 1 && poly != 0x2bc830a3) return false; 
    if(data.size()<1) return false;
    std::vector<uint8_t> p5(data.begin()+1, data.end()-6);
    if(!Bech32::convert_bits(p5, 5, 8, false, vchRet)) return false;
    return true;
}

// ============================================================================
//  MAIN
// ============================================================================
void print_help() {
    std::cout << "\n=======================================================\n";
    std::cout << "   Bloom Builder v7.5 - Multi-Coin / Hybrid Hashing   \n";
    std::cout << "=======================================================\n";
    std::cout << "Usage:\n";
    std::cout << "  build_bloom.exe --input <file> [options]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  --input <file>   : Input text file containing addresses.\n";
    std::cout << "                     Can be used multiple times for multiple files.\n";
    std::cout << "                     Example: --input btc.txt --input eth.txt\n\n";
    
    std::cout << "  --out <file>     : Output Bloom Filter file (.blf).\n";
    std::cout << "                     Default: 'out.blf'\n\n";
    
    std::cout << "  --p <rate>       : Desired False Positive Rate.\n";
    std::cout << "                     Default: 1e-9 (0.000000001)\n";
    std::cout << "                     Lower means bigger file size but fewer false hits.\n\n";
    
    std::cout << "Examples:\n";
    std::cout << "  1. Basic usage:\n";
    std::cout << "     build_bloom.exe --input addresses.txt\n\n";
    
    std::cout << "  2. High precision build for huge lists:\n";
    std::cout << "     build_bloom.exe --input all_btc.txt --out huge.blf --p 0.0000000001\n\n";
    
    std::cout << "Note: The tool automatically switches to 64-bit hashing if the\n";
    std::cout << "      dataset requires a filter larger than ~400MB to avoid collisions.\n";
    std::cout << "=======================================================\n";
}

int main(int argc, char* argv[]) {
    // Daca nu sunt argumente, afisam ajutorul extins
    if (argc < 2) { print_help(); return 0; }
    
    InitBase58(); 
    std::vector<std::string> inputs;
    std::string outputFile = "out.blf";
    double fpRate = DEFAULT_FP_RATE;

    // Parsare argumente
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg=="--input" && i+1<argc) inputs.push_back(argv[++i]);
        else if(arg=="--out" && i+1<argc) outputFile = argv[++i];
        else if(arg=="--p" && i+1<argc) fpRate = std::stod(argv[++i]);
        else if(arg=="--help" || arg=="-h") { print_help(); return 0; }
        else if (arg.find("--") == std::string::npos) {
            // Permite specificarea fisierelor fara flag explicit daca sunt la final
            inputs.push_back(arg);
        }
    }

    if(inputs.empty()) { 
        std::cerr << "Error: No input files specified.\n"; 
        print_help();
        return 1; 
    }

    std::cout << "--- Bloom Builder v7.5 (Auto-Detect Hybrid) ---\n";

    // 1. Numarare intrari (pentru calcul marime)
    uint64_t n=0;
    std::cout << "[INFO] Counting entries in input files...\n";
    for(const auto& file : inputs) {
        std::ifstream f(file);
        if(!f.is_open()) { std::cerr << "[WARN] Cannot open file: " << file << "\n"; continue; }
        std::string line;
        while(std::getline(f, line)) {
            // Curatam spatiile
            line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c){ return std::isspace(c); }), line.end());
            if(!line.empty() && line.length() > 10) n++;
        }
    }
    std::cout << "Total Entries: " << n << "\n";
    if(n==0) { std::cerr << "[ERR] No valid addresses found in input files.\n"; return 1; }

    // 2. Calcul marime necesara (m bits)
    double m_d = -1.0 * n * log(fpRate) / pow(log(2.0), 2.0);
    uint64_t m = (uint64_t)ceil(m_d);
    if (m < MIN_BLOOM_SIZE_BITS) m = MIN_BLOOM_SIZE_BITS;
    // Rotunjire la multiplu de 64
    m = ((m + 63) / 64) * 64; 
    
    // 3. DECIDERE AUTOMATA ALGORITM (HIBRID)
    bool useHighRes = false;
    uint8_t version = 3;

    if (m > BLOOM_32BIT_LIMIT) {
        useHighRes = true;
        version = 4; // Versiune pentru 64-bit Hash
        std::cout << "[AUTO] Large dataset detected (" << (m/8/1024/1024) << " MB).\n";
        std::cout << "[AUTO] Switched to 64-bit MurmurHash3 (High Precision/Collision Free).\n";
    } else {
        useHighRes = false;
        version = 3; // Versiune pentru 32-bit Hash
        std::cout << "[AUTO] Small/Medium dataset detected.\n";
        std::cout << "[AUTO] Using Standard 32-bit MurmurHash3 (Faster).\n";
    }

    // Calcul numar functii hash (k)
    uint32_t k = (uint32_t)round(((double)m / n) * log(2.0));
    if (k == 0) k = 1;
    
    std::cout << "Bloom Params: m=" << m << " bits (" << (m/8/1024/1024) << " MB) | k=" << k << " | Ver=" << (int)version << "\n";

    // Alocare memorie
    std::vector<uint8_t> bitarray;
    try {
        bitarray.resize(m/8, 0);
    } catch (const std::bad_alloc& e) {
        std::cerr << "[FATAL] Not enough RAM! Needed approx " << (m/8/1024/1024) << " MB.\n";
        return 1;
    }

    // 4. Procesare fisiere si populare filtru
    size_t addedCount = 0;
    size_t btcCount = 0;
    size_t ethCount = 0;

    for(const auto& file : inputs) {
        std::cout << "Processing file: " << file << "...\n";
        std::ifstream f(file);
        std::string line;
        while(std::getline(f, line)) {
            line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char c){ return std::isspace(c); }), line.end());
            if(line.empty()) continue;
            
            std::vector<uint8_t> payload;
            bool valid = false;

            // Detectie ETH
            if (line.size() == 42 && line.substr(0,2) == "0x") {
                payload = ParseEthAddress(line);
                if (payload.size() == 20) { valid = true; ethCount++; }
            }
            // Detectie BTC Legacy/P2SH (Base58)
            else if(line[0]=='1' || line[0]=='3') {
                if (DecodeBase58Check(line, payload)) {
                    // Base58Check are 1 byte version + 20 bytes hash + 4 bytes checksum
                    // DecodeBase58Check returneaza version+hash. Stergem versiunea (primul byte).
                    if (payload.size() == 21) { payload.erase(payload.begin()); valid = true; btcCount++; } 
                    else if (payload.size() == 20) { valid = true; btcCount++; }
                }
            }
            // Detectie BTC Bech32 (Segwit)
            else if(line.substr(0,3)=="bc1") {
                if (DecodeBech32(line, payload)) {
                    if (payload.size() == 20) { valid = true; btcCount++; }
                }
            }

            if(valid && payload.size() == 20) {
                uint64_t h1, h2;

                if (useHighRes) {
                    // --- MODUL HIGH PRECISION (64-BIT) ---
                    // Evita coliziunile la miliarde de intrari
                    uint64_t hash[2];
                    MurmurHash3_x64_128(payload.data(), 20, 0, hash);
                    h1 = hash[0];
                    h2 = hash[1];
                } else {
                    // --- MODUL STANDARD (32-BIT) ---
                    // Rapid, compatibil cu GPU-uri mai vechi daca e cazul
                    uint32_t r1 = MurmurHash3_x86_32(payload.data(), 20, 0xFBA4C795);
                    uint32_t r2 = MurmurHash3_x86_32(payload.data(), 20, 0x43876932);
                    h1 = (uint64_t)r1;
                    h2 = (uint64_t)r2;
                }

                // Setare biti in filtru
                for(uint32_t i=0; i<k; ++i) {
                    uint64_t idx = (h1 + i * h2) % m;
                    bitarray[idx/8] |= (1<<(idx%8));
                }
                addedCount++;
            }
        }
    }

    std::cout << "Successfully added " << addedCount << " unique addresses.\n";
    std::cout << "Stats: BTC=" << btcCount << " | ETH=" << ethCount << "\n";

    // 5. Salvare pe disc
    std::ofstream out(outputFile, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[ERR] Could not write to file: " << outputFile << "\n";
        return 1;
    }

    // Header Format: Magic(4) + Ver(1) + m(8) + k(4) + len(8) + data(...)
    out.write("BLM3", 4);
    out.write((char*)&version, 1); 
    write_be64(out, m); 
    write_be32(out, k); 
    write_be64(out, (uint64_t)bitarray.size());
    out.write((char*)bitarray.data(), bitarray.size());
    out.close();
    
    std::cout << "[DONE] Bloom Filter saved to: " << outputFile << " (Format Ver " << (int)version << ")\n";
    return 0;
}