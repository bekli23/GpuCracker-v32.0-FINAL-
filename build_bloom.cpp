#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <cstring>

// OpenSSL - Necesar pentru validarea checksum-ului adreselor
#include <openssl/sha.h>
#include <openssl/ripemd.h>

#define DEFAULT_FP_RATE 0.0000001
#define MIN_BLOOM_SIZE_BITS (8 * 1024 * 1024) // Minimum 1 MB (8388608 biți)

// --- REZOLVARE CONFLICTE TIPURI (Cross-Platform) ---
#ifdef _WIN32
    typedef unsigned long long ulong;
    typedef unsigned int uint;
#else
    #include <sys/types.h>
    // Pe Linux, ulong și uint sunt deja definite în sys/types.h
#endif

// ============================================================================
//  MURMURHASH3 (IDENTIC CU CEL DIN BLOOM.H / GPU)
// ============================================================================
inline uint32_t rotl32(uint32_t x, int8_t r) { return (x << r) | (x >> (32 - r)); }

inline uint32_t MurmurHash3_x86_32(const void* key, int len, uint32_t seed) {
    const uint8_t* data = (const uint8_t*)key;
    const int nblocks = len / 4;
    uint32_t h1 = seed;
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // Folosim memcpy pentru a evita erorile de aliniere (Bus Error) pe Linux
    const uint8_t* blocks = (data + nblocks * 4);
    for (int i = -nblocks; i; i++) {
        uint32_t k1;
        memcpy(&k1, blocks + (i * 4), 4);

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

// Helpers pentru scriere Big-Endian (Format BLM3)
void write_be64(std::ofstream& f, uint64_t val) {
    uint8_t bytes[8]; for (int i=0; i<8; ++i) bytes[7-i] = (val >> (i*8)) & 0xFF;
    f.write((char*)bytes, 8);
}
void write_be32(std::ofstream& f, uint32_t val) {
    uint8_t bytes[4]; for (int i=0; i<4; ++i) bytes[3-i] = (val >> (i*8)) & 0xFF;
    f.write((char*)bytes, 4);
}
void DoubleSHA256(const void* data, size_t len, uint8_t* output) {
    uint8_t hash1[SHA256_DIGEST_LENGTH]; SHA256((const uint8_t*)data, len, hash1);
    SHA256(hash1, SHA256_DIGEST_LENGTH, output);
}

// Logica de decodare Base58 pentru adrese Legacy/P2SH
const char* pszBase58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
int8_t mapBase58[256];
void InitBase58() { memset(mapBase58, -1, sizeof(mapBase58)); for (int i = 0; pszBase58[i]; i++) mapBase58[(uint8_t)pszBase58[i]] = i; }

bool DecodeBase58Check(const std::string& str, std::vector<uint8_t>& vchRet) {
    if (mapBase58['1'] == -1) InitBase58();
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
    if (data.size() <= 1) return false;
    vchRet.assign(data.begin() + 1, data.end()); 
    return true;
}

// Logica de decodare Bech32 pentru adrese Native SegWit
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

void print_help() {
    std::cout << "DESCRIPTION:\n";
    std::cout << "  High-speed Bitcoin address to Bloom Filter (.blf) converter.\n\n";
    std::cout << "USAGE:\n";
    std::cout << "  build_bloom.exe --input <file> [options]\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --input <file>      : Path to address list (.txt). Can be used multiple times.\n";
    std::cout << "  --out <file>        : Output filename (Default: out.blf)\n";
    std::cout << "  --p <rate>          : False Positive Rate (Default: 0.0000001)\n";
    std::cout << "  --help              : Show this message.\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) { print_help(); return 0; }

    InitBase58(); 
    std::vector<std::string> inputs;
    std::string outputFile = "out.blf";
    double fpRate = DEFAULT_FP_RATE;

    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if(arg=="--input" && i+1<argc) inputs.push_back(argv[++i]);
        else if(arg=="--out" && i+1<argc) outputFile = argv[++i];
        else if(arg=="--p" && i+1<argc) fpRate = std::stod(argv[++i]);
        else if(arg=="--help" || arg=="-h") { print_help(); return 0; }
    }

    if(inputs.empty()) { std::cerr << "Error: No input files.\n"; return 1; }

    std::cout << "--- Bloom Builder v6.0 (Double Hashing - GPU Compatible) ---\n";

    uint64_t n=0;
    for(const auto& file : inputs) {
        std::ifstream f(file);
        if(!f.is_open()) continue;
        std::string line;
        while(std::getline(f, line)) {
            if(!line.empty() && (line[0]=='1' || line[0]=='3' || line.substr(0,3)=="bc1")) n++;
        }
    }
    std::cout << "Total Valid Entries: " << n << "\n";
    if(n==0) return 1;

    // Calcul dimensiune optimă (m) și număr hash-uri (k)
    double m_d = -1.0 * n * log(fpRate) / pow(log(2.0), 2.0);
    uint64_t m = (uint64_t)ceil(m_d);
    if (m < MIN_BLOOM_SIZE_BITS) m = MIN_BLOOM_SIZE_BITS;
    m = ((m + 63) / 64) * 64; // Aliniere la 64 biți

    uint32_t k = 30; // FIXAT la 30 pentru compatibilitate cu loop-ul constant din GPU
    std::cout << "Bloom Params: m=" << m << " bits | k=" << k << "\n";

    std::vector<uint8_t> bitarray(m/8, 0);

    for(const auto& file : inputs) {
        std::ifstream f(file);
        std::string line;
        while(std::getline(f, line)) {
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
            if(line.empty()) continue;
            size_t first = line.find_first_not_of(" \t");
            if(std::string::npos == first) continue;
            std::string addr = line.substr(first, (line.find_last_not_of(" \t")-first+1));
            
            std::vector<uint8_t> payload;
            bool valid = false;
            if(addr[0]=='1' || addr[0]=='3') valid = DecodeBase58Check(addr, payload);
            else if(addr.substr(0,3)=="bc1") valid = DecodeBech32(addr, payload);

            if(valid && payload.size() == 20) {
                // Seed-uri critice Murmur3: 0xFBA4C795 și 0x43876932
                uint32_t h1 = MurmurHash3_x86_32(payload.data(), 20, 0xFBA4C795);
                uint32_t h2 = MurmurHash3_x86_32(payload.data(), 20, 0x43876932);

                for(uint32_t i=0; i<k; ++i) {
                    uint64_t idx = ((uint64_t)h1 + (uint64_t)i * h2) % m;
                    bitarray[idx/8] |= (1<<(idx%8));
                }
            }
        }
    }

    std::ofstream out(outputFile, std::ios::binary);
    out.write("BLM3", 4);
    uint8_t ver=3; out.write((char*)&ver, 1);
    write_be64(out, m); 
    write_be32(out, k); 
    write_be64(out, (uint64_t)bitarray.size());
    out.write((char*)bitarray.data(), bitarray.size());
    
    std::cout << "[DONE] Bloom Filter Created: " << outputFile << "\n";
    return 0;
}