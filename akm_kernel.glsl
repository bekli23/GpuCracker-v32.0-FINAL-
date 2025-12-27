#version 450
#extension GL_ARB_gpu_shader_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(local_size_x = 64) in;

// Bufferul de iesire (binding 0)
layout(std430, binding = 0) buffer OutputBuffer {
    uint result[];
} outputBuf;

// Argumente rapide (Push Constants)
layout(push_constant) uniform Constants {
    uint64_t baseSeed;
    uint points;
    uint useSequential;
    uint entropyBytes;
} pc;

// --- Helper Functions ---

// SplitMix64 (pentru initializare seed)
uint64_t splitmix64(inout uint64_t x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15UL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
    return z ^ (z >> 31);
}

// Rotire pe 64 biti
uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

// Xoshiro256** Next
uint64_t next(inout uint64_t s[4]) {
    const uint64_t res = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return res;
}

// Swap Endian (Little <-> Big)
uint swap_endian(uint val) {
    return ((val >> 24) & 0xff) | 
           ((val << 8) & 0xff0000) | 
           ((val >> 8) & 0xff00) | 
           ((val << 24) & 0xff000000);
}

// --- MAIN KERNEL ---
void main() {
    uint gid = gl_GlobalInvocationID.x;
    
    // State local RNG
    uint64_t s[4];

    // Initializare PRNG
    if (pc.useSequential == 0) {
        uint64_t seed = pc.baseSeed + gid;
        s[0] = splitmix64(seed);
        s[1] = splitmix64(seed);
        s[2] = splitmix64(seed);
        s[3] = splitmix64(seed);
    }

    // Calcul offset pentru scrierea counter-ului
    int offsetInt = int(pc.entropyBytes) / 4 - 2;
    if (offsetInt < 0) offsetInt = 0;
    uint counterOffset = uint(offsetInt);

    for (uint i = 0; i < pc.points; i++) {
        uint outIdx = (gid * pc.points) + i;
        uint baseOffset = outIdx * 8; 

        if (pc.useSequential == 1) {
            uint64_t uid = pc.baseSeed + outIdx;
            
            for(int k=0; k<8; k++) outputBuf.result[baseOffset + k] = 0;
            
            // Cast explicitly to uint64 to extract high/low bits then to uint
            uint high = uint(uid >> 32);
            uint low = uint(uid & 0xFFFFFFFFUL);

            outputBuf.result[baseOffset + counterOffset]     = swap_endian(high);
            outputBuf.result[baseOffset + counterOffset + 1] = swap_endian(low);
        } else {
            uint64_t r1 = next(s);
            uint64_t r2 = next(s);
            uint64_t r3 = next(s);
            uint64_t r4 = next(s);
            
            outputBuf.result[baseOffset + 0] = uint(r1 & 0xFFFFFFFFUL);
            outputBuf.result[baseOffset + 1] = uint(r1 >> 32);
            outputBuf.result[baseOffset + 2] = uint(r2 & 0xFFFFFFFFUL);
            outputBuf.result[baseOffset + 3] = uint(r2 >> 32);
            outputBuf.result[baseOffset + 4] = uint(r3 & 0xFFFFFFFFUL);
            outputBuf.result[baseOffset + 5] = uint(r3 >> 32);
            outputBuf.result[baseOffset + 6] = uint(r4 & 0xFFFFFFFFUL);
            outputBuf.result[baseOffset + 7] = uint(r4 >> 32);
        }
    }
}