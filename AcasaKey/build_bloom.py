import sys, math, struct, hashlib, argparse, os, base58

DEFAULT_FP_RATE = 0.0000001
MIN_BLOOM_SIZE_BITS = 8 * 1024 * 1024
BLOOM_K = 30  # Fixat pentru compatibilitate GPU

def murmur3_x86_32(data, seed):
    """Implementare MurmurHash3 bit-perfecta cu C++"""
    h1 = seed & 0xFFFFFFFF
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    nblocks = len(data) // 4
    for i in range(0, nblocks * 4, 4):
        k1 = struct.unpack("<I", data[i:i+4])[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
        h1 = ((h1 << 13) | (h1 >> 19)) & 0xFFFFFFFF
        h1 = (h1 * 5 + 0xe6546b64) & 0xFFFFFFFF
    tail = data[nblocks * 4:]
    k1 = 0
    if len(tail) >= 3: k1 ^= tail[2] << 16
    if len(tail) >= 2: k1 ^= tail[1] << 8
    if len(tail) >= 1: 
        k1 ^= tail[0]
        k1 = (k1 * c1) & 0xFFFFFFFF
        k1 = ((k1 << 15) | (k1 >> 17)) & 0xFFFFFFFF
        k1 = (k1 * c2) & 0xFFFFFFFF
        h1 ^= k1
    h1 ^= len(data)
    h1 ^= (h1 >> 16); h1 = (h1 * 0x85ebca6b) & 0xFFFFFFFF
    h1 ^= (h1 >> 13); h1 = (h1 * 0xc2b2ae35) & 0xFFFFFFFF
    h1 ^= (h1 >> 16)
    return h1

def decode_address_to_h160(addr):
    """Decodează adresa în Hash160 (20 bytes)"""
    try:
        if addr.startswith('1') or addr.startswith('3'):
            return base58.b58decode_check(addr)[1:]
        # Adresele bc1 necesită librărie bech32 (omisa pentru simplitate, adăugabilă)
    except: return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Python Bloom Builder v6.0")
    parser.add_argument("--input", action='append', required=True, help="Fisiere text cu adrese")
    parser.add_argument("--out", default="out.blf", help="Fisier Bloom iesire")
    parser.add_argument("--p", type=float, default=DEFAULT_FP_RATE, help="False Positive Rate")
    args = parser.parse_args()

    # 1. Numărare intrări valide
    valid_addresses = []
    print(f"[*] Scanare fisiere...")
    for path in args.input:
        with open(path, 'r') as f:
            for line in f:
                addr = line.strip()
                if addr.startswith(('1', '3', 'bc1')): valid_addresses.append(addr)
    
    n = len(valid_addresses)
    if n == 0: print("[!] Nu s-au gasit adrese valide."); return
    print(f"[*] Intrari gasite: {n}")

    # 2. Calcul parametri Bloom
    m_d = -1.0 * n * math.log(args.p) / (math.log(2.0) ** 2.0)
    m = int(math.ceil(m_d))
    if m < MIN_BLOOM_SIZE_BITS: m = MIN_BLOOM_SIZE_BITS
    m = ((m + 63) // 64) * 64
    print(f"[*] Parametri: m={m} bits, k={BLOOM_K}")

    # 3. Creare BitArray
    bitarray = bytearray(m // 8)

    # 4. Populează filtrul
    for addr in valid_addresses:
        payload = decode_address_to_h160(addr)
        if payload and len(payload) == 20:
            h1 = murmur3_x86_32(payload, 0xFBA4C795)
            h2 = murmur3_x86_32(payload, 0x43876932)
            for i in range(BLOOM_K):
                idx = (h1 + i * h2) % m
                bitarray[idx // 8] |= (1 << (idx % 8))

    # 5. Scriere fișier BLM3 (Format Big-Endian)
    with open(args.out, 'wb') as f:
        f.write(b"BLM3")
        f.write(struct.pack("B", 3)) # Versiune
        f.write(struct.pack(">Q", m)) # m (64-bit Big Endian)
        f.write(struct.pack(">I", BLOOM_K)) # k (32-bit Big Endian)
        f.write(struct.pack(">Q", len(bitarray))) # lungime date
        f.write(bitarray)
    
    print(f"[+] Bloom Filter creat: {args.out}")

if __name__ == "__main__": main()