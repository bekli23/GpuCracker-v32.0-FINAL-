# 🚀 GpuCracker v50.0

**GpuCracker** is a high-performance, modular cryptocurrency seed and private key recovery tool. It leverages massive hardware acceleration via **CUDA** and **OpenCL** to perform billions of cryptographic checks per second against target address databases using optimized Bloom Filters.

## 🌟 Key Features

* **8 Operating Modes**: Mnemonic, AKM, Check, Scan, XPRV, Brainwallet, Pubkey, and BSGS
* **Hybrid Architecture**: GPU for raw mathematical processing while CPU handles coordination
* **Multi-Backend Support**: Native optimization for NVIDIA (CUDA) and AMD/Intel (OpenCL)
* **Massive Speed**: Custom GPU engine capable of high-speed scanning with zero CPU bottlenecks
* **Ultra-Fast Filtering**: Integrated BLM3 Bloom Filter support for millions of targets
* **RPC Support**: Direct Bitcoin Core integration for balance checking

---

## 🛠️ System & Build Requirements

### 1. Software Prerequisites

| OS | Compiler | Dependencies |
|----|----------|--------------|
| Windows 10/11 | Visual Studio 2022 | CUDA 12.4, Boost, OpenSSL |
| Linux (Ubuntu/Debian) | GCC 11+ | CUDA 12.4, libsecp256k1, OpenCL |

### 2. Dependencies Installation

**Windows (vcpkg):**
```bash
vcpkg install openssl:x64-windows secp256k1:x64-windows boost-multiprecision:x64-windows
```

**Linux:**
```bash
sudo apt update
sudo apt install build-essential libssl-dev libsecp256k1-dev ocl-icd-opencl-dev libomp-dev libboost-all-dev
```

### 3. Build Instructions

**Windows (Visual Studio):**
```bash
# Open GpuCracker.sln in Visual Studio 2022
# Set: Release | x64
# Build: Ctrl+Shift+B
```

**Linux:**
```bash
make clean
make -j$(nproc)
```

---

## 📚 Mode Overview & Analysis

| Mode | Use Case | Speed | GPU Support | Best For |
|------|----------|-------|-------------|----------|
| **mnemonic** | BIP39 seed recovery | Medium | ✅ Yes | Lost wallet seeds |
| **akm** | Advanced Key Mapping | Fast | ✅ Yes | Cryptographic puzzles |
| **check** | Address verification | Very Fast | ✅ Yes | Database validation |
| **scan** | Blockchain scanning | Fast | ✅ Yes | Active monitoring |
| **xprv-mode** | Extended key derivation | Medium | ✅ Yes | HD wallet analysis |
| **brainwallet** | Password-based wallets | Slow | ✅ Yes | Weak passwords |
| **pubkey** | Public key brute force | Medium | ⚠️ CPU Only | Bitcoin puzzles |
| **bsgs** | Baby Step Giant Step | Medium | ❌ No | DLP attacks |

---

## 🔑 Mode 1: MNEMONIC (BIP39 Recovery)

### Description
Recovers BIP39 mnemonic phrases (12-24 words) using PBKDF2-HMAC-SHA512. Supports infinite random generation or sequential scanning.

### Parameters
```
--words N              Word count: 12, 15, 18, 21, 24
--language LANG        english, chinese_simplified, chinese_traditional, 
                       french, italian, japanese, korean, spanish
--order MODE           random, sequential, schematic
--prefix WORDS         Fixed prefix (e.g., "abandon abandon")
--brute PATTERN        Brute force pattern (e.g., "?????")
--infinite             Run until stopped
--entropy MODE         default, hex, bin, dice, random
```

### Basic Examples
```bash
# Standard 12-word brute force
GpuCracker.exe --mode mnemonic --words 12 --infinite --bloom-keys btc.blf

# 24-word with specific language
GpuCracker.exe --mode mnemonic --words 24 --language spanish --bloom-keys btc.blf

# Sequential with prefix
GpuCracker.exe --mode mnemonic --words 12 --order sequential \
    --prefix "abandon abandon abandon" --bloom-keys targets.blf

# Brute force last 3 words
GpuCracker.exe --mode mnemonic --words 12 \
    --prefix "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon" \
    --brute "???" --bloom-keys btc.blf
```

### Advanced Combinations
```bash
# Multi-GPU with specific devices
GpuCracker.exe --mode mnemonic --words 12 --infinite \
    --type cuda --device 0,1,2 --blocks 1024 --threads 512 \
    --bloom-keys btc.blf

# With RPC balance checking
GpuCracker.exe --mode mnemonic --words 12 --infinite \
    --rpc-check --rpc-user bitcoin --rpc-pass secret \
    --rpc-min-balance 100000

# Schematic entropy mode
GpuCracker.exe --mode mnemonic --words 12 --order schematic \
    --entropy schematic --bloom-keys btc.blf

# Show real generation speed
GpuCracker.exe --mode mnemonic --words 12 --infinite --speed
```

### Performance Analysis
| Hardware | Words | Speed (seeds/s) |
|----------|-------|-----------------|
| CPU (8 cores) | 12 | ~50,000 |
| GTX 1060 | 12 | ~500,000 |
| RTX 3090 | 12 | ~5,000,000 |
| RTX 4090 | 12 | ~10,000,000 |

---

## 🔑 Mode 2: AKM (Advanced Key Mapping)

### Description
Specialized for cryptographic puzzles using custom hexadecimal word mapping and Base-N math. Optimized for Bitcoin Challenge puzzles.

### Parameters
```
--akm-bit BITS         Target bit range: 66,67,68,71 or "66,67,68"
--akm-profile NAME     auto-linear, auto-random, akm3-puzzle71
--akm-mode MODE        random, sequential, schematic, wave
--akm-word N           Words for schematic mode
--akm-phrase TEXT      Starting phrase
--akm-file FILE        Load phrases from file
--akm-entropy MODE     Entropy generation mode
--akm-rotate-sec N     Rotate profile every N seconds
```

### Basic Examples
```bash
# Puzzle 71 optimized
GpuCracker.exe --mode akm --akm-bit 71 --profile akm3-puzzle71 \
    --bloom-keys puzzle71.blf --infinite

# Multiple bit ranges with rotation
GpuCracker.exe --mode akm --akm-bit 66,67,68 --profile auto-linear \
    --akm-rotate-sec 300 --bloom-keys puzzles.blf

# Schematic mode with 10 words
GpuCracker.exe --mode akm --akm-bit 68 --akm-mode schematic \
    --akm-word 10 --profile auto-linear

# Wave pattern generation
GpuCracker.exe --mode akm --akm-bit 71 --akm-mode wave \
    --bloom-keys puzzle71.blf
```

### Advanced Combinations
```bash
# Full puzzle setup with custom entropy
GpuCracker.exe --mode akm --akm-bit 71 --profile akm3-puzzle71 \
    --akm-entropy hex --akm-phrase "00000000000000000000" \
    --type cuda --device 0 --bloom-keys puzzle71.blf --win-file found.txt

# Multi-coin support
GpuCracker.exe --mode akm --akm-bit 66 --profile auto-linear \
    --multi-coin btc,eth,ltc --bloom-keys multi.blf

# List available profiles
GpuCracker.exe --mode akm --list-profiles
```

### Profile Reference
| Profile | Description | Best For |
|---------|-------------|----------|
| `auto-linear` | Sequential linear scan | General puzzles |
| `auto-random` | Random distribution | Large search spaces |
| `akm3-puzzle71` | Optimized for 71-bit puzzle | Bitcoin Challenge 71 |

### Performance Analysis
| Hardware | Profile | Speed (keys/s) |
|----------|---------|----------------|
| CPU | auto-linear | ~100,000 |
| GTX 1060 | akm3-puzzle71 | ~10,000,000 |
| RTX 4090 | akm3-puzzle71 | ~200,000,000 |

---

## 🔑 Mode 3: CHECK (Address Verification)

### Description
Fast address verification mode for checking if generated addresses match a target database without full derivation.

### Parameters
```
--check-mode MODE      verification, generation
--check-file FILE      Address list file
--check-format FORMAT  text, json, csv
```

### Basic Examples
```bash
# Verify addresses against bloom filter
GpuCracker.exe --mode check --check-mode verification \
    --check-file addresses.txt --bloom-keys btc.blf

# Generate and check simultaneously
GpuCracker.exe --mode check --check-mode generation \
    --mnemonic "word1 word2 ..." --bloom-keys btc.blf
```

### Advanced Combinations
```bash
# Batch verification with custom format
GpuCracker.exe --mode check --check-file wallets.json \
    --check-format json --bloom-keys btc.blf --win-file matches.txt

# With specific address types
GpuCracker.exe --mode check --setaddress SEGWIT,NATIVE \
    --bloom-keys btc.blf
```

---

## 🔑 Mode 4: SCAN (Blockchain Scanning)

### Description
Continuous scanning mode for monitoring blockchain activity. Can scan blocks, transactions, or addresses.

### Parameters
```
--scan-type TYPE       blocks, mempool, addresses
--scan-start N         Starting block height
--scan-end N           Ending block height (0 = current)
--scan-threads N       Number of scanner threads
--scan-interval N      Seconds between scans
```

### Basic Examples
```bash
# Scan recent blocks
GpuCracker.exe --mode scan --scan-type blocks \
    --scan-start 800000 --bloom-keys targets.blf

# Mempool monitoring
GpuCracker.exe --mode scan --scan-type mempool \
    --scan-interval 10 --bloom-keys targets.blf

# Address activity scan
GpuCracker.exe --mode scan --scan-type addresses \
    --scan-file addresses.txt
```

### Advanced Combinations
```bash
# Full blockchain scan with RPC
GpuCracker.exe --mode scan --scan-type blocks \
    --scan-start 0 --scan-end 850000 \
    --rpc-check --rpc-user user --rpc-pass pass \
    --bloom-keys historical.blf

# Real-time mempool monitoring with alert
GpuCracker.exe --mode scan --scan-type mempool \
    --scan-interval 5 --bloom-keys highvalue.blf \
    --win-file alerts.txt
```

---

## 🔑 Mode 5: XPRV-MODE (Extended Private Key)

### Description
Derives child keys from extended private keys (xprv) following BIP32 hierarchy.

### Parameters
```
--xprv KEY             Extended private key
--xprv-path PATH       Derivation path (e.g., "m/44'/0'/0'/0")
--xprv-start N         Start index
--xprv-end N           End index
--xprv-depth N         Maximum derivation depth
```

### Basic Examples
```bash
# Derive from xprv
GpuCracker.exe --mode xprv-mode \
    --xprv "xprv9s21ZrQH143K3GJpoapnV8SFfuw6..." \
    --xprv-path "m/44'/0'/0'/0" \
    --xprv-start 0 --xprv-end 100000 \
    --bloom-keys btc.blf

# Full wallet scan
GpuCracker.exe --mode xprv-mode \
    --xprv-file wallet.xprv \
    --xprv-depth 5 --bloom-keys targets.blf
```

### Advanced Combinations
```bash
# Multi-account derivation
GpuCracker.exe --mode xprv-mode \
    --xprv "xprv9s21ZrQH143K..." \
    --xprv-path "m/44'/0'/0'" \
    --xprv-start 0 --xprv-end 1000000 \
    --type cuda --device 0,1 \
    --bloom-keys btc.blf --win-file derived.txt

# Change addresses only
GpuCracker.exe --mode xprv-mode \
    --xprv "xprv9s21ZrQH143K..." \
    --xprv-path "m/44'/0'/0'/1" \
    --bloom-keys btc.blf
```

---

## 🔑 Mode 6: BRAINWALLET (Password-Based Recovery)

### Description
Recovers brain wallets - addresses derived from passwords/passphrases. Tests password combinations against target addresses.

### Parameters
```
--brain-wordlist FILE   Password wordlist
--brain-min N           Minimum password length
--brain-max N           Maximum password length
--brain-charset SET     Character set (ascii, alphanumeric, full)
--brain-rules RULES     Password transformation rules
```

### Basic Examples
```bash
# Wordlist-based brain wallet attack
GpuCracker.exe --mode brainwallet \
    --brain-wordlist passwords.txt \
    --bloom-keys btc.blf

# Brute force short passwords
GpuCracker.exe --mode brainwallet \
    --brain-min 4 --brain-max 8 \
    --brain-charset alphanumeric \
    --bloom-keys btc.blf
```

### Advanced Combinations
```bash
# With custom rules (uppercase, numbers, symbols)
GpuCracker.exe --mode brainwallet \
    --brain-wordlist passwords.txt \
    --brain-rules "u,s,n" \
    --type cuda --bloom-keys btc.blf

# Multi-GPU password cracking
GpuCracker.exe --mode brainwallet \
    --brain-min 8 --brain-max 12 \
    --brain-charset full \
    --type cuda --device 0,1,2,3 \
    --blocks 2048 --threads 512 \
    --bloom-keys btc.blf
```

### Performance Analysis
| Password Length | Charset | Est. Time (RTX 4090) |
|-----------------|---------|----------------------|
| 4 | alphanumeric | Seconds |
| 6 | alphanumeric | Minutes |
| 8 | alphanumeric | Days |
| 10 | alphanumeric | Years |

---

## 🔑 Mode 7: PUBKEY (Public Key Recovery)

### ⚠️ IMPORTANT: GPU Mode Limitation
**Currently Pubkey mode only works in CPU mode. GPU (CUDA) support is under development.**

The GPU kernel exists but requires additional fixes for proper hash160 calculation and modular arithmetic. Use CPU mode for reliable operation.

### Description
Brute-force search for private keys from public addresses within specific bit ranges. Optimized for Bitcoin puzzles (currently CPU-only).

### Parameters
```
--pub-address ADDR     Target Bitcoin address
--pub-bit N            Bit range (e.g., 71 for 2^71 search)
--pub-bit-start N      Starting bit
--pub-bit-end N        Ending bit
--pub-type TYPE        auto, p2pkh, p2sh, p2wpkh
--pubkey-blocks N      GPU blocks (default: 512) - CPU only
--pubkey-threads N     GPU threads per block (default: 512) - CPU only
```

### Basic Examples
```bash
# Puzzle 71 with CPU (RECOMMENDED)
GpuCracker.exe --mode pubkey \
    --pub-address 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so \
    --pub-bit 71 --type cpu

# Smaller ranges
GpuCracker.exe --mode pubkey \
    --pub-address 1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU \
    --pub-bit 40 --type cpu

# Custom bit range
GpuCracker.exe --mode pubkey \
    --pub-address 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so \
    --pub-bit-start 66 --pub-bit-end 71 \
    --type cpu
```

### Advanced Combinations
```bash
# Multi-threaded CPU (recommended for now)
GpuCracker.exe --mode pubkey \
    --pub-address 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so \
    --pub-bit 71 --type cpu \
    --cpu-cores 16 \
    --bloom-keys puzzle71.blf

# With custom address type
GpuCracker.exe --mode pubkey \
    --pub-address bc1q... \
    --pub-bit 71 --pub-type p2wpkh \
    --type cpu
```

### Performance Analysis (CPU Mode)
| Hardware | Speed (keys/s) | Time for 2^71 |
|----------|----------------|---------------|
| CPU (8 cores) | ~80,000 | ~6,000 years |
| CPU (16 cores) | ~160,000 | ~3,000 years |
| CPU (32 cores) | ~320,000 | ~1,500 years |
| CPU (64 cores) | ~640,000 | ~750 years |

### 🔧 GPU Mode Status
| Feature | Status | Notes |
|---------|--------|-------|
| CPU Mode | ✅ Working | Fully functional |
| CUDA GPU | ⚠️ In Development | Hash160 calculation needs fixes |
| OpenCL GPU | ❌ Not Available | Future implementation |

### Known Issues
1. **GPU mode falls back to CPU** - Even with `--type cuda`, pubkey mode currently uses CPU
2. **Hash160 mismatch in GPU kernel** - The GPU kernel produces different hash160 results than CPU
3. **Modular arithmetic** - Point multiplication on GPU needs verification

### Workaround for GPU Speed
For GPU acceleration, use **AKM mode** instead which has working GPU support:
```bash
# Use AKM mode for GPU puzzle solving
GpuCracker.exe --mode akm --akm-bit 71 --type cuda --device 0
```

---

## 🔑 Mode 8: BSGS (Baby Step Giant Step)

### Description
Solves the Elliptic Curve Discrete Logarithm Problem (ECDLP) using the Baby Step Giant Step algorithm. Recovers private key from public key.

### Parameters
```
--bsgs-pub HEX          Target public key (hex)
--bsgs-max N            Maximum key value (default: 2^40)
--bsgs-range BITS       Search range as bits (e.g., 135 for 2^135)
--bsgs-start OFFSET     Start search from offset
--bsgs-threads N        CPU threads (0 = auto)
```

### Basic Examples
```bash
# Search with default range (2^40)
GpuCracker.exe --mode bsgs \
    --bsgs-pub 031f6a332d3c5c4f2de2378c012f429cd109ba07d69690c6c701b6bb87860d6640 \
    --bsgs-threads 8

# Custom range with offset
GpuCracker.exe --mode bsgs \
    --bsgs-pub 0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798 \
    --bsgs-range 48 --bsgs-start 140737488355328

# Maximum range (256-bit support)
GpuCracker.exe --mode bsgs \
    --bsgs-pub 02... \
    --bsgs-range 135 --bsgs-threads 16
```

### Advanced Combinations
```bash
# Resume search from specific point
GpuCracker.exe --mode bsgs \
    --bsgs-pub 03... \
    --bsgs-range 50 --bsgs-start 562949953421312 \
    --bsgs-threads 16 --win-file bsgs_found.txt

# Large range with monitoring
GpuCracker.exe --mode bsgs \
    --bsgs-pub 03... \
    --bsgs-max 1099511627776 \
    --bsgs-threads 8 --verbose
```

### Algorithm Analysis
```
BSGS Complexity: O(√n) time, O(√n) memory

For range [0, N]:
- Baby steps: m = √N points stored in hash table
- Giant steps: √N iterations searching for match
- When found: k = i*m + j

Example for 2^48 range:
- m = 2^24 = 16,777,216 baby steps
- Memory: ~1.3 GB
- Time: Proportional to 2^24 operations
```

### Practical Limits
| Range | Memory Required | Time (1M steps/s) | Feasible? |
|-------|-----------------|-------------------|-----------|
| 2^40 | ~80 MB | ~17 min | ✅ Yes |
| 2^48 | ~1.3 GB | ~4.7 hours | ✅ Yes |
| 2^56 | ~20 GB | ~5 days | ⚠️ Hard |
| 2^64 | ~320 GB | ~3 months | ❌ No |
| 2^135 | ~10^20 GB | ~10^30 years | ❌ Impossible |

**Note**: For ranges > 48 bits, consider Pollard's Rho algorithm or distributed GPU BSGS.

---

## ⚙️ Global Options & Combinations

### Backend Selection
```bash
--type cuda          # NVIDIA GPUs (fastest)
--type opencl        # AMD/Intel GPUs
--type cpu           # CPU only
```

### GPU Configuration
```bash
--device ID          # Specific GPU (0, 1, 2, ... or -1 for all)
--blocks N           # CUDA blocks (default: 512)
--threads N          # Threads per block (default: 512)
```

### Bloom Filter Options
```bash
--bloom-keys FILE.blf       # Single filter
--bloom-keys FILE1,FILE2    # Multiple filters
--bloom-size SIZE           # Custom size
```

### RPC Configuration
```bash
--rpc-check                 # Enable RPC checking
--rpc-host 127.0.0.1       # Bitcoin Core host
--rpc-port 8332            # RPC port
--rpc-user username        # Username
--rpc-pass password        # Password
--rpc-min-balance 100000   # Minimum balance (satoshis)
```

### Output Options
```bash
--win-file FILE            # Save found keys
--verbose                  # Detailed output
--show-real-speed          # Actual generation rate
--quiet                    # Minimal output
```

### Configuration File (gpucracker.conf)
```ini
[general]
mode = mnemonic
words = 12
language = english
type = cuda
device = 0
bloom-keys = btc.blf
win-file = found.txt

[rpc]
host = 127.0.0.1
port = 8332
user = bitcoin
pass = secret

[pubkey]
target_address = 13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so
bit_start = 0
bit_end = 71
```

---

## 📊 Bloom Filter Setup (`build_bloom`)

### Build from Address List
```bash
# Single file
build_bloom.exe --input addresses.txt --out database.blf --p 0.0000001

# Multiple files
build_bloom.exe --input legacy.txt --input bech32.txt --out combined.blf

# Custom false positive rate
build_bloom.exe --input addresses.txt --out database.blf --p 0.000001 --size 1000000000
```

### Filter Specifications
| Target Addresses | False Positive Rate | Filter Size |
|------------------|---------------------|-------------|
| 1 million | 0.0000001 | ~32 MB |
| 10 million | 0.0000001 | ~320 MB |
| 100 million | 0.0000001 | ~3.2 GB |

---

## 🛡️ Attack Classes

| Class | Hardware | Speed (Est.) | Status |
|-------|----------|--------------|--------|
| **Class A** | Multi-core CPU | ~10K-100K h/s | Active (OpenMP) |
| **Class B** | GPU (GTX/RTX) | ~1M-1B h/s | Active |
| **Class C** | Specialized ASIC | ~100M h/s | N/A |
| **Class D** | ASIC Cluster | 1B+ h/s | N/A |

---

## ⚠️ Security & Legal Notice

**This tool is for legitimate recovery of your own wallets only!**

- Always verify you own the addresses before attempting recovery
- Use strong, unique passwords for brain wallets
- Keep backup copies of your seeds
- GPU mining generates significant heat - ensure proper cooling

**The developers assume no liability for misuse of this software.**

---

## 🐛 Troubleshooting

### Common Issues

**"Failed to load Bloom filter"**
```bash
# Ensure BLM3 format
build_bloom.exe --input addresses.txt --out filter.blf
```

**"CUDA out of memory"**
```bash
# Reduce blocks/threads
GpuCracker.exe ... --blocks 256 --threads 256
```

**"No OpenCL devices found"**
```bash
# Check drivers
sudo apt install nvidia-opencl-dev  # Linux
# Or use --type cpu for testing
```

**BSGS "Range too large" warning**
- This is expected for ranges > 48 bits
- Consider using pubkey mode instead for large ranges

---

## 📈 Performance Tips

1. **Use CUDA for NVIDIA** - 10x faster than OpenCL
2. **Optimize Bloom filter size** - Larger = fewer false positives
3. **Use multiple GPUs** - Linear scaling with `--device 0,1,2,3`
4. **Enable RPC for balance checking** - More accurate than bloom filters
5. **Profile your search** - Use `--show-real-speed` to optimize

---

## 🤝 Contributing

Pull requests welcome! Areas for contribution:
- Additional language support
- New attack modes
- GPU kernel optimizations
- Documentation improvements

---

## 📜 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- [secp256k1](https://github.com/bitcoin-core/secp256k1) - Bitcoin cryptography library
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - NVIDIA GPU computing
- [Boost](https://www.boost.org/) - C++ libraries

---

**Happy Hunting! 🔍**
