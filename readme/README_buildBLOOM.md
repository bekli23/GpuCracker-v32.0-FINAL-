
---

# Bloom Filter Builder v6.0 - Technical Documentation

This utility is a high-performance pre-processor for **GpuCracker**. It converts large text files containing Bitcoin addresses into an optimized binary Bloom Filter (`.blf`), allowing the GPU to verify billions of seeds against millions of target addresses with near-zero latency.

---

## 1. Overview & Compatibility

The builder uses a **Double Hashing** strategy based on **MurmurHash3**, which is bit-perfectly identical to the implementation in the main GPU kernels (`GpuCore.cu` and `mnemonic_gpu.cu`).

* **Format**: BLM3 (Binary Bloom Filter v3).
* **Supported Addresses**:
* **Legacy / P2SH**: Base58Check decoding.
* **Native SegWit**: Bech32/Bech32m decoding (bc1 addresses).


* **Critical Seeds**: Uses Murmur3 seeds `0xFBA4C795` and `0x43876932` to ensure GPU matching.

---

## 2. Build Requirements

To compile `build_bloom.cpp`, you need:

* **Compiler**: GCC, Clang, or MSVC (v143 recommended).
* **Libraries**: **OpenSSL** (required for SHA256 and RIPEMD160 logic during address verification).

### Compilation Example (Windows/vcpkg):


g++ build_bloom.cpp -o build_bloom.exe -lssl -lcrypto



---

## 3. Usage & CLI Arguments

The tool scans your input files, counts valid addresses, and automatically calculates the optimal bit-array size based on your desired False Positive Rate.

| Flag | Purpose | Default |
| --- | --- | --- |
| `--input <file>` | Path to the address list (text). Can be used multiple times to merge lists. | Required |
| `--out <file>` | Output filename for the binary filter. | `out.blf` |
| `--p <rate>` | **False Positive Rate**. Lower means more accuracy but larger file size. | `0.0000001` |

---

## 4. Examples

### A. Basic Generation

Convert a single list of addresses into a filter:


build_bloom.exe --input addresses.txt --out database.blf



### B. High Accuracy / Large Scale

For databases with millions of addresses where you want zero false alarms:


build_bloom.exe --input btc_richlist.txt --out secure.blf --p 0.000000001



### C. Merging Multiple Sources

Combine multiple text files into a single binary filter:


build_bloom.exe --input legacy.txt --input segwit.txt --out combined.blf



---

## 5. Technical Specifications (BLM3)

The generated binary file contains a 25-byte header:

1. **Magic Bytes**: `BLM3` (4 bytes).
2. **Version**: `0x03` (1 byte).
3. **Bit Size (M)**: Big-endian 64-bit integer.
4. **Hash Count (K)**: Big-endian 32-bit integer (Fixed to **30** for GPU compatibility).
5. **Data Length**: Big-endian 64-bit integer.

**Important**: The `k` value is hardcoded to 30 in the builder to match the constant loop in the GPU search kernels, ensuring maximum search throughput.

---

### Recomandări Utile:

* **List Format**: Make sure `.txt` files contain only one address per line. Blank lines or comments are automatically ignored.
* **Speed**: The utility is optimized to process lists of millions of addresses in a few seconds.
* **Usage**: After generation, use the resulting file with the `--bloom-keys` flag in the main `GpuCracker` application.