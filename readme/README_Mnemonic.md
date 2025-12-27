# Mnemonic Mode (BIP39) - Technical Documentation

The **Mnemonic Module** in **GpuCracker v32.0 (FINAL)** is a high-performance recovery engine designed to verify and brute-force BIP39-compliant mnemonic phrases. It utilizes a massive parallel architecture to perform heavy cryptographic operations—including PBKDF2 hashing and Elliptic Curve cryptography—directly on the GPU.

---

## 1. Core Logic: Seed to Address

The module follows the standard BIP39 and BIP32/BIP44/BIP49/BIP84 specifications to derive Bitcoin addresses from mnemonic phrases.

### The Derivation Pipeline

1. **Entropy Generation**: The system generates entropy randomly or sequentially based on the selected `--mnemonic-order`.
2. **Mnemonic Encoding**: Entropy is mapped to words using standard BIP39 wordlists (e.g., English, French).
3. **Seed Derivation (PBKDF2)**: The mnemonic is hashed using **PBKDF2-HMAC-SHA512** with **2048 iterations** to produce a 512-bit seed.
4. **HD Wallet Derivation**: The 512-bit seed is used to derive private keys for various paths (Legacy, SegWit, etc.).
5. **Address Generation**: Private keys are converted to public keys via Secp256k1 and then into hashed Bitcoin addresses (Hash160).

---

## 2. Key Features

* **Supported Word Counts**: Handles 12, 15, 18, 21, and 24-word phrases.
* **Multiple Derivation Paths**: Simultaneously checks target addresses across several standards:
* `m/0/0` (Standard BIP32/Legacy).
* `m/44'/0'/0'/0/0` (BIP44 Legacy).
* `m/49'/0'/0'/0/0` (BIP49 Nested SegWit).
* `m/84'/0'/0'/0/0` (BIP84 Native SegWit/Bech32).


* **Multi-Language Support**: Compatible with any standard BIP39 wordlist file located in the `bip39/` directory.

---

## 3. GPU Acceleration (Class B Attack)

Unlike many tools that perform hashing on the CPU, GpuCracker executes the entire **PBKDF2-HMAC-SHA512 (2048 rounds)** and **Secp256k1 multiplication** cycle inside the GPU kernel.

* **Fast Bloom Filtering**: Derived Hash160 values are instantly compared against a high-speed Bloom Filter stored in GPU memory.
* **Zero CPU Bottleneck**: The GPU handles the "muscle work," while the CPU only reconstructs the final phrase when a "HIT" is detected.
* **Class B Strategy**: Optimized for high-end consumer GPUs (GTX 1080 to RTX 4090), achieving speeds significantly higher than CPU-only methods.

---

## 4. CLI Usage & Parameters

To run in Mnemonic mode, use `--mode mnemonic` with the following options:

### Essential Commands

* `--langs NAME`: Select the wordlist language (default: `english`).
* `--words N`: Set the phrase length (12, 15, 18, 21, 24).
* `--mnemonic-order [random|sequential]`: Choose how seeds are generated.
* `--bloom-keys FILE`: Load the target address filter.
* `--setaddress TYPE`: Filter the output for specific address types (e.g., `P2WPKH`, `P2SH`).

### Example Command


GpuCracker.exe --mode mnemonic --langs english --words 12 --mnemonic-order random --bloom-keys my_addresses.blf --type cuda --device 0 --blocks 1024 --threads 512



---

## 5. System Status Indicators

When running in Mnemonic mode, the interace provides real-time statistics:

* **Total Addresses**: Total derived addresses across all active paths (calculated as `Seeds * 4`).
* **VRAM Usage**: Real-time memory monitoring for CUDA devices.
* **OpenMP Status**: Reports if CPU multi-threading is active for supporting tasks.