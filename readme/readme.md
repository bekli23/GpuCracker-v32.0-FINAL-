Sigur, iată fișierul `README.md` complet, gata de utilizare.

Am inclus toate argumentele definite în `args.h`, am structurat exemplele pentru modul AKM și am adăugat și o secțiune specială pentru utilitarul `akm_seed2priv` (bazat pe codul `seed2priv_main.cpp` pe care mi l-ai arătat anterior), deoarece cele două merg mână în mână.

---

markdown
# GpuCracker v9.10 - Modular Seed & Key Recovery Tool

**GpuCracker** is a high-performance, hybrid CPU/GPU tool designed for recovering cryptocurrency keys. It features a modular architecture supporting both standard BIP39 mnemonic recovery and specialized **AKM (Advanced Key Mode)** logic for cryptographic puzzles.

## 🚀 Key Features

* **Hybrid Processing:** Utilizes CPU and GPU (CUDA/OpenCL) simultaneously for maximum throughput.
* **Dual Operation Modes:**
    * `mnemonic`: Standard BIP39 brute-forcing (12/24 words).
    * `akm`: Advanced logic for specific puzzles (e.g., Puzzle 66, 67, 71).
* **Bloom Filter Integration:** Ultra-fast address checking using `.blf` databases.
* **Smart Profiles:** Preset logic for known puzzles (e.g., `akm3-puzzle71`).
* **Verification Tool:** Includes `akm_seed2priv` for verifying found seeds.

---

## 🛠 Prerequisites

Before running, ensure you have:
1.  **Bloom Filter File:** A `.blf` file containing target hashes (e.g., `btc.blf`).
2.  **Wordlists:**
    * For BIP39: `bip39/english.txt`
    * For AKM: `akm/wordlist_512_ascii.txt` (or similar).

---

## 🧩 AKM Mode (Advanced Key Mode)

This mode is designed for algorithmic puzzles where seeds are generated using non-standard methods (e.g., specific word lengths, math-based schematic logic).

### 1. Solving Puzzle 71 (Default Profile)
The default profile `akm3-puzzle71` generates phrases of lengths 3, 5, 10, 15, 20, 25, 30.


# Check range 71 continuously
GpuCracker.exe --mode akm --profile akm3-puzzle71 --akm-bit 71 --infinite --bloom-keys btc.blf



### 2. Solving Puzzle 66 & 67

You can target multiple bit ranges simultaneously.


# Check ranges 66 and 67
GpuCracker.exe --mode akm --akm-bit 66,67 --infinite --bloom-keys btc.blf



### 3. Loading Candidates from a File

If you have a list of pre-generated phrases to check.


# --input is an alias for --akm-file
GpuCracker.exe --mode akm --input candidates.txt --profile akm3-puzzle71 --bloom-keys btc.blf



### 4. Schematic Mode (Base-N Logic)

For puzzles that use mathematical word-to-index calculation instead of hashing.


# Use Schematic logic for Puzzle 68
GpuCracker.exe --mode akm --akm-mode schematic --akm-bit 68 --infinite --bloom-keys btc.blf



### 5. Custom Word Lengths

Force the generator to only use specific phrase lengths (e.g., only 12 words).


# Only generate 12-word phrases
GpuCracker.exe --mode akm --words 12 --infinite --bloom-keys btc.blf

# Generate 3 and 5 words only
GpuCracker.exe --mode akm --akm-word 3,5 --infinite --bloom-keys btc.blf



### Available Profiles

* `akm3-puzzle71` (Standard for current puzzles)
* `auto-linear` (Direct Index mapping)
* `akm2-core` (Legacy logic)
* `akm2-lab-v1` (Experimental)

---

## 🔑 Mnemonic Mode (BIP39 Standard)

Use this for recovering standard wallet seeds (Ledger, Trezor, Electrum, TrustWallet).

### 1. Standard Brute Force


# 12 Words, English
GpuCracker.exe --mode mnemonic --words 12 --infinite --bloom-keys btc.blf



### 2. Target Specific Address Types

Optimize speed by generating only the address type you need.

* **Types:** `ALL` (default), `LEGACY` (starts with 1), `SEGWIT` (starts with bc1), `P2SH` (starts with 3).


GpuCracker.exe --mode mnemonic --words 24 --setaddress SEGWIT --infinite --bloom-keys btc.blf



---

## 🔬 Helper Tool: akm_seed2priv

Use this standalone utility to verify a seed found by GpuCracker and view its Private Key and addresses.

**Usage:**


akm_seed2priv.exe --phrase "word1 word2 ..." --profile <PROFILE_NAME> [--mode schematic]



**Examples:**


# Verify a standard Puzzle 71 seed
akm_seed2priv.exe --phrase "alpha beta gamma" --profile akm3-puzzle71

# Verify a Schematic seed
akm_seed2priv.exe --phrase "alpha beta" --profile auto-linear --mode schematic



---

## ⚙️ Performance Tuning

### GPU Configuration

If `auto` doesn't detect your card correctly, or to tune performance:

| Flag | Description | Example |
| --- | --- | --- |
| `--type` | Backend (`cuda`, `opencl`) | `--type cuda` |
| `--device` | GPU ID (0, 1, 2...) | `--device 0` |
| `--blocks` | CUDA Blocks | `--blocks 512` |
| `--threads` | Threads per Block | `--threads 256` |
| `--points` | Keys per thread | `--points 4` |

### General Flags

| Flag | Description |
| --- | --- |
| `--infinite` | Run until stopped manually. |
| `--count N` | Stop after checking N seeds. |
| `--win FILE` | File to save found keys (Default: `win.txt`). |
| `--cores N` | Limit CPU usage to N cores. |
| `--quiet` | Minimal output mode. |

---

**Disclaimer:** This software is for educational purposes and security research only. The authors are not responsible for any misuse of this tool.



