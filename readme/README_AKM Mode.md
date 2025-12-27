# AKM Mode (Advanced Key Mapping) - Technical Documentation

The **AKM Mode** in **GpuCracker v32.0 (FINAL)** is a high-performance, deterministic engine designed to transform mnemonic phrases (word lists) into Bitcoin private keys using custom mapping rules instead of standard BIP39 PBKDF2. This module is specifically optimized for solving cryptographic puzzles and recovering phrases with known patterns.

---

## 1. Core Logic: Phrase to Private Key

Unlike BIP39, which uses 2048 rounds of PBKDF2, AKM uses **Hexadecimal Concatenation** and **Base-N math** to generate keys nearly instantaneously.

### The Conversion Process

1. **Tokenization**: Each word in a phrase is converted into a 4-to-8 character hexadecimal token based on the active profile.
2. **Building the BigHex**: These tokens are concatenated until a 64-character hex string (32 bytes / 256 bits) is formed.
3. **Bit Masking (Range Control)**: If `--akm-bit <N>` is used, the program clears all bits above , forces bit  to 1, and treats the resulting value as the final private key.

---

## 2. Operation Modes

AKM supports two fundamental ways to generate and search for keys:

### A. Random / Hash Mode (`--akm-mode random`)

* **Workflow**: The GPU generates a random 64-bit "Seed". This seed is used as a starting point to generate a 256-bit private key directly on the hardware.
* **CPU Reconstruction**: If the GPU finds a "Hit" in the Bloom Filter, the CPU takes the winning seed, reconstructs the random sequence, and maps it back to human-readable words using the AKM wordlist.

### B. Schematic Mode (`--akm-mode schematic`)

* **Logic**: Treats the phrase as a single massive number in a base equal to the wordlist size ().
* **Use Case**: Ideal for sequential searches or brute-forcing phrases where the mathematical relationship between words is linear.

---

## 3. Profile Types

Profiles define the "rules" for how a word becomes hex.

| Profile | Type | Description |
| --- | --- | --- |
| `auto-linear` | Automatic | Automatically maps a word's index in the list to its hex value (Index 10 = `a`, Index 256 = `100`). |
| `akm3-puzzle71` | Hardcoded | Optimized for specific puzzles; contains hundreds of manual rules (e.g., `abis=100`). |
| `custom.txt` | External | Users can provide a `.txt` file with `word=hex` pairs to define their own logic. |

---

## 4. CLI Usage & Parameters

To trigger AKM mode, use `--mode akm` followed by your configuration:

### Essential Commands

* `--profile NAME`: Select the mapping logic (e.g., `auto-linear` or `akm3-puzzle71`).
* `--akm-bit LIST`: Limit the search to specific bit ranges (e.g., `71,72`).
* `--akm-word N`: Set the fixed phrase length (e.g., `10` or `12` words).
* `--akm-mode random|schematic`: Choose the generation strategy.
* `--input FILE`: Provide a text file containing specific phrases to check against the Bloom Filter.

### Example Command


GpuCracker.exe --mode akm --profile auto-linear --akm-bit 71 --akm-word 10 --akm-mode random --bloom-keys my_filter.blf --type cuda --device 0



---

## 5. Hybrid Architecture & Performance

The AKM module utilizes a **Hybrid CPU/GPU** approach to maximize throughput:

* **GPU (The Muscle)**: Handles the raw mathematical generation and Bloom Filter comparisons. It does not "understand" words; it only sees seeds and hashes.
* **CPU (The Intelligence)**: Manages profile loading, UI updates, and reconstructing winning seeds back into word phrases once a match is found.

### Attack Classes (Class B Logic)

The UI indicates system status according to the hardware capability:

* **Class A**: Multi-core CPU (~10K h/s).
* **Class B**: High-end GPU (GTX 1080/RTX 4090) (~1M h/s) — **Active Experimental Status**.
* **Class C/D**: ASIC / Supercluster levels (~100M - 1B+ h/s).