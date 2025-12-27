# AKM Profiles & Modes - User Guide

This document explains the **AKM (Abstract Key Mapping)** system within `GpuCracker` and how to use advanced scanning modes and custom profiles.

## 1. Operation Modes (`--akm-mode`)

There are two fundamental generation modes available, changing the search strategy entirely.

### A. Schematic Mode (`--akm-mode schematic`)
* **How it works:** The GPU takes a sequential number (Seed Index: 0, 1, 2, 3...) and converts it into words using a mathematical base (Base-N).
* **Best for:** **Mathematical Puzzles (e.g., Puzzle 66, 71)** where keys are located within a specific, continuous range.
* **Output:** Generated phrases will look very similar (e.g., "abis abis abis..."), as only the "tail" of the number changes sequentially.
* **Key Parameters:**
    * `--akm-bit <N>`: Critical! Defines the range (e.g., 71 forces the key into the 2^70..2^71 interval).

### B. Random Mode (`--akm-mode random`)
* **How it works:** The GPU generates a completely random 256-bit Private Key (RNG) at every step.
* **Best for:** **General Hunting** (looking for collisions on rich, random addresses).
* **Output:** Phrases will be highly varied and chaotic (e.g., "mountain sky river fire...").
* **Key Parameters:**
    * `--akm-bit 256`: Recommended to search the entire Bitcoin keyspace.

---

## 2. Profile System (`--profile`)

An AKM Profile defines the **Word Dictionary** and the **Hexadecimal Rules** associated with each word.

### Internal Profiles (Built-in)
These are hardcoded into the `.exe` for maximum speed:
1.  **`akm2-core`**: Standard base profile.
2.  **`akm3-puzzle71`**: Optimized specifically for Challenge 71 (contains specific Romanian/English mappings required by the puzzle rules).
3.  **`akm2-fixed123-pack-v1`**: Experimental profile with fixed rules.

### External Profiles (Custom)
You can create your own profile without recompiling the software.

1.  Create a text file (e.g., `my_strategy.txt`) in the same folder as `GpuCracker.exe`.
2.  Write your rules in the format: `word=hex_value`.
3.  Run the tool adding: `--profile my_strategy`.

---

## 3. Verification Tool (`akm_seed2priv.exe`)

If `GpuCracker` finds a hit, use this tool to verify the phrase manually and get the exact Private Key (WIF).

**Important:** You must use the SAME settings as you did during the scan!

**Command for Puzzle 71 verification:**
```cmd
akm_seed2priv.exe --phrase "found words here..." --akm-bit 71 --mode schematic