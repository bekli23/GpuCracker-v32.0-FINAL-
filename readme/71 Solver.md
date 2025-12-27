Here is the English version of the `README.md` file, professional and ready to use for your project.

---

# 📄 README - GpuCracker: Bitcoin Puzzle #71 Solver

This document outlines the optimized configuration for running **GpuCracker** to solve **Bitcoin Puzzle #71** (target range  - ). The script uses `akm` (Advanced/Auto Kernel Mode) for high-speed randomized scanning on the GPU.

---

## 🚀 Run Command (Quick Start)

Use the following command line to start the cracking process.


GpuCracker.exe --mode akm --type auto --device 0 --blocks 512 --threads 512 --points 633 --cores 1 --infinite --bloom-keys puzzle.blf --win winner.txt --akm-mode random --akm-word 15 --profile auto-linear --akm-bit 71



### 📋 Required Files

Ensure the following files are in the same directory as the executable:

* `GpuCracker.exe`: The main application.
* `puzzle.blf`: The Bloom Filter file containing the target public addresses (must be correctly generated for Puzzle 71).

---

## ⚙️ Parameter Explanation

To help you understand the configuration, here is a breakdown of the arguments used:

### 1. Hardware & Performance

| Parameter | Value | Description |
| --- | --- | --- |
| `--device` | `0` | Selects the first GPU in the system. Use `1` for the second card, etc. |
| `--blocks` | `512` | Number of CUDA blocks allocated. *Note: Adjust based on available VRAM.* |
| `--threads` | `512` | Number of threads per block. Total threads = Blocks × Threads. |
| `--type` | `auto` | Automatically detects the GPU architecture for optimization. |
| `--profile` | `auto-linear` | Performance profile used for resource allocation. |

### 2. Search Logic (Puzzle Logic)

| Parameter | Value | Description |
| --- | --- | --- |
| `--akm-bit` | `71` | **Critical:** Defines the specific bit range for Puzzle 71 (...). |
| `--akm-mode` | `random` | Scans keys randomly within the range. Preferred for large ranges where sequential scanning is statistically less effective. |
| `--infinite` | (flag) | Runs the process indefinitely without stopping after a set number of keys. |
| `--bloom-keys` | `puzzle.blf` | Loads the target address database using a Bloom Filter (essential for high speed). |

### 3. Output & Results

| Parameter | Value | Description |
| --- | --- | --- |
| `--win` | `winner.txt` | If a private key is found (WIF/Hex), it will be automatically saved to this file. |

---

## 🔧 Optimization & Tuning

If the performance (Mkeys/s) is low or the GPU is unstable, consider adjusting these parameters:

1. **Blocks & Threads:**
* For high-end cards (e.g., RTX 3090/4090), you can try increasing `--blocks` to `1024` or higher (multiples of 128).
* If you encounter "Kernel launch failed" errors, try decreasing `--threads` to `256`.


2. **Points:**
* The value `--points 633` is specific to the elliptic curve point compression used in this mode. Only change this if you have advanced knowledge of the specific `akm` algorithm build.



---

## ⚠️ Important Note

> [!IMPORTANT]
> **GPU Cooling:** This process will keep your GPU at 100% load constantly. Ensure you have adequate ventilation and monitor temperatures (ideally keep it under 70°C for 24/7 operation).

---

### Windows `.bat` Script Example

To avoid typing the command manually every time, create a `start_puzzle71.bat` file with the following content. It includes a loop to auto-restart if the miner crashes.

batch
@echo off
title GpuCracker - Puzzle 71 Worker
:loop
GpuCracker.exe --mode akm --type auto --device 0 --blocks 512 --threads 512 --points 633 --cores 1 --infinite  --bloom-keys puzzle.blf --win winner.txt --akm-mode random --akm-word 10 --profile auto-linear  --akm-bit 71
echo.
echo Process stopped unexpectedly. Restarting in 10 seconds...
timeout /t 10
goto loop

