
# 🚀 GpuCracker 

**GpuCracker** is a high-performance, modular cryptocurrency seed and private key recovery tool. It leverages massive hardware acceleration via **CUDA**, **OpenCL**, and **Vulkan** to perform billions of cryptographic checks per second against target address databases using optimized Bloom Filters.

## 🌟 Key Features

* **Dual Operating Modes**: Standard BIP39 Mnemonic recovery and specialized AKM (Advanced Key Mapping) for cryptographic puzzles.
* **Hybrid Architecture**: Utilizes GPU for raw mathematical processing (Class B logic) while the CPU handles phrase reconstruction and UI updates.
* **Multi-Backend Support**: Native optimization for NVIDIA (CUDA) and cross-platform compatibility for AMD/Intel (OpenCL/Vulkan).
* **Massive Speed**: Includes a custom **Class B GPU engine** capable of high-speed scanning with zero CPU bottlenecks.
* **Ultra-Fast Filtering**: Integrated BLM3 Bloom Filter support for instantaneous address matching against millions of targets.

---

## 🛠️ System & Build Requirements

### 1. Software Prerequisites

* **OS**: Windows 10/11 (x64).
* **IDE**: Visual Studio 2022 (v143 build tools).
* **SDKs**: CUDA Toolkit 12.4 and Vulkan SDK 1.4.335.0.
* **Dependencies**: Handled via `vcpkg`. Install required libraries:
```bash
vcpkg install openssl:x64-windows secp256k1:x64-windows


### 2. Build Instructions

1. Open the `.vcxproj` file in Visual Studio 2022.
2. Set configuration to **Release | x64**.
3. Ensure **CUDA 12.4 Build Customizations** are enabled.
4. Build the solution (`Ctrl+Shift+B`). The binary `GpuCracker.exe` will be generated in `bin\x64\Release\`.
5. Build for linux
sudo apt update
sudo apt install build-essential libssl-dev libsecp256k1-dev ocl-icd-opencl-dev libomp-dev
make
Build Again
After applying these changes, clean your previous build attempts and run make again:

make clean
make -j$(nproc) or make

---

## 🔑 Operating Modes & Examples

### 1. Mnemonic Mode (BIP39 Standard)

Used for recovering standard wallet seeds (12-24 words) using PBKDF2-HMAC-SHA512 (2048 rounds).

* **Standard Brute Force (12 words, English)**:
```bash
GpuCracker.exe --mode mnemonic --words 12 --infinite --bloom-keys btc.blf --type cuda --device 0

```


* **Target Specific Address Types (SegWit only)**:
```bash
GpuCracker.exe --mode mnemonic --words 24 --setaddress SEGWIT --bloom-keys btc.blf

```



### 2. AKM Mode (Advanced Key Mapping)

Designed for cryptographic puzzles (like Bitcoin Challenge 71) using custom hexadecimal word mapping and Base-N math.

* **Solve Puzzle 71 (Optimized Profile)**:
```bash
GpuCracker.exe --mode akm --profile akm3-puzzle71 --akm-bit 71 --infinite --bloom-keys puzzle.blf --akm-mode random

```


* **Target Multiple Bit Ranges (66 and 67)**:
```bash
GpuCracker.exe --mode akm --akm-bit 66,67 --profile auto-linear --bloom-keys my_filter.blf

```


* **Schematic (Base-N) Sequential Scan**:
```bash
GpuCracker.exe --mode akm --akm-mode schematic --akm-bit 68 --akm-word 10 --profile auto-linear

```



---

## 📊 Bloom Filter Setup (`build_bloom`)

Before running a crack, you must convert your target Bitcoin addresses into a `.blf` file using the provided builder.

* **Generate Filter from Text List**:
```bash
build_bloom.exe --input addresses.txt --out database.blf --p 0.0000001

```


* **Merge Multiple Lists**:
```bash
build_bloom.exe --input legacy.txt --input bech32.txt --out combined.blf

```



---

## ⚙️ Performance Tuning

| Flag | Description | Recommendation |
| --- | --- | --- |
| `--type` | Backend selection | Use `cuda` for NVIDIA, `opencl` for others. |
| `--blocks` | GPU block allocation | Increase (e.g., 1024) for high-end cards (RTX 30/40). |
| `--threads` | Threads per block | Multiples of 128 (e.g., 256 or 512). |
| `--speed` | **Real Speed Mode** | Shows raw seed generation rate without multipliers. |
| `--device` | ID Selection | `-1` for all, or `0, 1, 2` for specific GPUs. |

---

## 🛡️ Attack Classes (System Capability)

The UI reports system status based on hardware capability:

| Class | Hardware | Speed (Est.) | Status |
| --- | --- | --- | --- |
| **Class A** | Laptop / Multi-core CPU | ~10K h/s | Active (OpenMP) |
| **Class B** | GPU (GTX 1080 / RTX 4090) | ~1M+ h/s | **Active (Experimental)** |
| **Class C** | Specialized ASIC | ~100M h/s | N/A |
| **Class D** | ASIC Supercluster | 1B+ h/s | N/A |

---

## ⚠️ Important Technical Notes

1. **VRAM Monitoring**: CUDA backend displays real-time memory usage.
2. **Bit Rotation**: If multiple bits are specified (`--akm-bit 71,72`), the program alternates ranges every data batch.
3. **Cooling**: High-intensity scanning will keep your GPU at 100% load. Ensure adequate ventilation.

---

