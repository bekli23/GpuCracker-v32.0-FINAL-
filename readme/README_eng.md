
---

# GpuCracker v32.0 (FINAL)

**GpuCracker** is a modular, high-performance tool designed for verifying and recovering BIP39 mnemonic phrases and custom AKM profiles. It leverages massive hardware acceleration via **CUDA**, **OpenCL**, and **Vulkan**.

---

## 🛠️ System & Build Requirements

### 1. Required Software

* **Operating System**: Windows 10/11 (x64).
* **IDE**: Visual Studio 2022 with **v143** build tools.
* **CUDA Toolkit**: Version **12.4** (required for NVIDIA backend).
* **Vulkan SDK**: Version **1.4.335.0**.
* **Package Manager**: **vcpkg** (for C++ dependency management).

### 2. Install Dependencies (vcpkg)

Before building, install the necessary libraries via terminal:


vcpkg install openssl:x64-windows secp256k1:x64-windows



*Note: The project automatically links these libraries from the standard vcpkg installation path.*

---

## 🚀 Build Instructions

1. Open the `.vcxproj` project file in **Visual Studio 2022**.
2. Set the build configuration to **Release** and platform to **x64**.
3. Ensure **CUDA 12.4 Build Customizations** are enabled for the project.
4. Run **Build Solution** (`Ctrl+Shift+B`).
5. **Post-Build**: The executable `GpuCracker.exe` is generated in `bin\x64\Release\`. Dictionaries from `bip39/` and `akm/` folders are copied automatically to the output directory.

---

## 💻 Usage (CLI Options)

### Operating Modes

* `--mode mnemonic`: Default mode for standard BIP39 phrases (12-24 words).
* `--mode akm`: Mode for custom AKM profiles (e.g., Puzzle 71/72).

### Essential Settings

* `--bloom-keys FILE`: Path to the Bloom Filter (.blf) containing target addresses.
* `--count N`: Automatically stop after checking **N** seeds (Precise stopping for Class B logic).
* `--setaddress TYPE`: Filter visible address types. Options: `ALL`, `LEGACY`, `P2PKH`, `P2SH`, `SEGWIT`, `TAPROOT`.

### Hardware Configuration

* `--type [cuda|opencl|vulkan|auto]`: Select the hardware backend.
* `--device N`: Target specific GPU ID (Default: -1 for auto-detect).
* `--blocks N`, `--threads N`, `--points N`: Performance tuning parameters.

---

## 🛡️ Attack Strategy (Classes of Attack)

The UI reports system status based on the following calculation power hierarchy:

| Class | Hardware | Estimated Speed | Status |
| --- | --- | --- | --- |
| **Class A** | Laptop / Multi-core CPU | ~10,000 h/s | Active (OpenMP) |
| **Class B** | GPU (e.g., GTX 1080/RTX 4090) | ~1,000,000 h/s | **Active (Experimental)** |
| **Class C** | Specialized ASIC | ~100,000,000 h/s | N/A |
| **Class D** | ASIC Supercluster | 1,000,000,000+ h/s | N/A |

---

## ⚠️ Important Technical Notes

1. **VRAM Monitoring**: CUDA backend displays real-time memory usage (e.g., `450/6144 MB`).
2. **Multi-Bit Rotation**: If multiple bits are specified (e.g., `--akm-bit 71,72`), the program rotates ranges every data batch to search both ranges in parallel.
3. **Hardware Filtering**: The `auto` type prevents duplicate device registration (e.g., preventing an NVIDIA card from appearing as both CUDA and OpenCL).
4. **GPU Load Optimization**: Visual derivations are limited to once per second to ensure the GPU maintains 100% load without CPU bottlenecks.

---
