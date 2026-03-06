# GpuCracker v50.0 🚀

> High-Performance Cryptocurrency Recovery & Security Analysis Tool

[![Version](https://img.shields.io/badge/version-50.0-blue.svg)](https://github.com/bekli23/GpuCracker/releases)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg)](#)

<p align="center">
  <img src="docs/images/gpucracker_logo.png" alt="GpuCracker Logo" width="400">
</p>

## ✨ Features

### 🔐 10 Operating Modes

| Mode | Description | GPU Support |
|------|-------------|-------------|
| **MNEMONIC** | BIP39 seed phrase recovery | ✅ CUDA/OpenCL |
| **AKM** | Advanced keyspace mapping | ✅ CUDA/OpenCL |
| **BSGS** | Baby-Step Giant-Step ECDLP | ✅ CUDA/OpenCL |
| **RHO** | Pollard's Rho algorithm | ✅ CUDA/OpenCL |
| **HYBRID** | Auto-select optimal algorithm | ✅ CUDA/OpenCL |
| **CHECK** | Key/address verification | ✅ CUDA/OpenCL |
| **SCAN** | Pattern/vanity scanning | ✅ CUDA/OpenCL |
| **XPRV** | Extended key operations | ✅ CUDA/OpenCL |
| **BRAINWALLET** | Password-based wallet recovery | ✅ CUDA/OpenCL |
| **PUBKEY** | Public key operations | ⚡ CPU/GPU |

### 🚀 Performance Highlights

- **10-100x** speedup with GPU acceleration
- **Billions** of cryptographic operations per second
- **100+** cryptocurrency support
- **Bloom filter** integration for O(1) address lookup

## 📊 Performance Benchmarks

### GPU Acceleration Speedup

| Algorithm | CPU (32 cores) | RTX 4090 | Speedup |
|-----------|---------------|----------|---------|
| SHA256 | 100 MH/s | 10 GH/s | 100x |
| PBKDF2 | 50 KH/s | 5 MH/s | 100x |
| BSGS Giant Steps | 1 M/s | 100 M/s | 100x |
| Pollard's Rho | 500 K/s | 50 M/s | 100x |

### Real-World Performance

| Scenario | Hardware | Time |
|----------|----------|------|
| BSGS 2^50 range | 4× RTX 4090 | ~3 hours |
| Mnemonic 12-word | RTX 4090 | 1B phrases/sec |
| Address scan (1B) | RTX 4090 | ~10 seconds |

## 🛠️ Quick Start

### Linux

```bash
# Install dependencies
sudo apt update
sudo apt install build-essential libssl-dev libsecp256k1-dev ocl-icd-opencl-dev

# Build with CUDA
make CUDA=1 OPENCL=1 -j$(nproc)

# Run benchmark
./GpuCracker --benchmark
```

### Windows

```powershell
# Install Visual Studio 2022, CUDA 12.4, vcpkg
vcpkg install openssl:x64-windows boost-multiprecision:x64-windows

# Open GpuCracker.sln in Visual Studio
# Build: Ctrl+Shift+B

# Run
.\GpuCracker.exe --benchmark
```

## 📖 Usage Examples

### BSGS ECDLP Solving

```bash
# Solve for private key in 2^60 range
./GpuCracker --mode bsgs \
  --bsgs-pub 0279BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798 \
  --bsgs-range 60 \
  --bsgs-gpu
```

### Mnemonic Recovery

```bash
# Brute force 12-word mnemonic
./GpuCracker --mode mnemonic \
  --words 12 \
  --bloom-keys btc_addresses.blf \
  --gpu-type cuda
```

### Multi-Target Search

```bash
# Search for 1000 public keys simultaneously
./GpuCracker --mode bsgs \
  --bsgs-targets pubkeys.txt \
  --bsgs-range 50 \
  --bsgs-gpu
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GpuCracker v50.0                         │
├─────────────────────────────────────────────────────────────┤
│  Modes: MNEMONIC │ AKM │ BSGS │ RHO │ HYBRID │ CHECK ...  │
├─────────────────────────────────────────────────────────────┤
│  GPU Layer: CUDA (NVIDIA) │ OpenCL (AMD/Intel) │ Vulkan    │
├─────────────────────────────────────────────────────────────┤
│  Crypto: secp256k1 │ SHA256 │ PBKDF2 │ Bloom Filters       │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Use Cases

### Legitimate Security Research
- Wallet implementation auditing
- Cryptographic algorithm analysis
- Penetration testing (authorized)

### Recovery Scenarios
- Lost mnemonic phrase recovery
- Corrupted wallet file repair
- Forgotten password recovery

### Forensic Analysis
- Blockchain transaction tracing
- Address clustering analysis
- Pattern recognition in key generation

## 📦 Supported Cryptocurrencies

### Major Coins
- Bitcoin (BTC)
- Ethereum (ETH)
- Litecoin (LTC)
- Bitcoin Cash (BCH)

### Full List (100+)
See [SUPPORTED_COINS.md](docs/SUPPORTED_COINS.md) for complete list.

## 🔧 System Requirements

### Minimum
- CPU: x86_64, 4 cores
- RAM: 8 GB
- GPU: Optional
- OS: Windows 10/11 or Linux

### Recommended
- CPU: x86_64, 16+ cores
- RAM: 64+ GB
- GPU: NVIDIA RTX 4090 or equivalent
- OS: Ubuntu 22.04 LTS

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [README_GENERAL.md](readme/README_GENERAL.md) | General documentation |
| [README_BUILD_LINUX.md](readme/README_BUILD_LINUX.md) | Linux build guide |
| [README_BUILD_WIN.md](readme/README_BUILD_WIN.md) | Windows build guide |
| [README_MODE_*.md](readme/) | Mode documentation (10 files) |
| [README_COMMAND_*.md](readme/) | Command reference (10 files) |

Total: **500+ documented commands**

## 🛡️ Security & Ethics

### Responsible Disclosure
This tool is designed for:
- ✅ Legitimate wallet recovery
- ✅ Authorized security testing
- ✅ Academic research
- ❌ Unauthorized access to others' funds

### Best Practices
- Always verify recovered keys in isolated environment
- Use air-gapped systems for high-value operations
- Follow responsible disclosure for vulnerabilities

## 🤝 Contributing

We welcome contributions!

```bash
# Fork and clone
git clone https://github.com/bekli23/GpuCracker.git

# Create branch
git checkout -b feature/amazing-feature

# Commit and push
git commit -m "Add amazing feature"
git push origin feature/amazing-feature

# Open Pull Request
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📈 Version History

### v50.0 (Current)
- Complete BSGS/Rho/Hybrid ECDLP implementation
- GPU acceleration for giant steps
- Multi-target batch search
- Bloom filter optimization
- Distributed cluster support

### v42.2 (Previous Stable)
- Basic BSGS implementation
- CPU-only operation
- Limited coin support

See [RELEASES.md](RELEASES.md) for full history.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

```
MIT License

Copyright (c) 2024-2026 GpuCracker Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## 🙏 Acknowledgments

- [libsecp256k1](https://github.com/bitcoin-core/secp256k1) - Bitcoin Core library
- [Boost](https://www.boost.org/) - C++ libraries
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone) - GPU computing
- [Khronos OpenCL](https://www.khronos.org/opencl/) - Heterogeneous computing

## 📞 Support

- 💬 [GitHub Discussions](https://github.com/bekli23/GpuCracker/discussions)
- 🐛 [Issue Tracker](https://github.com/bekli23/GpuCracker/issues)
- 📧 Email: support@gpucracker.dev

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=bekli23/GpuCracker&type=Date)](https://star-history.com/#bekli23/GpuCracker&Date)

---

<p align="center">
  Made with ❤️ by the GpuCracker Team
</p>

<p align="center">
  <a href="https://github.com/bekli23/GpuCracker">GitHub</a> •
  <a href="https://gpucracker.readthedocs.io">Documentation</a> •
  <a href="https://twitter.com/gpucracker">Twitter</a>
</p>
