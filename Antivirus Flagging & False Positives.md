Here is the translation and formatting of your text, optimized for a GitHub `README.md` file.

---

# ⚠️ Antivirus Flagging & False Positives

It is very common for files like `akm_seed2priv.exe` and `build_bloom.exe` to be detected as "infected" by antivirus software. This is a phenomenon known as a **False Positive**. These detections occur due to the specific technical nature of the **GpuCracker** project.

Below are the technical reasons why these programs are often flagged as suspicious:

### 1. Use of Cryptographic Libraries (OpenSSL & Secp256k1)

These programs utilize hashing functions (**SHA256**, **RIPEMD160**) and Elliptic Curve Cryptography (**secp256k1**). Many forms of malware—specifically ransomware or data stealers—use these exact same libraries to encrypt victim data or generate unauthorized wallet addresses. Antivirus engines recognize these cryptographic "signatures" and proactively block the file.

### 2. Absence of Digital Signatures

The executables compiled via Visual Studio do not have a **Digital Signature** (certificates issued by authorities like Microsoft or DigiCert). Windows Defender and other security suites automatically treat any unsigned `.exe` file that manipulates sensitive data (such as private keys) as a potential risk.

### 3. "HackTool" or "Miner" Behavior

* **`build_bloom.exe`**: This tool scans massive text files to generate binary databases. This heavy data-processing behavior mimics the patterns of "scrapers" or data-harvesting tools.
* **`akm_seed2priv.exe`**: Generating private keys from mnemonic phrases (brute-forcing) is behavior identical to password-cracking tools or cryptocurrency miners, which are often categorized as **"RiskWare"** or **"HackTool"**.

### 4. Direct Hardware Access (GPU)

The main application and its modules interact directly with GPU drivers (**CUDA/OpenCL**) for massive parallel computing. Programs that utilize 100% of GPU resources without being a video game are frequently misidentified as "crypto-jacking" malware (malicious miners installed without user consent).

---

## How to Verify the Safety of the Files

* **Source Code Transparency**: Because the source code is provided (`build_bloom.cpp`, `seed2priv_main.cpp`), you can inspect the logic yourself. You will find no functions that exfiltrate data to the internet (except for the optional API checker in `utils.h`, which is under your control).
* **VirusTotal**: If you upload the executable to [VirusTotal](https://www.virustotal.com/), you may see detections such as *Unsafe*, *Suspicious*, *HackTool*, or *RiskWare*. These labels confirm the program is flagged because of its **utility** (recovery/cracking), not because it contains a malicious payload or virus.
* **Adding Exclusions**: To run the project without interruption, you should add the `build` folder (where the `.exe` files are generated) to the **Exclusions** list in Windows Defender or your specific antivirus software.

**Conclusion:** These flags are not caused by a real virus, but by a conflict between automated security algorithms and the advanced math/cryptography required to solve Bitcoin puzzles.

---
