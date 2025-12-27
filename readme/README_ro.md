
---

# GpuCracker v32.0 (FINAL)

**GpuCracker** este o aplicație modulară de înaltă performanță concepută pentru verificarea și recuperarea frazelor mnemonice (BIP39) și a profilelor personalizate de tip AKM, utilizând accelerare hardware masivă prin CUDA, OpenCL și Vulkan.

---

## 🛠️ Cerințe de Sistem și Build

### 1. Software Necesar

* **Sistem de Operare**: Windows 10/11 (x64).
* **IDE**: Visual Studio 2022 cu setul de unelte **v143**.
* **CUDA Toolkit**: Versiunea **12.4** (necesară pentru backend-ul NVIDIA).
* **Vulkan SDK**: Versiunea **1.4.335.0**.
* **Manager Pachete**: **vcpkg** pentru gestionarea dependințelor C++.

### 2. Instalare Dependințe (vcpkg)

Înainte de build, instalează librăriile necesare folosind următoarea comandă în terminal:


vcpkg install openssl:x64-windows secp256k1:x64-windows



*Note: Proiectul caută automat include-urile și librăriile în folderul de instalare standard vcpkg.*

---

## 🚀 Instrucțiuni de Build

1. Deschide fișierul de proiect `.vcxproj` în Visual Studio 2022.
2. Setează configurația pe **Release** și platforma pe **x64**.
3. Asigură-te că **CUDA 12.4 Build Customizations** sunt activate pentru proiect.
4. **Build Solution** (`Ctrl+Shift+B`).
5. **Post-Build**: Executabilul va fi generat în folderul `bin\x64\Release\`. Dicționarele din folderele `bip39/` și `akm/` vor fi copiate automat în folderul de ieșire pentru a asigura rularea imediată.

---

## 💻 Utilizare (CLI Options)

### Moduri de Operare

* `--mode mnemonic`: Verifică fraze BIP39 standard (12-24 cuvinte).
* `--mode akm`: Rulează logica personalizată AKM bazată pe profile (ex: Puzzle 71/72).

### Setări Esențiale

* `--bloom-keys FILE`: Calea către filtrul Bloom (.blf) care conține adresele țintă.
* `--count N`: Se oprește automat după verificarea a **N** semințe (precizie ridicată în modul Class B).
* `--setaddress TYPE`: Filtrează adresele afișate. Opțiuni: `ALL`, `LEGACY`, `P2PKH`, `P2SH`, `SEGWIT`, `TAPROOT`.

### Configurare Hardware

* `--type [cuda|opencl|vulkan|auto]`: Selectează backend-ul hardware.
* `--device N`: ID-ul specific al GPU-ului (Default: -1 pentru auto-detectare NVIDIA/OpenCL).
* `--blocks N`, `--threads N`, `--points N`: Parametri pentru tuning-ul performanței GPU.

---

## 🛡️ Strategii de Atac (Classes of Attack)

Interfața raportează starea sistemului conform ierarhiei de putere de calcul:

| Clasa | Hardware | Viteză Estimată | Status |
| --- | --- | --- | --- |
| **Class A** | Laptop / CPU Multi-core | ~10,000 h/s | Activ (OpenMP) |
| **Class B** | GPU (ex: GTX 1080/RTX 3090) | ~1,000,000 h/s | **Activ (Experimental)** |
| **Class C** | ASIC Specializat | ~100,000,000 h/s | Nespecificat |
| **Class D** | ASIC Supercluster | 1,000,000,000+ h/s | Nespecificat |

---

## ⚠️ Note Importante

1. **Monitorizare VRAM**: Pentru backend-ul CUDA, programul afișează memoria utilizată în timp real (ex: `450/6144 MB`).
2. **Multi-Bit Rotation**: Dacă sunt specificați mai mulți biți (ex: `--akm-bit 71,72`), programul va alterna rangurile de căutare la fiecare pachet de date.
3. **Filtrare Hardware**: Modul `auto` previne duplicarea acelorași plăci video NVIDIA între interfețele CUDA și OpenCL.

---