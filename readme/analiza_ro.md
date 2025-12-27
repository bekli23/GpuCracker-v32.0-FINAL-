Aceasta este o analiză completă a proiectului GpuCracker / AKM Seed2Priv bazată pe toate fișierele încărcate.

Proiectul este un utilitar de înaltă performanță (HPC) destinat recuperării și auditării cheilor Bitcoin, având o arhitectură hibridă CPU-GPU. Este proiectat să funcționeze atât în mod clasic (BIP39 Mnemonic), cât și în moduri specifice de tip "Brainwallet" (AKM), folosind filtre Bloom pentru verificări rapide.

Mai jos este structura detaliată a arhitecturii și fluxul de date:

1. Arhitectura Generală
Proiectul este construit pe modelul Producer-Consumer, unde GPU-ul generează entropie/chei, iar CPU-ul (sau kernel-ul specializat GPU) verifică aceste chei împotriva unei baze de date.

Diagrama Fluxului de Date
Input: Argumente CLI (args.h) -> Configurare (Runner).

Generare (GPU): IGpuProvider (CUDA/OpenCL) -> Generează buffer de entropie (Random/Secvențial).

Procesare (Hibrid):

Calea A (Legacy/Mnemonic): Buffer -> CPU (OpenMP) -> SHA256/PBKDF2 -> Bip32 Derive -> Adresă.

Calea B (AKM High-Perf): GpuCore.cu -> Matematica directă pe GPU (BigInt + EC Multiply) -> Hash160 -> Bloom Check (totul pe placa video).

Verificare: BloomFilter (în memorie RAM) -> Verificare rapidă dacă hash-ul există.

Output: Consolă (Statistici Live) + Fișier Text (win.txt).

2. Analiza Modulelor Principale
A. Nucleul de Execuție (Runner - runner.h)
Acesta este "creierul" aplicației.

Rol: Inițializează hardware-ul, încarcă filtrele Bloom și gestionează firele de execuție (threads).

Worker Class B GPU: O funcție specializată (workerClassBGPU) care apelează direct kernel-ul launch_gpu_akm_search din GpuCore.cu pentru viteză maximă în modul AKM.

Worker Legacy Hybrid: Folosește OpenMP pentru a paralela procesarea pe CPU a datelor primite de la GPU (util pentru BIP39 unde PBKDF2 este greu de făcut eficient pe GPU).

B. Abstracția Hardware (IGpuProvider)
Interfața permite schimbarea dinamică a backend-ului.

CudaProvider: Folosește curand pentru generare rapidă și funcții native NVIDIA. Gestionează memoria pinned pentru transfer rapid.

OpenClProvider: Compilează kernel-ul la runtime (JIT) dintr-un string (kernelSource), asigurând compatibilitatea cu plăci AMD sau Intel ARC. Folosește CL_MEM_ALLOC_HOST_PTR pentru optimizarea transferului de memorie pe sisteme cu memorie partajată.

C. Logica AKM (Brainwallet Avansat)
Proiectul include o logică complexă pentru generarea cheilor din fraze non-standard.

Profile: Suportă profile precum akm3-puzzle71 sau akm2-lab-v1 care definesc reguli de substituție a cuvintelor în hex (ex: "mare" -> "0000").

Range & Masking: Poate limita cheile generate la un anumit interval de biți (ex: targetBits în GpuCore.cu), forțând bitul cel mai semnificativ pentru a rămâne într-un "Puzzle Range" specific.

D. Verificare Rapidă (BloomFilter)
Dublu Hashing: Folosește MurmurHash3 cu două seed-uri diferite (0xFBA4C795 și 0x43876932) pentru a mapa Hash160-ul adresei în bitarray. Această implementare este identică în build_bloom.cpp (generator) și GpuCore.cu (checker), garantând consistența.

3. Puncte Forte și Observații Tehnice
Optimizare Critică:

În GpuCore.cu, funcțiile matematice (add_cc, mad_lo_cc) sunt scrise în PTX Assembly (inline assembly pentru CUDA). Aceasta este cea mai rapidă metodă posibilă de a face aritmetică pe 256 de biți pe hardware NVIDIA.

Nu depinde de librării externe mari în kernel-ul critic; totul este "self-contained".

Flexibilitate:

Poate rula în mod infinit sau cu o limită fixă (--count).

Suportă atât Random Search cât și Secvențial (Odometru), util pentru a scana spații de chei contigue.

Dependențe:

Proiectul depinde puternic de OpenSSL (SHA, RIPEMD, HMAC) și secp256k1 (pentru generarea cheilor publice pe CPU).

Proiectele .vcxproj indică utilizarea vcpkg pentru gestionarea acestor pachete.

4. Instrucțiuni pentru Build (Visual Studio 2022/2026)
Deoarece ai "urcat tot proiectul", iată pașii pentru a compila corect soluția finală:

Prerechizite:

Instalează CUDA Toolkit 12.x.

Instalează vcpkg și rulează comanda: vcpkg install openssl:x64-windows secp256k1:x64-windows opencl:x64-windows.

Configurare Proiect:

Asigură-te că fișierele .vcxproj caută librăriile în folderul corect vcpkg/installed/x64-windows/include și lib.

Pentru GpuCracker.vcxproj, asigură-te că "Build Customizations" are bifat CUDA 12.x.

Ordinea de Compilare:

Compilează întâi build_bloom. Folosește-l pentru a genera un fișier .blf (baza de date) dintr-o listă de adrese text.

Compilează GpuCracker. Acesta este executabilul principal.

Concluzie
Proiectul este complet și pare funcțional din punct de vedere al codului sursă. Are o separare clară între logica de business (Runner), logica matematică (AKM/Utils) și logica de calcul paralel (Provideri GPU). Este un utilitar puternic, capabil să verifice miliarde de adrese pe secundă pe un GPU modern (RTX 30/40 series) folosind motorul "Class B" din GpuCore.cu.