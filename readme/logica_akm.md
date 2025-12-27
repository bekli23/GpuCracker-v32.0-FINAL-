
# Documentație Tehnică: Logica Profilelor AKM

Acest document descrie arhitectura și logica de funcționare a modului "AKM" (Advanced Key Mapping) din cadrul proiectului **GpuCracker**. Această logică este diferită de standardele clasice (BIP39) și este implementată specific în fișierele `akm.h`, `akm_extra.h` și `runner.h`.

## 1. Concept General

Un **Profil AKM** este un set de reguli deterministe care transformă o frază mnemonică (o listă de cuvinte) într-o cheie privată Bitcoin (256 biți). 

Spre deosebire de BIP39 (care folosește PBKDF2 cu 2048 runde), AKM utilizează o metodă de **concatenare hexazecimală** a token-urilor derivate din cuvinte. Aceasta permite o generare extrem de rapidă și personalizabilă, utilă pentru puzzle-uri criptografice specifice.

---

## 2. Procesul de Conversie (Phrase -> Private Key)

Funcția principală care guvernează această logică este `phrase_to_key` definită în `akm.h`. Procesul urmează acești pași:

### Pasul 1: Tokenizarea Cuvintelor
Fiecare cuvânt din frază este convertit într-un șir hexazecimal (token) pe baza regulilor profilului activ (`customHex` sau `specialRules`):

1.  **Regulă Fixă (Hardcoded):** Dacă un cuvânt este definit cu o valoare exactă de 8 caractere hex (ex: `soarele` -> `cafebabe`), se folosește acea valoare.
2.  **Regulă de Prefix:** Dacă un cuvânt are definit un prefix scurt (ex: `abis` -> `0`), restul caracterelor până la lungimea de 8 sunt generate făcând hash SHA256 al cuvântului.
3.  **Fallback (Hashing):** Dacă cuvântul nu are nicio regulă definită, se calculează SHA256-ul cuvântului și se utilizează primele 8 caractere hexazecimale.

### Pasul 2: Concatenarea (Building the BigHex)
Token-urile hexazecimale rezultate sunt concatenate într-un singur șir lung (`bigHex`).
* **Exemplu:** Dacă `cuvant_A` -> `1a` și `cuvant_B` -> `2b`, rezultatul intermediar este `1a2b`.
* Procesul se oprește când șirul atinge **64 de caractere hexazecimale** (32 bytes / 256 biți), care este lungimea unei chei private Bitcoin.

### Pasul 3: Conversia în Biți
Șirul `bigHex` este convertit direct într-un vector de bytes (`std::vector<uint8_t> key`). Aceasta reprezintă cheia privată brută înainte de aplicarea măștilor.

### Pasul 4: Aplicarea Măștilor (Range - Opțional)
Dacă utilizatorul specifică flag-ul `--akm-bit <N>` (ex: 71), se aplică o mască pe biți:
* Biții superiori sunt setați pe 0.
* Bitul specificat (N) este forțat pe 1.
* Aceasta limitează spațiul de căutare la intervalul matematic $2^{N-1} \dots 2^N$.

---

## 3. Tipuri de Profile Disponibile

Sistemul suportă trei tipuri de profile, gestionate în metoda `init` din `akm.h`:

### A. Profile Interne (Hardcoded)
Definite în `akm_extra.h` pentru performanță maximă.
* **`akm3-puzzle71`:** Profilul implicit. Conține sute de reguli specifice pentru puzzle-ul 71 (ex: `abis=100`, `acelasi=101`).
* **`akm2-lab-v1`:** Profil experimental cu reguli de test (ex: `jocul=dead`).

### B. Profilul `auto-linear`
* Acest mod nu folosește hashing.
* Mapează automat indexul cuvântului în lista încărcată la valoarea sa hexazecimală.
* **Exemplu:** Cuvântul de pe poziția 0 -> `0`, poziția 10 -> `a`, poziția 256 -> `100`.

### C. Profile Externe (Fișiere .txt)
Utilizatorul poate încărca un fișier text (ex: `my_custom.txt`) cu sintaxa:

word=HEX_VALUE
# Exemplu:
soarele=cafebabe  (Fix)
abis=0            (Prefix)



Sistemul parsează acest fișier și populează harta `customHex` dinamic.

---

## 4. Diferențe de Operare: Random vs. Schematic

Codul din `seed2priv_main.cpp` și `runner.h` distinge două moduri fundamentale de generare:

### Modul Hash (Random / Default AKM)

* Folosește logica de concatenare descrisă la Secțiunea 2.
* Fiecare cuvânt contribuie independent la biții finali ai cheii.
* Utilizat pentru brainwallets complexe.

### Modul Schematic (Base-N)

* Tratează fraza ca un număr matematic mare într-o bază egală cu numărul de cuvinte din dicționar.
* **Formula:** .
* Acesta este implementat în funcția `DecodeSchematic` din `seed2priv_main.cpp`.

---

## 5. Arhitectura Hibridă (CPU vs. GPU)

### GPU (Viteză Brută)

* Fișierul: `GpuCore.cu`.
* **Logica:** Kernelul CUDA **NU** cunoaște cuvinte sau profile AKM.
* Primește un `startSeed` (un număr pe 64 de biți) și generează chei private aleatoriu (folosind `curand`) sau secvențial.
* Aplică masca de biți (`targetBits`) direct pe numărul generat.
* Verifică hash-ul rezultat în Bloom Filter.

### CPU (Inteligenta & Reconstrucție)

* Fișierul: `runner.h`.
* **Logica:** Când GPU-ul găsește un "Seed" câștigător, îl trimite înapoi la CPU.
* CPU-ul folosește acel seed pentru a reconstrui secvența aleatoare.
* Abia aici, CPU-ul mapează numerele aleatoare înapoi la cuvinte folosind dicționarul AKM, pentru a afișa fraza utilizatorului.




