# 🔐 AKM (Advanced Key Mapper) v2

Deterministic **phrase → key → address** suite for experimental cryptographic mapping, mnemonic research, and deterministic key generation.

The **AKM suite** converts human-readable phrases into 256-bit private keys, complete with checksum validation, wordlist integrity checks, and multi-format Bitcoin address derivation.

> ⚠️ Not a BIP-39 implementation — AKM uses its own logic, word streams, and hex-packing algorithms.

---

## 🧩 Module Overview

| File | Purpose |
|------|----------|
| **`akm_stream.py`** | Core HKDF-based word stream engine — generates deterministic byte streams per word. |
| **`akm_words_512.py`** | Main library: 512-word list, fixed 8-hex token system, checksum (V1 + V2), special rules, anti-weak-key guard, and spec exporter. |
| **`akm_seed2priv.py`** | CLI tool: converts a phrase into a 256-bit private key + all Bitcoin address formats (P2PKH, P2SH, Bech32, Taproot). |
| **`akm_demo_gen.py`** | Demo phrase generator — tests multi-group lengths (3 / 5 / 10 / 15 / 20 / 25 / 30). |
| **`akm_examples.py`** | Contains canonical test vectors and ready-made example phrases. |
| **`akm_spec.json`** | Auto-exported spec file — includes parameter hashes, checksum variants, and test vectors for reproducibility. |
| **`wordlist_512_ascii.txt`** | Canonical 512-word list (Romanian-derived, ASCII normalized). |

---

## ⚙️ Key Concepts

### 1. Deterministic Word Streams

Each normalized word produces its own infinite HKDF-SHA256 stream:

```text
salt = "AKM2|wl:512|idx:{index}|word:{word}"
info = "AKM2-stream|ctr:{n}"
Each byte is deterministic, reproducible, and unique per word index.

2. Phrase → 256-bit Private Key
phrase_to_key(words) iteratively takes 8-hex tokens per word until 64 hex chars (32 bytes) are filled.

Example override:

css
 
["mare", "mare", "mare", "mare", "mare"]
→ special deterministic key (not all-zero)
Every word must exist in the 512-word list.

3. Fixed Tokenization (token8)
Each word maps to an 8-hex-digit token, derived as:

Base = SHA256(word)[:4]

If CUSTOM_HEX override → use it

Else if SPECIAL_RULES match → apply fixed8 / pad_nibble / repeat_last

Else → extend deterministically via HKDF

Use:

python
 
from akm_words_512 import explain_token
explain_token("apele")
to trace its logic (base + decision + token8).

4. Checksums
AKM supports two checksum versions:

Version	Words	Bits / Word	Seed Composition
V1	1	9 bits	`key
V2	2	10 bits	`key

Each checksum word is drawn from the same 512-word set.

5. Weak-Key Protection
Built-in check: keys such as
00...00, FF...FF, or repeating 4-nibble patterns are blocked.
Use --allow-weak only for explicit testing.

6. Version & Policy
Allowed total lengths: 3, 5, 10, 15, 20, 25, 30 (checksum included).

Each module carries its own version descriptor:

json
 
{
  "version": "AKM2",
  "wl": 512,
  "token": "hexpack8",
  "cs": { "v1": true, "v2": true, "v2_words": 2, "v2_bits_per_word": 10 }
}
Any parameter change = new version bump.

🖥️ CLI Usage
✅ Validate a phrase & generate addresses
bash
 
python akm_seed2priv.py --phrase "livada ora ora lumina livada"
Example output:

 
[✓] Checksum valid (livada)

=============================================================
PHRASE : livada ora ora lumina livada
CORE   : livada ora ora lumina
WORDS  : 5
PRIVATE HEX : <64-char hex>
PRIVATE INT : <decimal>
WIF (mainnet, compressed)  : KwDiD...
P2PKH           : 1A...
P2SH(P2WPKH)    : 3B...
P2WPKH (bech32) : bc1q...
P2TR (taproot)  : bc1p...
=============================================================
🔍 Calculate the expected checksum (no derivation)
bash
 
python akm_seed2priv.py --calc-cs --phrase "livada ora ora lumina X"
Output:

csharp
 
[i] CORE: livada ora ora lumina
[i] Expected CS: livada
CLI Options
Flag	Description
--phrase "..."	Phrase with checksum (final word).
--calc-cs	Compute expected checksum for given core phrase.
--in <file>	Load multiple phrases from a text file.
--stdin	Read phrases from standard input.
--no-wif	Suppress WIF output.

🧠 Library Highlights
python
 
from akm_words_512 import (
    AKM_PARAMS, WORDS, phrase_to_key,
    explain_token, checksum_words_v1,
    checksum_words_v2, export_spec_json, is_weak_key
)
Example
python
 
from akm_words_512 import phrase_to_key, checksum_words_v1, explain_token

phrase = ["apele", "apele", "apele", "apele", "apele"]
key = phrase_to_key(phrase)
print("Private key:", key.hex())
print("Checksum V1:", checksum_words_v1(phrase))
print("Token details:", explain_token("apele"))
🧾 Spec & Integrity
Each build exports a akm_spec.json snapshot with:

AKM_PARAMS

Hashes of WORDS / CUSTOM_HEX / SPECIAL_RULES

10–20 canonical test vectors (key + checksum)

If hashes differ between machines, your build isn’t bit-identical.

🧰 Developer Notes
Pure Python 3.8+ (no dependencies).

Deterministic HKDF-SHA256 — no RNG used.

Consistent outputs across all OS/platforms.

Minimal internal secp256k1 implementation.

Run export_spec_json() for cross-system consistency checks.

⚠️ Disclaimer
This project is for educational, artistic, and experimental purposes only.
Do not use AKM-derived keys or addresses to store real cryptocurrency funds.
No guarantee of entropy, cryptographic strength, or security audit.

🔎 Example: Token Explanation
bash
 
>>> from akm_words_512 import explain_token
>>> explain_token("apele")
{
  "word": "apele",
  "base4": "8f0a",
  "decision": "SPECIAL.fixed8",
  "token8": "00001111"
}
📖 License
MIT License — © 2025 Stefan Florin
Use, modify, and redistribute freely with attribution.

 
