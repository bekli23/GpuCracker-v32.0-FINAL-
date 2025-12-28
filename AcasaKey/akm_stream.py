#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_stream.py — algoritm “word-stream” pentru chei 256-bit din fraze de cuvinte

Ce face:
  - Normalizează cuvintele (ASCII, a-z).
  - Fiecare cuvânt -> flux determinist de bytes (HKDF-SHA256).
  - Fraza (L cuvinte) -> cheie 32 bytes prin intercalare round-robin, 1 byte/cuvânt/rotă.
  - Regula specială: dacă L=5 și toate cuvintele == "mare" => cheia e 0x...01 (nu 0).
  - Opțional: verifică faptul că toate cuvintele se află într-un wordlist (512).

Nu generează cuvinte, doar transformă fraza în cheie. Gândit să fie importabil.
Compatibil Python 3.8+.
"""

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import unicodedata
from typing import Iterable, Iterator, List, Optional, Set

AKM_VERSION = "AKM1"
DEFAULT_LIST_SIZE = 512   # doar pentru separare de domeniu în salt
KEY_LEN_BYTES = 32        # 256 biți

# ---------------- Normalizare ----------------

def normalize_word(w: str) -> str:
    """NFKD, lowercase, elimină diacritice și orice în afara [a-z]."""
    w = unicodedata.normalize("NFKD", w).lower()
    w = "".join(c for c in w if not unicodedata.combining(c))
    w = re.sub(r"[^a-z]", "", w)
    return w

def normalize_phrase(words: Iterable[str]) -> List[str]:
    out = [normalize_word(w) for w in words]
    if any(not w for w in out):
        raise ValueError("Cuvinte invalide după normalizare (a rezultat string gol).")
    return out

# ---------------- Wordlist (opțional) ----------------

def load_wordlist(path: str) -> List[str]:
    """Încărcă wordlist-ul și normalizează fiecare intrare (păstrează ordinea și duplicatele din fișier)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nu găsesc wordlist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    wl = [normalize_word(w) for w in raw]
    return wl

def enforce_words_in_list(words: Iterable[str], wordlist: Iterable[str]) -> None:
    """Aruncă eroare dacă vreun cuvânt nu e în wordlist (după normalizare)."""
    wl_set: Set[str] = set(wordlist)
    bad = [w for w in normalize_phrase(words) if w not in wl_set]
    if bad:
        raise ValueError(f"Cuvinte în afara wordlist-ului: {bad[:5]}{'...' if len(bad)>5 else ''}")

# ---------------- HKDF de bază ----------------

def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    if not salt:
        salt = bytes([0]*hashlib.sha256().digest_size)
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    hlen = hashlib.sha256().digest_size
    n = (length + hlen - 1) // hlen
    okm, t = b"", b""
    for i in range(1, n+1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]

# ---------------- Flux determinist per cuvânt ----------------

def word_stream(word: str,
                list_size: int = DEFAULT_LIST_SIZE,
                lang: str = "ro",
                stream_label: str = "stream") -> Iterator[int]:
    """
    Generator infinit de bytes pentru un cuvânt.
    Produce blocuri HKDF de 32B și le livrează byte cu byte.
    """
    w = normalize_word(word)
    salt = f"{AKM_VERSION}|lang:{lang}|list:{list_size}|word:{w}".encode("utf-8")
    ikm = w.encode("utf-8")
    prk = _hkdf_extract(salt, ikm)

    counter = 1
    buf = b""
    while True:
        if not buf:
            info = f"{AKM_VERSION}-{stream_label}|ctr:{counter}".encode("utf-8")
            buf = _hkdf_expand(prk, info, 32)
            counter += 1
        b = buf[0]
        buf = buf[1:]
        yield b  # int 0..255

# ---------------- Fraza -> Cheie 256-bit ----------------

def phrase_to_key(words: Iterable[str],
                  list_size: int = DEFAULT_LIST_SIZE,
                  lang: str = "ro",
                  override_special: bool = True,
                  require_in_wordlist: Optional[List[str]] = None) -> bytes:
    """
    Construcție cheie:
      - normalizează cuvintele
      - opțional verifică apartenența în wordlist
      - intercalează round-robin câte 1 byte din fluxul fiecărui cuvânt până la 32 bytes
      - regulă specială: L=5 și toate cu 'mare' => 0x...01
    """
    ww = normalize_phrase(words)
    if require_in_wordlist is not None:
        enforce_words_in_list(ww, require_in_wordlist)

    L = len(ww)
    if L == 0:
        raise ValueError("Fraza este goală.")

    # Regula specială cerută
    if override_special and L == 5 and all(w == "mare" for w in ww):
        return (b"\x00" * (KEY_LEN_BYTES - 1)) + b"\x01"

    gens: List[Iterator[int]] = [word_stream(w, list_size=list_size, lang=lang) for w in ww]

    out = bytearray()
    gi = 0
    while len(out) < KEY_LEN_BYTES:
        out.append(next(gens[gi]))
        gi += 1
        if gi == L:
            gi = 0
    return bytes(out)

# ---------------- Checksum opțional ----------------

def phrase_checksum_index(key32: bytes, list_size: int = DEFAULT_LIST_SIZE) -> int:
    """Index checksum derivat din cheie: HMAC(key, "AKM1-tag")[0] mod list_size."""
    tag = hmac.new(key32, f"{AKM_VERSION}-tag".encode("utf-8"), hashlib.sha256).digest()
    return tag[0] % list_size

# ---------------- Secțiune CLI minimală (opțională) ----------------

def _build_cli():
    p = argparse.ArgumentParser(description="AKM Stream (fraze -> cheie 256-bit, fără generare aleatoare)")
    p.add_argument("--phrase", "-p", help='Fraza între ghilimele: ex. "mare umbra culoare ulei pat"')
    p.add_argument("--phrase-file", help="Fișier text cu fraza (o linie).")
    p.add_argument("--wordlist", help="Fișier wordlist (opțional, pentru validare).")
    p.add_argument("--lang", default="ro", help="Etichetă limbă pentru salt (implicit ro).")
    p.add_argument("--list-size", type=int, default=DEFAULT_LIST_SIZE, help="Dimensiune listă (pentru salt).")
    p.add_argument("--json", action="store_true", help="Afișează output JSON.")
    return p

def _cli_main():
    ap = _build_cli()
    args = ap.parse_args()

    if not args.phrase and not args.phrase_file:
        ap.print_help()
        sys.exit(1)

    if args.phrase:
        words = re.split(r"\s+", args.phrase.strip())
    else:
        with open(args.phrase_file, "r", encoding="utf-8") as f:
            words = re.split(r"\s+", f.read().strip())

    wl = load_wordlist(args.wordlist) if args.wordlist else None
    key = phrase_to_key(words, list_size=args.list_size, lang=args.lang, require_in_wordlist=wl)
    cs = phrase_checksum_index(key, list_size=args.list_size)

    if args.json:
        print(json.dumps({
            "version": AKM_VERSION,
            "words_norm": normalize_phrase(words),
            "key_hex": key.hex(),
            "checksum_index": cs
        }, ensure_ascii=False, indent=2))
    else:
        print("words_norm:", " ".join(normalize_phrase(words)))
        print("key_hex   :", key.hex())
        print("checksum  :", cs)

if __name__ == "__main__":
    _cli_main()
