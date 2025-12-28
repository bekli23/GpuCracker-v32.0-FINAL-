#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_genaddr.py — Generate BTC addresses (u/c) from AKM phrases

Modes:
  - random
  - name
  - alpha            : same word repeated length times
  - alpha-last       : first (length-1) words are the same base word; last walks sorted WORDS
  - schematic        : mini-DSL to shape phrases

DSL blocks (space-separated):
  {w:WORD}                 -> fixed word
  {rep:WORD:N}             -> repeat WORD N times
  {list:W1|W2|...}         -> iterate a choice
  {range:START:STEP:COUNT} -> iterate sorted(WORDS)[START + k*STEP], COUNT items

All multi-valued blocks combine via cartesian product (bounded by --max in finite mode).
Core lengths must be one of {3,5,10,15,20,25,30}.

Extras:
  - --cs none|v1|v2
      * Pentru lungimi CANONICE (3/5/10/15/20/25/30), NU se adaugă cuvinte de checksum.
        Cheia se derivă exact din fraza generată.
  - --infinite  -> stream fraze + adrese până la Ctrl+C (fără CSV)
  - Realtime speed: scrie "[STATS] X addr/sec (total: N)" pe stderr, o singură linie dinamică.

Requires: akm_words_512.py
"""

from __future__ import annotations
import argparse
import sys
import os
import csv
import hashlib
import random
import time
from typing import List, Tuple, Iterable

from akm_words_512 import (
    WORDS,
    phrase_to_key,
    append_checksum_v1,
    append_checksum_v2,
    assert_in_list,
    is_allowed_core_length,
)

# ============================================================
# secp256k1 minimal
# ============================================================

P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424


def inv_mod(a: int, m: int) -> int:
    return pow(a, -1, m)


def point_add(Pt: Tuple[int, int] | None,
              Qt: Tuple[int, int] | None) -> Tuple[int, int] | None:
    if Pt is None:
        return Qt
    if Qt is None:
        return Pt
    x1, y1 = Pt
    x2, y2 = Qt
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if x1 == x2 and y1 == y2:
        s = (3 * x1 * x1) * inv_mod(2 * y1, P) % P
    else:
        s = (y2 - y1) * inv_mod((x2 - x1) % P, P) % P
    x3 = (s * s - x1 - x2) % P
    y3 = (s * (x1 - x3) - y1) % P
    return (x3, y3)


def scalar_mult(k: int,
                Pt: Tuple[int, int] = (Gx, Gy)) -> Tuple[int, int] | None:
    k %= N
    if k == 0:
        return None
    R = None
    Q = Pt
    while k:
        if k & 1:
            R = point_add(R, Q)
        Q = point_add(Q, Q)
        k >>= 1
    return R


# ============================================================
# hashing / encoding
# ============================================================

B58_ALPH = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def ripemd160(b: bytes) -> bytes:
    h = hashlib.new('ripemd160')
    h.update(b)
    return h.digest()


def b58encode(b: bytes) -> str:
    n = int.from_bytes(b, 'big')
    out = bytearray()
    while n > 0:
        n, r = divmod(n, 58)
        out.append(B58_ALPH[r])
    pad = 0
    for c in b:
        if c == 0:
            pad += 1
        else:
            break
    out.extend(b'1' * pad)
    out.reverse()
    return out.decode('ascii')


def b58check_encode(payload: bytes) -> str:
    chk = sha256(sha256(payload))[:4]
    return b58encode(payload + chk)


def convertbits(data: bytes,
                frombits: int,
                tobits: int,
                pad: bool = True) -> list[int]:
    acc = 0
    bits = 0
    ret: list[int] = []
    maxv = (1 << tobits) - 1
    for b in data:
        acc = (acc << frombits) | b
        bits += frombits
        while bits >= tobits:
            bits -= tobits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (tobits - bits)) & maxv)
    else:
        if bits >= frombits or ((acc << (tobits - bits)) & maxv):
            raise ValueError("convertbits invalid")
    return ret


def _bech32_polymod(values):
    GEN = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = (chk >> 25)
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GEN[i]
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_create_checksum(hrp: str,
                            data: list[int],
                            spec: str = 'bech32'):
    const = 1 if spec == 'bech32' else 0x2bc830a3
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0] * 6) ^ const
    return [(polymod >> (5 * (5 - i))) & 31 for i in range(6)]


def bech32_encode(hrp: str,
                  data: list[int],
                  spec: str = 'bech32') -> str:
    checksum = _bech32_create_checksum(hrp, data, spec=spec)
    combined = data + checksum
    return hrp + '1' + ''.join([BECH32_CHARSET[d] for d in combined])


# ============================================================
# BTC derivations
# ============================================================

def pubkey_from_priv(secret: int,
                     compressed: bool = True) -> bytes:
    Ppt = scalar_mult(secret)
    if Ppt is None:
        raise ValueError("Invalid secret -> point at infinity")
    x, y = Ppt
    xb = x.to_bytes(32, 'big')
    yb = y.to_bytes(32, 'big')
    if compressed:
        prefix = b'\x02' if (y % 2 == 0) else b'\x03'
        return prefix + xb
    return b'\x04' + xb + yb


def hash160_of_pub(pub: bytes) -> bytes:
    return ripemd160(sha256(pub))


def p2pkh(pub: bytes, mainnet: bool = True) -> str:
    ver = b'\x00' if mainnet else b'\x6f'
    return b58check_encode(ver + hash160_of_pub(pub))


def p2sh_from_script(script: bytes,
                     mainnet: bool = True) -> str:
    ver = b'\x05' if mainnet else b'\xc4'
    h = ripemd160(sha256(script))
    return b58check_encode(ver + h)


def p2wpkh_bech32(pub: bytes,
                  mainnet: bool = True) -> str:
    prog = hash160_of_pub(pub)
    hrp = 'bc' if mainnet else 'tb'
    data = [0] + convertbits(prog, 8, 5)
    return bech32_encode(hrp, data, spec='bech32')


def p2sh_p2wpkh(pub: bytes,
                mainnet: bool = True) -> str:
    prog = hash160_of_pub(pub)
    redeem = b'\x00\x14' + prog
    return p2sh_from_script(redeem, mainnet=mainnet)


def tagged_hash(tag: str, msg: bytes) -> bytes:
    th = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(th + th + msg).digest()


def p2tr_from_priv(secret: int,
                   mainnet: bool = True) -> str:
    Ppt = scalar_mult(secret)
    if Ppt is None:
        raise ValueError("Invalid secret")
    x, _ = Ppt
    xbytes = x.to_bytes(32, 'big')
    tweak = int.from_bytes(tagged_hash("TapTweak", xbytes), 'big') % N
    Q = point_add(Ppt, scalar_mult(tweak)) if tweak != 0 else Ppt
    if Q is None:
        raise ValueError("Tweaked point at infinity")
    qx, _ = Q
    out_x = qx.to_bytes(32, 'big')
    hrp = 'bc' if mainnet else 'tb'
    data = [1] + convertbits(out_x, 8, 5)
    return bech32_encode(hrp, data, spec='bech32m')


def priv_to_wif(priv32: bytes,
                compressed: bool = True,
                testnet: bool = False) -> str:
    ver = b'\xEF' if testnet else b'\x80'
    body = ver + priv32 + (b'\x01' if compressed else b'')
    return b58check_encode(body)


# ============================================================
# helpers
# ============================================================

ALLOWED_CORE = [n for n in (3, 5, 10, 15, 20, 25, 30) if is_allowed_core_length(n)]


def maybe_append_cs(words: List[str], cs_mode: str) -> List[str]:
    """
    Generator policy:
      - dacă lungimea este CANONICĂ (3/5/10/15/20/25/30), NU adăugăm checksum.
      - altfel, dacă cineva chiar generează fraze cu lungimi non-standard,
        se poate aplica checksum V1 / V2.
    """
    if cs_mode == "none":
        return words
    if is_allowed_core_length(len(words)):
        # nu atingem fraza, cheile rămân compatibile cu FINAL mode
        return words
    if cs_mode == "v1":
        return append_checksum_v1(words, cs=1)
    if cs_mode == "v2":
        return append_checksum_v2(words)
    raise ValueError("cs must be none|v1|v2")


def derive_all_from_phrase(phrase_words: List[str]) -> dict:
    assert_in_list(phrase_words)
    key = phrase_to_key(phrase_words)
    secret = int.from_bytes(key, 'big') % N
    if secret == 0:
        raise ValueError("Derived secret is zero")
    pub_c = pubkey_from_priv(secret, True)
    pub_u = pubkey_from_priv(secret, False)
    return {
        "phrase": " ".join(phrase_words),
        "priv_hex": key.hex(),
        "wif_c_main": priv_to_wif(key, True, False),
        "wif_u_main": priv_to_wif(key, False, False),
        "pub_c_hex": pub_c.hex(),
        "pub_u_hex": pub_u.hex(),
        "p2pkh_c": p2pkh(pub_c, True),
        "p2pkh_u": p2pkh(pub_u, True),
        "p2sh_p2wpkh": p2sh_p2wpkh(pub_c, True),
        "p2wpkh": p2wpkh_bech32(pub_c, True),
        "p2tr": p2tr_from_priv(secret, True),
    }


def random_phrase(rng: random.Random, length: int) -> List[str]:
    if not is_allowed_core_length(length):
        raise ValueError(f"Length must be one of {ALLOWED_CORE}")
    return [rng.choice(WORDS) for _ in range(length)]


def alpha_phrase_same_word(word: str, length: int) -> List[str]:
    if not is_allowed_core_length(length):
        raise ValueError(f"Length must be one of {ALLOWED_CORE}")
    assert_in_list([word])
    return [word] * length


def phrase_from_name(name: str, length: int) -> List[str]:
    if not is_allowed_core_length(length):
        raise ValueError(f"Length must be one of {ALLOWED_CORE}")
    h = hashlib.sha256(name.encode('utf-8')).digest()
    words: List[str] = []
    i = 0
    for _ in range(length):
        if i + 2 > len(h):
            h = hashlib.sha256(h).digest()
            i = 0
        idx = int.from_bytes(h[i:i+2], 'big') % len(WORDS)
        i += 2
        words.append(WORDS[idx])
    return words


def alpha_last_phrases(base_word: str,
                       length: int,
                       count: int,
                       start: int = 0,
                       step: int = 1) -> List[List[str]]:
    if not is_allowed_core_length(length):
        raise ValueError(f"Length must be one of {ALLOWED_CORE}")
    assert_in_list([base_word])
    sorted_words = sorted(WORDS)
    out: List[List[str]] = []
    idx = start % len(sorted_words)
    for _ in range(count):
        tail = sorted_words[idx]
        out.append([base_word] * (length - 1) + [tail])
        idx = (idx + step) % len(sorted_words)
    return out


# ============================================================
# SCHEMATIC DSL
# ============================================================

def _sorted_words() -> List[str]:
    return sorted(WORDS)


def _parse_block(token: str) -> List[List[str]]:
    """
    Return a list of 'segments', each segment is a list[str] words to append.
    Single-valued -> [[word]], multi-valued -> [[alt1],[alt2],...]
    """
    if not (token.startswith("{") and token.endswith("}")):
        return [[token]]

    body = token[1:-1].strip()

    if body.startswith("w:"):
        w = body[2:].strip()
        return [[w]]

    if body.startswith("rep:"):
        try:
            _, word, n = body.split(":", 2)
            n = int(n)
        except Exception:
            raise ValueError(f"Bad rep block: {token}")
        return [[word] * n]

    if body.startswith("list:"):
        opts = body[5:].strip()
        words = [x.strip() for x in opts.split("|") if x.strip()]
        if not words:
            raise ValueError(f"Empty list block: {token}")
        return [[w] for w in words]

    if body.startswith("range:"):
        try:
            _, s, step, cnt = body.split(":", 3)
            s = int(s)
            step = int(step)
            cnt = int(cnt)
        except Exception:
            raise ValueError(f"Bad range block: {token}")
        sw = _sorted_words()
        out: List[List[str]] = []
        idx = s % len(sw)
        for _ in range(cnt):
            out.append([sw[idx]])
            idx = (idx + step) % len(sw)
        return out

    raise ValueError(f"Unknown block: {token}")


def expand_schema(schema: str, max_total: int = 1000) -> List[List[str]]:
    """
    Cartesian product over blocks (finite).
    """
    tokens = [t for t in schema.strip().split() if t]
    if not tokens:
        raise ValueError("Empty schema")
    variants: List[List[List[str]]] = [_parse_block(tok) for tok in tokens]

    out: List[List[str]] = [[]]
    for pos_variants in variants:
        new_out: List[List[str]] = []
        for acc in out:
            for seg in pos_variants:
                phrase = acc + seg
                new_out.append(phrase)
                if len(new_out) >= max_total:
                    break
            if len(new_out) >= max_total:
                break
        out = new_out
        if not out:
            break
    return out


def ensure_allowed_length(ph: List[str]) -> None:
    if not is_allowed_core_length(len(ph)):
        raise ValueError(f"Phrase length {len(ph)} not in {ALLOWED_CORE} -> '{' '.join(ph)}'")


# ============================================================
# streaming helpers
# ============================================================

def stream_phrases_random(length: int, cs: str, seed: str | None):
    rng = random.Random(seed) if seed is not None else random.Random()
    while True:
        core = random_phrase(rng, length)
        yield maybe_append_cs(core, cs)


def stream_phrases_alpha(word: str, length: int, cs: str):
    core = alpha_phrase_same_word(word, length)
    while True:
        yield maybe_append_cs(core, cs)


def stream_phrases_alpha_last(base_word: str,
                              length: int,
                              cs: str,
                              start: int,
                              step: int):
    assert_in_list([base_word])
    sorted_words = _sorted_words()
    idx = start % len(sorted_words)
    while True:
        tail = sorted_words[idx]
        core = [base_word] * (length - 1) + [tail]
        yield maybe_append_cs(core, cs)
        idx = (idx + step) % len(sorted_words)


def stream_phrases_name(name_list: List[str], length: int, cs: str):
    cached = [phrase_from_name(nm, length) for nm in name_list]
    if not cached:
        return
    idx = 0
    while True:
        core = cached[idx]
        yield maybe_append_cs(core, cs)
        idx = (idx + 1) % len(cached)


def stream_phrases_schematic(schema: str, cs: str, max_total: int):
    base = expand_schema(schema, max_total=max_total)
    if not base:
        return
    checked: List[List[str]] = []
    for ph in base:
        ensure_allowed_length(ph)
        checked.append(ph)
    idx = 0
    while True:
        core = checked[idx]
        yield maybe_append_cs(core, cs)
        idx = (idx + 1) % len(checked)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Generate BTC addresses (u/c) from AKM phrases."
    )
    ap.add_argument("--mode",
                    choices=["random", "name", "alpha", "alpha-last", "schematic"],
                    required=True)
    ap.add_argument("--length", type=int, default=5,
                    help=f"Core length {ALLOWED_CORE}")
    ap.add_argument("--count", type=int, default=1,
                    help="How many phrases (ignored with --infinite)")
    ap.add_argument("--seed", type=str,
                    help="RNG seed for --mode random")
    ap.add_argument("--name", type=str,
                    help="Name for --mode name")
    ap.add_argument("--names-file", type=str,
                    help="File with names, one per line")
    ap.add_argument("--cs", choices=["none", "v1", "v2"], default="none",
                    help="Append checksum words (only for non-canonical lengths)")
    ap.add_argument("--csv", type=str,
                    help="Optional CSV output path (finite mode only)")
    ap.add_argument("--quiet", action="store_true",
                    help="Compact one-line output")

    # streaming
    ap.add_argument("--infinite", action="store_true",
                    help="Stream phrases indefinitely")

    # alpha / alpha-last
    ap.add_argument("--word", type=str,
                    help="Base word (alpha = repeated; alpha-last = prefix)")
    ap.add_argument("--start", type=int, default=0,
                    help="Start index in sorted WORDS for iterating tail")
    ap.add_argument("--step", type=int, default=1,
                    help="Step through sorted WORDS")

    # schematic
    ap.add_argument("--schema", type=str,
                    help='Mini-DSL schema, e.g. "{rep:abis:2} {range:0:1:512}"')
    ap.add_argument("--max", type=int, default=1000,
                    help="Max phrases to expand for schematic")

    args = ap.parse_args()

    if args.infinite and args.csv:
        ap.error("--csv is not supported together with --infinite")

    # ========================================================
    # INFINITE STREAM MODE
    # ========================================================

    if args.infinite:
        if args.mode in ["random", "name", "alpha", "alpha-last"]:
            if not is_allowed_core_length(args.length):
                ap.error(f"--length must be one of {ALLOWED_CORE}")

        # select generator
        if args.mode == "random":
            gen = stream_phrases_random(args.length, args.cs, args.seed)

        elif args.mode == "alpha":
            if not args.word:
                ap.error("--word is required for --mode alpha")
            gen = stream_phrases_alpha(args.word, args.length, args.cs)

        elif args.mode == "alpha-last":
            if not args.word:
                ap.error("--word is required for --mode alpha-last")
            gen = stream_phrases_alpha_last(args.word, args.length,
                                            args.cs, args.start, args.step)

        elif args.mode == "name":
            names: List[str] = []
            if args.name:
                names.append(args.name)
            if args.names_file:
                if not os.path.exists(args.names_file):
                    ap.error(f"names file not found: {args.names_file}")
                with open(args.names_file, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if s:
                            names.append(s)
            if not names:
                ap.error("Provide --name or --names-file for --mode name")
            gen = stream_phrases_name(names, args.length, args.cs)

        else:  # schematic
            if not args.schema:
                ap.error("--schema is required for --mode schematic")
            gen = stream_phrases_schematic(args.schema, args.cs, args.max)

        total = 0
        last_report = time.time()
        last_count = 0

        try:
            for phrase_words in gen:
                r = derive_all_from_phrase(phrase_words)
                total += 1

                now = time.time()
                if now - last_report >= 1.0:
                    interval = max(now - last_report, 1e-6)
                    delta = total - last_count
                    speed = delta / interval
                    sys.stderr.write(
                        f"\r[STATS] {speed:8.2f} addr/sec (total: {total})"
                    )
                    sys.stderr.flush()
                    last_report = now
                    last_count = total

                if args.quiet:
                    # seed | P2PKH(c) | P2PKH(u) | bech32 | taproot
                    print(
                        f"{r['phrase']}|{r['p2pkh_c']}|{r['p2pkh_u']}|{r['p2wpkh']}|{r['p2tr']}",
                        flush=True,
                    )
                else:
                    print("\n" + "=" * 68)
                    print("PHRASE : ", r["phrase"])
                    print("PRIV   : ", r["priv_hex"])
                    print("WIF c  : ", r["wif_c_main"])
                    print("WIF u  : ", r["wif_u_main"])
                    print("PUB c  : ", r["pub_c_hex"])
                    print("PUB u  : ", r["pub_u_hex"])
                    print("-- ADDRESSES --")
                    print("P2PKH (u)      :", r["p2pkh_u"])
                    print("P2PKH (c)      :", r["p2pkh_c"])
                    print("P2SH(P2WPKH)   :", r["p2sh_p2wpkh"])
                    print("P2WPKH (bech)  :", r["p2wpkh"])
                    print("P2TR (taproot) :", r["p2tr"])
                    print("=" * 68, flush=True)

        except KeyboardInterrupt:
            sys.stderr.write("\n[+] Stopped by user (Ctrl+C).\n")
        return

    # ========================================================
    # FINITE MODE
    # ========================================================

    phrases: List[List[str]] = []

    if args.mode == "random":
        if not is_allowed_core_length(args.length):
            ap.error(f"--length must be one of {ALLOWED_CORE}")
        rng = random.Random(args.seed) if args.seed else random.Random()
        for _ in range(args.count):
            core = random_phrase(rng, args.length)
            phrases.append(maybe_append_cs(core, args.cs))

    elif args.mode == "name":
        if not is_allowed_core_length(args.length):
            ap.error(f"--length must be one of {ALLOWED_CORE}")
        names: List[str] = []
        if args.name:
            names.append(args.name)
        if args.names_file:
            if not os.path.exists(args.names_file):
                ap.error(f"names file not found: {args.names_file}")
            with open(args.names_file, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        names.append(s)
        if not names:
            ap.error("Provide --name or --names-file for --mode name")
        for nm in names[: args.count]:
            core = phrase_from_name(nm, args.length)
            phrases.append(maybe_append_cs(core, args.cs))

    elif args.mode == "alpha":
        if not is_allowed_core_length(args.length):
            ap.error(f"--length must be one of {ALLOWED_CORE}")
        if not args.word:
            ap.error("--word is required for --mode alpha")
        core = alpha_phrase_same_word(args.word, args.length)
        for _ in range(args.count):
            phrases.append(maybe_append_cs(core, args.cs))

    elif args.mode == "alpha-last":
        if not is_allowed_core_length(args.length):
            ap.error(f"--length must be one of {ALLOWED_CORE}")
        if not args.word:
            ap.error("--word is required for --mode alpha-last")
        phs = alpha_last_phrases(args.word, args.length,
                                 args.count, args.start, args.step)
        for core in phs:
            phrases.append(maybe_append_cs(core, args.cs))

    else:  # schematic finite
        if not args.schema:
            ap.error("--schema is required for --mode schematic")
        expanded = expand_schema(args.schema, max_total=args.max)
        if not expanded:
            ap.error("Schema resulted in 0 phrases")
        for ph in expanded[: args.count]:
            ensure_allowed_length(ph)
            phrases.append(maybe_append_cs(ph, args.cs))

    rows = [derive_all_from_phrase(ph) for ph in phrases]

    if args.csv:
        fieldnames = [
            "phrase", "priv_hex", "wif_c_main", "wif_u_main",
            "pub_c_hex", "pub_u_hex", "p2pkh_u", "p2pkh_c",
            "p2sh_p2wpkh", "p2wpkh", "p2tr",
        ]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in fieldnames})
        if not args.quiet:
            print(f"[+] CSV written: {args.csv}")

    if not args.quiet:
        for r in rows:
            print("\n" + "=" * 68)
            print("PHRASE : ", r["phrase"])
            print("PRIV   : ", r["priv_hex"])
            print("WIF c  : ", r["wif_c_main"])
            print("WIF u  : ", r["wif_u_main"])
            print("PUB c  : ", r["pub_c_hex"])
            print("PUB u  : ", r["pub_u_hex"])
            print("-- ADDRESSES --")
            print("P2PKH (u)      :", r["p2pkh_u"])
            print("P2PKH (c)      :", r["p2pkh_c"])
            print("P2SH(P2WPKH)   :", r["p2sh_p2wpkh"])
            print("P2WPKH (bech)  :", r["p2wpkh"])
            print("P2TR (taproot) :", r["p2tr"])
            print("=" * 68)
        print("\nDone.")

    if args.quiet and not args.csv:
        for r in rows:
            print(
                f"{r['phrase']}|{r['p2pkh_c']}|{r['p2pkh_u']}|{r['p2wpkh']}|{r['p2tr']}"
            )


if __name__ == "__main__":
    main()
