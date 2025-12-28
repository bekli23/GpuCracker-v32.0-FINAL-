#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_seed2priv.py — AKM phrase → private key + checksum V1 (FINAL mode)
---------------------------------------------------------------------
Reguli:
 - Lungime totală permisă: 3, 5, 10, 15, 20, 25, 30 (INCLUDE checksum-ul).
 - Ultimul cuvânt este checksum (V1). Nu se adaugă nimic în plus.
 - Opțional: --calc-cs doar calculează ce ar trebui să fie checksum-ul
   pentru CORE = fraza fără ultimul cuvânt.

Profile:
 - Profilele sunt definite în:
       akm_words_512.py (profil core "akm2-core")
       akm_words_512_extra.py (AKM_EXTRA_PROFILES)
 - Poți:
       * lista profilele:    --list-profiles
       * alege un profil:    --profile akm2-lab-v1
 - FĂRĂ niciun export de AKM_PROFILE în environment.

Necesită:
  - akm_words_512.py (versiunea cu profile + *_v1)
  - akm_words_512_extra.py (opțional, pentru profile extra)
"""

from __future__ import annotations
import argparse
import sys
import hashlib
from typing import Iterable, Tuple, List

import akm_words_512 as akm
from akm_words_512 import (
    phrase_to_key,
    assert_in_list,
    checksum_words_v1,
    verify_checksum_v1,
    strip_checksum_v1,
)

# ------------ setări de politică ------------
ALLOWED_TOTAL_LENGTHS = {3, 5, 10, 15, 20, 25, 30}
CS_WORDS = 1  # în FINAL mode, ultimul cuvânt este checksum

# ---------------- secp256k1 minimal (affine) ----------------
P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
Gx = 55066263022277343669578718895168534326250603453777594175500187360389116729240
Gy = 32670510020758816978083085130507043184471273380659243275938904335757337482424


def inv_mod(a: int, m: int) -> int:
    return pow(a, -1, m)


def point_add(Pt: Tuple[int, int] | None, Qt: Tuple[int, int] | None) -> Tuple[int, int] | None:
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


def scalar_mult(k: int, Pt: Tuple[int, int] = (Gx, Gy)) -> Tuple[int, int] | None:
    if k % N == 0:
        return None
    k = k % N
    R = None
    Q = Pt
    while k:
        if (k & 1) != 0:
            R = point_add(R, Q)
        Q = point_add(Q, Q)
        k >>= 1
    return R


# ---------------- hashing / encoding utils ----------------
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


def convertbits(data: bytes, frombits: int, tobits: int, pad: bool = True) -> list[int]:
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
    GENERATORS = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for v in values:
        b = (chk >> 25)
        chk = ((chk & 0x1ffffff) << 5) ^ v
        for i in range(5):
            if (b >> i) & 1:
                chk ^= GENERATORS[i]
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _bech32_create_checksum(hrp: str, data: list[int], spec='bech32'):
    const = 1 if spec == 'bech32' else 0x2bc830a3
    values = _bech32_hrp_expand(hrp) + data
    polymod = _bech32_polymod(values + [0] * 6) ^ const
    return [(polymod >> (5 * (5 - i))) & 31 for i in range(6)]


def bech32_encode(hrp: str, data: list[int], spec='bech32') -> str:
    checksum = _bech32_create_checksum(hrp, data, spec=spec)
    combined = data + checksum
    return hrp + '1' + ''.join([BECH32_CHARSET[d] for d in combined])


# ---------------- BTC address generation ----------------
def pubkey_from_priv(secret: int, compressed: bool = True) -> bytes:
    P = scalar_mult(secret)
    if P is None:
        raise ValueError("Invalid secret -> point at infinity")
    x, y = P
    xb = x.to_bytes(32, 'big')
    yb = y.to_bytes(32, 'big')
    if compressed:
        prefix = b'\x02' if (y % 2 == 0) else b'\x03'
        return prefix + xb
    else:
        return b'\x04' + xb + yb


def hash160_of_pub(pub: bytes) -> bytes:
    return ripemd160(sha256(pub))


def p2pkh(pub: bytes, mainnet: bool = True) -> str:
    ver = b'\x00' if mainnet else b'\x6f'
    return b58check_encode(ver + hash160_of_pub(pub))


def p2sh_from_script(script: bytes, mainnet: bool = True) -> str:
    ver = b'\x05' if mainnet else b'\xc4'
    h = ripemd160(sha256(script))
    return b58check_encode(ver + h)


def p2wpkh_bech32(pub: bytes, mainnet: bool = True) -> str:
    prog = hash160_of_pub(pub)
    hrp = 'bc' if mainnet else 'tb'
    data = [0] + convertbits(prog, 8, 5)
    return bech32_encode(hrp, data, spec='bech32')


def p2sh_p2wpkh(pub: bytes, mainnet: bool = True) -> str:
    prog = hash160_of_pub(pub)
    redeem = b'\x00\x14' + prog
    return p2sh_from_script(redeem, mainnet=mainnet)


def tagged_hash(tag: str, msg: bytes) -> bytes:
    tag_hash = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(tag_hash + tag_hash + msg).digest()


def p2tr_from_point(P: Tuple[int, int], mainnet: bool = True) -> Tuple[str, bytes]:
    x, y = P
    xbytes = x.to_bytes(32, 'big')
    tweak = int.from_bytes(tagged_hash("TapTweak", xbytes), 'big') % N
    Q = point_add(P, scalar_mult(tweak)) if tweak != 0 else P
    if Q is None:
        raise ValueError("Tweaked point at infinity")
    qx, qy = Q
    out_x = qx.to_bytes(32, 'big')
    hrp = 'bc' if mainnet else 'tb'
    data = [1] + convertbits(out_x, 8, 5)
    return bech32_encode(hrp, data, spec='bech32m'), out_x


# ---------------- WIF encoding ----------------
def priv_to_wif(priv32: bytes, compressed: bool = True, testnet: bool = False) -> str:
    ver = b'\xEF' if testnet else b'\x80'
    body = ver + priv32 + (b'\x01' if compressed else b'')
    return b58check_encode(body)


# ---------------- Phrase -> Key ----------------
def phrase_to_priv_int(words: Iterable[str]) -> int:
    key_bytes = phrase_to_key(words)
    return int.from_bytes(key_bytes, 'big')


# ---------------- Helpers: length / profiles ----------------
def _enforce_total_length_or_fail(words: List[str]) -> bool:
    total = len(words)
    if total not in ALLOWED_TOTAL_LENGTHS:
        allowed = ", ".join(str(x) for x in sorted(ALLOWED_TOTAL_LENGTHS))
        print(
            f"[E] Lungime TOTALĂ nepermisă: {total} cuvinte.\n"
            f"    Lungimi permise: {allowed}.\n"
            f"    În FINAL mode, fraza include checksum-ul în acest total."
        )
        return False
    return True


def _available_profiles() -> List[str]:
    """
    Citește profilele disponibile din akm_words_512 + akm_words_512_extra.
    Nu depindem de env AKM_PROFILE.
    """
    # Dacă avem AKM_KNOWN_PROFILES, folosim aia direct:
    if hasattr(akm, "AKM_KNOWN_PROFILES"):
        return list(getattr(akm, "AKM_KNOWN_PROFILES"))
    # fallback: core + extra brute
    names: List[str] = ["akm2-core"]
    extra = getattr(akm, "AKM_EXTRA_PROFILES", {})
    if isinstance(extra, dict):
        for k in sorted(extra.keys()):
            if k not in names:
                names.append(k)
    return names


def _print_profiles_and_exit() -> None:
    names = _available_profiles()
    active = getattr(akm, "get_active_profile", lambda: "akm2-core")()
    print("[AKM] Profile disponibile:")
    for n in names:
        mark = " (activ)" if n == active else ""
        print(f"  - {n}{mark}")
    sys.exit(0)


def _apply_profile_from_cli(name: str | None) -> None:
    if not name:
        return
    if not hasattr(akm, "set_profile"):
        sys.stderr.write("[AKM] set_profile() nu există în akm_words_512. Actualizează fișierul.\n")
        sys.exit(1)
    try:
        akm.set_profile(name)
    except Exception as e:
        sys.stderr.write(f"[AKM] Eroare la set_profile('{name}'): {e}\n")
        sys.exit(1)


# ---------------- Main processing (FINAL mode) ----------------
def process_phrase_final(phrase: str, show_wif: bool = True, calc_cs: bool = False) -> None:
    words = [w for w in phrase.strip().split() if w]
    if not words:
        print("[E] Frază goală")
        return

    try:
        assert_in_list(words)
    except Exception as e:
        print(f"[E] cuvinte necunoscute: {e}")
        return

    if not _enforce_total_length_or_fail(words):
        return

    if calc_cs:
        if len(words) <= CS_WORDS:
            print("[E] Prea puține cuvinte ca să calculez checksum pentru CORE.")
            return
        core = words[:-CS_WORDS]
        exp = checksum_words_v1(core, cs=CS_WORDS)
        print(f"[i] CORE: {' '.join(core)}")
        print(f"[i] Expected CS (V1): {' '.join(exp)}")
        return

    ok, info = verify_checksum_v1(words, cs=CS_WORDS)
    if not ok:
        exp = info.get("expected", [])
        print("[!] Checksum V1 invalid.")
        print(f"    Expected CS: {' '.join(exp) if exp else '(necunoscut)'}")
    else:
        got = info.get("got", [])
        print(f"[✓] Checksum V1 valid ({' '.join(got)})")

    core_words = strip_checksum_v1(words, cs=CS_WORDS)

    priv_int = phrase_to_priv_int(core_words)
    priv_bytes = priv_int.to_bytes(32, 'big')

    pub_c = pubkey_from_priv(priv_int, True)
    pub_u = pubkey_from_priv(priv_int, False)
    P = scalar_mult(priv_int)

    print("\n" + "=" * 64)
    print("PHRASE :", " ".join(words))
    print("CORE   :", " ".join(core_words))
    print("WORDS  :", len(words))
    print("PRIVATE HEX :", priv_bytes.hex())
    print("PRIVATE INT :", priv_int)
    if show_wif:
        print("\nWIF (mainnet, compressed)  :", priv_to_wif(priv_bytes, True, False))
        print("WIF (mainnet, uncompressed):", priv_to_wif(priv_bytes, False, False))
        print("WIF (testnet, compressed)  :", priv_to_wif(priv_bytes, True, True))
        print("WIF (testnet, uncompressed):", priv_to_wif(priv_bytes, False, True))

    print("\nPUBKEY compressed  :", pub_c.hex())
    print("PUBKEY uncompressed:", pub_u.hex())

    print("\n-- ADDRESSES (MAINNET) --")
    print("P2PKH           :", p2pkh(pub_c, True))
    print("P2SH(P2WPKH)    :", p2sh_p2wpkh(pub_c, True))
    print("P2WPKH (bech32) :", p2wpkh_bech32(pub_c, True))
    try:
        if P is None:
            raise ValueError("point at infinity")
        tr, _ = p2tr_from_point(P, True)
        print("P2TR (taproot)  :", tr)
    except Exception as e:
        print("P2TR error:", e)
    print("=" * 64 + "\n")


# ---------------- CLI ----------------
def main():
    profiles = _available_profiles()

    ap = argparse.ArgumentParser(
        description="AKM phrase → private key + checksum V1 (FINAL mode) cu suport de profile."
    )
    # profile / introspecție
    ap.add_argument(
        "--list-profiles",
        action="store_true",
        help="Listează profilele AKM disponibile și ieși."
    )
    ap.add_argument(
        "--profile",
        choices=profiles,
        help="Selectează profilul AKM (ex: akm2-lab-v1)."
    )

    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--phrase",
        help='Fraza între ghilimele (ultimul cuvânt e checksum), ex: "livada ora ora lumina livada"',
    )
    src.add_argument("--in", dest="infile", help="Fișier cu fraze, una pe linie")
    src.add_argument("--stdin", action="store_true", help="Citește fraze din stdin (una pe linie)")

    ap.add_argument(
        "--calc-cs",
        action="store_true",
        help="Calculează checksum V1 pentru CORE = fraza fără ultimul cuvânt și afișează așteptatul.",
    )
    ap.add_argument("--no-wif", action="store_true", help="Nu afișa WIF.")

    args = ap.parse_args()

    # dacă vrea doar listă de profile, nu-l obligăm la frază
    if args.list_profiles:
        _print_profiles_and_exit()

    # aplicăm profilul ales, dacă e cazul
    _apply_profile_from_cli(args.profile)

    show_wif = not args.no_wif

    # validare sursă frază
    if not (args.phrase or args.infile or args.stdin):
        ap.error("Trebuie să dai una din: --phrase / --in / --stdin (sau --list-profiles).")

    if args.phrase:
        process_phrase_final(args.phrase, show_wif=show_wif, calc_cs=args.calc_cs)
    elif args.infile:
        with open(args.infile, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    process_phrase_final(line.strip(), show_wif=show_wif, calc_cs=args.calc_cs)
    else:
        for line in sys.stdin:
            if line.strip():
                process_phrase_final(line.strip(), show_wif=show_wif, calc_cs=args.calc_cs)


if __name__ == "__main__":
    main()
