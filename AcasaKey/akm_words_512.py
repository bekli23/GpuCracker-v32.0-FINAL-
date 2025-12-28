#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_words_512.py — 512 cuvinte unice + HKDF stream + HEX pack fix
+ Checksum (V1 și V2) + Reguli speciale + Versiuni/Spec + Anti-weak guard
+ Profile (core + extra din akm_words_512_extra.py)

Expune:
  - WORDS, INDEX, WORDSET
  - AKM_PARAMS, domain_sep(), integrity_hashes()
  - word_stream(word), stream_block(word, nbytes)
  - phrase_to_key(words), phrase_to_key_hexpack_fixed(words)
  - explain_token(word)  -> dict cu baza, decizie, token8
  - checksum_words_v1(core, cs=1), append_checksum_v1, verify_checksum_v1, strip_checksum_v1
  - checksum_words_v2(core), append_checksum_v2, verify_checksum_v2, strip_checksum_v2
  - is_weak_key(key32), enforce_not_weak(key32, allow_weak=False)
  - list_special_overrides()
  - export_spec_json(path="akm_spec.json", include_vectors=True)
  - policy helpers: ALLOWED_CORE_LENGTHS, is_allowed_core_length(n)
  - profile API:
        * AKM_KNOWN_PROFILES  (listă de nume profile)
        * set_profile(name)
        * get_active_profile()

IMPORTANT:
  - SPECIAL_RULES se aplică DOAR când cuvântul NU are override în CUSTOM_HEX.
  - Profile extra se definesc în akm_words_512_extra.py ca AKM_EXTRA_PROFILES.
"""

from __future__ import annotations

import re
import hmac
import hashlib
import unicodedata
import json
import os
import sys
from functools import lru_cache
from typing import Iterable, Iterator, List, Set, Dict, Tuple

# ============================================================
# 0) PARAMETRI GLOBALI / VERSIONARE / POLITICI (CORE)
# ============================================================

AKM_PARAMS: Dict[str, object] = {
    "version": "AKM2",        # profilul core stabil
    "wl": 512,
    "token": "hexpack8",
    "cs": {"v1": True, "v2": True, "v2_words": 2, "v2_bits_per_word": 10},
}

# profil de lungimi permis pentru "core" (separat de derivare)
ALLOWED_CORE_LENGTHS: Set[int] = {3, 5, 10, 15, 20, 25, 30}


def is_allowed_core_length(n: int) -> bool:
    return n in ALLOWED_CORE_LENGTHS


def _canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8", "strict")


def domain_sep(tag: str) -> bytes:
    """
    Domain separator stabil, derivat din AKM_PARAMS.
    Se schimbă dacă se schimbă profilul (AKM_PARAMS).
    """
    base = b"AKM|" + hashlib.sha256(_canonical_json(AKM_PARAMS)).digest()
    return b"%s|" % tag.encode("utf-8", "strict") + base


# ============================================================
# 1) LISTA DE BAZĂ + UNICIZARE
# ============================================================

BASE_WORDS = [
    'abis','acelasi','acoperis','adanc','adapost','adorare','afectiune','aer',
    'ajun','albastru','alb','alge','altar','amintire','amurg','anotimp','apa',
    'apele','apus','apuseni','apusor','aroma','artar','arta','asfintit',
    'asteptare','atingere','aur','aurora','autumn','avion','balta','barca',
    'barza','baterie','batran','bec','bezna','binecuvantare','blandete','boboc',
    'bogatie','bolta','brad','brat','bruma','bucata','bucurie','bujor','burg',
    'camp','campie','cafea','calator','cald','candela','caprioara','caramida',
    'carare','carte','catel','cautare','casa','ceas','cer','cerb','chip',
    'ciocarlie','ciutura','clar','clipa','clopot','coborare','colina','colt',
    'copac','copil','corabie','cord','corn','crang','credinta','crestere',
    'crestet','crin','cuc','cufar','culoare','culme','curcubeu','curte','cupru',
    'cuvant','cutie','daurire','deal','deget','delusor','departare','desert',
    'dimineata','dor','dorinta','drag','draga','drum','drumet','durere',
    'duminica','ecou','efemer','elixir','emisfera','enigma','eter','eternitate',
    'fag','fagure','fantana','farmec','fata','felinar','fenic','fereastra',
    'fericire','feriga','fier','fierar','film','fior','flacara','flamura',
    'floare','fluture','fosnet','fotografie','frag','frate','frezie','frig',
    'fruct','frumusete','frunza','frunte','fulger','furnica','galaxie','galben',
    'gand','gandire','garoafa','gheata','ghetar','ghinda','ghiozdan','glas',
    'glorie','grad','gradina','grai','granita','gust','gura','har','harfa',
    'iarba','iarna','icoana','implinire','inger','insula','insorire','intindere',
    'intuneric','inviere','iubire','iz','izvor','izvoras','joc','jocul','lac',
    'lacrima','laur','lebada','legenda','lemn','leu','libertate','linie',
    'livada','loc','luna','lumina','lume','lunca','lup','lut','manunchi',
    'margine','mare','maree','marina','masa','masina','matase','memorie','mers',
    'metal','mesteacan','mijloc','minune','miere','mireasma','mister','miros',
    'molid','mugur','munte','mur','muzica','natura','negru','nepasare','nisip',
    'noapte','noptiera','nor','norii','noroc','noroi','nunta','ocean','ochi',
    'odihna','oglinda','om','ora','oras','orasel','paine','palma','pana',
    'pamant','parfum','pas','pasare','pasune','pat','pajiste','paun','paznic',
    'pecete','pescuit','perete','peste','petala','piatra','pictura','pierdut',
    'pilde','piper','pisica','placere','plaja','planta','plimbare','ploi',
    'ploiuta','ploaie','plop','poarta','podea','povara','poveste','poteca',
    'prajitura','praf','prastie','prezent','prieten','privighetoare','privire',
    'profunzime','pulbere','punct','punere','putere','radacina','ram','ramura',
    'rau','raza','razboi','realitate','rece','reflex','regat','repaus',
    'rezonanta','risc','rosu','roua','rugaciune','rugina','sabie','sac','salcam',
    'salcie','sanie','sarut','sat','satean','scaun','sclipici','sclipire',
    'scorbura','scrisoare','scut','seara','secunda','semn','senin','seva',
    'sfarsit','sfoara','singuratate','soare','soarec','sofer','somn','soparla',
    'spectru','spic','spuma','stanca','steara','stea','stiuta','stralucire',
    'strugure','subtire','suflare','suflet','susur','talaz','tare','taram',
    'tavan','tacere','teanc','tihna','timp','timpurie','tingire','toamna',
    'trandafir','trandafiriu','tremur','tren','trezire','tristete','trunchi',
    'tunet','ulei','umbra','umbrela','unda','urcare','urma','urs','usa','vale',
    'val','valea','valuri','vant','vantura','vara','vatra','vaza','vedere',
    'verde','verdeata','veselie','via','viata','viitor','vis','visare','viziune',
    'vlastar','vin','vino','voce','vraja','vrabie','vrere','vulpe','vultur',
    'zapada','zar','zambet','zbor','zboras','zenit','zi','zid','zmeu','zori',
    'zvon','acord','adevar','alegere','altaras','anotimpuri','aprinzator',
    'armonie','atingeri','balansoar','balsam','baston','baza','bine','bucurie',
    'cadere','caravana','cascade','cerinta','chemare','cometă','condei','contur',
    'crestin','cruce','daruire','departari','desen','destin','dovada','drumul',
    'faguri','fantezie','finete','fir','flacari','fulg','ganduri','glinda',
    'inaltime','inteles','lumini','margaretă','miez','minte','norocel','ochiuri',
    'orhidee','parere','piersica','ploii','povesti','pulberi','rana','rasarit',
    'salbatic','scanteie','sclipire','soarele','strop','tainic','timpuri',
    'trai','urme','vanturi','vise','zare','ziua','adieri','amurguri','aripi',
    'batai','bucium','cerneală','cunună','doruri','ecouri','emisii','fald',
    'farmecuri','fosnire','frunzar','glasuri','iubiri','izbuc','lacuri',
    'luminiș','murmur','nemurire','noian','oglindiri','paradis','petale',
    'pietriș','plămadă','pribeag','rădăcini','răgaz','străluciri','tainuri',
    'țărână','șoaptă'
]
TARGET = 512


def _strip_diacritics(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def _normalize_ascii_letters(w: str) -> str:
    w = _strip_diacritics(w.lower())
    w = re.sub(r'[^a-z]', '', w)
    return w


_SYL_A = ["la","le","li","lo","lu","ra","re","ri","ro","ru",
          "na","ne","ni","no","nu","ta","te","ti","to","tu",
          "ma","me","mi","mo","mu","sa","se","si","so","su",
          "pa","pe","pi","po","pu","va","ve","vi","vo","vu"]
_SYL_B = ["ba","be","bi","bo","bu","ca","ce","ci","co","cu",
          "fa","fe","fi","fo","fu","ga","ge","gi","go","gu",
          "ha","he","hi","ho","hu","ja","je","ji","jo","ju",
          "ka","ke","ki","ko","ku"]
_SYL_C = ["ra","re","ri","ro","ru","la","le","li","lo","lu",
          "na","ne","ni","no","nu","ta","te","ti","to","tu"]


def _synth_word(counter: int) -> str:
    a = _SYL_A[counter % len(_SYL_A)]
    b = _SYL_B[(counter // len(_SYL_A)) % len(_SYL_B)]
    c = _SYL_C[(counter // (len(_SYL_A) * len(_SYL_B))) % len(_SYL_C)]
    return a + b + c


def _build_unique_512(base: List[str]) -> List[str]:
    used: Set[str] = set()
    out: List[str] = []
    synth_i = 0
    for raw in base:
        if len(out) >= TARGET:
            break
        norm = _normalize_ascii_letters(raw)
        if not norm:
            continue
        if norm in used:
            while True:
                sw = _synth_word(synth_i)
                synth_i += 1
                if sw not in used:
                    out.append(sw)
                    used.add(sw)
                    break
        else:
            out.append(norm)
            used.add(norm)
    while len(out) < TARGET:
        sw = _synth_word(synth_i)
        synth_i += 1
        if sw not in used:
            out.append(sw)
            used.add(sw)
    return out[:TARGET]


WORDS: List[str] = _build_unique_512(BASE_WORDS)
INDEX: Dict[str, int] = {w: i for i, w in enumerate(WORDS)}
WORDSET: Set[str] = set(WORDS)

# ============================================================
# 2) CUSTOM_HEX / SPECIAL_RULES (CORE) + SNAPSHOT
# ============================================================

CUSTOM_HEX: Dict[str, str] = {
    "mare": "0000",
    "acelasi": "1010",
    "acoperis": "2022",
    "adanc": "2200",
    "adapost": "3030",
    "adorare": "0330",
    "afectiune": "0afe",
    "aer": "0ae1",
    "ajun": "0a1a",
    "albastru": "1ab5",
    "alb": "0a1c",
    "alge": "a000",
    "altar": "a170",
    "amintire": "a000",
    "amurg": "d000",
    "anotimp": "1111",
    "apa": "00a0",
}

SPECIAL_RULES: Dict[str, Dict[str, str]] = {
    # EXEMPLE: te poți juca cu ele, dar pentru compatibilitate
    # profilele custom se fac în akm_words_512_extra.py
    "abis": {"fixed8": "00000001"},
    "ora": {"fixed8": "12991299"},
    "lac": {"fixed8": "01991550"},
    "lumina": {"fixed8": "1234c0de"},
}

# snapshot core pentru profile
CORE_AKM_PARAMS = json.loads(json.dumps(AKM_PARAMS))
CORE_CUSTOM_HEX = dict(CUSTOM_HEX)
CORE_SPECIAL_RULES = dict(SPECIAL_RULES)

# placeholdere pentru derived
WORD_HEX: Dict[str, str] = {}
_SPECIAL_RULES_NORM: Dict[str, Dict[str, str]] = {}
_CUSTOM_KEYS_NORM: Set[str] = set()

# ============================================================
# 3) HKDF STREAM + CACHE
# ============================================================

KEY_LEN_BYTES = 32


def _hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def _hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    hlen = hashlib.sha256().digest_size
    n = (length + hlen - 1) // hlen
    okm, t = b"", b""
    for i in range(1, n + 1):
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
    return okm[:length]


@lru_cache(maxsize=1024)
def _stream_prefix(word: str, nbytes: int = 64) -> bytes:
    """Cache: primele n bytes din stream-ul cuvântului (profil-dependent)."""
    idx = INDEX[word]
    salt = (domain_sep("AKM-stream") +
            f"wl:{AKM_PARAMS['wl']}|idx:{idx}|word:{word}".encode("utf-8", "strict"))
    prk = _hkdf_extract(salt, word.encode("utf-8", "strict"))
    info = domain_sep("stream-block|ctr:1")
    return _hkdf_expand(prk, info, nbytes)


def stream_block(word: str, nbytes: int) -> bytes:
    if nbytes <= 64:
        return _stream_prefix(word, 64)[:nbytes]
    idx = INDEX[word]
    salt = (domain_sep("AKM-stream") +
            f"wl:{AKM_PARAMS['wl']}|idx:{idx}|word:{word}".encode("utf-8", "strict"))
    prk = _hkdf_extract(salt, word.encode("utf-8", "strict"))
    out = bytearray(_stream_prefix(word, 64))
    ctr = 2
    while len(out) < nbytes:
        info = domain_sep(f"stream-block|ctr:{ctr}")
        out.extend(_hkdf_expand(prk, info, 32))
        ctr += 1
    return bytes(out[:nbytes])


def word_stream(word: str) -> Iterator[int]:
    if word not in WORDSET:
        raise ValueError(f"Unknown word: {word}")
    buf = bytearray(stream_block(word, 64))
    idx = 64
    while True:
        if buf:
            b = buf[0]
            del buf[0]
            yield b
            continue
        more = stream_block(word, idx + 32)[idx:]
        idx += 32
        buf.extend(more)


def assert_in_list(words: Iterable[str]) -> None:
    missing = [w for w in words if w not in WORDSET]
    if missing:
        raise ValueError(f"Cuvinte în afara listei: {missing[:5]}{'...' if len(missing) > 5 else ''}")


# ============================================================
# 4) TOKENIZARE: CUSTOM_HEX + SPECIAL_RULES + explain_token
#     + infrastructură de rebuild pentru profile
# ============================================================

def _build_word_hex() -> Dict[str, str]:
    m: Dict[str, str] = {}
    for w in WORDS:
        h4 = hashlib.sha256(w.encode("utf-8", "strict")).hexdigest()[:4]
        m[w] = h4
    for w, hx in CUSTOM_HEX.items():
        wn = _normalize_ascii_letters(w)
        if wn in WORDSET:
            if not re.fullmatch(r"[0-9a-fA-F]+", hx):
                raise ValueError(f"CUSTOM_HEX invalid pentru {w}: {hx}")
            m[wn] = hx.lower()
    return m


def _validate_special_rules() -> None:
    for raw, rule in SPECIAL_RULES.items():
        wn = _normalize_ascii_letters(raw)
        if not wn:
            raise ValueError(f"SPECIAL_RULES cheie invalidă: {raw}")

        # fixed8: exact 8 hex
        if "fixed8" in rule:
            fx = rule["fixed8"]
            if not re.fullmatch(r"[0-9a-fA-F]{8}", fx):
                raise ValueError(f"SPECIAL_RULES fixed8 invalid pentru {raw}: {fx}")

        # fixed1 / fixed2 / fixed3: 1,2,3 hex nibbles (prefix for token)
        if "fixed1" in rule:
            f1 = rule["fixed1"]
            if not re.fullmatch(r"[0-9a-fA-F]{1}", f1):
                raise ValueError(f"SPECIAL_RULES fixed1 invalid pentru {raw}: {f1}")
        if "fixed2" in rule:
            f2 = rule["fixed2"]
            if not re.fullmatch(r"[0-9a-fA-F]{2}", f2):
                raise ValueError(f"SPECIAL_RULES fixed2 invalid pentru {raw}: {f2}")
        if "fixed3" in rule:
            f3 = rule["fixed3"]
            if not re.fullmatch(r"[0-9a-fA-F]{3}", f3):
                raise ValueError(f"SPECIAL_RULES fixed3 invalid pentru {raw}: {f3}")

        if "pad_nibble" in rule:
            pn = rule["pad_nibble"]
            if not re.fullmatch(r"[0-9a-fA-F]", pn):
                raise ValueError(f"SPECIAL_RULES pad_nibble invalid pentru {raw}: {pn}")
        if "repeat_last" in rule and rule["repeat_last"] not in (True, False, "1"):
            raise ValueError(f"SPECIAL_RULES repeat_last invalid pentru {raw}")


def _rebuild_derived_tables() -> None:
    """
    Rebuild:
      - WORD_HEX
      - _SPECIAL_RULES_NORM
      - _CUSTOM_KEYS_NORM
    după schimbarea profilului (CUSTOM_HEX / SPECIAL_RULES / AKM_PARAMS).
    """
    global WORD_HEX, _SPECIAL_RULES_NORM, _CUSTOM_KEYS_NORM
    _validate_special_rules()
    WORD_HEX = _build_word_hex()
    _SPECIAL_RULES_NORM = {_normalize_ascii_letters(k): v for k, v in SPECIAL_RULES.items()}
    _CUSTOM_KEYS_NORM = {_normalize_ascii_letters(k) for k in CUSTOM_HEX.keys()}


# build initial tables pentru profilul core
_rebuild_derived_tables()


def list_special_overrides() -> List[Tuple[str, Dict[str, str]]]:
    return [(k, SPECIAL_RULES[k]) for k in sorted(SPECIAL_RULES)]


def _extend_with_stream_hex(word: str, base_hex: str, target_len: int) -> str:
    """
    Completează cu HKDF stream până la target_len.
    Folosit pentru fixed1/fixed2/fixed3: prefix fix, rest HKDF.
    """
    need = max(0, target_len - len(base_hex))
    if need == 0:
        return base_hex[:target_len]
    need_bytes = (need + 1) // 2
    extra = stream_block(word, need_bytes)
    hx = base_hex + extra.hex()
    return hx[:target_len]


def _apply_special_extend(word: str, base4: str) -> Tuple[str, str]:
    """
    Aplică SPECIAL_RULES dacă word NU e în CUSTOM_HEX.
    Returnează (token8, reason) sau ("","") dacă nu s-a aplicat.

    Suportă:
      - fixed8  : token8 complet fix (8 hex)
      - fixed1  : primul nibble fix, rest HKDF
      - fixed2  : primele 2 nibbles fixe, rest HKDF
      - fixed3  : primele 3 nibbles fixe, rest HKDF
      - pad_nibble  : joacă cu baza de 4 hex (legacy)
      - repeat_last : joacă cu ultimul nibble (legacy)
    """
    wn = _normalize_ascii_letters(word)
    if wn in _CUSTOM_KEYS_NORM:
        return "", ""
    rule = _SPECIAL_RULES_NORM.get(wn)
    if not rule:
        return "", ""

    # fixed8: tot tokenul de 8 hex este fix
    if "fixed8" in rule:
        fx = rule["fixed8"].lower()
        return fx, "SPECIAL.fixed8"

    # fixed1 / fixed2 / fixed3: prefix fix, rest HKDF
    if "fixed3" in rule:
        prefix = rule["fixed3"].lower()
        token8 = _extend_with_stream_hex(word, prefix, 8)
        return token8, "SPECIAL.fixed3"

    if "fixed2" in rule:
        prefix = rule["fixed2"].lower()
        token8 = _extend_with_stream_hex(word, prefix, 8)
        return token8, "SPECIAL.fixed2"

    if "fixed1" in rule:
        prefix = rule["fixed1"].lower()
        token8 = _extend_with_stream_hex(word, prefix, 8)
        return token8, "SPECIAL.fixed1"

    # restul rămân cum erau (nu te mai doare capul de ele acum)
    if "pad_nibble" in rule:
        pn = rule["pad_nibble"].lower()
        # aici păstrăm vechiul comportament (joacă pe baza de 4)
        return (base4 + pn * 2)[:4], "SPECIAL.pad"

    if "repeat_last" in rule:
        last = base4[-1]
        return (base4 + last * 2)[:8], "SPECIAL.repeat_last"

    return "", ""


def explain_token(word: str) -> Dict[str, str]:
    """
    Returnează explicație: baza (4 hex), decizie (CUSTOM/SPECIAL/HKDF),
    token8 final și sursa.
    """
    if word not in WORDSET:
        raise ValueError(f"Unknown word: {word}")
    base4 = WORD_HEX[word][:4].lower()
    wn = _normalize_ascii_letters(word)

    # CUSTOM_HEX are prioritate totală
    if wn in _CUSTOM_KEYS_NORM:
        base_full = WORD_HEX[word]
        token8 = base_full[:8] if len(base_full) >= 8 else _extend_with_stream_hex(word, base_full, 8)
        return {"word": word, "base4": base4, "decision": "CUSTOM", "token8": token8}

    # SPECIAL_RULES (fixed8 / fixed1/2/3 / etc.)
    spec_hex, reason = _apply_special_extend(word, base4)
    if spec_hex:
        # dacă regula a returnat mai puțin de 8, completăm cu HKDF
        token8 = spec_hex
        if len(token8) < 8:
            token8 = _extend_with_stream_hex(word, token8, 8)
        return {"word": word, "base4": base4, "decision": reason, "token8": token8}

    # fallback HKDF extend
    token8 = _extend_with_stream_hex(word, base4, 8)
    return {"word": word, "base4": base4, "decision": "HKDF.extend", "token8": token8}


def _token8_for_word(word: str) -> str:
    return explain_token(word)["token8"]


# ============================================================
# 5) DERIVARE CHEIE
# ============================================================

def phrase_to_key_hexpack_fixed(words: Iterable[str]) -> bytes:
    w = list(words)
    assert_in_list(w)
    # override tradițional: toate "mare" -> FF..FF (folosit de tine la teste)
    if len(w) >= 1 and all(x == "mare" for x in w):
        return b'\xff' * 32
    target_hex_len = 64
    out_hex: List[str] = []
    i = 0
    while sum(len(x) for x in out_hex) < target_hex_len:
        out_hex.append(_token8_for_word(w[i]))
        i += 1
        if i == len(w):
            i = 0
    big_hex = "".join(out_hex)[:target_hex_len]
    return bytes.fromhex(big_hex)


def phrase_to_key(words: Iterable[str]) -> bytes:
    return phrase_to_key_hexpack_fixed(words)


# ============================================================
# 6) ANTI-WEAK GUARD
# ============================================================

def is_weak_key(key32: bytes) -> bool:
    if len(key32) != 32:
        return True
    if all(b == 0x00 for b in key32):
        return True
    if all(b == 0xFF for b in key32):
        return True
    hx = key32.hex()
    chunk = hx[:4]
    if chunk * 16 == hx:
        return True
    return False


def enforce_not_weak(key32: bytes, allow_weak: bool = False) -> None:
    if not allow_weak and is_weak_key(key32):
        raise ValueError("Cheie detectată ca slabă. Folosește --allow-weak dacă insiști.")


# ============================================================
# 7) CHECKSUM V1 (9 biți/cuvânt)
# ============================================================

_CS_SALT_V1 = domain_sep("AKM-CS-v1")


def _gen_indices_bitpacked(seed: bytes, bits_per_index: int, count: int) -> List[int]:
    out: List[int] = []
    buf = 0
    blen = 0
    ctr = 0
    mask = (1 << bits_per_index) - 1
    while len(out) < count:
        h = hashlib.sha256(seed + ctr.to_bytes(4, 'big')).digest()
        ctr += 1
        for b in h:
            buf = (buf << 8) | b
            blen += 8
            while blen >= bits_per_index and len(out) < count:
                blen -= bits_per_index
                idx = (buf >> blen) & mask
                out.append(idx)
                buf &= (1 << blen) - 1
    return out


def checksum_words_v1(core_words: Iterable[str], cs: int = 1) -> List[str]:
    cw = list(core_words)
    assert_in_list(cw)
    key = phrase_to_key(cw)
    meta = len(cw).to_bytes(2, 'big')
    seed = key + _CS_SALT_V1 + meta
    idxs = _gen_indices_bitpacked(seed, 9, cs)
    return [WORDS[i] for i in idxs]


def append_checksum_v1(core_words: Iterable[str], cs: int = 1) -> List[str]:
    cw = list(core_words)
    assert_in_list(cw)
    return cw + checksum_words_v1(cw, cs=cs)


def verify_checksum_v1(words_with_cs: Iterable[str], cs: int = 1) -> Tuple[bool, Dict[str, object]]:
    ww = list(words_with_cs)
    assert_in_list(ww)
    n = len(ww)
    if n in ALLOWED_CORE_LENGTHS:
        return True, {"mode": "no_cs", "core_len": n, "expected": [], "got": []}
    if n <= cs:
        return False, {"error": "fraza e prea scurtă pentru checksum", "cs": cs, "length": n}
    core = ww[:-cs]
    got_cs = ww[-cs:]
    exp_cs = checksum_words_v1(core, cs=cs)
    ok = (got_cs == exp_cs)
    return ok, {"mode": "with_cs", "core_len": len(core), "expected": exp_cs, "got": got_cs}


def strip_checksum_v1(words_with_cs: Iterable[str], cs: int = 1) -> List[str]:
    ww = list(words_with_cs)
    assert_in_list(ww)
    n = len(ww)
    if n in ALLOWED_CORE_LENGTHS:
        return ww[:]
    if n <= cs:
        raise ValueError("Nu ai suficiente cuvinte pentru a elimina checksum-ul.")
    return ww[:-cs]


# ============================================================
# 8) CHECKSUM V2 (2 cuvinte, 10 biți/cuvânt)
# ============================================================

_CS_V2_BITS = AKM_PARAMS["cs"]["v2_bits_per_word"]
_CS_V2_WORDS = AKM_PARAMS["cs"]["v2_words"]
_CS_SALT_V2 = domain_sep("AKM-CS-v2")


def _words_hash() -> bytes:
    return hashlib.sha256("\n".join(WORDS).encode("utf-8", "strict")).digest()


def _akm_params_hash() -> bytes:
    return hashlib.sha256(_canonical_json(AKM_PARAMS)).digest()


def checksum_words_v2(core_words: Iterable[str]) -> List[str]:
    cw = list(core_words)
    assert_in_list(cw)
    key = phrase_to_key(cw)
    seed = key + _words_hash() + _akm_params_hash() + _CS_SALT_V2
    idxs = _gen_indices_bitpacked(seed, int(_CS_V2_BITS), int(_CS_V2_WORDS))
    idxs = [i % len(WORDS) for i in idxs]
    return [WORDS[i] for i in idxs]


def append_checksum_v2(core_words: Iterable[str]) -> List[str]:
    cw = list(core_words)
    assert_in_list(cw)
    return cw + checksum_words_v2(cw)


def verify_checksum_v2(words_with_cs: Iterable[str]) -> Tuple[bool, Dict[str, object]]:
    ww = list(words_with_cs)
    assert_in_list(ww)
    cs = int(_CS_V2_WORDS)
    n = len(ww)
    if n in ALLOWED_CORE_LENGTHS:
        return True, {"mode": "no_cs", "core_len": n, "expected": [], "got": []}
    if n <= cs:
        return False, {"error": "fraza e prea scurtă pentru checksum V2", "cs": cs, "length": n}
    core = ww[:-cs]
    got_cs = ww[-cs:]
    exp_cs = checksum_words_v2(core)
    ok = (got_cs == exp_cs)
    return ok, {"mode": "with_cs", "core_len": len(core), "expected": exp_cs, "got": got_cs}


def strip_checksum_v2(words_with_cs: Iterable[str]) -> List[str]:
    cs = int(_CS_V2_WORDS)
    ww = list(words_with_cs)
    assert_in_list(ww)
    n = len(ww)
    if n in ALLOWED_CORE_LENGTHS:
        return ww[:]
    if n <= cs:
        raise ValueError("Nu ai suficiente cuvinte pentru a elimina checksum-ul V2.")
    return ww[:-cs]


# ============================================================
# 9) INTEGRITATE + EXPORT SPEC
# ============================================================

def integrity_hashes() -> Dict[str, str]:
    words_hash = hashlib.sha256("\n".join(WORDS).encode("utf-8", "strict")).hexdigest()
    custom_hex_hash = hashlib.sha256(_canonical_json(CUSTOM_HEX)).hexdigest()
    special_rules_hash = hashlib.sha256(_canonical_json(SPECIAL_RULES)).hexdigest()
    akm_params_hash = hashlib.sha256(_canonical_json(AKM_PARAMS)).hexdigest()
    return {
        "WORDS_HASH": words_hash,
        "CUSTOM_HEX_HASH": custom_hex_hash,
        "SPECIAL_RULES_HASH": special_rules_hash,
        "AKM_PARAMS_HASH": akm_params_hash,
    }


def _safe_key_hex(words: List[str]) -> str:
    try:
        return phrase_to_key(words).hex()
    except Exception as e:
        return f"<error:{e}>"


def export_spec_json(path: str = "akm_spec.json", include_vectors: bool = True) -> None:
    data = {
        "AKM_PARAMS": AKM_PARAMS,
        "INTEGRITY": integrity_hashes(),
        "TOKEN_MODE": "hexpack8",
        "HAS_CS_V1": bool(AKM_PARAMS["cs"]["v1"]),
        "HAS_CS_V2": bool(AKM_PARAMS["cs"]["v2"]),
        "SPECIAL_RULES": SPECIAL_RULES,
        "CUSTOM_HEX_KEYS": sorted(list(CUSTOM_HEX.keys())),
        "ACTIVE_PROFILE": get_active_profile(),
    }
    if include_vectors:
        vectors = []
        samples = [
            ["mare", "mare", "mare"],
            ["apele", "apele", "apele", "apele", "apele"],
            ["aer", "apa", "abis", "ajun", "alb"],
        ]
        for wds in samples:
            vectors.append({
                "phrase": wds,
                "key_hex": _safe_key_hex(wds),
                "token8": [_token8_for_word(w) for w in wds],
                "cs_v1_mode": verify_checksum_v1(wds, 1)[1].get("mode"),
                "cs_v1": checksum_words_v1(wds, 1),
                "cs_v2_mode": verify_checksum_v2(wds)[1].get("mode"),
                "cs_v2": checksum_words_v2(wds),
            })
        data["TEST_VECTORS"] = vectors

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# 10) PROFILE: extra din akm_words_512_extra.py
# ============================================================

try:
    from akm_words_512_extra import AKM_EXTRA_PROFILES  # type: ignore
except Exception as e:
    AKM_EXTRA_PROFILES = {}
    sys.stderr.write(f"[AKM] Warning: cannot import akm_words_512_extra.AKM_EXTRA_PROFILES: {e}\n")

# listă publică pentru CLI-uri (akm_seed2priv, akm_genaddr etc.)
AKM_KNOWN_PROFILES: List[str] = ["akm2-core"] + sorted(AKM_EXTRA_PROFILES.keys())

AKM_ACTIVE_PROFILE: str = "akm2-core"


def _apply_profile(profile_name: str) -> None:
    """
    Aplică un profil definit în AKM_EXTRA_PROFILES peste CORE_*.
    Dacă profilul e 'akm2-core', revine la setările de bază.
    """
    global AKM_PARAMS, CUSTOM_HEX, SPECIAL_RULES, AKM_ACTIVE_PROFILE
    global _CS_V2_BITS, _CS_V2_WORDS, _CS_SALT_V1, _CS_SALT_V2

    if profile_name == "akm2-core":
        AKM_PARAMS = json.loads(json.dumps(CORE_AKM_PARAMS))
        CUSTOM_HEX.clear()
        CUSTOM_HEX.update(CORE_CUSTOM_HEX)
        SPECIAL_RULES.clear()
        SPECIAL_RULES.update(CORE_SPECIAL_RULES)
    else:
        prof = AKM_EXTRA_PROFILES.get(profile_name)
        if not prof:
            raise ValueError(f"Unknown AKM profile: {profile_name}")

        AKM_PARAMS = json.loads(json.dumps(CORE_AKM_PARAMS))
        if "AKM_PARAMS" in prof and isinstance(prof["AKM_PARAMS"], dict):
            for k, v in prof["AKM_PARAMS"].items():
                AKM_PARAMS[k] = v

        CUSTOM_HEX.clear()
        CUSTOM_HEX.update(CORE_CUSTOM_HEX)
        if "CUSTOM_HEX" in prof and isinstance(prof["CUSTOM_HEX"], dict):
            CUSTOM_HEX.update(prof["CUSTOM_HEX"])

        SPECIAL_RULES.clear()
        SPECIAL_RULES.update(CORE_SPECIAL_RULES)
        if "SPECIAL_RULES" in prof and isinstance(prof["SPECIAL_RULES"], dict):
            SPECIAL_RULES.update(prof["SPECIAL_RULES"])

    _rebuild_derived_tables()
    _stream_prefix.cache_clear()

    _CS_V2_BITS = AKM_PARAMS["cs"]["v2_bits_per_word"]
    _CS_V2_WORDS = AKM_PARAMS["cs"]["v2_words"]
    globals()["_CS_SALT_V1"] = domain_sep("AKM-CS-v1")
    globals()["_CS_SALT_V2"] = domain_sep("AKM-CS-v2")

    AKM_ACTIVE_PROFILE = profile_name


def set_profile(profile_name: str) -> None:
    """
    API public: scripturile pot selecta profilul activ.
    """
    if profile_name not in AKM_KNOWN_PROFILES:
        raise ValueError(f"Profil necunoscut: {profile_name}")
    _apply_profile(profile_name)


def get_active_profile() -> str:
    return AKM_ACTIVE_PROFILE


_env_prof = os.environ.get("AKM_PROFILE", "").strip()
if _env_prof:
    try:
        set_profile(_env_prof)
    except Exception as e:
        sys.stderr.write(f"[AKM] Warning: cannot apply profile '{_env_prof}': {e}\n")


# ============================================================
# 11) MAIN DEMO
# ============================================================

if __name__ == "__main__":
    print("AKM_PARAMS     :", json.dumps(AKM_PARAMS, ensure_ascii=False))
    print("ACTIVE_PROFILE :", get_active_profile())
    print("KNOWN_PROFILES :", AKM_KNOWN_PROFILES)
    print("INTEGRITY      :", integrity_hashes())
    for w in ["mare", "apele", "aer", "apa", "ochiuri", "privire"]:
        print("EXPLAIN", w, "->", explain_token(w))

    demo = ["apele", "apele", "apele", "apele", "apele"]
    key = phrase_to_key(demo)
    print("KEY(demo)      :", key.hex())
    print("is_weak_key    :", is_weak_key(key))
    try:
        enforce_not_weak(key, allow_weak=False)
        print("weak-guard     : OK (nu s-a considerat slabă)")
    except Exception as e:
        print("weak-guard     :", e)

    print("V1 verify(demo):", verify_checksum_v1(demo, 1))
    print("V2 verify(demo):", verify_checksum_v2(demo))

    export_spec_json("akm_spec.json", include_vectors=True)
    print("spec.json scris în:", os.path.abspath("akm_spec.json"))
