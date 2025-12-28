#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_words_512_extra.py — Extra profiles for AKM word engine.

Acesta este fișierul de joacă pentru:
  - CUSTOM_HEX
  - SPECIAL_RULES
  - AKM_PARAMS (version tag, policy checksum, etc.)

Core:
  - akm_words_512.py definește profilul stabil "akm2-core".
  - Aici definim AKM_EXTRA_PROFILES = { nume_profil: dict_spec }.
  - Un profil poate conține:
        "AKM_PARAMS"   : dict (override parțial peste AKM_PARAMS core)
        "CUSTOM_HEX"   : dict (override / extra pentru maparea word -> hex)
        "SPECIAL_RULES": dict (override / extra pentru reguli speciale:
                               fixed8, fixed1, fixed2, fixed3, pad_nibble,
                               repeat_last etc.)

Cum se aplică (vezi akm_words_512._apply_profile):
  - "akm2-core"       => revine la CORE_AKM_PARAMS / CORE_CUSTOM_HEX / CORE_SPECIAL_RULES
  - alt nume profil   => pornește de la core și aplică override din acest fișier

Selectare profil:
  - din cod:
        import akm_words_512 as akm
        akm.set_profile("akm2-fixed123-pack-v1")
  - sau prin CLI la akm_seed2priv.py / akm_genaddr.py:
        --profile akm2-fixed123-pack-v1

IMPORTANT:
  - Nu modifica direct core-ul din akm_words_512.py dacă vrei compatibilitate
    pe termen lung. Adaugă experimente aici ca profile noi.
"""

from __future__ import annotations
from typing import List

# ======================================================================
#  AKM_EXTRA_PROFILES: toate profilele custom
# ======================================================================

AKM_EXTRA_PROFILES = {
    # ================================================================
    # 1) akm2-lab-v1 — profil mic de laborator
    #    - schimbă version tag: "AKM2-lab-v1"
    #    - adaugă câteva CUSTOM_HEX
    #    - adaugă SPECIAL_RULES simple
    # ================================================================
    "akm2-lab-v1": {
        "AKM_PARAMS": {
            "version": "AKM2-lab-v1",
            # Dacă vrei, poți suprascrie și cs aici:
            # "cs": {"v1": True, "v2": True, "v2_words": 2, "v2_bits_per_word": 10},
        },

        "CUSTOM_HEX": {
            # Cuvintele sunt normalizate în akm_words_512 (lowercase, fără diacritice).
            "jocul": "dead",     # doar exemplu vizual
            "ochiuri": "face",   # simplu de recunoscut în hex
            "privire": "0e0e",
        },

        "SPECIAL_RULES": {
            # fixed8: token-ul hex devine EXACT valoarea pusă aici (8 hex).
            "soarele": {
                "fixed8": "cafebabe"
            },
            # pad_nibble: extinde base4 până la 8 hex cu nibble-ul dat.
            "norocel": {
                "pad_nibble": "7"
            },
            # repeat_last: extinde base4 până la 8 hex repetând ultimul nibble.
            "paradis": {
                "repeat_last": True
            },
        },
    },

    # ================================================================
    # 2) akm2-lab-strict-cs — profil cu politică de checksum diferită
    #    - schimbă doar AKM_PARAMS["version"] și AKM_PARAMS["cs"]
    #    - nu umblă la CUSTOM_HEX / SPECIAL_RULES (ramân ca în core)
    # ================================================================
    "akm2-lab-strict-cs": {
        "AKM_PARAMS": {
            "version": "AKM2-lab-strict-cs",
            "cs": {
                "v1": False,
                "v2": True,
                "v2_words": 3,
                "v2_bits_per_word": 10,
            },
        },
        # fără CUSTOM_HEX / SPECIAL_RULES => se folosește core
    },

    # ================================================================
    # 3) akm2-fixed123-pack-v1 — profil TEMPLATE pentru fixed1 / fixed2 / fixed3
    #
    # Ideea:
    #   - anumite cuvinte au token-ul forțat cu:
    #       "fixed1": "a"   # EXACT "a" în hex
    #       "fixed2": "12"  # EXACT "12"
    #       "fixed3": "123" # EXACT "123"
    #   - nu se face auto-completare la 8 hex; valoarea este folosită direct.
    #
    # Recomandare:
    #   - FOLosești fixed1/fixed2/fixed3 doar pentru cuvinte "marker",
    #     restul entropiei vine din celelalte cuvinte (HKDF/hybrid).
    #
    # Cum îl folosești:
    #   - clonezi acest profil, schimbi numele și listele de cuvinte /
    #     hex-urile din CUSTOM_HEX și SPECIAL_RULES.
    # ================================================================
    "akm2-fixed123-pack-v1": {
        "AKM_PARAMS": {
            "version": "AKM2-fixed123-pack-v1",
            # Dacă vrei alt policy de checksum aici, poți modifica:
            # "cs": {"v1": True, "v2": True, "v2_words": 2, "v2_bits_per_word": 10},
        },

        # ----------------------------------------------------------------
        # AICI MODIFICI / EXTINZI CUVINTELE CU fixed1 / fixed2 / fixed3
        # ----------------------------------------------------------------
        #
        # NOTĂ:
        #   - CUSTOM_HEX are prioritate absolută: dacă pui "mare": "00ff",
        #     atunci explain_token("mare") va porni de la hex-ul ăsta.
        #   - SPECIAL_RULES cu fixed1/fixed2/fixed3 se aplică DOAR dacă
        #     cuvântul NU are override în CUSTOM_HEX.
        #
        # Ideea pentru "canibalizare":
        #   - CUSTOM_HEX îți dă un început specific (ex: "00", "12", "abc"),
        #   - SPECIAL_RULES fixed1/2/3 îți pot forța EXACT acea valoare
        #     ca token principal (nu token8 complet ca la fixed8).
        # ----------------------------------------------------------------
        "CUSTOM_HEX": {
            # exemplu: cuvinte-ancoră pentru fixed1/fixed2/fixed3
            # (poți schimba TOT ce e mai jos, e doar schemă)
            "abis": "CB46",
            "acelasi": "12",
            "acoperis": "123",
            "adanc": "b",
            "adapost": "34",
            "adorare": "345",
            "aer": "c",
            "ajun": "56",
            "albastru": "567",
            "alb": "d",
            "alge": "89",
            "altar": "89a",
            "amintire": "e",
            "amurg": "cd",
            "anotimp": "cde",
            "apa": "f",
            "apele": "01",
            "apus": "012",
            "apuseni": "02",
            "arta": "234",
            # etc... aici poți continua până la ~50 cuvinte dacă vrei.
        },

        "SPECIAL_RULES": {
            # ----------------------------------------------------------------
            # fixed1: token-ul hex devine EXACT un singur nibble, ex: "a"
            # ----------------------------------------------------------------
            "abis": {
                "fixed1": "CB46",      # token = "a"
            },
            "adanc": {
                "fixed1": "b",      # token = "b"
            },
            "amintire": {
                "fixed1": "e",      # token = "e"
            },
            "bujor": {
                "fixed3": "000",      # token = "f"
            },

            # ----------------------------------------------------------------
            # fixed2: token-ul hex devine EXACT 2 nibble-uri, ex: "12"
            # ----------------------------------------------------------------
            "acelasi": {
                "fixed2": "12",     # token = "12"
            },
            "ajun": {
                "fixed2": "56",     # token = "56"
            },
            "apele": {
                "fixed2": "f1",     # token = "01"
            },
            "apuseni": {
                "fixed2": "d2",     # token = "02"
            },

            # ----------------------------------------------------------------
            # fixed3: token-ul hex devine EXACT 3 nibble-uri, ex: "123"
            # ----------------------------------------------------------------
            "acoperis": {
                "fixed3": "25E",    # token = "123"
            },
            "adorare": {
                "fixed3": "345",    # token = "345"
            },
            "albastru": {
                "fixed3": "567",    # token = "567"
            },
            "altar": {
                "fixed3": "89a",    # token = "89a"
            },
            "anotimp": {
                "fixed3": "cde",    # token = "cde"
            },
            "arta": {
                "fixed3": "234",    # token = "234"
            },

            # ----------------------------------------------------------------
            # poți combina cu fixed8 pentru câteva cuvinte speciale
            # (caz extrem: override total cu 8 hex)
            # ----------------------------------------------------------------
            "ochiuri": {
                "fixed8": "facefeed"
            },
            "privire": {
                "fixed8": "0e0e0e0e"
            },
        },
    },
    
    "akm3-experimental-v1": {
         "AKM_PARAMS": {
             "version": "AKM3-experimental-v1",
             # poți schimba și altele:
             # "token": "hexpack8",
             # "cs": {"v1": True, "v2": False, "v2_words": 0, "v2_bits_per_word": 10},
            
         },
         "CUSTOM_HEX": {
             # "mare": "f00d",
             # "abis": "0bad",
         },
         "SPECIAL_RULES": {
             # "apele": {"fixed8": "00001111"},
             # "lac": {"fixed2": "12"},
            # "nor": {"fixed3": "abc"},
             "abis": {
                "fixed8": "7CB46000",      # token = "a"
            },
         },
     },
     
   "akm3-puzzle71": {
    "AKM_PARAMS": {
        "version": "AKM3-puzzle71",
        # checksum/policy le setezi tu dacă ai nevoie
        # "cs": {"v1": True, "v2": True, "v2_words": 2, "v2_bits_per_word": 10},
    },

    "CUSTOM_HEX": {
        "abis": "100",
        "acelasi": "101",
        "acoperis": "102",
        "adanc": "103",
        "adapost": "104",
        "adorare": "105",
        "afectiune": "106",
        "aer": "107",
        "ajun": "108",
        "albastru": "109",
        "alb": "10a",
        "alge": "10b",
        "altar": "10c",
        "amintire": "10d",
        "amurg": "10e",
        "anotimp": "10f",
        "apa": "110",
        "apele": "111",
        "apus": "112",
        "apuseni": "113",
        "apusor": "114",
        "aroma": "115",
        "artar": "116",
        "arta": "117",
        "asfintit": "118",
        "asteptare": "119",
        "atingere": "11a",
        "aur": "11b",
        "aurora": "11c",
        "autumn": "11d",
        "avion": "11e",
        "balta": "11f",
        "barca": "120",
        "barza": "121",
        "baterie": "122",
        "batran": "123",
        "bec": "124",
        "bezna": "125",
        "binecuvantare": "126",
        "blandete": "127",
        "boboc": "128",
        "bogatie": "129",
        "bolta": "12a",
        "brad": "12b",
        "brat": "12c",
        "bruma": "12d",
        "bucata": "12e",
        "bucurie": "12f",
        "bujor": "130",
        "burg": "131",
        "camp": "132",
        "campie": "133",
        "cafea": "134",
        "calator": "135",
        "cald": "136",
        "candela": "137",
        "caprioara": "138",
        "caramida": "139",
        "carare": "13a",
        "carte": "13b",
        "catel": "13c",
        "cautare": "13d",
        "casa": "13e",
        "ceas": "13f",
        "cer": "140",
        "cerb": "141",
        "chip": "142",
        "ciocarlie": "143",
        "ciutura": "144",
        "clar": "145",
        "clipa": "146",
        "clopot": "147",
        "coborare": "148",
        "colina": "149",
        "colt": "14a",
        "copac": "14b",
        "copil": "14c",
        "corabie": "14d",
        "cord": "14e",
        "corn": "14f",
        "crang": "150",
        "credinta": "151",
        "crestere": "152",
        "crestet": "153",
        "crin": "154",
        "cuc": "155",
        "cufar": "156",
        "culoare": "157",
        "culme": "158",
        "curcubeu": "159",
        "curte": "15a",
        "cupru": "15b",
        "cuvant": "15c",
        "cutie": "15d",
        "daurire": "15e",
        "deal": "15f",
        "deget": "160",
        "delusor": "161",
        "departare": "162",
        "desert": "163",
        "dimineata": "164",
        "dor": "165",
        "dorinta": "166",
        "drag": "167",
        "draga": "168",
        "drum": "169",
        "drumet": "16a",
        "durere": "16b",
        "duminica": "16c",
        "ecou": "16d",
        "efemer": "16e",
        "elixir": "16f",
        "emisfera": "170",
        "enigma": "171",
        "eter": "172",
        "eternitate": "173",
        "fag": "174",
        "fagure": "175",
        "fantana": "176",
        "farmec": "177",
        "fata": "178",
        "felinar": "179",
        "fenic": "17a",
        "fereastra": "17b",
        "fericire": "17c",
        "feriga": "17d",
        "fier": "17e",
        "fierar": "17f",
        "film": "180",
        "fior": "181",
        "flacara": "182",
        "flamura": "183",
        "floare": "184",
        "fluture": "185",
        "fosnet": "186",
        "fotografie": "187",
        "frag": "188",
        "frate": "189",
        "frezie": "18a",
        "frig": "18b",
        "fruct": "18c",
        "frumusete": "18d",
        "frunza": "18e",
        "frunte": "18f",
        "fulger": "190",
        "furnica": "191",
        "galaxie": "192",
        "galben": "193",
        "gand": "194",
        "gandire": "195",
        "garoafa": "196",
        "gheata": "197",
        "ghetar": "198",
        "ghinda": "199",
        "ghiozdan": "19a",
        "glas": "19b",
        "glorie": "19c",
        "grad": "19d",
        "gradina": "19e",
        "grai": "19f",
        "granita": "1a0",
        "gust": "1a1",
        "gura": "1a2",
        "har": "1a3",
        "harfa": "1a4",
        "iarba": "1a5",
        "iarna": "1a6",
        "icoana": "1a7",
        "implinire": "1a8",
        "inger": "1a9",
        "insula": "1aa",
        "insorire": "1ab",
        "intindere": "1ac",
        "intuneric": "1ad",
        "inviere": "1ae",
        "iubire": "1af",
        "iz": "1b0",
        "izvor": "1b1",
        "izvoras": "1b2",
        "joc": "1b3",
        "jocul": "1b4",
        "lac": "1b5",
        "lacrima": "1b6",
        "laur": "1b7",
        "lebada": "1b8",
        "legenda": "1b9",
        "lemn": "1ba",
        "leu": "1bb",
        "libertate": "1bc",
        "linie": "1bd",
        "livada": "1be",
        "loc": "1bf",
        "luna": "1c0",
        "lumina": "1c1",
        "lume": "1c2",
        "lunca": "1c3",
        "lup": "1c4",
        "lut": "1c5",
        "manunchi": "1c6",
        "margine": "1c7",
    },

    "SPECIAL_RULES": {
        "abis": {"fixed1": "0"},
        "acelasi": {"fixed2": "01"},
        "acoperis": {"fixed3": "102"},
        "adanc": {"fixed1": "3"},
        "adapost": {"fixed2": "04"},
        "adorare": {"fixed3": "105"},
        "afectiune": {"fixed1": "6"},
        "aer": {"fixed2": "07"},
        "ajun": {"fixed3": "108"},
        "albastru": {"fixed1": "9"},
        "alb": {"fixed2": "0a"},
        "alge": {"fixed3": "10b"},
        "altar": {"fixed1": "c"},
        "amintire": {"fixed2": "0d"},
        "amurg": {"fixed3": "10e"},
        "anotimp": {"fixed1": "f"},
        "apa": {"fixed2": "10"},
        "apele": {"fixed3": "111"},
        "apus": {"fixed1": "2"},
        "apuseni": {"fixed2": "13"},
        "apusor": {"fixed3": "114"},
        "aroma": {"fixed1": "5"},
        "artar": {"fixed2": "16"},
        "arta": {"fixed3": "117"},
        "asfintit": {"fixed1": "8"},
        "asteptare": {"fixed2": "19"},
        "atingere": {"fixed3": "11a"},
        "aur": {"fixed1": "b"},
        "aurora": {"fixed2": "1c"},
        "autumn": {"fixed3": "11d"},
        "avion": {"fixed1": "e"},
        "balta": {"fixed2": "1f"},
        "barca": {"fixed3": "120"},
        "barza": {"fixed1": "1"},
        "baterie": {"fixed2": "22"},
        "batran": {"fixed3": "123"},
        "bec": {"fixed1": "4"},
        "bezna": {"fixed2": "25"},
        "binecuvantare": {"fixed3": "126"},
        "blandete": {"fixed1": "7"},
        "boboc": {"fixed2": "28"},
        "bogatie": {"fixed3": "129"},
        "bolta": {"fixed1": "a"},
        "brad": {"fixed2": "2b"},
        "brat": {"fixed3": "12c"},
        "bruma": {"fixed1": "d"},
        "bucata": {"fixed2": "2e"},
        "bucurie": {"fixed3": "12f"},
        "bujor": {"fixed1": "0"},
        "burg": {"fixed2": "31"},
        "camp": {"fixed3": "132"},
        "campie": {"fixed1": "3"},
        "cafea": {"fixed2": "34"},
        "calator": {"fixed3": "135"},
        "cald": {"fixed1": "6"},
        "candela": {"fixed2": "37"},
        "caprioara": {"fixed3": "138"},
        "caramida": {"fixed1": "9"},
        "carare": {"fixed2": "3a"},
        "carte": {"fixed3": "13b"},
        "catel": {"fixed1": "c"},
        "cautare": {"fixed2": "3d"},
        "casa": {"fixed3": "13e"},
        "ceas": {"fixed1": "f"},
        "cer": {"fixed2": "40"},
        "cerb": {"fixed3": "141"},
        "chip": {"fixed1": "2"},
        "ciocarlie": {"fixed2": "43"},
        "ciutura": {"fixed3": "144"},
        "clar": {"fixed1": "5"},
        "clipa": {"fixed2": "46"},
        "clopot": {"fixed3": "147"},
        "coborare": {"fixed1": "8"},
        "colina": {"fixed2": "49"},
        "colt": {"fixed3": "14a"},
        "copac": {"fixed1": "b"},
        "copil": {"fixed2": "4c"},
        "corabie": {"fixed3": "14d"},
        "cord": {"fixed1": "e"},
        "corn": {"fixed2": "4f"},
        "crang": {"fixed3": "150"},
        "credinta": {"fixed1": "1"},
        "crestere": {"fixed2": "52"},
        "crestet": {"fixed3": "153"},
        "crin": {"fixed1": "4"},
        "cuc": {"fixed2": "55"},
        "cufar": {"fixed3": "156"},
        "culoare": {"fixed1": "7"},
        "culme": {"fixed2": "58"},
        "curcubeu": {"fixed3": "159"},
        "curte": {"fixed1": "a"},
        "cupru": {"fixed2": "5b"},
        "cuvant": {"fixed3": "15c"},
        "cutie": {"fixed1": "d"},
        "daurire": {"fixed2": "5e"},
        "deal": {"fixed3": "15f"},
        "deget": {"fixed1": "0"},
        "delusor": {"fixed2": "61"},
        "departare": {"fixed3": "162"},
        "desert": {"fixed1": "3"},
        "dimineata": {"fixed2": "64"},
        "dor": {"fixed3": "165"},
        "dorinta": {"fixed1": "6"},
        "drag": {"fixed2": "67"},
        "draga": {"fixed3": "168"},
        "drum": {"fixed1": "9"},
        "drumet": {"fixed2": "6a"},
        "durere": {"fixed3": "16b"},
        "duminica": {"fixed1": "c"},
        "ecou": {"fixed2": "6d"},
        "efemer": {"fixed3": "16e"},
        "elixir": {"fixed1": "f"},
        "emisfera": {"fixed2": "70"},
        "enigma": {"fixed3": "171"},
        "eter": {"fixed1": "2"},
        "eternitate": {"fixed2": "73"},
        "fag": {"fixed3": "174"},
        "fagure": {"fixed1": "5"},
        "fantana": {"fixed2": "76"},
        "farmec": {"fixed3": "177"},
        "fata": {"fixed1": "8"},
        "felinar": {"fixed2": "79"},
        "fenic": {"fixed3": "17a"},
        "fereastra": {"fixed1": "b"},
        "fericire": {"fixed2": "7c"},
        "feriga": {"fixed3": "17d"},
        "fier": {"fixed1": "e"},
        "fierar": {"fixed2": "7f"},
        "film": {"fixed3": "180"},
        "fior": {"fixed1": "1"},
        "flacara": {"fixed2": "82"},
        "flamura": {"fixed3": "183"},
        "floare": {"fixed1": "4"},
        "fluture": {"fixed2": "85"},
        "fosnet": {"fixed3": "186"},
        "fotografie": {"fixed1": "7"},
        "frag": {"fixed2": "88"},
        "frate": {"fixed3": "189"},
        "frezie": {"fixed1": "a"},
        "frig": {"fixed2": "8b"},
        "fruct": {"fixed3": "18c"},
        "frumusete": {"fixed1": "d"},
        "frunza": {"fixed2": "8e"},
        "frunte": {"fixed3": "18f"},
        "fulger": {"fixed1": "0"},
        "furnica": {"fixed2": "91"},
        "galaxie": {"fixed3": "192"},
        "galben": {"fixed1": "3"},
        "gand": {"fixed2": "94"},
        "gandire": {"fixed3": "195"},
        "garoafa": {"fixed1": "6"},
        "gheata": {"fixed2": "97"},
        "ghetar": {"fixed3": "198"},
        "ghinda": {"fixed1": "9"},
        "ghiozdan": {"fixed2": "9a"},
        "glas": {"fixed3": "19b"},
        "glorie": {"fixed1": "c"},
        "grad": {"fixed2": "9d"},
        "gradina": {"fixed3": "19e"},
        "grai": {"fixed1": "f"},
        "granita": {"fixed2": "a0"},
        "gust": {"fixed3": "1a1"},
        "gura": {"fixed1": "2"},
        "har": {"fixed2": "a3"},
        "harfa": {"fixed3": "1a4"},
        "iarba": {"fixed1": "5"},
        "iarna": {"fixed2": "a6"},
        "icoana": {"fixed3": "1a7"},
        "implinire": {"fixed1": "8"},
        "inger": {"fixed2": "a9"},
        "insula": {"fixed3": "1aa"},
        "insorire": {"fixed1": "b"},
        "intindere": {"fixed2": "ac"},
        "intuneric": {"fixed3": "1ad"},
        "inviere": {"fixed1": "e"},
        "iubire": {"fixed2": "af"},
        "iz": {"fixed3": "1b0"},
        "izvor": {"fixed1": "1"},
        "izvoras": {"fixed2": "b2"},
        "joc": {"fixed3": "1b3"},
        "jocul": {"fixed1": "4"},
        "lac": {"fixed2": "b5"},
        "lacrima": {"fixed3": "1b6"},
        "laur": {"fixed1": "7"},
        "lebada": {"fixed2": "b8"},
        "legenda": {"fixed3": "1b9"},
        "lemn": {"fixed1": "a"},
        "leu": {"fixed2": "bb"},
        "libertate": {"fixed3": "1bc"},
        "linie": {"fixed1": "d"},
        "livada": {"fixed2": "be"},
        "loc": {"fixed3": "1bf"},
        "luna": {"fixed1": "0"},
        "lumina": {"fixed2": "c1"},
        "lume": {"fixed3": "1c2"},
        "lunca": {"fixed1": "3"},
        "lup": {"fixed2": "c4"},
        "lut": {"fixed3": "1c5"},
        "manunchi": {"fixed1": "6"},
        "margine": {"fixed2": "c7"},
    },
},


    # ================================================================
    # 4) TEMPLATE GOL pentru viitoare profile
    #
    # Copiezi blocul ăsta, redenumești cheia (ex: "akm3-experimental-v1")
    # și modifici câmpurile după cum ai chef.
    # ================================================================
    # how we make one profile for puzzle 71 
    # "akm3-experimental-v1": {
    #     "AKM_PARAMS": {
    #         "version": "AKM3-experimental-v1",
    #         # poți schimba și altele:
    #         # "token": "hexpack8",
    #         # "cs": {"v1": True, "v2": False, "v2_words": 0, "v2_bits_per_word": 10},
    #     },
    #     "CUSTOM_HEX": {
    #         # "mare": "f00d",
    #         # "abis": "0bad",
    #     },
    #     "SPECIAL_RULES": {
    #         # "apele": {"fixed8": "00001111"},
    #         # "lac": {"fixed2": "12"},
    #         # "nor": {"fixed3": "abc"},
    #     },
    # },
}


def list_profiles() -> List[str]:
    """
    Util pentru debugging / introspecție.
    (Nu este folosit direct de motorul core, dar e util dacă imporți
    acest fișier direct sau faci un tool de listare.)
    """
    return sorted(AKM_EXTRA_PROFILES.keys())


if __name__ == "__main__":
    # Mic demo
    print("Available AKM_EXTRA_PROFILES:")
    for name in list_profiles():
        prof = AKM_EXTRA_PROFILES[name]
        print(f" - {name}")
        if "AKM_PARAMS" in prof:
            ver = prof["AKM_PARAMS"].get("version", "<no-version>")
            print(f"    version: {ver}")
        if "CUSTOM_HEX" in prof:
            print(f"    custom_hex entries: {len(prof['CUSTOM_HEX'])}")
        if "SPECIAL_RULES" in prof:
            print(f"    special_rules entries: {len(prof['SPECIAL_RULES'])}")
