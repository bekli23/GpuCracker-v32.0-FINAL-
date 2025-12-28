#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_examples.py — Generează exemple aleatorii + cazuri semi-fixe pentru AKM words (512)

Moduri:
  • Exemple complet aleatorii pentru lungimi 3,5,10,15,20,25,30
  • Cazuri semi-fixe: toate "mare" + ultimul cuvânt aleatoriu
  • Include regula specială ['mare']*5 -> 0x..01

CLI:
  python akm_examples.py
  python akm_examples.py --per-length 3
  python akm_examples.py --no-special
"""

import argparse
import random
from typing import List
from akm_words_512 import WORDS, phrase_to_key, assert_in_list

DEFAULT_LENGTHS = [3, 5, 10, 15, 20, 25, 30]

def gen_phrase(rng: random.Random, length: int) -> List[str]:
    return [rng.choice(WORDS) for _ in range(length)]

def gen_mare_variant(rng: random.Random, length: int) -> List[str]:
    """Fraza formată din 'mare' + ultimul cuvânt aleator."""
    if length <= 1:
        return ['mare']
    last = rng.choice([w for w in WORDS if w != 'mare'])
    return ['mare'] * (length - 1) + [last]

def main():
    ap = argparse.ArgumentParser(description="Exemple aleatorii + cazuri semi-fixe pentru AKM wordlist")
    ap.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS,
                    help="Lungimi (implicit 3 5 10 15 20 25 30)")
    ap.add_argument("--per-length", type=int, default=3,
                    help="Câte exemple aleatorii per lungime (default 3)")
    ap.add_argument("--seed", type=int, default=42, help="Seed RNG (default 42)")
    ap.add_argument("--no-special", action="store_true", help="Omită cazul special ['mare']*5")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print(f"WORDS={len(WORDS)} | lengths={args.lengths} | per-length={args.per_length} | seed={args.seed}")

    # exemplul special
    if not args.no_special:
        special = ["mare"] * 5
        try:
            assert_in_list(special)
            k = phrase_to_key(special).hex()
            print("\n[SPECIAL] L=5  phrase:", " ".join(special))
            print("[SPECIAL] key   :", k)
        except Exception as e:
            print("[SPECIAL] eroare:", e)

    # rulare pe toate lungimile
    for L in args.lengths:
        print(f"\n=== L={L} ===")

        # exemplu fix: toate "mare" + 1 aleatoriu
        variant = gen_mare_variant(rng, L)
        try:
            key = phrase_to_key(variant).hex()
            print(f"  [FIXED] phrase: {' '.join(variant)}")
            print(f"          key:    {key}")
        except Exception as e:
            print(f"  [FIXED] eroare: {e}")

        # exemple complet aleatorii
        for i in range(1, args.per_length + 1):
            phrase = gen_phrase(rng, L)
            try:
                assert_in_list(phrase)
                key_hex = phrase_to_key(phrase).hex()
                print(f"  [{i}] phrase: {' '.join(phrase)}")
                print(f"      key:   {key_hex}")
            except Exception as e:
                print(f"  [{i}] eroare: {e}")

if __name__ == "__main__":
    main()
