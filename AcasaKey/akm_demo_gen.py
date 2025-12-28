#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
akm_demo_gen.py — Demo generator aleatoriu + viteză + exemple fraze/privkey

- Afișează fraze aleatorii + cheia privată (256-bit hex) pentru exemple.
- Opțiuni:
    --examples N     : câte exemple să afișeze per lungime (default 5)
    --print-all      : printează toate frazele generate (o să curgă mult text)
    --out-csv FILE   : salvează toate frazele + cheile în CSV
- Restul opțiunilor sunt aceleași: --lengths, --per-length, --seconds, --no-derive, --interval, --seed
"""

import argparse
import csv
import random
import time
import sys
from typing import List, Tuple, Iterable

from akm_words_512 import WORDS, phrase_to_key

DEFAULT_LENGTHS = [3, 5, 10, 15, 20, 25, 30]

# ---------- utilitare ----------

def gen_phrase(rng: random.Random, length: int) -> List[str]:
    return [rng.choice(WORDS) for _ in range(length)]

def human(n: float) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n/1_000:.2f}k"
    return f"{n:.2f}"

_UNITS = [
    ("ZH/s", 10**21),
    ("EH/s", 10**18),
    ("PH/s", 10**15),
    ("TH/s", 10**12),
    ("GH/s", 10**9),
    ("MH/s", 10**6),
    ("kH/s", 10**3),
    ("H/s",  1),
]

def hashrate(n: float) -> Tuple[str, str]:
    for unit, scale in _UNITS:
        v = n / scale
        if v >= 1:
            return (f"{v:.2f}", unit)
    return (f"{n:.2f}", "H/s")

def phrase_to_hexkey(phrase: Iterable[str]) -> str:
    """Wrapper: phrase_to_key returns bytes (32); convert to hex."""
    b = phrase_to_key(list(phrase))
    return b.hex()

# ---------- runner cu output exemplar ----------

def run_and_collect(rng: random.Random, length: int, count: int, derive: bool, interval: float,
                    examples: int, print_all: bool, csv_writer):
    """
    Rulează count pași (generare frază, eventual derivare cheie).
    Returnează: done_phr, done_key, elapsed
    Dacă csv_writer este furnizat, scrie toate rândurile în CSV.
    Dacă print_all True, printează fiecare frază + key pe măsură ce apar.
    În mod normal, păstrează doar primele `examples` pentru afișare finală.
    """
    start = time.time()
    last_report = start
    done_phr = 0
    done_key = 0
    shown_examples = 0
    sample_examples = []  # (phrase_str, key_hex or "")
    try:
        for i in range(count):
            phrase = gen_phrase(rng, length)
            done_phr += 1
            key_hex = ""
            if derive:
                key_hex = phrase_to_hexkey(phrase)
                done_key += 1
            phrase_str = " ".join(phrase)

            # scriere CSV full (dacă utilizatorul a cerut)
            if csv_writer is not None:
                csv_writer.writerow([length, phrase_str, key_hex])

            # print fiecare (dacă cerut)
            if print_all:
                if derive:
                    print(f"[L={length}] {phrase_str} -> {key_hex}")
                else:
                    print(f"[L={length}] {phrase_str}")

            # salvează primele few example pentru afișare compactă ulterior
            if shown_examples < examples:
                sample_examples.append((phrase_str, key_hex))
                shown_examples += 1

            now = time.time()
            if now - last_report >= interval:
                dt = now - start
                phr_rate = done_phr / dt if dt > 0 else 0.0
                key_rate = done_key / dt if dt > 0 else 0.0
                if derive:
                    kv, ku = hashrate(key_rate)
                    line = f"[L={length}] {done_phr}/{count} phrases | {human(phr_rate)} phr/s | {human(key_rate)} key/s ({kv} {ku})"
                else:
                    line = f"[L={length}] {done_phr}/{count} phrases | {human(phr_rate)} phr/s"
                print("\r" + line + " " * 20, end="", flush=True)
                last_report = now

        # final report
        dt = time.time() - start
        phr_rate = done_phr / dt if dt > 0 else 0.0
        key_rate = done_key / dt if dt > 0 else 0.0
        if derive:
            kv, ku = hashrate(key_rate)
            line = f"[L={length}] {done_phr}/{count} phrases | {human(phr_rate)} phr/s | {human(key_rate)} key/s ({kv} {ku})"
        else:
            line = f"[L={length}] {done_phr}/{count} phrases | {human(phr_rate)} phr/s"
        print("\r" + line + " " * 20)
        return done_phr, done_key, dt, sample_examples
    except KeyboardInterrupt:
        dt = time.time() - start
        phr_rate = done_phr / dt if dt > 0 else 0.0
        key_rate = done_key / dt if dt > 0 else 0.0
        print("\r" + f"[L={length}] INT {done_phr}/{count} | {human(phr_rate)} phr/s" + " " * 20)
        return done_phr, done_key, dt, sample_examples

def run_and_collect_seconds(rng: random.Random, length: int, seconds: float, derive: bool, interval: float,
                            examples: int, print_all: bool, csv_writer):
    start = time.time()
    end_time = start + seconds
    last_report = start
    done_phr = 0
    done_key = 0
    shown_examples = 0
    sample_examples = []
    try:
        while True:
            now = time.time()
            if now >= end_time:
                break
            phrase = gen_phrase(rng, length)
            done_phr += 1
            key_hex = ""
            if derive:
                key_hex = phrase_to_hexkey(phrase)
                done_key += 1
            phrase_str = " ".join(phrase)

            if csv_writer is not None:
                csv_writer.writerow([length, phrase_str, key_hex])

            if print_all:
                if derive:
                    print(f"[L={length}] {phrase_str} -> {key_hex}")
                else:
                    print(f"[L={length}] {phrase_str}")

            if shown_examples < examples:
                sample_examples.append((phrase_str, key_hex))
                shown_examples += 1

            if now - last_report >= interval:
                dt = now - start
                phr_rate = done_phr / dt if dt > 0 else 0.0
                key_rate = done_key / dt if dt > 0 else 0.0
                if derive:
                    kv, ku = hashrate(key_rate)
                    line = f"[L={length}] {done_phr} phrases | t={dt:.1f}s rem={max(0.0,end_time-now):.1f}s | {human(phr_rate)} phr/s | {human(key_rate)} key/s ({kv} {ku})"
                else:
                    line = f"[L={length}] {done_phr} phrases | t={dt:.1f}s rem={max(0.0,end_time-now):.1f}s | {human(phr_rate)} phr/s"
                print("\r" + line + " " * 8, end="", flush=True)
                last_report = now

        dt = time.time() - start
        phr_rate = done_phr / dt if dt > 0 else 0.0
        key_rate = done_key / dt if dt > 0 else 0.0
        if derive:
            kv, ku = hashrate(key_rate)
            line = f"[L={length}] {done_phr} phrases | t={dt:.1f}s | {human(phr_rate)} phr/s | {human(key_rate)} key/s ({kv} {ku})"
        else:
            line = f"[L={length}] {done_phr} phrases | t={dt:.1f}s | {human(phr_rate)} phr/s"
        print("\r" + line + " " * 12)
        return done_phr, done_key, dt, sample_examples
    except KeyboardInterrupt:
        dt = time.time() - start
        print("\r" + f"[L={length}] INT {done_phr} | t={dt:.1f}s" + " " * 12)
        return done_phr, done_key, dt, sample_examples

# ---------- CLI / main ----------

def main():
    ap = argparse.ArgumentParser(description="Demo generator aleatoriu + viteză + exemple (fraze + privkey)")
    ap.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS,
                    help="Lista de lungimi (implicit 3 5 10 15 20 25 30)")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--per-length", type=int, default=10000,
                       help="Număr de fraze per lungime (default 10000)")
    group.add_argument("--seconds", type=float, default=None,
                       help="Durata per lungime (secunde). Suprascrie --per-length.")
    ap.add_argument("--seed", type=int, default=42, help="Seed RNG (default 42)")
    ap.add_argument("--no-derive", action="store_true", help="Nu derivă chei (doar fraze)")
    ap.add_argument("--interval", type=float, default=1.0, help="Interval status live (sec)")
    ap.add_argument("--examples", type=int, default=5, help="Câte exemple să afișeze per lungime (default 5)")
    ap.add_argument("--print-all", action="store_true", help="Printează toate frazele generate (foarte mult output)")
    ap.add_argument("--out-csv", default=None, help="Salvează toate frazele+cheile în CSV (path)")

    args = ap.parse_args()
    for L in args.lengths:
        if L <= 0:
            print(f"Lungime invalidă: {L}", file=sys.stderr)
            sys.exit(2)

    rng = random.Random(args.seed)
    derive = not args.no_derive

    csv_file = None
    csv_writer = None
    if args.out_csv:
        csv_file = open(args.out_csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["length", "phrase", "privkey_hex"])

    print(f"Using WORDS=512 | lengths={args.lengths} | mode={'seconds='+str(args.seconds) if args.seconds else 'count='+str(args.per_length)} | derive={'ON' if derive else 'OFF'}")
    total_phr = 0
    total_key = 0
    total_time = 0.0

    # stocăm exemplele per lungime pentru afișare compactă
    examples_by_len = {}

    try:
        for L in args.lengths:
            if args.seconds is not None:
                p, k, dt, samples = run_and_collect_seconds(rng, L, args.seconds, derive, args.interval, args.examples, args.print_all, csv_writer)
            else:
                p, k, dt, samples = run_and_collect(rng, L, args.per_length, derive, args.interval, args.examples, args.print_all, csv_writer)
            total_phr += p
            total_key += k
            total_time += dt
            examples_by_len[L] = samples

    finally:
        if csv_file:
            csv_file.close()

    if total_time <= 0:
        total_time = 1e-9
    phr_rate = total_phr / total_time
    key_rate = total_key / total_time if derive else 0.0

    print("-" * 80)
    if derive:
        kv, ku = hashrate(key_rate)
        print(f"TOTAL: phrases={total_phr} in {total_time:.2f}s -> {human(phr_rate)} phr/s | {human(key_rate)} key/s ({kv} {ku})")
    else:
        print(f"TOTAL: phrases={total_phr} in {total_time:.2f}s -> {human(phr_rate)} phr/s")

    # afișăm exemple compacte
    print("\nExamples (first {} per length):".format(args.examples))
    for L in args.lengths:
        samples = examples_by_len.get(L, [])
        print(f"\n[L={L}]")
        if not samples:
            print("  (no samples)")
            continue
        for phr, key in samples:
            if derive:
                print(f"  {phr}  ->  {key}")
            else:
                print(f"  {phr}")

if __name__ == "__main__":
    main()
