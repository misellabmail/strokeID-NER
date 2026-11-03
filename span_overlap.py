"""Recompute Table 7 (span-level overlap, M3 vs GPT) and the cross-model
false-negative overlap from saved predictions, so both analyses use the SAME
runs as the main results table.

Usage
  # Table 7 for one seed / run pair
  python span_overlap.py --a runs/m3/seed42 --b runs/gpt0/seed42
  # every seed of A against every seed of B, then mean
  python span_overlap.py --a runs/m3 --b runs/gpt0 --all_seeds
  # false negatives missed by ALL listed models (seed 42 of each)
  python span_overlap.py --fn runs/m1/seed42 runs/m2/seed42 runs/m4/seed42 runs/m3/seed42
"""
import argparse
import glob
import os
from collections import Counter

from common import ENTITY_TYPES, get_spans, read_predictions


def load(run):
    p = run if run.endswith(".jsonl") else os.path.join(run, "test_predictions.jsonl")
    rows = sorted(read_predictions(p), key=lambda r: str(r["id"]))
    return rows


def overlap(a_rows, b_rows):
    c = Counter(); per = {t: Counter() for t in ENTITY_TYPES}
    for ra, rb in zip(a_rows, b_rows):
        assert ra["id"] == rb["id"] and ra["gold"] == rb["gold"]
        sa, sb = set(get_spans(ra["pred"])), set(get_spans(rb["pred"]))
        for s in get_spans(ra["gold"]):
            key = ("n1" if s in sa else "n0") + ("1" if s in sb else "0")
            c[key] += 1; per[s[0]][key] += 1
    return c, per


def show(c, per, name_a, name_b):
    n = sum(c.values())
    print(f"Gold spans: {n}")
    for k, lab in [("n11", "both correct"), ("n10", f"{name_a} only"), ("n01", f"{name_b} only"), ("n00", "both wrong")]:
        print(f"  {k} {lab:<22} {c[k]:5d}  {100*c[k]/n:5.1f}%")
    print(f"  ratio n10:n01 = {c['n10']/max(c['n01'],1):.1f}:1")
    for t in ENTITY_TYPES:
        p = per[t]
        print(f"    {t}: {name_a} only {p['n10']:4d} | {name_b} only {p['n01']:4d} | ratio {p['n10']/max(p['n01'],1):.1f}:1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--all_seeds", action="store_true")
    ap.add_argument("--fn", nargs="+", help="runs whose shared false negatives to count")
    o = ap.parse_args()
    if o.a and o.b:
        def nm(p):
            parts = p.rstrip("/").split("/")
            return "/".join(parts[-2:]) if parts[-1].startswith("seed") else parts[-1]
        na, nb = nm(o.a), nm(o.b)
        if o.all_seeds:
            tot = Counter(); k = 0
            for sa in sorted(glob.glob(os.path.join(o.a, "seed*"))):
                for sb in sorted(glob.glob(os.path.join(o.b, "seed*"))):
                    c, _ = overlap(load(sa), load(sb)); tot.update(c); k += 1
            print(f"Mean over {k} seed pairs:")
            for key in ("n11", "n10", "n01", "n00"):
                print(f"  {key} {tot[key]/k:.1f}")
        else:
            c, per = overlap(load(o.a), load(o.b)); show(c, per, na, nb)
    if o.fn:
        runs = [load(r) for r in o.fn]
        fn_any, fn_all = 0, 0
        typ_all = Counter()
        for rows in zip(*runs):
            gold = set(get_spans(rows[0]["gold"]))
            missed = [gold - set(get_spans(r["pred"])) for r in rows]
            union, inter = set().union(*missed), set.intersection(*missed)
            fn_any += len(union); fn_all += len(inter)
            for s in inter:
                typ_all[s[0]] += 1
        print(f"False-negative spans missed by at least one model: {fn_any}")
        print(f"Missed by ALL {len(runs)} models: {fn_all} ({100*fn_all/max(fn_any,1):.1f}% of the union)")
        print(f"  by type: {dict(typ_all)}")
        print("  NOTE: state which denominator the paper uses (union of all models' FNs here).")


if __name__ == "__main__":
    main()
