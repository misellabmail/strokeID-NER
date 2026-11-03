"""Rebuild Table 6 (joint vs. binary NER, XLM-RoBERTa-large) from multi-seed runs.

Joint  = runs/m3/seed*/      (one 9-label model, per-entity F1 read from it)
Binary = runs/m3_bin-X/seed* (one 3-label model per entity type X)
Both use identical settings and the same seeds, so the comparison is paired by seed.

Usage:  python table6.py --runs runs [--joint m3]
"""
import argparse
import glob
import json
import os

import numpy as np

from common import ENTITY_TYPES


def per_seed(path_glob, key):
    out = {}
    for p in sorted(glob.glob(path_glob)):
        m = json.load(open(p))
        out[m["seed"]] = m["test"]["per_entity"][key]
    return out


def fmt(v):
    v = np.array(v)
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--joint", default="m3")
    a = ap.parse_args()
    rows, joint_macro, bin_macro = [], {}, {}
    print("| Entity | Joint F1 | Binary F1 | Δ (binary − joint) | seeds |")
    print("|---|---|---|---|---|")
    for t in ENTITY_TYPES:
        j = per_seed(os.path.join(a.runs, a.joint, "seed*", "metrics.json"), t)
        b = per_seed(os.path.join(a.runs, f"{a.joint}_bin-{t}", "seed*", "metrics.json"), t)
        seeds = sorted(set(j) & set(b))
        if not seeds:
            print(f"| {t} | missing runs | | | |"); continue
        d = [b[s] - j[s] for s in seeds]
        for s in seeds:
            joint_macro.setdefault(s, []).append(j[s]); bin_macro.setdefault(s, []).append(b[s])
        print(f"| {t} | {fmt([j[s] for s in seeds])} | {fmt([b[s] for s in seeds])} | {fmt(d)} | {len(seeds)} |")
    full = [s for s in joint_macro if len(joint_macro[s]) == len(ENTITY_TYPES)]
    if full:
        jm = [np.mean(joint_macro[s]) for s in full]; bm = [np.mean(bin_macro[s]) for s in full]
        print(f"| **Macro** | {fmt(jm)} | {fmt(bm)} | {fmt(np.array(bm) - np.array(jm))} | {len(full)} |")
    print("\nJoint and binary numbers come from the same seeds and settings; Δ is mean ± SD of the per-seed difference.")


if __name__ == "__main__":
    main()
