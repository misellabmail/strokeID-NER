"""Seed-averaged results + query-level bootstrap 95% CIs (R2.3).

For every model we
  1. report mean +/- SD of macro P/R/F1 and per-entity F1 over seeds;
  2. resample the 275 test QUERIES with replacement (B times); in each resample
     compute macro F1 for every seed and average over seeds -> percentile CI of
     the seed-averaged macro F1;
  3. for chosen model pairs, compute the paired difference on the same resamples
     -> CI of the difference, and the share of resamples in which the ranking flips.

Usage
  python bootstrap_ci.py --runs runs --models gpt0 gpt5 m1 m2 m4 m4crf m5 m5crf m3 m3crf \
      --pairs m3:m4 m3:m5 m5:m4 m2:m1 m4crf:m4 m3crf:m3 m5crf:m5 m1:gpt5 --B 10000 --out results
Each model directory must contain seed*/test_predictions.jsonl
(GPT runs: one 'seed' directory per prompt/run).
"""
import argparse
import glob
import json
import os

import numpy as np

from common import ENTITY_TYPES, macro_from_totals, per_query_counts, read_predictions


def load_model(run_dir):
    seeds = sorted(glob.glob(os.path.join(run_dir, "seed*", "test_predictions.jsonl")))
    if not seeds:
        raise FileNotFoundError(f"no seed*/test_predictions.jsonl under {run_dir}")
    mats, ids = [], None
    for p in seeds:
        rows = read_predictions(p)
        rows.sort(key=lambda r: str(r["id"]))
        cur = [str(r["id"]) for r in rows]
        if ids is None:
            ids = cur
        assert cur == ids, f"query ids differ across seeds in {run_dir}"
        mats.append(np.array(per_query_counts([r["gold"] for r in rows], [r["pred"] for r in rows])))
    return [os.path.basename(os.path.dirname(p)) for p in seeds], ids, np.stack(mats)  # [S, Q, T, 3]


def macro_f1_batch(counts, idx):
    """counts [S,Q,T,3], idx [B,Q] -> macro F1 [B,S]."""
    summed = counts[:, idx].sum(axis=2)                       # [S,B,T,3]
    tp, npred, ngold = summed[..., 0], summed[..., 1], summed[..., 2]
    p = np.divide(tp, npred, out=np.zeros_like(tp, dtype=float), where=npred > 0)
    r = np.divide(tp, ngold, out=np.zeros_like(tp, dtype=float), where=ngold > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return f.mean(-1).T                                       # [B,S]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    data, ids0 = {}, None
    for m in a.models:
        seeds, ids, c = load_model(os.path.join(a.runs, m))
        if ids0 is None:
            ids0 = ids
        assert ids == ids0, f"{m}: test query ids differ from {a.models[0]}"
        data[m] = (seeds, c)
    Q = len(ids0)
    rng = np.random.default_rng(a.seed)
    idx = rng.integers(0, Q, size=(a.B, Q))

    summary, boot = {}, {}
    for m, (seeds, c) in data.items():
        per_seed = []
        for s in range(c.shape[0]):
            tot = c[s].sum(0)
            P, R, F, per = macro_from_totals(tot[:, 0], tot[:, 1], tot[:, 2])
            per_seed.append(dict(P=P, R=R, F1=F, **dict(zip(ENTITY_TYPES, per))))
        arr = {k: np.array([d[k] for d in per_seed]) for k in per_seed[0]}
        bs = []
        for start in range(0, a.B, 1000):           # chunk to bound memory
            bs.append(macro_f1_batch(c, idx[start:start + 1000]).mean(1))
        bs = np.concatenate(bs)
        boot[m] = bs
        summary[m] = dict(
            seeds=seeds, n_seeds=len(seeds),
            mean={k: float(v.mean()) for k, v in arr.items()},
            sd={k: float(v.std(ddof=1)) if len(v) > 1 else 0.0 for k, v in arr.items()},
            per_seed=per_seed,
            f1_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])

    pairs = {}
    for pr in a.pairs:
        x, y = pr.split(":")
        d = boot[x] - boot[y]
        obs = summary[x]["mean"]["F1"] - summary[y]["mean"]["F1"]
        pairs[pr] = dict(diff=float(obs), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                         share_resamples_reversed=float((d <= 0).mean() if obs > 0 else (d >= 0).mean()))

    json.dump(dict(n_queries=Q, B=a.B, models=summary, pairs=pairs),
              open(os.path.join(a.out, "bootstrap_summary.json"), "w"), indent=2)

    lines = ["| Model | Seeds | P | R | Macro F1 (mean ± SD) | 95% CI (query bootstrap) | " +
             " | ".join(ENTITY_TYPES) + " |", "|" + "---|" * (6 + len(ENTITY_TYPES))]
    for m, s in summary.items():
        mu, sd = s["mean"], s["sd"]
        lines.append(f"| {m} | {s['n_seeds']} | {mu['P']:.4f} | {mu['R']:.4f} | {mu['F1']:.4f} ± {sd['F1']:.4f} | "
                     f"[{s['f1_ci95'][0]:.4f}, {s['f1_ci95'][1]:.4f}] | " +
                     " | ".join(f"{mu[t]:.2f} ± {sd[t]:.2f}" for t in ENTITY_TYPES) + " |")
    if pairs:
        lines += ["", "| Comparison | ΔF1 | 95% CI of Δ | Share of resamples where ranking reverses |", "|---|---|---|---|"]
        for pr, d in pairs.items():
            lines.append(f"| {pr.replace(':', ' − ')} | {d['diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] | "
                         f"{d['share_resamples_reversed']:.3f} |")
    open(os.path.join(a.out, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
