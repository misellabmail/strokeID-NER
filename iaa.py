"""Inter-annotator agreement beyond token-level kappa (R2.1).

Reports, on the doubly annotated sample:
  * entity-level (span) agreement F1, exact match of boundaries + type
    (Hripcsak & Rothschild 2005: pairwise F1 approximates kappa for NER when
    the number of negative cases is undefined) - overall micro, macro, per type;
  * relaxed agreement F1 (same type, overlapping spans);
  * token-level Cohen's kappa on all tokens (the number currently reported) and
    on tokens where at least one annotator assigned an entity label (removes the
    inflation from O-O agreement the reviewer points out).

Usage: python iaa.py --a primary_100.jsonl --b second_100.jsonl [--out results/iaa.json]
Both files: same queries, same tokenisation (JSONL with tokens/labels or CoNLL).
"""
import argparse
import json
from collections import Counter

from common import ENTITY_TYPES, get_spans, load_split


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def f1(tp, na, nb):
    return 2 * tp / (na + nb) if na + nb else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--out", default=None)
    o = ap.parse_args()
    A, B = load_split(o.a), load_split(o.b)
    assert len(A) == len(B), "different number of queries"
    exact, relaxed, na, nb = Counter(), Counter(), Counter(), Counter()
    tok_a, tok_b = [], []
    disagreements = []
    for qa, qb in zip(A, B):
        assert qa["tokens"] == qb["tokens"], f"tokenisation differs in query {qa['id']}"
        tok_a += qa["labels"]; tok_b += qb["labels"]
        sa, sb = set(get_spans(qa["labels"])), set(get_spans(qb["labels"]))
        for t in ENTITY_TYPES:
            ta = [s for s in sa if s[0] == t]; tb = [s for s in sb if s[0] == t]
            na[t] += len(ta); nb[t] += len(tb)
            exact[t] += len(set(ta) & set(tb))
            # relaxed: greedy one-to-one matching of overlapping same-type spans
            used = set()
            for s in ta:
                for j, u in enumerate(tb):
                    if j not in used and not (u[2] < s[1] or u[1] > s[2]):
                        used.add(j); relaxed[t] += 1; break
        if sa != sb:
            disagreements.append({"id": qa["id"], "tokens": qa["tokens"],
                                  "only_a": sorted(sa - sb), "only_b": sorted(sb - sa)})
    per_exact = {t: f1(exact[t], na[t], nb[t]) for t in ENTITY_TYPES}
    per_relax = {t: f1(relaxed[t], na[t], nb[t]) for t in ENTITY_TYPES}
    ent_idx = [i for i, (x, y) in enumerate(zip(tok_a, tok_b)) if x != "O" or y != "O"]
    res = dict(
        n_queries=len(A), n_tokens=len(tok_a), spans_a=dict(na), spans_b=dict(nb),
        exact_f1_micro=f1(sum(exact.values()), sum(na.values()), sum(nb.values())),
        exact_f1_macro=sum(per_exact.values()) / len(ENTITY_TYPES), exact_f1_per_type=per_exact,
        relaxed_f1_micro=f1(sum(relaxed.values()), sum(na.values()), sum(nb.values())),
        relaxed_f1_per_type=per_relax,
        token_kappa_all=kappa(tok_a, tok_b),
        token_kappa_entity_tokens_only=kappa([tok_a[i] for i in ent_idx], [tok_b[i] for i in ent_idx]),
        n_entity_tokens=len(ent_idx),
        share_tokens_both_O=1 - len(ent_idx) / len(tok_a),
        n_queries_with_disagreement=len(disagreements))
    print(json.dumps(res, indent=2))
    if o.out:
        json.dump(dict(res, disagreements=disagreements), open(o.out, "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
