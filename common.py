"""Shared helpers for the StrokeID-NER revision experiments.

Data format (one of):
  * CoNLL/TSV: one "token<TAB>label" per line, blank line between queries.
  * JSONL: one {"id": ..., "tokens": [...], "labels": [...]} object per line.
"""
import json
import os
from collections import Counter

ENTITY_TYPES = ["DGN", "RF", "SYM", "TMP"]
LABELS = ["O", "B-SYM", "I-SYM", "B-DGN", "I-DGN", "B-TMP", "I-TMP", "B-RF", "I-RF"]


def load_split(path):
    """Return a list of dicts: {id, tokens, labels}."""
    if path.endswith(".jsonl"):
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line.strip():
                    d = json.loads(line)
                    d.setdefault("id", i)
                    out.append(d)
        return out
    out, toks, labs = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if toks:
                    out.append({"id": len(out), "tokens": toks, "labels": labs})
                    toks, labs = [], []
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            toks.append(parts[0])
            labs.append(parts[-1])
    if toks:
        out.append({"id": len(out), "tokens": toks, "labels": labs})
    return out


def find_split(data_dir, name):
    for ext in (".jsonl", ".tsv", ".conll", ".txt"):
        p = os.path.join(data_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No {name}.(jsonl|tsv|conll|txt) in {data_dir}")


def get_spans(labels):
    """Extract (type, start, end_inclusive) spans from a BIO sequence.

    Mirrors seqeval's default (non-strict) behaviour: an I- tag that does not
    continue a span of the same type starts a new span.
    """
    spans, cur = [], None
    for i, lab in enumerate(list(labels) + ["O"]):
        tag, typ = (lab.split("-", 1) + [None])[:2] if lab != "O" else ("O", None)
        if cur is not None and (tag in ("O", "B") or typ != cur[0]):
            spans.append((cur[0], cur[1], i - 1))
            cur = None
        if tag == "B" or (tag == "I" and cur is None):
            cur = (typ, i)
    return spans


def per_query_counts(gold_seqs, pred_seqs, types=ENTITY_TYPES):
    """Array-like counts [n_queries][type] -> (tp, n_pred, n_gold), exact match."""
    rows = []
    for g, p in zip(gold_seqs, pred_seqs):
        gs, ps = set(get_spans(g)), set(get_spans(p))
        row = []
        for t in types:
            gt = {s for s in gs if s[0] == t}
            pt = {s for s in ps if s[0] == t}
            row.append((len(gt & pt), len(pt), len(gt)))
        rows.append(row)
    return rows


def macro_from_totals(tp, npred, ngold):
    """tp/npred/ngold are per-type sequences. Returns macro P, R, F1 and per-type F1
    (seqeval 'macro avg' definition: unweighted mean of per-type scores)."""
    ps, rs, fs = [], [], []
    for a, b, c in zip(tp, npred, ngold):
        p = a / b if b else 0.0
        r = a / c if c else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        ps.append(p); rs.append(r); fs.append(f)
    n = len(fs)
    return sum(ps) / n, sum(rs) / n, sum(fs) / n, fs


def write_predictions(path, examples, preds):
    with open(path, "w", encoding="utf-8") as f:
        for ex, p in zip(examples, preds):
            assert len(p) == len(ex["tokens"]), (ex["id"], len(p), len(ex["tokens"]))
            f.write(json.dumps({"id": ex["id"], "tokens": ex["tokens"],
                                "gold": ex["labels"], "pred": p}, ensure_ascii=False) + "\n")


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def label_stats(examples):
    c = Counter()
    for ex in examples:
        for t, *_ in get_spans(ex["labels"]):
            c[t] += 1
    return dict(c)
