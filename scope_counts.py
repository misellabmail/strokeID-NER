"""Corpus counts for the annotation-scope rationale (R2.4).

Finds, across all splits:
  * medication / prescription mentions (lexicon + dosage patterns),
  * procedure / examination mentions (out of scope per guideline),
  * negated mentions: a negation cue followed (within 3 tokens) by a word that is
    annotated as an entity elsewhere in the corpus but is labelled O here.
Every hit is written out with its query so you can VERIFY BY HAND before quoting
a number in the paper - lexicon matching over- and under-counts.

Usage: python scope_counts.py --data_dir data --out results/scope
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

from common import find_split, get_spans, load_split

MEDS = {
    # antiplatelet / anticoagulant / thrombolytic
    "aspirin", "asetosal", "aspilet", "miniaspi", "clopidogrel", "cpg", "plavix", "cilostazol",
    "warfarin", "heparin", "rivaroxaban", "dabigatran", "alteplase", "rtpa", "rt-pa",
    # neuro "supplements" common in Indonesia
    "citicoline", "citicolin", "sitikolin", "piracetam", "neurobion", "mecobalamin", "mecobalamine",
    "betahistine", "flunarizine", "gabapentin", "pregabalin",
    # antihypertensives
    "amlodipin", "amlodipine", "captopril", "lisinopril", "ramipril", "candesartan", "valsartan",
    "losartan", "irbesartan", "telmisartan", "bisoprolol", "propranolol", "furosemide", "hct", "nifedipin",
    "nifedipine", "nicardipine", "clonidine", "spironolactone",
    # lipid / diabetes
    "simvastatin", "atorvastatin", "rosuvastatin", "metformin", "glimepiride", "glibenclamide", "insulin",
    # analgesic / other
    # "asam" deliberately excluded: "asam urat" (uric acid) is an RF
    "paracetamol", "ibuprofen", "mefenamat", "diazepam", "phenytoin", "fenitoin",
    "omeprazole", "lansoprazole", "ranitidin", "dexamethasone", "prednison",
}
GENERIC_MED = {"obat", "obatnya", "obat-obatan", "tablet", "kapsul", "resep", "suntik", "suntikan",
               "infus", "jamu", "herbal", "suplemen", "vitamin"}
DOSE = re.compile(r"^\d+(mg|ml|mcg|gr|g|iu)$", re.I)
PROC = {"ct", "ct-scan", "ctscan", "mri", "eeg", "ekg", "ecg", "rontgen", "operasi", "fisioterapi",
        "terapi", "cek", "lab", "laboratorium", "rawat", "opname", "kraniotomi", "angiografi", "dsa"}
NEG = {"tidak", "tdk", "tak", "bukan", "belum", "blm", "tanpa", "gak", "ga", "nggak", "ngga", "enggak",
       "engga", "gk", "g", "ndak", "nda"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--out", default="results/scope")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = []
    for s in ("train", "val", "test"):
        data += [dict(q, split=s) for q in load_split(find_split(a.data_dir, s))]
    N = len(data)

    # "entity words": words that START an entity span at least twice and are inside
    # an entity in at least half of all their occurrences (filters out function
    # words such as "saya", "pada" that happen to sit inside some spans).
    ent_words = defaultdict(Counter)
    starts, total = Counter(), Counter()
    for q in data:
        for w in q["tokens"]:
            total[w.lower()] += 1
        for t, i, j in get_spans(q["labels"]):
            starts[q["tokens"][i].lower()] += 1
            for w in q["tokens"][i:j + 1]:
                ent_words[w.lower()][t] += 1
    ent_words = {w: c for w, c in ent_words.items()
                 if starts[w] >= 2 and sum(c.values()) / total[w] >= 0.5 and len(w) > 2 and w not in NEG}

    hits = {"medication_specific": [], "medication_generic": [], "dosage": [], "procedure": [], "negated": []}
    q_with = {k: set() for k in hits}
    term_counts = {k: Counter() for k in hits}
    med_label = Counter()
    for q in data:
        toks = [t.lower() for t in q["tokens"]]
        for i, (w, lab) in enumerate(zip(toks, q["labels"])):
            for key, cond in (("medication_specific", w in MEDS), ("medication_generic", w in GENERIC_MED),
                              ("dosage", bool(DOSE.match(w))), ("procedure", w in PROC)):
                if cond:
                    hits[key].append(dict(id=q["id"], split=q["split"], i=i, token=q["tokens"][i], label=lab,
                                          context=" ".join(q["tokens"][max(0, i - 6):i + 7])))
                    q_with[key].add((q["split"], q["id"])); term_counts[key][w] += 1
                    if key == "medication_specific":
                        med_label[lab.split("-")[-1] if lab != "O" else "O"] += 1
            if w in NEG:
                for k in range(i + 1, min(i + 4, len(toks))):
                    if q["labels"][k] == "O" and toks[k] in ent_words and sum(ent_words[toks[k]].values()) >= 2:
                        typ = ent_words[toks[k]].most_common(1)[0][0]
                        hits["negated"].append(dict(id=q["id"], split=q["split"], cue=q["tokens"][i],
                                                    word=q["tokens"][k], usual_type=typ,
                                                    context=" ".join(q["tokens"][max(0, i - 4):k + 5])))
                        q_with["negated"].add((q["split"], q["id"])); term_counts["negated"][typ] += 1
                        break
    summary = {k: dict(mentions=len(v), queries=len(q_with[k]), pct_queries=round(100 * len(q_with[k]) / N, 1),
                       top_terms=term_counts[k].most_common(15)) for k, v in hits.items()}
    summary["n_queries"] = N
    summary["medication_specific_label_distribution"] = dict(med_label)
    json.dump(summary, open(os.path.join(a.out, "scope_summary.json"), "w"), indent=2, ensure_ascii=False)
    for k, v in hits.items():
        with open(os.path.join(a.out, f"hits_{k}.jsonl"), "w") as f:
            for h in v:
                f.write(json.dumps(h, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nHit lists for manual verification written to {a.out}/hits_*.jsonl")


if __name__ == "__main__":
    main()
