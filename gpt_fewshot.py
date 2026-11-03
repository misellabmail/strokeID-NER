"""GPT-5.4 zero-shot and few-shot conditions (R2.2c).

Reproduces gpt_zero_shot.ipynb exactly for k=0 (same system prompt, the query
sent as space-joined tokens, [{"token","label"}] output, temperature=0,
max_completion_tokens=4096, up to 5 retries, the same token-alignment recovery,
all-O after 5 failures). For k>0 the same k annotated TRAINING queries are added
as user/assistant demonstration turns before every test query.

Demonstrations: chosen once per --seed from the training split, greedily so all
four entity types appear, then filled up to k. Different seeds = different
demonstration sets (gives SD across demonstration choice).

Usage
  export OPENAI_API_KEY=...
  python gpt_fewshot.py --data_dir data --k 0 --seed 42 --out_dir runs/gpt0
  python gpt_fewshot.py --data_dir data --k 5 --seed 42 --out_dir runs/gpt5
Responses are cached in raw_responses.jsonl so an interrupted run resumes.
"""
import argparse
import json
import os
import random
import time

from common import ENTITY_TYPES, LABELS, find_split, get_spans, load_split, write_predictions

HERE = os.path.dirname(os.path.abspath(__file__))


def align_gpt_to_input(input_tokens, gpt_items):
    """
    Align GPT token-label pairs back to the original input token list.

    GPT sometimes merges multiple input tokens into one item (e.g. 'tidak bisa bicara'),
    splits one input token into multiple items (e.g. 'tiba-tiba' -> ['tiba','-','tiba']),
    or skips punctuation. This function recovers correct labels for all input tokens
    even when counts don't match, instead of discarding all labels.
    """
    result  = ['O'] * len(input_tokens)
    inp_idx = 0

    def try_merge_from(pos):
        """Check if gpt_tok matches a merge of input tokens starting at pos."""
        consumed = ''
        j = pos
        while j < len(input_tokens) and len(consumed) <= len(gpt_tok) + 5:
            consumed = (consumed + ' ' + input_tokens[j]).strip()
            if consumed.lower() == gpt_tok.lower():
                return j
            if len(consumed) > len(gpt_tok) + 2:
                break
            j += 1
        return None

    for item in gpt_items:
        if inp_idx >= len(input_tokens):
            break
        gpt_tok = item.get('token', '').strip()
        gpt_lab = item.get('label', 'O')
        if gpt_lab not in LABELS:
            gpt_lab = 'O'
        if not gpt_tok:
            continue

        def assign_merge(start, end, lab):
            """Assign lab to start, and the continuation tag to start+1..end."""
            for k in range(start, end + 1):
                if k == start:
                    result[k] = lab
                else:
                    result[k] = ('I-' + lab[2:]) if lab.startswith('B-') else lab

        #Try at current position
        if gpt_tok.lower() == input_tokens[inp_idx].lower():
            result[inp_idx] = gpt_lab
            inp_idx += 1
            continue

        merge_end = try_merge_from(inp_idx)
        if merge_end is not None:
            assign_merge(inp_idx, merge_end, gpt_lab)
            inp_idx = merge_end + 1
            continue

        if (len(gpt_tok) < len(input_tokens[inp_idx]) and
                gpt_tok.lower() in input_tokens[inp_idx].lower()):
            result[inp_idx] = gpt_lab
            inp_idx += 1
            continue

        #GPT skipped input tokens — look ahead up to 3 positions ─
        for lookahead in range(1, 4):
            ahead = inp_idx + lookahead
            if ahead >= len(input_tokens):
                break
            if gpt_tok.lower() == input_tokens[ahead].lower():
                inp_idx = ahead
                result[inp_idx] = gpt_lab
                inp_idx += 1
                break
            merge_end = try_merge_from(ahead)
            if merge_end is not None:
                inp_idx = ahead
                assign_merge(inp_idx, merge_end, gpt_lab)
                inp_idx = merge_end + 1
                break
            if (len(gpt_tok) < len(input_tokens[ahead]) and
                    gpt_tok.lower() in input_tokens[ahead].lower()):
                inp_idx = ahead
                result[inp_idx] = gpt_lab
                inp_idx += 1
                break
        # If still unmatched: GPT hallucinated a token — skip it silently

    return result




def pick_demos(train, k, seed):
    rng = random.Random(seed)
    pool = [q for q in train if 5 <= len(q["tokens"]) <= 60]
    rng.shuffle(pool)
    chosen, covered = [], set()
    for q in pool:                                   # first cover every entity type
        types = {t for t, *_ in get_spans(q["labels"])}
        if types - covered:
            chosen.append(q); covered |= types
        if covered >= set(ENTITY_TYPES) or len(chosen) == k:
            break
    for q in pool:                                   # then fill up to k
        if len(chosen) >= k:
            break
        if q not in chosen:
            chosen.append(q)
    return chosen[:k]


def answer_json(q):
    return json.dumps([{"token": t, "label": l} for t, l in zip(q["tokens"], q["labels"])], ensure_ascii=False)


def build_messages(system_prompt, demos, q):
    msgs = [{"role": "system", "content": system_prompt}]
    for d in demos:
        msgs.append({"role": "user", "content": " ".join(d["tokens"])})
        msgs.append({"role": "assistant", "content": answer_json(d)})
    msgs.append({"role": "user", "content": " ".join(q["tokens"])})
    return msgs


def parse(raw, tokens):
    """Same logic as the notebook. Returns (labels, status) or raises on bad JSON."""
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    if len(result) == len(tokens):
        return [l if l in LABELS else "O" for l in (item.get("label", "O") for item in result)], "exact"
    return align_gpt_to_input(tokens, result), "aligned"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--system_prompt", default=os.path.join(HERE, "prompts", "zero_shot_system.txt"))
    ap.add_argument("--model", default="gpt-5.4")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_retries", type=int, default=5)
    a = ap.parse_args()
    from openai import OpenAI
    client = OpenAI()

    out = os.path.join(a.out_dir, f"seed{a.seed}"); os.makedirs(out, exist_ok=True)
    train = load_split(find_split(a.data_dir, "train"))
    test = load_split(find_split(a.data_dir, "test"))
    demos = pick_demos(train, a.k, a.seed) if a.k else []
    json.dump([{"id": d["id"], "tokens": d["tokens"], "labels": d["labels"]} for d in demos],
              open(os.path.join(out, "demonstrations.json"), "w"), indent=1, ensure_ascii=False)
    system_prompt = open(a.system_prompt, encoding="utf-8").read()

    cache_p = os.path.join(out, "raw_responses.jsonl")
    cache = {}
    if os.path.exists(cache_p):
        for l in open(cache_p, encoding="utf-8"):
            r = json.loads(l); cache[str(r["id"])] = r
    preds, status = [], {"exact": 0, "aligned": 0, "failed_all_O": 0}
    with open(cache_p, "a", encoding="utf-8") as cf:
        for n, q in enumerate(test):
            rec = cache.get(str(q["id"]))
            if rec is None:
                labs, st, raws, wait = None, "failed_all_O", [], 2
                for attempt in range(a.max_retries):
                    try:
                        r = client.chat.completions.create(
                            model=a.model, messages=build_messages(system_prompt, demos, q),
                            temperature=0, max_completion_tokens=4096)
                        raw = r.choices[0].message.content or ""
                        raws.append(raw)
                        labs, st = parse(raw, q["tokens"])
                        break
                    except Exception as e:
                        err = str(e)
                        if "rate_limit" in err.lower() or "429" in err:
                            print(f"  rate limit, waiting {wait}s"); time.sleep(wait); wait = min(wait * 2, 60)
                        elif attempt < a.max_retries - 1:
                            time.sleep(wait)
                        else:
                            print(f"  query {q['id']}: failed after {a.max_retries} attempts: {err[:80]}")
                if labs is None:
                    labs = ["O"] * len(q["tokens"])
                rec = {"id": q["id"], "labels": labs, "status": st, "raw": raws}
                cf.write(json.dumps(rec, ensure_ascii=False) + "\n"); cf.flush()
                time.sleep(0.5)
            preds.append(rec["labels"]); status[rec["status"]] += 1
            if (n + 1) % 25 == 0:
                print(f"  {n+1}/{len(test)}")
    write_predictions(os.path.join(out, "test_predictions.jsonl"), test, preds)
    json.dump(dict(model=a.model, k=a.k, seed=a.seed, temperature=0, max_completion_tokens=4096,
                   parse_status=status, demo_ids=[d["id"] for d in demos]),
              open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print("parse status:", status, "-> report these counts in the paper")


if __name__ == "__main__":
    main()
