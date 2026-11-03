"""Unified fine-tuning script for the StrokeID-NER revision.

One script, one code path, for every encoder/decoder combination, so all
models share identical data handling, label alignment, checkpoint selection and
evaluation (answers R1.2 / R2.2 / R2.3).

Examples
  python train.py --model m3 --seed 42  --data_dir data --out_dir runs
  python train.py --model m5crf --seed 123 --data_dir data --out_dir runs
  python train.py --model m3 --seed 42 --binary RF ...   # binary-NER experiment

Outputs in <out_dir>/<model>[_bin-X]/seed<seed>/:
  test_predictions.jsonl, val_predictions.jsonl, metrics.json, train_log.json
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from common import (ENTITY_TYPES, LABELS, find_split, load_split, macro_from_totals,
                    per_query_counts, write_predictions)

# Hyperparameters follow manuscript Table 3. CRF variants use exactly the same
# settings as their linear counterpart (only the decoder differs). M5 (plain
# XLM-R-large) uses exactly the M3 settings (only the starting checkpoint differs).
BASE = dict(weight_decay=0.01, grad_clip=1.0, max_len=128)
IB_BASE = dict(BASE, encoder="indobenchmark/indobert-base-p2", epochs=10, lr=1e-5,
               batch=16, accum=1, warmup=0.20, fp16=False, grad_ckpt=False)
IB_LARGE = dict(BASE, encoder="indobenchmark/indobert-large-p2", epochs=10, lr=1e-5,
                batch=8, accum=1, warmup=0.20, fp16=False, grad_ckpt=False)
XLMR_NER = dict(BASE, encoder="cahya/xlm-roberta-large-indonesian-NER", epochs=10, lr=5e-6,
                batch=4, accum=2, warmup=0.10, fp16=True, grad_ckpt=True)
XLMR_PLAIN = dict(XLMR_NER, encoder="FacebookAI/xlm-roberta-large")

CONFIGS = {
    "m1": dict(IB_BASE, crf=False, name="IndoBERT-base + linear"),
    "m2": dict(IB_BASE, crf=True, name="IndoBERT-base + CRF"),
    "m4": dict(IB_LARGE, crf=False, name="IndoBERT-large + linear"),
    "m4crf": dict(IB_LARGE, crf=True, name="IndoBERT-large + CRF"),
    "m3": dict(XLMR_NER, crf=False, name="XLM-R-large (Indonesian-NER ckpt) + linear"),
    "m3crf": dict(XLMR_NER, crf=True, name="XLM-R-large (Indonesian-NER ckpt) + CRF"),
    "m5": dict(XLMR_PLAIN, crf=False, name="XLM-R-large (plain) + linear"),
    "m5crf": dict(XLMR_PLAIN, crf=True, name="XLM-R-large (plain) + CRF"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class Tagger(nn.Module):
    """Encoder -> dropout -> linear emissions over first sub-tokens -> (CRF | argmax)."""

    def __init__(self, encoder_name, num_labels, use_crf, dropout=0.1, grad_ckpt=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name, add_pooling_layer=False) \
            if "bert" in encoder_name.lower() or "roberta" in encoder_name.lower() \
            else AutoModel.from_pretrained(encoder_name)
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.use_crf = use_crf
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(num_labels, batch_first=True)

    def word_emissions(self, input_ids, attention_mask, word_index, word_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        em = self.head(self.dropout(h))                           # [B, T, L]
        idx = word_index.unsqueeze(-1).expand(-1, -1, em.size(-1))
        return torch.gather(em, 1, idx).float()                  # [B, W, L] (first sub-tokens)

    def loss(self, batch):
        em = self.word_emissions(batch["input_ids"], batch["attention_mask"],
                                 batch["word_index"], batch["word_mask"])
        y, m = batch["word_labels"], batch["word_mask"]
        if self.use_crf:
            return -self.crf(em, y, mask=m, reduction="mean")
        return nn.functional.cross_entropy(em[m], y[m])

    @torch.no_grad()
    def decode(self, batch):
        em = self.word_emissions(batch["input_ids"], batch["attention_mask"],
                                 batch["word_index"], batch["word_mask"])
        m = batch["word_mask"]
        if self.use_crf:
            return self.crf.decode(em, mask=m)
        am = em.argmax(-1)
        return [am[i][m[i]].tolist() for i in range(am.size(0))]


def encode(examples, tok, max_len, lab2id):
    """First-sub-token alignment. Words cut off by truncation are not seen by the
    model and are predicted as O at evaluation time (counted as misses)."""
    feats, n_trunc = [], 0
    for ex in examples:
        enc = tok(ex["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        wids = enc.word_ids()
        first = {}
        for pos, w in enumerate(wids):
            if w is not None and w not in first:
                first[w] = pos
        n_seen = len(first)
        if n_seen < len(ex["tokens"]):
            n_trunc += 1
        feats.append(dict(input_ids=enc["input_ids"], word_index=[first[w] for w in range(n_seen)],
                          word_labels=[lab2id[l] for l in ex["labels"][:n_seen]],
                          n_words=len(ex["tokens"])))
    return feats, n_trunc


def collate(feats, pad_id):
    B = len(feats)
    T = max(len(f["input_ids"]) for f in feats)
    W = max(max(len(f["word_index"]) for f in feats), 1)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    am = torch.zeros((B, T), dtype=torch.long)
    wi = torch.zeros((B, W), dtype=torch.long)
    wl = torch.zeros((B, W), dtype=torch.long)
    wm = torch.zeros((B, W), dtype=torch.bool)
    for i, f in enumerate(feats):
        n, k = len(f["input_ids"]), len(f["word_index"])
        ids[i, :n] = torch.tensor(f["input_ids"]); am[i, :n] = 1
        if k:
            wi[i, :k] = torch.tensor(f["word_index"]); wl[i, :k] = torch.tensor(f["word_labels"])
            wm[i, :k] = True
    # torchcrf requires the first timestep of every sequence to be unmasked.
    wm[:, 0] = True
    return dict(input_ids=ids, attention_mask=am, word_index=wi, word_labels=wl, word_mask=wm)


def predict(model, feats, examples, id2lab, device, bs, pad_id, fp16):
    model.eval()
    preds = []
    for s in range(0, len(feats), bs):
        chunk = feats[s:s + bs]
        batch = {k: v.to(device) for k, v in collate(chunk, pad_id).items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=fp16 and device.type == "cuda"):
            out = model.decode(batch)
        for f, seq in zip(chunk, out):
            labs = [id2lab[i] for i in seq[:len(f["word_index"])]]
            labs += ["O"] * (f["n_words"] - len(labs))   # truncated words -> O
            preds.append(labs)
    return preds


def score(examples, preds):
    rows = per_query_counts([e["labels"] for e in examples], preds)
    tot = np.array(rows).sum(0)   # [types, 3]
    P, R, F, per = macro_from_totals(tot[:, 0], tot[:, 1], tot[:, 2])
    return dict(precision=float(P), recall=float(R), f1=float(F), per_entity={k: float(v) for k, v in zip(ENTITY_TYPES, per)})


def to_binary(examples, target):
    out = []
    for ex in examples:
        labs = [l if l != "O" and l.split("-", 1)[1] == target else "O" for l in ex["labels"]]
        out.append(dict(ex, labels=labs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--binary", choices=ENTITY_TYPES, default=None,
                    help="Train a single-entity (binary) model for this type")
    ap.add_argument("--crf_lr", type=float, default=1e-3,
                    help="LR for CRF transition parameters (encoder LR is too small for them)")
    ap.add_argument("--encoder_override", default=None, help="for local testing only")
    ap.add_argument("--epochs_override", type=int, default=None, help="for local testing only")
    ap.add_argument("--lr_override", type=float, default=None, help="for local testing only")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.model])
    if args.encoder_override:
        cfg["encoder"] = args.encoder_override
    if args.epochs_override:
        cfg["epochs"] = args.epochs_override
    if args.lr_override:
        cfg["lr"] = args.lr_override
    tag = args.model + (f"_bin-{args.binary}" if args.binary else "")
    out = os.path.join(args.out_dir, tag, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(find_split(args.data_dir, "train"))
    val = load_split(find_split(args.data_dir, "val"))
    test = load_split(find_split(args.data_dir, "test"))
    if args.binary:
        train, val, test = (to_binary(x, args.binary) for x in (train, val, test))
        labels = ["O", f"B-{args.binary}", f"I-{args.binary}"]
    else:
        labels = LABELS
    lab2id = {l: i for i, l in enumerate(labels)}
    id2lab = {i: l for l, i in lab2id.items()}

    tok = AutoTokenizer.from_pretrained(cfg["encoder"])
    assert tok.is_fast, "A fast tokenizer is required for word_ids() alignment"
    pad_id = tok.pad_token_id
    ftr, tr_tr = encode(train, tok, cfg["max_len"], lab2id)
    fva, tr_va = encode(val, tok, cfg["max_len"], lab2id)
    fte, tr_te = encode(test, tok, cfg["max_len"], lab2id)
    print(f"[{tag} seed{args.seed}] {cfg['name']} | truncated queries: "
          f"train {tr_tr}, val {tr_va}, test {tr_te} (max_len={cfg['max_len']})")

    model = Tagger(cfg["encoder"], len(labels), cfg["crf"], grad_ckpt=cfg["grad_ckpt"]).to(device)

    # AdamW; no weight decay on bias / LayerNorm (manuscript Table 3 footnote).
    nd = ("bias", "LayerNorm.weight", "layer_norm.weight", "LayerNorm.bias")
    enc_dec = [p for n, p in model.named_parameters() if not n.startswith("crf.") and not any(x in n for x in nd)]
    enc_nod = [p for n, p in model.named_parameters() if not n.startswith("crf.") and any(x in n for x in nd)]
    groups = [dict(params=enc_dec, weight_decay=cfg["weight_decay"], lr=cfg["lr"]),
              dict(params=enc_nod, weight_decay=0.0, lr=cfg["lr"])]
    if cfg["crf"]:
        groups.append(dict(params=list(model.crf.parameters()), weight_decay=0.0, lr=args.crf_lr))
    opt = torch.optim.AdamW(groups)

    g = torch.Generator(); g.manual_seed(args.seed)
    loader = DataLoader(ftr, batch_size=cfg["batch"], shuffle=True, generator=g,
                        collate_fn=lambda b: collate(b, pad_id))
    steps = math.ceil(len(loader) / cfg["accum"]) * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * steps), steps)
    use_amp = cfg["fp16"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, best_epoch, log = -1.0, None, -1, []
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running, opt_step = 0.0, 0
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model.loss(batch) / cfg["accum"]
            scaler.scale(loss).backward()
            running += loss.item() * cfg["accum"]
            if (i + 1) % cfg["accum"] == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                opt_step += 1
        vp = predict(model, fva, val, id2lab, device, 32, pad_id, cfg["fp16"])
        vm = score(val, vp) if not args.binary else score_binary(val, vp, args.binary)
        log.append(dict(epoch=epoch, train_loss=running / len(loader), val_f1=vm["f1"]))
        print(f"  epoch {epoch:2d} loss {running/len(loader):.4f} val macro-F1 {vm['f1']:.4f}")
        if vm["f1"] > best_f1:
            best_f1, best_epoch = vm["f1"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    vp = predict(model, fva, val, id2lab, device, 32, pad_id, cfg["fp16"])
    tp = predict(model, fte, test, id2lab, device, 32, pad_id, cfg["fp16"])
    write_predictions(os.path.join(out, "val_predictions.jsonl"), val, vp)
    write_predictions(os.path.join(out, "test_predictions.jsonl"), test, tp)
    tm = score(test, tp) if not args.binary else score_binary(test, tp, args.binary)
    meta = dict(model=tag, name=cfg["name"], seed=args.seed, config=cfg, crf_lr=args.crf_lr if cfg["crf"] else None,
                best_epoch=best_epoch, best_val_f1=best_f1, test=tm,
                truncated_queries=dict(train=tr_tr, val=tr_va, test=tr_te),
                device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                torch=torch.__version__, transformers=__import__("transformers").__version__,
                minutes=round((time.time() - t0) / 60, 1))
    json.dump(meta, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    json.dump(log, open(os.path.join(out, "train_log.json"), "w"), indent=2)
    print(f"[{tag} seed{args.seed}] best epoch {best_epoch}, TEST macro-F1 {tm['f1']:.4f} "
          f"(P {tm['precision']:.4f} R {tm['recall']:.4f}) {tm['per_entity']}")


def score_binary(examples, preds, target):
    rows = per_query_counts([e["labels"] for e in examples], preds, types=[target])
    tp, npred, ngold = np.array(rows).sum(0)[0]
    P = tp / npred if npred else 0.0
    R = tp / ngold if ngold else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return dict(precision=float(P), recall=float(R), f1=float(F), per_entity={target: float(F)})


if __name__ == "__main__":
    main()
