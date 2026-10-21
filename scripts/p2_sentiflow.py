"""P2 / E1: SentiFlow versus LEX, A (base-paper GRU→CNN), A_rev, B (FinBERT), PROSUS; E1-b with Araci's split.

    python scripts/p2_sentiflow.py            # full run (needs the Word2Vec of P2 step 1)
    python scripts/p2_sentiflow.py --smoke    # code check: tiny Word2Vec on the labelled sentences, 1 seed,
                                              # 1 epoch, nothing written to results.json

Outputs: sentiflow.* keys; data/processed/sentiflow/ (per-seed predictions, models for corpus scoring).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import PROCESSED, RESULTS, frozen  # noqa: E402
from smartrl.data import sentiment_sets as S  # noqa: E402
from smartrl.runs import run_record  # noqa: E402
from smartrl.sentiflow import calibration as C  # noqa: E402
from smartrl.sentiflow import finbert, lexicon, word2vec  # noqa: E402
from smartrl.sentiflow import train as T  # noqa: E402
from smartrl.stats.classification import macro_f1, mcnemar_exact, paired_bootstrap_f1_diff  # noqa: E402

OUT = PROCESSED / "sentiflow"
DATASETS = ("phrasebank50", "phrasebank_all", "fiqa", "twitter")
POOLED_FROM = ("phrasebank50", "fiqa", "twitter")      # phrasebank_all ⊂ phrasebank50: not counted twice
PROSUS_DS = ("fiqa", "twitter")                         # PROSUS was fine-tuned on PhraseBank (SPEC §3.7)
PROSUS_LABEL = {"negative": 0, "neutral": 1, "positive": 2}


def split_frame(df: pd.DataFrame, split: str, datasets=POOLED_FROM) -> pd.DataFrame:
    return df[(df["split"] == split) & df["dataset"].isin(datasets)].reset_index(drop=True)


def eval_sets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {d: df[(df["split"] == "test") & (df["dataset"] == d)].reset_index(drop=True) for d in DATASETS}
    out["pooled"] = split_frame(df, "test")
    return out


def branch_b(tok, encoder, tr: pd.DataFrame, va: pd.DataFrame, seed: int, b_cfg: dict, cache: Path | None):
    """Branch B for one seed; trained once and cached (it depends only on the labelled sentences and the seed)."""
    from smartrl.sentiflow.models import BranchB
    if cache is not None and (cache / "branch_b_model.pt").exists():
        ck = torch.load(cache / "branch_b_model.pt", map_location="cpu", weights_only=False)
        mb = BranchB(encoder).to(T.DEV)
        mb.load_state_dict(ck["state"])
        return mb.eval(), ck["hist"]
    texts = lambda d: d["sentence"].astype(str).tolist()
    mb, hb = T.train_bert(tok, encoder, (texts(tr), tr["label"].to_numpy()), (texts(va), va["label"].to_numpy()), seed, b_cfg)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        torch.save({"state": mb.state_dict(), "hist": hb.__dict__}, cache / "branch_b_model.pt")
    return mb, hb.__dict__


def run_seed(seed: int, tr: pd.DataFrame, va: pd.DataFrame, evals: dict[str, pd.DataFrame], wv, cfg: dict,
             save_dir: Path | None, smoke: bool, b_cache: Path | None = None) -> dict:
    """Train A, A_rev, B and the gate for one seed; return probabilities on val and every evaluation set."""
    sf = cfg["sentiflow"]
    vocab = word2vec.Vocab(wv, tr["sentence"])
    emb = vocab.embedding_matrix(wv, seed)
    enc = lambda d: (*vocab.encode(d["sentence"], sf["branch_a"]["max_len"]), d["label"].to_numpy())
    a_cfg = {**sf["branch_a"], **({"max_epochs": 1} if smoke else {})}
    out: dict = {"hist": {}}
    sets = {"train": tr, "val": va, **evals}
    models = {}
    for kind in ("A", "A_REV"):
        m, h = T.train_rnn(kind, emb, enc(tr), enc(va), seed, {**a_cfg, **sf["branch_a_rev"]} if kind == "A_REV" else a_cfg)
        out["hist"][kind] = h.__dict__
        models[kind] = m
        for name, d in sets.items():
            ids, lens, _ = enc(d)
            out[(kind, name)] = T.predict_rnn(m, ids, lens)
    tok, encoder = finbert.load_pretrain()
    b_cfg = {**sf["branch_b"], **({"max_epochs": 1} if smoke else {})}
    texts = lambda d: d["sentence"].astype(str).tolist()
    mb, hb = branch_b(tok, encoder, tr, va, seed, b_cfg, b_cache)
    out["hist"]["B"] = hb
    for name, d in sets.items():
        out[("B", name)] = T.predict_bert(mb, tok, texts(d), b_cfg["max_len"])
    pack = lambda name: {"h_a": out[("A", name)][1], "h_b": out[("B", name)][1], "p_a": out[("A", name)][0],
                         "p_b": out[("B", name)][0], "y": sets[name]["label"].to_numpy()}
    gate, gh = T.train_fusion(pack("train"), pack("val"), seed, sf["fusion"])
    out["hist"]["fusion"] = gh
    for name in sets:
        out[("SF_raw", name)] = T.predict_fusion(gate, pack(name))
    y_val = va["label"].to_numpy()
    out["T_sf"] = C.fit_temperature(out[("SF_raw", "val")][0], y_val)
    out["T_b"] = C.fit_temperature(out[("B", "val")][0], y_val)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state": models["A"].state_dict(), "itos": vocab.itos, "cfg": a_cfg}, save_dir / "branch_a.pt")
        torch.save({"state": mb.state_dict()}, save_dir / "branch_b.pt")
        torch.save({"state": gate.state_dict(), "T_sf": out["T_sf"], "T_b": out["T_b"]}, save_dir / "gate.pt")
    del mb, models
    torch.cuda.empty_cache()
    return out


def metric_block(p: np.ndarray, y: np.ndarray, bins: int) -> dict:
    pred = p.argmax(1)
    return {"acc": float((pred == y).mean()), "f1": macro_f1(y, pred), "ece": C.ece(p, y, bins), "brier": C.brier(p, y)}


def summarise(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = float(v.mean())
        if len(v) > 1:
            out[f"{k}_sd"] = float(v.std(ddof=1))
    return out


def e1(df: pd.DataFrame, wv, cfg: dict, seeds: list[int], smoke: bool, reps: int) -> tuple[dict, dict]:
    bins = cfg["sentiflow"]["calibration"]["ece_bins"]
    tr, va = split_frame(df, "train"), split_frame(df, "val")
    evals = eval_sets(df)
    per_seed = []
    for s in seeds:
        t0 = time.time()
        save = None if smoke else OUT / f"seed{s}"
        per_seed.append(run_seed(s, tr, va, evals, wv, cfg, save, smoke, None if smoke else OUT / "branch_b" / f"seed{s}"))
        print(f"seed {s} done in {time.time() - t0:.0f}s", flush=True)
    res: dict = {}
    # LEX (deterministic): thresholds tuned on the pooled validation split
    pol_va = lexicon.polarity(va["sentence"].tolist())
    lo, hi, f1v = lexicon.tune(pol_va, va["label"].to_numpy())
    res["lex_thresholds"] = {"lo": lo, "hi": hi, "val_f1": f1v}
    # PROSUS zero-shot
    tok_p, mod_p, labels = finbert.load_prosus()
    mod_p = mod_p.to(T.DEV).eval()
    perm = [labels.index(k) for k in ("negative", "neutral", "positive")]
    detail = {"per_seed": {}, "T_sf": [r["T_sf"] for r in per_seed], "T_b": [r["T_b"] for r in per_seed]}
    for ds, d in evals.items():
        y = d["label"].to_numpy()
        block = {}
        cal = lambda r, key, Tk: C.apply_temperature(r[(key, ds)][0], r[Tk])
        model_probs = {
            "A": [r[("A", ds)][0] for r in per_seed],
            "A_REV": [r[("A_REV", ds)][0] for r in per_seed],
            "B": [r[("B", ds)][0] for r in per_seed],
            "SF": [cal(r, "SF_raw", "T_sf") for r in per_seed],
        }
        for sm, plist in model_probs.items():
            block[sm] = summarise([metric_block(p, y, bins) for p in plist])
        pred_lex = lexicon.classify(lexicon.polarity(d["sentence"].tolist()), lo, hi)
        block["LEX"] = {"acc": float((pred_lex == y).mean()), "f1": macro_f1(y, pred_lex)}
        if ds in PROSUS_DS:
            texts = d["sentence"].astype(str).tolist()
            with torch.no_grad():
                P = []
                for i in range(0, len(texts), 256):
                    e = tok_p(texts[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors="pt").to(T.DEV)
                    P.append(torch.softmax(mod_p(**e).logits.float(), -1)[:, perm].cpu().numpy())
            block["PROSUS"] = metric_block(np.concatenate(P), y, bins)
        res[ds] = block
        preds = {k: [p.argmax(1) for p in v] for k, v in model_probs.items()}
        tests = {}
        for a, b, name in (("SF", "B", "sf_b"), ("SF", "A", "sf_a"), ("A", "A_REV", "a_arev")):
            ps = [mcnemar_exact(pa == y, pb == y)["p"] for pa, pb in zip(preds[a], preds[b])]
            tests[f"mcnemar_{name}_median_p"] = float(np.median(ps))
            if name == "sf_b":
                tests["mcnemar_sf_b_n_sig"] = int(sum(p < 0.05 for p in ps))
            bt = paired_bootstrap_f1_diff(y, preds[a], preds[b], reps=reps)
            tests[f"f1_diff_{name}"] = bt["est"]
            if name != "a_arev":
                tests[f"f1_diff_{name}_lo"], tests[f"f1_diff_{name}_hi"] = bt["lo"], bt["hi"]
            detail.setdefault("mcnemar_p", {}).setdefault(ds, {})[name] = ps
        res.setdefault("tests", {})[ds] = tests
        alphas = np.concatenate([r[("SF_raw", ds)][1] for r in per_seed])
        res.setdefault("gate", {})[ds] = {"alpha_mean": float(alphas.mean()), "alpha_sd": float(alphas.std(ddof=1))}
        detail.setdefault("alpha_hist", {})[ds] = np.histogram(alphas, bins=20, range=(0, 1))[0].tolist()
        if ds == "pooled":
            for key, Tk, name in (("SF_raw", "T_sf", "calibration"), ("B", "T_b", "calibration_b")):
                before = [metric_block(r[(key, ds)][0], y, bins) for r in per_seed]
                after = [metric_block(C.apply_temperature(r[(key, ds)][0], r[Tk]), y, bins) for r in per_seed]
                res[name] = {"temperature": float(np.mean([r[Tk] for r in per_seed])),
                             "ece_before": float(np.mean([m["ece"] for m in before])),
                             "ece_after": float(np.mean([m["ece"] for m in after])),
                             "brier_before": float(np.mean([m["brier"] for m in before])),
                             "brier_after": float(np.mean([m["brier"] for m in after]))}
                detail.setdefault("reliability", {})[name] = {
                    "before": C.reliability(np.concatenate([r[(key, ds)][0] for r in per_seed]), np.tile(y, len(per_seed)), bins),
                    "after": C.reliability(np.concatenate([C.apply_temperature(r[(key, ds)][0], r[Tk]) for r in per_seed]),
                                           np.tile(y, len(per_seed)), bins)}
    detail["histories"] = {s: r["hist"] for s, r in zip(seeds, per_seed)}
    return res, detail


def e1b(ar: pd.DataFrame, wv, cfg: dict, seeds: list[int], smoke: bool) -> dict:
    """SF and B trained and evaluated on Araci's 64/16/20 split of each PhraseBank set."""
    out = {}
    for ds in ("phrasebank_all", "phrasebank50"):
        d = ar[ar["dataset"] == ds]
        tr, va, te = (d[d["split"] == s].reset_index(drop=True) for s in ("train", "val", "test"))
        y = te["label"].to_numpy()
        rows = {"B": [], "SF": []}
        for s in seeds:
            r = run_seed(s, tr, va, {"test": te}, wv, cfg, None, smoke,
                         None if smoke else OUT / "branch_b_e1b" / ds / f"seed{s}")
            rows["B"].append({"acc": float((r[("B", "test")][0].argmax(1) == y).mean()),
                              "f1": macro_f1(y, r[("B", "test")][0].argmax(1))})
            psf = C.apply_temperature(r[("SF_raw", "test")][0], r["T_sf"])
            rows["SF"].append({"acc": float((psf.argmax(1) == y).mean()), "f1": macro_f1(y, psf.argmax(1))})
        out[ds] = {k: summarise(v) for k, v in rows.items()}
    return out


def pretrain_b() -> None:
    """Branch B for E1 (seeds 0-4, union of the training splits) and E1-b (Araci's split of each PhraseBank set)."""
    cfg = frozen()
    df, ar = S.load()
    tr, va = split_frame(df, "train"), split_frame(df, "val")
    jobs = [(OUT / "branch_b" / f"seed{s}", tr, va, s) for s in cfg["sentiflow"]["seeds"]]
    for ds in ("phrasebank_all", "phrasebank50"):
        d = ar[ar["dataset"] == ds]
        jobs += [(OUT / "branch_b_e1b" / ds / f"seed{s}", d[d["split"] == "train"].reset_index(drop=True),
                  d[d["split"] == "val"].reset_index(drop=True), s) for s in cfg["sentiflow"]["seeds"]]
    for cache, a, b, s in jobs:
        if (cache / "branch_b_model.pt").exists():
            continue
        with run_record("P2", "branch_b_pretrain", s, str(T.DEV), {"cache": str(cache.relative_to(PROCESSED))}):
            tok, enc = finbert.load_pretrain()
            mb, hist = branch_b(tok, enc, a, b, s, cfg["sentiflow"]["branch_b"], cache)
        print(cache.relative_to(PROCESSED), "best epoch", hist["best_epoch"], "val F1", round(hist["best_score"], 4), flush=True)
        del mb
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip_e1b", action="store_true")
    ap.add_argument("--pretrain_b", action="store_true", help="train and cache Branch B for every seed and split")
    args = ap.parse_args()
    if args.pretrain_b:
        pretrain_b()
        return
    cfg = frozen()
    df, ar = S.load()
    if args.smoke:
        tr = split_frame(df, "train")
        from gensim.models import Word2Vec
        wv = Word2Vec([word2vec.tokenize(s) for s in tr["sentence"]], vector_size=300, min_count=1, sg=0, epochs=1,
                      seed=2026, workers=1, hashfxn=word2vec._hash).wv
        seeds, reps = [0], 200
    else:
        wv = word2vec.load()
        seeds, reps = cfg["sentiflow"]["seeds"], 5000
    with run_record("P2", "E1_sentiflow" + ("_smoke" if args.smoke else ""), None, str(T.DEV),
                    {"seeds": seeds, "smoke": args.smoke}) as run:
        res, detail = e1(df, wv, cfg, seeds, args.smoke, reps)
        if not args.skip_e1b:
            res["e1b"] = e1b(ar, wv, cfg, seeds, args.smoke)
        run.set(summary={k: v for k, v in res.items() if k in ("pooled", "calibration")})
    target = RESULTS / ("p2_e1_smoke.json" if args.smoke else "p2_e1_detail.json")
    target.write_text(json.dumps(R._clean({"res": res, "detail": detail}), indent=1, default=str), encoding="utf-8")
    if args.smoke:
        print(json.dumps(R._clean(res["pooled"]), indent=1))
        return
    items = []
    for ds in (*DATASETS, "pooled"):
        for sm, m in res[ds].items():
            items += [(f"sentiflow.{ds}.{sm}.{k}", v) for k, v in m.items()]
        items += [(f"sentiflow.tests.{ds}.{k}", v) for k, v in res["tests"][ds].items()]
        items += [(f"sentiflow.gate.{ds}.{k}", v) for k, v in res["gate"][ds].items()]
    for name in ("calibration", "calibration_b"):
        items += [(f"sentiflow.{name}.{k}", v) for k, v in res[name].items()]
    for ds, block in res.get("e1b", {}).items():
        for sm, m in block.items():
            items += [(f"sentiflow.e1b.{ds}.{sm}.{k}", v) for k, v in m.items()]
    items.append(("p2.lex_thresholds", res["lex_thresholds"]))
    notes = ["sentiflow.pooled: union of the test splits of phrasebank50, fiqa and twitter (phrasebank_all is a subset "
             "of phrasebank50 and is not counted twice).",
             "sentiflow.<DS>.SF: calibrated SentiFlow; A, A_REV and B are the raw branches (their calibration is in "
             "sentiflow.calibration_b for B). sentiflow.calibration.temperature is the mean of the five per-seed temperatures."]
    R.update(items, notes)


if __name__ == "__main__":
    main()
