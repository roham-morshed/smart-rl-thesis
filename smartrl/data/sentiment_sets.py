"""Labelled sentiment datasets and their splits (SPEC §2.4). Labels: 0 = negative, 1 = neutral, 2 = positive.

phrasebank_all is a strict subset of phrasebank50 (every all-agree sentence is also a 50 %-agree sentence with the
same label), and SentiFlow trains on phrasebank50. A split of phrasebank_all drawn independently would put most of
its validation and test sentences inside phrasebank50's training split, and the overlap rule of §2.4 would then
delete them. Its splits are therefore nested: an all-agree sentence belongs to the split its phrasebank50 copy is
in (mode "nested"). Mode "independent" (own stratified split, then the overlap rule) is kept for comparison.
"""
from __future__ import annotations

import re
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import PROCESSED, RAW, frozen
from . import hf

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
SENT_DIR = RAW / "sentiment"


def norm(s: str) -> str:
    """Normalisation used for duplicate and overlap checks, the same as the news key of SPEC §2.3:
    lower case, non-alphanumerics removed, whitespace collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z\s]+", "", str(s).lower())).strip()


def _strat3(df: pd.DataFrame, fracs: tuple[float, float, float], seed: int) -> pd.Series:
    """Stratified train/val/test labels with the given fractions."""
    tr, va, te = fracs
    idx = np.arange(len(df))
    i_tr, i_rest = train_test_split(idx, train_size=tr, stratify=df["label"], random_state=seed)
    i_va, i_te = train_test_split(i_rest, train_size=va / (va + te), stratify=df["label"].iloc[i_rest],
                                  random_state=seed)
    s = pd.Series("", index=df.index)
    s.iloc[i_tr], s.iloc[i_va], s.iloc[i_te] = "train", "val", "test"
    return s


def _araci(df: pd.DataFrame, seed: int) -> pd.Series:
    """Araci (2019): 20 % test, then 20 % of the rest for validation (64/16/20), stratified."""
    idx = np.arange(len(df))
    i_rest, i_te = train_test_split(idx, test_size=0.20, stratify=df["label"], random_state=seed)
    i_tr, i_va = train_test_split(i_rest, test_size=0.20, stratify=df["label"].iloc[i_rest], random_state=seed)
    s = pd.Series("", index=df.index)
    s.iloc[i_tr], s.iloc[i_va], s.iloc[i_te] = "train", "val", "test"
    return s


def load_raw() -> dict[str, pd.DataFrame]:
    cfg = frozen()["sentiment_sets"]
    out = {}
    for key in ("phrasebank50", "phrasebank_all"):
        d = pd.read_parquet(SENT_DIR / f"{key}.parquet")
        d = pd.DataFrame({"sentence": d["sentence"].astype(str), "label": d["label"].map(LABELS)})
        out[key] = d.assign(k=d["sentence"].map(norm)).drop_duplicates("k").drop(columns="k").reset_index(drop=True)
    snap = glob(str(hf.cache_dir() / "datasets--pauri32--fiqa-2018" / "snapshots" / "*"))[0]
    f = pd.concat([pd.read_csv(f"{snap}/{s}.csv") for s in ("train", "validation", "test")], ignore_index=True)
    f = f.assign(k=f["sentence"].astype(str).map(norm))
    g = f.groupby("k", sort=False).agg(sentence=("sentence", "first"), score=("sentiment_score", "mean")).reset_index(drop=True)
    lo, hi = cfg["fiqa_thresholds"]
    g["label"] = np.where(g["score"] > hi, 2, np.where(g["score"] < lo, 0, 1))
    out["fiqa"] = g[["sentence", "label"]]
    tw_map = {0: 0, 1: 2, 2: 1}                    # 0 bearish -> neg, 1 bullish -> pos, 2 neutral -> neu
    for split in ("train", "valid"):
        d = pd.read_parquet(SENT_DIR / f"twitter_{split}.parquet")
        out[f"twitter_{split}"] = pd.DataFrame({"sentence": d["text"].astype(str), "label": d["label"].map(tw_map)})
    return out


def build(pb_all_mode: str = "nested") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Returns (main splits, Araci splits, report). Main: dataset, split, sentence, label."""
    seed = frozen()["sentiment_sets"]["split_seed"]
    raw = load_raw()
    parts = []
    pb50 = raw["phrasebank50"].copy()
    pb50["split"] = _strat3(pb50, (0.70, 0.15, 0.15), seed)
    parts.append(pb50.assign(dataset="phrasebank50"))
    pba = raw["phrasebank_all"].copy()
    if pb_all_mode == "nested":
        where = dict(zip(pb50["sentence"].map(norm), pb50["split"]))
        pba["split"] = pba["sentence"].map(norm).map(where)
        assert pba["split"].notna().all()
    else:
        pba["split"] = _strat3(pba, (0.70, 0.15, 0.15), seed)
    parts.append(pba.assign(dataset="phrasebank_all"))
    fq = raw["fiqa"].copy()
    fq["split"] = _strat3(fq, (0.70, 0.15, 0.15), seed)
    parts.append(fq.assign(dataset="fiqa"))
    tw_tr = raw["twitter_train"].assign(split="train")
    tw_v = raw["twitter_valid"].copy()
    i_va, i_te = train_test_split(np.arange(len(tw_v)), test_size=0.5, stratify=tw_v["label"], random_state=seed)
    tw_v["split"] = ""
    tw_v.iloc[i_va, tw_v.columns.get_loc("split")] = "val"
    tw_v.iloc[i_te, tw_v.columns.get_loc("split")] = "test"
    parts.append(pd.concat([tw_tr, tw_v]).assign(dataset="twitter"))
    df = pd.concat(parts, ignore_index=True)[["dataset", "split", "sentence", "label"]]
    # remove from validation and test every sentence that occurs in any training split (phrasebank_all's training
    # split is part of phrasebank50's, so the union of the four training splits is the SentiFlow training set)
    df["k"] = df["sentence"].map(norm)
    train_keys = set(df.loc[(df["split"] == "train") & (df["dataset"] != "phrasebank_all"), "k"])
    leak = (df["split"] != "train") & df["k"].isin(train_keys)
    report = {"removed_overlap": {d: int(leak[df["dataset"] == d].sum()) for d in df["dataset"].unique()}}
    df = df[~leak].drop(columns="k").reset_index(drop=True)
    report["sizes"] = {d: {s: int(((df["dataset"] == d) & (df["split"] == s)).sum()) for s in ("train", "val", "test")}
                       for d in ("phrasebank50", "phrasebank_all", "fiqa", "twitter")}
    report["labels"] = {d: {s: {int(k): int(v) for k, v in df[(df["dataset"] == d) & (df["split"] == s)]["label"]
                                .value_counts().sort_index().items()} for s in ("train", "val", "test")}
                        for d in ("phrasebank50", "phrasebank_all", "fiqa", "twitter")}
    report["pb_all_mode"] = pb_all_mode
    # Araci split (E1-b), drawn independently inside each PhraseBank set
    ar = []
    for key in ("phrasebank_all", "phrasebank50"):
        d = raw[key].copy()
        d["split"] = _araci(d, seed)
        ar.append(d.assign(dataset=key))
    ar = pd.concat(ar, ignore_index=True)[["dataset", "split", "sentence", "label"]]
    report["araci_sizes"] = {d: {s: int(((ar["dataset"] == d) & (ar["split"] == s)).sum()) for s in ("train", "val", "test")}
                             for d in ("phrasebank_all", "phrasebank50")}
    return df, ar, report


def save(df: pd.DataFrame, ar: pd.DataFrame) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "sentiment_sets.parquet", index=False)
    ar.to_parquet(PROCESSED / "sentiment_sets_araci.parquet", index=False)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(PROCESSED / "sentiment_sets.parquet"),
            pd.read_parquet(PROCESSED / "sentiment_sets_araci.parquet"))
