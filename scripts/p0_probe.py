"""P0 data-access probe (P0 plan, step 2-3): environment, prices, dividend ETFs, news corpus, labelled sentiment
sets, FinBERT checkpoints, LM dictionary, ETT files.

Writes results/p0_probe.json (full detail, including exact errors) and the `p0.*` and `meta.*` keys of
results/results.json. Each section can be run alone: python scripts/p0_probe.py --only yf,ett
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import RAW, RESULTS, ROOT, config_hash, frozen, paths  # noqa: E402
from smartrl.data import hf, prices  # noqa: E402
from smartrl.runs import git_commit, hardware, package_versions, run_record  # noqa: E402

PROBE_JSON = RESULTS / "p0_probe.json"
NEWS_REPO = "Brianferrell787/financial-news-multisource"


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- environment
def probe_env() -> dict:
    out: dict = {"hardware": hardware()}
    du = shutil.disk_usage(ROOT)
    out["disk_free_gb"] = du.free / 2**30
    fonts = Path(paths()["fonts_dir"])
    cfg = frozen()["figures"]
    found = sorted(p.name for p in fonts.glob("*") if "nazanin" in p.name.lower())
    out["font_b_nazanin"] = found
    vazir = sorted(p.name for p in fonts.glob("*") if "vazir" in p.name.lower())
    local_vazir = sorted(p.name for p in (ROOT.parent / "thesis_writing" / "figures" / "diagrams").glob("Vazirmatn*.ttf"))
    out["font_vazirmatn"] = vazir or local_vazir
    out["font_used"] = cfg["font"] if found else cfg["fallback_font"]
    try:
        import torch
        out["torch_matmul_ok"] = bool(torch.cuda.is_available() and
                                      float((torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda"))[0, 0]) == 64.0)
    except Exception as e:
        out["torch_matmul_ok"] = _err(e)
    return out


# ---------------------------------------------------------------- prices and dividend ETFs
def _series_stats(df: pd.DataFrame) -> dict:
    d = pd.to_datetime(df["date"])
    vol = df["volume"].fillna(0)
    return {"first_date": str(d.min().date()), "last_date": str(d.max().date()), "rows": int(len(df)),
            "missing_close": int(df["close"].isna().sum()), "share_zero_volume": float((vol <= 0).mean()),
            "n_dividends": int((df["dividends"].fillna(0) > 0).sum())}


def probe_yf() -> dict:
    cfg = frozen()
    end = cfg["dates"]["raw_end"]
    out: dict = {"tickers": {}, "etf_check": {}, "markets": {}}
    frames: dict[str, pd.DataFrame] = {}
    tickers = []
    for mkt, m in cfg["markets"].items():
        tickers += [m["index"]] + [t for t in (m["total_return"], m["dividend_etf"]) if t]
    for t in tickers:
        try:
            df = prices.fetch(t, end_inclusive=end)
            frames[t] = df
            out["tickers"][t] = {"ok": True, **_series_stats(df)}
        except Exception as e:
            out["tickers"][t] = {"ok": False, "error": _err(e)}
    lo, hi = (pd.Timestamp(x) for x in cfg["dividend_etf_check"]["window"])
    thr = cfg["dividend_etf_check"]["min_return_correlation"]
    firsts = []
    for mkt, m in cfg["markets"].items():
        idx = frames.get(m["index"])
        if idx is None:
            continue
        firsts.append(pd.to_datetime(idx["date"]).min())
        cal = pd.to_datetime(idx["date"])
        mk = {"index": m["index"], "first_date": str(cal.min().date()), "last_date": str(cal.max().date()),
              "rows": int(len(idx))}
        for other in (m["total_return"], m["dividend_etf"]):
            if other and other in frames:
                od = set(pd.to_datetime(frames[other]["date"]))
                in_win = cal[(cal >= max(lo, pd.to_datetime(frames[other]["date"]).min())) & (cal <= hi)]
                mk[f"{other}_index_days_missing_2015_2024"] = int(sum(d not in od for d in in_win))
        out["markets"][mkt] = mk
        etf = m["dividend_etf"] or m["total_return"]
        if etf and etf in frames:
            a = idx.set_index("date")["close"]
            b = frames[etf].set_index("date")["close"]
            j = pd.concat([a, b], axis=1, keys=["idx", "etf"]).dropna()
            j = j[(j.index >= lo) & (j.index <= hi)]
            ra = j["idx"].pct_change()
            rb = j["etf"].pct_change()
            ok = ra.notna() & rb.notna() & (j["etf"].diff() != 0)   # ignore stale (unchanged) ETF quotes too
            corr_all = float(ra.corr(rb))
            corr_traded = float(ra[ok].corr(rb[ok]))
            divs = frames[etf]
            divs = divs[divs["dividends"].fillna(0) > 0]
            res = {"ticker": etf, "kind": "dividend_etf" if m["dividend_etf"] else "total_return",
                   "corr_daily_returns_2015_2024": corr_all, "corr_traded_days_only": corr_traded,
                   "n_common_days": int(len(j)), "n_dividend_events": int(len(divs)),
                   "dividend_first": str(pd.to_datetime(divs["date"]).min().date()) if len(divs) else None,
                   "dividend_last": str(pd.to_datetime(divs["date"]).max().date()) if len(divs) else None}
            if m["dividend_etf"]:
                res["passes"] = bool(corr_all >= thr and len(divs) > 0)
                # trailing-12-month yield of the ETF (dividends of the last 365 days / close), yearly mean
                e = frames[etf].set_index("date").sort_index()
                ttm = e["dividends"].fillna(0).rolling("365D").sum() / e["close"]
                res["ttm_yield_mean_by_year"] = {str(y): float(v) for y, v in ttm.groupby(ttm.index.year).mean().items()
                                                 if y >= 2008}
            out["etf_check"][mkt] = res
    if firsts:
        out["first_common_index_date"] = str(max(firsts).date())
    return out


# ---------------------------------------------------------------- ETT
def probe_ett() -> dict:
    import requests
    out = {}
    d = RAW / "ett"
    d.mkdir(parents=True, exist_ok=True)
    for name in ("ETTh1", "ETTm1"):
        url = f"https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
        p = d / f"{name}.csv"
        try:
            if not p.exists():
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                p.write_bytes(r.content)
            df = pd.read_csv(p)
            out[name] = {"ok": True, "url": url, "rows": int(len(df)), "columns": list(df.columns),
                         "first": str(df["date"].iloc[0]), "last": str(df["date"].iloc[-1])}
        except Exception as e:
            out[name] = {"ok": False, "url": url, "error": _err(e)}
    return out


# ---------------------------------------------------------------- labelled sentiment sets
def _phrasebank(cache: Path) -> dict:
    from huggingface_hub import hf_hub_download
    zp = hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset",
                         cache_dir=cache, token=hf.token())
    out = {}
    dst = RAW / "sentiment"
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        for key, member in (("phrasebank50", "Sentences_50Agree.txt"), ("phrasebank_all", "Sentences_AllAgree.txt")):
            name = next(n for n in z.namelist() if n.endswith(member))
            rows = []
            for line in z.read(name).decode("latin-1").splitlines():
                line = line.strip()
                if not line:
                    continue
                sent, _, lab = line.rpartition("@")
                rows.append((sent.strip(), lab.strip()))
            df = pd.DataFrame(rows, columns=["sentence", "label"])
            df.to_parquet(dst / f"{key}.parquet", index=False)
            out[key] = {"ok": True, "rows": int(len(df)), "labels": df["label"].value_counts().to_dict(),
                        "source": "takala/financial_phrasebank data/FinancialPhraseBank-v1.0.zip"}
    return out


def _twitter(cache: Path) -> dict:
    from huggingface_hub import hf_hub_download
    out = {}
    dst = RAW / "sentiment"
    for split, f in (("train", "sent_train.csv"), ("valid", "sent_valid.csv")):
        p = hf_hub_download("zeroshot/twitter-financial-news-sentiment", f, repo_type="dataset", cache_dir=cache,
                            token=hf.token())
        df = pd.read_csv(p)
        df.to_parquet(dst / f"twitter_{split}.parquet", index=False)
        out[split] = {"rows": int(len(df)), "columns": list(df.columns),
                      "labels": {str(k): int(v) for k, v in df["label"].value_counts().items()}}
    return {"ok": True, **out}


FIQA_CANDIDATES = ("pauri32/fiqa-2018", "TheFinAI/fiqa-sentiment-classification", "TheFinAI/en-fiqa-absa",
                   "ChanceFocus/fiqa-sentiment-classification")


def _fiqa(cache: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf.token())
    out = {}
    for repo in FIQA_CANDIDATES:
        try:
            files = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.endswith((".csv", ".parquet", ".json", ".jsonl"))]
            splits = {}
            for f in files:
                p = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache, token=hf.token())
                df = (pd.read_csv(p) if f.endswith(".csv") else pd.read_parquet(p) if f.endswith(".parquet")
                      else pd.read_json(p, lines=f.endswith(".jsonl")))
                info = {"rows": int(len(df)), "columns": list(df.columns)}
                score_col = next((c for c in ("score", "sentiment_score") if c in df.columns), None)
                if score_col:
                    s = pd.to_numeric(df[score_col], errors="coerce")
                    info.update(score_col=score_col, score_min=float(s.min()), score_max=float(s.max()),
                                n_pos=int((s > 0.1).sum()), n_neg=int((s < -0.1).sum()),
                                n_neu=int(((s >= -0.1) & (s <= 0.1)).sum()))
                if "sentence" in df.columns:
                    info["example"] = str(df["sentence"].iloc[0])[:120]
                    info["n_unique_sentences"] = int(df["sentence"].astype(str).str.strip().str.lower().nunique())
                splits[f] = info
            has = all(("sentence" in v["columns"] and "score_col" in v) for v in splits.values()) and bool(splits)
            out[repo] = {"ok": True, "has_sentence_and_score": has, "files": splits}
        except Exception as e:
            out[repo] = {"ok": False, "error": _err(e)}
    return out


def probe_sets() -> dict:
    cache = hf.cache_dir()
    out = {}
    for key, fn in (("phrasebank", _phrasebank), ("twitter", _twitter), ("fiqa", _fiqa)):
        try:
            out[key] = fn(cache)
        except Exception as e:
            out[key] = {"ok": False, "error": _err(e), "trace": traceback.format_exc()[-1500:]}
    return out


# ---------------------------------------------------------------- models and lexicon
def probe_models() -> dict:
    import torch
    from smartrl.sentiflow import finbert
    out = {}
    for repo in (finbert.PRETRAIN, finbert.PROSUS):
        try:
            if repo == finbert.PROSUS:
                tok, m, labels = finbert.load_prosus()
            else:
                (tok, m), labels = finbert.load_pretrain(), None
            text = "Operating profit rose to EUR 13.1 mn from EUR 8.7 mn."
            enc = tok([text], return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                o = m(**enc)
            out[repo] = {"ok": True, "path": str(finbert.snapshot(repo)),
                         "n_params": int(sum(x.numel() for x in m.parameters())),
                         "vocab": int(len(tok.get_vocab())), "tokens": tok.convert_ids_to_tokens(enc["input_ids"][0]),
                         "id2label": labels,
                         "output_shape": list((o.logits if hasattr(o, "logits") else o.last_hidden_state).shape)}
        except Exception as e:
            out[repo] = {"ok": False, "error": _err(e), "trace": traceback.format_exc()[-1500:]}
    try:
        import pysentiment2 as ps
        lm = ps.LM()
        s = lm.get_score(lm.tokenize("The company reported a severe loss and weak demand, but strong growth."))
        out["pysentiment2_LM"] = {"ok": True, "example_score": s}
    except Exception as e:
        out["pysentiment2_LM"] = {"ok": False, "error": _err(e), "trace": traceback.format_exc()[-1500:]}
    return out


# ---------------------------------------------------------------- news corpus inventory
def probe_hf_news(read_dates: bool = True) -> dict:
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    tok = hf.token()
    out: dict = {"repo": NEWS_REPO, "token_found": bool(tok)}
    if not tok:
        out.update(ok=False, error="HF_TOKEN not set (user env var or process env)")
        return out
    api = HfApi(token=tok)
    try:
        who = api.whoami()
        out["hf_user"] = who.get("name")
    except Exception as e:
        out.update(ok=False, error=f"token rejected: {_err(e)}")
        return out
    try:
        files = [f for f in api.list_repo_tree(NEWS_REPO, repo_type="dataset", path_in_repo="data", recursive=True)
                 if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
    except Exception as e:
        out.update(ok=False, error=f"cannot list files (terms accepted?): {_err(e)}")
        return out
    fs = HfFileSystem(token=tok)
    subsets: dict[str, dict] = {}
    for f in files:
        sub = f.path.split("/")[1]
        s = subsets.setdefault(sub, {"files": 0, "bytes": 0, "rows": 0, "date_min": None, "date_max": None,
                                     "n_files_without_stats": 0})
        s["files"] += 1
        s["bytes"] += int(f.size)
        try:
            with fs.open(f"datasets/{NEWS_REPO}/{f.path}", "rb") as fh:
                pf = pq.ParquetFile(fh)
                md = pf.metadata
                s["rows"] += md.num_rows
                names = pf.schema_arrow.names
                di = names.index("date") if "date" in names else None
                lo = hi = None
                if di is not None:
                    for rg in range(md.num_row_groups):
                        st = md.row_group(rg).column(di).statistics
                        if st is None or not st.has_min_max:
                            lo = hi = None
                            break
                        mn, mx = str(st.min), str(st.max)
                        lo = mn if lo is None or mn < lo else lo
                        hi = mx if hi is None or mx > hi else hi
                    if lo is None and read_dates:
                        s["n_files_without_stats"] += 1
                        col = pf.read(columns=["date"]).column("date").to_pandas().astype(str)
                        lo, hi = col.min(), col.max()
                if lo is not None:
                    s["date_min"] = lo if s["date_min"] is None or lo < s["date_min"] else s["date_min"]
                    s["date_max"] = hi if s["date_max"] is None or hi > s["date_max"] else s["date_max"]
                s["columns"] = names
        except Exception as e:
            s.setdefault("errors", []).append(f"{f.path}: {_err(e)}")
    for s in subsets.values():
        s["gb"] = s["bytes"] / 2**30
    out.update(ok=True, n_files=len(files), total_gb=sum(s["bytes"] for s in subsets.values()) / 2**30,
               total_rows=sum(s["rows"] for s in subsets.values()), subsets=subsets)
    return out


SECTIONS = {"env": probe_env, "yf": probe_yf, "ett": probe_ett, "sets": probe_sets, "models": probe_models,
            "hf": probe_hf_news}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=",".join(SECTIONS))
    args = ap.parse_args()
    probe = json.loads(PROBE_JSON.read_text(encoding="utf-8")) if PROBE_JSON.exists() else {}
    with run_record("P0", "probe", None, "cpu", {"sections": args.only}) as run:
        for name in args.only.split(","):
            print(f"== {name}", flush=True)
            try:
                probe[name] = SECTIONS[name]()
            except Exception as e:
                probe[name] = {"ok": False, "error": _err(e), "trace": traceback.format_exc()[-2000:]}
            probe[name]["probed_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            PROBE_JSON.write_text(json.dumps(R._clean(probe), indent=1, ensure_ascii=False, default=str),
                                  encoding="utf-8")
            print(json.dumps(R._clean(probe[name]), indent=1, default=str)[:4000], flush=True)
        run.set(outputs=[str(PROBE_JSON.relative_to(ROOT))])
    R.update([("meta.config_hash", config_hash()), ("meta.git_commit", git_commit()),
              ("meta.packages", package_versions()), ("p0.probe", probe)])


if __name__ == "__main__":
    main()
