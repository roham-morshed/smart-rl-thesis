"""Pre-flight check of P5 evaluate -> analyze -> outputs -> contract keys, written only to a scratch folder.

Missing B7 (SAC) policies are replaced by the conf2021 policy of the same market and seed, so every code path
runs on the full fold structure. Nothing is written to results/, and its numbers are never used.

    python scripts/p5_dryrun.py <scratch_dir> <n_jobs>
"""
"""
import contextlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCR = Path(sys.argv[1])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def patch():
    import smartrl.results as R
    from smartrl.evaluation import confirm as C
    import p5_confirm as P
    from smartrl.evaluation import analysis as A, outputs as O
    from smartrl.figures import style as S
    rj = SCR / "results.json"
    o_load, o_update = R.__dict__.setdefault("_o_load", R.load), R.__dict__.setdefault("_o_update", R.update)
    R.load = lambda path=rj: o_load(path)
    R.update = lambda items, notes=(), path=rj: o_update(items, notes, path)
    o_pd = C.__dict__.setdefault("_o_pd", C.policy_dir)

    def pd_(fold, arm, market, seed):
        d = o_pd(fold, arm, market, seed)
        if arm == "B7" and not (d / "info.json").exists():
            d = o_pd("conf2021", arm, market, seed)
        return d
    C.policy_dir = pd_
    for m in (P, A, O):
        m.LOGS = SCR / "logs"
    O.TABLES = SCR / "tables"
    S.FIGURES = SCR / "figures"
    P.run_record = lambda *a, **k: contextlib.nullcontext()


def task(f, m):
    patch()
    import p5_confirm as P
    t0 = time.time()
    out = P.evaluate_one(f, m)
    return f, m, out, time.time() - t0


def main():
    (SCR / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results" / "results.json", SCR / "results.json")
    patch()
    import pandas as pd
    from joblib import Parallel, delayed
    import p5_confirm as P
    from smartrl.data import folds
    from smartrl.config import MARKETS
    t0 = time.time()
    tasks = [(f, m) for f in folds.fold_names()[1:] for m in MARKETS]
    res = Parallel(n_jobs=int(sys.argv[2]), backend="loky", verbose=10)(delayed(task)(f, m) for f, m in tasks)
    for f, m, _, s in res:
        print(f"evaluate {f} {m}: {s / 60:.1f} min", flush=True)
    for variant in P.VARIANTS:
        parts = [df for r in res for df in r[2].get(variant, [])]
        pd.concat(parts, ignore_index=True).to_parquet(SCR / "logs" / f"returns_{variant}.parquet", index=False)
    print(f"EVALUATE OK {(time.time() - t0) / 60:.1f} min", flush=True)
    t1 = time.time()
    from smartrl.evaluation import analysis
    analysis.run_all()
    print(f"ANALYZE OK {(time.time() - t1) / 60:.1f} min", flush=True)
    from smartrl.evaluation import outputs
    t2 = time.time()
    done = {}
    for k, fn in outputs.BUILDERS.items():
        try:
            done[k] = bool(fn(outputs.R.load()))
        except Exception:
            done[k] = "ERROR"
            print(f"OUTPUT {k} FAILED\n{traceback.format_exc()}", flush=True)
    print(f"OUTPUTS {(time.time() - t2) / 60:.1f} min:", done, flush=True)
    import smartrl.results as R
    tree = R.load()
    keys = [k.strip() for k in (ROOT / "docs" / "THESIS_KEYS.txt").read_text(encoding="utf-8").splitlines()
            if k.strip() and not k.startswith("#")]
    missing = [k for k in keys if R.lookup(tree, k) is R.MISSING]
    nulls = [k for k in keys if R.lookup(tree, k) is None]
    notes = [str(n) for n in (tree.get("notes") or [])]

    def explained(key):
        parts = key.split(".")
        return any(".".join(parts[:j]) in n for j in range(len(parts), 1, -1) for n in notes)
    unexpl = [k for k in nulls if not explained(k)]
    (SCR / "unexplained_null_keys.txt").write_text("\n".join(unexpl), encoding="utf-8")
    print(f"CONTRACT: {len(keys)} keys, {len(missing)} missing, {len(nulls)} null, {len(unexpl)} unexplained",
          flush=True)
    (SCR / "missing_keys.txt").write_text("\n".join(missing), encoding="utf-8")
    (SCR / "null_keys.txt").write_text("\n".join(nulls), encoding="utf-8")
    print("DRYRUN COMPLETE", flush=True)


if __name__ == "__main__":
    main()
