"""P1 (E0, prices): build data/processed/prices_<MKT>.parquet and the data.prices.* keys of results.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, PROCESSED, RESULTS, frozen  # noqa: E402
from smartrl.data import panel  # noqa: E402
from smartrl.runs import run_record  # noqa: E402


def main() -> None:
    cfg = frozen()
    end = cfg["dates"]["raw_end"]
    start = panel.first_common_date(end)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    items, details = [], {}
    with run_record("P1", "build_prices", None, "cpu", {"start": start, "end": end}) as run:
        for mkt in MARKETS:
            df, st = panel.build_market(mkt, start, end)
            df.to_parquet(panel.processed_path(mkt), index=False)
            details[mkt] = st
            for k in ("start", "end", "n_days", "dividend_source", "n_flagged_returns"):
                items.append((f"data.prices.{mkt}.{k}", st[k]))
            print(mkt, json.dumps(st), flush=True)
        run.set(outputs=[str(panel.processed_path(m).name) for m in MARKETS])
    items.append(("p1.prices", details))
    notes = []
    if details["SSE"]["dividend_source"].startswith("none"):
        notes.append("data.prices.SSE.dividend_source: no cited SSE Composite dividend yield was found (SSE and "
                     "CSIndex sources searched), so the SSE sleeve earns the index price return without dividends; "
                     "this understates SSE total returns of invested days and is a limitation (SPEC §2.1 fallback).")
    R.update(items, notes)
    (RESULTS / "p1_prices.json").write_text(json.dumps(details, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
