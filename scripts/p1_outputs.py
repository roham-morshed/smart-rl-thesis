"""P1 outputs (E0): table T01 (prices), table T02 and figure F01 (news; skipped until the news corpus is built)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.config import MARKETS, TABLES, frozen  # noqa: E402
from smartrl.figures import style  # noqa: E402


def t01(res: dict) -> None:
    cfg = frozen()
    rows = []
    for m in MARKETS:
        p = res["data"]["prices"][m]
        mk = cfg["markets"][m]
        rows.append({"market": m, "index_ticker": mk["index"], "pnl_series": mk["total_return"] or "price + dy/252",
                     "dividend_source": p["dividend_source"], "start": p["start"], "end": p["end"],
                     "n_days": p["n_days"], "n_flagged_returns": p["n_flagged_returns"],
                     "volume_used": res["p1"]["prices"][m]["volume_used"]})
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "T01.csv", index=False)


def t02_f01(res: dict) -> bool:
    if "news" not in res.get("data", {}) or "market" not in res["data"]["news"]:
        return False
    n = res["data"]["news"]
    rows = [{"kind": "source", "name": s, "n_raw": v.get("n_raw"), "n_kept": v.get("n_kept"),
             "first_date": v.get("first_date"), "last_date": v.get("last_date")} for s, v in n["source"].items()]
    rows += [{"kind": "tag", "name": t, "n_kept": v["n"]} for t, v in n["tag"].items()]
    rows += [{"kind": "market", "name": m, "mean_daily_local": v["mean_daily_local"],
              "share_days_with_local": v["share_days_with_local"], "mean_daily_global": v["mean_daily_global"]}
             for m, v in n["market"].items()]
    rows.append({"kind": "total", "name": "all", "n_raw": n["n_raw"], "n_kept": n["n_kept"],
                 "n_duplicates": n["n_duplicates"]})
    pd.DataFrame(rows).to_csv(TABLES / "T02.csv", index=False)
    fig, ax = style.figure("F01")
    for m in MARKETS:
        s = pd.Series(n["market"][m]["share_days_with_local_by_year"]).astype(float)
        s.index = s.index.astype(int)
        s = s[(s.index >= 2008) & (s.index <= 2026)]
        ax.plot(s.index, s.values, marker="o", ms=3, lw=1.4, label=style.market(m))
    ax.set_xlabel(style.label("Year", "سال (Year)"))
    ax.set_ylabel(style.label("Share of trading days with local news", "سهم روزهای دارای خبر محلی (Share of days)"))
    style.ticks(ax, "x", 0)
    style.ticks(ax, "y", 0, percent=True)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", frameon=False)
    style.save(fig, "F01")
    return True


def main() -> None:
    res = R.load()
    t01(res)
    print("T01 written")
    print("T02/F01 written" if t02_f01(res) else "T02/F01 skipped: news corpus not built yet")


if __name__ == "__main__":
    main()
