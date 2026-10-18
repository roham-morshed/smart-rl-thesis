"""P1 (E0, labelled sets): build the SentiFlow splits of SPEC §2.4 and the data.sentiment_sets.* keys."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl import results as R  # noqa: E402
from smartrl.data import sentiment_sets as S  # noqa: E402
from smartrl.runs import run_record  # noqa: E402


def main() -> None:
    with run_record("P1", "sentiment_sets", None, "cpu", {"pb_all_mode": "nested"}):
        df, ar, rep = S.build("nested")
        S.save(df, ar)
        _, _, rep_ind = S.build("independent")
    items = []
    for d, sz in rep["sizes"].items():
        items += [(f"data.sentiment_sets.{d}.n_{s}", n) for s, n in sz.items()]
    rep["alternative_independent_pb_all"] = {"sizes": rep_ind["sizes"]["phrasebank_all"],
                                             "removed_overlap": rep_ind["removed_overlap"]["phrasebank_all"]}
    items.append(("p1.sentiment_sets", rep))
    R.update(items)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
