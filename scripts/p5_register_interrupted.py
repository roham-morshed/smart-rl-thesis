"""Close the run records of P5 SAC stages that were stopped from outside (user pauses, a desktop-app update).

Their meta.json files have a start but no end, so compute.* missed their hours although the 37 policies they
saved are used in P5. Conservative end: the save time of the last B7 policy written during the stage, so only
compute that produced a used policy is counted. Status "interrupted"; runs.compute_totals counts ok + interrupted.

    python scripts/p5_register_interrupted.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.config import PROCESSED, RUNS  # noqa: E402
from smartrl.runs import _registry  # noqa: E402

REASONS = {
    "P5_train_sac_sna_20261021T183431_8e33c7": "stopped on the user's request for a laptop shutdown (2026-10-22 10:14 local)",
    "P5_train_sac_sna_20261022T075131_b40bc3": "killed by a desktop-app update that ended its child processes (2026-10-22 ~17:30 local)",
    "P5_train_sac_sna_20261022T141604_0cd4bd": "stopped on the user's request for a pause (2026-10-22 20:03 local)",
}


def main() -> None:
    saves = sorted(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
                   for p in (PROCESSED / "policies").glob("*/B7/*/seed*/info.json"))
    starts = sorted(json.loads((d / "meta.json").read_text(encoding="utf-8"))["start_utc"]
                    for d in RUNS.glob("P5_train_sac_*"))
    starts = [datetime.fromisoformat(s) for s in starts]
    con = _registry()
    for run_id, reason in REASONS.items():
        mp = RUNS / run_id / "meta.json"
        meta = json.loads(mp.read_text(encoding="utf-8"))
        if meta.get("status"):
            print(run_id, "already closed:", meta["status"])
            continue
        t0 = datetime.fromisoformat(meta["start_utc"])
        nxt = min([s for s in starts if s > t0], default=None)
        mine = [s for s in saves if s > t0 and (nxt is None or s < nxt)]
        end = max(mine)
        meta.update({"end_utc": end.isoformat(timespec="seconds"), "seconds": (end - t0).total_seconds(),
                     "status": "interrupted", "policies_saved": len(mine)})
        meta["events"].append({"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "message": f"{reason}; closed afterwards with end = save time of the last policy"})
        mp.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        with con:
            con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                run_id, meta["experiment"], meta["name"], meta["seed"], meta["config_hash"], meta["git_commit"],
                meta["start_utc"], meta["end_utc"], meta["seconds"], meta["device"], "interrupted",
                json.dumps({**meta["params"], "policies_saved": len(mine)})))
        print(f"{run_id}: {len(mine)} policies, {meta['seconds'] / 3600:.2f} h")
    con.close()


if __name__ == "__main__":
    main()
