"""Build every table and figure whose inputs exist (results/tables, results/figures).

    python scripts/make_outputs.py            # all
    python scripts/make_outputs.py T06 F05    # selected IDs
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.evaluation import outputs  # noqa: E402


if __name__ == "__main__":
    done = outputs.build(sys.argv[1:] or None)
    print("built:", [k for k, v in done.items() if v], "| waiting for inputs:", [k for k, v in done.items() if not v])
