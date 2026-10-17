"""Read and update results/results.json (keys of docs/RESULTS_SCHEMA.md).

Some identifiers contain dots (PPO configuration ids such as R2_l10_d0.5_lr3e-4). They must stay one JSON
key, so writers pass the key as a list of parts; `put("a.b.c", v)` is a shortcut for dot-free keys only.
Lookup follows the same longest-match rule as tests/test_results_contract.py.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import RESULTS

RESULTS_JSON = RESULTS / "results.json"
MISSING = object()


def _clean(v: Any) -> Any:
    """Convert numpy scalars and arrays to plain JSON types; non-finite floats become None."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            v = v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def load(path: Path = RESULTS_JSON) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"meta": {}, "compute": {}, "notes": []}


def save(tree: dict, path: Path = RESULTS_JSON) -> None:
    """Atomic write so that an interrupted run never leaves a truncated results.json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tree = _clean(tree)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=1, allow_nan=False)
    os.replace(tmp, path)


def lookup(tree: dict, key: str) -> Any:
    parts, node, i = key.split("."), tree, 0
    while i < len(parts):
        if not isinstance(node, dict):
            return MISSING
        for j in range(len(parts), i, -1):
            cand = ".".join(parts[i:j])
            if cand in node:
                node, i = node[cand], j
                break
        else:
            return MISSING
    return node


def put_parts(tree: dict, parts: Sequence[str], value: Any) -> None:
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(str(p), {})
    node[str(parts[-1])] = _clean(value)


def put(tree: dict, key: str | Sequence[str], value: Any) -> None:
    parts = key.split(".") if isinstance(key, str) else list(key)
    put_parts(tree, parts, value)


def add_note(tree: dict, text: str) -> None:
    notes = tree.setdefault("notes", [])
    if text not in notes:
        notes.append(text)


def update(items: Iterable[tuple[str | Sequence[str], Any]], notes: Iterable[str] = (),
           path: Path = RESULTS_JSON) -> dict:
    """Load, write several keys and notes, stamp meta.created_utc, save. Returns the new tree."""
    tree = load(path)
    for key, value in items:
        put(tree, key, value)
    for n in notes:
        add_note(tree, n)
    tree.setdefault("meta", {})["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    save(tree, path)
    return tree
