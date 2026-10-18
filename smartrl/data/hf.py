"""Hugging Face access: token lookup and the local cache folder of configs/paths.yaml."""
from __future__ import annotations

import os
from pathlib import Path

from ..config import paths


def token() -> str | None:
    """HF_TOKEN from the process environment, else from the Windows user environment (set with `setx`)."""
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                tok, _ = winreg.QueryValueEx(k, "HF_TOKEN")
                if tok:
                    os.environ["HF_TOKEN"] = tok
                    return tok
        except OSError:
            return None
    return None


def cache_dir() -> Path:
    p = Path(paths()["hf_cache"])
    p.mkdir(parents=True, exist_ok=True)
    return p
