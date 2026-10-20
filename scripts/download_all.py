"""Every large download the project still needs, in one resumable command.

    python scripts/download_all.py                 # download with parallel connections until everything is complete
    python scripts/download_all.py --check         # only report what is missing
    python scripts/download_all.py --links         # (re)write the link list for manual download

Files go to data/raw/news_multisource/<subset>/<file>.parquet (the same place a manual download should use).
A file counts as complete only when its size equals the size in manifest.json. Partial downloads are kept as
<file>.part and resumed (HTTP Range); stalled or dropped connections are retried until the file is complete.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.config import RAW  # noqa: E402
from smartrl.data import hf  # noqa: E402

NEWS_REPO = "Brianferrell787/financial-news-multisource"
NEWS_DIR = RAW / "news_multisource"
MANIFEST = NEWS_DIR / "manifest.json"
BASE_URL = f"https://huggingface.co/datasets/{NEWS_REPO}/resolve/main/"
_lock = threading.Lock()


def manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def target(f: dict) -> Path:
    return NEWS_DIR / f["subset"] / Path(f["path"]).name


def complete(f: dict) -> bool:
    p = target(f)
    return p.exists() and p.stat().st_size == f["size"]


def organize() -> int:
    """Move parquet files saved anywhere under data/raw/news_multisource (e.g. all in one folder by a download
    manager) into their subset folders; names are matched with manifest.json. Returns the number moved."""
    by_name = {Path(f["path"]).name: f for f in manifest()}
    moved = 0
    for p in NEWS_DIR.rglob("*.parquet"):
        f = by_name.get(p.name)
        if f is None or p == target(f):
            continue
        dst = target(f)
        if dst.exists() and dst.stat().st_size == f["size"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(p, dst)
        moved += 1
    for f in manifest():                                  # a finished file makes its partial download obsolete
        part = target(f).with_suffix(".parquet.part")
        if complete(f) and part.exists():
            part.unlink()
    return moved


def write_links() -> Path:
    """LINKS.txt: one URL per line, only the files that are not complete yet."""
    files = [f for f in sorted(manifest(), key=lambda x: (x["subset"], x["path"])) if not complete(f)]
    out = NEWS_DIR / "LINKS.txt"
    out.write_text("\n".join(f"{BASE_URL}{f['path']}?download=true" for f in files) + "\n", encoding="utf-8")
    (NEWS_DIR / "LINKS_urls_only.txt").unlink(missing_ok=True)
    for s in {f["subset"] for f in manifest()}:
        (NEWS_DIR / s).mkdir(parents=True, exist_ok=True)
    return out


def fetch(f: dict, header_file: str, log) -> bool:
    """Resume one file with curl until it is complete or `max_tries` attempts failed."""
    p = target(f)
    part = p.with_suffix(p.suffix + ".part")
    p.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 400):
        if complete(f):
            return True
        have = part.stat().st_size if part.exists() else 0
        if have > f["size"]:
            part.unlink()
            have = 0
        cmd = ["curl", "-sS", "-L", "-C", "-", "-H", f"@{header_file}", "--connect-timeout", "30",
               "--speed-limit", "2000", "--speed-time", "90", "-o", str(part), BASE_URL + f["path"]]
        r = subprocess.run(cmd, capture_output=True, text=True)
        size = part.stat().st_size if part.exists() else 0
        if size == f["size"]:
            os.replace(part, p)
            return True
        if r.returncode == 33:                        # server ignored the range: start over
            part.unlink(missing_ok=True)
        if attempt % 5 == 1:
            log(f"  retry {attempt} {f['path']}: {size / 2**20:.1f}/{f['size'] / 2**20:.1f} MB (curl {r.returncode} "
                f"{r.stderr.strip()[:120]})")
        time.sleep(min(10 * attempt, 120))
    return complete(f)


def run(workers: int) -> None:
    tok = hf.token()
    if not tok:
        sys.exit("HF_TOKEN not found")
    files = manifest()
    todo = sorted([f for f in files if not complete(f)], key=lambda x: x["size"])
    total = sum(f["size"] for f in files)
    log_path = Path(__file__).resolve().parents[1] / "results" / "logs" / "download_news.log"
    t0 = time.time()

    def log(msg):
        with _lock:
            print(msg, flush=True)

    def done_bytes():
        return sum(f["size"] for f in files if complete(f)) + sum(
            target(f).with_suffix(".parquet.part").stat().st_size for f in files
            if not complete(f) and target(f).with_suffix(".parquet.part").exists())

    log(f"news: {len(files)} files, {total / 2**30:.2f} GB; to download {len(todo)} files with {workers} connections")
    fd, header = tempfile.mkstemp(suffix=".hdr")
    with os.fdopen(fd, "w") as h:
        h.write(f"Authorization: Bearer {tok}\n")
    stop = threading.Event()

    def progress():
        last = done_bytes()
        while not stop.wait(300):
            now = done_bytes()
            n_ok = sum(complete(f) for f in files)
            log(f"[{time.strftime('%H:%M')}] {now / 2**30:.2f}/{total / 2**30:.2f} GB, {n_ok}/{len(files)} files, "
                f"{(now - last) / 300 / 2**20:.2f} MB/s over the last 5 min")
            last = now
    threading.Thread(target=progress, daemon=True).start()
    try:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda f: (f["path"], fetch(f, header, log)), todo))
    finally:
        stop.set()
        os.unlink(header)
    bad = [p for p, ok in results if not ok]
    log(f"done in {(time.time() - t0) / 3600:.2f} h; incomplete: {len(bad)} {bad[:5]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--links", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.links:
        organize()
        print(write_links())
        return
    moved = organize()
    if moved:
        print(f"moved {moved} manually downloaded files into their subset folders", flush=True)
    files = manifest()
    missing = [f for f in files if not complete(f)]
    print(f"news: {len(files)} files, {sum(f['size'] for f in files) / 2**30:.2f} GB; complete "
          f"{sum(f['size'] for f in files if complete(f)) / 2**30:.2f} GB; missing {len(missing)} files", flush=True)
    if args.check or not missing:
        return
    run(args.workers)


if __name__ == "__main__":
    main()
