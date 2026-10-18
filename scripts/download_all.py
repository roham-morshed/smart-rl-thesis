"""Every large download the project still needs, in one resumable command (run when internet time is available).

    python scripts/download_all.py            # download, then verify
    python scripts/download_all.py --check    # only report what is missing

Currently: the news subsets approved at G0 (configs/frozen.yaml: news.subsets_keep) from
Brianferrell787/financial-news-multisource into the Hugging Face cache of configs/paths.yaml. Interrupted runs
resume where they stopped; files already complete are skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smartrl.config import frozen  # noqa: E402
from smartrl.data import hf  # noqa: E402

NEWS_REPO = "Brianferrell787/financial-news-multisource"


def remote_files(subsets: list[str]) -> list:
    from huggingface_hub import HfApi
    api = HfApi(token=hf.token())
    files = []
    for s in subsets:
        files += [f for f in api.list_repo_tree(NEWS_REPO, repo_type="dataset", path_in_repo=f"data/{s}", recursive=True)
                  if getattr(f, "size", None) is not None and f.path.endswith(".parquet")]
    return files


def local_path(path_in_repo: str) -> Path | None:
    from huggingface_hub import try_to_load_from_cache
    p = try_to_load_from_cache(NEWS_REPO, path_in_repo, cache_dir=hf.cache_dir(), repo_type="dataset")
    return Path(p) if isinstance(p, str) else None


def status(files: list) -> tuple[list, int]:
    missing, done_bytes = [], 0
    for f in files:
        p = local_path(f.path)
        if p is not None and p.exists() and p.stat().st_size == f.size:
            done_bytes += f.size
        else:
            missing.append(f)
    return missing, done_bytes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if not hf.token():
        sys.exit("HF_TOKEN not found (set it with: setx HF_TOKEN \"hf_...\")")
    subsets = frozen()["news"]["subsets_keep"]
    files = remote_files(subsets)
    total = sum(f.size for f in files)
    missing, done = status(files)
    print(f"news: {len(files)} files, {total / 2**30:.2f} GB; complete {done / 2**30:.2f} GB; missing {len(missing)} files",
          flush=True)
    if args.check or not missing:
        return
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    for k, f in enumerate(sorted(missing, key=lambda x: x.size), 1):
        for attempt in range(6):
            try:
                hf_hub_download(NEWS_REPO, f.path, repo_type="dataset", cache_dir=hf.cache_dir(), token=hf.token())
                break
            except Exception as e:  # network drop: wait and resume the same file
                print(f"  retry {attempt + 1} for {f.path}: {type(e).__name__}: {e}", flush=True)
                time.sleep(30 * (attempt + 1))
        else:
            print(f"  FAILED {f.path}; rerun the script to resume", flush=True)
            continue
        done += f.size
        rate = done / max(time.time() - t0, 1) / 2**20
        print(f"[{k}/{len(missing)}] {f.path} ({f.size / 2**20:.0f} MB) — {done / 2**30:.2f}/{total / 2**30:.2f} GB, "
              f"{rate:.1f} MB/s", flush=True)
    missing, done = status(files)
    print(f"done: missing {len(missing)} files; complete {done / 2**30:.2f} of {total / 2**30:.2f} GB", flush=True)


if __name__ == "__main__":
    main()
