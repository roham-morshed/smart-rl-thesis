"""Domain Word2Vec of SPEC §3.1 (CBOW on deduplicated headlines dated <= 2019-12-31)."""
from __future__ import annotations

import re
import zlib
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from ..config import PROCESSED, frozen

TOKEN_RE = re.compile(r"[a-z0-9$%']+")
W2V_PATH = PROCESSED / "word2vec_cbow300.kv"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


def _hash(s: str) -> int:
    """Deterministic replacement of Python's randomised str hash (gensim seeds each vector with hashfxn)."""
    return zlib.crc32(s.encode("utf-8"))


class _Corpus:
    """Re-iterable token stream (gensim makes one pass for the vocabulary and one per epoch)."""

    def __init__(self, headlines: Iterable[str] | callable):
        self.src = headlines

    def __iter__(self) -> Iterator[list[str]]:
        it = self.src() if callable(self.src) else self.src
        for h in it:
            toks = tokenize(h)
            if toks:
                yield toks


def train(headlines, path: Path = W2V_PATH, workers: int = 1):
    """workers=1 together with the deterministic hash makes the vectors reproducible run to run."""
    from gensim.models import Word2Vec
    c = frozen()["sentiflow"]["word2vec"]
    model = Word2Vec(sentences=_Corpus(headlines), vector_size=c["vector_size"], window=c["window"],
                     min_count=c["min_count"], sg=c["sg"], negative=c["negative"], epochs=c["epochs"],
                     seed=c["seed"], workers=workers, hashfxn=_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(path))
    return model.wv


def load(path: Path = W2V_PATH):
    from gensim.models import KeyedVectors
    return KeyedVectors.load(str(path))


class Vocab:
    """Ids for Branch A: 0 = PAD, 1 = UNK, then the Word2Vec vocabulary, then training-set words missing from it."""

    def __init__(self, wv, train_texts: Iterable[str]):
        self.itos = ["<pad>", "<unk>"] + list(wv.index_to_key)
        self.n_w2v = len(wv.index_to_key)
        known = set(wv.key_to_index)
        extra = sorted({t for s in train_texts for t in tokenize(s) if t not in known})
        self.itos += extra
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, texts: Iterable[str], max_len: int) -> tuple[np.ndarray, np.ndarray]:
        texts = list(texts)
        ids = np.zeros((len(texts), max_len), dtype=np.int64)
        lens = np.zeros(len(texts), dtype=np.int64)
        for i, s in enumerate(texts):
            t = [self.stoi.get(w, 1) for w in tokenize(s)][:max_len] or [1]
            ids[i, :len(t)] = t
            lens[i] = len(t)
        return ids, lens

    def embedding_matrix(self, wv, seed: int) -> np.ndarray:
        """Word2Vec rows for known words; N(0, 0.1) for UNK and out-of-vocabulary training words; zeros for PAD."""
        rng = np.random.default_rng(seed)
        m = rng.normal(0.0, 0.1, (len(self.itos), wv.vector_size)).astype(np.float32)
        m[0] = 0.0
        m[2:2 + self.n_w2v] = wv.vectors
        return m
