"""Loading the FinBERT checkpoints of SPEC §3.4 and §3.7.

`yiyanghkust/finbert-pretrain` ships only config.json (without model_type), vocab.txt (lower-case FinVocab) and
pytorch_model.bin, so the auto classes cannot build it; the BERT classes are named explicitly.
"""
from __future__ import annotations

from pathlib import Path

from ..data import hf

PRETRAIN = "yiyanghkust/finbert-pretrain"
PROSUS = "ProsusAI/finbert"


def snapshot(repo: str) -> Path:
    """Local snapshot of `repo`; the Hub is contacted only when the files are not in the cache yet (offline runs)."""
    from huggingface_hub import snapshot_download
    kw = dict(cache_dir=hf.cache_dir(), allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
              ignore_patterns=["*.h5", "*.msgpack"])
    try:
        return Path(snapshot_download(repo, local_files_only=True, **kw))
    except Exception:
        return Path(snapshot_download(repo, token=hf.token(), **kw))


def load_pretrain():
    """(tokenizer, BertModel) of finbert-pretrain; the encoder is not fine-tuned on any sentiment set."""
    from transformers import BertModel, BertTokenizer
    p = snapshot(PRETRAIN)
    # transformers 5: the WordPiece vocabulary is passed as `vocab` (a `vocab_file` keyword is silently ignored)
    tok = BertTokenizer(vocab=str(p / "vocab.txt"), do_lower_case=True)
    assert len(tok.get_vocab()) > 30000, "FinVocab not loaded"
    model = BertModel.from_pretrained(p, add_pooling_layer=False)
    return tok, model


def load_prosus():
    """(tokenizer, classifier, label order) of ProsusAI/finbert (fine-tuned on PhraseBank; zero-shot use only)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    p = snapshot(PROSUS)
    tok = AutoTokenizer.from_pretrained(p)
    model = AutoModelForSequenceClassification.from_pretrained(p)
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    return tok, model, labels
