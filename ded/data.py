"""Text normalisation, vocabulary, tokenisers and datasets."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABELS = {'negative': 0, 'neutral': 1, 'positive': 2}
LABEL_NAMES = ['negative', 'neutral', 'positive']
SOURCE_LABEL_MAP = {'bearish': 'negative', 'neutral': 'neutral', 'bullish': 'positive'}

BERT_CHECKPOINTS = {'finbert': 'ProsusAI/finbert', 'distilbert': 'distilbert-base-uncased'}

_URL = re.compile(r'https?://\S+|www\.\S+')
_MENTION = re.compile(r'@[A-Za-z0-9_]+')


def normalise(text: str) -> str:
    """Lowercase, replace URLs and user mentions, normalise whitespace.

    Cashtags, hashtags, stopwords and spelling are kept unchanged.
    """
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = _URL.sub('[URL]', text)
    text = _MENTION.sub('[USER]', text)
    return ' '.join(text.split())


class Vocabulary:
    PAD, UNK = '[PAD]', '[UNK]'
    PAD_ID, UNK_ID = 0, 1

    def __init__(self, token_to_id: Dict[str, int]):
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: Sequence[str], max_size: int = 10_000) -> 'Vocabulary':
        counts = Counter(tok for t in texts for tok in t.split())
        tokens = [tok for tok, _ in counts.most_common(max_size - 2)]
        mapping = {cls.PAD: cls.PAD_ID, cls.UNK: cls.UNK_ID}
        mapping.update({tok: i + 2 for i, tok in enumerate(tokens)})
        return cls(mapping)

    def save(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        with open(path, encoding='utf-8') as f:
            return cls(json.load(f))

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __call__(self, texts: Sequence[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        ids = torch.full((len(texts), max_length), self.PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(texts), max_length), dtype=torch.long)
        for row, text in enumerate(texts):
            toks = text.split()[:max_length] or [self.UNK]
            ids[row, :len(toks)] = torch.tensor([self.token_to_id.get(t, self.UNK_ID) for t in toks])
            mask[row, :len(toks)] = 1
        return {'input_ids': ids, 'attention_mask': mask}


class BertTokenizer:
    """Wrapper with the same call signature as Vocabulary."""

    def __init__(self, checkpoint: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.PAD_ID = self.tok.pad_token_id

    def __len__(self) -> int:
        return len(self.tok)

    def __call__(self, texts: Sequence[str], max_length: int = 128) -> Dict[str, torch.Tensor]:
        enc = self.tok(list(texts), padding='max_length', truncation=True,
                       max_length=max_length, return_tensors='pt')
        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']}


def load_tokenizer(kind: str, dataset_dir: str):
    if kind in BERT_CHECKPOINTS:
        return BertTokenizer(BERT_CHECKPOINTS[kind])
    return Vocabulary.load(os.path.join(dataset_dir, 'vocabulary.json'))


def read_split(dataset_dir: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(dataset_dir, f'{split}.csv'))


def stratified_holdout(labels: np.ndarray, fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split training indices into fit and validation indices, stratified by label."""
    rng = np.random.RandomState(seed)
    fit, val = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = int(round(len(idx) * fraction))
        val.extend(idx[:n_val])
        fit.extend(idx[n_val:])
    return np.sort(np.array(fit)), np.sort(np.array(val))


class TextDataset(Dataset):
    """Returns the row index so that teacher targets are matched by sample ID."""

    def __init__(self, texts: List[str], labels: List[int], ids: List[int]):
        self.texts, self.labels, self.ids = list(texts), list(labels), list(ids)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i], self.ids[i]


def collate(tokenizer, max_length: int):
    def fn(batch):
        texts, labels, ids = zip(*batch)
        enc = tokenizer(texts, max_length)
        enc['labels'] = torch.tensor(labels, dtype=torch.long)
        enc['ids'] = torch.tensor(ids, dtype=torch.long)
        return enc
    return fn
