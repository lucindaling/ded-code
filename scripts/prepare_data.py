"""Build the four study datasets from the Hugging Face collection.

For each source: select rows by the `url` field, drop empty texts and exact
duplicates, normalise the text, split once into 75% training and 25% test data
by stratified sampling on the label (seed 42), and build the vocabulary for
recurrent encoders from the training partition.

Usage: python scripts/prepare_data.py --out data
"""
import argparse
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ded import load_config  # noqa: E402
from ded.data import LABELS, SOURCE_LABEL_MAP, Vocabulary, normalise  # noqa: E402


def load_records(cfg):
    from datasets import load_dataset
    ds = load_dataset(cfg['data']['hf_dataset'], split='train', revision=cfg['data']['hf_revision'])
    int2str = ds.features['sentiment'].int2str
    for ex in ds:
        yield {'text': ex.get('tweet') or '', 'label': int2str(int(ex['sentiment'])),
               'url': ex.get('url') or ''}


def select_source(records, url_match):
    seen, rows = set(), []
    for r in records:
        if url_match not in r['url']:
            continue
        text = r['text'].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append({'text': normalise(text), 'label': LABELS[SOURCE_LABEL_MAP[r['label']]]})
    return [r for r in rows if r['text']]


def stratified_split(rows, test_fraction, seed):
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r['label'], []).append(r)
    train, test = [], []
    for _, items in sorted(by_label.items()):
        items = list(items)
        rng.shuffle(items)
        n_test = round(len(items) * test_fraction)
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    records = list(load_records(cfg))
    for name, url_match in cfg['data']['sources'].items():
        rows = select_source(records, url_match)
        train, test = stratified_split(rows, cfg['data']['test_fraction'], cfg['seed'])
        folder = os.path.join(args.out, name)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(train).to_csv(os.path.join(folder, 'train.csv'), index=False)
        pd.DataFrame(test).to_csv(os.path.join(folder, 'test.csv'), index=False)
        vocab = Vocabulary.build([r['text'] for r in train], cfg['data']['vocab_size'])
        vocab.save(os.path.join(folder, 'vocabulary.json'))
        counts = pd.Series([r['label'] for r in rows]).value_counts().sort_index().tolist()
        print(f'{name}: total={len(rows)} neg/neu/pos={counts} train={len(train)} '
              f'test={len(test)} vocab={len(vocab)}')


if __name__ == '__main__':
    main()
