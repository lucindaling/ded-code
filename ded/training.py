"""Shared training loop.

AdamW with weight decay 0.0005, at most 30 epochs, early stopping with
patience 5 on validation Macro-F1. Learning rate 1e-3 and batch size 64 for
recurrent encoders, 2e-5 and 16 for Transformer encoders. The validation set
is a stratified hold-out of the training partition; the test partition is
not used for training or model selection.
"""
from __future__ import annotations

import json
import os
import random
from typing import Callable, Dict

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from .data import TextDataset, collate, read_split, stratified_holdout

TRANSFORMERS = ('finbert', 'distilbert')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hyperparameters(kind: str, cfg: Dict) -> Dict:
    t = cfg['training']
    is_tf = kind in TRANSFORMERS
    return dict(lr=t['lr_transformer'] if is_tf else t['lr_recurrent'],
                batch_size=t['batch_transformer'] if is_tf else t['batch_recurrent'],
                weight_decay=t['weight_decay'], epochs=t['max_epochs'],
                patience=t['patience'], max_length=cfg['data']['max_length'],
                val_fraction=t['val_fraction'], seed=cfg['seed'])


def make_loaders(dataset_dir: str, tokenizer, hp: Dict):
    """Return fit and validation loaders. Row IDs index the training partition."""
    train = read_split(dataset_dir, 'train')
    labels = train['label'].to_numpy()
    fit_idx, val_idx = stratified_holdout(labels, hp['val_fraction'], hp['seed'])
    texts = train['text'].astype(str).tolist()

    def subset(idx):
        return TextDataset([texts[i] for i in idx], [int(labels[i]) for i in idx], list(idx))

    gen = torch.Generator().manual_seed(hp['seed'])
    fit = DataLoader(subset(fit_idx), batch_size=hp['batch_size'], shuffle=True,
                     generator=gen, collate_fn=collate(tokenizer, hp['max_length']))
    val = DataLoader(subset(val_idx), batch_size=64, shuffle=False,
                     collate_fn=collate(tokenizer, hp['max_length']))
    return fit, val, train


@torch.no_grad()
def macro_f1(predict: Callable, loader, device) -> float:
    preds, gold = [], []
    for batch in loader:
        p = predict(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        preds.extend(p.argmax(-1).cpu().tolist())
        gold.extend(batch['labels'].tolist())
    return float(f1_score(gold, preds, average='macro'))


def fit(model, params, step: Callable, predict: Callable, fit_loader, val_loader,
        hp: Dict, device, out_dir: str, extra_state: Callable = lambda: {}):
    """Train with AdamW and early stopping on validation Macro-F1."""
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.AdamW(params, lr=hp['lr'], weight_decay=hp['weight_decay'])
    best, bad, history = -1.0, 0, []
    for epoch in range(1, hp['epochs'] + 1):
        model.train()
        logs = []
        for batch in fit_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, parts = step(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            logs.append(parts)
        model.eval()
        score = macro_f1(predict, val_loader, device)
        row = {'epoch': epoch, 'val_macro_f1': score,
               **{k: float(np.mean([p[k] for p in logs])) for k in logs[0]}}
        history.append(row)
        print(json.dumps(row))
        if score > best:
            best, bad = score, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, **extra_state()},
                       os.path.join(out_dir, 'best.pt'))
        else:
            bad += 1
            if bad >= hp['patience']:
                break
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    return best
