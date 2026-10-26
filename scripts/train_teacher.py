"""Train a teacher (NoDefense) with cross-entropy on clean text.

Teachers: full-width BiLSTM, BiGRU, HAN, or FinBERT (teacher of DistilBERT).

Usage: python scripts/train_teacher.py --dataset stockrel --kind bigru
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ded import load_config  # noqa: E402
from ded.data import load_tokenizer  # noqa: E402
from ded.models import Teacher  # noqa: E402
from ded.training import fit, hyperparameters, make_loaders, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--kind', required=True, choices=['bilstm', 'bigru', 'han', 'finbert'])
    ap.add_argument('--data', default='data')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hp = hyperparameters(args.kind, cfg)
    set_seed(hp['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = os.path.join(args.data, args.dataset)
    tok = load_tokenizer(args.kind, dataset_dir)
    fit_loader, val_loader, _ = make_loaders(dataset_dir, tok, hp)

    model = Teacher(args.kind, vocab_size=len(tok)).to(device)

    def step(batch):
        loss = F.cross_entropy(model(batch['input_ids'], batch['attention_mask']), batch['labels'])
        return loss, {'ce': float(loss.detach())}

    def predict(ids, mask):
        return model(ids, mask)

    out_dir = os.path.join(args.runs, args.dataset, 'teacher', args.kind)
    best = fit(model, model.parameters(), step, predict, fit_loader, val_loader, hp, device, out_dir)
    print(f'best validation macro-F1 {best:.4f}; model saved in {out_dir}')


if __name__ == '__main__':
    main()
