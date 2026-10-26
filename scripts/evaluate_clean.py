"""Clean evaluation on the held-out test partition.

Reports accuracy, Macro-F1 and class-wise F1 = 2TP / (2TP + FP + FN).

Usage
  python scripts/evaluate_clean.py --dataset stockrel --kind bigru --run ded
  python scripts/evaluate_clean.py --dataset stockrel --kind bigru --run teacher
  python scripts/evaluate_clean.py --dataset stockrel --kind distilbert --run ded --rule single
"""
import argparse
import json
import os
import sys

import torch
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ded import load_config  # noqa: E402
from ded.data import LABEL_NAMES, load_tokenizer, read_split  # noqa: E402
from ded.inference import Classifier  # noqa: E402
from ded.models import Teacher, build_student  # noqa: E402


def load(args, cfg, device):
    dataset_dir = os.path.join(args.data, args.dataset)
    max_length = cfg['data']['max_length']
    if args.run == 'teacher':
        kind = 'finbert' if args.kind == 'distilbert' else args.kind
        tok = load_tokenizer(kind, dataset_dir)
        model = Teacher(kind, vocab_size=len(tok)).to(device)
        state = torch.load(os.path.join(args.runs, args.dataset, 'teacher', kind, 'best.pt'),
                           map_location=device)
        model.load_state_dict(state['model'])
        return Classifier(model, tok, device, max_length=max_length, is_teacher=True)
    tok = load_tokenizer(args.kind, dataset_dir)
    state = torch.load(os.path.join(args.runs, args.dataset, 'student', args.kind, args.run, 'best.pt'),
                       map_location=device)
    meta = state['meta']
    model = build_student(args.kind, meta['variant'], len(tok), meta['qubits'], meta['layers'],
                          meta['proj_width']).to(device)
    model.load_state_dict(state['model'])
    inf = cfg['inference']
    return Classifier(model, tok, device, rule=args.rule or meta['inference'], n_views=inf['n_views'],
                      rho=inf['rho'] if args.rho is None else args.rho, max_length=max_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--kind', required=True)
    ap.add_argument('--run', default='ded')
    ap.add_argument('--rule', default=None, choices=['alpha', 'uniform', 'single'])
    ap.add_argument('--rho', type=float, default=None)
    ap.add_argument('--data', default='data')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clf = load(args, cfg, device)
    test = read_split(os.path.join(args.data, args.dataset), 'test')
    preds = clf.predict(test['text'].astype(str).tolist()).numpy()
    gold = test['label'].to_numpy()
    per_class = f1_score(gold, preds, average=None, labels=[0, 1, 2])
    print(json.dumps({'accuracy': float(accuracy_score(gold, preds)),
                      'macro_f1': float(f1_score(gold, preds, average='macro')),
                      **{f'f1_{n}': float(v) for n, v in zip(LABEL_NAMES, per_class)}}, indent=2))


if __name__ == '__main__':
    main()
