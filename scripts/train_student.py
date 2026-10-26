"""Train a DED, VQC-DED or control student from a frozen teacher on clean text.

Teacher targets are computed once for the whole training partition and
gathered by sample ID in every batch.

Examples
  python scripts/train_student.py --dataset stockrel --kind bigru --variant ded
  python scripts/train_student.py --dataset stockrel --kind bigru --variant vqc_ded --qubits 5 --layers 4
  python scripts/train_student.py --dataset stockrel --kind distilbert --variant ded
  python scripts/train_student.py --dataset stockrel --kind bigru --variant ded --temperature 2
  python scripts/train_student.py --dataset stockrel --kind bigru --variant ded --proj-width 32

Variants: see ded.models.VARIANTS (ded, vqc_ded, kd, kd_only, hard_label,
coupled, no_detach, no_kl, no_mean, no_shape, no_evi, no_soft, evi_only,
belief_only, uncertainty_only, linear_bottleneck, matched_mlp, periodic,
vqc_prec, vqc_both).
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ded import load_config  # noqa: E402
from ded.data import load_tokenizer  # noqa: E402
from ded.models import (DEDLoss, LOSS_SETS, Teacher, UncertaintyProjector,  # noqa: E402
                        VARIANTS, build_student, count_parameters)
from ded.training import fit, hyperparameters, make_loaders, set_seed  # noqa: E402

TEACHER_OF = {'bilstm': 'bilstm', 'bigru': 'bigru', 'han': 'han', 'distilbert': 'finbert'}


@torch.no_grad()
def teacher_targets(teacher, train_df, tokenizer, max_length, temperature, device):
    """Temperature-scaled teacher probabilities for every training row, by row ID."""
    teacher.eval()
    texts = train_df['text'].astype(str).tolist()
    probs = []
    for start in range(0, len(texts), 64):
        enc = tokenizer(texts[start:start + 64], max_length)
        logits = teacher(enc['input_ids'].to(device), enc['attention_mask'].to(device))
        probs.append((logits / temperature).softmax(-1).cpu())
    return torch.cat(probs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--kind', required=True, choices=list(TEACHER_OF))
    ap.add_argument('--variant', default='ded', choices=list(VARIANTS))
    ap.add_argument('--qubits', type=int, default=None)
    ap.add_argument('--layers', type=int, default=None)
    ap.add_argument('--temperature', type=float, default=None)
    ap.add_argument('--proj-width', type=int, default=None)
    ap.add_argument('--data', default='data')
    ap.add_argument('--runs', default='runs')
    ap.add_argument('--config', default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    hp = hyperparameters(args.kind, cfg)
    T = args.temperature or cfg['ded']['temperature']
    n_qubits = args.qubits or cfg['vqc']['n_qubits']
    n_layers = args.layers or cfg['vqc']['n_layers']
    set_seed(hp['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_dir = os.path.join(args.data, args.dataset)

    student_tok = load_tokenizer(args.kind, dataset_dir)
    teacher_kind = TEACHER_OF[args.kind]
    teacher_tok = load_tokenizer(teacher_kind, dataset_dir)
    fit_loader, val_loader, train_df = make_loaders(dataset_dir, student_tok, hp)

    teacher = Teacher(teacher_kind, vocab_size=len(teacher_tok)).to(device)
    ckpt = torch.load(os.path.join(args.runs, args.dataset, 'teacher', teacher_kind, 'best.pt'),
                      map_location=device)
    teacher.load_state_dict(ckpt['model'])
    for p in teacher.parameters():
        p.requires_grad_(False)
    p_t_all = teacher_targets(teacher, train_df, teacher_tok, hp['max_length'], T, device).to(device)
    del teacher

    student = build_student(args.kind, args.variant, len(student_tok), n_qubits, n_layers,
                            args.proj_width).to(device)
    projector = UncertaintyProjector().to(device)
    w = cfg['ded']['loss_weights']
    loss_fn = DEDLoss(LOSS_SETS[VARIANTS[args.variant][4]], T,
                      (w['kl'], w['mean'], w['shape'], w['evi'], w['soft']), cfg['ded']['kd_alpha'])
    print(f'student parameters (incl. DEH): {count_parameters(student):,}')

    def step(batch):
        p_t = p_t_all[batch['ids']]
        out = student(batch['input_ids'], batch['attention_mask'])
        return loss_fn(out, p_t, projector(p_t), batch['labels'])

    def predict(ids, mask):
        return student(ids, mask)['m']

    params = list(student.parameters()) + list(projector.parameters())
    name = args.variant
    if args.proj_width:
        name += f'_proj{args.proj_width}'
    if args.temperature:
        name += f'_T{args.temperature:g}'
    if args.qubits or args.layers:
        name += f'_n{n_qubits}_L{n_layers}'
    out_dir = os.path.join(args.runs, args.dataset, 'student', args.kind, name)
    meta = {'variant': args.variant, 'kind': args.kind, 'temperature': T, 'qubits': n_qubits,
            'layers': n_layers, 'proj_width': args.proj_width, 'inference': VARIANTS[args.variant][5]}
    best = fit(student, params, step, predict, fit_loader, val_loader, hp, device, out_dir,
               extra_state=lambda: {'projector': projector.state_dict(), 'meta': meta})
    print(f'best validation macro-F1 {best:.4f}; model saved in {out_dir}')


if __name__ == '__main__':
    main()
