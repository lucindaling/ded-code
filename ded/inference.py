"""Inference rules: single pass, uniform smoothing (m-smooth) and alpha-smooth.

alpha-smooth masks each attended token independently with probability rho,
replacing its ID by the padding ID while keeping the attention mask, and
aggregates the N views with weights S_i / sum_j S_j. The N views of an input
are processed together as one batch.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


def make_views(input_ids: torch.Tensor, attention_mask: torch.Tensor, n_views: int,
               rho: float, pad_id: int, generator: Optional[torch.Generator] = None):
    ids = input_ids.repeat_interleave(n_views, dim=0)
    mask = attention_mask.repeat_interleave(n_views, dim=0)
    if rho > 0:
        draw = torch.rand(ids.shape, generator=generator, device='cpu').to(ids.device)
        drop = (draw < rho) & mask.bool()
        ids = torch.where(drop, torch.full_like(ids, pad_id), ids)
    return ids, mask


@torch.no_grad()
def predict_proba(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                  rule: str = 'alpha', n_views: int = 100, rho: float = 0.15,
                  pad_id: int = 0, generator: Optional[torch.Generator] = None,
                  max_views_per_pass: int = 4096) -> torch.Tensor:
    """Return the aggregated class probabilities m_bar for each input."""
    if rule == 'single':
        return model(input_ids, attention_mask)['m']
    per_pass = max(1, max_views_per_pass // n_views)
    outputs = []
    for start in range(0, input_ids.shape[0], per_pass):
        ids, mask = make_views(input_ids[start:start + per_pass],
                               attention_mask[start:start + per_pass],
                               n_views, rho, pad_id, generator)
        out = model(ids, mask)
        m = out['m'].reshape(-1, n_views, out['m'].shape[-1])
        if rule == 'alpha':
            w = out['S'].reshape(-1, n_views, 1)
        elif rule == 'uniform':
            w = torch.ones_like(m[..., :1])
        else:
            raise ValueError(rule)
        outputs.append((w * m).sum(1) / w.sum(1))
    return torch.cat(outputs, 0)


class Classifier:
    """Text-level wrapper around a trained model and its inference rule."""

    def __init__(self, model, tokenizer, device, rule: str = 'alpha', n_views: int = 100,
                 rho: float = 0.15, max_length: int = 128, seed: int = 42,
                 is_teacher: bool = False):
        self.model, self.tokenizer, self.device = model.eval(), tokenizer, device
        self.rule, self.n_views, self.rho = rule, n_views, rho
        self.max_length, self.is_teacher = max_length, is_teacher
        self.generator = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def proba(self, texts: Sequence[str], batch_size: int = 64) -> torch.Tensor:
        rows = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(texts[start:start + batch_size], self.max_length)
            ids = enc['input_ids'].to(self.device)
            mask = enc['attention_mask'].to(self.device)
            if self.is_teacher:
                rows.append(self.model(ids, mask).softmax(-1).cpu())
            else:
                rows.append(predict_proba(self.model, ids, mask, self.rule, self.n_views,
                                          self.rho, self.tokenizer.PAD_ID, self.generator).cpu())
        return torch.cat(rows, 0)

    def predict(self, texts: Sequence[str]) -> torch.Tensor:
        return self.proba(texts).argmax(-1)
