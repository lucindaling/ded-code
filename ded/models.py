"""Encoders, teachers, DED heads, VQC and classical control heads."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum import StatevectorVQC, encode_angles

# Widths reported in the paper. Teachers: 512-d embeddings; students: 128-d
# embeddings and a 64-d representation h.
ENCODER_CONFIGS = {
    'teacher': {
        'bilstm': dict(embed=512, hidden=256, out=128),
        'bigru': dict(embed=512, hidden=256, out=128),
        'han': dict(embed=512, hidden=128, out=128, sentence_attention=True),
    },
    'student': {
        'bilstm': dict(embed=128, hidden=128, out=64),
        'bigru': dict(embed=128, hidden=128, out=64),
        # one sentence per tweet: attention over a single sentence state is
        # the identity, so the student omits the sentence-attention layer
        'han': dict(embed=128, hidden=64, out=64, sentence_attention=False),
    },
}


class RNNEncoder(nn.Module):
    def __init__(self, cell: str, vocab_size: int, embed: int, hidden: int, out: int,
                 layers: int = 2, dropout: float = 0.3):
        super().__init__()
        rnn = nn.LSTM if cell == 'bilstm' else nn.GRU
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.rnn = rnn(embed, hidden, num_layers=layers, batch_first=True,
                       bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.proj = nn.Linear(2 * hidden, out)
        self.dropout = nn.Dropout(dropout)
        self.dim = out

    def forward(self, input_ids, attention_mask):
        lengths = attention_mask.sum(1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embedding(input_ids), lengths, batch_first=True, enforce_sorted=False)
        _, state = self.rnn(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        h = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        return self.dropout(F.relu(self.proj(h)))


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.context = nn.Linear(dim, 1, bias=False)

    def forward(self, states, mask):
        scores = self.context(torch.tanh(self.proj(states))).squeeze(-1)
        weights = scores.masked_fill(mask == 0, -1e9).softmax(-1)
        return (states * weights.unsqueeze(-1)).sum(1)


class HANEncoder(nn.Module):
    """Hierarchical attention network; each tweet is one sentence."""

    def __init__(self, vocab_size: int, embed: int, hidden: int, out: int,
                 sentence_attention: bool = True, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed, padding_idx=0)
        self.word_gru = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
        self.word_attention = _Attention(2 * hidden)
        self.sent_gru = nn.GRU(2 * hidden, hidden, batch_first=True, bidirectional=True)
        self.sent_attention = _Attention(2 * hidden) if sentence_attention else None
        self.proj = nn.Linear(2 * hidden, out)
        self.dropout = nn.Dropout(dropout)
        self.dim = out

    def forward(self, input_ids, attention_mask):
        words, _ = self.word_gru(self.embedding(input_ids))
        sentence = self.word_attention(words, attention_mask).unsqueeze(1)
        doc, _ = self.sent_gru(sentence)
        if self.sent_attention is not None:
            doc = self.sent_attention(doc, torch.ones(doc.shape[:2], device=doc.device))
        else:
            doc = doc[:, 0]
        return self.dropout(F.relu(self.proj(doc)))


class BertEncoder(nn.Module):
    """[CLS] representation of FinBERT or DistilBERT with dropout 0.1."""

    def __init__(self, checkpoint: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.dropout = nn.Dropout(dropout)
        self.dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state[:, 0])


def build_encoder(kind: str, role: str, vocab_size: Optional[int] = None) -> nn.Module:
    if kind == 'finbert':
        return BertEncoder('ProsusAI/finbert')
    if kind == 'distilbert':
        return BertEncoder('distilbert-base-uncased')
    cfg = ENCODER_CONFIGS[role][kind]
    if kind == 'han':
        return HANEncoder(vocab_size, **cfg)
    return RNNEncoder(kind, vocab_size, **cfg)


class Teacher(nn.Module):
    def __init__(self, kind: str, vocab_size: Optional[int] = None, num_classes: int = 3):
        super().__init__()
        self.encoder = build_encoder(kind, 'teacher', vocab_size)
        self.classifier = nn.Linear(self.encoder.dim, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encoder(input_ids, attention_mask))


# ---------------------------------------------------------------------------
# Mean Branch variants
# ---------------------------------------------------------------------------
class LinearMean(nn.Module):
    """Classical DED Mean Branch: softmax(W_m h + b_m)."""

    def __init__(self, d: int, k: int = 3):
        super().__init__()
        self.out = nn.Linear(d, k)

    def forward(self, h):
        return self.out(h)


class VQCMean(nn.Module):
    """Dressed VQC: h' = W_r h + b_r, theta = pi h', <Z_j>, linear readout."""

    def __init__(self, d: int, k: int = 3, n_qubits: int = 5, n_layers: int = 4):
        super().__init__()
        self.project = nn.Linear(d, n_qubits)
        self.circuit = StatevectorVQC(n_qubits, n_layers)
        self.readout = nn.Linear(n_qubits, k)

    def forward(self, h):
        return self.readout(self.circuit(encode_angles(self.project(h))))


class _ScaleOffset(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n))
        self.offset = nn.Parameter(torch.zeros(n))

    def forward(self, x):
        return self.scale * x + self.offset


class _Periodic(nn.Module):
    """Four sine layers z <- sin(a * z + b) with trainable a, b in R^5 (40 params)."""

    def __init__(self, n: int = 5, layers: int = 4):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(layers, n))
        self.phase = nn.Parameter(torch.zeros(layers, n))

    def forward(self, z):
        for a, b in zip(self.scale, self.phase):
            z = torch.sin(a * z + b)
        return z


class BottleneckMean(nn.Module):
    """Classical heads sharing the VQC projection (d -> 5) and readout (5 -> 3).

    linear:   no module between projection and readout
    mlp:      5x5 affine (30) + ReLU + elementwise scale/offset (10) + tanh = 40 params
    periodic: four sine layers with scale and phase = 40 params
    """

    def __init__(self, d: int, kind: str, k: int = 3, n: int = 5):
        super().__init__()
        self.project = nn.Linear(d, n)
        if kind == 'linear':
            self.module = nn.Identity()
        elif kind == 'mlp':
            self.module = nn.Sequential(nn.Linear(n, n), nn.ReLU(), _ScaleOffset(n), nn.Tanh())
        elif kind == 'periodic':
            self.module = _Periodic(n)
        else:
            raise ValueError(kind)
        self.readout = nn.Linear(n, k)

    def forward(self, h):
        return self.readout(self.module(self.project(h)))


class ProjectionMLPMean(nn.Module):
    """Classical projection head of width w: Linear(d, w) + ReLU + Linear(w, k)."""

    def __init__(self, d: int, width: int, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, k))

    def forward(self, h):
        return self.net(h)


# ---------------------------------------------------------------------------
# Precision Branch variants
# ---------------------------------------------------------------------------
class LinearPrecision(nn.Module):
    """S = Softplus(W_s h + b_s) + 1."""

    def __init__(self, d: int):
        super().__init__()
        self.out = nn.Linear(d, 1)

    def forward(self, h):
        return F.softplus(self.out(h)).squeeze(-1) + 1.0


class VQCPrecision(nn.Module):
    """Placement control: precision computed through a five-qubit circuit."""

    def __init__(self, d: int, n_qubits: int = 5, n_layers: int = 4):
        super().__init__()
        self.project = nn.Linear(d, n_qubits)
        self.circuit = StatevectorVQC(n_qubits, n_layers)
        self.out = nn.Linear(n_qubits, 1)

    def forward(self, h):
        q = self.circuit(encode_angles(self.project(h)))
        return F.softplus(self.out(q)).squeeze(-1) + 1.0


class DEDStudent(nn.Module):
    """Student encoder + Decoupled Evidential Head.

    mean:      module mapping h to logits z_s; m = softmax(z_s)
    precision: module mapping h to S > 1, or None for single-output students
    detach:    if True, the Precision Branch reads detach(h)
    coupled:   standard evidential head, alpha = softplus(z) + 1, S = sum(alpha)
    """

    def __init__(self, encoder: nn.Module, mean: nn.Module,
                 precision: Optional[nn.Module] = None,
                 detach: bool = True, coupled: bool = False):
        super().__init__()
        self.encoder, self.mean, self.precision = encoder, mean, precision
        self.detach, self.coupled = detach, coupled

    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        h = self.encoder(input_ids, attention_mask)
        logits = self.mean(h)
        if self.coupled:
            alpha = F.softplus(logits) + 1.0
            S = alpha.sum(-1)
            m = alpha / S.unsqueeze(-1)
            return {'logits': logits, 'm': m, 'S': S, 'alpha': alpha}
        m = logits.softmax(-1)
        if self.precision is None:
            S = torch.ones(m.shape[0], device=m.device)
        else:
            S = self.precision(h.detach() if self.detach else h)
        return {'logits': logits, 'm': m, 'S': S, 'alpha': S.unsqueeze(-1) * m}


class UncertaintyProjector(nn.Module):
    """g_phi: teacher probabilities (3) -> 32 -> 1 with ReLU and Softplus."""

    def __init__(self, k: int = 3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, 32), nn.ReLU(), nn.Linear(32, 1), nn.Softplus())

    def forward(self, p_t):
        return self.net(p_t).squeeze(-1)


# ---------------------------------------------------------------------------
# Student variants used in the paper
# ---------------------------------------------------------------------------
LOSS_SETS = {
    'ded': ('kl', 'mean', 'shape', 'evi', 'soft'),
    'no_kl': ('mean', 'shape', 'evi', 'soft'),
    'no_mean': ('kl', 'shape', 'evi', 'soft'),
    'no_shape': ('kl', 'mean', 'evi', 'soft'),
    'no_evi': ('kl', 'mean', 'shape', 'soft'),
    'no_soft': ('kl', 'mean', 'shape', 'evi'),
    'evi_only': ('evi',),
    'belief_only': ('mean', 'shape', 'ce'),
    'uncertainty_only': ('evi', 'kl'),
    'hard_label': ('ce',),
    'kd': ('kd_ce',),
    'kd_only': ('kd',),
}

VARIANTS = {
    # name: (mean head, precision head, detach, coupled, losses, inference rule)
    'ded': ('linear', 'linear', True, False, 'ded', 'alpha'),
    'vqc_ded': ('vqc', 'linear', True, False, 'ded', 'alpha'),
    'no_detach': ('linear', 'linear', False, False, 'ded', 'alpha'),
    # coupled evidential head trained with the same distillation objective
    'coupled': ('linear', None, True, True, 'ded', 'alpha'),
    'hard_label': ('linear', None, True, False, 'hard_label', 'single'),
    'kd': ('linear', None, True, False, 'kd', 'single'),
    'kd_only': ('linear', None, True, False, 'kd_only', 'single'),
    'linear_bottleneck': ('bottleneck_linear', 'linear', True, False, 'ded', 'alpha'),
    'matched_mlp': ('bottleneck_mlp', 'linear', True, False, 'ded', 'alpha'),
    'periodic': ('bottleneck_periodic', 'linear', True, False, 'ded', 'alpha'),
    'vqc_prec': ('linear', 'vqc', True, False, 'ded', 'alpha'),
    'vqc_both': ('vqc', 'vqc', True, False, 'ded', 'alpha'),
}
for _name in ('no_kl', 'no_mean', 'no_shape', 'no_evi', 'no_soft',
              'evi_only', 'belief_only', 'uncertainty_only'):
    VARIANTS[_name] = ('linear', 'linear', True, False, _name, 'alpha')


def build_student(kind: str, variant: str, vocab_size: Optional[int] = None,
                  n_qubits: int = 5, n_layers: int = 4,
                  proj_width: Optional[int] = None, k: int = 3) -> DEDStudent:
    encoder = build_encoder(kind, 'student', vocab_size)
    d = encoder.dim
    mean_kind, prec_kind, detach, coupled, _, _ = VARIANTS[variant]
    if proj_width is not None:
        mean = ProjectionMLPMean(d, proj_width, k)
    elif mean_kind == 'linear':
        mean = LinearMean(d, k)
    elif mean_kind == 'vqc':
        mean = VQCMean(d, k, n_qubits, n_layers)
    else:
        mean = BottleneckMean(d, mean_kind.split('_', 1)[1], k)
    if prec_kind == 'linear':
        precision = LinearPrecision(d)
    elif prec_kind == 'vqc':
        precision = VQCPrecision(d, n_qubits, n_layers)
    else:
        precision = None
    return DEDStudent(encoder, mean, precision, detach=detach, coupled=coupled)


class DEDLoss(nn.Module):
    """Weighted DED objective with optional component sets for the ablations.

    L = l1 L_kl + l2 L_mean + l3 L_shape + l4 L_evi + l5 L_soft,
    (l1..l5) = (1, 0.5, 1, 0.1, 0.001), T = 5.
    """

    def __init__(self, components, temperature: float = 5.0,
                 weights=(1.0, 0.5, 1.0, 0.1, 0.001), kd_alpha: float = 0.5):
        super().__init__()
        self.components = set(components)
        self.T = temperature
        self.w = dict(zip(('kl', 'mean', 'shape', 'evi', 'soft'), weights))
        self.kd_alpha = kd_alpha

    def forward(self, out, p_t, S_t, labels):
        m, S, logits = out['m'], out['S'], out['logits']
        T, terms = self.T, {}
        if 'kl' in self.components:
            terms['kl'] = self.w['kl'] * F.kl_div((m + 1e-8).log(), p_t, reduction='batchmean')
        if 'mean' in self.components:
            terms['mean'] = self.w['mean'] * ((m - p_t) ** 2).sum(-1).mean()
        if 'shape' in self.components:
            alpha_t = S_t.unsqueeze(-1) * p_t
            terms['shape'] = self.w['shape'] * ((out['alpha'] - alpha_t) ** 2).sum(-1).mean()
        if 'evi' in self.components:
            m_y = m.gather(1, labels.unsqueeze(1)).squeeze(1)
            terms['evi'] = self.w['evi'] * (S * (1.0 - m_y)).mean()
        if 'soft' in self.components:
            terms['soft'] = self.w['soft'] * T ** 2 * F.kl_div(
                (logits / T).log_softmax(-1), p_t, reduction='batchmean')
        if 'ce' in self.components:
            terms['ce'] = F.cross_entropy(logits, labels)
        if 'kd' in self.components:
            terms['kd'] = T ** 2 * F.kl_div((logits / T).log_softmax(-1), p_t, reduction='batchmean')
        if 'kd_ce' in self.components:
            kd = T ** 2 * F.kl_div((logits / T).log_softmax(-1), p_t, reduction='batchmean')
            terms['kd_ce'] = self.kd_alpha * kd + (1 - self.kd_alpha) * F.cross_entropy(logits, labels)
        total = sum(terms.values())
        return total, {k: float(v.detach()) for k, v in terms.items()}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
