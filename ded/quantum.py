"""Exact state-vector simulation of the VQC Mean Branch circuit.

The circuit starts in |0...0>, and each of the L layers applies
    E(h') = prod_j R_Y(pi * h'_j)          (data re-uploading)
    prod_j R_Z(psi_{l,j}) R_Y(phi_{l,j})   (trainable rotations)
    CNOT(0,1) CNOT(1,2) ... CNOT(n-1,0)    (directed ring)
and the outputs are the Pauli-Z expectations <Z_j>. Qubit 0 is the most
significant bit of the computational-basis index (PennyLane convention).
Amplitudes are complex64 and gradients come from autograd.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class StatevectorVQC(nn.Module):
    def __init__(self, n_qubits: int = 5, n_layers: int = 4):
        super().__init__()
        self.n = n_qubits
        self.layers = n_layers
        self.phi = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits))
        self.psi = nn.Parameter(0.1 * torch.randn(n_layers, n_qubits))
        self.register_buffer('ring', self._compose_ring(n_qubits), persistent=False)

    @staticmethod
    def _compose_ring(n: int) -> torch.Tensor:
        """Index map of one CNOT ring, so that new_state = state[:, perm].

        After CNOT(c, t) the amplitude of basis b equals the old amplitude of
        b with the target bit flipped whenever the control bit of b is set.
        """
        dim = 2 ** n
        perm = torch.arange(dim)
        for control in range(n):
            target = (control + 1) % n
            c_bit = 1 << (n - 1 - control)
            t_bit = 1 << (n - 1 - target)
            idx = torch.arange(dim)
            src = torch.where((idx & c_bit) != 0, idx ^ t_bit, idx)
            perm = perm[src]
        return perm

    def _gate(self, state: torch.Tensor, gate: torch.Tensor, wire: int) -> torch.Tensor:
        # state: (B, 2**n); gate: (B, 2, 2) or (2, 2)
        b = state.shape[0]
        view = state.reshape(b, 2 ** wire, 2, 2 ** (self.n - wire - 1))
        if gate.dim() == 2:
            out = torch.einsum('kl,balc->bakc', gate, view)
        else:
            out = torch.einsum('bkl,balc->bakc', gate, view)
        return out.reshape(b, -1)

    @staticmethod
    def _ry(theta: torch.Tensor) -> torch.Tensor:
        c, s = torch.cos(theta / 2), torch.sin(theta / 2)
        g = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        return g.to(torch.complex64)

    @staticmethod
    def _rz(theta: torch.Tensor) -> torch.Tensor:
        z = torch.zeros_like(theta)
        e_minus = torch.polar(torch.ones_like(theta), -theta / 2)
        e_plus = torch.polar(torch.ones_like(theta), theta / 2)
        zc = torch.complex(z, z)
        return torch.stack([torch.stack([e_minus, zc], -1), torch.stack([zc, e_plus], -1)], -2)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        """angles: (B, n) encoding angles pi*h'. Returns (B, n) <Z_j>."""
        b = angles.shape[0]
        state = torch.zeros(b, 2 ** self.n, dtype=torch.complex64, device=angles.device)
        state[:, 0] = 1.0
        for layer in range(self.layers):
            for j in range(self.n):
                state = self._gate(state, self._ry(angles[:, j]), j)
            for j in range(self.n):
                state = self._gate(state, self._ry(self.phi[layer, j]), j)
                state = self._gate(state, self._rz(self.psi[layer, j]), j)
            state = state[:, self.ring]
        probs = (state.real ** 2 + state.imag ** 2).reshape(b, *([2] * self.n))
        expectations = []
        for j in range(self.n):
            p = probs.movedim(j + 1, 1).reshape(b, 2, -1).sum(-1)
            expectations.append(p[:, 0] - p[:, 1])
        return torch.stack(expectations, dim=-1)


def encode_angles(projected: torch.Tensor) -> torch.Tensor:
    """Angle map used by VQC-DED: theta = pi * h' (no tanh)."""
    return math.pi * projected
