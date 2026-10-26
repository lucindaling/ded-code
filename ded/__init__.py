"""Decoupled Evidential Distillation (DED) and VQC-DED."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str = None) -> dict:
    with open(path or os.path.join(ROOT, 'configs', 'default.json')) as f:
        return json.load(f)
