"""Default HIFTool options for PSRF-DiffNet."""

from __future__ import annotations

from types import SimpleNamespace


def make_options(**overrides):
    values = {
        "sf": 32,
        "hsi_channel": 31,
        "msi_channel": 3,
        "patch_size": 128,
        "width": 64,
        "blocks": 6,
        "diffusion_steps": 8,
        "nonregistration_probability": 0.75,
        "learning_rate": 1e-4,
        "batch_size": 8,
        "clip_grad_norm": 1.0,
        "requires_specialized_loss": True,
        "source_url": "https://github.com/Jiahuiqu/PSRF-DiffNet",
        "source_commit": "d4e88fa3fe2f13eb5a67e53ead99f6370713ab9b",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def args_parser(argv=None):
    del argv
    return make_options()


opt = make_options()

