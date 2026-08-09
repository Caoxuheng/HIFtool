"""Default HIFTool options for BHSR-Net."""

from __future__ import annotations

from types import SimpleNamespace


def make_options(**overrides):
    values = {
        "sf": 32,
        "hsi_channel": 31,
        "msi_channel": 3,
        "patch_size": 128,
        "stages": 10,
        "sigma1": 1.0,
        "sigma2": 2.0,
        "blur_level": 0.0,
        "update_init": 0.003,
        "learning_rate": 1e-3,
        "batch_size": 8,
        "requires_specialized_loss": True,
        "source_url": "https://github.com/Dou0405/BHSR-Net",
        "source_commit": "d5fa43ae36b2ac2831856cbace0de1c9116a8749",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def args_parser(argv=None):
    del argv
    return make_options()


opt = make_options()

