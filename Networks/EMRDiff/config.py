"""Default HIFTool options for EMR-Diff."""

from __future__ import annotations

from types import SimpleNamespace


def make_options(**overrides):
    values = {
        "sf": 32,
        "hsi_channel": 31,
        "msi_channel": 3,
        "patch_size": 256,
        "diffusion_steps": 5,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "requires_specialized_loss": True,
        "source_url": "https://github.com/luocz55/EMR-Diff",
        "source_commit": "cc851ec7b597bbe259c1cdc413a13e19a386d228",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def args_parser(argv=None):
    # HIFTool model_generator must not consume the parent program's CLI.
    del argv
    return make_options()


opt = make_options()

