import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qmb1_active_duty_slope_loss as active
from qmb1_thermal_rollout_new import PhysThermalRolloutModel


class ToySlopeModel(PhysThermalRolloutModel):
    def __init__(self):
        super().__init__(
            n_targets=2,
            n_controls=4,
            n_diagnostics=0,
            target_mean=np.zeros(2, dtype=np.float32),
            target_std=np.ones(2, dtype=np.float32),
            control_mean=np.zeros(4, dtype=np.float32),
            control_std=np.ones(4, dtype=np.float32),
            diagnostic_mean=np.zeros(0, dtype=np.float32),
            diagnostic_std=np.ones(0, dtype=np.float32),
            target_cols=["inner_a", "middle_b"],
            duty_index=0,
            real_power_indices=[1, 2, 3],
            hidden_dim=8,
        )

    def rollout(self, target_seq_norm, control_seq_norm, dt_seq, warmup_steps):
        steps = target_seq_norm.shape[1] - warmup_steps
        start = target_seq_norm[:, warmup_steps - 1 : warmup_steps]
        # Deliberately wrong: predict cooling while the synthetic truth rises.
        t = torch.arange(1, steps + 1, device=target_seq_norm.device, dtype=target_seq_norm.dtype).view(1, -1, 1)
        pred = start - 0.002 * t
        return pred.expand(-1, -1, target_seq_norm.shape[-1]), {}


def make_batch():
    warmup = 5
    horizon = 120
    total = warmup + horizon
    y = torch.zeros(1, total, 2)
    # True rises by about 1.44 K/h over 600 s.
    for k in range(warmup, total):
        y[:, k, :] = (k - warmup + 1) * 0.002
    u = torch.zeros(1, total, 4)
    u[:, :, 0] = 0.48  # duty
    u[:, :, 1:] = torch.tensor([2.0, 2.0, 1.0])  # total real power = 5 W
    dt = torch.ones(1, total - 1) * 5.0
    return y, u, dt


def test_active_duty_slope_loss_penalizes_wrong_cooling_sign():
    model = ToySlopeModel()
    ns = argparse.Namespace(
        warmup_seconds=20,
        resample_seconds=5,
        active_slope_min_power_w=4.0,
        active_slope_max_power_w=6.0,
        active_slope_power_width_w=0.25,
        active_slope_min_duty=0.40,
        active_slope_max_duty=0.55,
        active_slope_duty_width=0.02,
        active_slope_power_change_scale_w=0.05,
        active_slope_duty_change_scale=0.01,
        active_slope_true_threshold_k_per_hour=0.3,
        active_slope_true_width_k_per_hour=0.2,
        active_slope_pred_margin_k_per_hour=0.1,
        active_slope_target_patterns="",
    )
    loss, logs = active._active_slope_loss(model, make_batch(), ns, torch.device("cpu"))
    assert loss.item() > 1.0
    assert logs["active_slope_gate"] > 0.5
    assert logs["active_slope_true_kph"] > 0.3
    assert logs["active_slope_pred_kph"] < 0.0


if __name__ == "__main__":
    test_active_duty_slope_loss_penalizes_wrong_cooling_sign()
