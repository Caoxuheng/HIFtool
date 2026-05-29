from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn.functional as F

import qmb1_thermal_rollout_new as base

_orig_parser = base.build_arg_parser
_orig_loss = base.compute_training_loss


def _mask(model, patterns: str, device, dtype):
    cols = list(getattr(model, "target_cols", []) or [])
    pats = [p.strip().lower() for p in str(patterns or "").split(",") if p.strip()]
    if not cols or not pats:
        return torch.ones(int(getattr(model, "n_targets", 1)), device=device, dtype=dtype).view(1, -1)
    vals = [1.0 if any(p in c.lower() for p in pats) else 0.0 for c in cols]
    if sum(vals) == 0:
        vals = [1.0 for _ in cols]
    return torch.tensor(vals, device=device, dtype=dtype).view(1, -1)


def _active_slope_loss(model, batch, ns, device):
    y = batch[0].to(device, non_blocking=True)
    u = batch[1].to(device, non_blocking=True)
    dt = batch[2].to(device, non_blocking=True)
    w = max(1, int(round(float(getattr(ns, "warmup_seconds", 20.0)) / max(float(getattr(ns, "resample_seconds", 5.0)), 1e-6))))
    if y.shape[1] <= w:
        return y.sum() * 0.0, {"active_slope_gate": 0.0}
    pred_n, _ = model.rollout(y, u, dt, w)
    if pred_n.shape[1] <= 0:
        return y.sum() * 0.0, {"active_slope_gate": 0.0}
    pred = model.denorm_target(pred_n)
    true = model.denorm_target(y[:, w:])
    start = model.denorm_target(y[:, w - 1])
    dur_h = dt[:, w - 1 : w - 1 + pred_n.shape[1]].clamp_min(1e-6).sum(dim=1).view(-1, 1) / 3600.0
    dur_h = dur_h.clamp_min(1e-6)
    pred_slope_ch = (pred[:, -1, :] - start) / dur_h
    true_slope_ch = (true[:, -1, :] - start) / dur_h
    m = _mask(model, getattr(ns, "active_slope_target_patterns", ""), y.device, y.dtype)
    denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
    pred_slope = (pred_slope_ch * m).sum(dim=-1, keepdim=True) / denom
    true_slope = (true_slope_ch * m).sum(dim=-1, keepdim=True) / denom

    uk = model.denorm_control(u)
    ur = uk[:, w : w + pred_n.shape[1], :]
    pidx = list(getattr(model, "real_power_indices", []))
    didx = int(getattr(model, "duty_index", 0))
    p = ur[:, :, pidx].clamp_min(0.0).sum(dim=-1)
    d = ur[:, :, didx].clamp_min(0.0)
    pm = p.mean(dim=1, keepdim=True)
    dm = d.mean(dim=1, keepdim=True)
    pc = (p[:, 1:] - p[:, :-1]).abs().mean(dim=1, keepdim=True) if p.shape[1] > 1 else p.new_zeros(p.shape[0], 1)
    dc = (d[:, 1:] - d[:, :-1]).abs().mean(dim=1, keepdim=True) if d.shape[1] > 1 else d.new_zeros(d.shape[0], 1)

    pmin = float(getattr(ns, "active_slope_min_power_w", 4.0)); pmax = float(getattr(ns, "active_slope_max_power_w", 6.0))
    dmin = float(getattr(ns, "active_slope_min_duty", 0.40)); dmax = float(getattr(ns, "active_slope_max_duty", 0.55))
    pw = max(float(getattr(ns, "active_slope_power_width_w", 0.25)), 1e-6)
    dw = max(float(getattr(ns, "active_slope_duty_width", 0.02)), 1e-6)
    pcs = max(float(getattr(ns, "active_slope_power_change_scale_w", 0.05)), 1e-6)
    dcs = max(float(getattr(ns, "active_slope_duty_change_scale", 0.01)), 1e-6)
    thr = float(getattr(ns, "active_slope_true_threshold_k_per_hour", 0.3))
    tw = max(float(getattr(ns, "active_slope_true_width_k_per_hour", 0.2)), 1e-6)
    margin = float(getattr(ns, "active_slope_pred_margin_k_per_hour", 0.1))

    gate = torch.sigmoid((pm - pmin) / pw) * torch.sigmoid((pmax - pm) / pw)
    gate = gate * torch.sigmoid((dm - dmin) / dw) * torch.sigmoid((dmax - dm) / dw)
    gate = gate * torch.exp(-pc / pcs) * torch.exp(-dc / dcs)
    gate = gate * torch.sigmoid((true_slope - thr) / tw)
    gate = gate.clamp(0.0, 1.0)
    loss = (gate * F.relu(margin - pred_slope).square()).sum() / gate.sum().clamp_min(1e-6)
    return loss, {
        "active_slope_loss": float(loss.detach().cpu()),
        "active_slope_gate": float(gate.detach().mean().cpu()),
        "active_slope_true_kph": float(true_slope.detach().mean().cpu()),
        "active_slope_pred_kph": float(pred_slope.detach().mean().cpu()),
    }


def compute_training_loss(*args: Any, **kwargs: Any):
    total, logs = _orig_loss(*args, **kwargs)
    ns = kwargs.get("args", None)
    for item in args:
        if isinstance(item, argparse.Namespace):
            ns = item
    lam = float(getattr(ns, "lambda_active_slope", 0.0)) if ns is not None else 0.0
    if lam <= 0:
        return total, logs
    model = kwargs.get("model", args[0] if args else None)
    batch = kwargs.get("batch", args[1] if len(args) > 1 else None)
    if model is None or batch is None:
        return total, logs
    device = next(model.parameters()).device
    extra, extra_logs = _active_slope_loss(model, batch, ns, device)
    total = total + lam * extra
    logs.update(extra_logs)
    logs["loss"] = float(total.detach().cpu())
    return total, logs


def build_arg_parser():
    p = _orig_parser()
    p.add_argument("--lambda-active-slope", type=float, default=0.0)
    p.add_argument("--active-slope-min-power-w", type=float, default=4.0)
    p.add_argument("--active-slope-max-power-w", type=float, default=6.0)
    p.add_argument("--active-slope-power-width-w", type=float, default=0.25)
    p.add_argument("--active-slope-min-duty", type=float, default=0.40)
    p.add_argument("--active-slope-max-duty", type=float, default=0.55)
    p.add_argument("--active-slope-duty-width", type=float, default=0.02)
    p.add_argument("--active-slope-power-change-scale-w", type=float, default=0.05)
    p.add_argument("--active-slope-duty-change-scale", type=float, default=0.01)
    p.add_argument("--active-slope-true-threshold-k-per-hour", type=float, default=0.3)
    p.add_argument("--active-slope-true-width-k-per-hour", type=float, default=0.2)
    p.add_argument("--active-slope-pred-margin-k-per-hour", type=float, default=0.1)
    p.add_argument("--active-slope-target-patterns", type=str, default="")
    return p


base.compute_training_loss = compute_training_loss
base.build_arg_parser = build_arg_parser

if __name__ == "__main__":
    base.main()
