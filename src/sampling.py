from __future__ import annotations
import torch
import numpy as np
from scipy.stats import qmc
from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, T_MIN, T_MAX,
    A_PIPE, B_PIPE, A_STEP, B_STEP, Z_STEP,
    Z0_BUNCH, V_BUNCH,
    SIGMA_X, SIGMA_Y, SIGMA_Z,
    DEVICE
)
from src.geometry import is_interior


"""
HELPERS
"""
def _to_tensor(arr: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float32, device=device)

def _lhs_unit(n: int, d: int, seed: int | None = None) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n=n)


"""
INTERIOR PDE SAMPLING
"""
def sample_interior_lhs(
        n: int,
        seed: int | None = None,
        device: str = DEVICE,
) -> torch.Tensor:
    oversample = 3
    accepted: list[np.ndarray] = []
    total = 0
    rng = np.random.default_rng(seed)

    while total < n:
        raw = _lhs_unit(n * oversample, 4, seed=rng.integers(0, 2**31))

        # Scale to physical domain
        raw[:, 0] = raw[:, 0] * (X_MAX - X_MIN) + X_MIN
        raw[:, 1] = raw[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
        raw[:, 2] = raw[:, 2] * (Z_MAX - Z_MIN) + Z_MIN
        raw[:, 3] = raw[:, 3] * (T_MAX - T_MIN) + T_MIN

        x_ = torch.tensor(raw[:, 0], dtype=torch.float32)
        y_ = torch.tensor(raw[:, 1], dtype=torch.float32)
        z_ = torch.tensor(raw[:, 2], dtype=torch.float32)

        mask = is_interior(x_, y_, z_, tol=1e-6).numpy()
        raw_ok = raw[mask]
        accepted.append(raw_ok)
        total += len(raw_ok)
    
    pts_np = np.concatenate(accepted, axis=0)[:n]
    return _to_tensor(pts_np, device)


"""
IMPORTANCE SAMPLING NEAR THE BUNCH TRAJECTORY
"""
def sample_importance(
        n: int,
        n_sigma: float = 4.0,
        seed: int | None = None,
        device: str = DEVICE,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)

    t = rng.uniform(T_MIN, T_MAX, n).astype(np.float32)

    z_c = Z0_BUNCH + V_BUNCH * t

    x = rng.normal(0.0, n_sigma * SIGMA_X, n).astype(np.float32)
    y = rng.normal(0.0, n_sigma * SIGMA_Y, n).astype(np.float32)
    z = (z_c + rng.normal(0.0, n_sigma * SIGMA_X, n)).astype(np.float32)

    x = np.clip(x, X_MIN + 1e-6, X_MAX - 1e-6)
    y = np.clip(y, Y_MIN + 1e-6, Y_MAX - 1e-6)
    z = np.clip(z, Z_MIN + 1e-6, Z_MAX - 1e-6)

    x_ = torch.tensor(x)
    y_ = torch.tensor(y)
    z_ = torch.tensor(z)
    mask = is_interior(x_, y_, z_, tol=1e-6).numpy()

    pts_np = np.stack([x[mask], y[mask], z[mask], t[mask]], axis=1)

    if len(pts_np) < n:
        pad = sample_interior_lhs(n - len(pts_np), seed=seed, device=device)
        pts_np = np.concatenate(
            [pts_np, pad.cpu().numpy()], axis=0
        )[:n]
    
    return _to_tensor(pts_np[:n], device)


"""
BOUNDARY PEC WALL SAMPLING
"""
def _uniform(low: float, high: float, n: int, rng) -> np.ndarray:
    return rng.uniform(low, high, n).astype(np.float32)

def sample_boundary(
        n: int,
        seed: int | None = None,
        device: str = DEVICE,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)

    # Upstream pipe
    dz_main = Z_STEP - Z_MIN
    dz_step = Z_MAX - Z_STEP

    # Face areas
    area_main_x = 2 * B_PIPE * dz_main
    area_main_y = 2 * A_PIPE * dz_main
    area_step_x = 2 * B_STEP * dz_step
    area_step_y = 2 * A_STEP * dz_step

    area_shoulder = (
        2 * (A_PIPE - A_STEP) * 2 * B_PIPE
        + 2 * (B_PIPE - B_STEP) * 2 * A_STEP
    )

    total_area = (area_main_x + area_main_y + area_step_x + area_step_y + area_shoulder)

    def n_for(area):
        return max(4, int(round(n * area / total_area)))
    
    counts = {
        "main_x": n_for(area_main_x),
        "main_y": n_for(area_main_y),
        "step_x": n_for(area_step_x),
        "step_y": n_for(area_step_y),
        "shoulder": n_for(area_shoulder)
    }

    result: dict[str, torch.Tensor] = {}

    nm = counts["main_x"]
    x_ = np.where(rng.integers(0, 2, nm) == 0, A_PIPE, -A_PIPE).astype(np.float32)
    y_ = _uniform(-B_PIPE, B_PIPE, nm, rng)
    z_ = _uniform(Z_MIN, Z_MAX, nm, rng)
    t_ = _uniform(T_MIN, T_MAX, nm, rng)
    result["main_x"] = _to_tensor(np.stack([x_, y_, z_, t_], 1), device)

    nm = counts["main_y"]
    x_ = _uniform(-A_PIPE, A_PIPE, nm, rng)
    y_ = np.where(rng.integers(0, 2, nm) == 0, B_PIPE, -B_PIPE).astype(np.float32)
    z_ = _uniform(Z_MIN, Z_MAX, nm, rng)
    t_ = _uniform(T_MIN, T_MAX, nm, rng)
    result["main_y"] = _to_tensor(np.stack([x_, y_, z_, t_], 1), device)

    ns = counts["step_x"]
    x_   = np.where(rng.integers(0, 2, ns) == 0, A_STEP, -A_STEP).astype(np.float32)
    y_   = _uniform(-B_STEP, B_STEP, ns, rng)
    z_   = _uniform(Z_STEP, Z_MAX,  ns, rng)
    t_   = _uniform(T_MIN,  T_MAX,  ns, rng)
    result["step_x"] = _to_tensor(np.stack([x_, y_, z_, t_], 1), device)

    ns = counts["step_y"]
    x_   = _uniform(-A_STEP, A_STEP, ns, rng)
    y_   = np.where(rng.integers(0, 2, ns) == 0, B_STEP, -B_STEP).astype(np.float32)
    z_   = _uniform(Z_STEP, Z_MAX,  ns, rng)
    t_   = _uniform(T_MIN,  T_MAX,  ns, rng)
    result["step_x"] = _to_tensor(np.stack([x_, y_, z_, t_], 1), device)

    nsh = counts["shoulder"]
    # Generate points in the annular region by rejection sampling
    cands_x  = _uniform(-A_PIPE, A_PIPE, nsh * 4, rng)
    cands_y  = _uniform(-B_PIPE, B_PIPE, nsh * 4, rng)
    in_outer = (np.abs(cands_x) <= A_PIPE) & (np.abs(cands_y) <= B_PIPE)
    in_inner = (np.abs(cands_x) <  A_STEP) & (np.abs(cands_y) <  B_STEP)
    in_annulus = in_outer & (~in_inner)
    x_ = cands_x[in_annulus][:nsh]
    y_ = cands_y[in_annulus][:nsh]
    if len(x_) < nsh:
        # pad with valid shoulder points on the outer strip in x
        pad_n = nsh - len(x_)
        x_pad = _uniform(A_STEP, A_PIPE, pad_n, rng) * np.where(
            rng.integers(0, 2, pad_n) == 0, 1, -1
        ).astype(np.float32)
        y_pad = _uniform(-B_PIPE, B_PIPE, pad_n, rng)
        x_ = np.concatenate([x_, x_pad])
        y_ = np.concatenate([y_, y_pad])
    z_  = np.full(nsh, Z_STEP, dtype=np.float32)
    t_  = _uniform(T_MIN, T_MAX, nsh, rng)
    result["shoulder"] = _to_tensor(np.stack([x_[:nsh], y_[:nsh], z_, t_], 1), device)

    return result


"""
INITIAL CONDITION SAMPLING
"""
def sample_ic(
        n: int,
        seed: int | None = None,
        device: str = DEVICE,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    accepted: list[np.ndarray] = []
    total = 0
    while total < n:
        x_ = _uniform(X_MIN, X_MAX, n * 3, rng)
        y_ = _uniform(Y_MIN, Y_MAX, n * 3, rng)
        z_ = _uniform(Y_MIN, Y_MAX, n * 3, rng)
        t_ = np.full(n * 3, T_MIN, dtype=np.float32)

        xt = torch.tensor(x_)
        yt = torch.tensor(y_)
        zt = torch.tensor(z_)
        mask = is_interior(xt, yt, zt, tol=1e-6).numpy()

        block = np.stack([x_[mask], y_[mask], z_[mask], t_[mask]], axis=1)
        accepted.append(block)
        total += len(block)
    
    pts_np = np.concatenate(accepted, axis=0)[:n]
    return _to_tensor(pts_np, device)


"""
COMBINED PDE SAMPLING (LHS + IMPORTANCE)
"""
def sample_pde_points(
        n_lhs: int,
        n_imp: int,
        seed: int | None = None,
        device: str = DEVICE,
) -> torch.Tensor:
    lhs = sample_interior_lhs(n_lhs, seed=seed, device=device)
    imp = sample_importance(n_imp, seed=seed, device=device)
    return torch.cat([lhs, imp], dim=0)