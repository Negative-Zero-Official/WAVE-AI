"""
evaluation/visualize.py — WAVE-AI Visualization Utilities
===========================================================
Generates publication-quality plots for:

    1.  Loss history (total + component breakdown)
    2.  2D spatial slices of E and B field components
    3.  Scalar / vector potential slices
    4.  Longitudinal wakefield  Ez(z, t=const)
    5.  Charge-density source snapshot
    6.  Collocation-point distribution check

All plots are saved as PNG to the OUTPUT_DIR defined in config.py.
"""

from __future__ import annotations
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for headless servers)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, T_MIN, T_MAX,
    A_PIPE, B_PIPE, A_STEP, B_STEP, Z_STEP,
    OUTPUT_DIR, DEVICE,
)
from src.physics import compute_em_fields, normalize, charge_density


# Helpers

def _save(fig, name: str) -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK]  Figure saved → {path}")
    return path


def _symlog_norm(data: np.ndarray, linthresh: float | None = None):
    """Symmetric-log normalisation, auto-selecting linthresh if needed."""
    absmax = np.abs(data).max()
    if absmax == 0:
        return mcolors.Normalize(vmin=-1, vmax=1)
    if linthresh is None:
        linthresh = max(absmax * 1e-3, 1e-30)
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=-absmax, vmax=absmax)


def _step_outline(ax, color="black", lw=1.5):
    """Draw the step-collimator cross-section outline on an (x or y) vs z axes."""
    # Main pipe walls (z < Z_STEP)
    ax.plot([Z_MIN, Z_STEP], [ A_PIPE,  A_PIPE], color=color, lw=lw)
    ax.plot([Z_MIN, Z_STEP], [-A_PIPE, -A_PIPE], color=color, lw=lw)
    # Step shoulder
    ax.plot([Z_STEP, Z_STEP], [A_STEP, A_PIPE], color=color, lw=lw)
    ax.plot([Z_STEP, Z_STEP], [-A_PIPE, -A_STEP], color=color, lw=lw)
    # Step pipe walls (z > Z_STEP)
    ax.plot([Z_STEP, Z_MAX], [ A_STEP,  A_STEP], color=color, lw=lw)
    ax.plot([Z_STEP, Z_MAX], [-A_STEP, -A_STEP], color=color, lw=lw)


# 1.  Loss history

def plot_loss_history(
    history: list[dict] | str,
    save_name: str = "loss_history",
) -> str:
    """
    Plot total and component loss curves over training epochs.

    Parameters
    ----------
    history  : list of loss dicts OR path to loss_history.json
    save_name: base filename (without extension)
    """
    if isinstance(history, str):
        with open(history) as f:
            history = json.load(f)

    epochs = [d["epoch"] for d in history]
    keys   = ["total", "pde", "bc", "ic", "pde_gauge"]
    labels = {
        "total":     "Total",
        "pde":       "PDE (wave eq.)",
        "bc":        "BC  (PEC walls)",
        "ic":        "IC  (t = 0)",
        "pde_gauge": "Lorenz gauge",
    }
    colors = ["black", "steelblue", "tomato", "seagreen", "orange"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scale in zip(axes, ["linear", "log"]):
        for key, col in zip(keys, colors):
            vals = [d.get(key, float("nan")) for d in history]
            if scale == "linear":
                ax.plot(epochs, vals, label=labels[key], color=col, lw=1.5)
            else:
                ax.semilogy(epochs, vals, label=labels[key], color=col, lw=1.5)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Training Loss ({scale} scale)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("WAVE-AI Training Loss History", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, save_name)


# 2.  2D E-field slice  (z-x plane at y=0)

@torch.no_grad()
def plot_efield_slice_zx(
    model,
    t_val: float,
    n_z: int = 150,
    n_x: int = 100,
    component: str = "Ez",
    device: str = DEVICE,
    save_name: str | None = None,
) -> str:
    """
    Plot a selected E-field component on the z-x plane at y = 0.

    Parameters
    ----------
    model     : trained WAVENetwork
    t_val     : fixed time value (s)
    component : "Ex", "Ey", or "Ez"
    """
    model.eval()

    z_vals = np.linspace(Z_MIN + 1e-4, Z_MAX - 1e-4, n_z, dtype=np.float32)
    x_vals = np.linspace(X_MIN + 1e-4, X_MAX - 1e-4, n_x, dtype=np.float32)

    ZZ, XX = np.meshgrid(z_vals, x_vals)   # shape (n_x, n_z)
    N = n_x * n_z

    # Flatten and filter to interior only
    x_flat = torch.tensor(XX.ravel(), device=device).unsqueeze(1)
    y_flat = torch.zeros_like(x_flat)
    z_flat = torch.tensor(ZZ.ravel(), device=device).unsqueeze(1)
    t_flat = torch.full_like(x_flat, t_val)

    x_flat.requires_grad_(True)
    y_flat.requires_grad_(True)
    z_flat.requires_grad_(True)
    t_flat.requires_grad_(True)

    with torch.enable_grad():
        f = compute_em_fields(model, x_flat, y_flat, z_flat, t_flat)

    E_field = f[component].detach().cpu().numpy().reshape(n_x, n_z)

    # Mask out conductor region at z > Z_STEP outside step aperture
    for iz, z_ in enumerate(z_vals):
        a = A_STEP if z_ >= Z_STEP else A_PIPE
        for ix, x_ in enumerate(x_vals):
            if abs(x_) >= a:
                E_field[ix, iz] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))
    norm = _symlog_norm(E_field[~np.isnan(E_field)])
    im = ax.pcolormesh(ZZ, XX, E_field, norm=norm, cmap="RdBu_r", shading="auto")
    _step_outline(ax)
    ax.set_xlabel("z  (m)")
    ax.set_ylabel("x  (m)")
    ax.set_title(f"WAVE-AI  —  {component}  at  y=0,  t = {t_val:.2e} s")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(X_MIN, X_MAX)
    plt.colorbar(im, ax=ax, label=f"{component}  (V/m)")
    fig.tight_layout()

    name = save_name or f"{component}_slice_zx_t{int(t_val*1e11):04d}"
    return _save(fig, name)


# 3.  Charge-density snapshot

def plot_charge_density_snapshot(
    t_val: float,
    n_z: int = 200,
    n_x: int = 80,
    device: str = DEVICE,
    save_name: str = "charge_density_snapshot",
) -> str:
    """Visualise the source charge density on the z–x plane at y = 0."""
    z_vals = np.linspace(Z_MIN, Z_MAX, n_z, dtype=np.float32)
    x_vals = np.linspace(X_MIN, X_MAX, n_x, dtype=np.float32)
    ZZ, XX = np.meshgrid(z_vals, x_vals)

    x_t = torch.tensor(XX.ravel(), device=device).unsqueeze(1)
    y_t = torch.zeros_like(x_t)
    z_t = torch.tensor(ZZ.ravel(), device=device).unsqueeze(1)
    t_t = torch.full_like(x_t, t_val)

    with torch.no_grad():
        rho = charge_density(x_t, y_t, z_t, t_t)
    rho_np = rho.cpu().numpy().reshape(n_x, n_z)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.pcolormesh(ZZ, XX, rho_np, cmap="inferno", shading="auto")
    _step_outline(ax, color="cyan")
    ax.set_xlabel("z  (m)")
    ax.set_ylabel("x  (m)")
    ax.set_title(f"Charge density  ρ(x, y=0, z, t={t_val:.2e} s)  [C/m³]")
    plt.colorbar(im, ax=ax, label="ρ  (C/m³)")
    fig.tight_layout()
    return _save(fig, save_name)


# 4.  Wakefield  Ez(z) on axis at fixed t

@torch.no_grad()
def plot_wakefield_on_axis(
    model,
    t_val: float,
    n_z: int = 300,
    device: str = DEVICE,
    save_name: str | None = None,
) -> str:
    """
    Plot the longitudinal wakefield Ez(z) on the beam axis (x=y=0) at time t.
    """
    model.eval()
    z_vals = np.linspace(Z_MIN + 1e-4, Z_MAX - 1e-4, n_z, dtype=np.float32)

    x_ = torch.zeros(n_z, 1, device=device).requires_grad_(True)
    y_ = torch.zeros(n_z, 1, device=device).requires_grad_(True)
    z_ = torch.tensor(z_vals, device=device).unsqueeze(1).requires_grad_(True)
    t_ = torch.full((n_z, 1), t_val, device=device).requires_grad_(True)

    with torch.enable_grad():
        f = compute_em_fields(model, x_, y_, z_, t_)

    Ez = f["Ez"].detach().cpu().numpy().squeeze()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(z_vals, Ez, color="steelblue", lw=1.5, label="$E_z$  on axis")
    ax.axvline(Z_STEP, color="gray", ls="--", lw=1, label="Step (z=0)")
    ax.set_xlabel("z  (m)")
    ax.set_ylabel("$E_z$  (V/m)")
    ax.set_title(f"Longitudinal wakefield  $E_z$ on axis  (t = {t_val:.2e} s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    name = save_name or f"wakefield_Ez_t{int(t_val * 1e11):04d}"
    return _save(fig, name)


# 5.  Collocation-point distribution check

def plot_sampling_distribution(
    pts: torch.Tensor,
    title: str = "Collocation points",
    save_name: str = "sampling_dist",
) -> str:
    """
    Scatter plot of (z, x) coordinates of the collocation points,
    providing a visual sanity-check that points cover the domain.
    """
    arr = pts.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.scatter(arr[:, 2], arr[:, 0], s=0.5, alpha=0.3, color="steelblue")
    _step_outline(ax, color="red", lw=1)
    ax.set_xlabel("z  (m)")
    ax.set_ylabel("x  (m)")
    ax.set_title("z – x projection")
    ax.set_xlim(Z_MIN, Z_MAX)
    ax.set_ylim(X_MIN, X_MAX)

    ax = axes[1]
    ax.scatter(arr[:, 3] * 1e9, arr[:, 2], s=0.5, alpha=0.3, color="tomato")
    ax.set_xlabel("t  (ns)")
    ax.set_ylabel("z  (m)")
    ax.set_title("t – z projection")

    fig.suptitle(f"{title}  (N = {len(arr):,})", fontsize=11)
    fig.tight_layout()
    return _save(fig, save_name)


# 6.  Potential slices

@torch.no_grad()
def plot_potential_slice(
    model,
    t_val: float,
    component: int = 0,
    n_z: int = 150,
    n_x: int = 80,
    device: str = DEVICE,
    save_name: str | None = None,
) -> str:
    """
    Plot a potential component on the z–x plane at y = 0.

    component : 0 = Φ,  1 = Ax,  2 = Ay,  3 = Az
    """
    comp_labels = ["Φ  (V)", "Ax  (T·m)", "Ay  (T·m)", "Az  (T·m)"]
    model.eval()

    z_vals = np.linspace(Z_MIN + 1e-4, Z_MAX - 1e-4, n_z, dtype=np.float32)
    x_vals = np.linspace(X_MIN + 1e-4, X_MAX - 1e-4, n_x, dtype=np.float32)
    ZZ, XX = np.meshgrid(z_vals, x_vals)
    N = n_x * n_z

    xn, yn, zn, tn = normalize(
        torch.tensor(XX.ravel(), device=device).unsqueeze(1),
        torch.zeros(N, 1, device=device),
        torch.tensor(ZZ.ravel(), device=device).unsqueeze(1),
        torch.full((N, 1), t_val, device=device),
    )
    coords = torch.cat([xn, yn, zn, tn], dim=1)
    pots = model(coords).detach().cpu().numpy()[:, component].reshape(n_x, n_z)

    # Mask conductor
    for iz, z_ in enumerate(z_vals):
        a = A_STEP if z_ >= Z_STEP else A_PIPE
        for ix, x_ in enumerate(x_vals):
            if abs(x_) >= a:
                pots[ix, iz] = np.nan

    fig, ax = plt.subplots(figsize=(10, 4))
    valid = pots[~np.isnan(pots)]
    norm = _symlog_norm(valid) if len(valid) > 0 else None
    im = ax.pcolormesh(ZZ, XX, pots, norm=norm, cmap="RdBu_r", shading="auto")
    _step_outline(ax)
    ax.set_xlabel("z  (m)")
    ax.set_ylabel("x  (m)")
    ax.set_title(
        f"Potential  {comp_labels[component]}  at  y=0,  t={t_val:.2e} s"
    )
    plt.colorbar(im, ax=ax, label=comp_labels[component])
    fig.tight_layout()

    label = ["phi", "Ax", "Ay", "Az"][component]
    name = save_name or f"potential_{label}_t{int(t_val * 1e11):04d}"
    return _save(fig, name)