"""
evaluation/metrics.py — Quantitative Evaluation Metrics
=========================================================
Provides error metrics and physics-consistency checks for the
trained WAVE-AI model.

Metrics
-------
    relative_l2_error   — Relative L² norm between predicted and reference
    gauge_violation     — RMS Lorenz-gauge residual (should → 0 at convergence)
    div_B_violation     — ∇·B = 0 check (always exactly 0 for B = ∇×A analytically,
                        but numerical derivatives may introduce small error)
    field_energy_density — ε₀|E|²/2 + |B|²/(2μ₀) at given points
    peak_wakefield      — Maximum |Ez| on the beam axis over time
"""

from __future__ import annotations
import torch
import numpy as np
from config import (
    EPSILON_0, MU_0, INV_C2, C_LIGHT,
    Z_MIN, Z_MAX, T_MIN, T_MAX, DEVICE,
)
from src.physics import compute_em_fields, compute_pde_residuals, normalize

# Generic

def relative_l2_error(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Relative L² error:  ‖pred − ref‖₂ / (‖ref‖₂ + ε)
    """
    diff_norm = torch.norm(predicted.flatten() - reference.flatten(), p=2).item()
    ref_norm  = torch.norm(reference.flatten(), p=2).item()
    return diff_norm / (ref_norm + eps)


def mse(predicted: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean squared error."""
    return ((predicted - reference) ** 2).mean().item()

# Physics-consistency metrics

@torch.no_grad()
def gauge_violation_rms(
    model,
    pts: torch.Tensor,
    device: str = DEVICE,
) -> float:
    """
    Evaluate the Lorenz-gauge residual  ∇·A + (1/c²)∂Φ/∂t  (SI units: T/m)
    at the given collocation points and return the RMS value.

    Requires autograd, so torch.no_grad() is disabled inside via a context switch.
    """
    # We need gradients for the gauge check
    model.eval()
    x = pts[:, 0:1].clone().detach().to(device).requires_grad_(True)
    y = pts[:, 1:2].clone().detach().to(device).requires_grad_(True)
    z = pts[:, 2:3].clone().detach().to(device).requires_grad_(True)
    t = pts[:, 3:4].clone().detach().to(device).requires_grad_(True)

    with torch.enable_grad():
        res = compute_pde_residuals(model, x, y, z, t)
    gauge_raw = res["gauge_raw"].detach()
    rms = gauge_raw.pow(2).mean().sqrt().item()
    return rms


@torch.no_grad()
def div_B_rms(
    model,
    pts: torch.Tensor,
    device: str = DEVICE,
) -> float:
    """
    ∇·B should be identically zero because B = ∇×A  (in exact arithmetic).
    This metric measures the numerical divergence from the predicted A.
    Returns RMS of  ∂Bx/∂x + ∂By/∂y + ∂Bz/∂z.
    """
    from src.physics import _grad1
    model.eval()
    x = pts[:, 0:1].clone().detach().to(device).requires_grad_(True)
    y = pts[:, 1:2].clone().detach().to(device).requires_grad_(True)
    z = pts[:, 2:3].clone().detach().to(device).requires_grad_(True)
    t = pts[:, 3:4].clone().detach().to(device).requires_grad_(True)

    with torch.enable_grad():
        fields = compute_em_fields(model, x, y, z, t)
        Bx, By, Bz = fields["Bx"], fields["By"], fields["Bz"]
        dBx_dx = _grad1(Bx, x)
        dBy_dy = _grad1(By, y)
        dBz_dz = _grad1(Bz, z)
        div_B  = (dBx_dx + dBy_dy + dBz_dz).detach()

    return div_B.pow(2).mean().sqrt().item()


# Field energy

@torch.no_grad()
def field_energy_density(
    model,
    pts: torch.Tensor,
    device: str = DEVICE,
) -> torch.Tensor:
    """
    Electromagnetic energy density at the given points (J/m³):
        u = ε₀|E|²/2 + |B|²/(2μ₀)

    Returns Tensor of shape (N,).
    """
    model.eval()
    x = pts[:, 0:1].clone().detach().to(device).requires_grad_(True)
    y = pts[:, 1:2].clone().detach().to(device).requires_grad_(True)
    z = pts[:, 2:3].clone().detach().to(device).requires_grad_(True)
    t = pts[:, 3:4].clone().detach().to(device).requires_grad_(True)

    with torch.enable_grad():
        f = compute_em_fields(model, x, y, z, t)

    E2 = (f["Ex"]**2 + f["Ey"]**2 + f["Ez"]**2).detach().squeeze()
    B2 = (f["Bx"]**2 + f["By"]**2 + f["Bz"]**2).detach().squeeze()
    return 0.5 * EPSILON_0 * E2 + B2 / (2.0 * MU_0)


# Wakefield diagnostics

@torch.no_grad()
def longitudinal_wakefield_on_axis(
    model,
    n_z: int = 200,
    n_t: int = 50,
    device: str = DEVICE,
) -> dict[str, np.ndarray]:
    """
    Sample Ez on the beam axis (x=y=0) over a grid of (z, t) values.

    Returns
    -------
    dict with keys "z", "t", "Ez"  (numpy arrays)
    Ez : shape (n_t, n_z)
    """
    model.eval()
    z_grid = np.linspace(Z_MIN + 1e-4, Z_MAX - 1e-4, n_z, dtype=np.float32)
    t_grid = np.linspace(T_MIN, T_MAX, n_t, dtype=np.float32)

    Ez_out = np.zeros((n_t, n_z), dtype=np.float32)

    for i, t_val in enumerate(t_grid):
        x_ = torch.zeros(n_z, 1, device=device)
        y_ = torch.zeros(n_z, 1, device=device)
        z_ = torch.tensor(z_grid, device=device).unsqueeze(1).requires_grad_(True)
        t_ = torch.full((n_z, 1), t_val, device=device).requires_grad_(True)

        with torch.enable_grad():
            f = compute_em_fields(
                model,
                x_.requires_grad_(True),
                y_.requires_grad_(True),
                z_, t_,
            )
        Ez_out[i] = f["Ez"].detach().cpu().numpy().squeeze()

    return {"z": z_grid, "t": t_grid, "Ez": Ez_out}



# Summary report


def print_metrics_report(
    model,
    pts: torch.Tensor,
    device: str = DEVICE,
) -> None:
    """
    Print a formatted metrics report for the trained model.
    """
    print("\n" + "="*55)
    print("  WAVE-AI  — Physics Consistency Metrics")
    print("="*55)

    gauge = gauge_violation_rms(model, pts, device)
    print(f"  Lorenz gauge RMS violation  : {gauge:.4e}  T/m")

    divB  = div_B_rms(model, pts, device)
    print(f"  ∇·B RMS                     : {divB:.4e}  T/m")

    energy = field_energy_density(model, pts, device)
    print(f"  Mean field energy density   : {energy.mean().item():.4e}  J/m³")
    print(f"  Max  field energy density   : {energy.max().item():.4e}  J/m³")

    print("="*55 + "\n")