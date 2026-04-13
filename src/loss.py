from __future__ import annotations
import math
import torch
import numpy as np
from config import (
    EPSILON_0, LAMBDA_BC, LAMBDA_IC, LAMBDA_GAUGE, LAMBDA_PDE, LAMBDA_PHI_OVERRIDE, 
    LAMBDA_REG, LAMBDA_SMOOTH, PHI_REF, A_REF, E_REF, Q_TOTAL, SIGMA_X, SIGMA_Z,
    SPECTRAL_FILTER_ENABLED, FILTER_CUTOFF_FREQ, C_LIGHT, T_MIN, V_BUNCH, Z0_BUNCH, Z_MAX, Z_MIN
)
from src.physics import compute_pde_residuals, compute_em_fields, tangential_E_on_face
from src.geometry import classify_boundary
from src.physics import normalize   # for IC forward pass


"""
HELPERS
"""

def _mse(tensor: torch.Tensor) -> torch.Tensor:
    # Use float64 accumulation for squared residuals to avoid
    # float32 overflow on very large PDE residuals during early training.
    return tensor.pow(2).mean(dtype=torch.float64).to(tensor.dtype)

def _make_leaf(pts: torch.Tensor) -> tuple[torch.Tensor, ...]:
    x = pts[:, 0:1].clone().detach().requires_grad_(True)
    y = pts[:, 1:2].clone().detach().requires_grad_(True)
    z = pts[:, 2:3].clone().detach().requires_grad_(True)
    t = pts[:, 3:4].clone().detach().requires_grad_(True)
    return x, y, z, t


def _compute_regularization(model) -> torch.Tensor:
    """Tikhonov regularization: penalize large final-layer weights."""
    if LAMBDA_REG < 1e-6:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    # Get final layer weights
    final_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            final_layer = module
    
    if final_layer is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    # Regularize: ||W_L||_F^2 (Frobenius norm)
    reg = torch.sum(final_layer.weight ** 2)
    return LAMBDA_REG * reg


def _spectral_filter_low_pass(coeffs: torch.Tensor, spatial_extent: float) -> torch.Tensor:
    """Apply low-pass spectral filter to 1D field (assumes valid shape)."""
    if not SPECTRAL_FILTER_ENABLED or len(coeffs.shape) != 1:
        return coeffs
    
    # Compute frequency domain
    fft = torch.fft.rfft(coeffs)
    freqs = torch.fft.rfftfreq(len(coeffs), d=spatial_extent / len(coeffs))
    
    # Create low-pass filter (smooth cutoff)
    cutoff_norm = FILTER_CUTOFF_FREQ / (C_LIGHT * 1e9)  # Normalize by c
    mask = torch.exp(-0.5 * (torch.abs(freqs) / cutoff_norm) ** 2)
    
    # Apply filter in frequency domain
    fft_filtered = fft * mask
    filtered = torch.fft.irfft(fft_filtered, n=len(coeffs))
    
    return filtered[:len(coeffs)]


"""
PDE RESIDUAL LOSS
"""
def pde_loss(model, pts: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    x, y, z, t = _make_leaf(pts)
    res = compute_pde_residuals(model, x, y, z, t)

    l_phi = _mse(res["phi"])
    l_ax = _mse(res["ax"])
    l_ay = _mse(res["ay"])
    l_az = _mse(res["az"])
    l_gauge = _mse(res["gauge"])

    for name, term in [
        ("pde_phi", l_phi),
        ("pde_ax", l_ax),
        ("pde_ay", l_ay),
        ("pde_az", l_az),
        ("pde_gauge", l_gauge),
    ]:
        if not torch.isfinite(term):
            print(f"WARNING: Non-finite PDE term {name}: {term.item()}")

    # PHASE 2 FIX: Strengthen scalar potential enforcement
    loss = 2.0 * l_phi * (1.0 + LAMBDA_PHI_OVERRIDE) + l_ax + l_ay + l_az + LAMBDA_GAUGE * l_gauge
    
    # PHASE 2 FIX: Add regularization to reduce high-frequency artifacts
    reg = _compute_regularization(model)
    loss = loss + reg

    detail = {
        "pde_phi": l_phi.item(),
        "pde_ax": l_ax.item(),
        "pde_ay": l_ay.item(),
        "pde_az": l_az.item(),
        "pde_gauge": l_gauge.item(),
        "pde_reg": reg.item() if torch.isfinite(reg) else 0.0,
    }

    return loss, detail


"""
BOUNDARY CONDITONS
"""
def bc_loss(
        model,
        bc_pts: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    detail: dict[str, float] = {}

    for face, pts in bc_pts.items():
        if pts.shape[0] == 0:
            continue
        x, y, z, t = _make_leaf(pts)
        fields = compute_em_fields(model, x, y, z, t)
        Et1, Et2 = tangential_E_on_face(fields, face)
        face_loss = _mse(Et1 / E_REF) + _mse(Et2 / E_REF)
        if not torch.isfinite(face_loss):
            print(f"WARNING: Non-finite BC loss on face {face}: {face_loss.item()}")
        detail[f"bc_{face}"] = face_loss.item()
        total_loss += face_loss
    
    return total_loss, detail


"""
INITIAL CONDITION LOSS (ALL POTENTIALS = 0 AT T=T_MIN)
"""
def ic_loss(
        model,
        ic_pts: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    xn, yn, zn, tn = normalize(
        ic_pts[:, 0:1],
        ic_pts[:, 1:2],
        ic_pts[:, 2:3],
        ic_pts[:, 3:4],
    )
    coords = torch.cat([xn, yn, zn, tn], dim=1)
    pots = model(coords)

    phi_pred = pots[:, 0:1]
    Ax_pred  = pots[:, 1:2]
    Ay_pred  = pots[:, 2:3]
    Az_pred  = pots[:, 3:4]

    # Physical coordinates
    x_phys = ic_pts[:, 0:1]
    y_phys = ic_pts[:, 1:2]
    z_phys = ic_pts[:, 2:3]

    # Relativistic Gaussian Approximation
    r_perp2 = x_phys**2 + y_phys**2
    z_rel = z_phys - (Z0_BUNCH + V_BUNCH * T_MIN)

    gamma = 1.0 / math.sqrt(1 - (V_BUNCH / C_LIGHT)**2)

    sigma_x = SIGMA_X
    sigma_z = SIGMA_Z

    phi_init = PHI_REF * torch.exp(
        -r_perp2 / (2 * sigma_x**2)
        - (gamma * z_rel)**2 / (2 * sigma_z**2)
    )

    Az_init = phi_init / C_LIGHT

    l_phi = _mse((phi_pred - phi_init) / PHI_REF)
    l_ax = _mse(Ax_pred / A_REF)
    l_ay = _mse(Ay_pred / A_REF)
    l_az = _mse((Az_pred - Az_init) / A_REF)

    for name, term in [
        ("ic_phi", l_phi),
        ("ic_ax", l_ax),
        ("ic_ay", l_ay),
        ("ic_az", l_az),
    ]:
        if not torch.isfinite(term):
            print(f"WARNING: Non-finite IC term {name}: {term.item()}")

    loss = 5.0 * l_phi + l_ax + l_ay + l_az
    detail = {
        "ic_phi": l_phi.item(),
        "ic_ax": l_ax.item(),
        "ic_ay": l_ay.item(),
        "ic_az": l_az.item()
    }

    return loss, detail


"""
COMPOSITE TOTAL LOSS
"""
def total_loss(
        model,
        pde_pts: torch.Tensor,
        bc_pts: dict[str, torch.Tensor],
        ic_pts: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    l_pde, d_pde = pde_loss(model, pde_pts)
    l_bc, d_bc = bc_loss(model, bc_pts)
    l_ic, d_ic = ic_loss(model, ic_pts)

    # Check for NaN/inf in individual losses
    losses = [l_pde, l_bc, l_ic]
    loss_names = ["pde", "bc", "ic"]
    for loss, name in zip(losses, loss_names):
        if not torch.isfinite(loss):
            print(f"WARNING: Non-finite {name} loss: {loss.item()}")

    loss = LAMBDA_PDE * l_pde + LAMBDA_BC * l_bc + LAMBDA_IC * l_ic
    
    # Final check
    if not torch.isfinite(loss):
        print(f"WARNING: Non-finite total loss: {loss.item()}")

    loss_dict: dict[str, float] = {
        "total": loss.item(),
        "pde": l_pde.item(),
        "bc": l_bc.item(),
        "ic": l_ic.item(),
        **d_pde,
        **d_bc,
        **d_ic
    }

    return loss, loss_dict