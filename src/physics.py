from __future__ import annotations
from typing import Tuple
import torch
import math
from config import (
    C_LIGHT, EPSILON_0, MU_0, INV_C2,
    Q_TOTAL, SIGMA_X, SIGMA_Y, SIGMA_Z, Z0_BUNCH, V_BUNCH,
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, T_MIN, T_MAX,
    RES_PHI_SCALE, RES_A_SCALE, RES_GAUGE_SCALE,
)


"""
COORDINATE NORMALIZATION (DIFFERENTIABLE)
"""

_X_MID  = (X_MAX + X_MIN) / 2.0;  _X_HALF = (X_MAX - X_MIN) / 2.0
_Y_MID  = (Y_MAX + Y_MIN) / 2.0;  _Y_HALF = (Y_MAX - Y_MIN) / 2.0
_Z_MID  = (Z_MAX + Z_MIN) / 2.0;  _Z_HALF = (Z_MAX - Z_MIN) / 2.0
_T_MID  = (T_MAX + T_MIN) / 2.0;  _T_HALF = (T_MAX - T_MIN) / 2.0

def normalize(x, y, z, t):
    return (
        (x - _X_MID) / _X_HALF,
        (y - _Y_MID) / _Y_HALF,
        (z - _Z_MID) / _Z_HALF,
        (t - _T_MID) / _T_HALF,
    )


"""
SOURCE TERMS
"""
def _rho0():
    return Q_TOTAL / ((2.0 * math.pi)**1.5 * SIGMA_X * SIGMA_Y * SIGMA_Z)

_RHO0 = _rho0()

def charge_density(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
) -> torch.Tensor:
    z_c = Z0_BUNCH + V_BUNCH * t
    exponent = (
        -(x ** 2) / (2.0 * SIGMA_X ** 2)
        - (y ** 2) / (2.0 * SIGMA_Y ** 2)
        - ((z - z_c) ** 2) / (2.0 * SIGMA_Z ** 2)
    )

    return _RHO0 * torch.exp(exponent)

def current_density(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = charge_density(x, y, z, t)
    zeros = torch.zeros_like(rho)
    return zeros, zeros, rho * V_BUNCH


"""
AUTOGRAD HELPERS
"""
def _grad1(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=u,
        inputs=v,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        allow_unused=False
    )[0]

def _grad2(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    u_v = _grad1(u, v)
    return _grad1(u_v, v)


"""
CORE: FORWARD PASS + RESIDUAL COMPUTATION
"""
def compute_pde_residuals(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    xn, yn, zn, tn = normalize(x, y, z, t)
    coords = torch.cat([xn, yn, zn, tn], dim=1)
    pots = model(coords)

    phi = pots[:, 0:1]
    Ax = pots[:, 1:2]
    Ay = pots[:, 2:3]
    Az = pots[:, 3:4]

    # D'Alembertian Helper
    def dalembert(u):
        u_xx = _grad2(u, x)
        u_yy = _grad2(u, y)
        u_zz = _grad2(u, z)
        u_tt = _grad2(u, t)
        return u_xx + u_yy + u_zz + u_tt
    
    # Source Terms
    rho = charge_density(x, y, z, t)
    Jx, Jy, Jz = current_density(x, y, z, t)

    # PDE Residuals (SI)
    res_phi_raw = dalembert(phi) + rho / EPSILON_0
    res_ax_raw = dalembert(Ax) + MU_0 * Jx
    res_ay_raw = dalembert(Ay) + MU_0 + Jy
    res_az_raw = dalembert(Az) + MU_0 + Jz

    # Lorenz Gauge
    dAx_dx = _grad1(Ax, x)
    dAy_dy = _grad1(Ay, y)
    dAz_dz = _grad1(Az, z)
    dphi_dt = _grad1(phi, t)
    res_gauge_raw = dAx_dx + dAy_dy + dAz_dz + INV_C2 * dphi_dt

    # Normalize for numerically balanced loss
    safe_phi = max(RES_PHI_SCALE, 1e-30)
    safe_a = max(RES_A_SCALE, 1e-30)
    safe_gauge = max(RES_GAUGE_SCALE, 1e-30)

    return {
        "phi":       res_phi_raw  / safe_phi,
        "ax":        res_ax_raw   / safe_a,
        "ay":        res_ay_raw   / safe_a,
        "az":        res_az_raw   / safe_a,
        "gauge":     res_gauge_raw / safe_gauge,
        "phi_raw":   res_phi_raw,
        "ax_raw":    res_ax_raw,
        "ay_raw":    res_ay_raw,
        "az_raw":    res_az_raw,
        "gauge_raw": res_gauge_raw,
    }


"""
ELECTROMAGNETIC FIELD COMPUTATION E AND B
"""
def compute_em_fields(
        model,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
) -> dict[str, torch.Tensor]:
    xn, yn, zn, tn = normalize(x, y, z, t)
    coords = torch.cat([xn, yn, zn, tn], dim=1)
    pots = model(coords)

    phi = pots[:, 0:1]
    Ax  = pots[:, 1:2]
    Ay  = pots[:, 2:3]
    Az  = pots[:, 3:4]

    # Potential gradients
    dphi_dx = _grad1(phi, x)
    dphi_dy = _grad1(phi, y)
    dphi_dz = _grad1(phi, z)

    dAx_dt  = _grad1(Ax, t)
    dAy_dt  = _grad1(Ay, t)
    dAz_dt  = _grad1(Az, t)

    # E field
    Ex = -dphi_dx - dAx_dt
    Ey = -dphi_dy - dAy_dt
    Ez = -dphi_dz - dAz_dt

    # Curl of A for B field
    dAz_dy  = _grad1(Az, y)
    dAy_dz  = _grad1(Ay, z)
    dAx_dz  = _grad1(Ax, z)
    dAz_dx  = _grad1(Az, x)
    dAy_dx  = _grad1(Ay, x)
    dAx_dy  = _grad1(Ax, y)

    Bx = dAz_dy  - dAy_dz
    By = dAx_dz  - dAz_dx
    Bz = dAy_dx  - dAx_dy

    return {
        "Ex": Ex, "Ey": Ey, "Ez": Ez,
        "Bx": Bx, "By": By, "Bz": Bz,
    }


"""
PEC BOUNDARY CONDITION ENFORCEMENT
"""
def tangential_E_on_face(
        fields: dict[str, torch.Tensor],
        face: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    Ex, Ey, Ez = fields["Ex"], fields["Ey"], fields["Ez"]

    if face in ("main_x", "step_x"):
        return Ey, Ez
    elif face in ("main_y", "step_y"):
        return Ex, Ez
    elif face == "shoulder":
        return Ex, Ey
    else:
        raise ValueError(f"Unknown face: {face}")