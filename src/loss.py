from __future__ import annotations
import torch
from config import LAMBDA_BC, LAMBDA_IC, LAMBDA_GAUGE
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

    loss = l_phi + l_ax + l_ay + l_az + LAMBDA_GAUGE * l_gauge

    detail = {
        "pde_phi": l_phi.item(),
        "pde_ax": l_ax.item(),
        "pde_ay": l_ay.item(),
        "pde_az": l_az.item(),
        "pde_gauge": l_gauge.item()
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
        face_loss = _mse(Et1) + _mse(Et2)
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

    l_phi = _mse(pots[:, 0:1])
    l_ax = _mse(pots[:, 1:2])
    l_ay = _mse(pots[:, 2:3])
    l_az = _mse(pots[:, 3:4])

    loss = l_phi + l_ax + l_ay + l_az
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

    loss = l_pde + LAMBDA_BC * l_bc + LAMBDA_IC * l_ic
    
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