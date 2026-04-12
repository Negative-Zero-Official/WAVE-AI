from __future__ import annotations
from typing import Tuple
import torch
from config import (
    A_PIPE, B_PIPE, A_STEP, B_STEP, Z_STEP,
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, T_MIN, T_MAX
)

"""
APERTURE QUERY
"""
def aperture_at(z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    upstream = (z < Z_STEP)
    a = torch.where(upstream, torch.full_like(z, A_PIPE), torch.full_like(z, A_STEP))
    b = torch.where(upstream, torch.full_like(z, B_PIPE), torch.full_like(z, B_STEP))
    return a, b


"""
INTERIOR CHECK
"""
def is_interior(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
        tol: float = 0.0,
) -> torch.Tensor:
    a, b = aperture_at(z)
    return (torch.abs(x) < a - tol) & (torch.abs(y) < b - tol)


"""
PEC BOUNDARY FACE HELPERS
"""
def on_main_x_wall(x, z, tol=1e-6):
    return (torch.abs(torch.abs(x) - A_PIPE) < tol) & (z <= Z_STEP + tol)

def on_main_y_wall(y, z, tol=1e-6):
    return (torch.abs(torch.abs(y) - B_PIPE) < tol) & (z <= Z_STEP - tol)

def on_step_x_wall(x, z, tol=1e-6):
    return (torch.abs(torch.abs(x) - A_STEP) < tol) & (z >= Z_STEP - tol)

def on_step_y_wall(y, z, tol=1e-6):
    return (torch.abs(torch.abs(y) - B_STEP) < tol) & (z >= Z_STEP - tol)

def on_shoulder(x, y, z, tol=1e-6):
    at_step = torch.abs(z - Z_STEP) < tol
    in_annulus = (
        ((torch.abs(x) >= A_STEP - tol) & (torch.abs(x) <= A_PIPE + tol)) |
        ((torch.abs(y) >= B_STEP - tol) & (torch.abs(y) <= B_PIPE + tol))
    )

    return at_step & in_annulus


"""
OUTWARD SURFACE NORMALS
"""
FACE_NORMALS = {
    "main_x": (1.0, 0.0, 0.0),
    "main_y": (0.0, 1.0, 0.0),
    "step_x": (1.0, 0.0, 0.0),
    "step_y": (0.0, 1.0, 0.0),
    "shoulder": (0.0, 0.0, 1.0),
}


"""
CONVENIENCE: CLASSIFY A TENSOR OF BOUNDARY POINTS
"""

def classify_boundary(
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
) -> dict[str, torch.BoolTensor]:
    return {
        "main_x": on_main_x_wall(x, z),
        "main_y": on_main_y_wall(y, z),
        "step_x": on_step_x_wall(x, z),
        "step_y": on_step_y_wall(y, z),
        "shoulder": on_shoulder(x, y, z),
    }