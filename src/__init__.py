# src/__init__.py
from src.network  import WAVENetwork
from src.physics  import (
    charge_density,
    current_density,
    compute_pde_residuals,
    compute_em_fields,
    normalize,
)
from src.geometry import (
    is_interior,
    aperture_at,
    classify_boundary,
)
from src.sampling import (
    sample_interior_lhs,
    sample_importance,
    sample_boundary,
    sample_ic,
    sample_pde_points,
)
from src.loss    import pde_loss, bc_loss, ic_loss, total_loss
from src.trainer import Trainer

__all__ = [
    "WAVENetwork",
    "charge_density", "current_density",
    "compute_pde_residuals", "compute_em_fields", "normalize",
    "is_interior", "aperture_at", "classify_boundary",
    "sample_interior_lhs", "sample_importance", "sample_boundary",
    "sample_ic", "sample_pde_points",
    "pde_loss", "bc_loss", "ic_loss", "total_loss",
    "Trainer",
]