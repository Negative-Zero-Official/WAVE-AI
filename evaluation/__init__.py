# evaluation/__init__.py
from evaluation.metrics   import (
    relative_l2_error,
    mse,
    gauge_violation_rms,
    div_B_rms,
    field_energy_density,
    longitudinal_wakefield_on_axis,
    print_metrics_report,
)
from evaluation.visualize import (
    plot_loss_history,
    plot_efield_slice_zx,
    plot_charge_density_snapshot,
    plot_wakefield_on_axis,
    plot_sampling_distribution,
    plot_potential_slice,
)

__all__ = [
    "relative_l2_error", "mse",
    "gauge_violation_rms", "div_B_rms",
    "field_energy_density",
    "longitudinal_wakefield_on_axis",
    "print_metrics_report",
    "plot_loss_history",
    "plot_efield_slice_zx",
    "plot_charge_density_snapshot",
    "plot_wakefield_on_axis",
    "plot_sampling_distribution",
    "plot_potential_slice",
]