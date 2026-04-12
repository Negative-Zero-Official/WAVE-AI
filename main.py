"""
main.py — WAVE-AI Entry Point
==============================
Usage
-----
    # Full training (Adam + L-BFGS)
    python main.py

    # Quick smoke-test (fewer epochs, smaller batches)
    python main.py --quick

    # Resume from a checkpoint
    python main.py --resume checkpoints/wave_ai_adam_ep02000.pt

    # Evaluate only (no training)
    python main.py --eval-only --resume checkpoints/wave_ai_final_ep05500.pt

    # Override training lengths
    python main.py --adam-epochs 3000 --lbfgs-epochs 300

    # Skip L-BFGS phase
    python main.py --adam-epochs 5000 --no-lbfgs
"""

import argparse
import os
import sys
import time
import torch

#  Make sure the project root is on the Python path 
sys.path.insert(0, os.path.dirname(__file__))

import config as cfg
from src.network  import WAVENetwork
from src.trainer  import Trainer
from src.sampling import (
    sample_pde_points, sample_boundary, sample_ic,
)
from evaluation.metrics   import print_metrics_report
from evaluation.visualize import (
    plot_loss_history,
    plot_efield_slice_zx,
    plot_charge_density_snapshot,
    plot_wakefield_on_axis,
    plot_sampling_distribution,
    plot_potential_slice,
)

# Argument parser

def parse_args():
    p = argparse.ArgumentParser(
        description="WAVE-AI: 3D Physics-Informed Neural Modeling of Transient Fields"
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke-test mode: short epochs, small batches, fast feedback.",
    )
    p.add_argument(
        "--resume", type=str, default=None, metavar="CKPT",
        help="Path to a checkpoint file to resume training from.",
    )
    p.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; only run evaluation & plots (requires --resume).",
    )
    p.add_argument(
        "--adam-epochs", type=int, default=None,
        help="Override number of Adam epochs.",
    )
    p.add_argument(
        "--lbfgs-epochs", type=int, default=None,
        help="Override number of L-BFGS epochs.",
    )
    p.add_argument(
        "--no-lbfgs", action="store_true",
        help="Skip the L-BFGS phase entirely.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Compute device: 'cuda' or 'cpu'. Default: auto-detect.",
    )
    return p.parse_args()

# Quick-mode overrides

QUICK_OVERRIDES = dict(
    N_PDE           = 4_000,
    N_IMPORTANCE    = 1_000,
    N_BC            =   800,
    N_IC            =   400,
    BATCH_PDE       = 1_000,
    BATCH_BC        =   200,
    BATCH_IC        =   100,
    N_EPOCHS_ADAM   =   200,
    N_EPOCHS_LBFGS  =    50,
    SAVE_INTERVAL   =   100,
    LOG_INTERVAL    =    20,
)


def apply_quick_mode():
    """Monkey-patch config for a fast sanity-check run."""
    for k, v in QUICK_OVERRIDES.items():
        setattr(cfg, k, v)
    print("  [QUICK MODE] Reduced epochs and batch sizes for fast testing.\n")


# Evaluation + plotting

def run_evaluation(model, device: str) -> None:
    """Generate all diagnostic plots and print physics metrics."""
    print("\n Evaluation & Visualisation ")
    t_mid = (cfg.T_MIN + cfg.T_MAX) / 2.0
    t_end = cfg.T_MAX * 0.9

    # 1.  Charge-density snapshot
    print("  Plotting source charge density …")
    plot_charge_density_snapshot(t_val=t_mid)

    # 2.  E-field slices at two time snapshots
    for t_val in [t_mid, t_end]:
        for comp in ["Ez", "Ex"]:
            print(f"  Plotting {comp} slice at t = {t_val:.2e} s …")
            plot_efield_slice_zx(model, t_val=t_val, component=comp, device=device)

    # 3.  Wakefield on axis
    for t_val in [t_mid, t_end]:
        print(f"  Plotting wakefield Ez on axis at t = {t_val:.2e} s …")
        plot_wakefield_on_axis(model, t_val=t_val, device=device)

    # 4.  Potential slices  (Φ and Az)
    for comp_idx in [0, 3]:
        print(f"  Plotting potential component {comp_idx} …")
        plot_potential_slice(model, t_val=t_mid, component=comp_idx, device=device)

    # 5.  Physics consistency metrics
    print("  Computing physics consistency metrics …")
    test_pts = sample_pde_points(
        n_lhs=2000, n_imp=500, seed=77, device=device
    )
    print_metrics_report(model, test_pts, device=device)

    print(" Evaluation complete \n")


# Main

def main() -> None:
    args = parse_args()

    #  Device 
    device = args.device or cfg.DEVICE
    cfg.DEVICE = device

    #  Reproducibility 
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    #  Quick mode 
    if args.quick:
        apply_quick_mode()

    #  Epoch overrides 
    n_adam  = args.adam_epochs  or cfg.N_EPOCHS_ADAM
    n_lbfgs = args.lbfgs_epochs or cfg.N_EPOCHS_LBFGS
    if args.no_lbfgs:
        n_lbfgs = 0

    #  Print banner & config summary 
    print(f"  Device      : {device}")
    print(f"  Adam epochs : {n_adam}")
    print(f"  LBFGS epochs: {n_lbfgs}")
    print(f"  PDE points  : {cfg.N_PDE + cfg.N_IMPORTANCE} (LHS + importance)")
    print(f"  BC points   : {cfg.N_BC}")
    print(f"  IC points   : {cfg.N_IC}")
    print()

    #  Build model 
    model = WAVENetwork(
        in_features  = 4,
        out_features = 4,
        hidden_size  = cfg.HIDDEN_SIZE,
        num_layers   = cfg.NUM_LAYERS,
        omega_0      = cfg.OMEGA_0,
    ).to(device)
    print(f"  Model       : {model}\n")

    #  Build trainer 
    trainer = Trainer(model=model, device=device, seed=args.seed)

    #  Optionally resume 
    if args.resume:
        trainer.load_checkpoint(args.resume)

    #  Training 
    if not args.eval_only:
        t_start = time.time()

        # Adam phase
        if n_adam > 0:
            trainer.train_adam(n_epochs=n_adam)

        # L-BFGS refinement
        if n_lbfgs > 0:
            trainer.train_lbfgs(n_epochs=n_lbfgs)

        elapsed = time.time() - t_start
        print(f"\n  Total training time: {elapsed / 60:.1f} min\n")

        # Save final checkpoint and loss history
        trainer.save_checkpoint(trainer.start_epoch, phase="final")
        trainer._save_history()

        # Plot loss curves
        if trainer.history:
            print("  Plotting loss history …")
            plot_loss_history(trainer.history)

    #  Evaluation 
    run_evaluation(model, device)

    #  Sampling visualisation 
    print("  Plotting collocation-point distribution …")
    pde_pts = sample_pde_points(
        n_lhs=cfg.N_PDE,
        n_imp=cfg.N_IMPORTANCE,
        seed=args.seed,
        device=device,
    )
    plot_sampling_distribution(pde_pts, title="PDE collocation points")

    print("\n  [OK]  All done.  Outputs are in:", os.path.abspath(cfg.OUTPUT_DIR))


if __name__ == "__main__":
    main()