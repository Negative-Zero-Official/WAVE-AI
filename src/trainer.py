"""
src/trainer.py — WAVE-AI Training Engine
=========================================
Implements the two-phase training strategy common to PINNs:

Phase 1 — Adam optimizer (stochastic, large-scale exploration)
Phase 2 — L-BFGS refinement (quasi-Newton, fine-grained convergence)

During Adam, collocation points are *resampled* every epoch from the LHS /
importance distributions so the network sees diverse training points and
does not overfit to a fixed set.

During L-BFGS, a *fixed* point set is used because the closure may be called
multiple times per step (due to line search) and changing the data mid-step
would break the second-order curvature estimates.

Checkpointing saves model weights and the full loss history so training can
be resumed after interruption.
"""

from __future__ import annotations
import os
import time
import json
import math
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import (
    N_PDE, N_IMPORTANCE, N_BC, N_IC,
    N_LBFGS_PDE, N_LBFGS_IMPORTANCE, N_LBFGS_BC, N_LBFGS_IC,
    BATCH_PDE, BATCH_BC, BATCH_IC,
    LR_ADAM, N_EPOCHS_ADAM,
    LR_LBFGS, N_EPOCHS_LBFGS, LBFGS_MAX_ITER, LBFGS_HISTORY,
    CHECKPOINT_DIR, OUTPUT_DIR,
    LOG_INTERVAL, SAVE_INTERVAL,
    DEVICE,
)
from src.sampling import sample_pde_points, sample_boundary, sample_ic
from src.loss import total_loss


# 
# Helpers
# 

def _ensure_dirs():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR,     exist_ok=True)


def _format_loss(ld: dict) -> str:
    """Compact one-line string of the most important loss components."""
    return (
        f"total={ld['total']:.4e}  "
        f"pde={ld['pde']:.4e}  "
        f"bc={ld['bc']:.4e}  "
        f"ic={ld['ic']:.4e}  "
        f"gauge={ld.get('pde_gauge', float('nan')):.4e}"
    )


def _random_batch(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Draw a random sub-batch of n rows from pts (without replacement if possible)."""
    N = pts.shape[0]
    if N <= n:
        return pts
    idx = torch.randperm(N, device=pts.device)[:n]
    return pts[idx]


def _random_batch_dict(
    bc_dict: dict[str, torch.Tensor],
    n_total: int,
) -> dict[str, torch.Tensor]:
    """
    Randomly sub-sample boundary points from each face so the total
    count across all faces does not exceed n_total.
    """
    keys = list(bc_dict.keys())
    n_per_face = max(1, n_total // len(keys))
    return {k: _random_batch(v, n_per_face) for k, v in bc_dict.items()}


# 
# Main Trainer class
# 

class Trainer:
    """
    Two-phase PINN trainer for WAVE-AI.

    Parameters
    ----------
    model    : WAVENetwork — the neural network to train
    device   : str        — 'cuda' or 'cpu'
    seed     : int        — random seed for reproducibility
    """

    def __init__(
        self,
        model,
        device: str = DEVICE,
        seed: int = 42,
    ) -> None:
        self.model  = model.to(device)
        self.device = device
        self.seed   = seed
        self.history: list[dict[str, float]] = []   # loss log per epoch
        self.start_epoch = 0                         # supports resume
        _ensure_dirs()

    #  Pre-generate a large pool of boundary & IC points 

    def _resample_pools(self, seed_offset: int = 0):
        """Refresh the full pool of boundary and IC points."""
        self.bc_pool  = sample_boundary(N_BC,  seed=self.seed + seed_offset)
        self.ic_pool  = sample_ic(N_IC,         seed=self.seed + seed_offset + 1)

    #  Single loss evaluation  (used both by Adam and L-BFGS closure) 

    def _eval_loss(
        self,
        pde_pts: torch.Tensor,
        bc_pts:  dict[str, torch.Tensor],
        ic_pts:  torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return total_loss(self.model, pde_pts, bc_pts, ic_pts)

    # Phase 1: Adam

    def train_adam(self, n_epochs: int = N_EPOCHS_ADAM) -> None:
        """
        Adam optimisation with cosine learning-rate annealing and
        fresh collocation-point re-sampling each epoch.
        """
        print(f"\n{'='*60}")
        print(f"  WAVE-AI — Adam phase  ({n_epochs} epochs)")
        print(f"  Device  : {self.device}")
        print(f"  Network : {self.model}")
        print(f"{'='*60}\n")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR_ADAM)
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=LR_ADAM * 0.01)

        # Pre-generate boundary and IC pools (expensive — reuse across epochs)
        self._resample_pools(seed_offset=0)

        t0 = time.time()
        epoch_iter = tqdm(
            range(self.start_epoch, self.start_epoch + n_epochs),
            total=n_epochs,
            desc="Adam",
            unit="epoch",
            leave=True,
        )

        for epoch in epoch_iter:
            self.model.train()
            optimizer.zero_grad()

            #  Resample PDE points every epoch 
            pde_pts = sample_pde_points(
                n_lhs=BATCH_PDE,
                n_imp=BATCH_PDE // 4,
                seed=epoch,
                device=self.device,
            )

            #  Sub-sample BC and IC from the large pools 
            bc_pts  = _random_batch_dict(self.bc_pool,  BATCH_BC)
            ic_pts  = _random_batch(self.ic_pool, BATCH_IC)

            #  Forward + loss 
            loss, ld = self._eval_loss(pde_pts, bc_pts, ic_pts)

            #  Backward 
            loss.backward()

            # Gradient clipping prevents exploding gradients in early training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            ld["epoch"] = epoch
            ld["lr"]    = scheduler.get_last_lr()[0]
            self.history.append(ld)
            epoch_iter.set_postfix(total=f"{ld['total']:.4e}", lr=f"{ld['lr']:.2e}")

            #  Logging 
            if epoch % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                epoch_iter.write(
                    f"[Adam {epoch:5d}/{self.start_epoch + n_epochs - 1}]  "
                    f"{_format_loss(ld)}  "
                    f"lr={ld['lr']:.2e}  {elapsed:.1f}s"
                )

            #  Checkpointing 
            if epoch % SAVE_INTERVAL == 0 and epoch > self.start_epoch:
                self.save_checkpoint(epoch, phase="adam")

            #  Refresh boundary/IC pool occasionally 
            if epoch % 1000 == 999:
                self._resample_pools(seed_offset=epoch)

        self.start_epoch += n_epochs
        print(f"\nAdam phase complete.  Final {_format_loss(self.history[-1])}\n")

    # Phase 2: L-BFGS

    def train_lbfgs(self, n_epochs: int = N_EPOCHS_LBFGS) -> None:
        """
        L-BFGS refinement using a FIXED set of training points.
        Points are fixed so that successive closure evaluations are consistent
        with the curvature estimates built by the quasi-Newton method.
        """
        print(f"\n{'='*60}")
        print(f"  WAVE-AI — L-BFGS phase  ({n_epochs} epochs)")
        print(f"{'='*60}\n")

        # Fixed training set for the entire L-BFGS phase (reduced size for efficiency)
        pde_pts_fixed = sample_pde_points(
            n_lhs=N_LBFGS_PDE,
            n_imp=N_LBFGS_IMPORTANCE,
            seed=self.seed + 9999,
            device=self.device,
        )
        bc_pts_fixed  = sample_boundary(N_LBFGS_BC,  seed=self.seed + 10000)
        ic_pts_fixed  = sample_ic(N_LBFGS_IC,         seed=self.seed + 10001)

        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=LR_LBFGS,
            max_iter=LBFGS_MAX_ITER,
            history_size=LBFGS_HISTORY,
            tolerance_change=1e-6,  # Relaxed from 1e-9
            tolerance_grad=1e-5,    # Relaxed from 1e-7
            line_search_fn="strong_wolfe",  # Re-enabled with relaxed tolerances
        )

        # Mutable closure state (so we can log it after each step)
        closure_ld: dict[str, float] = {}
        closure_call_count = 0
        lbfgs_failed = False

        def _zero_gradients():
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

        def closure():
            nonlocal closure_call_count
            closure_call_count += 1
            start_time = time.time()

            optimizer.zero_grad()
            loss, ld = self._eval_loss(pde_pts_fixed, bc_pts_fixed, ic_pts_fixed)

            # Check for NaN/inf in loss
            if not torch.isfinite(loss):
                print(f"WARNING: Non-finite loss detected in closure call {closure_call_count}: {loss.item()}")
                _zero_gradients()
                raise RuntimeError(
                    f"Invalid loss in L-BFGS closure (call {closure_call_count})"
                )

            loss.backward()

            # Check gradients before clipping
            total_grad_norm = torch.norm(
                torch.stack([
                    torch.norm(p.grad.detach())
                    for p in self.model.parameters()
                    if p.grad is not None
                ])
            )
            if not torch.isfinite(total_grad_norm):
                print(f"WARNING: Infinite gradients detected before clipping in closure call {closure_call_count}")
                _zero_gradients()
                raise RuntimeError(
                    f"Infinite gradients in L-BFGS closure (call {closure_call_count})"
                )

            # Gradient clipping prevents exploding gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            # Check for NaN/inf in gradients after clipping
            has_invalid_grad = any(
                not torch.isfinite(p.grad).all()
                for p in self.model.parameters()
                if p.grad is not None
            )
            if has_invalid_grad:
                print(f"WARNING: NaN/inf gradients detected in closure call {closure_call_count}")
                _zero_gradients()
                raise RuntimeError(
                    f"Invalid gradients in L-BFGS closure (call {closure_call_count})"
                )

            closure_ld.update(ld)
            closure_ld["closure_calls"] = closure_call_count
            closure_ld["grad_norm"] = grad_norm.item()
            closure_ld["closure_time"] = time.time() - start_time

            return loss

        t0 = time.time()
        epoch_iter = tqdm(
            range(n_epochs),
            total=n_epochs,
            desc="L-BFGS",
            unit="epoch",
            leave=True,
        )
        
        for epoch in epoch_iter:
            closure_call_count = 0  # Reset per epoch

            step_start = time.time()
            self.model.train()
            try:
                optimizer.step(closure)
            except RuntimeError as exc:
                message = str(exc)
                if "Invalid loss" in message or "Infinite gradients" in message or "Invalid gradients" in message:
                    epoch_iter.write(
                        f"[LBFGS {epoch:4d}/{n_epochs - 1}]  Aborted: {message}"
                    )
                    lbfgs_failed = True
                    break
                raise
            step_time = time.time() - step_start

            closure_ld["epoch"] = self.start_epoch + epoch
            closure_ld["step_time"] = step_time
            self.history.append(dict(closure_ld))
            epoch_iter.set_postfix(
                total=f"{closure_ld['total']:.4e}",
                calls=closure_ld.get("closure_calls", 0),
                grad_norm=f"{closure_ld.get('grad_norm', 0):.2e}",
                step_time=f"{step_time:.1f}s"
            )
            
            if epoch % LOG_INTERVAL == 0:
                elapsed = time.time() - t0
                epoch_iter.write(
                    f"[LBFGS {epoch:4d}/{n_epochs - 1}]  "
                    f"{_format_loss(closure_ld)}  "
                    f"calls={closure_ld.get('closure_calls', 0)}  "
                    f"grad_norm={closure_ld.get('grad_norm', 0):.2e}  "
                    f"step_time={step_time:.1f}s  {elapsed:.1f}s total"
                )

            if epoch % SAVE_INTERVAL == 0 and epoch > 0:
                self.save_checkpoint(self.start_epoch + epoch, phase="lbfgs")

        if lbfgs_failed:
            self.start_epoch += epoch
            print("\nL-BFGS phase aborted early due to invalid closure evaluation.\n")
        else:
            self.start_epoch += n_epochs
            print(f"\nL-BFGS phase complete.  Final {_format_loss(self.history[-1])}\n")

    # Full training

    def train(
        self,
        n_adam:  int = N_EPOCHS_ADAM,
        n_lbfgs: int = N_EPOCHS_LBFGS,
    ) -> None:
        """Run Adam phase followed by L-BFGS refinement."""
        self.train_adam(n_epochs=n_adam)
        self.train_lbfgs(n_epochs=n_lbfgs)
        self.save_checkpoint(self.start_epoch, phase="final")
        self._save_history()

    # Persistence

    def save_checkpoint(self, epoch: int, phase: str = "adam") -> None:
        path = os.path.join(
            CHECKPOINT_DIR, f"wave_ai_{phase}_ep{epoch:05d}.pt"
        )
        torch.save(
            {
                "epoch":       epoch,
                "model_state": self.model.state_dict(),
                "history":     self.history,
            },
            path,
        )
        tqdm.write(f"  ✓  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.history     = ckpt.get("history", [])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        print(f"  ✓  Loaded checkpoint from {path}  (epoch {self.start_epoch - 1})")

    def _save_history(self) -> None:
        path = os.path.join(OUTPUT_DIR, "loss_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  ✓  Loss history saved → {path}")

    def load_history(self, path: str | None = None) -> list[dict]:
        if path is None:
            path = os.path.join(OUTPUT_DIR, "loss_history.json")
        with open(path) as f:
            self.history = json.load(f)
        return self.history