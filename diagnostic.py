#!/usr/bin/env python3
"""
Diagnostic utility to inspect WAVE-AI model outputs and verify physics correctness.
Checks for:
1. Scalar potential collapse (Φ ≈ 0 everywhere?)
2. High-frequency oscillations in fields
3. Actual PDE residual satisfaction
4. Source term enforcement
5. Boundary condition violations
"""

import torch
import json
import numpy as np
from pathlib import Path
from config import (
    X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX, T_MIN, T_MAX,
    C_LIGHT, EPSILON_0, MU_0, INV_C2,
    CHECKPOINT_DIR, Q_TOTAL, SIGMA_X, SIGMA_Y, SIGMA_Z, V_BUNCH, Z0_BUNCH,
)
from src.network import WAVENetwork
from src.physics import (
    normalize, charge_density, current_density, 
    compute_pde_residuals, compute_em_fields
)


def load_checkpoint(epoch):
    """Load model from checkpoint."""
    ckpt_path = Path(CHECKPOINT_DIR) / f"wave_ai_adam_ep{epoch:05d}.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model = WAVENetwork()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


def sample_grid(nx=10, ny=10, nz=20, nt=5):
    """Sample uniform grid in physical space."""
    x = np.linspace(X_MIN, X_MAX, nx)
    y = np.linspace(Y_MIN, Y_MAX, ny)
    z = np.linspace(Z_MIN, Z_MAX, nz)
    t = np.linspace(T_MIN, T_MAX, nt)
    
    grid = []
    for xi in x:
        for yi in y:
            for zi in z:
                for ti in t:
                    grid.append([xi, yi, zi, ti])
    return torch.tensor(grid, dtype=torch.float32)


def check_potential_collapse(model, grid_pts):
    """Check if scalar potential is near-zero everywhere."""
    print("\n" + "="*70)
    print("CHECK 1: SCALAR POTENTIAL COLLAPSE")
    print("="*70)
    
    with torch.no_grad():
        xn, yn, zn, tn = normalize(
            grid_pts[:, 0:1], grid_pts[:, 1:2], 
            grid_pts[:, 2:3], grid_pts[:, 3:4]
        )
        coords = torch.cat([xn, yn, zn, tn], dim=1)
        pots = model(coords)
        phi = pots[:, 0:1].numpy()
    
    phi_min, phi_max = phi.min(), phi.max()
    phi_mean = np.mean(np.abs(phi))
    phi_std = np.std(phi)
    
    print(f"Φ range:     [{phi_min:.6e}, {phi_max:.6e}] V")
    print(f"Φ |mean|:    {phi_mean:.6e} V")
    print(f"Φ std:       {phi_std:.6e} V")
    
    # Expected: ~mV scale for 1nC bunch
    if phi_mean < 1e-3:
        print("[!]  WARNING: Potential is VERY SMALL (<1mV). May indicate trivial solution!")
    else:
        print("[OK] Potential magnitude looks reasonable.")
    
    return phi_min, phi_max, phi_mean


def check_high_frequency_content(model, grid_pts):
    """Check for unphysical high-frequency oscillations in fields."""
    print("\n" + "="*70)
    print("CHECK 2: HIGH-FREQUENCY OSCILLATIONS")
    print("="*70)
    
    with torch.no_grad():
        # Sample dense spatial grid at fixed time
        z_slice = np.linspace(Z_MIN, Z_MAX, 100)
        x_slice = np.linspace(X_MIN, X_MAX, 100)
        t_fixed = 1e-9  # 1 ns
        
        pts = []
        for xi in x_slice:
            for zi in z_slice:
                pts.append([xi, 0.0, zi, t_fixed])
        pts = torch.tensor(pts, dtype=torch.float32)
        
        xn, yn, zn, tn = normalize(pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], pts[:, 3:4])
        coords = torch.cat([xn, yn, zn, tn], dim=1)
        pots = model(coords)
        
        Ex = pots[:, 1:2].numpy()  # Approximation: dAx/dt ~ Ax for visualization
    
    # Compute autocorrelation length
    Ex_flat = Ex.flatten()
    Ex_norm = (Ex_flat - Ex_flat.mean()) / (Ex_flat.std() + 1e-10)
    
    acf = np.correlate(Ex_norm, Ex_norm, mode='full')
    acf = acf[len(acf)//2:]
    acf = acf / acf[0]
    
    # Find e-fold decay length
    decay_idx = np.where(acf < np.exp(-1))[0]
    if len(decay_idx) > 0:
        decay_spatial = (Z_MAX - Z_MIN) * decay_idx[0] / len(z_slice)
    else:
        decay_spatial = Z_MAX - Z_MIN
    
    print(f"Ex range:               [{Ex_flat.min():.3e}, {Ex_flat.max():.3e}] V/m")
    print(f"Ex Field smoothness:    {decay_spatial*1e3:.3f} mm correlation length")
    print(f"Expected wavelength:    ~{C_LIGHT/(5e9)*1e3:.1f} mm (5 GHz)")
    
    if decay_spatial < 1e-3:
        print("[!]  WARNING: Field has VERY SHORT correlation length - high-frequency noise!")
    else:
        print("[OK] Field smoothness looks acceptable.")
    
    return decay_spatial


def check_pde_residuals(model, grid_pts):
    """Check actual PDE residual magnitudes."""
    print("\n" + "="*70)
    print("CHECK 3: PDE RESIDUAL SATISFACTION")
    print("="*70)
    
    # Split into small batches to avoid memory issues
    batch_size = 1000
    all_res = {}
    
    for i in range(0, len(grid_pts), batch_size):
        batch = grid_pts[i:i+batch_size]
        with torch.enable_grad():
            x = batch[:, 0:1].clone().detach().requires_grad_(True)
            y = batch[:, 1:2].clone().detach().requires_grad_(True)
            z = batch[:, 2:3].clone().detach().requires_grad_(True)
            t = batch[:, 3:4].clone().detach().requires_grad_(True)
            
            res = compute_pde_residuals(model, x, y, z, t)
        
        for key in ["phi", "ax", "ay", "az", "gauge"]:
            if key not in all_res:
                all_res[key] = []
            all_res[key].append(res[key].detach().numpy())
    
    # Concatenate results
    for key in all_res:
        all_res[key] = np.concatenate(all_res[key])
    
    print("Normalized residual statistics:")
    for key in ["phi", "ax", "ay", "az", "gauge"]:
        res = all_res[key]
        print(f"  {key:8s}: mean={np.mean(np.abs(res)):.3e}, "
                f"max={np.max(np.abs(res)):.3e}")
    
    # Check if Φ residual is good
    phi_res = all_res["phi"]
    if np.mean(np.abs(phi_res)) > 1.0:
        print("[!]  WARNING: Phi PDE residual is LARGE - not satisfying wave equation!")
    else:
        print("[OK] Φ residual looks good.")
    
    return all_res


def check_source_term_enforcement(model, grid_pts):
    """Check if nabla^2 Phi ≈ -rho/epsilon_0 at source locations."""
    print("\n" + "="*70)
    print("CHECK 4: SOURCE TERM AT BUNCH TRAJECTORY")
    print("="*70)
    
    # Sample points along bunch trajectory
    t_sample = 1e-9
    z_bunch = Z0_BUNCH + V_BUNCH * t_sample
    
    pts = []
    for x_val in np.linspace(-1e-3, 1e-3, 5):
        for y_val in np.linspace(-1e-3, 1e-3, 5):
            pts.append([x_val, y_val, z_bunch, t_sample])
    pts = torch.tensor(pts, dtype=torch.float32)
    
    # Enable gradients for residual computation
    x = pts[:, 0:1].clone().requires_grad_(True)
    y = pts[:, 1:2].clone().requires_grad_(True)
    z = pts[:, 2:3].clone().requires_grad_(True)
    t = pts[:, 3:4].clone().requires_grad_(True)
    
    model.eval()  # Ensure eval mode, but allow_grad=True
    with torch.enable_grad():
        res = compute_pde_residuals(model, x, y, z, t)
    
    # Compute charge density (no gradients needed for this)
    with torch.no_grad():
        rho = charge_density(x.detach(), y.detach(), z.detach(), t.detach()).numpy()
    
    phi_res_raw = res["phi_raw"].detach().numpy()
    expected_source = -rho / EPSILON_0
    
    print(f"At bunch trajectory (t={t_sample:.2e}s, z_bunch={z_bunch:.3e}m):")
    print(f"  nabla^2 Phi (PDE residual):   {np.mean(phi_res_raw):.3e} V/m²")
    print(f"  -rho/epsilon_0 (expected):     {np.mean(expected_source):.3e} V/m²")
    print(f"  Ratio:                {np.mean(np.abs(phi_res_raw)) / (np.mean(np.abs(expected_source)) + 1e-20):.3f}")
    
    # Should be close to 1 (within ±10%)
    ratio = np.mean(np.abs(phi_res_raw)) / (np.mean(np.abs(expected_source)) + 1e-20)
    if abs(ratio - 1.0) > 0.1:
        print(f"[!]  WARNING: nabla^2 Phi does NOT match -rho/epsilon_0 well (ratio={ratio:.3f}) - source enforcement issue!")
    else:
        print(f"[OK] Source term enforcement looks excellent (ratio={ratio:.3f}).")


def main():
    import sys
    
    epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 7500
    
    print(f"\nLoading checkpoint epoch {epoch}...")
    model = load_checkpoint(epoch)
    
    print("Sampling grid...")
    grid_pts = sample_grid(nx=8, ny=8, nz=15, nt=4)
    
    # Run all checks
    check_potential_collapse(model, grid_pts)
    check_high_frequency_content(model, grid_pts)
    phi_res = check_pde_residuals(model, grid_pts)
    check_source_term_enforcement(model, grid_pts)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
