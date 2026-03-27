import torch
import matplotlib.pyplot as plt
import numpy as np
from network import WAVENetwork
from pinn_solver import get_gradient
from physics_source import t0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_and_visualize(model_path):
    print("Loading model for inference...")

    model = WAVENetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create structured test grid
    # 2D slice: x = 2, t = 2ns. Vary y and z
    b = 0.025
    z_min, z_max = -1.0, 1.0
    t_test = 3.33e-9 / t0

    # Create linear spaces
    y_vals = np.linspace(-b, b, 100)
    z_vals = np.linspace(z_min, z_max, 400)

    Y, Z = np.meshgrid(y_vals, z_vals)

    # Flatten grid into column vectors for the NN
    Y_flat = Y.flatten()[:, None]
    Z_flat = Z.flatten()[:, None]

    # Create the constant X and T vectors
    X_flat = np.zeros_like(Y_flat)
    T_flat = np.full_like(Y_flat, t_test)

    # Convert to PyTorch tensors
    # REQUIRES gradients as we need to calculate electric field
    x_test = torch.tensor(X_flat, dtype=torch.float32, device=device, requires_grad=True)
    y_test = torch.tensor(Y_flat, dtype=torch.float32, device=device, requires_grad=True)
    z_test = torch.tensor(Z_flat, dtype=torch.float32, device=device, requires_grad=True)
    t_test = torch.tensor(T_flat, dtype=torch.float32, device=device, requires_grad=True)

    # Model inference
    Phi, Ax, Ay, Az = model(x_test, y_test, z_test, t_test)

    # Reconstruct Physics
    dPhi_dz = get_gradient(Phi, z_test)
    dAz_dt = get_gradient(Az, t_test)

    Ez = -dPhi_dz - dAz_dt

    # Visualization
    # Detach from PyTorch graph and shape back into 2D grid dimensions
    Ez_grid = Ez.detach().cpu().numpy().reshape(Z.shape)

    plt.figure(figsize=(12, 4))

    # Create contour
    contour = plt.contourf(Z, Y, Ez_grid, levels=50, cmap='RdBu_r')
    plt.colorbar(contour, label='Longitudinal Electric Field Ez (V/m)')

    plt.title("PINN Predicted Wakefield Snapshot at t = 3.33 ns\n(Slice at x=0)")
    plt.xlabel("Logitudinal Position z (m)")
    plt.ylabel("Transverse Position y (m)")

    # Draw lines to represent pipe walls
    # plt.axhline(y=b, color='black', linewidth=3, label='PEC Wall')
    # plt.axhline(y=-b, color='black', linewidth=3)
    b_step = 0.015
    step_z_start = -0.1
    step_z_end = 0.1

    plt.plot(
        [z_min, step_z_start, step_z_start, step_z_end, step_z_end, z_max],
        [b, b, b_step, b_step, b, b],
        color='black', linewidth=3, label='PEC Wall'
    )
    
    plt.plot(
        [z_min, step_z_start, step_z_start, step_z_end, step_z_end, z_max],
        [-b, -b, -b_step, -b_step, -b, -b],
        color='black', linewidth=3
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig("wakefield_output.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    test_and_visualize("WAVEAI.pth")