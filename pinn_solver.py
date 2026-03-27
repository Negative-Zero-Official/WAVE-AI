import torch
from torch import optim
from scipy.stats import qmc
from tqdm import tqdm
import matplotlib.pyplot as plt
from physics_source import c, mu_0, epsilon_0, t0, rho_func, J_func
from network import WAVENetwork

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Gradient Utilities
"""
def get_gradient(output, input_var):
    return torch.autograd.grad(
        output, input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True
    )[0]

def d_alembertian(u, x, y, z, t):
    du_dx = get_gradient(u, x)
    du_dy = get_gradient(u, y)
    du_dz = get_gradient(u, z)
    du_dt = get_gradient(u, t)

    d2u_dx2 = get_gradient(du_dx, x)
    d2u_dy2 = get_gradient(du_dy, y)
    d2u_dz2 = get_gradient(du_dz, z)
    d2u_dt2 = get_gradient(du_dt, t)

    laplacian = d2u_dx2 + d2u_dy2 + d2u_dz2

    return laplacian - (1.0 / c**2) * d2u_dt2

"""
Loss Computation
"""
def compute_losses(model, points_domain, points_bc_x, points_bc_y, points_bc_step, points_ic):
    # CONTINUOUS DOMAIN (PDE) LOSSES
    x, y, z, t = [p.requires_grad_(True) for p in points_domain]

    Phi, Ax, Ay, Az = model(x, y, z, t)
    Jx, Jy, Jz = J_func(x, y, z, t)
    rho = rho_func(x, y, z, t)

    # D'Alembertian wave equations (PDE residuals)
    res_Phi = d_alembertian(Phi, x, y, z, t) + (rho / epsilon_0)
    res_Ax = d_alembertian(Ax, x, y, z, t) + (mu_0 * Jx)
    res_Ay = d_alembertian(Ay, x, y, z, t) + (mu_0 * Jy)
    res_Az = d_alembertian(Az, x, y, z, t) + (mu_0 * Jz)

    # Lorenz Gauge condition: div(A) + (1/c^2) * dPhi/dt = 0
    div_A = get_gradient(Ax, x) + get_gradient(Ay, y) + get_gradient(Az, z)
    dPhi_dt = get_gradient(Phi, t)
    res_gauge = div_A + (1.0 / c**2) * dPhi_dt

    loss_pde = torch.mean(res_Phi**2 + res_Ax**2 + res_Ay**2 + res_Az**2 + res_gauge**2)

    # BOUNDARY CONDITION LOSSES (PEC Walls)
    # To enforce E_tangential = 0, we need to compute the E field E = -grad(Phi) - dA/dt

    # Left/Right walls (x = -a and x = a)
    xb, yb, zb, tb = [p.requires_grad_(True) for p in points_bc_x]
    Phi_xb, Ax_xb, Ay_xb, Az_xb = model(xb, yb, zb, tb)

    Ey_xb = -get_gradient(Phi_xb, yb) - get_gradient(Ay_xb, tb)
    Ez_xb = -get_gradient(Phi_xb, zb) - get_gradient(Az_xb, tb)

    loss_bc_x = torch.mean(Ey_xb**2 + Ez_xb**2) # Tangential components must be 0

    # Top/Bottom walls (y = -b and y = b)
    xc, yc, zc, tc = [p.requires_grad_(True) for p in points_bc_y]
    Phi_yc, Ax_yc, Ay_yc, Az_yc = model(xc, yc, zc, tc)

    Ex_yc = -get_gradient(Phi_yc, xc) - get_gradient(Ax_yc, tc)
    Ez_yc = -get_gradient(Phi_yc, zc) - get_gradient(Az_yc, tc)
    
    loss_bc_y = torch.mean(Ex_yc**2 + Ez_yc**2) # Tangential components must be 0

    loss_bc = loss_bc_x + loss_bc_y

    # GEOMETRIC STEP LOSS (Collimator)
    xs, ys, zs, ts = [p.requires_grad_(True) for p in points_bc_step]
    Phi_s, Ax_s, Ay_s, Az_s = model(xs, ys, zs, ts)

    # On vertical Z-walls, the tangential fields are Ex and Ey
    Ex_s = -get_gradient(Phi_s, xs) - get_gradient(Ax_s, ts)
    Ey_s = -get_gradient(Phi_s, ys) - get_gradient(Ay_s, ts)

    loss_bc_step = torch.mean(Ex_s**2 + Ey_s**2)
    loss_bc += loss_bc_step

    # INITIAL CONDITIONS (t=0)
    x0, y0, z0, t0 = [p.requires_grad_(True) for p in points_ic]
    Phi0, Ax0, Ay0, Az0 = model(x0, y0, z0, t0)

    loss_ic = torch.mean(Phi0**2 + Ax0**2 + Ay0**2 + Az0**2)

    return loss_pde, loss_bc, loss_ic

"""
Data Generation
"""
def generate_lhs_samples(num_samples, bounds):
    """
    Generates Latin Hypercube Sampling (LHS) samples mapped to specific physical bounds.

    Args:
        num_samples (int): The number of points to generate
        bounds (list of tuples): [(min_x, max_x), (min_y, max_y), (min_z, max_z), (min_t, max_t)]

    Returns:
        list of PyTorch tensors: 4 PyTorch tensors representing x, y, z, and t coordinates
    """
    dimensions = len(bounds)

    # Initialize the LHS sampler for 4 dimensions
    sampler = qmc.LatinHypercube(d=dimensions)

    # Generate points in the range [0, 1]
    sample = sampler.random(n=num_samples)

    # Store the scaled coordinates in a list
    scaled_tensors = []

    # Scale the [0, 1] samples to actual physical dimensions
    for i in range(dimensions):
        lower_bound, upper_bound = bounds[i]

        # Scale: scaled_value = lower + sample * (upper - lower)
        scaled_column = lower_bound + sample[:, i] * (upper_bound - lower_bound)

        # Convert to PyTorch tensor, reshape to (N, 1) and require gradients for Autograd
        tensor_col = torch.tensor(scaled_column, dtype=torch.float32, device=device).view(-1, 1)
        # tensor_col.requires_grad_(True)

        scaled_tensors.append(tensor_col)
    
    return scaled_tensors # Returns [x_tensor, y_tensor, z_tensor, t_tensor]

"""
Model Training
"""
def train():
    model = WAVENetwork().to(device)

    # Define the physical dimensions of the simulation
    # For demonstration purposes, consider a rectangular pipe that is 10cm wide and 5cm tall
    a = 0.05 # x goes from -0.05m to 0.05m
    b = 0.025 # y goes from -0.025m to 0.025m

    # The length of the pipe being simulated (taken 2m)
    z_min, z_max = -5.0, 5.0
    
    # Time window of simulation (taken 10 nanoseconds)
    t_min, t_max = 0.0, 1e-8 / t0

    # Set normalization in the model
    model.set_normalization(-a, a, -b, b, z_min, z_max, t_min, t_max)

    # GENERATE DATA POINTS
    # Domain points (interior)
    domain_bounds = [(-a, a), (-b, b), (z_min, z_max), (t_min, t_max)]

    # Generate 10,000 distributed LHS points inside the 4D volume
    print("Generating LHS Domain points...")
    domain_pts = generate_lhs_samples(10000, domain_bounds)

    # For boundary conditions, lock one dimension and use LHS for the remaining 3
    # e.g., for the left/right walls, x is fixed at -a or +a, so we only need to LHS sample y, z, and t
    print("Generating LHS Boundary Points...")

    # Y-Z-T bounds for the X-walls
    bc_x_bounds = [(-b, b), (z_min, z_max), (t_min, t_max)]
    bc_x_samples = generate_lhs_samples(2000, bc_x_bounds)

    # Construct the final points for the x=-a wall
    x_wall_left = torch.full((2000, 1), -a, dtype=torch.float32, device=device, requires_grad=True)
    bc_left_wall_pts = [x_wall_left, bc_x_samples[0], bc_x_samples[1], bc_x_samples[2]]

    # Repeat this logic for the right, top, and bottom walls

    x_wall_right = torch.full((2000, 1), a, dtype=torch.float32, device=device, requires_grad=True)
    bc_right_wall_pts = [x_wall_right, bc_x_samples[0], bc_x_samples[1], bc_x_samples[2]]

    # Combine left an right walls into a single list of 4 tensors [x, y, z, t]
    points_bc_x = [torch.cat([left, right], dim=0) for left, right in zip(bc_left_wall_pts, bc_right_wall_pts)]

    # X-Z-T bounds for the Y-walls
    step_z_start = -0.1
    step_z_end = 0.1
    b_step = 0.015 # Pipe pinches from 0.025 to 0.015

    bc_y_bounds = [(-a, a), (z_min, z_max), (t_min, t_max)]
    bc_y_samples = generate_lhs_samples(2000, bc_y_bounds)

    z_vals = bc_y_samples[1]

    y_wall_bottom_vals = torch.full((2000, 1), -b, dtype=torch.float32, device=device)
    y_wall_bottom_vals = torch.where(
        (z_vals > step_z_start) & (z_vals < step_z_end),
        torch.tensor(-b_step, dtype=torch.float32, device=device),
        y_wall_bottom_vals
    )
    y_wall_bottom = y_wall_bottom_vals.clone().detach().requires_grad_(True)
    bc_bottom_wall_pts = [bc_y_samples[0], y_wall_bottom, bc_y_samples[1], bc_y_samples[2]]

    y_wall_top_vals = torch.full((2000, 1), b, dtype=torch.float32, device=device)
    y_wall_top_vals = torch.where(
        (z_vals > step_z_start) & (z_vals < step_z_end),
        torch.tensor(b_step, dtype=torch.float32, device=device),
        y_wall_top_vals
    )
    y_wall_top = y_wall_top_vals.clone().detach().requires_grad_(True)
    bc_top_wall_pts = [bc_y_samples[0], y_wall_top, bc_y_samples[1], bc_y_samples[2]]

    points_bc_y = [torch.cat([bottom, top], dim=0) for bottom, top in zip(bc_bottom_wall_pts, bc_top_wall_pts)]

    # Step points generation
    bc_step_bounds_top = [(-a, a), (b_step, b), (t_min, t_max)]
    bc_step_samples_top = generate_lhs_samples(1000, bc_step_bounds_top)

    bc_step_bounds_bottom = [(-a, a), (-b, -b_step), (t_min, t_max)]
    bc_step_samples_bottom = generate_lhs_samples(1000, bc_step_bounds_bottom)

    z_face_start = torch.full((1000, 1), step_z_start, dtype=torch.float32, device=device, requires_grad=True)
    z_face_end = torch.full((1000, 1), step_z_end, dtype=torch.float32, device=device, requires_grad=True)

    # Front Top Face
    f_start_top = [bc_step_samples_top[0], bc_step_samples_top[1], z_face_start, bc_step_samples_top[2]]
    f_end_top = [bc_step_samples_top[0], bc_step_samples_top[1], z_face_end, bc_step_samples_top[2]]

    f_start_bottom = [bc_step_samples_bottom[0], bc_step_samples_bottom[1], z_face_start, bc_step_samples_bottom[2]]
    f_end_bottom = [bc_step_samples_bottom[0], bc_step_samples_bottom[1], z_face_end, bc_step_samples_bottom[2]]

    points_bc_step = [
        torch.cat([f_start_top[i], f_end_top[i], f_start_bottom[i], f_end_bottom[i]], dim=0)
        for i in range(4)
    ]

    # Initial condition points (t=0)
    ic_bounds = [(-a, a), (-b, b), (z_min, z_max), (0.0, 0.0)]
    points_ic = generate_lhs_samples(5000, ic_bounds)

    # TRAINING LOOP
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    epochs = 10000
    lambda_bc = 100.0 # Prioritize walls
    lambda_ic = 100.0 # Initial conditions are crucial - give high weight

    # OLD CODE: TODO: Remove if deemed unnecessary
    # Weighting factor for the boundary conditions
    # In PINNs, it is notoriously hard to make the network respect the boundaries
    # Multiplying the BC loss by a scalar (like 10 or 100) forces the optimizer to prioritize it
    # lambda_bc = 1000.0

    losses_pde = []
    losses_bc = []
    losses_ic = []
    losses = []

    loop = tqdm(range(epochs), leave=True)
    for _ in loop:
        optimizer.zero_grad()

        # 1. Compute the losses
        loss_pde, loss_bc, loss_ic = compute_losses(model, domain_pts, points_bc_x, points_bc_y, points_bc_step, points_ic)

        # 2. Total loss equation
        total_loss = loss_pde + lambda_bc * loss_bc + lambda_ic * loss_ic

        # 3. Backpropagation
        total_loss.backward()

        # Record losses
        losses_pde.append(loss_pde.item())
        losses_bc.append(loss_bc.item())
        losses_ic.append(loss_ic.item())
        losses.append(total_loss.item())

        # 4. Update weights
        optimizer.step()

        # 5. Update learning rate
        scheduler.step(total_loss.item())

        loop.set_description(f"Training")
        loop.set_postfix(loss=total_loss.item())
    
    print(f"Final loss = {losses[-1]}")
    print("Training complete. Saving model and datasets...")

    torch.save(model.state_dict(), "WAVEAI.pth")

    torch.save({
        "domain_pts" : domain_pts,
        "points_bc_x" : points_bc_x,
        "points_bc_y" : points_bc_y,
        "points_ic" : points_ic
    }, "WAVEAI_training_data.pth")

    print("Artifacts saved successfully.")

    plt.figure(figsize=(10, 6))
    plt.semilogy(losses, label="Total Loss")
    plt.semilogy(losses_pde, label="PDE Loss")
    plt.semilogy(losses_bc, label="BC Loss")
    plt.semilogy(losses_ic, label="IC Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('loss-curve.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    train()