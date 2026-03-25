import torch
from torch import optim
from scipy.stats import qmc
from tqdm import tqdm
import matplotlib.pyplot as plt
from physics_source import c, mu_0, epsilon_0, rho_func, J_func
from network import WAVENetwork

torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gradient(output, input_var):
    return torch.autograd.grad(
        output, input_var,
        grad_outputs=torch.ones_like(output),
        create_graph=True, retain_graph=True
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

def compute_losses(model, points_domain, points_bc_x, points_bc_y):
    # CONTINUOUS DOMAIN (PDE) LOSSES
    x, y, z, t = [p.requires_grad_(True) for p in points_domain]

    Phi, Ax, Ay, Az = model(x, y, z, t)
    Jx, Jy, Jz = J_func(x, y, z, t)
    rho = rho_func(x, y, z, t)

    # D'Alembertian wave equations
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

    return loss_pde, loss_bc

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

def train():
    model = WAVENetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Define the physical dimensions of the simulation
    # For demonstration purposes, consider a rectangular pipe that is 10cm wide and 5cm tall
    a = 0.05 # x goes from -0.05m to 0.05m
    b = 0.025 # y goes from -0.025m to 0.025m

    # The length of the pipe being simulated (taken 2m)
    z_min, z_max = -1.0, 1.0
    
    # Time window of simulation (taken 10 nanoseconds)
    t_min, t_max = 0.0, 1e-8

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
    bc_y_bounds = [(-a, a), (z_min, z_max), (t_min, t_max)]
    bc_y_samples = generate_lhs_samples(2000, bc_y_bounds)

    y_wall_bottom = torch.full((2000, 1), -b, dtype=torch.float32, device=device, requires_grad=True)
    bc_bottom_wall_pts = [bc_y_samples[0], y_wall_bottom, bc_y_samples[1], bc_y_samples[2]]

    y_wall_top = torch.full((2000, 1), b, dtype=torch.float32, device=device, requires_grad=True)
    bc_top_wall_pts = [bc_y_samples[0], y_wall_top, bc_y_samples[1], bc_y_samples[2]]

    points_bc_y = [torch.cat([bottom, top], dim=0) for bottom, top in zip(bc_bottom_wall_pts, bc_top_wall_pts)]

    # TRAINING LOOP
    epochs = 1000

    # Weighting factor for the boundary conditions
    # In PINNs, it is notoriously hard to make the network respect the boundaries
    # Multiplying the BC loss by a scalar (like 10 or 100) forces the optimizer to prioritize it
    lambda_bc = 1000.0

    losses = []

    loop = tqdm(range(epochs), leave=True)
    for epoch in loop:
        optimizer.zero_grad()

        # 1. Compute the losses
        loss_pde, loss_bc = compute_losses(model, domain_pts, points_bc_x, points_bc_y)

        # 2. Total loss equation
        total_loss = loss_pde + lambda_bc * loss_bc

        # 3. Backpropagation
        total_loss.backward()
        losses.append(total_loss.item())

        # 4. Update weights
        optimizer.step()

        loop.set_description(f"Epoch: {epoch+1}")
        loop.set_postfix(loss=total_loss.item())
    
    print(f"Final loss = {total_loss.item()}")
    print("Training complete. Saving model and datasets...")

    torch.save(model.state_dict(), "WAVEAI.pth")

    torch.save({
        "domain_pts" : domain_pts,
        "points_bc_x" : points_bc_x,
        "points_bc_y" : points_bc_y
    }, "WAVEAI_training_data.pth")

    print("Artifacts saved successfully.")

    plt.plot(losses)
    plt.show()
    plt.savefig('loss-curve.png', dpi=300)

if __name__ == "__main__":
    if (int(input("Begin process? (0/1): "))):
        train()
    else:
        print("Exitting program...")