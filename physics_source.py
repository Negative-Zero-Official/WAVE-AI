import torch
from scipy import constants

# Physical Constants (SI Units)
c = constants.c                     # Speed of Light (m/s)
epsilon_0 = constants.epsilon_0     # Vacuum Permittivity (F/m)
mu_0 = 1 / (epsilon_0 * c ** 2)     # Vacuum Permeability (H/m)
q_charge = constants.e              # Elementary Charge (C)
N_particles = 1e11                  # Chosen as it represents the Nominal Intensity for a proton bunch in the LHC

# Beam parameters
sigma_x = 0.001
sigma_y = 0.001
sigma_z = 0.05
v = c # Beam velocity approximated as speed of light

def rho_func(x, y, z, t):
    """
    Computes the 3D charge desnity of the moving particle bunch
    Models a Gaussian distribution traveling along the z-axis

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    total_charge = N_particles * q_charge
    norm_factor = total_charge / ((2 * constants.pi)**(3/2) * sigma_x * sigma_y * sigma_z)
    
    # The bunch moves: its center is at z = v*t
    z_center = v * t

    exp_term = torch.exp(
        -(x**2) / (2 * sigma_x**2)
        -(y**2) / (2 * sigma_y**2)
        -((z - z_center)**2) / (2 * sigma_z**2)
    )

    return norm_factor * exp_term

def J_func(x, y, z, t):
    """
    Computes the 3D current density vector (Jx, Jy, Jz).
    Since the beam moves strictly in the z-direction, Jx = Jy = 0

    Args:
        x (_type_): _description_
        y (_type_): _description_
        z (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    rho = rho_func(x, y, z, t)
    Jx = torch.zeros_like(rho)
    Jy = torch.zeros_like(rho)
    Jz = v * rho
    
    return Jx, Jy, Jz