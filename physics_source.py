import torch
from scipy import constants

# TODO: Remove old code if deemed unnecessary
# # Physical Constants (SI Units)
# c = constants.c                     # Speed of Light (m/s)
# epsilon_0 = constants.epsilon_0     # Vacuum Permittivity (F/m)
# mu_0 = 1 / (epsilon_0 * c ** 2)     # Vacuum Permeability (H/m)
# q_charge = constants.e              # Elementary Charge (C)
# N_particles = 1e11                  # Chosen as it represents the Nominal Intensity for a proton bunch in the LHC

# 1. NON-DIMENSIONAL SCALES
c_si = constants.c          # Speed of light in SI units
L0 = 1.0                    # 1 unit of scaled distance = 1 meter
t0 = L0 / c_si              # 1 unit of scaled time = ~3.33 nanoseconds

# 2. PDE CONSTANTS (Forced to 1.0)
c = 1.0
epsilon_0 = 1.0
mu_0 = 1.0

# 3. BUNCH PARAMETERS
sigma_x = 0.001 / L0
sigma_y = 0.001 / L0
sigma_z = 0.05 / L0
v = c # Beam velocity approximated as speed of light

# Add initial offset so that the bunch starts outside the domain (z_min = -1.0)
z0 = -1.0 / L0

def rho_func(x, y, z, t):
    """
    Computes the 3D charge desnity of the moving particle bunch
    Models a Gaussian distribution traveling along the z-axis
    """
    # Bunch center moves with velocity v
    z_center = z0 + v * t

    exp_term = torch.exp(
        -(x**2) / (2 * sigma_x**2)
        -(y**2) / (2 * sigma_y**2)
        -((z - z_center)**2) / (2 * sigma_z**2)
    )

    return exp_term

def J_func(x, y, z, t):
    """
    Computes the 3D current density vector (Jx, Jy, Jz).
    Since the beam moves strictly in the z-direction, Jx = Jy = 0
    """
    rho = rho_func(x, y, z, t)
    Jx = torch.zeros_like(rho)
    Jy = torch.zeros_like(rho)
    Jz = v * rho
    
    return Jx, Jy, Jz