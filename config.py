import math
import torch
import scipy.constants as constants

"""
Physical constants (SI)
"""
C_LIGHT = constants.c               # Speed of light (m/s)
EPSILON_0 = constants.epsilon_0     # Permittivity of vacuum (F/m)
MU_0 = 4.0 * math.pi * 1e-7         # Permeability of vacuum (H/m)

# Derived
INV_C2 = 1.0 / C_LIGHT**2           # 1/c^2 (s^2/m^2)


"""
Relativistic proton-bunch parameters
"""
BETA = 0.999                            # v / c
V_BUNCH = BETA * C_LIGHT                # Bunch velocity (m/s)
GAMMA = 1.0 / math.sqrt(1 - BETA**2)    # Lorentz factor (approx. 22.4)
Q_TOTAL = 1.0e-9                        # Total charge (1 nC)

# Traverse (lab-frame) and longitudinal RMS sizes
SIGMA_X = 1.0e-3    # m
SIGMA_Y = 1.0e-3    # m
SIGMA_Z = 5.0e-3    # m (5 mm) - lab-frame bunch length

# Initial position
Z0_BUNCH = -0.20    # m


"""
Waveguide / Step-Collimator Geometry
"""
# The rectangular waveguide has a sudden step at z = Z_STEP:
#   z < Z_STEP  ->  full aperture  +-A_PIPE  (x)  ×  +-B_PIPE  (y)
#   z >= Z_STEP  ->  reduced aperture +-A_STEP (x)  ×  +-B_STEP  (y)
#
# PEC walls are enforced on ALL surfaces (outer walls + step shoulder).

A_PIPE = 0.040      # Half-width of main pipe (m)
B_PIPE = 0.040      # Half-height of main pipe (m)
A_STEP = 0.020      # Half-width of small pipe (m)
B_STEP = 0.020      # Half-height of small pipe (m)
Z_STEP = 0.000      # z-coordinate of step face (m)


"""
SIMULATION DOMAIN (x, y, z, t)
"""
X_MIN, X_MAX = -A_PIPE, A_PIPE
Y_MIN, Y_MAX = -B_PIPE, B_PIPE
Z_MIN, Z_MAX = -0.30, 0.30
T_MIN, T_MAX = 0.0, 2.0e-9

# Domain half-extents (for normalization)
X_HALF = (X_MAX - X_MIN) / 2.0
Y_HALF = (Y_MAX - Y_MIN) / 2.0
Z_HALF = (Z_MAX - Z_MIN) / 2.0
T_HALF = (T_MAX - T_MIN) / 2.0

X_MID = (X_MAX + X_MIN) / 2.0
Y_MID = (Y_MAX + Y_MIN) / 2.0
Z_MID = (Z_MAX + Z_MIN) / 2.0
T_MID = (T_MAX + T_MIN) / 2.0


"""
Reference (normalization) scales
Used to make PDE residuals O(1)
"""
RHO_MAX = Q_TOTAL / ((2.0 * math.pi)**1.5 * SIGMA_X * SIGMA_Y * SIGMA_Z)
J_MAX = RHO_MAX * V_BUNCH

RES_PHI_SCALE = RHO_MAX / EPSILON_0
RES_A_SCALE = MU_0 * J_MAX

L_REF = A_PIPE
RES_GAUGE_SCALE = max(RES_A_SCALE * L_REF, 1e-20)

PHI_REF = Q_TOTAL / (4.0 * math.pi * EPSILON_0 * L_REF)
A_REF = PHI_REF / C_LIGHT
E_REF = PHI_REF / L_REF
B_REF = E_REF / C_LIGHT


"""
WAVENetwork Architecture
"""
HIDDEN_SIZE = 128
NUM_LAYERS = 4
OMEGA_0 = 10.0


"""
Collocation / boundary / IC sampling counts
"""

N_PDE = 40_000
N_IMPORTANCE = 10_000
N_BC = 8_000
N_IC = 4_000

# Reduced point sets for L-BFGS phase (faster convergence)
N_LBFGS_PDE = 4_000
N_LBFGS_IMPORTANCE = 2_000
N_LBFGS_BC = 4_000
N_LBFGS_IC = 2_000

# Batch sizes used each training iteration
BATCH_PDE = 8_000
BATCH_BC = 2_000
BATCH_IC = 1_000


"""
Training Hyper-Parameters
"""
LR_ADAM = 1e-4
N_EPOCHS_ADAM = 8_000       # Adam phase (increased for better warm-start)

LR_LBFGS = 0.01             # Reduced further from 0.1 to improve L-BFGS stability
N_EPOCHS_LBFGS = 200        # Fewer epochs since each is more expensive
LBFGS_MAX_ITER = 30         # More L-BFGS iterations per epoch
LBFGS_HISTORY = 50


"""
LOSS WEIGHTS - CRITICAL FIX
"""
LAMBDA_PDE = 10.0
LAMBDA_BC = 1.5
LAMBDA_IC = 5.0
LAMBDA_GAUGE = 10.0

LAMBDA_PHI_OVERRIDE = 0.5   # Moderate boost to Φ (cannot remove; A depends on it)
LAMBDA_REG = 1e-3           # Tikhonov regularization to suppress oscillations
LAMBDA_SMOOTH = 0.0         # Smoothness penalty (||∇²u||²) (disabled for now)
OMEGA_0_PHASE2 = 20.0       # Reserve for Phase 3 if needed


"""
Spectral Filtering (Phase 2)
"""
SPECTRAL_FILTER_ENABLED = False  # Keep disabled; ω₀=30 provides sufficient smoothness
FILTER_TYPE = "lowpass"        # "lowpass" or "bandpass"
FILTER_CUTOFF_FREQ = 15e9      # Reserve for Phase 3: 15 GHz cutoff (wavelength ~2cm)


"""
Data Sampling Adjustments (Phase 4)
"""
N_IMPORTANCE_INCREASED = 20_000  # Double importance sampling for better source enforcement


"""
Paths & Logging
"""
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"

LOG_INTERVAL = 100
SAVE_INTERVAL = 500


"""
Device
"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"