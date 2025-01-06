import torch
import numpy as np
import random
import h5py
from scipy.constants import *


def set_seed(seed):
    """
    Set seed for reproducibility across random number generators.

    Parameters:
        seed (int): The seed value to use for Python, NumPy, and PyTorch.
    """
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy random
    torch.manual_seed(seed)  # PyTorch CPU seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU seed
        torch.cuda.manual_seed_all(seed)  # Seed all GPUs

    # Ensure deterministic behavior in PyTorch (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_simulation(filename, device="cpu"):
    """
    Load simulation data from an HDF5 file.

    Parameters:
        filename (str): Name of the HDF5 file (without extension).
        device (str): Target device for the data ("cpu" or "cuda").

    Returns:
        dict: Dictionary containing datasets loaded from the file.
    """
    data_dict = {}

    with h5py.File(filename + ".h5", "r") as f:
        for key in f.keys():
            # Load all datasets into a dictionary
            arr = f[key][...]
            data_dict[key] = np.array(arr)

    print(f"[INFO] Loaded datasets from {filename}: {list(data_dict.keys())}")
    return data_dict


class Normalization:
    """
    Utility class to handle normalization and denormalization of physical quantities
    based on reference units for plasma simulations.
    """

    def __init__(self, 
                 omega_r,  # Plasma frequency scale (1/s)
                 length_ref=None,  # Reference length scale (m)
                 mass_e=m_e,  # Reference electron mass (kg)
                 charge_e=e):  # Reference electron charge (C)
        """
        Initialize normalization constants and calculate derived units.

        Parameters:
            omega_r (float): Plasma frequency scale in radians per second.
            length_ref (float, optional): Reference length scale in meters. Defaults to c/omega_r.
            mass_e (float): Reference electron mass in kilograms.
            charge_e (float): Reference electron charge in coulombs.
        """
        self.omega_r = omega_r
        self.mass_e = mass_e
        self.charge_e = charge_e

        # If length_ref is not provided, calculate it using the plasma frequency
        if length_ref is None:
            self.v_ref = c  # Speed of light (m/s)
            self.length_ref = self.v_ref / self.omega_r  # Reference length (m)
        else:
            self.length_ref = length_ref
            self.v_ref = self.length_ref * self.omega_r  # Reference velocity (m/s)

        self.t_ref = 1.0 / self.omega_r  # Reference time scale (s)

    # Conversion methods for length
    def to_nondim_length(self, x_dim):
        """Convert length from dimensional [m] to nondimensional form."""
        return x_dim / self.length_ref

    def to_dim_length(self, x_nondim):
        """Convert length from nondimensional to dimensional [m]."""
        return x_nondim * self.length_ref

    # Conversion methods for time
    def to_nondim_time(self, t_dim):
        """Convert time from dimensional [s] to nondimensional form."""
        return t_dim * self.omega_r

    def to_dim_time(self, t_nondim):
        """Convert time from nondimensional to dimensional [s]."""
        return t_nondim / self.omega_r

    # Conversion methods for velocity
    def to_nondim_velocity(self, v_dim):
        """Convert velocity from dimensional [m/s] to nondimensional form."""
        return v_dim / self.v_ref

    def to_dim_velocity(self, v_nondim):
        """Convert velocity from nondimensional to dimensional [m/s]."""
        return v_nondim * self.v_ref

    # Conversion methods for mass
    def to_nondim_mass(self, m_dim):
        """Convert mass from dimensional [kg] to nondimensional form."""
        return m_dim / self.mass_e

    def to_dim_mass(self, m_nondim):
        """Convert mass from nondimensional to dimensional [kg]."""
        return m_nondim * self.mass_e

    # Conversion methods for charge
    def to_nondim_charge(self, q_dim):
        """Convert charge from dimensional [C] to nondimensional form."""
        return q_dim / self.charge_e

    def to_dim_charge(self, q_nondim):
        """Convert charge from nondimensional to dimensional [C]."""
        return q_nondim * self.charge_e

    # Conversion methods for density
    def to_nondim_density(self, n_dim):
        """Convert density from dimensional [m^-3] to nondimensional form."""
        return n_dim / self.n0

    def to_dim_density(self, n_nondim):
        """Convert density from nondimensional to dimensional [m^-3]."""
        return n_nondim * self.n0
