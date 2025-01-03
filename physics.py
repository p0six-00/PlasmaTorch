import torch
from scipy.constants import *
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(1234)

def maxwell_boltzmann_distribution(num_particles, temperature, mass, device=device):
    """
    Generates a Maxwell-Boltzmann velocity distribution for a set of particles.

    Parameters:
        num_particles (int): Number of particles to sample velocities for.
        temperature (float): Temperature of the system in Kelvin.
        mass (float): Mass of each particle in kilograms.
        device (str): Target device for the computation ("cpu" or "cuda").

    Returns:
        torch.Tensor: A tensor of shape (num_particles, 3) representing the 
                      velocities (vx, vy, vz) of the particles in m/s.
    """
    # Boltzmann constant in Joules per Kelvin (SI units)
    kB = 1.380649e-23  # [J/K]

    # Standard deviation of the velocity distribution (1D Gaussian)
    std = np.sqrt(kB * temperature / mass)

    # Sample velocities from a normal distribution with mean 0 and calculated std
    vx = torch.normal(0, std, size=(num_particles,), device=device)
    vy = torch.normal(0, std, size=(num_particles,), device=device)
    vz = torch.zeros((num_particles,), device=device)  # No velocity in z-direction by default

    # Stack velocities into a (num_particles, 3) tensor
    return torch.stack([vx, vy, vz], dim=1)

def check_CFL_condition(dt, dx):
    """
    Checks whether the Courant-Friedrichs-Lewy (CFL) condition is satisfied.

    The CFL condition ensures numerical stability in explicit time integration
    schemes, especially for wave propagation problems.

    Parameters:
        dt (float): Time step size (s).
        dx (float): Spatial step size (m).

    Returns:
        tuple: 
            - bool: True if the CFL condition is satisfied, False otherwise.
            - float: The computed CFL number.
    """
    # The maximum velocity in the system, assumed to be the speed of light (m/s)
    max_velocity = c  # Speed of light [m/s]

    # Compute the CFL number: CFL = max_velocity * dt / dx
    cfl_number = max_velocity * dt / dx

    # Return whether the CFL condition is satisfied and the computed CFL number
    return cfl_number <= 1, cfl_number
