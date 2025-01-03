import torch
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(1234)
    

def place_random(num_particles, Lx, Ly, device=device):
    """
    Places particles randomly within the simulation domain.

    Parameters:
        num_particles (int): Number of particles to place.
        Lx, Ly (float): Dimensions of the simulation domain.
        device (str): Target device ("cpu" or "cuda").

    Returns:
        torch.Tensor: Tensor of shape (num_particles, 3) with random positions (x, y, z).
    """
    positions = torch.rand((num_particles, 3), device=device) * torch.tensor([Lx, Ly, 0], device=device)
    return positions


def place_uniform(num_particles, Nx, Ny, Lx, Ly, device=device):
    """
    Places particles uniformly on a grid within the simulation domain.

    Parameters:
        num_particles (int): Number of particles to place.
        Nx, Ny (int): Number of grid points along x and y directions.
        Lx, Ly (float): Dimensions of the simulation domain.
        device (str): Target device ("cpu" or "cuda").

    Returns:
        torch.Tensor: Tensor of shape (num_particles, 3) with uniform positions (x, y, z).
    """
    if Nx * Ny < num_particles:
        raise ValueError(f"Cannot place {num_particles} particles with Nx*Ny={Nx*Ny} < {num_particles}.")

    # Generate uniform grid points
    x_vals = torch.linspace(0, Lx, Nx, device=device)
    y_vals = torch.linspace(0, Ly, Ny, device=device)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')

    # Flatten and add a z-coordinate
    all_positions = torch.stack([X.flatten(), Y.flatten(), torch.zeros_like(X.flatten())], dim=-1)

    # Select the first 'num_particles' positions
    positions = all_positions[:num_particles, :]
    return positions


def nearest_grid_point_deposition(space, quantity, positions, Nx, Ny, dx, dy):
    """
    Deposits particle quantities onto the nearest grid points.

    Parameters:
        space: Simulation space object.
        quantity (torch.Tensor): Quantity to deposit (e.g., charge).
        positions (torch.Tensor): Particle positions.
        Nx, Ny (int): Grid dimensions.
        dx, dy (float): Grid spacing in x and y directions.

    Returns:
        torch.Tensor: Grid with deposited quantities.
    """
    grid = torch.zeros((Nx, Ny), device=positions.device)

    # Find grid indices for each particle
    ix = (positions[:, 0] / dx).long()
    iy = (positions[:, 1] / dy).long()

    # Flatten the grid and perform parallel deposition
    flat_grid = grid.flatten()
    flat_grid = torch.index_add(flat_grid, 0, ix * Ny + iy, quantity)

    # Reshape the grid back to 2D
    grid = flat_grid.view(Nx, Ny)
    grid /= dx * dy  # Normalize the grid values
    return grid


def cloud_in_cell_deposition(space, quantity, positions, Nx, Ny, dx, dy, device=device):
    """
    Deposits particle quantities onto the grid using the Cloud-in-Cell method.

    Parameters:
        space: Simulation space object.
        quantity (torch.Tensor): Quantity to deposit (e.g., charge).
        positions (torch.Tensor): Particle positions.
        Nx, Ny (int): Grid dimensions.
        dx, dy (float): Grid spacing in x and y directions.
        device (str): Target device ("cpu" or "cuda").

    Returns:
        torch.Tensor: Grid with deposited quantities.
    """
    grid = torch.zeros((Nx, Ny), device=device)

    # Calculate indices of the lower-left corner of the cell containing each particle
    ix = torch.floor(positions[:, 0] / dx).long()
    iy = torch.floor(positions[:, 1] / dy).long()

    # Clamp indices to grid boundaries
    ix = torch.clamp(ix, 0, Nx - 1)
    iy = torch.clamp(iy, 0, Ny - 1)

    # Calculate relative distances within the cell
    wx = (positions[:, 0] / dx) - ix
    wy = (positions[:, 1] / dy) - iy

    # Compute weighted contributions to surrounding cells
    contributions = {
        'Q11': quantity * (1 - wx) * (1 - wy),
        'Q12': quantity * (1 - wx) * wy,
        'Q21': quantity * wx * (1 - wy),
        'Q22': quantity * wx * wy,
    }

    # Calculate indices of the surrounding cells
    ixp1 = torch.clamp(ix + 1, 0, Nx - 1)
    iyp1 = torch.clamp(iy + 1, 0, Ny - 1)

    idx_Q11 = (ix * Ny + iy).long()
    idx_Q12 = (ix * Ny + iyp1).long()
    idx_Q21 = (ixp1 * Ny + iy).long()
    idx_Q22 = (ixp1 * Ny + iyp1).long()

    # Deposit contributions onto the grid
    flat_grid = grid.flatten()
    flat_grid = torch.index_add(flat_grid, 0, idx_Q11, contributions['Q11'])
    flat_grid = torch.index_add(flat_grid, 0, idx_Q12, contributions['Q12'])
    flat_grid = torch.index_add(flat_grid, 0, idx_Q21, contributions['Q21'])
    flat_grid = torch.index_add(flat_grid, 0, idx_Q22, contributions['Q22'])

    # Reshape the grid back to 2D and normalize
    grid = flat_grid.view(Nx, Ny)
    grid /= dx * dy
    return grid


def apply_boundaries(space, particles_position, particles_velocity, particles_charge, particles_mass, particles_specie, Lx, Ly, type="periodic"):
    """
    Applies boundary conditions to particles in the simulation.

    Parameters:
        space: Simulation space object.
        particles_position (torch.Tensor): Particle positions.
        particles_velocity (torch.Tensor): Particle velocities.
        Lx, Ly (float): Dimensions of the simulation domain.
        type (str): Type of boundary condition ("periodic", "reflecting", "absorbing").

    Returns:
        tuple: Updated particle positions and velocities.
    """
    if type == "periodic":
        # Apply periodic boundary conditions
        particles_position[:, 0] %= Lx
        particles_position[:, 1] %= Ly

    elif type == "reflecting":
        # Reflecting boundaries: Invert velocity at domain edges
        mask_x_low = particles_position[:, 0] < 0
        mask_x_high = particles_position[:, 0] > Lx
        particles_velocity[mask_x_low | mask_x_high, 0] *= -1
        particles_position[:, 0] = torch.clamp(particles_position[:, 0], 0, Lx)

        mask_y_low = particles_position[:, 1] < 0
        mask_y_high = particles_position[:, 1] > Ly
        particles_velocity[mask_y_low | mask_y_high, 1] *= -1
        particles_position[:, 1] = torch.clamp(particles_position[:, 1], 0, Ly)

    elif type == "absorbing":
        # Absorbing boundaries: Deactivate particles outside the domain
        mask_inside = (particles_position[:, 0] >= 0) & (particles_position[:, 0] <= Lx) & \
                      (particles_position[:, 1] >= 0) & (particles_position[:, 1] <= Ly)
        space.particles_active[~mask_inside] = False

    else:
        raise ValueError(f"Unknown boundary type '{type}'. Supported types: 'periodic', 'reflecting', 'absorbing'.")

    # Handle interactions with solid boundaries
    i_indices = torch.clamp((particles_position[:, 0] / space.dx).long(), 0, space.Nx - 1)
    j_indices = torch.clamp((particles_position[:, 1] / space.dy).long(), 0, space.Ny - 1)

    inside_solid = space.solid_mask[i_indices, j_indices]

    # Reflecting solid boundaries
    inside_solid_reflecting = inside_solid & (space.solid_type[i_indices, j_indices] == 1)
    particles_velocity[inside_solid_reflecting] *= -1

    # Absorbing solid boundaries
    inside_solid_absorbing = inside_solid & (space.solid_type[i_indices, j_indices] == 2)
    space.particles_active[inside_solid_absorbing] = False

    return particles_position, particles_velocity
