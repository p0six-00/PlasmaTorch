import torch
from scipy.constants import *
from utils import *
device = "cuda" if torch.cuda.is_available() else "cpu"

set_seed(1234)

def poisson_equation_gauss_seidel(phi, rho, fixed_potential_mask, fixed_potential_value,
                                  dx, max_iter=10000, tol=1e-7, verbose=False):
    """
    Solves the Poisson equation ∇²φ = -ρ/ε₀ using the Gauss-Seidel iterative method.

    Parameters:
        phi (torch.Tensor): Initial guess for the potential, updated in-place.
        rho (torch.Tensor): Charge density distribution.
        fixed_potential_mask (torch.Tensor): Boolean mask indicating fixed potential points.
        fixed_potential_value (torch.Tensor): Values of fixed potentials.
        dx (float): Grid spacing (assumed equal in x and y directions).
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance for the residual.
        verbose (bool): If True, prints residual information during iterations.

    Returns:
        torch.Tensor: Converged potential distribution.
    """
    residual = float('inf')  # Initial residual
    iter_count = 0

    while residual > tol and iter_count < max_iter:
        phi_old = phi.clone()  # Save old potential for residual calculation

        # Update potential using Gauss-Seidel formula for interior points
        phi[1:-1, 1:-1] = 0.25 * (
            phi[2:, 1:-1] + phi[:-2, 1:-1] +
            phi[1:-1, 2:] + phi[1:-1, :-2] +
            dx**2 * rho[1:-1, 1:-1] / epsilon_0
        )
        
        # Apply fixed potential constraints
        phi[fixed_potential_mask] = fixed_potential_value[fixed_potential_mask]

        # Calculate residual as the maximum change in φ
        residual = torch.max(torch.abs(phi - phi_old))
        iter_count += 1

        if verbose and iter_count % 100 == 0:
            print(f"Iteration: {iter_count}, Residual: {residual:.2e}")
    
    if verbose:
        print(f"Convergence reached in {iter_count} iterations with residual {residual:.2e}")
    
    return phi


def poisson_equation_jacobi(phi, rho, fixed_potential_mask, fixed_potential_value,
                            dx, max_iter=10000, tol=1e-7, verbose=False):
    """
    Solves the Poisson equation ∇²φ = -ρ/ε₀ using the Jacobi iterative method.

    Parameters:
        Similar to `poisson_equation_gauss_seidel`.

    Returns:
        torch.Tensor: Converged potential distribution.
    """
    Nx, Ny = phi.shape
    phi_new = torch.empty_like(phi)

    for k in range(max_iter):
        phi_old = phi  # Use φ_old to read values for Jacobi updates

        # Jacobi update for interior points
        phi_new[1:-1, 1:-1] = 0.25 * (
            phi_old[2:, 1:-1] + phi_old[:-2, 1:-1] +
            phi_old[1:-1, 2:] + phi_old[1:-1, :-2] +
            dx**2 * rho[1:-1, 1:-1] / epsilon_0
        )

        # Apply fixed potential constraints
        phi_new[fixed_potential_mask] = fixed_potential_value[fixed_potential_mask]

        # Calculate residual
        r = torch.max(torch.abs(phi_new - phi_old))
        phi, phi_new = phi_new.clone(), phi.clone()

        if verbose and k % 50 == 0:
            print(f"Iteration = {k}, Residual = {r.item():.2e}")
        if r < tol:
            break

    if verbose:
        print(f"Convergence reached in {k} iterations with residual {r.item():.2e}")
    return phi

def poisson_equation_red_black_gauss_seidel(phi, rho, fixed_potential_mask, fixed_potential_value,
                              dx, max_iter=10000, tol=1e-7, verbose=False):
    """
    Solves the Poisson equation ∇²φ = -ρ/ε₀ using the Red-Black Gauss-Seidel method.

    Parameters:
        Similar to `poisson_equation_gauss_seidel`, with an additional red-black decomposition.

    Returns:
        torch.Tensor: Converged potential distribution.
    """
    Nx, Ny = phi.shape

    # Crea maschere booleane “red” e “black”
    # (i+j) % 2 == 0 -> red, ==1 -> black
    i_idx = torch.arange(Nx, device=phi.device).view(-1, 1)
    j_idx = torch.arange(Ny, device=phi.device).view(1, -1)
    # Broadcasting per ottenere shape Nx x Ny
    sum_idx = i_idx + j_idx

    red_mask = (sum_idx % 2 == 0)
    black_mask = ~red_mask  # complementare

    # Funzione residuo per monitorare la convergenza
    def get_residual(old_phi, new_phi):
        return torch.max(torch.abs(new_phi - old_phi))

    for k in range(max_iter):
        phi_old = phi.clone()  # Copia la soluzione attuale per calcolare il residuo a fine step

        # 1) Update RED points
        #    phi_red = 0.25*(phi[up]+phi[down]+phi[left]+phi[right]) + dx^2/(4 eps0)*rho
        #    Solo per le celle “red_mask” e non soggette a potenziale fisso
        #    NB: I vicini di un nodo red sono black, e li leggiamo in “phi” già aggiornato?
        #    Nel Gauss-Seidel, “phi” viene usato in place, perché si suppone di avere i black già
        #    aggiornati all'iterazione (k), ma in questo schema “red-black” i black sono dell’iter k-1
        #    e i red verranno aggiornati a k.

        # Per non complicare eccessivamente la maschera ai bordi, aggiorniamo solo [1:-1,1:-1].
        # Negli esempi reali andrebbero gestiti i bordi con opportune BC.
        interior_mask = torch.zeros_like(red_mask)
        interior_mask[1:-1, 1:-1] = True

        update_mask_red = red_mask & interior_mask & (~fixed_potential_mask)

        phi_up    = phi.roll(-1, dims=0)  # shift in i = i+1
        phi_down  = phi.roll( 1, dims=0)  # shift i = i-1
        phi_right = phi.roll(-1, dims=1)  # shift j = j+1
        phi_left  = phi.roll( 1, dims=1)  # shift j = j-1

        phi[update_mask_red] = 0.25 * (
            phi_up[update_mask_red] +
            phi_down[update_mask_red] +
            phi_right[update_mask_red] +
            phi_left[update_mask_red]
        ) + (dx**2 / (4.0 * epsilon_0)) * rho[update_mask_red]

        # Imposta i nodi a potenziale fisso
        phi[fixed_potential_mask] = fixed_potential_value[fixed_potential_mask]

        # 2) Update BLACK points
        update_mask_black = black_mask & interior_mask & (~fixed_potential_mask)

        # Ricalcoliamo i shift su phi appena aggiornato (perché adesso i RED sono k e i BLACK da aggiornare)
        phi_up    = phi.roll(-1, dims=0)
        phi_down  = phi.roll( 1, dims=0)
        phi_right = phi.roll(-1, dims=1)
        phi_left  = phi.roll( 1, dims=1)

        phi[update_mask_black] = 0.25 * (
            phi_up[update_mask_black] +
            phi_down[update_mask_black] +
            phi_right[update_mask_black] +
            phi_left[update_mask_black]
        ) + (dx**2 / (4.0 * epsilon_0)) * rho[update_mask_black]

        # Imposta i nodi a potenziale fisso
        phi[fixed_potential_mask] = fixed_potential_value[fixed_potential_mask]

        # Calcolo residuo = max|phi_new - phi_old|
        r = get_residual(phi_old, phi)

        if verbose and k % 50 == 0:
            print(f"[Red-Black] Iter = {k}, Residuo = {r.item():.2e}")

        if r < tol:
            break

    if verbose:
        print(f"[Red-Black] Convergenza in {k} iterazioni, residuo = {r.item():.2e}")
        
    return phi
def compute_electric_field(space, phi, dx, dy, device=device):
    """
    Computes the electric field components (Ex, Ey) from the potential φ.

    Parameters:
        phi (torch.Tensor): Potential distribution.
        dx, dy (float): Grid spacing in x and y directions.

    Returns:
        tuple: Electric field components (Ex, Ey, Ez).
    """
    Ex = torch.zeros_like(phi, device=device)
    Ey = torch.zeros_like(phi, device=device)

    # Central difference for interior points
    Ex[1:-1, :] = -(phi[2:, :] - phi[:-2, :]) / (2 * dx)
    Ey[:, 1:-1] = -(phi[:, 2:] - phi[:, :-2]) / (2 * dy)

    # Forward/backward difference for boundaries
    Ex[0, :] = -(phi[1, :] - phi[0, :]) / dx
    Ex[-1, :] = -(phi[-1, :] - phi[-2, :]) / dx
    Ey[:, 0] = -(phi[:, 1] - phi[:, 0]) / dy
    Ey[:, -1] = -(phi[:, -1] - phi[:, -2]) / dy

    return Ex, Ey, 0


def boris_method(space, particles_position, particles_velocity, particles_charge, particles_mass, fields, dt, dx, dy, Nx, Ny, Lx, Ly):
    """
    Updates particle positions and velocities using the Boris algorithm.

    Parameters:
        particles_position, particles_velocity: Particle states.
        particles_charge, particles_mass: Particle properties.
        fields: Electric and magnetic fields.
        dt: Time step size.
        dx, dy: Grid spacing.

    Returns:
        tuple: Updated positions and velocities.
    """
    num_particles = particles_position.shape[0]

    # Interpolate fields to particle positions
    electric_field_at_particles = interpolate_fields(fields[:, :, 0], particles_position, Nx, Ny, dx, dy)
    magnetic_field_at_particles = interpolate_fields(fields[:, :, 1], particles_position, Nx, Ny, dx, dy)

    half_dt_qm = 0.5 * dt * particles_charge / particles_mass
    half_dt_qm = half_dt_qm.unsqueeze(dim=1)

    # Update velocity with electric field (half step)
    v_minus = particles_velocity + half_dt_qm * electric_field_at_particles

    # Magnetic field rotation
    t = half_dt_qm * magnetic_field_at_particles
    t_magnitude = torch.norm(t, dim=1, keepdim=True)
    s = 2 * t / (1 + t_magnitude**2)

    v_prime = v_minus + torch.cross(v_minus, t)
    v_plus = v_prime + torch.cross(v_prime, s)

    # Update velocity with electric field (half step)
    particles_velocity = v_plus + half_dt_qm * electric_field_at_particles

    # Update particle positions
    particles_position += particles_velocity * dt

    return particles_position, particles_velocity


def interpolate_fields(field_grid, particles_position, Nx, Ny, dx, dy):
    """
    Performs bilinear interpolation of field values at particle positions.

    Parameters:
        field_grid (torch.Tensor): Field values on the grid.
        particles_position (torch.Tensor): Particle positions.

    Returns:
        torch.Tensor: Interpolated field values at particle positions.
    """
    grid_x = (particles_position[:, 0] / dx).long()
    grid_y = (particles_position[:, 1] / dy).long()

    x0 = torch.clamp(grid_x, 0, Nx - 2)
    x1 = x0 + 1
    y0 = torch.clamp(grid_y, 0, Ny - 2)
    y1 = y0 + 1

    wx1 = (particles_position[:, 0] - x0 * dx) / dx
    wx0 = 1 - wx1
    wy1 = (particles_position[:, 1] - y0 * dy) / dy
    wy0 = 1 - wy1

    interpolated_field = (
        field_grid[x0, y0] * wx0.unsqueeze(dim=1) * wy0.unsqueeze(dim=1) +
        field_grid[x1, y0] * wx1.unsqueeze(dim=1) * wy0.unsqueeze(dim=1) +
        field_grid[x0, y1] * wx0.unsqueeze(dim=1) * wy1.unsqueeze(dim=1) +
        field_grid[x1, y1] * wx1.unsqueeze(dim=1) * wy1.unsqueeze(dim=1)
    )

    return interpolated_field