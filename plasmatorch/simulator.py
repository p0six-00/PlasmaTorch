import torch
import numpy as np
import random
from tqdm import tqdm
import h5py
import os
from pathlib import Path

from deposition import *
from fields import *
from helper import *
from utils import *
from physics import *

# Set the device to CUDA if available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Main simulation class
class PlasmaTorch:
    """
    Represents the simulation space for particle interactions, electromagnetic fields,
    and other physical phenomena.
    """
    def __init__(self, simulation_name, Lx, Ly, Nx, Ny, simulation_time, dt, 
                 boundaries_type="reflecting", save_frequency=1, seed=1234, backend=device, verbose=False):
        """
        Initialize the simulation environment.

        Parameters:
        - simulation_name: Name of the simulation file.
        - Lx, Ly: Dimensions of the simulation grid.
        - Nx, Ny: Number of grid points in each direction.
        - simulation_time: Total time of the simulation.
        - dt: Time step size.
        - boundaries_type: Type of boundary conditions (e.g., "reflecting").
        - save_frequency: Frequency (in steps) of saving simulation data.
        - seed: Random seed for reproducibility.
        - backend: Device to use ("cuda" or "cpu").
        - verbose: Whether to print additional information.
        """
        set_seed(seed)
        self.simulation_name = simulation_name + ".h5"
        
        # Initialize grid parameters
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.dx, self.dy = Lx / Nx, Ly / Ny  # Grid spacing

        #self.normalization = normalization #! To be implemented
        self.dt = dt
        self.simulation_time = simulation_time
        self.boundaries_type = boundaries_type

        # Initialize grid and field arrays
        self.grid = torch.zeros((Nx, Ny, 2), device=backend)
        self.fields = torch.zeros((Nx, Ny, 2, 3), device=backend)
        self.particles_position = torch.rand((0, 3), device=backend) * torch.tensor([Lx, Ly, 0], device=backend)
        self.particles_velocity = torch.zeros((0, 3), device=backend)

        # Define solid masks and boundary conditions
        self.solid_mask = torch.zeros((Nx, Ny), dtype=torch.bool, device=backend)
        self.solid_type = torch.zeros((Nx, Ny), dtype=torch.int16, device=backend)
        self.type_of_solid_surfaces = np.array(["void", "reflecting", "absorbing"])
        self.fixed_potential_mask = torch.zeros((Nx, Ny), dtype=torch.bool, device=backend)
        self.fixed_potential_value = torch.zeros((Nx, Ny), device=backend)

        # Particle and species registers
        self.total_species = 0
        self.particles_specie = torch.tensor([], dtype=torch.int16, device=backend)
        self.particles_active = torch.zeros((0,), dtype=torch.bool, device=backend)
        self.mass_register = torch.tensor([], device=backend)
        self.charge_register = torch.tensor([], device=backend)
        self.name_register = np.array([])

        # Simulation control variables
        self.steps = int(simulation_time / dt)
        self.current_step = 0
        self.save_frequency = save_frequency
        self.poisson_tolerance = 1e-9

        # Energy and history trackers
        self.fields_chronology = []
        self.particles_position_chronology = []
        self.particles_velocity_chronology = []
        self.grid_chronology = []
        self.particles_specie_chronology = []
        self.particles_active_chronology = []
        self.kinetic_energy_chronology = []
        self.potential_energy_chronology = []
        self.mechanic_energy_chronology = []

        # Set simulation device and Poisson solver method
        self.device = backend
        self.verbose = verbose
        self.poisson_method = "jacobi"

        # Initialize simulation
        self._build_simulation(remove=True)

    def update(self):
        """
        Perform one step of the simulation.
        Updates particle positions, fields, and energies.
        """
        active_mask = self.particles_active.clone()
        particles_specie_active = self.particles_specie[active_mask]
        self.particles_charge = self.charge_register[particles_specie_active]
        self.particles_mass = self.mass_register[particles_specie_active]

        # Update grid density and potential
        self.grid[:, :, 0] = cloud_in_cell_deposition(self, self.particles_charge, self.particles_position[active_mask], self.Nx, self.Ny, self.dx, self.dy, device=self.device)
        self.grid[:, :, 1] = self.solve_poisson_equation()

        # Compute electric field
        self.fields[:, :, 0, 0], self.fields[:, :, 0, 1], self.fields[:, :, 0, 2] = compute_electric_field(self, self.grid[:, :, 1], self.dx, self.dy, device=self.device)

        # Save data and energies at specified intervals
        if self.current_step % self.save_frequency == 0:
            self.kinetic_energy, self.potential_energy, self.mechanic_energy = self.measure_energies()
            self.save_data()
            self.save_chronologies_to_file()
            self.clear_chronologies()

        # Update particle positions and velocities using Boris method
        self.particles_position[active_mask], self.particles_velocity[active_mask] = boris_method(self, self.particles_position[active_mask], self.particles_velocity[active_mask], self.particles_charge, self.particles_mass, self.fields, self.dt, self.dx, self.dy, self.Nx, self.Ny, self.Lx, self.Ly)
        self.particles_position[active_mask], self.particles_velocity[active_mask] = apply_boundaries(self, self.particles_position[active_mask], self.particles_velocity[active_mask], self.particles_charge, self.particles_mass, particles_specie_active, self.Lx, self.Ly, self.boundaries_type)

        self.current_step += 1

    def simulate(self, steps=None):
        """
        Run the simulation for a specified number of steps.
        """
        if steps is None:
            steps = self.steps

        with torch.no_grad():
            for step in tqdm(range(steps), unit=" step", leave=True):
                self.update()

    def save_data(self):
        """
        Save the current simulation state to memory.
        """
        self.fields_chronology.append(self.fields.clone().to("cpu"))
        self.particles_position_chronology.append(self.particles_position.clone().to("cpu"))
        self.particles_velocity_chronology.append(self.particles_velocity.clone().to("cpu"))
        self.grid_chronology.append(self.grid.clone().to("cpu"))
        self.particles_specie_chronology.append(self.particles_specie.clone().to("cpu"))
        self.particles_active_chronology.append(self.particles_active.clone().to("cpu"))
        self.kinetic_energy_chronology.append(self.kinetic_energy)
        self.potential_energy_chronology.append(self.potential_energy)
        self.mechanic_energy_chronology.append(self.mechanic_energy)
        
    def solve_poisson_equation(self):
        """
        Solves the Poisson equation for the electric potential using the specified method.
        
        Returns:
            Updated electric potential grid.
        """
        if self.poisson_method == "jacobi":
            self.grid[:, :, 1] = poisson_equation_jacobi(
                self.grid[:, :, 1], self.grid[:, :, 0], self.fixed_potential_mask, 
                self.fixed_potential_value, self.dx, tol=self.poisson_tolerance, verbose=self.verbose
            )
        elif self.poisson_method == "gauss_seidel":
            self.grid[:, :, 1] = poisson_equation_gauss_seidel(
                self.grid[:, :, 1], self.grid[:, :, 0], self.fixed_potential_mask, 
                self.fixed_potential_value, self.dx, tol=self.poisson_tolerance, verbose=self.verbose
            )
        elif self.poisson_method == "red-black":
            self.grid[:, :, 1] = poisson_equation_red_black_gauss_seidel(
                self.grid[:, :, 1], self.grid[:, :, 0], self.fixed_potential_mask, 
                self.fixed_potential_value, self.dx, tol=self.poisson_tolerance, verbose=self.verbose
            )
        return self.grid[:, :, 1]

    def create_new_specie(self, name, num_particles, temperature=None, distribution="zero", 
                        Nx=None, Ny=None, Lx=None, Ly=None, mass=1, charge=1, 
                        disposition="random", position=None):
        """
        Creates a new particle species and adds it to the simulation.

        Parameters:
            name (str): Name of the species.
            num_particles (int): Number of particles.
            temperature (float, optional): Temperature for velocity distribution.
            distribution (str): Type of velocity distribution ("zero" or "maxwell-boltzmann").
            mass (float): Particle mass.
            charge (float): Particle charge.
            disposition (str): Particle positioning method ("random", "uniform", "specific").
            position (tuple, optional): Specific position if disposition is "specific".
        """
        if disposition == "random":
            new_particles_position = place_random(num_particles, self.Lx, self.Ly, device=self.device)
        elif disposition == "uniform":
            new_particles_position = place_uniform(num_particles, self.Nx, self.Ny, self.Lx, self.Ly, device=self.device)
        elif disposition == "specific":
            new_particles_position = torch.zeros((num_particles, 3), device=self.device) + \
                                    torch.cat([torch.tensor(position, device=self.device), 
                                                torch.tensor([0], device=self.device)], dim=0)

        if distribution == "zero":
            new_particles_velocity = torch.zeros(num_particles, 3, device=self.device)
        elif distribution == "maxwell-boltzmann":
            new_particles_velocity = maxwell_boltzmann_distribution(num_particles, temperature, mass, device=self.device)

        if name in self.name_register:
            id = np.where(self.name_register == name)[0][0]
        else:
            id = self.total_species
            self.total_species += 1
            self.name_register = np.append(self.name_register, name)
            self.mass_register = torch.cat((self.mass_register, torch.tensor([mass], device=self.device)))
            self.charge_register = torch.cat((self.charge_register, torch.tensor([charge], device=self.device)))

        new_particles_specie = torch.full((num_particles,), id, device=self.device)
        new_particles_active = torch.full((num_particles,), True, device=self.device)

        # Update particle registers
        self.particles_position = torch.vstack([self.particles_position, new_particles_position])
        self.particles_velocity = torch.vstack([self.particles_velocity, new_particles_velocity])
        self.particles_specie = torch.cat([self.particles_specie, new_particles_specie])
        self.particles_active = torch.cat([self.particles_active, new_particles_active])

        # Update simulation
        self._build_simulation(remove=True)

    def measure_energies(self):
        """
        Measures the kinetic, potential, and total mechanical energy of the system.

        Returns:
            Tuple containing kinetic energy, potential energy, and total energy.
        """
        active_mask = self.particles_active
        particle_masses = self.mass_register[self.particles_specie[active_mask]]
        v = self.particles_velocity[active_mask]
        kinetic_energy = 0.5 * torch.sum(particle_masses * (v[:, 0]**2 + v[:, 1]**2))
        rho = self.grid[:, :, 0]
        phi = self.grid[:, :, 1]
        potential_energy = 0.5 * self.dx * self.dy * torch.sum(rho * phi)
        total_energy = kinetic_energy + potential_energy
        return kinetic_energy.item(), potential_energy.item(), total_energy.item()

    def define_constant_magnetic_field(self, x_min, x_max, y_min, y_max, bz):
        """
        Defines a constant magnetic field within a specified rectangular region.

        Parameters:
            x_min, x_max, y_min, y_max: Region boundaries in physical units.
            bz: Magnitude of the magnetic field in the z-direction.
        """
        i_min, i_max, j_min, j_max = self.get_index_cell_from_rectangle(x_min, x_max, y_min, y_max)
        self.fields[i_min:i_max + 1, j_min:j_max + 1, 1, :] = bz

    def add_rectangle_object(self, x_min, x_max, y_min, y_max, solid=True, type="reflecting", potential=None):
        """
        Adds a rectangular object to the simulation space.

        Parameters:
            x_min, x_max, y_min, y_max: Rectangle boundaries.
            solid (bool): Whether the rectangle is solid.
            type (str): Type of solid surface ("reflecting", "absorbing").
            potential (float, optional): Fixed potential value within the rectangle.
        """
        i_min, i_max, j_min, j_max = self.get_index_cell_from_rectangle(x_min, x_max, y_min, y_max)

        if solid:
            self.solid_mask[i_min:i_max + 1, j_min:j_max + 1] = True
            self.solid_type[i_min:i_max + 1, j_min:j_max + 1] = (self.type_of_solid_surfaces == type).argmax()

        if potential is not None:
            self.fixed_potential_mask[i_min:i_max + 1, j_min:j_max + 1] = True
            self.fixed_potential_value[i_min:i_max + 1, j_min:j_max + 1] = potential

        # Update simulation
        self._build_simulation(remove=True)

    def get_index_cell_from_rectangle(self, x_min, x_max, y_min, y_max):
        """
        Converts physical coordinates of a rectangle to grid indices.

        Parameters:
            x_min, x_max, y_min, y_max: Physical boundaries.

        Returns:
            Indices (i_min, i_max, j_min, j_max) corresponding to the rectangle.
        """
        i_min = int(x_min / self.dx)
        i_max = int(x_max / self.dx)
        j_min = int(y_min / self.dy)
        j_max = int(y_max / self.dy)
        i_min = max(0, i_min)
        i_max = min(self.Nx - 1, i_max)
        j_min = max(0, j_min)
        j_max = min(self.Ny - 1, j_max)
        return i_min, i_max, j_min, j_max

    def _build_simulation(self, remove = False):
        """
        Builds the simulation file and initializes necessary datasets.
        """
        if remove:
            simulation_file = Path(self.simulation_name)
            if simulation_file.is_file():
                os.remove(self.simulation_name)
        with h5py.File(self.simulation_name, "a") as f:
            self._append_dataset(f, "solid_mask", self.solid_mask.to("cpu").numpy())
            self._append_dataset(f, "solid_type", self.solid_type.to("cpu").numpy())
            self._append_dataset(f, "fixed_potential_mask", self.fixed_potential_mask.to("cpu").numpy())
            self._append_dataset(f, "fixed_potential_value", self.fixed_potential_value.to("cpu").numpy())
            self._append_dataset(f, "mass_register", self.mass_register.to("cpu").numpy())
            self._append_dataset(f, "charge_register", self.charge_register.to("cpu").numpy())

    def save_chronologies_to_file(self):
        """
        Saves accumulated simulation histories to the HDF5 file.
        """
        if self.verbose:
            print(f"[INFO] Saving chronologies to '{self.simulation_name}'...")

        with h5py.File(self.simulation_name, "a") as f:
            # Save each type of chronology if data exists
            if self.fields_chronology:
                self._append_dataset(f, "fields_chronology", torch.stack(self.fields_chronology).to("cpu").numpy())
            
            # Esempio: particles_position_chronology
            if len(self.particles_position_chronology) > 0:
                data = torch.stack(self.particles_position_chronology).to("cpu").numpy()
                self._append_dataset(f, "particles_position_chronology", data)

            # Esempio: particles_velocity_chronology
            if len(self.particles_velocity_chronology) > 0:
                data = torch.stack(self.particles_velocity_chronology).to("cpu").numpy()
                self._append_dataset(f, "particles_velocity_chronology", data)

            # Esempio: grid_chronology
            if len(self.grid_chronology) > 0:
                data = torch.stack(self.grid_chronology).to("cpu").numpy()
                self._append_dataset(f, "grid_chronology", data)

            # Esempio: particles_specie_chronology
            if len(self.particles_specie_chronology) > 0:
                data = torch.stack(self.particles_specie_chronology).to("cpu").numpy()
                self._append_dataset(f, "particles_specie_chronology", data)
                
            # Esempio: particles_active_chronology
            if len(self.particles_active_chronology) > 0:
                data = torch.stack(self.particles_active_chronology).to("cpu").numpy()
                self._append_dataset(f, "particles_active_chronology", data)

            # Esempio: kinetic_energy_chronology (lista di float, converti in np.array)
            if len(self.kinetic_energy_chronology) > 0:
                data = np.array(self.kinetic_energy_chronology, dtype=np.float64)
                self._append_dataset(f, "kinetic_energy_chronology", data)

            # Esempio: potential_energy_chronology
            if len(self.potential_energy_chronology) > 0:
                data = np.array(self.potential_energy_chronology, dtype=np.float64)
                self._append_dataset(f, "potential_energy_chronology", data)

            # Esempio: mechanic_energy_chronology
            if len(self.mechanic_energy_chronology) > 0:
                data = np.array(self.mechanic_energy_chronology, dtype=np.float64)
                self._append_dataset(f, "mechanic_energy_chronology", data)

        if self.verbose:
            print(f"[INFO] Salvati i dati di cronologia (step={self.current_step}).")

    def _append_dataset(self, h5file, dset_name, new_data):
        """
        Appends data to an existing dataset in the HDF5 file or creates it if absent.

        Parameters:
            h5file: HDF5 file object.
            dset_name (str): Dataset name.
            new_data (array): Data to append.
        """
        if dset_name not in h5file:
            maxshape = (None,) + new_data.shape[1:]
            h5file.create_dataset(dset_name, data=new_data, maxshape=maxshape, chunks=True)
        else:
            dset = h5file[dset_name]
            old_size = dset.shape[0]
            new_size = old_size + new_data.shape[0]
            dset.resize(new_size, axis=0)
            dset[old_size:new_size, ...] = new_data

    def clear_chronologies(self):
        """
        Clears the memory of all stored chronologies and releases GPU cache if applicable.
        """
        self.fields_chronology = []
        self.particles_position_chronology = []
        self.particles_velocity_chronology = []
        self.grid_chronology = []
        self.particles_specie_chronology = []
        self.particles_active_chronology = []
        self.kinetic_energy_chronology = []
        self.potential_energy_chronology = []
        self.mechanic_energy_chronology = []

        if self.device == "cuda":
            torch.cuda.empty_cache()

        if self.verbose:
            print("[INFO] Chronologies cleared from memory.")
