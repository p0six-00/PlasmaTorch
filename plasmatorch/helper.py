import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from utils import *

set_seed(1234)
def draw_histogram_xy(data_xy):
    """
    Draws histograms for the x and y positions of particles.

    Parameters:
        data_xy (numpy.ndarray): Array of shape (num_particles, 2) containing x and y positions.
    """
    x_positions = data_xy[:, 0]
    y_positions = data_xy[:, 1]

    # Create histograms
    plt.figure(figsize=(14, 6))

    # Histogram for x positions
    plt.subplot(1, 2, 1)
    plt.hist(x_positions, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Histogram of X Positions")
    plt.xlabel("X Position")
    plt.ylabel("Frequency")

    # Histogram for y positions
    plt.subplot(1, 2, 2)
    plt.hist(y_positions, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Histogram of Y Positions")
    plt.xlabel("Y Position")
    plt.ylabel("Frequency")

    # Display plots
    plt.tight_layout()
    plt.show()


def dynamic_slider(simulation, Lx, Ly):
    """
    Creates an interactive visualization with a slider to explore simulation results over time.

    Parameters:
        simulation (dict): Dictionary containing simulation data with the following keys:
            - 'particles_position_chronology': Time evolution of particle positions.
            - 'particles_velocity_chronology': Time evolution of particle velocities.
            - 'fields_chronology': Time evolution of field data (density, potential, etc.).
            - 'grid_chronology': Time evolution of grid quantities (density, potential, etc.).
            - 'particles_specie_chronology': Time evolution of particle species.
            - 'particles_active_chronology': Time evolution of active particle masks.
            - 'kinetic_energy_chronology': Kinetic energy over time.
            - 'potential_energy_chronology': Potential energy over time.
            - 'mechanic_energy_chronology': Total mechanical energy over time.
            - 'solid_mask': Boolean mask for solid regions in the simulation.
            - 'fixed_potential_value': Fixed potential values for specific regions.
        Lx, Ly (float): Dimensions of the simulation domain.
    """
    # Extract data from the simulation dictionary
    particles_position_chronology = simulation['particles_position_chronology']
    particles_velocity_chronology = simulation['particles_velocity_chronology']
    fields_chronology = simulation['fields_chronology']
    grid_chronology = simulation['grid_chronology']
    particles_specie_chronology = simulation['particles_specie_chronology']
    particles_active_chronology = simulation['particles_active_chronology']
    solid_mask = simulation['solid_mask']
    fixed_potential_value = simulation['fixed_potential_value']

    # Setup the figure
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.2)  # Add space for the slider

    # Density distribution plot
    im_density = axs[0, 0].imshow(
        grid_chronology[0, :, :, 0].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis'
    )
    axs[0, 0].set_title("Density Distribution")
    plt.colorbar(im_density, ax=axs[0, 0], label='Density')

    # Overlays for fixed potentials, magnetic fields, and solids
    axs[0, 1].imshow(fixed_potential_value.T, cmap='viridis', alpha=0.1, origin='lower', extent=[0, Lx, 0, Ly])
    axs[0, 1].imshow(fields_chronology[0, :, :, 1, 2].T, cmap='plasma', alpha=0.1, origin='lower', extent=[0, Lx, 0, Ly])
    axs[0, 1].imshow(~solid_mask.T, cmap='gray', origin='lower', alpha=0.9, extent=[0, Lx, 0, Ly])

    # Particle trajectory scatter plot
    unique_species = np.unique(particles_specie_chronology)
    scatter_plots = []
    colors = plt.cm.tab10.colors  # Default color palette

    for i, specie in enumerate(unique_species):
        species_mask = particles_specie_chronology[0] == specie
        scatter = axs[0, 1].scatter(
            particles_position_chronology[0, species_mask, 0],
            particles_position_chronology[0, species_mask, 1],
            s=1,
            color=colors[i % len(colors)],
            label=f"Species {specie}"
        )
        scatter_plots.append(scatter)

    axs[0, 1].set_xlim(0, Lx)
    axs[0, 1].set_ylim(0, Ly)
    axs[0, 1].set_title("Particle Trajectories")
    axs[0, 1].legend(loc="upper right")

    # Electrostatic potential distribution plot
    im_potential = axs[1, 0].imshow(
        grid_chronology[0, :, :, 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis'
    )
    axs[1, 0].set_title("Electrostatic Potential Distribution")
    plt.colorbar(im_potential, ax=axs[1, 0], label='Potential')

    # Slider for time control
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    time_slider = Slider(ax_slider, "Time", 0, len(grid_chronology) - 1, valinit=0, valstep=1)

    # Update function for the slider
    def update(val):
        time_index = int(time_slider.val)  # Get the current time index
        
        # Update density plot
        im_density.set_data(grid_chronology[time_index, :, :, 0].T)

        # Update particle positions for each species
        for i, specie in enumerate(unique_species):
            species_mask = particles_specie_chronology[time_index] == specie
            scatter_plots[i].set_offsets(
                particles_position_chronology[time_index][species_mask & particles_active_chronology[time_index]]
            )

        # Update potential plot
        im_potential.set_data(grid_chronology[time_index, :, :, 1].T)

        fig.canvas.draw_idle()

    # Connect the slider to the update function
    time_slider.on_changed(update)

    # Show the interactive plot
    plt.show()
