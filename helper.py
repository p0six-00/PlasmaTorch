import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from utils import *

set_seed(1234)

def draw_histogram_xy(data_xy):
    
    x_positions = data_xy[:, 0]
    y_positions = data_xy[:, 1]
    
        # Disegna gli istogrammi
    plt.figure(figsize=(14, 6))

    # Istogramma delle posizioni x
    plt.subplot(1, 2, 1)
    plt.hist(x_positions, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Istogramma delle Posizioni X")
    plt.xlabel("Posizione X")
    plt.ylabel("Frequenza")

    # Istogramma delle posizioni y
    plt.subplot(1, 2, 2)
    plt.hist(y_positions, bins=30, alpha=0.7, edgecolor='black')
    plt.title("Istogramma delle Posizioni Y")
    plt.xlabel("Posizione Y")
    plt.ylabel("Frequenza")

    # Mostra i grafici
    plt.tight_layout()
    plt.show()
    
def dynamic_slider(simulation, Lx, Ly):
    
    
    particles_position_chronology = simulation['particles_position_chronology']
    particles_velocity_chronology = simulation['particles_velocity_chronology']
    fields_chronology = simulation['fields_chronology']
    grid_chronology = simulation['grid_chronology']
    particles_specie_chronology = simulation['particles_specie_chronology']
    particles_active_chronology = simulation['particles_active_chronology']
    kinetic_energy_chronology = simulation['kinetic_energy_chronology']
    potential_energy_chronology = simulation['potential_energy_chronology']
    mechanic_energy_chronology = simulation['mechanic_energy_chronology']
    solid_mask = simulation['solid_mask']
    fixed_potential_value = simulation['fixed_potential_value']

    # Configurazione della figura
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.2)  # Spazio extra per il widget slider

    # Configurazione iniziale dell'immagine della densità
    im_density = axs[0, 0].imshow(grid_chronology[0, :, :, 0].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axs[0, 0].set_title("Distribuzione della Densità")
    plt.colorbar(im_density, ax=axs[0, 0], label='Densità')
    

    # Overlay dei potenziali fissi
    #potential_overlay = axs[0, 1].imshow(fixed_potential_value.T, cmap='viridis', alpha=0.1, origin='lower',  vmax =fixed_potential_value.max(), vmin = fixed_potential_value.min(), extent=[0, Lx, 0, Ly])

    # Overlay zone campo magnetico
    #magnetic_overlay = axs[0, 1].imshow(fields_chronology[0, :, :, 1, 2].T, cmap='plasma', alpha=0.1, origin='lower',  vmax =fields_chronology[0, :, :, 1, 2].max(), vmin = fields_chronology[0, :, :, 1, 2].min(), extent=[0, Lx, 0, Ly])

    # Overlay delle superfici solide
    solid_overlay = axs[0, 1].imshow(~solid_mask.T, cmap='gray', origin='lower',  alpha=0.9, extent=[0, Lx, 0, Ly])
    
    # Configurazione iniziale dello scatter plot delle particelle con colori per specie
    unique_species = np.unique(particles_specie_chronology)
    scatter_plots = []
    colors = plt.cm.tab10.colors  # Tavolozza di colori predefinita

    for i, specie in enumerate(unique_species):
        species_mask = particles_specie_chronology[0] == specie
        scatter = axs[0, 1].scatter(
            particles_position_chronology[0, species_mask, 0], 
            particles_position_chronology[0, species_mask, 1], 
            s=1, 
            color=colors[i % len(colors)], 
            label=f"Specie {specie}"
        )
        scatter_plots.append(scatter)

    axs[0, 1].set_xlim(0, Lx)
    axs[0, 1].set_ylim(0, Ly)
    axs[0, 1].set_title("Traiettoria delle Particelle")
    axs[0, 1].legend(loc="upper right")

    
    # Configurazione iniziale dell'immagine del potenziale elettrico
    im_potential = axs[1, 0].imshow(grid_chronology[0, :, :, 1].T, extent=[0, Lx, 0, Ly], origin='lower', cmap='viridis')
    axs[1, 0].set_title("Distribuzione del Potenziale Elettrostatico")
    plt.colorbar(im_potential, ax=axs[1, 0], label='Potenziale (V)')

    # Slider per il controllo del tempo
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])  # Posizione dello slider [x, y, larghezza, altezza]
    time_slider = Slider(ax_slider, "Time", 0, len(grid_chronology) - 1, valinit=0, valstep=1)

    # Funzione per aggiornare le visualizzazioni
    def update(val):
        time_index = int(time_slider.val)  # Ottieni l'indice temporale
        
        # Aggiorna la densità
        im_density.set_data(grid_chronology[time_index, :, :, 0].T)
        
        # Aggiorna le posizioni delle particelle per specie
        for i, specie in enumerate(unique_species):
            species_mask = particles_specie_chronology[time_index] == specie
            scatter_plots[i].set_offsets(particles_position_chronology[time_index][species_mask & particles_active_chronology[time_index]])
        
        # Aggiorna il potenziale elettrico
        im_potential.set_data(grid_chronology[time_index, :, :, 1].T)
        
        fig.canvas.draw_idle()

    # Collega lo slider alla funzione di aggiornamento
    time_slider.on_changed(update)

    plt.show()
