import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch.autograd.profiler as profiler

from simulator import *
from utils import *
set_seed(1234)

device = "cuda" if torch.cuda.is_available() else "cpu"

num_particles = 25
Nx, Ny = 200, 200  # Numero di nodi in x e y
Lx, Ly = 1.0, 1.0  # Dimensioni del dominio
dx, dy = Lx / Nx, Ly / Ny  # Passo della griglia

dt = 0.01
simulation_time = 5
boundaries_type = "reflecting"

space = Space(Lx, Ly, Nx, Ny, simulation_time, dt, boundaries_type)
steps = space.steps

space.create_new_specie("electron", charge = -0.01, mass = 0.1, num_particles= 1000)
space.create_new_specie("proton", charge = 0.01, mass = 1, num_particles= 5)

#with profiler.profile(record_shapes=True) as prof:
for step in tqdm(range(steps), unit=" step", leave=True):
    #draw_histogram_xy(particles_position)
    space.update()
    
    
    

#print(prof.key_averages().table(sort_by="cuda_time_total"))

particles_position_chronology = space.particles_position_chronology
particles_velocity_chronology = space.particles_velocity_chronology
fields_chronology = space.fields_chronology
grid_chronology = space.grid_chronology
particles_position_chronology = torch.stack(particles_position_chronology).cpu().numpy()
particles_velocity_chronology = torch.stack(particles_velocity_chronology).cpu().numpy()
fields_chronology = torch.stack(fields_chronology).cpu().numpy()
grid_chronology = torch.stack(grid_chronology).cpu().numpy()


dynamic_slider(grid_chronology, particles_position_chronology, Lx, Ly)


