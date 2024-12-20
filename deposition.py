import torch

def linear_interpolation_deposition(grid, quantity, positions, Nx, Ny, dx, dy):
    # Deposizione CIC
    # num_particles = len(positions)
    # for i in range(num_particles):
    #     x, y = positions[i]
    #     ix, iy = int(x / dx), int(y / dy)  # Trova il nodo più vicino
    #     grid[ix, iy] += quantity[i]  # Aggiungi la carica al nodo
    
    ix = (positions[:, 0] / dx).long()  # Trova gli indici x
    iy = (positions[:, 1] / dy).long()  # Trova gli indici y
    return torch.index_add(grid, 0, ix * Ny + iy, quantity)  # Deposizione parallela
    
    
    
    def nearest_grid_point_deposition(grid, quantity, positions, Nx, Ny, dx, dy):
        # num_particles = len(positions)
        # for i in range(num_particles):
        #     x, y = positions[i]
        #     ix, iy = int(round(x / dx)), int(round(y / dy))  # Trova il nodo più vicino
        #     grid[ix, iy] += quantity[i]  # Aggiungi la carica al nodo

        ix = torch.floor(positions[:, 0] / dx).long()  # Indice x della cella inferiore
        iy = torch.floor(positions[:, 1] / dy).long()  # Indice y della cella inferiore

        # Step 2: Calcolo delle distanze relative (peso CIC)
        wx = (positions[:, 0] / dx) - ix  # Peso x verso la cella superiore
        wy = (positions[:, 1] / dy) - iy  # Peso y verso la cella superiore

        # Step 3: Calcolo dei contributi ponderati per le 4 celle circostanti
        contributions = {
            'Q11': quantity * (1 - wx) * (1 - wy),  # Peso verso la cella inferiore sinistra
            'Q12': quantity * (1 - wx) * wy,       # Peso verso la cella inferiore destra
            'Q21': quantity * wx * (1 - wy),       # Peso verso la cella superiore sinistra
            'Q22': quantity * wx * wy,             # Peso verso la cella superiore destra
        }

        # Step 4: Indici lineari per le 4 celle circostanti
        ixp1 = (ix + 1) % Nx  # Gestione dei bordi (periodicità)
        iyp1 = (iy + 1) % Ny

        idx_Q11 = ix * Ny + iy
        idx_Q12 = ix * Ny + iyp1
        idx_Q21 = ixp1 * Ny + iy
        idx_Q22 = ixp1 * Ny + iyp1

        # Step 5: Deposizione delle cariche usando torch.index_add
        flat_grid = grid.flatten()  # Reshape della griglia per usare indici lineari
        torch.index_add(flat_grid, 0, idx_Q11, contributions['Q11'])
        torch.index_add(flat_grid, 0, idx_Q12, contributions['Q12'])
        torch.index_add(flat_grid, 0, idx_Q21, contributions['Q21'])
        torch.index_add(flat_grid, 0, idx_Q22, contributions['Q22'])

        # Reshape della griglia al formato 2D
        grid = flat_grid.view(Nx, Ny)
        
        return grid