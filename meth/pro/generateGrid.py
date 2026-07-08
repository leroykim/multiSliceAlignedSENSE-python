import math
import numpy as np
import torch

def generateGrid(N, fo=0, Nres=None, cent=None, nor=1, gpu=0, rGrid0=None):
    if Nres is None:
        Nres = N
    NDims = len(N)
    res = [N[i] / Nres[i] for i in range(NDims)]
    rGrid = []
    
    device = torch.device('cuda' if gpu > 0 else 'cpu')
    
    for m in range(NDims):
        if cent is None or len(cent) == 0:
            if fo == 0:
                if rGrid0 is None:
                    start = -math.floor(N[m]/2) + 0.5 * (res[m] - 1)
                    stop = math.ceil(N[m]/2) - 1
                    n_elems = int(round((stop - start) / res[m])) + 1
                    grid_val = start + np.arange(n_elems) * res[m]
                else:
                    r0_flat = rGrid0[m].reshape(-1)
                    start = r0_flat[0].item() + 0.5 * (res[m] - 1)
                    end_val = r0_flat[-1].item()
                    n_elems = int(round((end_val - start) / res[m])) + 1
                    grid_val = start + np.arange(n_elems) * res[m]
            else:
                grid_val = np.arange(-math.floor(Nres[m]/2), math.ceil(Nres[m]/2))
        else:
            if fo == 0:
                start = 1 + 0.5 * (res[m] - 1)
                n_elems = int(round((N[m] - start) / res[m])) + 1
                grid_val = (start + np.arange(n_elems) * res[m]) - cent[m]
            else:
                grid_val = np.arange(1, Nres[m] + 1) - cent[m]
        
        # Convert to PyTorch tensor
        grid_tensor = torch.tensor(grid_val, dtype=torch.float32, device=device)
        
        # Reshape to place elements in dimension m, and 1s elsewhere
        shape = [1] * NDims
        shape[m] = len(grid_tensor)
        grid_tensor = grid_tensor.view(*shape)
        
        if nor != 0 or fo != 0:
            grid_tensor = grid_tensor / N[m]
            
        if fo != 0:
            grid_tensor = fo * 2 * math.pi * grid_tensor
            
        rGrid.append(grid_tensor)
        
    return rGrid
