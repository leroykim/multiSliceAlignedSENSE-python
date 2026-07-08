import torch
import numpy as np

def extractSlabs(x, NT, ext, direc, yZR=None):
    N = list(x.shape)
    
    # Pad N to have at least 5 dimensions for direct indexing/mapping
    while len(N) < 5:
        N.append(1)
        
    NTh = NT // 2
    if ext == 1:
        NSl = N[2]
    else:
        # NSl is the size of the 6th dimension (if it exists)
        NSl = N[5] if len(N) > 5 else 1
        
    # Pad N to at least 6 dimensions for safety in reshape
    while len(N) < 6:
        N.append(1)
        
    x_reshaped = x.view(*N)
    
    # Compute range index using modulo for wrapping
    range_idx = np.zeros((NSl, NT), dtype=np.int64)
    for s in range(NSl):
        r = np.arange(s - NTh, s + NTh + 1)
        r = r % NSl
        range_idx[s, :] = r
        
    range_idx = torch.tensor(range_idx, dtype=torch.long, device=x.device)
    
    if ext == 1:
        if direc == 1:
            # y(:,:,:,:,:,s) = x(:,:,range(s,:))
            # Output y shape should be (N[0], N[1], NT, N[3], N[4], NSl)
            y = torch.zeros((N[0], N[1], NT, N[3], N[4], NSl), dtype=x.dtype, device=x.device)
            for s in range(NSl):
                y[:, :, :, :, :, s] = x_reshaped[:, :, range_idx[s]].reshape(N[0], N[1], NT, N[3], N[4])
        else:
            # y(:,:,NTh+1,:,:,:) = permute(x,[1 2 6 4 5 3])
            # Output y shape is (N[0], N[1], NT, N[3], N[4], N[5])
            if yZR is not None and len(yZR) > ext and yZR[ext] is not None and len(yZR[ext]) > direc and yZR[ext][direc] is not None:
                y = yZR[ext][direc]
            else:
                y = torch.zeros((N[0], N[1], NT, N[3], N[4], N[5]), dtype=x.dtype, device=x.device)
            # Use slice NTh:NTh+1 to preserve the dimension of size 1, dynamically appending extra dimensions for permute
            perm_vector = [0, 1, 5, 3, 4, 2] + list(range(6, x_reshaped.ndim))
            y[:, :, NTh : NTh + 1, :, :, :] = x_reshaped.permute(*perm_vector)
    else:
        if direc == 1:
            # y = permute(x(:,:,NTh+1,:,:,:),[1 2 6 4 5 3])
            # x_reshaped has shape (N0, N1, NT, N3, N4, NSl)
            # Use slice NTh:NTh+1 and dynamically append extra dimensions for permute
            perm_vector = [0, 1, 5, 3, 4, 2] + list(range(6, x_reshaped.ndim))
            y = x_reshaped[:, :, NTh : NTh + 1, :, :, :].permute(*perm_vector).contiguous()
        else:
            # y(:,:,range(s,:)) = y(:,:,range(s,:)) + x(:,:,:,:,:,s)
            # Output y shape is (N[0], N[1], NSl) since ext=0, direc=0
            if yZR is not None and len(yZR) > ext and yZR[ext] is not None and len(yZR[ext]) > direc and yZR[ext][direc] is not None:
                y = yZR[ext][direc]
                y.zero_()
            else:
                y = torch.zeros((N[0], N[1], NSl), dtype=x.dtype, device=x.device)
            for s in range(NSl):
                y[:, :, range_idx[s]] += x_reshaped[:, :, :, :, :, s].reshape(N[0], N[1], NT)
                
    return y
