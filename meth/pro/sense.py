import math
import torch

def sense(x, m, NS, NY, FOV):
    if NS == NY:
        return x
        
    oddFactSENSE = 2 * math.ceil((math.ceil(NS / NY) - 1) / 2) + 1
    oFRed = (oddFactSENSE - 3) // 2
    oFRedT = oFRed * NY
    
    nDimsIn = len(x.shape)
    dim = m - 1
    
    if dim != 0:
        perm = list(range(nDimsIn))
        perm[0] = dim
        perm[dim] = 0
        x = x.permute(*perm)
        
    N = list(x.shape)
    x = x.reshape(N[0], -1)
    
    xb = x[FOV[1] : NS - FOV[0], :].clone()
    
    for s in range(1, oFRed + 1):
        idx1_start = FOV[1] - s * NY
        idx1_end = NS - FOV[0] - s * NY
        idx2_start = FOV[1] + s * NY
        idx2_end = NS - FOV[0] + s * NY
        xb += x[idx1_start : idx1_end, :] + x[idx2_start : idx2_end, :]
        
    end_idx1 = FOV[0] - oFRedT
    start_idx2 = NS - FOV[0] + oFRedT
    xb[0 : end_idx1, :] += x[start_idx2 : NS, :]
    
    start_idx1 = NY - FOV[1] + oFRedT
    end_idx2 = FOV[1] - oFRedT
    xb[start_idx1 : NY, :] += x[0 : end_idx2, :]
    
    new_shape = [NY] + N[1:]
    xb = xb.reshape(*new_shape)
    
    if dim != 0:
        xb = xb.permute(*perm)
        
    return xb
