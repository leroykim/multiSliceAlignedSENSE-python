import math
import torch

def isense(x, m, NS, NY, FOV):
    if NS == NY:
        return x
        
    oddFactSENSE = 2 * math.ceil((math.ceil(NS / NY) - 1) / 2) + 1
    oFRed = oddFactSENSE - 2
    
    nDimsIn = len(x.shape)
    dim = m - 1
    
    if dim != 0:
        perm = list(range(nDimsIn))
        perm[0] = dim
        perm[dim] = 0
        x = x.permute(*perm)
        
    NX = list(x.shape)
    x = x.reshape(NX[0], -1)
    
    indUnf = []
    indUnf.extend(range(FOV[0], NX[0]))
    for _ in range(oFRed):
        indUnf.extend(range(NX[0]))
    indUnf.extend(range(NX[0] - FOV[1]))
    
    indUnf = torch.tensor(indUnf, dtype=torch.long, device=x.device)
    x = x[indUnf, :]
    
    new_shape = [NS] + NX[1:]
    x = x.reshape(*new_shape)
    
    if dim != 0:
        x = x.permute(*perm)
        
    return x
