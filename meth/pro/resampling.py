import math
import numpy as np
import torch
from .mirroring import mirroring
from .fftGPU import fftGPU
from .ifftGPU import ifftGPU

def resampling(x, Nres, fo=0, gpu=0, mirror=None):
    N = list(x.shape)
    nDimsIn = len(N)
    
    Nres = list(Nres)
    while len(Nres) > 0 and Nres[-1] == 1:
        Nres.pop()
    nDimsOu = len(Nres)
    
    if mirror is None:
        mirror = [0] * nDimsOu
        
    if nDimsOu > nDimsIn:
        raise ValueError("Resampling dimensionality is larger than image dimensionality")
        
    Nor = N[:nDimsOu]
    mirror = mirror[:nDimsOu]
    
    NorM = [Nor[i] + mirror[i] * Nor[i] for i in range(nDimsOu)]
    NresM = [Nres[i] + mirror[i] * Nres[i] for i in range(nDimsOu)]
    Nmin = [min(NorM[i], NresM[i]) for i in range(nDimsOu)]
    Nmax = [max(NorM[i], NresM[i]) for i in range(nDimsOu)]
    
    zeroF = [math.ceil((Nmax[i] + 1) / 2) - 1 for i in range(nDimsOu)]
    orig = [zeroF[i] - math.ceil((Nmin[i] - 1) / 2) for i in range(nDimsOu)]
    fina = [zeroF[i] + math.floor((Nmin[i] - 1) / 2) for i in range(nDimsOu)]
    
    for m in range(nDimsOu):
        if Nor[m] != Nres[m]:
            NNres = Nres[:m] + [NresM[m]] + N[m+1:]
            if m != 0:
                perm = list(range(nDimsIn))
                perm[0] = m
                perm[m] = 0
                x = x.permute(*perm)
                NNres = [NNres[p] for p in perm]
            
            mirror_flags = [0] * len(x.shape)
            mirror_flags[0] = mirror[m]
            
            x = mirroring(x, mirror_flags, 1)
            
            if fo == 0:
                x = fftGPU(x, 1, gpu) / NorM[m]
            
            x = torch.fft.fftshift(x, dim=0)
            
            if Nor[m] > Nres[m]:
                xRes = x[orig[m] : fina[m] + 1]
            else:
                xRes = torch.zeros(NNres, dtype=x.dtype, device=x.device)
                xRes[orig[m] : fina[m] + 1] = x
                
            x = xRes
            x = torch.fft.ifftshift(x, dim=0)
            
            if fo == 0:
                x = ifftGPU(x, 1, gpu) * NresM[m]
                
            x = mirroring(x, mirror_flags, 0)
            
            if m != 0:
                x = x.permute(*perm)
                
    return x
