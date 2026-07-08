import math
import numpy as np
import torch
from ..pro.generateGrid import generateGrid
from ..pro.filtering import filtering

def tukeywin(N, r=0.5):
    if r <= 0:
        return np.ones(N, dtype=np.float32)
    elif r >= 1:
        return np.hanning(N).astype(np.float32)
    
    x = np.linspace(0, 1, N)
    w = np.ones(N, dtype=np.float32)
    idx1 = x < r / 2.0
    w[idx1] = 0.5 * (1.0 + np.cos(2.0 * np.pi / r * (x[idx1] - r / 2.0)))
    idx2 = x > 1.0 - r / 2.0
    w[idx2] = 0.5 * (1.0 + np.cos(2.0 * np.pi / r * (x[idx2] - 1.0 + r / 2.0)))
    return w

def gibbsRingingFilter(x, NDims, gibbsRing, gpu=0):
    N = list(x.shape)
    # Convert gibbsRing to list if it's a scalar or numpy array
    if isinstance(gibbsRing, (int, float)):
        gibbsRing = [gibbsRing]
    elif isinstance(gibbsRing, np.ndarray):
        gibbsRing = gibbsRing.tolist()
        
    NGR = len(gibbsRing)
    if NGR == 1:
        gibbs_val = gibbsRing[0]
        NNDims = N[:NDims]
        device = torch.device('cuda' if gpu > 0 else 'cpu')
        
        kGrid = generateGrid(NNDims, 1, NNDims, None, 1, gpu)
        # Compute radial coordinates using PyTorch broadcasting
        kkrad = torch.zeros(NNDims, dtype=torch.float32, device=device)
        for m in range(NDims):
            kkrad += kGrid[m] ** 2
        kkrad = torch.sqrt(kkrad)
        
        tuk = torch.ones(NNDims, dtype=torch.float32, device=device)
        alpha = 1.0 - gibbs_val
        if gibbs_val != 0:
            fkk = 0.5 * (1.0 + torch.cos(math.pi * ((kkrad - math.pi * alpha) / ((1.0 - alpha) * math.pi))))
            mask = (kkrad >= math.pi * alpha)
            tuk = torch.where(mask, fkk, tuk)
        tuk[kkrad >= math.pi] = 0.0
        
        # If x has more dimensions than NDims, we need to match dimensions
        # e.g., if x has shape (N1, N2, N3, N4) and tuk has shape (N1, N2, N3),
        # we can view tuk as (N1, N2, N3, 1)
        while tuk.ndim < x.ndim:
            tuk = tuk.unsqueeze(-1)
            
        tuk = torch.fft.ifftshift(tuk)
        x = filtering(x, tuk, gpu)
    else:
        NDims = min(NDims, NGR)
        for m in range(NDims):
            n_m = N[m]
            tuk_len = n_m + 1 - (n_m % 2)
            tuk = tukeywin(tuk_len, gibbsRing[m])
            tuk = tuk[:n_m]
            
            shape = [1] * len(x.shape)
            shape[m] = n_m
            tuk_tensor = torch.tensor(tuk, dtype=torch.complex64 if torch.is_complex(x) else torch.float32, device=x.device).view(*shape)
            tuk_tensor = torch.fft.ifftshift(tuk_tensor, dim=m)
            x = filtering(x, tuk_tensor, gpu)
            
    return x
