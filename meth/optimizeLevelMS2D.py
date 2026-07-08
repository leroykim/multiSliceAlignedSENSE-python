import math
import torch
import numpy as np
from .pro.computeROI import computeROI
from .pro.extractROI import extractROI
from .pro.extractSlabs import extractSlabs
from .pro.sliceProfile import sliceProfile
from .pro.fftGPU import fftGPU
from .solveXMS2D import solveXMS2D
from .solveTMS2D import solveTMS2D

def optimizeLevelMS2D(x, y, T, S, W, Ak, xGrid, mNorm, parX, debug=0, res=1.0, estT=True, gpu=0, SlTh=1.0, SlOv=0.0, outlD=None, alpha=40.0):
    reg = 0.001
    ROIsec = 0
    ROI = computeROI(W, ROIsec)
    
    threeD = getattr(parX, 'threeD', 1)
    correct = getattr(parX, 'correct', 1)
    
    if threeD > 0:
        NzSlab = 7 if correct != 0 else 3
    else:
        NzSlab = 1
        
    H = sliceProfile(NzSlab, SlTh, SlOv, parX)
    device = x.device
    H = H.to(device)
    Nz = W.shape[2]
    Ha = torch.zeros(Nz, dtype=torch.complex64, device=device)
    Hb = torch.zeros(Nz, dtype=torch.complex64, device=device)
    
    if threeD == 0:
        alpha = 0.0
        
    H[0, 0, 0] = 1.0
    Ha[0] = 1.0
    Ha[1] = -1.0
    Hb[0] = 1.0
    Hb[-1] = -1.0
    
    Ha_fft = torch.fft.fft(Ha)
    Hb_fft = torch.fft.fft(Hb)
    Hsmooth = alpha * (torch.conj(Ha_fft) * Ha_fft + torch.conj(Hb_fft) * Hb_fft)
    Hsmooth = Hsmooth.view(1, 1, Nz)
    
    y = extractROI(y, ROI, 2, 1)
    S = extractROI(S, ROI, 2, 1)
    W = extractROI(W, ROI, 2, 1)
    x = extractROI(x, ROI, 2, 1)
    Ak = extractROI(Ak, ROI, 2, 1)
    xGrid_2 = extractROI(xGrid[1], ROI, 2, 1)
    
    N = list(W.shape)
    N[2] = NzSlab
    
    kGrid = [None] * 3
    
    kGrid_val0 = np.arange(-math.floor(N[0]/2), math.ceil(N[0]/2)) / res
    kGrid[0] = torch.tensor(kGrid_val0, dtype=torch.float32, device=device).view(-1, 1, 1)
    
    kGrid_val1 = np.arange(-math.floor(N[1]/2), math.ceil(N[1]/2)) / res
    kGrid[1] = torch.tensor(kGrid_val1, dtype=torch.float32, device=device).view(1, -1, 1)
    
    kGrid_val2 = np.arange(-math.floor(N[2]/2), math.ceil(N[2]/2)) / res
    kGrid[2] = torch.tensor(kGrid_val2, dtype=torch.float32, device=device).view(1, 1, -1)
    
    for m in range(3):
        kGrid[m] = 2.0 * math.pi * kGrid[m] / N[m]
        
    yZ = [[None, None], [None, None]]
    yZ[1][1] = torch.zeros((1, 1, N[2], 1, 1, xGrid[2].shape[2]), dtype=torch.float32, device=device)
    xGrid_3 = extractSlabs(xGrid[2], N[2], 1, 1, yZ)
    
    per_mat = [
        [0, 2, 1],
        [1, 0, 2]
    ]
    xkGrid = [[None]*3 for _ in range(2)]
    xGrid_all = [xGrid[0], xGrid_2, xGrid_3]
    for n in range(2):
        for m in range(3):
            idx_x = per_mat[1 - n][m]
            idx_k = per_mat[n][m]
            k_tensor = kGrid[idx_k]
            while k_tensor.ndim < xGrid_all[idx_x].ndim:
                k_tensor = k_tensor.unsqueeze(-1)
            xkGrid[n][m] = xGrid_all[idx_x] * k_tensor
            
    fact = [
        [0, 1, 2, 0, 0, 1],
        [0, 1, 2, 1, 2, 2]
    ]
    kkGrid = [None] * 6
    for m in range(6):
        kkGrid[m] = kGrid[fact[0][m]] * kGrid[fact[1][m]]
        
    y = y / mNorm
    x = x / mNorm
    
    winic = 1.0
    NT_shape = list(T.shape)
    w = torch.ones(NT_shape[0:6], dtype=torch.float32, device=device) * winic
    flagw = torch.zeros(NT_shape[0:6], dtype=torch.long, device=device)
    
    nExtern = 1000
    nX = 5
    ndim_t = 1
    
    SH = torch.conj(S)
    Precond = torch.sum(torch.real(SH * S), dim=3)
    Precond = 1.0 / (Precond + reg)
    
    y = fftGPU(y, 1, gpu)
    
    for n in range(1, nExtern + 1):
        xant = x.clone()
        
        BlSz_arr = [0, 0]
        if res == 1 and torch.sum(torch.abs(T)) != 0:
            BlSz_arr[0] = 1
            BlSz_arr[1] = math.ceil(NT_shape[5] / 2.0)
        else:
            BlSz_arr[0] = math.floor(NT_shape[4] / 2.0)
            BlSz_arr[1] = NT_shape[5]
            if BlSz_arr[0] == 0:
                BlSz_arr[0] = 1
                
        if correct > 0 and estT:
            x = solveXMS2D(x, y, W, T, H, Hsmooth, S, SH, Precond, Ak, xkGrid, kGrid, nX, 0.0, NzSlab, gpu, outlD, BlSz_arr)
        else:
            x = solveXMS2D(x, y, W, T, H, Hsmooth, S, SH, Precond, Ak, xkGrid, kGrid, float('inf'), getattr(parX, 'toler', 1e-6), NzSlab, gpu, outlD, BlSz_arr)
            
        if correct > 0 and estT:
            BlSz_T = 1
            T, w, flagw, outlD = solveTMS2D(x, y, T, H, S, Ak, xkGrid, kkGrid, kGrid, ndim_t, w, flagw, NzSlab, gpu, getattr(parX, 'outlP', 1.2), getattr(parX, 'thplc', 2), BlSz_T)
            
            diff = x - xant
            diff_val = torch.real(diff * torch.conj(diff))
            err_val = torch.max(diff_val).item()
            if debug > 0:
                print(f"  -> Alt-Min Iteration {n:03d}/{nExtern} - Max Diff Error: {err_val:.2e}")
            if err_val < getattr(parX, 'toler', 1e-6):
                if debug > 0:
                    print(f"  -> Converged at iteration {n:03d} (Error: {err_val:.2e} < toler)")
                break
        else:
            break
            
    x = x * mNorm
    x = extractROI(x, ROI, 2, 0)
    
    return x, T, outlD
