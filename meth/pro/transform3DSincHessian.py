import torch
from .fftGPU import fftGPU
from .ifftGPU import ifftGPU

def transform3DSincHessian(xB, GB, GC, et, etg, eth, mH, gpu=0, F=None, FH=None):
    def bc(factor, ref):
        """Append singleton dims to factor so it broadcasts against ref."""
        f = factor
        while f.ndim < ref.ndim:
            f = f.unsqueeze(-1)
        return f

    F_1 = F[0] if F is not None else None
    FH_1 = FH[0] if FH is not None else None
    F_2 = F[1] if F is not None else None
    FH_2 = FH[1] if FH is not None else None
    F_3 = F[2] if F is not None else None
    FH_3 = FH[2] if FH is not None else None
    
    G = None
    
    # Translation parameters
    if 1 <= mH <= 6:
        x_local = xB[3] * bc(eth[0][mH - 1], xB[3])
        for n in range(1, 4):
            FH_n = FH[n - 1] if FH is not None else None
            x_local = ifftGPU(x_local, n, gpu, FH_n)
        G = x_local
        
    # Translation-rotation parameters
    for n in range(1, 4):
        for m in range(1, 4):
            if (6 + 3 * (n - 1) + m) == mH:
                x_local = GB[n - 1] * bc(etg[0][m - 1], GB[n - 1])
                for o in range(1, 4):
                    FH_o = FH[o - 1] if FH is not None else None
                    x_local = ifftGPU(x_local, o, gpu, FH_o)
                G = x_local
                
    # Rotation cross terms
    # First-second
    if mH == 16:
        x_0 = GC[0] * bc(et[1][1], GC[0])
        x_1 = GC[0] * bc(etg[1][1], GC[0])
        
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 1, gpu, F_1)
        x_1 = ifftGPU(x_1, 3, gpu, FH_3)
        x_1 = fftGPU(x_1, 1, gpu, F_1)
        
        x_1 = bc(etg[2][1], x_0) * x_0 + bc(et[2][1], x_1) * x_1
        x_0 = bc(et[2][1], x_0) * x_0
        
        x_0 = ifftGPU(x_0, 1, gpu, FH_1)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_1 = ifftGPU(x_1, 1, gpu, FH_1)
        x_1 = fftGPU(x_1, 3, gpu, F_3)
        
        x_0 = bc(etg[1][1], x_0) * x_0 + bc(et[1][1], x_1) * x_1
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        x_0 = ifftGPU(x_0, 2, gpu, FH_2)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_0 = x_0 * bc(et[2][2], x_0)
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    # First-third
    if mH == 17:
        x_1 = GC[1] * bc(etg[1][2], GC[1])
        x_0 = GC[1] * bc(et[1][2], GC[1])
        
        x_0 = ifftGPU(x_0, 2, gpu, FH_2)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_1 = ifftGPU(x_1, 2, gpu, FH_2)
        x_1 = fftGPU(x_1, 3, gpu, F_3)
        
        x_1 = bc(etg[2][2], x_0) * x_0 + bc(et[2][2], x_1) * x_1
        x_0 = bc(et[2][2], x_0) * x_0
        
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_1 = ifftGPU(x_1, 3, gpu, FH_3)
        x_1 = fftGPU(x_1, 2, gpu, F_2)
        
        x_0 = bc(etg[1][2], x_0) * x_0 + bc(et[1][2], x_1) * x_1
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    # Second-third
    if mH == 18:
        x_1 = GC[2] * bc(etg[1][2], GC[2])
        x_0 = GC[2] * bc(et[1][2], GC[2])
        
        x_0 = ifftGPU(x_0, 2, gpu, FH_2)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_1 = ifftGPU(x_1, 2, gpu, FH_2)
        x_1 = fftGPU(x_1, 3, gpu, F_3)
        
        x_1 = bc(etg[2][2], x_0) * x_0 + bc(et[2][2], x_1) * x_1
        x_0 = bc(et[2][2], x_0) * x_0
        
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_1 = ifftGPU(x_1, 3, gpu, FH_3)
        x_1 = fftGPU(x_1, 2, gpu, F_2)
        
        x_0 = bc(etg[1][2], x_0) * x_0 + bc(et[1][2], x_1) * x_1
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    # Rotation second order
    # First rotation
    if mH == 19:
        x_0 = xB[0] * bc(et[1][0], xB[0])
        x_1 = xB[0] * bc(eth[1][0], xB[0])
        x_2 = xB[0] * bc(etg[1][0], xB[0])
        x_3 = xB[0] * bc(etg[1][0], xB[0])
        
        for m_idx in range(4):
            target = [x_0, x_1, x_2, x_3][m_idx]
            target = ifftGPU(target, 1, gpu, FH_1)
            target = fftGPU(target, 2, gpu, F_2)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            
        x_1 = bc(eth[2][0], x_0) * x_0 + bc(et[2][0], x_1) * x_1
        x_4 = bc(etg[2][0], x_0) * x_0 + bc(et[2][0], x_2) * x_2
        x_5 = bc(etg[2][0], x_0) * x_0 + bc(et[2][0], x_3) * x_3
        x_0 = bc(et[2][0], x_0) * x_0
        x_2 = bc(etg[2][0], x_2) * x_2
        x_3 = bc(etg[2][0], x_3) * x_3
        
        for m_idx in range(6):
            target = [x_0, x_1, x_2, x_3, x_4, x_5][m_idx]
            target = ifftGPU(target, 2, gpu, FH_2)
            target = fftGPU(target, 1, gpu, F_1)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            elif m_idx == 4: x_4 = target
            elif m_idx == 5: x_5 = target
            
        x_0 = bc(eth[1][0], x_0) * x_0 + bc(et[1][0], x_1) * x_1
        x_1 = bc(etg[1][0], x_4) * x_4 + bc(et[1][0], x_2) * x_2
        x_2 = bc(etg[1][0], x_5) * x_5 + bc(et[1][0], x_3) * x_3
        x_0 = x_0 + x_1 + x_2
        x_0 = ifftGPU(x_0, 1, gpu, FH_1)
        
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_0 = x_0 * bc(et[1][1], x_0)
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 1, gpu, F_1)
        x_0 = x_0 * bc(et[2][1], x_0)
        x_0 = ifftGPU(x_0, 1, gpu, FH_1)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_0 = x_0 * bc(et[1][1], x_0)
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        x_0 = ifftGPU(x_0, 2, gpu, FH_2)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_0 = x_0 * bc(et[2][2], x_0)
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    # Second rotation
    if mH == 20:
        x_0 = xB[1] * bc(et[1][1], xB[1])
        x_1 = xB[1] * bc(eth[1][1], xB[1])
        x_2 = xB[1] * bc(etg[1][1], xB[1])
        x_3 = xB[1] * bc(etg[1][1], xB[1])
        
        for m_idx in range(4):
            target = [x_0, x_1, x_2, x_3][m_idx]
            target = ifftGPU(target, 3, gpu, FH_3)
            target = fftGPU(target, 1, gpu, F_1)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            
        x_1 = bc(eth[2][1], x_0) * x_0 + bc(et[2][1], x_1) * x_1
        x_4 = bc(etg[2][1], x_0) * x_0 + bc(et[2][1], x_2) * x_2
        x_5 = bc(etg[2][1], x_0) * x_0 + bc(et[2][1], x_3) * x_3
        x_0 = bc(et[2][1], x_0) * x_0
        x_2 = bc(etg[2][1], x_2) * x_2
        x_3 = bc(etg[2][1], x_3) * x_3
        
        for m_idx in range(6):
            target = [x_0, x_1, x_2, x_3, x_4, x_5][m_idx]
            target = ifftGPU(target, 1, gpu, FH_1)
            target = fftGPU(target, 3, gpu, F_3)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            elif m_idx == 4: x_4 = target
            elif m_idx == 5: x_5 = target
            
        x_0 = bc(eth[1][1], x_0) * x_0 + bc(et[1][1], x_1) * x_1
        x_1 = bc(etg[1][1], x_4) * x_4 + bc(et[1][1], x_2) * x_2
        x_2 = bc(etg[1][1], x_5) * x_5 + bc(et[1][1], x_3) * x_3
        x_0 = x_0 + x_1 + x_2
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        x_0 = ifftGPU(x_0, 2, gpu, FH_2)
        x_0 = fftGPU(x_0, 3, gpu, F_3)
        x_0 = x_0 * bc(et[2][2], x_0)
        x_0 = ifftGPU(x_0, 3, gpu, FH_3)
        x_0 = fftGPU(x_0, 2, gpu, F_2)
        x_0 = x_0 * bc(et[1][2], x_0)
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    # Third rotation
    if mH == 21:
        x_0 = xB[2] * bc(et[1][2], xB[2])
        x_1 = xB[2] * bc(eth[1][2], xB[2])
        x_2 = xB[2] * bc(etg[1][2], xB[2])
        x_3 = xB[2] * bc(etg[1][2], xB[2])
        
        for m_idx in range(4):
            target = [x_0, x_1, x_2, x_3][m_idx]
            target = ifftGPU(target, 2, gpu, FH_2)
            target = fftGPU(target, 3, gpu, F_3)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            
        x_1 = bc(eth[2][2], x_0) * x_0 + bc(et[2][2], x_1) * x_1
        x_4 = bc(etg[2][2], x_0) * x_0 + bc(et[2][2], x_2) * x_2
        x_5 = bc(etg[2][2], x_0) * x_0 + bc(et[2][2], x_3) * x_3
        x_0 = bc(et[2][2], x_0) * x_0
        x_2 = bc(etg[2][2], x_2) * x_2
        x_3 = bc(etg[2][2], x_3) * x_3
        
        for m_idx in range(6):
            target = [x_0, x_1, x_2, x_3, x_4, x_5][m_idx]
            target = ifftGPU(target, 3, gpu, FH_3)
            target = fftGPU(target, 2, gpu, F_2)
            if m_idx == 0: x_0 = target
            elif m_idx == 1: x_1 = target
            elif m_idx == 2: x_2 = target
            elif m_idx == 3: x_3 = target
            elif m_idx == 4: x_4 = target
            elif m_idx == 5: x_5 = target
            
        x_0 = bc(eth[1][2], x_0) * x_0 + bc(et[1][2], x_1) * x_1
        x_1 = bc(etg[1][2], x_4) * x_4 + bc(et[1][2], x_2) * x_2
        x_2 = bc(etg[1][2], x_5) * x_5 + bc(et[1][2], x_3) * x_3
        x_0 = x_0 + x_1 + x_2
        
        for m in [1, 3]:
            F_m = F[m - 1] if F is not None else None
            x_0 = fftGPU(x_0, m, gpu, F_m)
        x_0 = x_0 * bc(et[0], x_0)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            x_0 = ifftGPU(x_0, m, gpu, FH_m)
        G = x_0
        
    return G
