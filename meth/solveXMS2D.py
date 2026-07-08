import math
import torch
from .pro.fftGPU import fftGPU
from .pro.ifftGPU import ifftGPU
from .pro.isense import isense
from .pro.sense import sense
from .pro.extractSlabs import extractSlabs
from .pro.filtering import filtering
from .pro.precomputeFactors3DTransform import precomputeFactors3DTransform
from .pro.transform3DSinc import transform3DSinc

def solveXMS2D(x, y, W, T, H, Hsmooth, S, SH, Precond, Ak, xkGrid, kGrid, nX, toler, NzSlab, gpu, outlD, BlSz):
    N = list(x.shape)
    NT_shape = list(T.shape)
    
    NRun = [0, 0]
    NRunRem = [0, 0]
    vS = [[], []]
    for p in range(2):
        dim_size = NT_shape[4 + p]
        block_size = BlSz[p]
        NRun[p] = math.ceil(dim_size / block_size)
        NRunRem[p] = dim_size % block_size
        for s in range(NRun[p]):
            if s != NRun[p] - 1 or NRunRem[p] == 0:
                vS[p].append(slice(s * block_size, (s + 1) * block_size))
            else:
                vS[p].append(slice(s * block_size, s * block_size + NRunRem[p]))
                
    if torch.sum(torch.abs(T)) != 0:
        etDir = [[None] * NRun[1] for _ in range(NRun[0])]
        etInv = [[None] * NRun[1] for _ in range(NRun[0])]
        for s in range(NRun[0]):
            for t in range(NRun[1]):
                xkGridAux = [
                    [xkGrid[0][0], xkGrid[0][1], xkGrid[0][2]],
                    [xkGrid[1][0], xkGrid[1][1], xkGrid[1][2]]
                ]
                xkGridAux[0][2] = xkGrid[0][2][:, :, :, :, :, vS[1][t]]
                xkGridAux[1][1] = xkGrid[1][1][:, :, :, :, :, vS[1][t]]
                
                T_slice = T[:, :, :, :, vS[0][s], vS[1][t], :]
                etDir[s][t], _, _ = precomputeFactors3DTransform(xkGridAux, [], kGrid, T_slice, 1, 0, gpu)
                etInv[s][t], _, _ = precomputeFactors3DTransform(xkGridAux, [], kGrid, T_slice, 0, 0, gpu)
                
    NS = list(S.shape)
    NY = list(y.shape)
    over = (NS[0] - NY[0]) / 2.0
    disc = (3.0 * NY[0] - NS[0]) / 2.0
    FOV = [math.floor(over), math.ceil(over)]
    iFOV = [math.floor(disc), math.ceil(disc)]
    
    AkOutlDisc = Ak * outlD
    
    yEnd_shape = (N[0], N[1], NzSlab, 1, 1, N[2])
    yEnd = torch.zeros(yEnd_shape, dtype=torch.complex64, device=x.device)
    
    yZR = [[None, None], [None, None]]
    
    for s in range(NRun[0]):
        if torch.sum(torch.abs(T)) != 0:
            yS = y * outlD[:, :, :, :, vS[0][s]]
            Ak_slice = Ak[:, :, :, :, vS[0][s]]
            while Ak_slice.ndim < yS.ndim:
                Ak_slice = Ak_slice.unsqueeze(-1)
            yS = yS * Ak_slice
        else:
            yS = y
            
        yS = ifftGPU(yS, 1, gpu)
        yS = isense(yS, 1, NS[0], NY[0], iFOV)
        SH_broadcast = SH
        while SH_broadcast.ndim < yS.ndim:
            SH_broadcast = SH_broadcast.unsqueeze(-1)
        yS = torch.sum(yS * SH_broadcast, dim=3, keepdim=True)
        
        NZR = list(yS.shape)
        while len(NZR) < 5:
            NZR.append(1)
            
        if s == 0:
            yZR[0][1] = torch.zeros(NZR, dtype=torch.complex64, device=x.device)
            yZR[1][0] = torch.zeros((NZR[0], NZR[1], NzSlab, NZR[3], NZR[4], NZR[2]), dtype=torch.complex64, device=x.device)
            
        yS = extractSlabs(yS, NzSlab, 1, 0, yZR)
        yS = filtering(yS, H, gpu)
        
        for t in range(NRun[1]):
            if torch.sum(torch.abs(T)) != 0:
                etS = etInv[s][t]
                yS_slice = yS[:, :, :, :, :, vS[1][t]]
                transformed, _ = transform3DSinc(yS_slice, etS, 0, gpu)
                yEnd[:, :, :, :, :, vS[1][t]] += torch.sum(transformed, dim=4, keepdim=True)
            else:
                yS_slice = yS[:, :, :, :, :, vS[1][t]]
                yEnd[:, :, :, :, :, vS[1][t]] += torch.sum(yS_slice, dim=4, keepdim=True)
                
    y_final = yEnd
    NZR = list(y_final.shape)
    while len(NZR) < 6:
        NZR.append(1)
    yZR[0][0] = torch.zeros((NZR[0], NZR[1], NZR[5]), dtype=torch.complex64, device=x.device)
    yZR[1][1] = torch.zeros(NZR, dtype=torch.complex64, device=x.device)
    
    y_final = extractSlabs(y_final, NzSlab, 0, 0, yZR)
    y_final = W * y_final
    
    def applyCG(x_in):
        xB = filtering(x_in, Hsmooth, gpu)
        x_curr = extractSlabs(x_in, NzSlab, 1, 1, yZR)
        NX_local = list(x_curr.shape)
        while len(NX_local) < 6:
            NX_local.append(1)
            
        xEnd = torch.zeros(NX_local, dtype=torch.complex64, device=x_in.device)
        
        for s in range(NRun[0]):
            if torch.sum(torch.abs(T)) != 0:
                xS_shape = NX_local[:4] + [vS[0][s].stop - vS[0][s].start, NX_local[5]]
                xS = torch.zeros(xS_shape, dtype=torch.complex64, device=x_in.device)
            else:
                xS = torch.zeros(NX_local, dtype=torch.complex64, device=x_in.device)
                
            for t in range(NRun[1]):
                if torch.sum(torch.abs(T)) != 0:
                    etS = etDir[s][t]
                    x_slice = x_curr[:, :, :, :, :, vS[1][t]]
                    transformed, _ = transform3DSinc(x_slice, etS, 1, gpu)
                    xS[:, :, :, :, :, vS[1][t]] = transformed
                else:
                    xS[:, :, :, :, :, vS[1][t]] = x_curr[:, :, :, :, :, vS[1][t]]
                    
            xS = filtering(xS, H, gpu)
            xS = extractSlabs(xS, NzSlab, 0, 1, yZR)
            S_broadcast = S
            while S_broadcast.ndim < xS.ndim:
                S_broadcast = S_broadcast.unsqueeze(-1)
            xS = xS * S_broadcast
            xS = sense(xS, 1, NS[0], NY[0], FOV)
            if torch.sum(torch.abs(T)) != 0:
                xS = fftGPU(xS, 1, gpu)
                Ak_slice = AkOutlDisc[:, :, :, :, vS[0][s]]
                while Ak_slice.ndim < xS.ndim:
                    Ak_slice = Ak_slice.unsqueeze(-1)
                xS = xS * Ak_slice
                xS = ifftGPU(xS, 1, gpu)
                
            SH_broadcast = SH
            while SH_broadcast.ndim < xS.ndim:
                SH_broadcast = SH_broadcast.unsqueeze(-1)
            xS = torch.sum(xS * SH_broadcast, dim=3, keepdim=True)
            xS = extractSlabs(xS, NzSlab, 1, 0, yZR)
            xS = filtering(xS, H, gpu)
            
            for t in range(NRun[1]):
                if torch.sum(torch.abs(T)) != 0:
                    etS = etInv[s][t]
                    xS_slice = xS[:, :, :, :, :, vS[1][t]]
                    transformed, _ = transform3DSinc(xS_slice, etS, 0, gpu)
                    xEnd[:, :, :, :, :, vS[1][t]] += torch.sum(transformed, dim=4, keepdim=True)
                else:
                    xS_slice = xS[:, :, :, :, :, vS[1][t]]
                    xEnd[:, :, :, :, :, vS[1][t]] += torch.sum(xS_slice, dim=4, keepdim=True)
                    
        x_out = xEnd
        x_out = extractSlabs(x_out, NzSlab, 0, 0, yZR)
        x_out = x_out + xB
        x_out = x_out * W
        return x_out

    Ap = applyCG(x)
    r = y_final - Ap
    z = Precond * r
    p = z
    rsold = torch.sum(torch.conj(z) * r)
    
    n_iter = 1
    while True:
        Ap = applyCG(p)
        denom = torch.sum(torch.conj(p) * Ap)
        al = torch.conj(rsold) / denom
        xup = al * p
        x = x + xup
        
        xup_val = torch.real(xup * torch.conj(xup))
        xup_max = torch.max(xup_val).item()
        
        if xup_max < toler or n_iter >= nX:
            if toler != 0:
                print(f"Iteration CG {n_iter:04d} - Error {xup_max:.2g}")
            break
            
        r = r - al * Ap
        z = Precond * r
        rsnew = torch.sum(torch.conj(z) * r)
        be = rsnew / rsold
        p = z + be * p
        rsold = rsnew
        
        if torch.sqrt(torch.abs(rsnew)) < 1e-10:
            break
        n_iter += 1
        
    return x
