import math
import torch
from .pro.fftGPU import fftGPU
from .pro.ifftGPU import ifftGPU
from .pro.sense import sense
from .pro.isense import isense
from .pro.extractSlabs import extractSlabs
from .pro.filtering import filtering
from .pro.precomputeFactors3DTransform import precomputeFactors3DTransform
from .pro.transform3DSinc import transform3DSinc
from .pro.transform3DSincGradient import transform3DSincGradient
from .pro.transform3DSincHessian import transform3DSincHessian

def solveTMS2D(x, y, T, H, S, Ak, xkGrid, kkGrid, kGrid, nT, w, flagw, NzSlab, gpu, outlP, thplc, BlSz):
    NS = list(S.shape)
    NY = list(y.shape)
    over = (NS[0] - NY[0]) / 2.0
    FOV = [math.floor(over), math.ceil(over)]
    
    multA = 1.2
    multB = 2.0
    
    NT_shape = list(T.shape)
    
    a = [
        [0, 1, 2, 0, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 3, 4, 3, 4, 5],
        [0, 1, 2, 1, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 4, 5, 5, 3, 4, 5]
    ]
    NHe = 21
    
    Eprev = torch.zeros(NT_shape[0:6], dtype=torch.float32, device=T.device)
    E = torch.zeros(NT_shape[0:6], dtype=torch.float32, device=T.device)
    dHe = torch.zeros((NHe, NT_shape[4], NT_shape[5]), dtype=torch.float32, device=T.device)
    dH = torch.zeros((NT_shape[6], NT_shape[4], NT_shape[5]), dtype=torch.float32, device=T.device)
    
    # Masking
    MaskThreshold = 0.2
    xabs = torch.abs(x)
    meanBody = torch.mean(xabs).item()
    xabs_mask = (xabs > meanBody * MaskThreshold)
    x = x * xabs_mask
    
    NZR = list(x.shape)
    while len(NZR) < 5:
        NZR.append(1)
        
    yZR = [[None, None], [None, None]]
    yZR[1][1] = torch.zeros((NZR[0], NZR[1], NzSlab, NZR[3], NZR[4], NZR[2]), dtype=torch.complex64, device=x.device)
    yZR[0][1] = torch.zeros((NZR[0], NZR[1], NZR[2], NZR[3], NT_shape[4]), dtype=torch.complex64, device=x.device)
    
    x = extractSlabs(x, NzSlab, 1, 1, yZR)
    
    def encoding(x_in, s_idx, vS_idx):
        x_val = filtering(x_in, H, gpu)
        x_val = extractSlabs(x_val, NzSlab, 0, 1, yZR)
        S_broadcast = S
        while S_broadcast.ndim < x_val.ndim:
            S_broadcast = S_broadcast.unsqueeze(-1)
        x_val = x_val * S_broadcast
        x_val = sense(x_val, 1, NS[0], NY[0], FOV)
        x_val = fftGPU(x_val, 1, gpu)
        return x_val
        
    for n in range(nT):
        w = w.clone()
        w[flagw == 2] /= multA
        w[flagw == 1] *= multB
        
        # Iteration over blocks
        for s in range(0, NT_shape[4], BlSz):
            vS = slice(s, min(s + BlSz, NT_shape[4]))
            len_vS = vS.stop - vS.start
            T_slice = T[:, :, :, :, vS, :, :]
            et, etg, eth = precomputeFactors3DTransform(xkGrid, kkGrid, kGrid, T_slice, 1, 2, gpu)
            
            transformed_I, xB = transform3DSinc(x, et, 1, gpu)
            xT = encoding(transformed_I, s, vS)
            y_slice = y[:, :, :, :, vS]
            while y_slice.ndim < xT.ndim:
                y_slice = y_slice.unsqueeze(-1)
            xT = xT - y_slice
            
            Ak_slice = Ak[:, :, :, :, vS]
            while Ak_slice.ndim < xT.ndim:
                Ak_slice = Ak_slice.unsqueeze(-1)
            xT = xT * Ak_slice
            xTH = torch.conj(xT)
            
            prod_Eprev = torch.real(xT * xTH)
            sum_dims_Eprev = [d for d in range(prod_Eprev.ndim) if d not in (2, 4)]
            summed_Eprev = torch.sum(prod_Eprev, dim=sum_dims_Eprev)
            Eprev[:, :, :, :, vS, :] = summed_Eprev.t().view(1, 1, 1, 1, len_vS, NT_shape[5])
            
            G, GB, GC = transform3DSincGradient(xB, et, etg, 0, gpu)
            GH = [None] * 6
            for m in range(6):
                G[m] = encoding(G[m], s, vS)
                Ak_slice = Ak[:, :, :, :, vS]
                while Ak_slice.ndim < G[m].ndim:
                    Ak_slice = Ak_slice.unsqueeze(-1)
                G[m] = G[m] * Ak_slice
                GH[m] = torch.conj(G[m])
                
            for m in range(NHe):
                GG = transform3DSincHessian(xB, GB, GC, et, etg, eth, m + 1, gpu)
                GG = encoding(GG, s, vS)
                Ak_slice = Ak[:, :, :, :, vS]
                while Ak_slice.ndim < GG.ndim:
                    Ak_slice = Ak_slice.unsqueeze(-1)
                GG = GG * Ak_slice
                xTH_broadcast = xTH
                while xTH_broadcast.ndim < GG.ndim:
                    xTH_broadcast = xTH_broadcast.unsqueeze(-1)
                GG = torch.real(GG * xTH_broadcast)
                GG = GG + torch.real(G[a[0][m]] * GH[a[1][m]])
                
                sum_dims_GG = [d for d in range(GG.ndim) if d not in (2, 4)]
                summed_GG = torch.sum(GG, dim=sum_dims_GG)
                dHe[m, vS, :] = summed_GG.t()
                
            for m in range(6):
                xTH_broadcast = xTH
                while xTH_broadcast.ndim < G[m].ndim:
                    xTH_broadcast = xTH_broadcast.unsqueeze(-1)
                G_val = torch.real(G[m] * xTH_broadcast)
                sum_dims_G = [d for d in range(G_val.ndim) if d not in (2, 4)]
                summed_G = torch.sum(G_val, dim=sum_dims_G)
                dH[m, vS, :] = summed_G.t()
                
        # Batched Newton update step
        dHe_double = dHe.double()
        w_2d = w[0, 0, 0, 0].double()
        
        MHe = torch.eye(6, dtype=torch.float64, device=T.device).view(1, 1, 6, 6).repeat(NT_shape[4], NT_shape[5], 1, 1) * 1000.0
        
        for k in range(NHe):
            idx1 = a[0][k]
            idx2 = a[1][k]
            val = dHe_double[k]
            if idx1 == idx2:
                MHe[:, :, idx1, idx2] = val + w_2d
            else:
                MHe[:, :, idx1, idx2] = val
                MHe[:, :, idx2, idx1] = val
                
        dH_batched = dH.double().permute(1, 2, 0).unsqueeze(-1)
        sol_batched = torch.linalg.solve(MHe, dH_batched)
        dH = sol_batched.squeeze(-1).permute(2, 0, 1).float()
                
        if thplc == 0:
            dH[[0, 1, 3], :, :] = 0.0
        elif thplc == 1:
            dH[[2, 4, 5], :, :] = 0.0
            
        Tup_diff = dH.permute(1, 2, 0).view(1, 1, 1, 1, NT_shape[4], NT_shape[5], 6)
        Tup = T - Tup_diff
        
        for s in range(0, NT_shape[4], BlSz):
            vS = slice(s, min(s + BlSz, NT_shape[4]))
            len_vS = vS.stop - vS.start
            Tup_slice = Tup[:, :, :, :, vS, :, :]
            et, _, _ = precomputeFactors3DTransform(xkGrid, kkGrid, kGrid, Tup_slice, 1, 0, gpu)
            
            transformed_I, _ = transform3DSinc(x, et, 1, gpu)
            xT = encoding(transformed_I, s, vS)
            y_slice = y[:, :, :, :, vS]
            while y_slice.ndim < xT.ndim:
                y_slice = y_slice.unsqueeze(-1)
            xT = xT - y_slice
            
            Ak_slice = Ak[:, :, :, :, vS]
            while Ak_slice.ndim < xT.ndim:
                Ak_slice = Ak_slice.unsqueeze(-1)
            xT = xT * Ak_slice
            xTH = torch.conj(xT)
            
            prod_E = torch.real(xT * xTH)
            sum_dims_E = [d for d in range(prod_E.ndim) if d not in (2, 4)]
            summed_E = torch.sum(prod_E, dim=sum_dims_E)
            E[:, :, :, :, vS, :] = summed_E.t().view(1, 1, 1, 1, len_vS, NT_shape[5])
            
        flagw = torch.where(E < Eprev, torch.tensor(2, device=T.device), torch.tensor(1, device=T.device))
        T = torch.where((flagw == 2).unsqueeze(-1), Tup, T)
        
        Eaux = torch.minimum(E, Eprev)
        Eaux_roll_left = torch.roll(Eaux, shifts=-1, dims=5)
        Eaux_roll_right = torch.roll(Eaux, shifts=1, dims=5)
        Eaux = 2.0 * Eaux / (Eaux_roll_left + Eaux_roll_right)
        
        outlD = (Eaux < outlP).permute(0, 1, 5, 2, 4, 3)
        outlD = outlD.squeeze(-1)
        
    return T, w, flagw, outlD
