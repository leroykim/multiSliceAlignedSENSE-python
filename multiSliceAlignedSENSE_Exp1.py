import os
import time
import math
import numpy as np
import scipy.io as sio
import torch

from meth.pro.sliceProfile import sliceProfile
from meth.pro.resampling import resampling
from meth.pro.generateGrid import generateGrid
from meth.pro.filtering import filtering
from meth.optimizeLevelMS2D import optimizeLevelMS2D
from meth.pos.gibbsRingingFilter import gibbsRingingFilter
from meth.pos.rotateMPS import rotateMPS
from meth.pos.writeNIFTI import writeNIFTI

def main():
    pathOu = './data'
    gpu = 1
    debug = 1
    
    # Verify/Load T2 compressed data
    nameX = os.path.join(pathOu, 'yT2Comp.mat')
    if not os.path.exists(nameX):
        print(f"Data file not found: {nameX}. Skipping execution of the script, but saving it for completeness.")
        return
        
    print(f"Loading {nameX}...")
    mat = sio.loadmat(nameX)
    
    # Extract variables
    y_np = mat['y']
    S_np = mat['S']
    W_np = mat['W']
    Ak_np = mat['Ak']
    SlTh = float(mat['SlTh'][0, 0])
    SlOv = float(mat['SlOv'][0, 0])
    
    Encoding = mat['Encoding'][0, 0]
    Rot = mat['Rot']
    Phi = mat['Phi']
    
    device = torch.device('cuda' if gpu > 0 and torch.cuda.is_available() else 'cpu')
    
    # Convert inputs to PyTorch complex/float tensors
    y = torch.from_numpy(y_np).to(device=device, dtype=torch.complex64)
    S = torch.from_numpy(S_np).to(device=device, dtype=torch.complex64)
    W = torch.from_numpy(W_np).to(device=device, dtype=torch.float32)
    Ak = torch.from_numpy(Ak_np).to(device=device, dtype=torch.float32)
    
    class ParX:
        toler = 1e-6
        gibbsRing = [0.4, 0.4]
        alpha = [40.0, 20.0]
        threeD = 0
        outl = 0
        correct = 0
        thplc = 0
        outlP = 1.2
        gpu = gpu
        
    parX = ParX()
    
    Vpar_threeD = [0, 1, 1, 1, 1, 1]
    Vpar_outl = [0, 0, 0, 1, 1, 1]
    Vpar_correct = [0, 0, 1, 1, 1, 1]
    Vpar_thplc = [0, 0, 2, 0, 1, 2]
    
    NA = list(Ak.shape)
    NX = list(W.shape)
    
    T = torch.zeros((1, 1, 1, 1, NA[4], NX[2], 6), dtype=torch.float32, device=device)
    outlD = torch.ones((1, 1, NX[2], 1, NA[4]), dtype=torch.float32, device=device)
    x = torch.zeros(NX, dtype=torch.complex64, device=device)
    mNorm = torch.max(torch.abs(y)).item()
    
    NY = list(y.shape)
    
    suffix = ['NMC', 'NMC-SP', 'MC-NOU', 'MC-NWP', 'MC-NTP', 'MC']
    suffixFull = [
        'Conventional uncorrected SENSE (NMC)',
        'Uncorrected with slice profile filter (NMC-SP)',
        'Corrected without outlier rejection (MC-NOU)',
        'Corrected without within-plane motion (MC-NWP)',
        'Corrected without through-plane motion (MC-NTP)',
        'Fully corrected (MC)'
    ]
    
    for recType in range(len(Vpar_threeD)):
        print(f"\nReconstructing {suffixFull[recType]}...")
        t_start_rec = time.time()
        
        # Reset variables
        x.zero_()
        T.zero_()
        outlD.fill_(1.0)
        
        parX.threeD = Vpar_threeD[recType]
        parX.outl = Vpar_outl[recType]
        parX.correct = Vpar_correct[recType]
        parX.thplc = Vpar_thplc[recType]
        
        if parX.correct == 0:
            estT = [0]
        else:
            estT = [1, 0]
            
        if not parX.outl:
            parX.outlP = float('inf')
        else:
            parX.outlP = 1.2
            
        L = len(estT)
        H = sliceProfile(NX[2], SlTh, SlOv, parX)
        
        for l in range(L):
            res = 2 ** (L - 1 - l)
            NXres = [int(math.floor(NX[0] / res)), int(math.floor(NX[1] / res)), NX[2]]
            NYres = [int(math.floor(NY[0] / res)), int(math.floor(NY[1] / res)), NY[2]]
            
            xRes = resampling(x, NXres, 0, gpu)
            SRes = resampling(S, NXres, 0, gpu)
            yRes = resampling(y, NYres, 0, gpu)
            WRes = resampling(W, NXres, 0, gpu)
            WRes = (torch.abs(WRes) > 0.5).to(dtype=torch.float32)
            
            AkRes = resampling(Ak, [NYres[0]], 1, gpu)
            cent = [1.0, 1.0, 1.0]
            xGrid = generateGrid(NX, 0, NXres, cent, 0, gpu)
            
            xRes, T, outlD = optimizeLevelMS2D(
                xRes, yRes, T, SRes, WRes, AkRes, xGrid, mNorm, parX,
                debug, float(res), bool(estT[l]), gpu, SlTh, SlOv, outlD, parX.alpha[l]
            )
            
            x = W * resampling(xRes, NX, 0, gpu)
            
        xF = x.permute(1, 0, 2)
        WF = W.permute(1, 0, 2)
        
        xF = filtering(xF, H, gpu)
        NDims = 2
        
        # Gibbs ringing filter (convert xF back to CPU numpy for final post-processing, or keep on GPU)
        xF = gibbsRingingFilter(xF, NDims, parX.gibbsRing, gpu)
        
        # Zero filling
        XRes_enc = int(Encoding['XRes'][0, 0])
        YRes_enc = int(Encoding['YRes'][0, 0])
        KyOversampling = float(Encoding['KyOversampling'][0, 0])
        
        N_res = [XRes_enc, int(round(YRes_enc * KyOversampling))]
        WF_res = resampling(WF, N_res, 0, gpu)
        xF_res = (torch.abs(WF_res) > 0.5).to(dtype=torch.complex64) * resampling(xF, N_res, 0, gpu)
        
        # Remove oversampling
        pady = (xF_res.shape[1] - YRes_enc) / 2.0
        padya = int(math.floor(pady))
        padyb = int(math.ceil(pady))
        if padyb > 0:
            xF_res = xF_res[:, padya:-padyb, :]
        else:
            xF_res = xF_res[:, padya:, :]
        
        # Zero fill to reconstruction size
        XReconRes = int(Encoding['XReconRes'][0, 0])
        YReconRes = int(Encoding['YReconRes'][0, 0])
        
        padi_x = (XReconRes - XRes_enc) / 2.0
        padi_y = (YReconRes - YRes_enc) / 2.0
        padia_x = int(math.floor(padi_x))
        padib_x = int(math.ceil(padi_x))
        padia_y = int(math.floor(padi_y))
        padib_y = int(math.ceil(padi_y))
        
        target_shape = list(xF_res.shape)
        target_shape[0] += padia_x + padib_x
        target_shape[1] += padia_y + padib_y
        
        xF_padded = torch.zeros(target_shape, dtype=xF_res.dtype, device=xF_res.device)
        xF_padded[padia_x:padia_x+xF_res.shape[0], padia_y:padia_y+xF_res.shape[1], :] = xF_res
        
        # Rotate image
        Rot_3x3 = Rot[:3, :3]
        xF_final = rotateMPS(xF_padded, Rot_3x3)
        
        # Convert to numpy on CPU for writing
        xF_final_np = xF_final.cpu().numpy()
        
        # Write to NIfTI file
        writeNIFTI(xF_final_np, Phi, pathOu, 'xT2', suffix[recType])
        print(f"Time reconstructing {suffixFull[recType]}: {time.time() - t_start_rec:.2f}s")

if __name__ == '__main__':
    main()
