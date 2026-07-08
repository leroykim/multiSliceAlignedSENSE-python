import math
import numpy as np
import torch
from .fftGPU import fftGPU

def sliceProfile(N, SlTh, SlOv, parX):
    H = np.zeros(N, dtype=np.float32)
    SlSp = SlTh - SlOv
    H[0] = 1.0
    
    if SlSp > SlTh:
        print(f"Slice thickness: {SlTh:6.3f}")
        print(f"Slice gap: {SlSp:6.3f}")
        raise ValueError("The slice overlap pattern has not been implemented in the reconstruction")
        
    if getattr(parX, 'threeD', 0) > 0:
        H[0] = SlSp
        SlRes = SlOv
        inda = 1
        indb = 0
        while SlRes > 0:
            if SlRes - 2 * SlSp > 0:
                H[inda] = SlSp
                H[-1 - indb] = SlSp
                inda += 1
                indb += 1
            else:
                H[inda] = SlRes / 2
                H[-1 - indb] = SlRes / 2
            SlRes = SlRes - 2 * SlSp
        H = H / SlTh
        
    device = torch.device('cuda' if getattr(parX, 'gpu', 0) > 0 else 'cpu')
    H_tensor = torch.tensor(H, dtype=torch.complex64, device=device).view(1, -1)
    H_tensor = fftGPU(H_tensor, 2, 0)
    H_tensor = H_tensor.view(1, N, 1).permute(0, 2, 1).contiguous()
    
    return H_tensor
