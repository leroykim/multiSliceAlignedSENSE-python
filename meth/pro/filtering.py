from .fftGPU import fftGPU
from .ifftGPU import ifftGPU

def filtering(x, H, gpu=0):
    NH = H.shape
    nDimsH = len(NH)
    for m in range(nDimsH):
        if NH[m] != 1:
            x = fftGPU(x, m + 1, gpu)
            
    # Append singleton dimensions to H until its number of dimensions matches x
    H_broadcast = H
    while H_broadcast.ndim < x.ndim:
        H_broadcast = H_broadcast.unsqueeze(-1)
        
    x = x * H_broadcast
    
    for m in range(nDimsH):
        if NH[m] != 1:
            x = ifftGPU(x, m + 1, gpu)
    return x
