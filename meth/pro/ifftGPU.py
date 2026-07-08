import torch

def ifftGPU(x, m, gpu=0, FH=None):
    dim = m - 1
    if not torch.is_complex(x):
        x = x.to(dtype=torch.complex64)
    return torch.fft.ifft(x, dim=dim)
