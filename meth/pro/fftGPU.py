import torch

def fftGPU(x, m, gpu=0, F=None):
    dim = m - 1
    if not torch.is_complex(x):
        x = x.to(dtype=torch.complex64)
    return torch.fft.fft(x, dim=dim)
