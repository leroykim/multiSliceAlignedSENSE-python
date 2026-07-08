import numpy as np
import torch

def rotateMPS(x, ARot):
    # ARot is a 3x3 rotation matrix (numpy array or list)
    if isinstance(ARot, torch.Tensor):
        ARot = ARot.detach().cpu().numpy()
    else:
        ARot = np.asarray(ARot)
    s = [int(np.nonzero(ARot[m, :])[0][0]) for m in range(3)]
    
    # In PyTorch or numpy, permute dimensions
    perm = s + list(range(3, x.ndim))
    if isinstance(x, torch.Tensor):
        x = x.permute(*perm)
    else:
        x = np.transpose(x, perm)
        
    v = [ARot[m, s[m]] for m in range(3)]
    
    if isinstance(x, torch.Tensor):
        if v[0] < 0:
            x = torch.flip(x, dims=[0])
        if v[1] < 0:
            x = torch.flip(x, dims=[1])
        if v[2] < 0:
            x = torch.flip(x, dims=[2])
    else:
        if v[0] < 0:
            x = np.flip(x, axis=0)
        if v[1] < 0:
            x = np.flip(x, axis=1)
        if v[2] < 0:
            x = np.flip(x, axis=2)
            
    return x
