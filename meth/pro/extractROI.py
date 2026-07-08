import torch
import numpy as np

def extractROI(x, ROI, dim, ext):
    # dim is 1-based index (1, 2, or 3)
    d = dim - 1
    
    if ext == 1:
        start = ROI[d, 0]
        end = ROI[d, 1] + 1
        if d == 0:
            return x[start:end, ...]
        elif d == 1:
            return x[:, start:end, ...]
        elif d == 2:
            return x[:, :, start:end, ...]
        else:
            # General case if more dimensions
            slices = [slice(None)] * x.ndim
            slices[d] = slice(start, end)
            return x[tuple(slices)]
    else:
        # Create output tensor of shape with original size along dimension d
        target_shape = list(x.shape)
        target_shape[d] = int(ROI[d, 2])
        
        if isinstance(x, torch.Tensor):
            y = torch.zeros(target_shape, dtype=x.dtype, device=x.device)
        else:
            y = np.zeros(target_shape, dtype=x.dtype)
            
        start = ROI[d, 0]
        end = ROI[d, 1] + 1
        
        if d == 0:
            y[start:end, ...] = x
        elif d == 1:
            y[:, start:end, ...] = x
        elif d == 2:
            y[:, :, start:end, ...] = x
        else:
            slices = [slice(None)] * y.ndim
            slices[d] = slice(start, end)
            y[tuple(slices)] = x
            
        return y
