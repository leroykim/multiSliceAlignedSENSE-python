import numpy as np
import torch

def computeROI(W, ext, strict=None):
    if isinstance(W, torch.Tensor):
        W_np = W.cpu().numpy()
    else:
        W_np = np.asarray(W)
        
    ND = W_np.ndim
    if strict is None:
        strict = [0] * ND
        
    N = W_np.shape
    ROI = np.zeros((ND, 6), dtype=np.int64)
    
    nonzero_coords = np.nonzero(W_np != 0)
    for nd in range(ND):
        if len(nonzero_coords[nd]) == 0:
            inf_lim = 0
            sup_lim = N[nd] - 1
        else:
            inf_lim = np.min(nonzero_coords[nd])
            sup_lim = np.max(nonzero_coords[nd])
            
        inf_lim = max(inf_lim - ext, 0)
        sup_lim = min(sup_lim + ext, N[nd] - 1)
        
        ROI[nd, 0] = inf_lim
        ROI[nd, 1] = sup_lim
        ROI[nd, 2] = N[nd]
        ROI[nd, 3] = sup_lim - inf_lim + 1
        ROI[nd, 4] = inf_lim
        ROI[nd, 5] = N[nd] - 1 - sup_lim
        
        if strict[nd]:
            left = ROI[nd, 4]
            right = ROI[nd, 5]
            if left < right:
                ROI[nd, 3] = ROI[nd, 3] + right - left
                ROI[nd, 1] = ROI[nd, 1] + right - left
                ROI[nd, 5] = left
            else:
                ROI[nd, 3] = ROI[nd, 3] + left - right
                ROI[nd, 0] = ROI[nd, 0] - (left - right)
                ROI[nd, 4] = right
                
    return ROI
