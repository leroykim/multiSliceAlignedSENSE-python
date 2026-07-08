import os
import numpy as np
import nibabel as nib

def writeNIFTI(x, Phi, pathO, ima, suffix=""):
    MT = np.asarray(Phi, dtype=np.float32).copy()
    MS = np.sqrt(np.sum(MT[0:3, 0:3] ** 2, axis=0))
    MTT = np.array([
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    MT = MTT @ MT
    MT[0:3, 3] = MT[0:3, 3] + MT[0:3, 2]/MS[2] + MT[0:3, 1]/MS[1] + MT[0:3, 0]/MS[0]
    
    if np.iscomplexobj(x):
        data = np.abs(x).astype(np.float32)
    else:
        data = x.astype(np.float32)
        
    img = nib.Nifti1Image(data, affine=MT)
    img.set_sform(MT, code=1)
    os.makedirs(pathO, exist_ok=True)
    nib.save(img, os.path.join(pathO, f"{ima}{suffix}.nii"))
