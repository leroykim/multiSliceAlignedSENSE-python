import torch
from .fftGPU import fftGPU
from .ifftGPU import ifftGPU

def transform3DSinc(I, et, di, gpu=0, F=None, FH=None):
    def et_bc(factor, ref):
        """Append singleton dims to factor so it broadcasts against ref."""
        f = factor
        while f.ndim < ref.ndim:
            f = f.unsqueeze(-1)
        return f

    orig_ndim = I.ndim
    tr = [[1, 3, 2], [2, 1, 3]]
    IB = [None] * 5

    # Pre-align I to the spatial+batch ndim of et[0]
    target_dim = et[0].ndim
    while I.ndim < target_dim:
        I = I.unsqueeze(-1)

    if di == 1:
        # Rotations
        for m in range(3):
            dim1 = tr[0][m]
            dim2 = tr[1][m]

            F_dim1  = F[dim1 - 1]  if F  is not None else None
            FH_dim1 = FH[dim1 - 1] if FH is not None else None
            F_dim2  = F[dim2 - 1]  if F  is not None else None
            FH_dim2 = FH[dim2 - 1] if FH is not None else None

            I = fftGPU(I, dim1, gpu, F_dim1)
            IB[m] = I
            I = I * et_bc(et[1][m], I)
            I = ifftGPU(I, dim1, gpu, FH_dim1)
            I = fftGPU(I, dim2, gpu, F_dim2)
            I = I * et_bc(et[2][m], I)
            I = ifftGPU(I, dim2, gpu, FH_dim2)
            I = fftGPU(I, dim1, gpu, F_dim1)
            I = I * et_bc(et[1][m], I)
            if m != 2:
                I = ifftGPU(I, dim1, gpu, FH_dim1)

        # Translation
        for m in range(1, 4):
            if m != tr[0][2]:
                F_m = F[m - 1] if F is not None else None
                I = fftGPU(I, m, gpu, F_m)
        IB[3] = I
        I = I * et_bc(et[0], I)
        for m in range(3, 0, -1):
            FH_m = FH[m - 1] if FH is not None else None
            I = ifftGPU(I, m, gpu, FH_m)

    else:
        # Back-translation
        for m in range(1, 4):
            F_m = F[m - 1] if F is not None else None
            I = fftGPU(I, m, gpu, F_m)
        I = I * et_bc(et[0], I)
        for m in range(3, 0, -1):
            if m != tr[0][2]:
                FH_m = FH[m - 1] if FH is not None else None
                I = ifftGPU(I, m, gpu, FH_m)

        # Back-rotations
        for m in range(2, -1, -1):
            dim1 = tr[0][m]
            dim2 = tr[1][m]
            F_dim1  = F[dim1 - 1]  if F  is not None else None
            FH_dim1 = FH[dim1 - 1] if FH is not None else None
            F_dim2  = F[dim2 - 1]  if F  is not None else None
            FH_dim2 = FH[dim2 - 1] if FH is not None else None

            if m != 2:
                I = fftGPU(I, dim1, gpu, F_dim1)
            I = I * et_bc(et[1][m], I)
            I = ifftGPU(I, dim1, gpu, FH_dim1)
            I = fftGPU(I, dim2, gpu, F_dim2)
            I = I * et_bc(et[2][m], I)
            I = ifftGPU(I, dim2, gpu, FH_dim2)
            I = fftGPU(I, dim1, gpu, F_dim1)
            I = I * et_bc(et[1][m], I)
            if m == 0:
                I = torch.sum(I, dim=4)
            I = ifftGPU(I, dim1, gpu, FH_dim1)

    # Squeeze I back to its original number of dimensions
    while I.ndim > orig_ndim:
        I = I.squeeze(-1)

    return I, IB
