import torch
from .fftGPU import fftGPU
from .ifftGPU import ifftGPU

def transform3DSincGradient(xB, et, etg, full, gpu=0, F=None, FH=None):
    def bc(factor, ref):
        """Append singleton dims to factor to broadcast against ref."""
        f = factor
        while f.ndim < ref.ndim:
            f = f.unsqueeze(-1)
        return f

    GB = [None] * 3
    GC = [None] * 3
    G  = [None] * 6

    F_1  = F[0]  if F  is not None else None
    FH_1 = FH[0] if FH is not None else None
    F_2  = F[1]  if F  is not None else None
    FH_2 = FH[1] if FH is not None else None
    F_3  = F[2]  if F  is not None else None
    FH_3 = FH[2] if FH is not None else None

    # Translation parameters
    for m in range(3):
        x_local = xB[3] * bc(etg[0][m], xB[3])
        for n in range(1, 4):
            FH_n = FH[n - 1] if FH is not None else None
            x_local = ifftGPU(x_local, n, gpu, FH_n)
        G[m] = x_local

    # First rotation
    x_0 = xB[0] * bc(et[1][0],  xB[0])
    x_1 = xB[0] * bc(etg[1][0], xB[0])

    x_0 = ifftGPU(x_0, 1, gpu, FH_1)
    x_0 = fftGPU(x_0,  2, gpu, F_2)
    x_1 = ifftGPU(x_1, 1, gpu, FH_1)
    x_1 = fftGPU(x_1,  2, gpu, F_2)

    x_1 = bc(etg[2][0], x_0) * x_0 + bc(et[2][0], x_1) * x_1
    x_0 = bc(et[2][0], x_0) * x_0

    x_0 = ifftGPU(x_0, 2, gpu, FH_2)
    x_0 = fftGPU(x_0,  1, gpu, F_1)
    x_1 = ifftGPU(x_1, 2, gpu, FH_2)
    x_1 = fftGPU(x_1,  1, gpu, F_1)

    x_0 = bc(etg[1][0], x_0) * x_0 + bc(et[1][0], x_1) * x_1
    x_0 = ifftGPU(x_0, 1, gpu, FH_1)

    x_0 = fftGPU(x_0, 3, gpu, F_3)
    GC[0] = x_0

    x_0 = x_0 * bc(et[1][1], x_0)
    x_0 = ifftGPU(x_0, 3, gpu, FH_3)

    x_0 = fftGPU(x_0, 1, gpu, F_1)
    x_0 = x_0 * bc(et[2][1], x_0)
    x_0 = ifftGPU(x_0, 1, gpu, FH_1)

    x_0 = fftGPU(x_0, 3, gpu, F_3)
    x_0 = x_0 * bc(et[1][1], x_0)
    x_0 = ifftGPU(x_0, 3, gpu, FH_3)

    x_0 = fftGPU(x_0, 2, gpu, F_2)
    GC[1] = x_0

    x_0 = x_0 * bc(et[1][2], x_0)
    x_0 = ifftGPU(x_0, 2, gpu, FH_2)

    x_0 = fftGPU(x_0, 3, gpu, F_3)
    x_0 = x_0 * bc(et[2][2], x_0)
    x_0 = ifftGPU(x_0, 3, gpu, FH_3)

    x_0 = fftGPU(x_0, 2, gpu, F_2)
    x_0 = x_0 * bc(et[1][2], x_0)

    for m in [1, 3]:
        F_m = F[m - 1] if F is not None else None
        x_0 = fftGPU(x_0, m, gpu, F_m)
    GB[0] = x_0

    x_0 = x_0 * bc(et[0], x_0)
    for m in range(3, 0, -1):
        FH_m = FH[m - 1] if FH is not None else None
        x_0 = ifftGPU(x_0, m, gpu, FH_m)
    G[3] = x_0

    # Second rotation
    x_0 = xB[1] * bc(et[1][1],  xB[1])
    x_1 = xB[1] * bc(etg[1][1], xB[1])

    x_0 = ifftGPU(x_0, 3, gpu, FH_3)
    x_0 = fftGPU(x_0,  1, gpu, F_1)
    x_1 = ifftGPU(x_1, 3, gpu, FH_3)
    x_1 = fftGPU(x_1,  1, gpu, F_1)

    x_1 = bc(etg[2][1], x_0) * x_0 + bc(et[2][1], x_1) * x_1
    x_0 = bc(et[2][1], x_0) * x_0

    x_0 = ifftGPU(x_0, 1, gpu, FH_1)
    x_0 = fftGPU(x_0,  3, gpu, F_3)
    x_1 = ifftGPU(x_1, 1, gpu, FH_1)
    x_1 = fftGPU(x_1,  3, gpu, F_3)

    x_0 = bc(etg[1][1], x_0) * x_0 + bc(et[1][1], x_1) * x_1
    x_0 = ifftGPU(x_0, 3, gpu, FH_3)

    x_0 = fftGPU(x_0, 2, gpu, F_2)
    GC[2] = x_0

    x_0 = x_0 * bc(et[1][2], x_0)
    x_0 = ifftGPU(x_0, 2, gpu, FH_2)

    x_0 = fftGPU(x_0, 3, gpu, F_3)
    x_0 = x_0 * bc(et[2][2], x_0)
    x_0 = ifftGPU(x_0, 3, gpu, FH_3)

    x_0 = fftGPU(x_0, 2, gpu, F_2)
    x_0 = x_0 * bc(et[1][2], x_0)

    for m in [1, 3]:
        F_m = F[m - 1] if F is not None else None
        x_0 = fftGPU(x_0, m, gpu, F_m)
    GB[1] = x_0

    x_0 = x_0 * bc(et[0], x_0)
    for m in range(3, 0, -1):
        FH_m = FH[m - 1] if FH is not None else None
        x_0 = ifftGPU(x_0, m, gpu, FH_m)
    G[4] = x_0

    # Third rotation
    x_0 = xB[2] * bc(et[1][2],  xB[2])
    x_1 = xB[2] * bc(etg[1][2], xB[2])

    x_0 = ifftGPU(x_0, 2, gpu, FH_2)
    x_0 = fftGPU(x_0,  3, gpu, F_3)
    x_1 = ifftGPU(x_1, 2, gpu, FH_2)
    x_1 = fftGPU(x_1,  3, gpu, F_3)

    x_1 = bc(etg[2][2], x_0) * x_0 + bc(et[2][2], x_1) * x_1
    x_0 = bc(et[2][2], x_0) * x_0

    x_0 = ifftGPU(x_0, 3, gpu, FH_3)
    x_0 = fftGPU(x_0,  2, gpu, F_2)
    x_1 = ifftGPU(x_1, 3, gpu, FH_3)
    x_1 = fftGPU(x_1,  2, gpu, F_2)

    x_0 = bc(etg[1][2], x_0) * x_0 + bc(et[1][2], x_1) * x_1

    for m in [1, 3]:
        F_m = F[m - 1] if F is not None else None
        x_0 = fftGPU(x_0, m, gpu, F_m)
    GB[2] = x_0

    x_0 = x_0 * bc(et[0], x_0)
    for m in range(3, 0, -1):
        FH_m = FH[m - 1] if FH is not None else None
        x_0 = ifftGPU(x_0, m, gpu, FH_m)
    G[5] = x_0

    if full:
        for i in range(len(G)):
            while G[i].ndim < 5:
                G[i] = G[i].unsqueeze(-1)
        G = torch.stack(G, dim=5)

    return G, GB, GC
