import torch

def broadcast_helper(grid_tensor, val_tensor):
    """
    Append singleton dimensions to grid_tensor until its number of dimensions 
    matches val_tensor.ndim. This aligns the spatial grid_tensor to the left
    so it broadcasts correctly against val_tensor (which contains batch dimensions
    like S_shot and NSlices).
    """
    extra_dims = val_tensor.ndim - grid_tensor.ndim
    if extra_dims > 0:
        for _ in range(extra_dims):
            grid_tensor = grid_tensor.unsqueeze(-1)
    return grid_tensor

def precomputeFactors3DTransform(xkGrid, kkGrid, kGrid, T, di, cg, gpu=0):
    """
    Python port of MATLAB precomputeFactors3DTransform.m
    Supports arbitrary batch dimensions in T (e.g. 7D production shape or 2D test shape).
    """
    device = T.device

    # --- permute T so param dim (size 6) moves to front ---
    ND = T.ndim
    perm = list(range(ND))
    perm[0] = ND - 1
    perm[ND - 1] = 0
    T_perm = T.permute(*perm)

    theta  = T_perm[3:6].to(torch.complex64)  # rotation  (3, ...)
    t_val  = T_perm[0:3].to(torch.complex64)  # translation (3, ...)

    tantheta2  = torch.tan(theta / 2.0)
    tantheta2j = ((-1.0) ** (di - 1)) * 1j * tantheta2
    sintheta   = ((-1.0) ** di)        * 1j * torch.sin(theta)
    t_val      = ((-1.0) ** di)        * 1j * t_val

    if cg > 0:
        tantheta    = torch.tan(theta)
        tanthetacuad = tantheta2 * tantheta2
        tanthetacuad = (1.0 + tanthetacuad) / 2.0
        costheta    = torch.cos(theta).to(torch.complex64)

    # MATLAB: per(1,:)=[1 3 2]; per(2,:)=[2 1 3]
    # Python 0-indexed: subtract 1 -> per[0]=[0,2,1], per[1]=[1,0,2]
    per = [[0, 2, 1], [1, 0, 2]]

    et  = [None, [None] * 3, [None] * 3]
    etg = [None, [None] * 3, [None] * 3] if cg > 0 else None
    eth = [None, [None] * 3, [None] * 3] if cg == 2 else None

    perm_back = list(range(ND))
    perm_back[0] = ND - 1
    perm_back[ND - 1] = 0

    # ---------------------------------------------------------------
    # et{2}{m}, et{3}{m}
    # ---------------------------------------------------------------
    for m in range(3):
        val2 = tantheta2j[m:m+1].permute(*perm_back)
        val3 = sintheta[m:m+1].permute(*perm_back)

        xk1 = xkGrid[0][m].to(torch.complex64)
        xk2 = xkGrid[1][m].to(torch.complex64)

        et[1][m] = torch.exp(val2 * broadcast_helper(xk1, val2))
        et[2][m] = torch.exp(val3 * broadcast_helper(xk2, val3))

        if cg > 0:
            val_cuad = tanthetacuad[m:m+1].permute(*perm_back)
            val_cos  = costheta[m:m+1].permute(*perm_back)

            etg[1][m] = val_cuad * (1j * broadcast_helper(xk1, val_cuad))
            etg[2][m] = val_cos  * (-1j * broadcast_helper(xk2, val_cos))

            if cg == 2:
                val_tan2 = tantheta2[m:m+1].permute(*perm_back)
                val_tan  = tantheta[m:m+1].permute(*perm_back)

                eth[1][m] = val_tan2 + etg[1][m]
                eth[2][m] = -val_tan + etg[2][m]

    if cg > 0:
        for mm in range(1, 3):   # mm = 1, 2
            for n in range(3):   # n = 0, 1, 2
                etg[mm][n] = etg[mm][n] * et[mm][n]
                if cg == 2:
                    eth[mm][n] = eth[mm][n] * etg[mm][n]
                sdim = per[mm - 1][n]
                etg[mm][n] = torch.fft.ifftshift(etg[mm][n], dim=sdim)
                if cg == 2:
                    eth[mm][n] = torch.fft.ifftshift(eth[mm][n], dim=sdim)

    # ifftshift et{2} and et{3}
    for m in range(3):
        for n in range(1, 3):
            sdim = per[n - 1][m]
            et[n][m] = torch.fft.ifftshift(et[n][m], dim=sdim)

    # ---------------------------------------------------------------
    # et{1}  (translation phase)
    # ---------------------------------------------------------------
    t0 = t_val[0:1].permute(*perm_back)
    t1 = t_val[1:2].permute(*perm_back)
    t2 = t_val[2:3].permute(*perm_back)

    k0 = kGrid[0].to(torch.complex64)
    k1 = kGrid[1].to(torch.complex64)
    k2 = kGrid[2].to(torch.complex64)

    # Ensure all kGrid tensors are 3-D for broadcasting
    while k0.ndim < 3: k0 = k0.unsqueeze(-1)
    while k1.ndim < 3: k1 = k1.unsqueeze(-1)
    while k2.ndim < 3: k2 = k2.unsqueeze(-1)

    et_1 = t0 * broadcast_helper(k0, t0) + t1 * broadcast_helper(k1, t1) + t2 * broadcast_helper(k2, t2)
    et[0] = torch.exp(et_1)

    if cg > 0:
        etg[0] = [None] * 3
        for m in range(3):
            km = [k0, k1, k2][m]
            etg[0][m] = (-1j * broadcast_helper(km, et[0])) * et[0]
            for n in range(3):
                etg[0][m] = torch.fft.ifftshift(etg[0][m], dim=n)

        if cg == 2:
            eth[0] = [None] * 6
            for m in range(6):
                kkm = kkGrid[m].to(torch.complex64)
                while kkm.ndim < 3: kkm = kkm.unsqueeze(-1)
                eth[0][m] = (-broadcast_helper(kkm, et[0])) * et[0]
                for n in range(3):
                    eth[0][m] = torch.fft.ifftshift(eth[0][m], dim=n)

    # Final ifftshift of et{1} along each spatial dim
    for m in range(3):
        et[0] = torch.fft.ifftshift(et[0], dim=m)

    if cg == 2:
        return et, etg, eth
    elif cg == 1:
        return et, etg, None
    else:
        return et, None, None
