import torch

def mirroring(x, mirror, di, ty=0):
    N = list(x.shape)
    nDimsOu = len(mirror)
    while len(N) < nDimsOu:
        N.append(1)
    nDimsIn = len(N)
    
    for m in range(nDimsOu):
        if mirror[m]:
            perm = list(range(nDimsIn))
            perm[0] = m
            perm[m] = 0
            x = x.permute(*perm)
            
            if di: # Mirror
                if ty == 0:
                    flipped = torch.flip(x, dims=[0])
                    x = torch.cat([x, flipped], dim=0)
                else:
                    flipped = torch.flip(x, dims=[0])
                    x = torch.cat([flipped, x, flipped], dim=0)
            else: # Demirror
                if ty == 0:
                    x = x[:x.shape[0] // 2]
                else:
                    length = x.shape[0] // 3
                    x = x[length : 2 * length]
                    
            x = x.permute(*perm)
    return x
