# multiSliceAlignedSENSE-python

A pure Python and PyTorch implementation of **multiSliceAlignedSENSE** (S2V motion correction and reconstruction) designed for fetal fMRI preprocessing.

This repository is a **1:1 mathematically validated port** of the original MATLAB-based [multiSliceAlignedSENSE](https://github.com/mriphysics/multiSliceAlignedSENSE) framework (Cordero-Grande et al.), optimized for modern GPU-accelerated computing clusters and free of MATLAB license constraints.

---

## 🌟 Key Features

*   **1:1 Mathematical Equivalence**: Validated against the original MATLAB outputs within a numerical tolerance of $10^{-4}$ to $10^{-5}$ across all core modules (including S2V interpolation, gradient/Hessian calculation, and optimization solvers).
*   **PyTorch & CUDA Powered**: Built on PyTorch, leveraging native CUDA acceleration for high-performance GPU computation.
*   **MATLAB-Free Deployment**: Easily deployable inside containerized environments (Docker, Singularity) and cloud clusters without expensive MATLAB Parallel Computing Toolbox licenses.
*   **General-Purpose SVR Solver**: While named after SENSE (Sensitivity Encoding), setting the coil sensitivity maps (`S`) to `1.0` and the sampling mask (`Ak`) to `1.0` allows it to function as a general-purpose **Slice-to-Volume Reconstruction (SVR)** motion correction tool for standard single-channel MRI acquisitions.

---

## 📦 Dependencies

Ensure you have the following packages installed (already configured in the `dhcp` environment):
*   Python 3.8+
*   PyTorch (CUDA compatible recommended)
*   SciPy
*   Nibabel
*   tqdm

---

## 🚀 Getting Started

### 1. Run the Unit Test Suite
To verify that all ported modules (SVR, SENSE, coordinate grids, Gibbs filtering, etc.) are working properly on your device:
```bash
python3 -m unittest tests/test_core_functions.py
```

### 2. Basic Usage Example
Below is a simplified example showing how to invoke the core alternating minimization solver (`optimizeLevelMS2D`):

```python
import torch
from meth.optimizeLevelMS2D import optimizeLevelMS2D
from meth.pro.generateGrid import generateGrid

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_flag = 1 if device == 'cuda' else 0

# Define shapes: (Nx, Ny, Nz)
Nx, Ny, Nz = 16, 16, 6
nCoils = 1
S_shot = 1

# Setup inputs
x = torch.randn((Nx, Ny, Nz), dtype=torch.complex64, device=device)
y = torch.randn((Nx, Ny, Nz, nCoils, S_shot), dtype=torch.complex64, device=device)
T = torch.zeros((1, 1, 1, 1, S_shot, Nz, 6), dtype=torch.float32, device=device)
S = torch.ones((Nx, Ny, Nz, nCoils), dtype=torch.complex64, device=device)  # Bypassing SENSE
W = torch.ones((Nx, Ny, Nz), dtype=torch.float32, device=device)
Ak = torch.ones((Nx, Ny, Nz, nCoils, S_shot), dtype=torch.float32, device=device)

# Generate coordinate grids
xGrid = generateGrid([Nx, Ny, Nz], 0, [Nx, Ny, Nz], [1.0, 1.0, 1.0], 0, gpu_flag)

class ReconstructionParams:
    threeD = 1
    correct = 1
    shape = 'se'
    thick = 3.0
    dist = 1.0
    toler = 1e-3
    outlP = 1.2
    thplc = 2

# Run solver
x_opt, T_opt, outlD_opt = optimizeLevelMS2D(
    x=x, y=y, T=T, S=S, W=W, Ak=Ak, xGrid=xGrid,
    mNorm=1.0, parX=ReconstructionParams(), debug=1,
    res=1.0, estT=True, gpu=gpu_flag, SlTh=1.0, SlOv=0.0
)

print("Optimized BOLD Volume:", x_opt.shape)
print("Optimized Motion Parameters:", T_opt.shape)
```

---

## 📜 How to Cite

If you use this Python/PyTorch implementation in your research, please cite this repository:

```bibtex
@software{multislice_aligned_sense_pytorch,
  author = {Dae-young Kim},
  title = {multiSliceAlignedSENSE-python: A Python port of S2V motion correction and reconstruction for fetal fMRI},
  url = {https://github.com/leroykim/multiSliceAlignedSENSE-python},
  year = {2026}
}
```

### References (Original MATLAB Implementation & Papers)
This code is a port based on the following original works:
1.  **Aligned SENSE / S2V Core Formulation:**
    > Cordero-Grande, L., Hughes, E. J., Hutter, J., Price, A. N., & Hajnal, J. V. (2018). *Three-dimensional motion corrected sensitivity encoding reconstruction for multi-shot multi-slice MRI: Application to neonatal brain imaging.* Magnetic Resonance in Medicine, 79(3), 1365-1376.
2.  **dHCP Fetal fMRI Pipeline Release Paper:**
    > Karolis, V. R., Cordero-Grande, L., Price, A. N., Hughes, E., Fitzgibbon, S. P., ... & Hajnal, J. V. (2025). *The developing Human Connectome Project fetal functional MRI release: Methods and data structures.* Imaging Neuroscience, Volume 3.
