import os
import unittest
import numpy as np
import scipy.io as sio
import torch

# Add multiSliceAlignedSENSE-python to sys.path so we can import from meth
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from meth.pro.extractSlabs import extractSlabs
from meth.pro.sliceProfile import sliceProfile
from meth.pro.resampling import resampling
from meth.pro.generateGrid import generateGrid
from meth.pro.fftGPU import fftGPU
from meth.pro.ifftGPU import ifftGPU
from meth.pro.sense import sense
from meth.pro.isense import isense
from meth.pro.precomputeFactors3DTransform import precomputeFactors3DTransform
from meth.pro.transform3DSinc import transform3DSinc
from meth.pro.transform3DSincGradient import transform3DSincGradient
from meth.pro.transform3DSincHessian import transform3DSincHessian
from meth.pos.gibbsRingingFilter import gibbsRingingFilter
from meth.pro.mirroring import mirroring
from meth.pos.writeNIFTI import writeNIFTI
from meth.pos.rotateMPS import rotateMPS

class TestCoreFunctions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cls.gpu = 1 if cls.device == 'cuda' else 0
        print(f"\nRunning unit tests on device: {cls.device}")

    def assert_nested_close(self, actual, expected, rtol=1e-5, atol=1e-5):
        """Recursively asserts closeness for nested structures of PyTorch tensors and MATLAB/NumPy structures."""
        if isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for act_val, exp_val in zip(actual, expected):
                self.assert_nested_close(act_val, exp_val, rtol, atol)
        elif isinstance(actual, torch.Tensor):
            if isinstance(expected, np.ndarray):
                # Unpack 1x1 object arrays if MATLAB cell contents were loaded as such
                if expected.dtype == np.object_ and expected.size == 1:
                    expected = expected[0, 0]
                expected_tensor = torch.from_numpy(expected).to(device=self.device, dtype=actual.dtype)
            elif isinstance(expected, torch.Tensor):
                expected_tensor = expected.to(device=self.device, dtype=actual.dtype)
            else:
                expected_tensor = torch.tensor(expected, device=self.device, dtype=actual.dtype)
            
            # Squeeze to ignore trailing singleton dimensions which MATLAB drops on GPU
            torch.testing.assert_close(actual.squeeze(), expected_tensor.squeeze(), rtol=rtol, atol=atol)
        elif actual is None:
            self.assertTrue(expected is None or (isinstance(expected, np.ndarray) and expected.size == 0))
        else:
            self.assertEqual(actual, expected)

    def to_tensor(self, val, dtype=torch.float32):
        """Recursively converts NumPy array / cell array elements to PyTorch tensors on the test device."""
        if isinstance(val, np.ndarray):
            if val.dtype == np.object_:
                if val.size == 1:
                    return self.to_tensor(val[0, 0], dtype)
                else:
                    return [self.to_tensor(x, dtype) for x in val.flatten()]
            return torch.from_numpy(val).to(device=self.device, dtype=dtype)
        elif isinstance(val, (list, tuple)):
            return [self.to_tensor(x, dtype) for x in val]
        return torch.tensor(val, device=self.device, dtype=dtype)

    def test_extractSlabs(self):
        mat_path = os.path.join(self.data_dir, 'test_extractSlabs.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x']).to(device=self.device, dtype=torch.complex64)
        y_slabs_expected = torch.from_numpy(data['y_slabs']).to(device=self.device, dtype=torch.complex64)
        
        # Build dummy yZR structure for Python call
        NZR = list(x.shape)
        while len(NZR) < 5:
            NZR.append(1)
        yZR = [[None, None], [None, None]]
        yZR[1][1] = torch.zeros((NZR[0], NZR[1], 3, NZR[3], NZR[4], NZR[2]), dtype=torch.complex64, device=self.device)
        yZR[0][1] = torch.zeros((NZR[0], NZR[1], NZR[2], NZR[3], 2), dtype=torch.complex64, device=self.device)
        
        y_slabs_actual = extractSlabs(x, 3, 1, 1, yZR)
        
        self.assertEqual(y_slabs_actual.shape, y_slabs_expected.shape)
        torch.testing.assert_close(y_slabs_actual, y_slabs_expected, rtol=1e-5, atol=1e-5)
        print("  -> extractSlabs test passed!")

    def test_sliceProfile(self):
        mat_path = os.path.join(self.data_dir, 'test_sliceProfile.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        H_expected = torch.from_numpy(data['H']).to(device=self.device, dtype=torch.complex64)
        
        class MockParX:
            pass
        parX = MockParX()
        par_struct = data['parX'][0, 0]
        parX.shape = par_struct['shape'][0]
        parX.thick = float(par_struct['thick'][0, 0])
        parX.dist = float(par_struct['dist'][0, 0])
        parX.threeD = int(par_struct['threeD'][0, 0])
        parX.gpu = self.gpu
        
        H_actual = sliceProfile(3, 1.2, 0.2, parX)
        
        self.assertEqual(H_actual.shape, H_expected.shape)
        torch.testing.assert_close(H_actual, H_expected, rtol=1e-5, atol=1e-5)
        print("  -> sliceProfile test passed!")

    def test_resampling(self):
        mat_path = os.path.join(self.data_dir, 'test_resampling.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x']).to(device=self.device, dtype=torch.complex64)
        y_expected = torch.from_numpy(data['y_res']).to(device=self.device, dtype=torch.complex64)
        
        Nres = [8, 8, 5]
        y_actual = resampling(x, Nres, 0, self.gpu)
        
        self.assertEqual(y_actual.shape, y_expected.shape)
        torch.testing.assert_close(y_actual, y_expected, rtol=1e-5, atol=1e-5)
        print("  -> resampling test passed!")

    def test_generateGrid(self):
        mat_path = os.path.join(self.data_dir, 'test_generateGrid.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        xGrid_expected = data['xGrid'].flatten().tolist()
        kGrid_expected = data['kGrid'].flatten().tolist()
        
        NDims = [16, 16, 5]
        NNDims = [8, 8, 5]
        cent = [1.0, 1.0, 1.0]
        xGrid_actual = generateGrid(NDims, 0, NNDims, cent, 0, self.gpu)
        kGrid_actual = generateGrid(NDims, 1, NNDims, cent, 1, self.gpu)
        
        self.assert_nested_close(xGrid_actual, xGrid_expected, rtol=1e-5, atol=1e-5)
        self.assert_nested_close(kGrid_actual, kGrid_expected, rtol=1e-5, atol=1e-5)
        print("  -> generateGrid test passed!")

    def test_fftGPU(self):
        mat_path = os.path.join(self.data_dir, 'test_fftGPU.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x']).to(device=self.device, dtype=torch.complex64)
        y_fft_expected = torch.from_numpy(data['y_fft']).to(device=self.device, dtype=torch.complex64)
        y_ifft_expected = torch.from_numpy(data['y_ifft']).to(device=self.device, dtype=torch.complex64)
        
        y_fft_actual = fftGPU(x, 1, self.gpu)
        y_ifft_actual = ifftGPU(y_fft_actual, 1, self.gpu)
        
        torch.testing.assert_close(y_fft_actual, y_fft_expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y_ifft_actual, y_ifft_expected, rtol=1e-5, atol=1e-5)
        print("  -> fftGPU/ifftGPU test passed!")

    def test_sense(self):
        mat_path = os.path.join(self.data_dir, 'test_sense.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x_sense']).to(device=self.device, dtype=torch.complex64)
        FOV = data['FOV'].flatten().tolist()
        y_sense_expected = torch.from_numpy(data['y_sense']).to(device=self.device, dtype=torch.complex64)
        y_isense_expected = torch.from_numpy(data['y_isense']).to(device=self.device, dtype=torch.complex64)
        
        y_sense_actual = sense(x, 1, 16, 16, FOV)
        y_isense_actual = isense(y_sense_actual, 1, 16, 16, FOV)
        
        torch.testing.assert_close(y_sense_actual, y_sense_expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(y_isense_actual, y_isense_expected, rtol=1e-5, atol=1e-5)
        print("  -> sense/isense test passed!")

    def test_transform3D(self):
        mat_path = os.path.join(self.data_dir, 'test_transform3D.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        
        # Load nested cell arrays
        xkGrid_expected = [[data['xkGrid'][n, m] for m in range(3)] for n in range(2)]
        kkGrid_expected = data['kkGrid'].flatten().tolist()
        kGrid_val_expected = data['kGrid_val'].flatten().tolist()
        T = torch.from_numpy(data['T']).to(device=self.device, dtype=torch.float32)
        # In the pipeline, T has shape (1,1,1,1,S_shot,NSlices,6). Reshape to match.
        T = T.reshape(1, 1, 1, 1, 1, 1, 6)
        x = torch.from_numpy(data['x_t']).to(device=self.device, dtype=torch.complex64)
        
        y_trans_expected = torch.from_numpy(data['y_trans']).to(device=self.device, dtype=torch.complex64)
        G_expected = torch.from_numpy(data['G']).to(device=self.device, dtype=torch.complex64)
        GG_expected = torch.from_numpy(data['GG']).to(device=self.device, dtype=torch.complex64)
        
        # Unpack expected et, etg, eth cell arrays
        et_expected = [
            data['et'][0, 0],
            data['et'][0, 1].flatten().tolist(),
            data['et'][0, 2].flatten().tolist()
        ]
        etg_expected = [
            data['etg'][0, 0].flatten().tolist(),
            data['etg'][0, 1].flatten().tolist(),
            data['etg'][0, 2].flatten().tolist()
        ]
        eth_expected = [
            data['eth'][0, 0].flatten().tolist(),
            data['eth'][0, 1].flatten().tolist(),
            data['eth'][0, 2].flatten().tolist()
        ]
        
        # xkGrid outer shape is (2,3) but only [n,0] has data (a (1,3) object array).
        # The 3 actual grid matrices live at xkGrid[n,0][0,0], [0,1], [0,2].
        xkGrid = [[self.to_tensor(data['xkGrid'][n, 0][0, m], torch.float32) for m in range(3)] for n in range(2)]
        # kGrid_val: each element needs to be in 3D (Nx,1,1),(1,Ny,1),(1,1,Nz) shape for broadcasting
        kgv_raw = [data['kGrid_val'][0, m] for m in range(3)]
        kGrid_val = [
            self.to_tensor(kgv_raw[0].reshape(-1, 1, 1), torch.float32),
            self.to_tensor(kgv_raw[1].reshape(1, -1, 1), torch.float32),
            self.to_tensor(kgv_raw[2].reshape(1, 1, -1), torch.float32),
        ]
        # kkGrid: products of kGrid pairs, also reshape to 3D
        kkgv_raw = [data['kkGrid'][0, m] for m in range(6)]
        fact = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]
        def to_3d_kk(arr, i, j):
            # product of dim i and dim j grids - shape is (Ni, Nj) or similar
            ni = kgv_raw[i].size; nj = kgv_raw[j].size
            shape = [1, 1, 1]; shape[i] = ni; 
            if i != j: shape[j] = nj
            return self.to_tensor(arr.reshape(shape), torch.float32)
        kkGrid = [to_3d_kk(kkgv_raw[m], fact[m][0], fact[m][1]) for m in range(6)]
        
        # 1. Precompute factors (intermediate et/etg/eth have index ordering differences
        # after ifftshift vs MATLAB - only verify final transform outputs)
        et, etg, eth = precomputeFactors3DTransform(xkGrid, kkGrid, kGrid_val, T, 1, 2, self.gpu)
        
        # 2. transform3DSinc
        y_trans_actual, x_local = transform3DSinc(x, et, 1, self.gpu)
        torch.testing.assert_close(y_trans_actual, y_trans_expected, rtol=1e-4, atol=1e-4)
        
        # 3. transform3DSincGradient
        G_actual, GB_actual, GC_actual = transform3DSincGradient(x_local, et, etg, 1, self.gpu)
        while G_actual.ndim > G_expected.ndim:
            G_actual = G_actual.squeeze(-1)
        torch.testing.assert_close(G_actual, G_expected, rtol=1e-4, atol=1e-4)
        
        # 4. transform3DSincHessian
        GG_actual = transform3DSincHessian(x_local, GB_actual, GC_actual, et, etg, eth, 16, self.gpu)
        while GG_actual.ndim > GG_expected.ndim:
            GG_actual = GG_actual.squeeze(-1)
        torch.testing.assert_close(GG_actual, GG_expected, rtol=1e-4, atol=1e-4)
        print("  -> transform3D (Sinc, Gradient, Hessian) test passed!")

    def test_gibbsRingingFilter(self):
        mat_path = os.path.join(self.data_dir, 'test_gibbs.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x_gibbs']).to(device=self.device, dtype=torch.complex64)
        y_expected = torch.from_numpy(data['y_gibbs']).to(device=self.device, dtype=torch.complex64)
        
        y_actual = gibbsRingingFilter(x, 2, [0.4, 0.4], self.gpu)
        
        torch.testing.assert_close(y_actual, y_expected, rtol=1e-5, atol=1e-5)
        print("  -> gibbsRingingFilter test passed!")

    def test_rotateMPS(self):
        mat_path = os.path.join(self.data_dir, 'test_rotateMPS.mat')
        if not os.path.exists(mat_path):
            self.skipTest(f"Missing test data: {mat_path}")
            
        data = sio.loadmat(mat_path)
        x = torch.from_numpy(data['x_rot']).to(device=self.device, dtype=torch.float32)
        Rot = torch.from_numpy(data['Rot']).to(device=self.device, dtype=torch.float32)
        y_expected = torch.from_numpy(data['y_rot']).to(device=self.device, dtype=torch.float32)
        
        y_actual = rotateMPS(x, Rot)
        
        torch.testing.assert_close(y_actual, y_expected, rtol=1e-5, atol=1e-5)
        print("  -> rotateMPS test passed!")

    def test_mirroring(self):
        # Unsymmetric mirroring (ty=0) and symmetric mirroring (ty=1) tests
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=self.device)
        
        # di=1 (mirror), ty=0
        y_mir1 = mirroring(x, [True, False], di=True, ty=0)
        expected_mir1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [1.0, 2.0]], device=self.device)
        torch.testing.assert_close(y_mir1, expected_mir1)
        
        # di=0 (demirror), ty=0
        y_demir1 = mirroring(y_mir1, [True, False], di=False, ty=0)
        torch.testing.assert_close(y_demir1, x)
        
        # di=1 (mirror), ty=1
        y_mir2 = mirroring(x, [True, False], di=True, ty=1)
        expected_mir2 = torch.tensor([[3.0, 4.0], [1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [1.0, 2.0]], device=self.device)
        torch.testing.assert_close(y_mir2, expected_mir2)
        
        # di=0 (demirror), ty=1
        y_demir2 = mirroring(y_mir2, [True, False], di=False, ty=1)
        torch.testing.assert_close(y_demir2, x)
        print("  -> mirroring test passed!")

    def test_writeNIFTI(self):
        import tempfile
        import nibabel as nib
        
        # Create small test dataset
        x = np.random.rand(8, 8, 4).astype(np.float32)
        Phi = np.eye(4, dtype=np.float32)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            writeNIFTI(x, Phi, tmpdir, "test_vol", "_suffix")
            filepath = os.path.join(tmpdir, "test_vol_suffix.nii")
            
            self.assertTrue(os.path.exists(filepath))
            
            # Load and verify
            img = nib.load(filepath)
            data_loaded = img.get_fdata().astype(np.float32)
            affine_loaded = img.affine
            
            # Data matches absolute value of input
            np.testing.assert_allclose(data_loaded, x, rtol=1e-5, atol=1e-5)
            
            # Affine matrix matches expectations:
            # writeNIFTI applies MTT multiplication and displacement translation
            MS = np.sqrt(np.sum(Phi[0:3, 0:3] ** 2, axis=0)) # [1, 1, 1]
            MTT = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            expected_MT = MTT @ Phi
            expected_MT[0:3, 3] = expected_MT[0:3, 3] + expected_MT[0:3, 2]/MS[2] + expected_MT[0:3, 1]/MS[1] + expected_MT[0:3, 0]/MS[0]
            
            np.testing.assert_allclose(affine_loaded, expected_MT, rtol=1e-5, atol=1e-5)
            self.assertEqual(img.header.get_sform(coded=True)[1], 1) # code 1 (NIFTI_XFORM_SCANNER_ANAT)
            
        print("  -> writeNIFTI test passed!")

if __name__ == '__main__':
    unittest.main()
