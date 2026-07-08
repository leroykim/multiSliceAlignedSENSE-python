function generate_all_test_data()
    % Add original MATLAB codebase to path
    addpath(fullfile(pwd, '../../../multiSliceAlignedSENSE-1.0.1'));
    addpath(fullfile(pwd, '../../../multiSliceAlignedSENSE-1.0.1/meth'));
    addpath(fullfile(pwd, '../../../multiSliceAlignedSENSE-1.0.1/meth/pro'));
    addpath(fullfile(pwd, '../../../multiSliceAlignedSENSE-1.0.1/meth/pos'));

    % Create output data directory if not exists
    out_dir = '../data';
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end
    
    fprintf('Generating test data for SENSE core functions...\n');

    %% 1. extractSlabs
    fprintf('  -> Generating test_extractSlabs.mat\n');
    x = randn([16, 16, 8]) + 1i * randn([16, 16, 8]);
    NZR = size(x);
    NZR(end+1:5) = 1;
    NzSlab = 3;
    yZR{2}{2} = single(zeros([NZR(1:2) NzSlab NZR(4:5) NZR(3)]));
    yZR{1}{2} = single(zeros([NZR(1:4) 2])); % Assume NT(5) is 2
    
    y_slabs = extractSlabs(x, NzSlab, 1, 1, yZR);
    save(fullfile(out_dir, 'test_extractSlabs.mat'), 'x', 'y_slabs');

    %% 2. sliceProfile
    fprintf('  -> Generating test_sliceProfile.mat\n');
    parX.shape = 'se';
    parX.thick = 3.0;
    parX.dist = 1.0;
    parX.threeD = 1;
    H = sliceProfile(3, 1.2, 0.2, parX);
    save(fullfile(out_dir, 'test_sliceProfile.mat'), 'parX', 'H');

    %% 3. resampling
    fprintf('  -> Generating test_resampling.mat\n');
    x = randn([16, 16, 5]) + 1i * randn([16, 16, 5]);
    y_res = resampling(x, [8, 8, 5], 0, 0);
    save(fullfile(out_dir, 'test_resampling.mat'), 'x', 'y_res');

    %% 4. generateGrid
    fprintf('  -> Generating test_generateGrid.mat\n');
    xGrid = generateGrid([16, 16, 5], 0, [8, 8, 5], [1.0, 1.0, 1.0], 0, 0);
    kGrid = generateGrid([16, 16, 5], 1, [8, 8, 5], [1.0, 1.0, 1.0], 1, 0);
    save(fullfile(out_dir, 'test_generateGrid.mat'), 'xGrid', 'kGrid');

    %% 5. fftGPU / ifftGPU
    fprintf('  -> Generating test_fftGPU.mat\n');
    x = randn([16, 16, 5]) + 1i * randn([16, 16, 5]);
    y_fft = fftGPU(x, 1, 0);
    y_ifft = ifftGPU(y_fft, 1, 0);
    save(fullfile(out_dir, 'test_fftGPU.mat'), 'x', 'y_fft', 'y_ifft');

    %% 6. sense / isense
    fprintf('  -> Generating test_sense.mat\n');
    x_sense = randn([16, 16, 5]) + 1i * randn([16, 16, 5]);
    FOV = [0, 0];
    y_sense = sense(x_sense, 1, 16, 16, FOV);
    y_isense = isense(y_sense, 1, 16, 16, FOV);
    save(fullfile(out_dir, 'test_sense.mat'), 'x_sense', 'FOV', 'y_sense', 'y_isense');

    %% 7. precomputeFactors3DTransform / transform3DSinc / Gradient / Hessian
    fprintf('  -> Generating test_transform3D.mat\n');
    NDims = [16, 16, 6];
    xGrid_val = generateGrid(NDims, 0, NDims, [1.0, 1.0, 1.0], 0, 0);
    kGrid_val = generateGrid(NDims, 1, NDims, [1.0, 1.0, 1.0], 1, 0);
    
    per_map = [1 3 2; 2 1 3];
    xkGrid = cell(2, 3);
    for n=1:2
        for m=1:3
            xkGrid{n}{m} = bsxfun(@times, xGrid_val{per_map(3-n,m)}, kGrid_val{per_map(n,m)});
        end
    end
    
    fact_map = [1 2 3 1 1 2; 1 2 3 2 3 3];
    kkGrid = cell(1, 6);
    for m=1:6
        kkGrid{m} = bsxfun(@times, kGrid_val{fact_map(1,m)}, kGrid_val{fact_map(2,m)});
    end
    
    T = reshape([0.5, -0.2, 0.1, 0.05, -0.03, 0.02], [1,1,1,1,1,1,6]);
    [et, etg, eth] = precomputeFactors3DTransform(xkGrid, kkGrid, kGrid_val, T, 1, 2, 0);
    x_t = randn(NDims) + 1i * randn(NDims);
    [y_trans, x_local] = transform3DSinc(x_t, et, 1, 0);
    [G, GB, GC] = transform3DSincGradient(x_local, et, etg, 1, 0);
    GG = transform3DSincHessian(x_local, GB, GC, et, etg, eth, 16, 0);
    save(fullfile(out_dir, 'test_transform3D.mat'), 'xkGrid', 'kkGrid', 'kGrid_val', 'T', 'et', 'etg', 'eth', 'x_t', 'y_trans', 'x_local', 'G', 'GG');

    %% 8. gibbsRingingFilter
    fprintf('  -> Generating test_gibbs.mat\n');
    x_gibbs = randn([16, 16, 5]) + 1i * randn([16, 16, 5]);
    y_gibbs = gibbsRingingFilter(x_gibbs, 2, [0.4, 0.4], 0);
    save(fullfile(out_dir, 'test_gibbs.mat'), 'x_gibbs', 'y_gibbs');

    %% 9. rotateMPS
    fprintf('  -> Generating test_rotateMPS.mat\n');
    x_rot = randn([16, 16, 5]);
    Rot = [0, 1, 0; -1, 0, 0; 0, 0, 1];
    y_rot = rotateMPS(x_rot, Rot);
    save(fullfile(out_dir, 'test_rotateMPS.mat'), 'x_rot', 'Rot', 'y_rot');

    %% 10. mirroring
    fprintf('  -> Generating test_mirroring.mat\n');
    x_mir = randn([8, 8, 4]) + 1i * randn([8, 8, 4]);
    mirror_flags = [1, 0, 1];
    
    y_mir1 = mirroring(x_mir, mirror_flags, 1, 0);
    y_demir1 = mirroring(y_mir1, mirror_flags, 0, 0);
    
    y_mir2 = mirroring(x_mir, mirror_flags, 1, 1);
    y_demir2 = mirroring(y_mir2, mirror_flags, 0, 1);
    
    save(fullfile(out_dir, 'test_mirroring.mat'), 'x_mir', 'mirror_flags', ...
         'y_mir1', 'y_demir1', 'y_mir2', 'y_demir2');

    fprintf('Successfully generated all test data!\n');
end
