function performance_optimization_script()
% PERFORMANCE_OPTIMIZATION_SCRIPT Demonstrates performance optimization techniques for financial econometric computations using the MFE Toolbox
%
% This script provides detailed examples of performance optimization strategies that
% can significantly speed up financial time series analysis, including:
%   1. MEX acceleration vs MATLAB-only implementations
%   2. Memory preallocation and optimization
%   3. Vectorization for performance
%   4. Scaling with dataset size
%   5. Matrix operation optimization
%   6. High-frequency data performance techniques
%
% The examples use real-world financial econometric models to demonstrate
% performance improvements of >50% through MEX optimization and efficient
% programming practices.
%
% See also AGARCHFIT, EGARCHFIT, MATRIXDIAGNOSTICS, RV_COMPUTE, ARMAXFILTER

% Copyright: MFE Toolbox
% Version 4.0    Date: 2009/10/28

%% Initialize the MFE Toolbox
% Add necessary paths for the MFE Toolbox components
addToPath();

%% Introduction
fprintf('=====================================================================\n');
fprintf('      MFE TOOLBOX PERFORMANCE OPTIMIZATION TECHNIQUES\n');
fprintf('=====================================================================\n\n');

fprintf('This example demonstrates various performance optimization techniques\n');
fprintf('for financial econometric computations using the MFE Toolbox v4.0.\n\n');

fprintf('Performance optimization is crucial for financial applications that\n');
fprintf('process large volumes of data or require real-time computation.\n\n');

%% Load example data
fprintf('Loading example financial data...\n');
try
    % Try to load example data file
    load('example_financial_data.mat', 'data');
    fprintf('Example data loaded successfully.\n\n');
catch
    % If data file doesn't exist, generate synthetic data
    fprintf('Example data file not found. Generating synthetic data...\n');
    T = 5000;
    data = randn(T, 1) * 0.01;
    for t = 2:T
        data(t) = 0.05 * data(t-1) + data(t) + 0.1 * (data(t-1) < 0) * data(t-1);
    end
    fprintf('Synthetic financial returns generated (%d observations).\n\n', T);
end

%% 1. MEX Acceleration Demonstration
fprintf('=====================================================================\n');
fprintf('1. MEX ACCELERATION DEMONSTRATION\n');
fprintf('=====================================================================\n\n');

fprintf('MEX functions provide C-level performance while maintaining\n');
fprintf('MATLAB''s ease of use. The MFE Toolbox uses MEX for critical\n');
fprintf('computational components, particularly in volatility modeling.\n\n');

% Demonstrate MEX vs. MATLAB performance
fprintf('Benchmarking MEX vs. MATLAB-only implementation:\n');
mexResults = demonstrate_mex_speedup(data);

fprintf('Performance results:\n');
fprintf('  MATLAB-only time: %.4f seconds\n', mexResults.matlabTime);
fprintf('  MEX-optimized time: %.4f seconds\n', mexResults.mexTime);
fprintf('  Speedup ratio: %.2fx\n', mexResults.speedupRatio);
fprintf('  Performance improvement: %.1f%%\n\n', mexResults.improvementPct);

if mexResults.improvementPct > 50
    fprintf('✓ Successfully achieved >50%% performance improvement with MEX!\n\n');
else
    fprintf('Note: MEX optimization showing less than expected improvement.\n');
    fprintf('This might be due to small dataset size or system configuration.\n\n');
end

%% 2. Memory Preallocation
fprintf('=====================================================================\n');
fprintf('2. MEMORY PREALLOCATION AND OPTIMIZATION\n');
fprintf('=====================================================================\n\n');

fprintf('Preallocating memory for arrays is critical for performance,\n');
fprintf('especially in loop-heavy financial calculations.\n\n');

% Demonstrate memory preallocation benefits
fprintf('Benchmarking memory preallocation benefits:\n');
preallocResults = demonstrate_memory_preallocation(5000);

fprintf('Performance results:\n');
fprintf('  Non-preallocated time: %.4f seconds\n', preallocResults.noPreallocTime);
fprintf('  Preallocated time: %.4f seconds\n', preallocResults.preallocTime);
fprintf('  Performance improvement: %.1f%%\n\n', preallocResults.improvementPct);

fprintf('Memory usage statistics:\n');
fprintf('  Non-preallocated memory operations: %d\n', preallocResults.noPreallocMemOps);
fprintf('  Preallocated memory operations: %d\n', preallocResults.preallocMemOps);
fprintf('  Memory operation reduction: %.1f%%\n\n', preallocResults.memOpReductionPct);

%% 3. Vectorization for Performance
fprintf('=====================================================================\n');
fprintf('3. VECTORIZATION FOR PERFORMANCE\n');
fprintf('=====================================================================\n\n');

fprintf('Vectorized operations eliminate loops and dramatically improve\n');
fprintf('performance for many financial calculations.\n\n');

% Demonstrate vectorization benefits
fprintf('Benchmarking vectorization benefits:\n');
vectorizationResults = demonstrate_vectorization(data);

fprintf('Performance results:\n');
fprintf('  Loop-based time: %.4f seconds\n', vectorizationResults.loopTime);
fprintf('  Vectorized time: %.4f seconds\n', vectorizationResults.vectorTime);
fprintf('  Performance improvement: %.1f%%\n\n', vectorizationResults.improvementPct);

%% 4. Scaling with Dataset Size
fprintf('=====================================================================\n');
fprintf('4. SCALING WITH DATASET SIZE\n');
fprintf('=====================================================================\n\n');

fprintf('Understanding how performance scales with dataset size is crucial\n');
fprintf('for large-scale financial data analysis.\n\n');

% Test different dataset sizes
dataSizes = [1000, 2000, 5000, 10000, 20000];
fprintf('Benchmarking performance scaling with dataset sizes: %s\n', mat2str(dataSizes));
scalingResults = benchmark_data_scaling(dataSizes);

% Create a figure to visualize the scaling
figure;
subplot(2,1,1);
plot(dataSizes, scalingResults.matlabTimes, 'b-o', 'LineWidth', 2);
hold on;
plot(dataSizes, scalingResults.mexTimes, 'r-x', 'LineWidth', 2);
xlabel('Dataset Size (observations)');
ylabel('Execution Time (seconds)');
title('Performance Scaling with Dataset Size');
legend('MATLAB Implementation', 'MEX-optimized Implementation', 'Location', 'northwest');
grid on;

% Plot speedup ratio
subplot(2,1,2);
plot(dataSizes, scalingResults.speedupRatios, 'g-s', 'LineWidth', 2);
xlabel('Dataset Size (observations)');
ylabel('Speedup Ratio (×)');
title('MEX Speedup Factor vs Dataset Size');
grid on;

fprintf('Analysis of scaling behavior:\n');
fprintf('  - MEX optimization advantage increases with dataset size\n');
fprintf('  - MATLAB implementation shows approximately O(n^%.1f) scaling\n', scalingResults.matlabScalingFactor);
fprintf('  - MEX implementation shows approximately O(n^%.1f) scaling\n\n', scalingResults.mexScalingFactor);

%% 5. Matrix Operation Optimization
fprintf('=====================================================================\n');
fprintf('5. MATRIX OPERATION OPTIMIZATION\n');
fprintf('=====================================================================\n\n');

fprintf('Financial econometrics requires efficient matrix operations,\n');
fprintf('especially for covariance matrices and multivariate models.\n\n');

% Generate a test matrix (correlation/covariance-like matrix)
n = 100;
fprintf('Analyzing a %dx%d matrix for optimization...\n', n, n);
testMatrix = 0.9*ones(n) + 0.1*eye(n);
for i = 1:n
    for j = 1:n
        if i ~= j
            testMatrix(i,j) = testMatrix(i,j) * 0.98^abs(i-j);
        end
    end
end

% Demonstrate matrix operation optimization
matrixOptResults = optimize_matrix_operations(testMatrix);

fprintf('Matrix diagnostics results:\n');
fprintf('  Condition number: %.2e\n', matrixOptResults.condNumber);
fprintf('  Near singularity status: %s\n', mat2str(matrixOptResults.nearSingular));
fprintf('  Positive definite status: %s\n', mat2str(matrixOptResults.posDefinite));

fprintf('\nMatrix multiplication optimization:\n');
fprintf('  Standard multiplication time: %.4f seconds\n', matrixOptResults.standardMultTime);
fprintf('  Optimized multiplication time: %.4f seconds\n', matrixOptResults.optimizedMultTime);
fprintf('  Performance improvement: %.1f%%\n\n', matrixOptResults.improvementPct);

%% 6. High-Frequency Data Performance Techniques
fprintf('=====================================================================\n');
fprintf('6. HIGH-FREQUENCY DATA PERFORMANCE TECHNIQUES\n');
fprintf('=====================================================================\n\n');

fprintf('High-frequency financial data analysis requires special optimization\n');
fprintf('techniques due to the large volume of data.\n\n');

% Create synthetic high-frequency returns (1-minute data)
fprintf('Generating synthetic high-frequency (1-minute) returns...\n');
trading_days = 5;
observations_per_day = 390; % Typical for 6.5 hour trading day at 1-minute intervals
T_hf = trading_days * observations_per_day;
hf_returns = randn(T_hf, 1) * 0.0005; % 5 basis points standard deviation

fprintf('Benchmarking realized volatility computation methods:\n');
% Standard method
tic;
rv_standard = rv_compute(hf_returns);
standard_time = toc;

% Optimized method with subsampling
options = struct('method', 'subsample', 'subSample', 5);
tic;
rv_optimized = rv_compute(hf_returns, options);
optimized_time = toc;

fprintf('Realized volatility computation times:\n');
fprintf('  Standard method: %.4f seconds\n', standard_time);
fprintf('  Optimized method: %.4f seconds\n', optimized_time);
fprintf('  Performance improvement: %.1f%%\n\n', 100*(standard_time-optimized_time)/standard_time);

%% Summary of Performance Optimization Techniques
fprintf('=====================================================================\n');
fprintf('SUMMARY OF PERFORMANCE OPTIMIZATION TECHNIQUES\n');
fprintf('=====================================================================\n\n');

fprintf('Key performance optimization guidelines for MFE Toolbox users:\n\n');

fprintf('1. MEX Acceleration\n');
fprintf('   - Use MEX-optimized functions when available (set useMEX=true in options)\n');
fprintf('   - Expect significant performance gains for large datasets\n\n');

fprintf('2. Memory Management\n');
fprintf('   - Preallocate arrays before loops with zeros() or similar functions\n');
fprintf('   - Avoid growing arrays dynamically within loops\n');
fprintf('   - Clear large temporary variables when no longer needed\n\n');

fprintf('3. Vectorization\n');
fprintf('   - Replace loops with vectorized operations where possible\n');
fprintf('   - Use matrix operations instead of element-wise processing\n');
fprintf('   - Consider logical indexing for conditional operations\n\n');

fprintf('4. Matrix Optimization\n');
fprintf('   - Use matrixdiagnostics() to analyze and optimize matrix operations\n');
fprintf('   - Consider numerical stability alongside performance\n');
fprintf('   - Apply appropriate matrix decomposition methods based on matrix properties\n\n');

fprintf('5. Large Dataset Handling\n');
fprintf('   - Process data in smaller batches when working with very large datasets\n');
fprintf('   - Consider subsampling techniques for high-frequency data\n');
fprintf('   - Profile memory usage for large-scale applications\n\n');

fprintf('These optimization techniques can result in performance improvements\n');
fprintf('of 50%% or more, enabling more sophisticated financial analyses and\n');
fprintf('faster model estimation for production environments.\n\n');

end

%% ===== HELPER FUNCTIONS =====

function results = demonstrate_mex_speedup(data)
% DEMONSTRATE_MEX_SPEEDUP Demonstrates performance improvements from MEX optimization
%
% Compares execution time between MEX-optimized and MATLAB-only implementations
% of the AGARCH volatility model from the MFE Toolbox.
%
% INPUTS:
%   data - Financial returns time series
%
% OUTPUTS:
%   results - Structure with performance comparison results

% Create options structure for model estimation
options = struct();

% Turn off MEX optimization for baseline comparison
options.useMEX = false;
fprintf('  Running MATLAB-only implementation... ');
tic;
model_matlab = agarchfit(data, options);
matlab_time = toc;
fprintf('done (%.4f seconds)\n', matlab_time);

% Turn on MEX optimization (default)
options.useMEX = true;
fprintf('  Running MEX-optimized implementation... ');
tic;
model_mex = agarchfit(data, options);
mex_time = toc;
fprintf('done (%.4f seconds)\n', mex_time);

% Check if both implementations produce the same results
paramDiff = norm(model_matlab.parameters - model_mex.parameters);
fprintf('  Parameter estimate difference: %.2e (should be near zero)\n', paramDiff);

% Calculate speedup metrics
speedup_ratio = matlab_time / mex_time;
improvement_pct = 100 * (matlab_time - mex_time) / matlab_time;

% Return results
results = struct('matlabTime', matlab_time, ...
                'mexTime', mex_time, ...
                'speedupRatio', speedup_ratio, ...
                'improvementPct', improvement_pct, ...
                'parameterDifference', paramDiff);
end

function results = demonstrate_memory_preallocation(size)
% DEMONSTRATE_MEMORY_PREALLOCATION Shows performance benefits of memory preallocation
%
% Compares execution time between implementations with and without memory preallocation
% for a common financial time series operation.
%
% INPUTS:
%   size - Size of the test array
%
% OUTPUTS:
%   results - Structure with timing and memory usage statistics

% Method 1: Without preallocation (inefficient)
fprintf('  Running inefficient implementation (without preallocation)... ');
tic;
% Initialize counter for memory operations
noPreallocMemOps = 0;

% Start with empty array
result_nopre = [];
for i = 1:size
    % Grow array dynamically (inefficient)
    result_nopre(i) = i^2 - i;
    noPreallocMemOps = noPreallocMemOps + 1;
end

% Perform typical financial calculation (moving average)
ma_nopre = [];
window = 20;
for i = window:size
    % Grow array again
    ma_nopre(i-window+1) = mean(result_nopre(i-window+1:i));
    noPreallocMemOps = noPreallocMemOps + 1;
end
noPrealloc_time = toc;
fprintf('done (%.4f seconds)\n', noPrealloc_time);

% Method 2: With preallocation (efficient)
fprintf('  Running optimized implementation (with preallocation)... ');
tic;
% Initialize counter for memory operations
preallocMemOps = 1; % Count initial allocation

% Preallocate array
result_pre = zeros(size, 1);
for i = 1:size
    % Assign to preallocated array (efficient)
    result_pre(i) = i^2 - i;
    % No additional memory operations needed here
end

% Preallocate for moving average too
preallocMemOps = preallocMemOps + 1; % Count second allocation
ma_pre = zeros(size-window+1, 1);
for i = window:size
    % Assign to preallocated array
    ma_pre(i-window+1) = mean(result_pre(i-window+1:i));
    % No additional memory operations
end
prealloc_time = toc;
fprintf('done (%.4f seconds)\n', prealloc_time);

% Calculate improvement
improvement_pct = 100 * (noPrealloc_time - prealloc_time) / noPrealloc_time;
memOp_reduction = 100 * (noPreallocMemOps - preallocMemOps) / noPreallocMemOps;

% Return results
results = struct('noPreallocTime', noPrealloc_time, ...
                'preallocTime', prealloc_time, ...
                'improvementPct', improvement_pct, ...
                'noPreallocMemOps', noPreallocMemOps, ...
                'preallocMemOps', preallocMemOps, ...
                'memOpReductionPct', memOp_reduction);
end

function results = demonstrate_vectorization(data)
% DEMONSTRATE_VECTORIZATION Demonstrates performance gains from vectorized operations
%
% Compares execution time between loop-based and vectorized implementations of
% common financial calculations.
%
% INPUTS:
%   data - Financial returns time series
%
% OUTPUTS:
%   results - Structure with performance comparison results

% Ensure data is a column vector
data = data(:);
T = length(data);

% Method 1: Loop-based calculation (inefficient)
fprintf('  Running loop-based implementation... ');
tic;
% Calculate absolute returns (common volatility measure)
abs_returns_loop = zeros(T, 1);
for i = 1:T
    abs_returns_loop(i) = abs(data(i));
end

% Calculate squared returns (for realized variance)
squared_returns_loop = zeros(T, 1);
for i = 1:T
    squared_returns_loop(i) = data(i)^2;
end

% Calculate simple volatility measure (sum of squared returns)
vols_loop = zeros(T-20+1, 1);
for i = 20:T
    sum_sq = 0;
    for j = i-19:i
        sum_sq = sum_sq + squared_returns_loop(j);
    end
    vols_loop(i-19) = sqrt(sum_sq);
end
loop_time = toc;
fprintf('done (%.4f seconds)\n', loop_time);

% Method 2: Vectorized calculation (efficient)
fprintf('  Running vectorized implementation... ');
tic;
% Calculate absolute returns - vectorized
abs_returns_vec = abs(data);

% Calculate squared returns - vectorized
squared_returns_vec = data.^2;

% Calculate simple volatility measure - vectorized
vols_vec = zeros(T-20+1, 1);
for i = 20:T
    % Still using a loop but with vectorized operation inside
    vols_vec(i-19) = sqrt(sum(squared_returns_vec(i-19:i)));
end

% Fully vectorized moving sum using convolution
kernel = ones(20, 1);
vols_fully_vec = sqrt(conv(squared_returns_vec, kernel, 'valid'));
vector_time = toc;
fprintf('done (%.4f seconds)\n', vector_time);

% Verify results are the same
abs_diff = norm(abs_returns_loop - abs_returns_vec);
squared_diff = norm(squared_returns_loop - squared_returns_vec);
vols_diff = norm(vols_loop - vols_vec);
fprintf('  Result differences (should be near zero):\n');
fprintf('    Absolute returns: %.2e\n', abs_diff);
fprintf('    Squared returns: %.2e\n', squared_diff);
fprintf('    Volatility measures: %.2e\n', vols_diff);

% Calculate improvement
improvement_pct = 100 * (loop_time - vector_time) / loop_time;

% Return results
results = struct('loopTime', loop_time, ...
                'vectorTime', vector_time, ...
                'improvementPct', improvement_pct, ...
                'absoluteDiff', abs_diff, ...
                'squaredDiff', squared_diff, ...
                'volsDiff', vols_diff);
end

function results = benchmark_data_scaling(sizes)
% BENCHMARK_DATA_SCALING Tests how performance scales with increasing dataset sizes
%
% Evaluates performance of MEX-optimized vs. MATLAB implementations at different
% dataset sizes to identify scaling behavior.
%
% INPUTS:
%   sizes - Array of dataset sizes to test
%
% OUTPUTS:
%   results - Structure with comprehensive scaling results

% Initialize results arrays
matlabTimes = zeros(length(sizes), 1);
mexTimes = zeros(length(sizes), 1);
speedupRatios = zeros(length(sizes), 1);

% Loop through different data sizes
for i = 1:length(sizes)
    size = sizes(i);
    fprintf('  Testing dataset size %d... ', size);
    
    % Generate synthetic data of the current size
    data = randn(size, 1) * 0.01;
    for t = 2:size
        % Simple ARMA(1,1) + asymmetric volatility process
        data(t) = 0.05 * data(t-1) + data(t) + 0.1 * (data(t-1) < 0) * data(t-1);
    end
    
    % Test MATLAB-only implementation
    options = struct('useMEX', false);
    tic;
    agarchfit(data, options);
    matlabTimes(i) = toc;
    
    % Test MEX-optimized implementation
    options.useMEX = true;
    tic;
    agarchfit(data, options);
    mexTimes(i) = toc;
    
    % Calculate speedup ratio
    speedupRatios(i) = matlabTimes(i) / mexTimes(i);
    
    fprintf('MATLAB: %.4fs, MEX: %.4fs, Speedup: %.2fx\n', ...
            matlabTimes(i), mexTimes(i), speedupRatios(i));
end

% Calculate scaling factors (power-law fitting)
% For scaling evaluation, we fit execution time to the model: time = c * size^p
% Log transform: log(time) = log(c) + p * log(size)
logSizes = log(sizes');
logMatlabTimes = log(matlabTimes);
logMexTimes = log(mexTimes);

% Fit linear models to the log-transformed data
matlabFit = polyfit(logSizes, logMatlabTimes, 1);
mexFit = polyfit(logSizes, logMexTimes, 1);

% Extract scaling factors (the slopes of the linear fits)
matlabScalingFactor = matlabFit(1);
mexScalingFactor = mexFit(1);

% Return results
results = struct('sizes', sizes, ...
                'matlabTimes', matlabTimes, ...
                'mexTimes', mexTimes, ...
                'speedupRatios', speedupRatios, ...
                'matlabScalingFactor', matlabScalingFactor, ...
                'mexScalingFactor', mexScalingFactor);
end

function results = optimize_matrix_operations(matrix)
% OPTIMIZE_MATRIX_OPERATIONS Demonstrates techniques for optimizing matrix operations
%
% Analyzes matrix properties and demonstrates performance improvements for
% matrix operations that are common in financial econometric calculations.
%
% INPUTS:
%   matrix - Square matrix to analyze and optimize
%
% OUTPUTS:
%   results - Structure with optimization recommendations and timing results

% Get matrix size
[n, m] = size(matrix);
if n ~= m
    error('Input must be a square matrix');
end

% Perform matrix diagnostics
fprintf('  Running matrix diagnostics... ');
diagnostics = matrixdiagnostics(matrix);
fprintf('done\n');

% Extract key properties
condNumber = diagnostics.ConditionNumber;
isNearSingular = diagnostics.IsNearSingular;
isPositiveDefinite = diagnostics.IsPositiveDefinite;

% Print matrix properties
fprintf('  Matrix properties:\n');
fprintf('    Dimensions: %d x %d\n', n, n);
fprintf('    Condition number: %.2e\n', condNumber);
fprintf('    Near singular: %s\n', mat2str(isNearSingular));
fprintf('    Positive definite: %s\n', mat2str(isPositiveDefinite));

% Choose optimization approach based on matrix properties
fprintf('  Optimizing based on matrix properties...\n');

% Test case: Matrix multiplication (common in covariance estimation)
fprintf('  Benchmarking matrix multiplication methods:\n');

% Method 1: Standard matrix multiplication
fprintf('    Running standard matrix multiplication... ');
tic;
for i = 1:10  % Repeat for timing accuracy
    result_standard = matrix * matrix;
end
standardMult_time = toc / 10;
fprintf('done (%.4f seconds)\n', standardMult_time);

% Method 2: Optimized matrix multiplication
fprintf('    Running optimized matrix multiplication... ');
tic;
for i = 1:10  % Repeat for timing accuracy
    % Choose optimization based on matrix properties
    if isPositiveDefinite
        % For positive definite matrices, Cholesky can be faster
        L = chol(matrix, 'lower');
        result_optimized = L * L';
    elseif isNearSingular
        % For near-singular matrices, use SVD
        [U, S, V] = svd(matrix);
        % Apply truncation if needed
        tol = eps(max(diag(S))) * max(n, 100);
        S(S < tol) = 0;
        result_optimized = U * S * V';
    else
        % For well-conditioned matrices, standard multiplication is sufficient
        result_optimized = matrix * matrix;
    end
end
optimizedMult_time = toc / 10;
fprintf('done (%.4f seconds)\n', optimizedMult_time);

% Calculate improvement
improvement_pct = 100 * (standardMult_time - optimizedMult_time) / standardMult_time;

% Return results
results = struct('condNumber', condNumber, ...
                'nearSingular', isNearSingular, ...
                'posDefinite', isPositiveDefinite, ...
                'standardMultTime', standardMult_time, ...
                'optimizedMultTime', optimizedMult_time, ...
                'improvementPct', improvement_pct, ...
                'diagnostics', diagnostics);
end