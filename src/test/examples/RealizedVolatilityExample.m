%% ========================================================================
%  REALIZED VOLATILITY ANALYSIS EXAMPLE
%  ========================================================================
%  This example demonstrates the usage of the MFE Toolbox's realized
%  volatility functions for analyzing high-frequency financial data.
%  It covers the complete workflow from loading data to computing various
%  realized volatility measures, detecting jumps, and visualizing results.
%
%  Functions demonstrated:
%  - rv_compute: Standard realized volatility
%  - bv_compute: Bipower variation (jump-robust)
%  - rv_kernel: Kernel-based realized volatility (noise-robust)
%  - realized_spectrum: Spectral realized volatility
%  - jump_test: Statistical tests for price jumps
%
%  Copyright: Kevin Sheppard
%  kevin.sheppard@economics.ox.ac.uk
%  ========================================================================

%% Introduction to Realized Volatility
% Realized volatility (RV) measures the quadratic variation of financial asset
% prices using high-frequency data. It provides a more accurate estimate of
% the true volatility compared to traditional low-frequency methods.
%
% The basic RV estimator is the sum of squared intraday returns. However,
% this estimator is affected by market microstructure noise and jumps.
% More advanced estimators address these issues:
%
% 1. Bipower Variation (BV): Robust to jumps, but affected by noise
% 2. Kernel-based RV: Robust to microstructure noise
% 3. Spectral RV: Frequency domain approach for noise robustness
%
% Jump tests help identify discontinuities in price processes, which are
% important for risk management and option pricing.

disp('==================================================================');
disp('              REALIZED VOLATILITY ANALYSIS EXAMPLE                ');
disp('==================================================================');
disp(' ');
disp('This example demonstrates high-frequency volatility estimation');
disp('methods implemented in the MFE Toolbox.');
disp(' ');

%% Data Loading
% Load high-frequency returns data from MAT file
% This dataset contains 1-minute return data for multiple assets
disp('Loading high-frequency data...');

% Load the data file
load('../../test/data/high_frequency_data.mat');

% Extract returns, timestamps, and reference values
returns = high_frequency_data.high_frequency_returns;
timestamps = high_frequency_data.timestamps;
reference_values = high_frequency_data.reference_values;

% Get dimensions of the returns data
[nObs, nAssets] = size(returns);

% Display basic information about the dataset
disp(['Dataset contains ' num2str(nObs) ' observations for ' num2str(nAssets) ' assets']);
disp(['Date range: ' datestr(timestamps(1)) ' to ' datestr(timestamps(end))]);
disp(['Sampling frequency: ' num2str(high_frequency_data.sampling_frequency) ' observations per day']);
disp(' ');

% For this example, we'll focus on the first asset
assetIdx = 1;
assetReturns = returns(:, assetIdx);

%% Basic Realized Volatility Calculation
% The standard realized volatility is computed as the sum of squared returns
disp('Computing standard realized volatility...');

% Configure options for realized volatility calculation
rvOptions = struct();
rvOptions.scale = 1; % No scaling (annualization would use values like 252)

% Calculate realized volatility
rv = rv_compute(assetReturns, rvOptions);

% Display the results
disp(['Realized Volatility (variance): ' num2str(rv)]);
disp(['Realized Volatility (std dev): ' num2str(sqrt(rv))]);

% Compare with reference value if available
if isfield(reference_values, 'realized_volatility')
    disp(['Reference RV value: ' num2str(reference_values.realized_volatility(assetIdx))]);
    disp(['Difference: ' num2str(rv - reference_values.realized_volatility(assetIdx))]);
end
disp(' ');

% Plot the high-frequency returns
figure;
subplot(2,1,1);
plot(timestamps(1:length(assetReturns)), assetReturns);
title('High-Frequency Returns');
xlabel('Time');
ylabel('Returns');
grid on;

% Plot the squared returns to visualize volatility clustering
subplot(2,1,2);
plot(timestamps(1:length(assetReturns)), assetReturns.^2);
title('Squared Returns');
xlabel('Time');
ylabel('Squared Returns');
grid on;

%% Bipower Variation Calculation
% Bipower variation (BV) is a jump-robust estimator that uses the product 
% of adjacent absolute returns, making it less sensitive to large jumps
disp('Computing bipower variation (jump-robust estimator)...');

% Configure options for bipower variation calculation
bvOptions = struct();

% Calculate bipower variation
bv = bv_compute(assetReturns, bvOptions);

% Display the results
disp(['Bipower Variation (variance): ' num2str(bv)]);
disp(['Bipower Variation (std dev): ' num2str(sqrt(bv))]);

% Compare with reference value if available
if isfield(reference_values, 'bipower_variation')
    disp(['Reference BV value: ' num2str(reference_values.bipower_variation(assetIdx))]);
    disp(['Difference: ' num2str(bv - reference_values.bipower_variation(assetIdx))]);
end

% Compare RV and BV
disp(['RV/BV Ratio: ' num2str(rv/bv)]);
disp(['Difference (RV - BV): ' num2str(rv - bv)]);
disp(' ');

%% Kernel-based Realized Volatility
% Kernel-based estimators are robust to market microstructure noise
% They apply a weighting kernel to the realized autocovariances
disp('Computing kernel-based realized volatility (noise-robust estimator)...');

% Try different kernel types and bandwidths
kernelTypes = {'Bartlett-Parzen', 'Quadratic', 'Cubic', 'Tukey-Hanning'};
kernelRV = zeros(length(kernelTypes), 1);

for i = 1:length(kernelTypes)
    % Configure kernel options
    kernelOptions = struct();
    kernelOptions.kernelType = kernelTypes{i};
    kernelOptions.bandwidth = 10; % Fixed bandwidth for comparison
    
    % Calculate kernel-based realized volatility
    kernelRV(i) = rv_kernel(assetReturns, kernelOptions);
    
    % Display results
    disp([kernelTypes{i} ' Kernel RV: ' num2str(kernelRV(i))]);
end

% Compare with reference value if available
if isfield(reference_values, 'kernel_estimates')
    disp(['Reference Kernel RV value: ' num2str(reference_values.kernel_estimates(assetIdx))]);
end
disp(' ');

% Plot the kernel-based RV estimates
figure;
bar(1:length(kernelTypes), kernelRV);
set(gca, 'XTickLabel', kernelTypes);
title('Kernel-Based Realized Volatility Estimates');
ylabel('Realized Variance');
grid on;

%% Realized Volatility Spectrum
% Spectral methods analyze volatility in the frequency domain
% They can identify noise patterns and provide robust estimators
disp('Computing realized volatility spectrum...');

% Configure spectral options
spectrumOptions = struct();
spectrumOptions.windowType = 'Parzen';
spectrumOptions.compareKernel = true;
spectrumOptions.compareBenchmark = true;

% Calculate spectral realized volatility
[spectrumRV, diagnostics] = realized_spectrum(assetReturns, spectrumOptions);

% Display results
disp(['Spectral Realized Volatility: ' num2str(spectrumRV)]);
if isfield(diagnostics, 'benchmark')
    disp(['Benchmark RV: ' num2str(diagnostics.benchmark)]);
end
if isfield(diagnostics, 'kernel')
    disp(['Kernel RV: ' num2str(diagnostics.kernel)]);
end
disp(' ');

% Plot the spectral density
figure;
plot(diagnostics.frequencies{1}, diagnostics.spectrum{1});
title('Realized Volatility Spectrum');
xlabel('Frequency');
ylabel('Spectral Density');
grid on;

%% Jump Detection
% Jump tests determine whether price processes contain statistically 
% significant jumps, which have implications for risk modeling
disp('Detecting jumps in high-frequency price data...');

% Configure jump test options
jumpOptions = struct();
jumpOptions.alpha = 0.05; % 5% significance level

% Perform jump test
jumpResults = jump_test(assetReturns, jumpOptions);

% Display jump test results
disp(['Jump Test Z-Statistic: ' num2str(jumpResults.zStatistic)]);
disp(['Jump Test p-value: ' num2str(jumpResults.pValue)]);

% Display jump detection at different significance levels
disp('Jump Detection:');
for i = 1:length(jumpResults.significanceLevels)
    alpha = jumpResults.significanceLevels(i);
    detected = jumpResults.jumpDetected(i);
    disp(['  at ' num2str(alpha*100) '% significance: ' num2str(detected)]);
end

% Display jump and continuous components
disp(['Jump Component (% of total): ' num2str(jumpResults.jumpComponent*100) '%']);
disp(['Continuous Component (% of total): ' num2str(jumpResults.contComponent/rv*100) '%']);
disp(' ');

% Plot the jump test results
figure;
subplot(2,1,1);
bar([jumpResults.contComponent, jumpResults.jumpComponent*rv]);
set(gca, 'XTickLabel', {'Continuous', 'Jump'});
title('Volatility Decomposition');
ylabel('Variance Component');
grid on;

subplot(2,1,2);
plot([1 2], [jumpResults.rv, jumpResults.bv], 'o-');
hold on;
plot([1 2], [jumpResults.rv, jumpResults.bv], 'rs', 'MarkerSize', 10);
set(gca, 'XTickLabel', {'RV', 'BV'});
title('Realized Volatility vs. Bipower Variation');
grid on;
ylim([min(jumpResults.bv)*0.9, max(jumpResults.rv)*1.1]);

%% Comparative Analysis
% Compare different realized volatility estimators
disp('Comparing different realized volatility estimators...');

% Collect all estimators 
estimatorNames = {'Standard RV', 'Bipower Variation', ...
                 'Kernel RV (Bartlett)', 'Kernel RV (Quadratic)', ...
                 'Kernel RV (Cubic)', 'Kernel RV (Tukey)', ...
                 'Spectral RV'};
             
estimatorValues = [rv, bv, kernelRV', spectrumRV];

% Create a bar chart comparing all estimators
figure;
bar(estimatorValues);
set(gca, 'XTickLabel', estimatorNames);
title('Comparison of Realized Volatility Estimators');
ylabel('Realized Variance');
grid on;
xtickangle(45);

% Calculate percentage differences from the standard RV
diffFromRV = (estimatorValues - rv) / rv * 100;
disp('Percentage difference from standard RV:');
for i = 2:length(estimatorNames)
    disp(['  ' estimatorNames{i} ': ' num2str(diffFromRV(i)) '%']);
end
disp(' ');

%% Conclusion
% Summarize findings and provide recommendations
disp('==================================================================');
disp('CONCLUSION');
disp('==================================================================');
disp('This example demonstrated several realized volatility estimators:');
disp('1. Standard RV - Simple sum of squared returns');
disp('   * Fast and easy to compute');
disp('   * Sensitive to jumps and microstructure noise');
disp(' ');
disp('2. Bipower Variation - Jump-robust estimator');
disp('   * Robust to price jumps');
disp('   * Still affected by microstructure noise');
disp(' ');
disp('3. Kernel-based RV - Noise-robust estimator');
disp('   * Robust to microstructure noise through kernel weighting');
disp('   * Different kernels offer varying trade-offs in bias/variance');
disp(' ');
disp('4. Spectral RV - Frequency domain approach');
disp('   * Robust to specific noise patterns');
disp('   * Good for analyzing cyclical volatility components');
disp(' ');
disp('Key findings:');
if jumpResults.jumpDetected(2)  % At 5% significance
    disp('- Significant jumps detected in the price process');
    disp('- Jump component represents a substantial portion of total volatility');
    disp('- Consider using robust estimators for risk management');
else
    disp('- No significant jumps detected at 5% level');
    disp('- Standard RV may be sufficient for this series');
end

% Add recommendations based on microstructure noise evidence
rvBvDiff = abs(rv - bv) / rv * 100;
rvKernelDiff = abs(rv - kernelRV(1)) / rv * 100;

if rvKernelDiff > 10
    disp('- Substantial difference between standard RV and kernel RV');
    disp('  suggests presence of significant microstructure noise');
    disp('- Recommended: Use kernel or spectral estimators for this data');
else
    disp('- Limited evidence of microstructure noise');
    disp('- Standard RV or BV may be adequate for this sampling frequency');
end

disp('==================================================================');