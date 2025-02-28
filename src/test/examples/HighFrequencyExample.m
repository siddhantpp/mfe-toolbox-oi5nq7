%% HighFrequencyExample.m
% This script demonstrates advanced high-frequency financial data analysis capabilities
% of the MFE Toolbox. It includes examples of various volatility estimation techniques,
% jump detection methods, and spectral analysis for high-frequency financial data.
%
% The script showcases:
% 1. Realized volatility computation
% 2. Jump-robust volatility estimation using bipower variation
% 3. Testing for price jumps in high-frequency data
% 4. Kernel-based volatility estimators robust to microstructure noise
% 5. Spectral analysis of high-frequency return data
%
% This is an educational example with detailed explanations and visualizations
% to help understand high-frequency financial econometrics methods.

% Global constants
DAILY_SESSIONS = 252; % Typical number of trading days in a year
FIGURE_POSITION = [100, 100, 800, 600]; % Standard figure position and size

function runHighFrequencyExample()
    % Main function that demonstrates high-frequency financial data analysis using the MFE Toolbox
    
    % Display informative header
    disp('================================================================');
    disp('  HIGH-FREQUENCY FINANCIAL DATA ANALYSIS EXAMPLE');
    disp('  Using MFE Toolbox for Advanced Econometric Analysis');
    disp('================================================================');
    disp(' ');
    
    % Load high-frequency data from test data directory
    disp('Loading high-frequency financial data...');
    try
        load('../data/high_frequency_data.mat');
        
        % Extract data components
        returns = high_frequency_data.high_frequency_returns;
        timestamps = high_frequency_data.timestamps;
        reference_values = high_frequency_data.reference_values;
        sampling_frequency = high_frequency_data.sampling_frequency;
        
        % Display basic information about the dataset
        [T, N] = size(returns);
        disp(['Dataset contains ' num2str(T) ' observations for ' num2str(N) ' assets']);
        disp(['Sampling frequency: ' num2str(sampling_frequency) ' observations per day']);
        disp(['Date range: ' datestr(timestamps(1)) ' to ' datestr(timestamps(end))]);
        disp(' ');
        
        % Part 1: Standard Realized Volatility Estimation
        disp('=== PART 1: STANDARD REALIZED VOLATILITY ESTIMATION ===');
        disp('Realized volatility (RV) is computed as the sum of squared intraday returns.');
        disp('Under suitable assumptions, RV provides a consistent estimate of integrated');
        disp('volatility in continuous-time stochastic volatility models.');
        disp(' ');
        
        % Calculate standard realized volatility
        rv = rv_compute(returns);
        
        % Display realized volatility results
        disp('Realized volatility estimates (variance):');
        for i = 1:N
            disp(['  Asset ' num2str(i) ': ' num2str(rv(i), '%.6f') ...
                  ' (Daily volatility: ' num2str(sqrt(rv(i) * DAILY_SESSIONS) * 100, '%.2f') '%)']);
        end
        disp(' ');
        
        % Part 2: Jump-Robust Volatility Estimation
        disp('=== PART 2: JUMP-ROBUST VOLATILITY ESTIMATION ===');
        disp('Bipower variation (BV) is a jump-robust estimator of integrated volatility.');
        disp('It uses products of adjacent absolute returns to filter out jumps while');
        disp('still capturing the continuous component of price variation.');
        disp(' ');
        
        % Calculate bipower variation (jump-robust estimator)
        bv = bv_compute(returns);
        
        % Display bipower variation results
        disp('Bipower variation estimates (variance):');
        for i = 1:N
            disp(['  Asset ' num2str(i) ': ' num2str(bv(i), '%.6f') ...
                  ' (Daily volatility: ' num2str(sqrt(bv(i) * DAILY_SESSIONS) * 100, '%.2f') '%)']);
        end
        disp(' ');
        
        % Compare RV and BV to assess potential jump components
        disp('Comparison of RV and BV for jump component analysis:');
        disp('   RV captures both continuous and jump components of price variation,');
        disp('   while BV is robust to jumps. Therefore, the difference RV-BV can');
        disp('   be used to estimate the jump component of price variation.');
        disp(' ');
        
        for i = 1:N
            jumpComponent = max(0, rv(i) - bv(i));
            jumpPercent = 100 * jumpComponent / rv(i);
            disp(['   Asset ' num2str(i) ': Jump component is ' num2str(jumpPercent, '%.2f') ...
                  '% of total variance']);
        end
        disp(' ');
        
        % Part 3: Jump Detection Tests
        disp('=== PART 3: JUMP DETECTION TESTS ===');
        disp('The ratio statistic from Barndorff-Nielsen and Shephard (2004, 2006)');
        disp('provides a formal statistical test for detecting jumps in high-frequency data.');
        disp('The test is based on the RV/BV ratio, which converges to 1 in the absence of jumps.');
        disp(' ');
        
        % Run jump detection test
        jumpResults = jump_test(returns);
        
        % Display jump test results
        disp('Jump test results:');
        disp('  - Z-statistics (values > 1.96 indicate jumps at 5% level):');
        disp(['    ' num2str(jumpResults.zStatistic, '%.4f')]);
        
        disp('  - p-values:');
        disp(['    ' num2str(jumpResults.pValue, '%.4f')]);
        
        disp('  - Jump detection summary:');
        for i = 1:3
            level = jumpResults.significanceLevels(i);
            detected = sum(jumpResults.jumpDetected(i,:));
            disp(['    At ' num2str(level*100) '% significance: ' ...
                  num2str(detected) ' out of ' num2str(N) ' assets show evidence of jumps']);
        end
        disp(' ');
        
        % Analyze jump components in more detail
        jumpAnalysis = analyzeJumpComponents(returns);
        
        % Part 4: Noise-Robust Volatility Estimation
        disp('=== PART 4: NOISE-ROBUST VOLATILITY ESTIMATION ===');
        disp('Microstructure noise can bias standard RV estimators, especially at very');
        disp('high sampling frequencies. Kernel-based realized volatility estimators use');
        disp('weighted autocovariances to provide noise-robust estimates.');
        disp(' ');
        
        % Compare different kernel estimators
        kernelResults = compareKernelEstimators(returns);
        
        % Display kernel estimator results for first asset
        disp('Kernel-based realized volatility estimates for Asset 1:');
        for i = 1:length(kernelResults.kernelTypes)
            for j = 1:length(kernelResults.bandwidths)
                kernelType = kernelResults.kernelTypes{i};
                bandwidth = kernelResults.bandwidths(j);
                rvKernel = kernelResults.estimates(i,j,1);
                
                disp(['  - ' kernelType ' kernel with bandwidth ' num2str(bandwidth) ...
                    ': ' num2str(rvKernel, '%.6f') ' (' ...
                    num2str(sqrt(rvKernel * DAILY_SESSIONS) * 100, '%.2f') '% annualized)']);
            end
        end
        disp(' ');
        
        % Part 5: Spectral Analysis
        disp('=== PART 5: SPECTRAL ANALYSIS OF HIGH-FREQUENCY RETURNS ===');
        disp('Spectral methods estimate integrated volatility in the frequency domain,');
        disp('which can effectively filter out high-frequency noise components while');
        disp('preserving the genuine volatility signal.');
        disp(' ');
        
        % Set options for spectral analysis
        spectrumOptions = struct();
        spectrumOptions.compareBenchmark = true;
        spectrumOptions.compareKernel = true;
        
        % Compute realized spectrum for Asset 1
        [spectrumRV, spectrumDiag] = realized_spectrum(returns(:,1), spectrumOptions);
        
        % Display spectral analysis results
        disp('Spectral realized volatility estimates (Asset 1):');
        disp(['  - Spectral RV: ' num2str(spectrumRV, '%.6f') ...
              ' (' num2str(sqrt(spectrumRV * DAILY_SESSIONS) * 100, '%.2f') '% annualized)']);
        
        if spectrumOptions.compareBenchmark
            disp(['  - Standard RV: ' num2str(spectrumDiag.benchmark, '%.6f') ...
                  ' (' num2str(sqrt(spectrumDiag.benchmark * DAILY_SESSIONS) * 100, '%.2f') '% annualized)']);
        end
        
        if spectrumOptions.compareKernel
            disp(['  - Kernel RV:   ' num2str(spectrumDiag.kernel, '%.6f') ...
                  ' (' num2str(sqrt(spectrumDiag.kernel * DAILY_SESSIONS) * 100, '%.2f') '% annualized)']);
        end
        disp(' ');
        
        % Create comprehensive results structure
        results = struct();
        results.rv = rv;
        results.bv = bv;
        results.jumpTest = jumpResults;
        results.jumpAnalysis = jumpAnalysis;
        results.kernelResults = kernelResults;
        results.spectrumRV = spectrumRV;
        results.spectrumDiag = spectrumDiag;
        
        % Create visualizations
        disp('Creating visualizations of results...');
        figHandles = createVolatilityPlots(results, timestamps, returns);
        disp(['Created ' num2str(length(figHandles)) ' figures with visualizations']);
        disp(' ');
        
        % Summarize findings
        disp('================================================================');
        disp('SUMMARY OF HIGH-FREQUENCY ANALYSIS FINDINGS:');
        disp('----------------------------------------------------------------');
        
        % Volatility summary
        disp('Volatility Estimates (Asset 1, annualized):');
        disp(['- Standard RV:         ' num2str(sqrt(rv(1) * DAILY_SESSIONS) * 100, '%.2f') '%']);
        disp(['- Jump-robust BV:      ' num2str(sqrt(bv(1) * DAILY_SESSIONS) * 100, '%.2f') '%']);
        disp(['- Kernel-based RV:     ' num2str(sqrt(kernelResults.estimates(1,1,1) * DAILY_SESSIONS) * 100, '%.2f') '%']);
        disp(['- Spectral RV:         ' num2str(sqrt(spectrumRV * DAILY_SESSIONS) * 100, '%.2f') '%']);
        disp(' ');
        
        % Jump summary
        jumpDetected = any(jumpResults.jumpDetected(2,:));
        if jumpDetected
            disp('Jump Analysis:');
            disp(['- Jump components detected in ' num2str(sum(jumpResults.jumpDetected(2,:))) ...
                  ' out of ' num2str(N) ' assets at 5% significance level']);
            disp(['- Average jump contribution: ' num2str(mean(jumpResults.jumpComponent) * 100, '%.2f') ...
                  '% of total variance']);
        else
            disp('Jump Analysis: No significant jumps detected at 5% level.');
        end
        disp(' ');
        
        % Noise robustness summary
        disp('Noise Robustness Analysis:');
        disp('- Kernel estimators provide noise-robust alternatives to standard RV');
        disp('- Spectral methods effectively filter out high-frequency noise components');
        disp('- The choice of kernel type and bandwidth affects the bias-variance tradeoff');
        
        disp('================================================================');
        
    catch e
        % Error handling
        disp('ERROR: An error occurred while running the example:');
        disp(['Message: ' e.message]);
        disp('Stack trace:');
        disp(e.stack);
    end
end

function jumpAnalysis = analyzeJumpComponents(returns)
    % Analyzes the continuous and jump components of price variation based on realized 
    % volatility and bipower variation
    
    % Calculate realized volatility and bipower variation
    rv = rv_compute(returns);
    bv = bv_compute(returns);
    
    % Run jump test
    jumpTest = jump_test(returns);
    
    % Get dimensions
    [~, N] = size(returns);
    
    % Initialize jump analysis structure
    jumpAnalysis = struct();
    jumpAnalysis.rv = rv;
    jumpAnalysis.bv = bv;
    jumpAnalysis.testStat = jumpTest.zStatistic;
    jumpAnalysis.pValue = jumpTest.pValue;
    
    % Calculate continuous component as min(RV, BV)
    jumpAnalysis.continuousComponent = min(rv, bv);
    
    % Calculate jump component as max(0, RV-BV)
    jumpAnalysis.jumpComponent = max(0, rv - bv);
    
    % Calculate jump ratio (proportion of total variance)
    jumpAnalysis.jumpRatio = jumpAnalysis.jumpComponent ./ rv;
    
    % Determine significance of jumps at different confidence levels
    jumpAnalysis.significant99 = jumpTest.jumpDetected(1,:);
    jumpAnalysis.significant95 = jumpTest.jumpDetected(2,:);
    jumpAnalysis.significant90 = jumpTest.jumpDetected(3,:);
    
    % Count significant jumps at different confidence levels
    jumpAnalysis.numSignificant99 = sum(jumpAnalysis.significant99);
    jumpAnalysis.numSignificant95 = sum(jumpAnalysis.significant95);
    jumpAnalysis.numSignificant90 = sum(jumpAnalysis.significant90);
    
    % Analyze jump characteristics
    if any(jumpAnalysis.significant95)
        % Calculate mean and max jump sizes for significant jumps
        sigJumps = jumpAnalysis.jumpComponent(jumpAnalysis.significant95);
        jumpAnalysis.meanJumpSize = mean(sigJumps);
        jumpAnalysis.maxJumpSize = max(sigJumps);
        
        % Calculate jump contribution to total variance
        jumpAnalysis.totalJumpContribution = sum(jumpAnalysis.jumpComponent) / sum(rv);
    else
        jumpAnalysis.meanJumpSize = 0;
        jumpAnalysis.maxJumpSize = 0;
        jumpAnalysis.totalJumpContribution = 0;
    end
end

function kernelResults = compareKernelEstimators(returns)
    % Compares different kernel estimators for realized volatility calculation
    % with varying bandwidth parameters
    
    % Define array of kernel types
    kernelTypes = {'Bartlett-Parzen', 'Tukey-Hanning', 'Quadratic'};
    
    % Define array of bandwidth parameters
    bandwidths = [10, 20, 30];
    
    % Get dimensions
    [~, N] = size(returns);
    
    % Initialize results array
    estimates = zeros(length(kernelTypes), length(bandwidths), N);
    
    % Compute kernel estimates for each combination
    for i = 1:length(kernelTypes)
        for j = 1:length(bandwidths)
            % Set options for this kernel configuration
            kernelOptions = struct();
            kernelOptions.kernelType = kernelTypes{i};
            kernelOptions.bandwidth = bandwidths(j);
            
            % Compute kernel-based realized volatility
            rv_kernel_result = rv_kernel(returns, kernelOptions);
            
            % Store the results
            estimates(i, j, :) = rv_kernel_result;
        end
    end
    
    % Calculate standard realized volatility as benchmark
    rv = rv_compute(returns);
    
    % Compute efficiency metric (ratio of kernel estimate to standard RV)
    efficiency = zeros(length(kernelTypes), length(bandwidths), N);
    for i = 1:N
        efficiency(:,:,i) = estimates(:,:,i) ./ rv(i);
    end
    
    % Find the best kernel configuration for each asset
    [~, bestIdxRow] = min(abs(efficiency - 1), [], 1:2);
    bestKernel = zeros(1, N);
    bestBandwidth = zeros(1, N);
    
    for i = 1:N
        [bestKernel(i), bestBandwidth(i)] = ind2sub([length(kernelTypes), length(bandwidths)], bestIdxRow(i));
    end
    
    % Return results in a structure
    kernelResults = struct();
    kernelResults.kernelTypes = kernelTypes;
    kernelResults.bandwidths = bandwidths;
    kernelResults.estimates = estimates;
    kernelResults.standardRV = rv;
    kernelResults.efficiency = efficiency;
    kernelResults.bestKernel = bestKernel;
    kernelResults.bestBandwidth = bestBandwidth;
end

function figHandles = createVolatilityPlots(results, timestamps, returns)
    % Creates visualizations for different realized volatility estimators and jump tests
    
    % Initialize array for figure handles
    figHandles = zeros(6, 1);
    
    % Get dimensions
    N = length(results.rv);
    
    % Figure 1: Time series plot of returns (first asset)
    figHandles(1) = figure('Position', FIGURE_POSITION);
    
    % Plot returns time series for the first asset
    % We need to align timestamps with returns appropriately
    if exist('returns', 'var')
        plot(timestamps(1:length(returns)), returns(:,1));
    else
        % Create mock data if original returns not available
        t = linspace(timestamps(1), timestamps(end), 100);
        mockReturns = 0.001 * randn(100, 1);
        plot(t, mockReturns);
    end
    
    title('High-Frequency Returns (Asset 1)');
    xlabel('Time');
    ylabel('Returns');
    datetick('x', 'HH:MM'); % Format x-axis as hours:minutes
    grid on;
    
    % Figure 2: Compare realized volatility and bipower variation
    figHandles(2) = figure('Position', FIGURE_POSITION);
    
    % Convert variance to volatility (standard deviation) and annualize
    annualizedRV = sqrt(results.rv * DAILY_SESSIONS) * 100;
    annualizedBV = sqrt(results.bv * DAILY_SESSIONS) * 100;
    
    % Create grouped bar chart
    X = 1:N;
    bar(X, [annualizedRV; annualizedBV]');
    title('Comparison of Realized Volatility and Bipower Variation');
    xlabel('Asset');
    ylabel('Annualized Volatility (%)');
    legend('Realized Volatility', 'Bipower Variation (Jump-Robust)');
    set(gca, 'XTick', X);
    grid on;
    
    % Figure 3: Jump components
    figHandles(3) = figure('Position', FIGURE_POSITION);
    
    % Convert variance components to percentages
    contComponent = results.jumpAnalysis.continuousComponent ./ results.rv * 100;
    jumpComponent = results.jumpAnalysis.jumpComponent ./ results.rv * 100;
    
    % Create stacked bar chart
    bar(X, [contComponent; jumpComponent]', 'stacked');
    title('Decomposition of Price Variation into Continuous and Jump Components');
    xlabel('Asset');
    ylabel('Percentage of Total Variance (%)');
    legend('Continuous Component', 'Jump Component');
    set(gca, 'XTick', X);
    grid on;
    
    % Add text annotations showing significance
    for i = 1:N
        if results.jumpTest.jumpDetected(2,i)  % 5% significance level
            text(i, 102, '*', 'FontSize', 14, 'HorizontalAlignment', 'center');
        end
    end
    
    % Figure 4: Comparison of kernel estimators (for asset 1)
    figHandles(4) = figure('Position', FIGURE_POSITION);
    
    % Extract data for asset 1
    kernelData = squeeze(results.kernelResults.estimates(:,:,1));
    
    % Convert to annualized volatility
    annualizedKernel = sqrt(kernelData * DAILY_SESSIONS) * 100;
    
    % Create 3D bar chart
    bar3(annualizedKernel);
    
    title('Kernel Estimators for Realized Volatility (Asset 1)');
    xlabel('Bandwidth');
    ylabel('Kernel Type');
    zlabel('Annualized Volatility (%)');
    
    % Set the x-tick labels to the actual bandwidth values
    xticks(1:length(results.kernelResults.bandwidths));
    xticklabels(arrayfun(@num2str, results.kernelResults.bandwidths, 'UniformOutput', false));
    
    % Set the y-tick labels to the kernel types
    yticks(1:length(results.kernelResults.kernelTypes));
    yticklabels(results.kernelResults.kernelTypes);
    
    % Add reference line for standard RV
    hold on;
    stdRV = sqrt(results.rv(1) * DAILY_SESSIONS) * 100;
    refX = [0.5, length(results.kernelResults.bandwidths)+0.5];
    refY = [0.5, length(results.kernelResults.kernelTypes)+0.5];
    refZ = [stdRV, stdRV; stdRV, stdRV];
    surf(refX, refY, refZ, 'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
    hold off;
    
    % Figure 5: Spectral analysis
    figHandles(5) = figure('Position', FIGURE_POSITION);
    
    if isfield(results, 'spectrumDiag') && isfield(results.spectrumDiag, 'spectrum')
        % Plot spectral density
        semilogy(results.spectrumDiag.frequencies{1}, results.spectrumDiag.spectrum{1});
        title('Realized Volatility Spectrum (Asset 1)');
        xlabel('Frequency');
        ylabel('Spectral Density (log scale)');
        grid on;
        
        % Add vertical line at cutoff frequency if available
        if isfield(results.spectrumDiag, 'window') && any(results.spectrumDiag.window{1} > 0)
            hold on;
            % Find where window becomes zero - this is the cutoff
            nonZeroIdx = find(results.spectrumDiag.window{1} > 0, 1, 'last');
            cutoffFreq = results.spectrumDiag.frequencies{1}(nonZeroIdx);
            plot([cutoffFreq, cutoffFreq], ylim, 'r--', 'LineWidth', 2);
            text(cutoffFreq*1.1, mean(ylim), 'Cutoff Frequency', 'Color', 'r');
            hold off;
        end
    else
        % Create mock spectral plot if real data not available
        freq = linspace(0, 0.5, 100);
        spectrum = 0.1 * exp(-10 * freq);
        semilogy(freq, spectrum);
        title('Realized Volatility Spectrum (Asset 1) - Example');
        xlabel('Frequency');
        ylabel('Spectral Density (log scale)');
        grid on;
    end
    
    % Figure 6: Comparative bar chart of all estimators for Asset 1
    figHandles(6) = figure('Position', FIGURE_POSITION);
    
    % Create comparison data for asset 1 (annualized volatility)
    compData = [
        sqrt(results.rv(1) * DAILY_SESSIONS) * 100,                      % Standard RV
        sqrt(results.bv(1) * DAILY_SESSIONS) * 100,                      % Bipower variation
        sqrt(results.kernelResults.estimates(1,1,1) * DAILY_SESSIONS) * 100,  % Kernel (first type, first bandwidth)
        sqrt(results.spectrumRV * DAILY_SESSIONS) * 100                 % Spectral
    ];
    
    % Create bar chart with custom colors
    barColors = [0.2 0.6 0.8; 0.8 0.4 0.2; 0.3 0.7 0.4; 0.6 0.3 0.6];
    b = bar(compData);
    colormap(barColors);
    
    title('Comparison of Volatility Estimators (Asset 1, Annualized)');
    xlabel('Estimator');
    ylabel('Volatility (%)');
    xticklabels({'Standard RV', 'Bipower', 'Kernel', 'Spectral'});
    
    % Add value labels above bars
    for i = 1:length(compData)
        text(i, compData(i)+0.5, num2str(compData(i), '%.2f'), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
    end
    
    grid on;
    
    % Add annotation about estimator properties
    dim = [0.15 0.15 0.3 0.2];
    str = {'Estimator Properties:', ...
           '• Standard RV: Fast but sensitive to jumps & noise', ...
           '• Bipower: Robust to jumps but sensitive to noise', ...
           '• Kernel: Robust to noise but complex tuning', ...
           '• Spectral: Robust to noise, frequency domain filtering'};
    annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
               'BackgroundColor', [0.9 0.9 0.9], 'EdgeColor', [0.7 0.7 0.7]);
end