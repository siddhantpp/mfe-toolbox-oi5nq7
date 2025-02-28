function runHighFrequencyExample()
% HIGH-FREQUENCY VOLATILITY ANALYSIS EXAMPLE
% This example demonstrates the use of high-frequency volatility estimation techniques
% using the MFE Toolbox, including realized volatility computation, bipower variation,
% kernel-based estimators, and jump detection for financial return data.

    % Display introductory message
    disp('=======================================================');
    disp('HIGH-FREQUENCY VOLATILITY ANALYSIS EXAMPLE');
    disp('MFE Toolbox Version 4.0 (28-Oct-2009)');
    disp('=======================================================');
    disp('This example demonstrates various techniques for estimating');
    disp('volatility from high-frequency financial return data, including:');
    disp('  - Standard realized volatility (RV)');
    disp('  - Bipower variation (BV) for jump robustness');
    disp('  - Kernel-based realized volatility for noise robustness');
    disp('  - Jump detection and analysis');
    disp('=======================================================');
    
    % Load example high-frequency data
    disp('Loading high-frequency financial data...');
    load('./data/example_high_frequency_data.mat', 'high_frequency_returns', 'timestamps', 'sampling_frequency');
    
    % For simplicity, focus on the first asset in the dataset
    returns = high_frequency_returns(:, 1);
    
    % Display basic information about the data
    disp(['Dataset contains ' num2str(length(returns)) ' observations at ' ...
          num2str(sampling_frequency) '-minute frequency']);
    
    % 1. Compute standard realized volatility
    disp('Computing standard realized volatility (RV)...');
    rv = rv_compute(returns);
    
    % 2. Compute bipower variation (robust to jumps)
    disp('Computing bipower variation (BV)...');
    bv = bv_compute(returns);
    
    % 3. Compute kernel-based realized volatility (robust to microstructure noise)
    disp('Computing kernel-based realized volatility...');
    % Try different kernel types and bandwidths
    options_bp = struct('kernelType', 'Bartlett-Parzen', 'bandwidth', 10);
    options_th = struct('kernelType', 'Tukey-Hanning', 'bandwidth', 10);
    rv_kernel_bp = rv_kernel(returns, options_bp);
    rv_kernel_th = rv_kernel(returns, options_th);
    
    % 4. Perform jump test
    disp('Performing jump test...');
    jump_results = jump_test(returns);
    
    % 5. Summarize volatility measures
    disp('=======================================================');
    disp('VOLATILITY ESTIMATION RESULTS:');
    disp(['Standard Realized Volatility (RV): ' num2str(sqrt(rv)*100) '% (annualized: ' num2str(sqrt(rv*252)*100) '%)']);
    disp(['Bipower Variation (BV):           ' num2str(sqrt(bv)*100) '% (annualized: ' num2str(sqrt(bv*252)*100) '%)']);
    disp(['Kernel RV (Bartlett-Parzen):      ' num2str(sqrt(rv_kernel_bp)*100) '% (annualized: ' num2str(sqrt(rv_kernel_bp*252)*100) '%)']);
    disp(['Kernel RV (Tukey-Hanning):        ' num2str(sqrt(rv_kernel_th)*100) '% (annualized: ' num2str(sqrt(rv_kernel_th*252)*100) '%)']);
    disp(['Jump Component (RV-BV)/RV:        ' num2str((rv-bv)/rv*100) '%']);
    disp('=======================================================');
    
    % 6. Create visualizations of results
    % Figure 1: Plot the high-frequency return series
    figure(1);
    plot(timestamps, returns);
    title('High-Frequency Returns');
    xlabel('Time');
    ylabel('Returns (%)');
    grid on;
    datetick('x', 'dd-mmm-yyyy HH:MM', 'keepticks');
    
    % Figure 2: Compare volatility estimators
    figure(2);
    estimators = {'RV', 'BV', 'Kernel (BP)', 'Kernel (TH)'};
    vol_values = [sqrt(rv)*100, sqrt(bv)*100, sqrt(rv_kernel_bp)*100, sqrt(rv_kernel_th)*100];
    bar(vol_values);
    set(gca, 'XTickLabel', estimators);
    title('Volatility Estimator Comparison');
    ylabel('Volatility (%)');
    grid on;
    
    % Figure 3: Jump component analysis
    figure(3);
    subplot(2,1,1);
    plot(timestamps, jump_results.jumpComponent);
    title('Jump Component');
    xlabel('Time');
    ylabel('Relative Contribution');
    grid on;
    datetick('x', 'dd-mmm-yyyy HH:MM', 'keepticks');
    
    subplot(2,1,2);
    bar(jump_results.ratio);
    title(['RV/BV Ratio (Values > 1 indicate jumps), p-value: ' num2str(jump_results.pValue)]);
    grid on;
    
    % Figure 4: Autocorrelation of volatility measures
    figure(4);
    lags = 20;
    acf_rv = sacf(returns.^2, lags);
    [~, ~, acf_ci] = sacf(returns.^2, lags);
    
    plot(1:lags, acf_rv, 'bo-', 'LineWidth', 1.5);
    hold on;
    plot(1:lags, acf_ci(:,1), 'r--', 1:lags, acf_ci(:,2), 'r--');
    hold off;
    title('Volatility Persistence');
    xlabel('Lag');
    ylabel('Autocorrelation');
    legend('ACF of Squared Returns', '95% Confidence Bounds');
    grid on;
    
    % 7. Perform additional tests for volatility properties
    disp('=======================================================');
    disp('VOLATILITY STATIONARITY ANALYSIS:');
    
    % Test for stationarity in the volatility series
    adf_result = adf_test(returns.^2);
    disp(['ADF Test for Squared Returns - Statistic: ' num2str(adf_result.stat) ...
          ', p-value: ' num2str(adf_result.pval)]);
    if adf_result.pval < 0.05
        disp('  Volatility series is stationary (reject unit root null at 5% level)');
    else
        disp('  Cannot reject unit root in volatility series');
    end
    
    % Test for serial correlation in volatility
    lb_result = ljungbox(returns.^2, 10);
    disp(['Ljung-Box Test for Volatility - Statistic: ' num2str(lb_result.stats(end)) ...
          ', p-value: ' num2str(lb_result.pvals(end))]);
    if lb_result.pvals(end) < 0.05
        disp('  Significant autocorrelation in volatility series');
    else
        disp('  No significant autocorrelation in volatility series');
    end
    
    % 8. Analyze relationship between different volatility estimators
    disp('=======================================================');
    disp('VOLATILITY ESTIMATOR COMPARISON:');
    
    % Compare estimators
    volatility_results = struct(...
        'rv', rv, ...
        'bv', bv, ...
        'kernel_bp', rv_kernel_bp, ...
        'kernel_th', rv_kernel_th);
    
    compareVolatilityEstimators(volatility_results);
    
    % 9. Detailed jump analysis
    disp('=======================================================');
    disp('JUMP COMPONENT ANALYSIS:');
    
    analyzeJumpComponents(jump_results);
    
    % 10. Conclusion and usage notes
    disp('=======================================================');
    disp('CONCLUSION AND RECOMMENDATIONS:');
    disp('- Standard RV is sensitive to jumps and microstructure noise');
    disp('- Bipower Variation (BV) provides jump-robust volatility estimation');
    disp('- Kernel-based estimators reduce the impact of microstructure noise');
    disp('- Jump tests help identify significant price discontinuities');
    disp('=======================================================');
    disp('For practical applications:');
    disp('1. Use BV or kernel-based estimators when jumps or noise are concerns');
    disp('2. Daily aggregation of intraday measures reduces sampling error');
    disp('3. Consider optimal sampling frequency to balance bias and variance');
    disp('4. Test for jumps to properly interpret volatility measures');
    disp('=======================================================');
end

function comparison = compareVolatilityEstimators(volatility_results)
    % Helper function that compares the performance and characteristics of different 
    % volatility estimators using both numerical metrics and visualization.
    %
    % INPUTS:
    %   volatility_results - Structure containing different volatility estimates
    %
    % OUTPUTS:
    %   comparison - Structure with comparison metrics and statistics
    
    % Extract volatility measures
    rv = volatility_results.rv;
    bv = volatility_results.bv;
    kernel_bp = volatility_results.kernel_bp;
    kernel_th = volatility_results.kernel_th;
    
    % Calculate correlations between estimators
    corr_rv_bv = corr([rv, bv]);
    corr_rv_bv = corr_rv_bv(1,2);
    
    corr_rv_kbp = corr([rv, kernel_bp]);
    corr_rv_kbp = corr_rv_kbp(1,2);
    
    corr_bv_kbp = corr([bv, kernel_bp]);
    corr_bv_kbp = corr_bv_kbp(1,2);
    
    % Calculate relative bias (assuming RV as baseline)
    rel_bias_bv = (bv - rv) / rv * 100;
    rel_bias_kbp = (kernel_bp - rv) / rv * 100;
    rel_bias_kth = (kernel_th - rv) / rv * 100;
    
    % Calculate mean absolute differences
    mad_rv_bv = abs(rv - bv);
    mad_rv_kbp = abs(rv - kernel_bp);
    mad_bv_kbp = abs(bv - kernel_bp);
    
    % Calculate variance ratios
    var_ratio_bv_rv = bv/rv;
    var_ratio_kbp_rv = kernel_bp/rv;
    var_ratio_kbp_bv = kernel_bp/bv;
    
    % Display results
    disp(['Correlation between RV and BV:          ' num2str(corr_rv_bv)]);
    disp(['Correlation between RV and Kernel(BP):  ' num2str(corr_rv_kbp)]);
    disp(['Correlation between BV and Kernel(BP):  ' num2str(corr_bv_kbp)]);
    disp(' ');
    disp(['Relative Bias of BV vs RV:              ' num2str(rel_bias_bv) '%']);
    disp(['Relative Bias of Kernel(BP) vs RV:      ' num2str(rel_bias_kbp) '%']);
    disp(['Relative Bias of Kernel(TH) vs RV:      ' num2str(rel_bias_kth) '%']);
    disp(' ');
    disp(['Mean Absolute Difference RV-BV:         ' num2str(mad_rv_bv)]);
    disp(['Mean Absolute Difference RV-Kernel(BP): ' num2str(mad_rv_kbp)]);
    disp(['Mean Absolute Difference BV-Kernel(BP): ' num2str(mad_bv_kbp)]);
    disp(' ');
    disp(['Variance Ratio BV/RV:                   ' num2str(var_ratio_bv_rv)]);
    disp(['Variance Ratio Kernel(BP)/RV:           ' num2str(var_ratio_kbp_rv)]);
    disp(['Variance Ratio Kernel(BP)/BV:           ' num2str(var_ratio_kbp_bv)]);
    
    % Create output structure
    comparison = struct(...
        'correlations', struct('rv_bv', corr_rv_bv, 'rv_kbp', corr_rv_kbp, 'bv_kbp', corr_bv_kbp), ...
        'relative_bias', struct('bv', rel_bias_bv, 'kernel_bp', rel_bias_kbp, 'kernel_th', rel_bias_kth), ...
        'mean_abs_diff', struct('rv_bv', mad_rv_bv, 'rv_kbp', mad_rv_kbp, 'bv_kbp', mad_bv_kbp), ...
        'variance_ratios', struct('bv_rv', var_ratio_bv_rv, 'kbp_rv', var_ratio_kbp_rv, 'kbp_bv', var_ratio_kbp_bv));
end

function analysis = analyzeJumpComponents(jump_results)
    % Helper function that analyzes the jump components detected in the high-frequency 
    % return series, including patterns, magnitudes, and significance.
    %
    % INPUTS:
    %   jump_results - Structure containing jump test results
    %
    % OUTPUTS:
    %   analysis - Structure with jump analysis results
    
    % Extract jump information
    jumpComp = jump_results.jumpComponent;
    contComp = jump_results.contComponent;
    pValues = jump_results.pValue;
    jumpDetected = jump_results.jumpDetected;
    
    % Calculate jump statistics
    jumpMean = mean(jumpComp);
    jumpMax = max(jumpComp);
    jumpPct = sum(jumpComp) / (sum(jumpComp) + sum(contComp)) * 100;
    
    % Count significant jumps at different significance levels
    sigJumps10pct = sum(jumpDetected(1,:));
    sigJumps5pct = sum(jumpDetected(2,:));
    sigJumps1pct = sum(jumpDetected(3,:));
    
    % Find location of maximum jump
    [~, maxJumpIdx] = max(jumpComp);
    
    % Temporal pattern analysis - identify if jumps cluster in time
    % (Simplified approach for demonstration)
    jumps_binary = jumpComp > mean(jumpComp) + 2*std(jumpComp);
    jump_runs = diff([0; jumps_binary; 0]);
    jump_starts = find(jump_runs == 1);
    jump_ends = find(jump_runs == -1) - 1;
    jump_lengths = jump_ends - jump_starts + 1;
    
    % Display results
    disp(['Average Jump Component:                 ' num2str(jumpMean)]);
    disp(['Maximum Jump Component:                 ' num2str(jumpMax)]);
    disp(['Proportion of Variance Due to Jumps:    ' num2str(jumpPct) '%']);
    disp(' ');
    disp(['Significant Jumps (10% level):          ' num2str(sigJumps10pct)]);
    disp(['Significant Jumps (5% level):           ' num2str(sigJumps5pct)]);
    disp(['Significant Jumps (1% level):           ' num2str(sigJumps1pct)]);
    disp(' ');
    disp(['Number of Jump Clusters:                ' num2str(length(jump_starts))]);
    disp(['Average Jump Cluster Length:            ' num2str(mean(jump_lengths))]);
    disp(['P-value of Maximum Jump:                ' num2str(pValues(maxJumpIdx))]);
    
    % Create output structure
    analysis = struct(...
        'jumpMean', jumpMean, ...
        'jumpMax', jumpMax, ...
        'jumpPct', jumpPct, ...
        'sigJumps', struct('alpha10', sigJumps10pct, 'alpha05', sigJumps5pct, 'alpha01', sigJumps1pct), ...
        'maxJumpLocation', maxJumpIdx, ...
        'maxJumpPValue', pValues(maxJumpIdx), ...
        'jumpClusters', struct('count', length(jump_starts), 'avgLength', mean(jump_lengths)));
end