%% Bootstrap Examples for Financial Time Series Analysis
% This script demonstrates the application of bootstrap methods to financial
% time series analysis using the MFE Toolbox. The examples illustrate how to
% implement various bootstrap techniques for dependent data, including:
%
% * Block bootstrap
% * Stationary bootstrap
% * Bootstrap variance estimation
% * Bootstrap confidence intervals
%
% These methods are particularly useful for financial time series which
% typically exhibit temporal dependence structures such as autocorrelation
% and volatility clustering.
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

function run_bootstrap_examples()
    % Main function that runs all bootstrap examples in sequence
    
    % Display header
    disp('====================================================================');
    disp('   Bootstrap Methods for Financial Time Series Analysis Examples');
    disp('====================================================================');
    disp(' ');
    disp('These examples demonstrate bootstrap methods for time series data with');
    disp('temporal dependencies, specifically designed for financial applications.');
    disp(' ');
    
    % Load sample financial returns data or generate synthetic data
    try
        load('../data/financial_returns.mat', 'returns', 'dates', 'assets');
        disp('Using sample financial returns data from financial_returns.mat');
        disp(['Data contains ' num2str(size(returns, 2)) ' assets with ' num2str(size(returns, 1)) ' observations.']);
        
        % Select a single asset for demonstration
        assetIdx = 1;
        data = returns(:, assetIdx);
        disp(['Using returns for asset: ' assets{assetIdx}]);
    catch
        % Generate synthetic data if sample data is not available
        disp('Sample financial returns data not found. Generating synthetic data...');
        T = 1000;  % Length of time series
        ar_coef = 0.2;  % Autoregressive coefficient
        vol_persistence = 0.9;  % Volatility persistence
        data = generate_sample_data(T, ar_coef, vol_persistence);
        disp(['Generated synthetic time series with ' num2str(T) ' observations.']);
    end
    
    disp(' ');
    
    % Run demonstrations of different bootstrap methods
    
    % Block bootstrap demonstration
    disp('Running block bootstrap demonstration...');
    block_results = demonstrate_block_bootstrap(data);
    disp(' ');
    
    % Stationary bootstrap demonstration
    disp('Running stationary bootstrap demonstration...');
    stationary_results = demonstrate_stationary_bootstrap(data);
    disp(' ');
    
    % Bootstrap variance estimation
    disp('Running bootstrap variance estimation demonstration...');
    variance_results = demonstrate_bootstrap_variance(data);
    disp(' ');
    
    % Bootstrap confidence intervals
    disp('Running bootstrap confidence intervals demonstration...');
    ci_results = demonstrate_bootstrap_confidence_intervals(data);
    disp(' ');
    
    % Summary
    disp('====================================================================');
    disp('                        Summary of Results');
    disp('====================================================================');
    disp(' ');
    disp('Block Bootstrap Results:');
    disp(['  Mean of original data: ' num2str(mean(data))]);
    disp(['  Mean of bootstrap means: ' num2str(mean(block_results.bootstrap_means))]);
    disp(['  Bootstrap standard error: ' num2str(std(block_results.bootstrap_means))]);
    disp(' ');
    
    disp('Stationary Bootstrap Results:');
    disp(['  Mean of bootstrap means: ' num2str(mean(stationary_results.bootstrap_means))]);
    disp(['  Bootstrap standard error: ' num2str(std(stationary_results.bootstrap_means))]);
    disp(' ');
    
    disp('Bootstrap Variance Estimation:');
    disp(['  Variance of returns: ' num2str(var(data))]);
    disp(['  Bootstrap variance estimate: ' num2str(variance_results.block_bootstrap.variance)]);
    disp(['  Bootstrap standard error: ' num2str(variance_results.block_bootstrap.std_error)]);
    disp(['  95% Confidence interval: [' num2str(variance_results.block_bootstrap.conf_lower) ', ' ...
          num2str(variance_results.block_bootstrap.conf_upper) ']']);
    disp(' ');
    
    disp('Bootstrap Confidence Intervals for Mean:');
    disp(['  Original mean: ' num2str(ci_results.original_statistic)]);
    disp(['  95% Percentile CI: [' num2str(ci_results.percentile.lower) ', ' ...
          num2str(ci_results.percentile.upper) ']']);
    disp(['  95% Bias-Corrected CI: [' num2str(ci_results.bc.lower) ', ' ...
          num2str(ci_results.bc.upper) ']']);
    disp(' ');
    
    disp('======================================================================');
    disp('                       End of Bootstrap Examples');
    disp('======================================================================');
    disp(' ');
    disp('For more information on bootstrap methods, see the MFE Toolbox documentation.');
    disp('For customizing these examples, you can modify:');
    disp('- Block size for block bootstrap');
    disp('- Probability parameter for stationary bootstrap');
    disp('- Number of bootstrap replications');
    disp('- Statistics of interest (mean, variance, quantiles, etc.)');
    disp('- Confidence levels');
end

function data = generate_sample_data(T, ar_coef, vol_persistence)
    % Generates synthetic financial time series data with autocorrelation and 
    % volatility clustering for bootstrap demonstrations when real data is not available
    
    % Initialize arrays for returns and volatility
    data = zeros(T, 1);
    volatility = zeros(T, 1);
    
    % Generate initial values
    volatility(1) = 0.01; % Initial volatility (1%)
    data(1) = volatility(1) * randn(1);
    
    % Generate time series with volatility clustering and autocorrelation
    for t = 2:T
        % Update volatility - simple GARCH(1,1)-like process
        volatility(t) = sqrt(0.05^2 * (1 - vol_persistence) + ...
                       vol_persistence * volatility(t-1)^2 + ...
                       0.1 * data(t-1)^2);
        
        % Generate return with AR component and time-varying volatility
        data(t) = ar_coef * data(t-1) + volatility(t) * randn(1);
    end
end

function fig = plot_bootstrap_distribution(bootstrap_statistics, original_statistic, ci_result, title_str)
    % Creates histogram and summary plots of bootstrap statistics distribution
    
    % Create new figure
    fig = figure;
    
    % Create histogram of bootstrap statistics
    histogram(bootstrap_statistics, 'Normalization', 'probability', 'EdgeColor', 'none');
    hold on;
    
    % Add vertical line for original statistic
    line([original_statistic original_statistic], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '-');
    
    % Add vertical lines for confidence interval bounds
    line([ci_result.lower ci_result.lower], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    line([ci_result.upper ci_result.upper], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    
    % Add labels and title
    title(title_str);
    xlabel('Statistic Value');
    ylabel('Probability');
    
    % Add legend
    legend('Bootstrap Distribution', 'Original Statistic', 'Confidence Bounds', 'Location', 'Best');
    
    % Add summary statistics as text
    text_x = min(xlim) + 0.7 * (max(xlim) - min(xlim));
    text_y = max(ylim) * 0.9;
    text_str = {['Mean: ' num2str(mean(bootstrap_statistics))], ...
                ['Std Dev: ' num2str(std(bootstrap_statistics))], ...
                ['95% CI: [' num2str(ci_result.lower) ', ' num2str(ci_result.upper) ']']};
    text(text_x, text_y, text_str, 'FontSize', 10);
    
    hold off;
end

function results = demonstrate_block_bootstrap(data)
    % Demonstrates the application of block bootstrap to financial time series data
    
    % Display information about block bootstrap
    disp('----------------------------------------------------------------');
    disp('                      BLOCK BOOTSTRAP');
    disp('----------------------------------------------------------------');
    disp('Block bootstrap resamples blocks of consecutive observations to');
    disp('preserve the dependence structure within each block. This method');
    disp('is appropriate for financial time series with autocorrelation,');
    disp('heteroskedasticity, and other forms of temporal dependence.');
    disp(' ');
    
    % Set parameters for block bootstrap
    T = length(data);
    % Block size chosen based on T^(1/3) heuristic
    block_length = ceil(T^(1/3));
    % Number of bootstrap replications
    B = 1000;
    
    disp(['Time series length: ' num2str(T)]);
    disp(['Using block length: ' num2str(block_length)]);
    disp(['Number of bootstrap replications: ' num2str(B)]);
    disp(' ');
    
    % Apply block bootstrap
    disp('Applying block bootstrap...');
    bs_data = block_bootstrap(data, block_length, B);
    
    % Calculate mean for each bootstrap sample
    bs_means = zeros(B, 1);
    for i = 1:B
        bs_means(i) = mean(bs_data(:, :, i));
    end
    
    % Visualize results
    figure;
    
    % Plot original data and a sample of bootstrap series
    subplot(2, 1, 1);
    plot(1:T, data, 'b-', 'LineWidth', 1.5);
    hold on;
    % Plot 5 bootstrap samples
    for i = 1:5
        plot(1:T, bs_data(:, :, i), '--', 'LineWidth', 0.5);
    end
    hold off;
    title('Original Data vs. Bootstrap Samples');
    xlabel('Time');
    ylabel('Value');
    legend('Original Data', 'Bootstrap Samples', 'Location', 'Best');
    
    % Plot histogram of bootstrap means
    subplot(2, 1, 2);
    histogram(bs_means, 30, 'Normalization', 'probability', 'EdgeColor', 'none');
    hold on;
    % Add vertical line for the original mean
    original_mean = mean(data);
    line([original_mean original_mean], ylim, 'Color', 'r', 'LineWidth', 2);
    hold off;
    title('Distribution of Bootstrap Means');
    xlabel('Mean Value');
    ylabel('Probability');
    
    % Calculate and display bootstrap statistics
    boot_mean = mean(bs_means);
    boot_std = std(bs_means);
    boot_ci = prctile(bs_means, [2.5, 97.5]);
    
    disp('Bootstrap Statistics:');
    disp(['  Original Mean: ' num2str(original_mean)]);
    disp(['  Bootstrap Mean of Means: ' num2str(boot_mean)]);
    disp(['  Bootstrap Standard Error: ' num2str(boot_std)]);
    disp(['  95% Confidence Interval: [' num2str(boot_ci(1)) ', ' num2str(boot_ci(2)) ']']);
    
    % Return results
    results = struct('bootstrap_data', bs_data, ...
                    'bootstrap_means', bs_means, ...
                    'original_mean', original_mean, ...
                    'bootstrap_mean', boot_mean, ...
                    'bootstrap_std_error', boot_std, ...
                    'confidence_interval', boot_ci);
end

function results = demonstrate_stationary_bootstrap(data)
    % Demonstrates the application of stationary bootstrap to financial time series data
    
    % Display information about stationary bootstrap
    disp('----------------------------------------------------------------');
    disp('                    STATIONARY BOOTSTRAP');
    disp('----------------------------------------------------------------');
    disp('Stationary bootstrap uses random block lengths drawn from a');
    disp('geometric distribution with parameter p. This makes the resampled');
    disp('series stationary, unlike fixed block bootstrap. The expected');
    disp('block length is 1/p. This method is particularly suitable for');
    disp('financial time series with heteroskedasticity and temporal');
    disp('dependence.');
    disp(' ');
    
    % Set parameters for stationary bootstrap
    T = length(data);
    % Expected block length similar to block bootstrap (T^(1/3))
    expected_block_length = ceil(T^(1/3));
    p = 1 / expected_block_length;
    % Number of bootstrap replications
    B = 1000;
    
    disp(['Time series length: ' num2str(T)]);
    disp(['Using probability parameter p: ' num2str(p)]);
    disp(['Expected block length (1/p): ' num2str(1/p)]);
    disp(['Number of bootstrap replications: ' num2str(B)]);
    disp(' ');
    
    % Apply stationary bootstrap
    disp('Applying stationary bootstrap...');
    bs_data = stationary_bootstrap(data, p, B);
    
    % Calculate mean for each bootstrap sample
    bs_means = zeros(B, 1);
    for i = 1:B
        bs_means(i) = mean(bs_data(:, :, i));
    end
    
    % Visualize results
    figure;
    
    % Plot original data and a sample of bootstrap series
    subplot(2, 1, 1);
    plot(1:T, data, 'b-', 'LineWidth', 1.5);
    hold on;
    % Plot 5 bootstrap samples
    for i = 1:5
        plot(1:T, bs_data(:, :, i), '--', 'LineWidth', 0.5);
    end
    hold off;
    title('Original Data vs. Stationary Bootstrap Samples');
    xlabel('Time');
    ylabel('Value');
    legend('Original Data', 'Bootstrap Samples', 'Location', 'Best');
    
    % Plot histogram of bootstrap means
    subplot(2, 1, 2);
    histogram(bs_means, 30, 'Normalization', 'probability', 'EdgeColor', 'none');
    hold on;
    % Add vertical line for the original mean
    original_mean = mean(data);
    line([original_mean original_mean], ylim, 'Color', 'r', 'LineWidth', 2);
    hold off;
    title('Distribution of Stationary Bootstrap Means');
    xlabel('Mean Value');
    ylabel('Probability');
    
    % Calculate and display bootstrap statistics
    boot_mean = mean(bs_means);
    boot_std = std(bs_means);
    boot_ci = prctile(bs_means, [2.5, 97.5]);
    
    disp('Stationary Bootstrap Statistics:');
    disp(['  Original Mean: ' num2str(original_mean)]);
    disp(['  Bootstrap Mean of Means: ' num2str(boot_mean)]);
    disp(['  Bootstrap Standard Error: ' num2str(boot_std)]);
    disp(['  95% Confidence Interval: [' num2str(boot_ci(1)) ', ' num2str(boot_ci(2)) ']']);
    
    % Return results
    results = struct('bootstrap_data', bs_data, ...
                    'bootstrap_means', bs_means, ...
                    'original_mean', original_mean, ...
                    'bootstrap_mean', boot_mean, ...
                    'bootstrap_std_error', boot_std, ...
                    'confidence_interval', boot_ci);
end

function results = demonstrate_bootstrap_variance(data)
    % Demonstrates variance estimation using bootstrap methods for financial time series
    
    % Display information about bootstrap variance estimation
    disp('----------------------------------------------------------------');
    disp('               BOOTSTRAP VARIANCE ESTIMATION');
    disp('----------------------------------------------------------------');
    disp('Bootstrap variance estimation uses resampling to provide robust');
    disp('variance estimates for statistics of interest. This is particularly');
    disp('useful for financial time series where classical variance estimators');
    disp('may be biased due to temporal dependencies like autocorrelation');
    disp('and volatility clustering.');
    disp(' ');
    
    % Define the statistic of interest - here we use variance of returns
    statistic_fn = @var;
    
    % Define options for bootstrap variance estimation
    options = struct();
    options.bootstrap_type = 'block';  % Use block bootstrap
    options.block_size = ceil(length(data)^(1/3));
    options.replications = 1000;
    options.conf_level = 0.95;
    
    disp(['Statistic: Variance of returns']);
    disp(['Using bootstrap type: ' options.bootstrap_type]);
    disp(['Block size: ' num2str(options.block_size)]);
    disp(['Number of bootstrap replications: ' num2str(options.replications)]);
    disp(['Confidence level: ' num2str(options.conf_level)]);
    disp(' ');
    
    % Apply bootstrap variance estimation
    disp('Applying bootstrap variance estimation...');
    var_results = bootstrap_variance(data, statistic_fn, options);
    
    % Create a similar structure with different bootstrap method for comparison
    options.bootstrap_type = 'stationary';
    options.p = 1 / ceil(length(data)^(1/3));
    disp(['Comparing with stationary bootstrap (p = ' num2str(options.p) ')...']);
    var_results_stationary = bootstrap_variance(data, statistic_fn, options);
    
    % Visualize the distribution of bootstrap variance estimates
    figure;
    
    % Plot histogram of block bootstrap variance estimates
    subplot(2, 1, 1);
    histogram(var_results.bootstrap_stats, 30, 'Normalization', 'probability', 'EdgeColor', 'none');
    hold on;
    % Add vertical line for the original variance
    original_var = var(data);
    line([original_var original_var], ylim, 'Color', 'r', 'LineWidth', 2);
    % Add confidence interval bounds
    line([var_results.conf_lower var_results.conf_lower], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    line([var_results.conf_upper var_results.conf_upper], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    hold off;
    title('Distribution of Block Bootstrap Variance Estimates');
    xlabel('Variance Value');
    ylabel('Probability');
    legend('Bootstrap Distribution', 'Original Variance', 'Confidence Bounds', 'Location', 'Best');
    
    % Plot histogram of stationary bootstrap variance estimates
    subplot(2, 1, 2);
    histogram(var_results_stationary.bootstrap_stats, 30, 'Normalization', 'probability', 'EdgeColor', 'none');
    hold on;
    % Add vertical line for the original variance
    line([original_var original_var], ylim, 'Color', 'r', 'LineWidth', 2);
    % Add confidence interval bounds
    line([var_results_stationary.conf_lower var_results_stationary.conf_lower], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    line([var_results_stationary.conf_upper var_results_stationary.conf_upper], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '--');
    hold off;
    title('Distribution of Stationary Bootstrap Variance Estimates');
    xlabel('Variance Value');
    ylabel('Probability');
    legend('Bootstrap Distribution', 'Original Variance', 'Confidence Bounds', 'Location', 'Best');
    
    % Display results
    disp('Bootstrap Variance Estimation Results:');
    disp(['  Original Variance: ' num2str(original_var)]);
    disp(' ');
    disp('Block Bootstrap Results:');
    disp(['  Bootstrap Variance Estimate (Mean): ' num2str(var_results.mean)]);
    disp(['  Bootstrap Standard Error: ' num2str(var_results.std_error)]);
    disp(['  95% Confidence Interval: [' num2str(var_results.conf_lower) ', ' num2str(var_results.conf_upper) ']']);
    disp(' ');
    disp('Stationary Bootstrap Results:');
    disp(['  Bootstrap Variance Estimate (Mean): ' num2str(var_results_stationary.mean)]);
    disp(['  Bootstrap Standard Error: ' num2str(var_results_stationary.std_error)]);
    disp(['  95% Confidence Interval: [' num2str(var_results_stationary.conf_lower) ', ' num2str(var_results_stationary.conf_upper) ']']);
    
    % Return results (combine both bootstrap methods)
    results = struct('block_bootstrap', var_results, ...
                    'stationary_bootstrap', var_results_stationary, ...
                    'original_variance', original_var);
end

function results = demonstrate_bootstrap_confidence_intervals(data)
    % Demonstrates computing confidence intervals using various bootstrap methods
    
    % Display information about bootstrap confidence intervals
    disp('----------------------------------------------------------------');
    disp('               BOOTSTRAP CONFIDENCE INTERVALS');
    disp('----------------------------------------------------------------');
    disp('Bootstrap confidence intervals provide robust interval estimates');
    disp('for statistics of interest in financial time series. Multiple');
    disp('methods for computing intervals are available, including:');
    disp('  - Percentile: Direct percentiles of bootstrap distribution');
    disp('  - Basic: Symmetric around original estimate');
    disp('  - Bias-Corrected (BC): Adjusts for median bias');
    disp('  - BCa: Adjusts for both bias and skewness');
    disp(' ');
    
    % Define the statistic of interest - here we use the mean
    statistic_fn = @mean;
    
    % Base options for bootstrap confidence intervals
    options = struct();
    options.bootstrap_type = 'block';
    options.block_size = ceil(length(data)^(1/3));
    options.replications = 1000;
    options.conf_level = 0.95;
    
    disp(['Statistic: Mean of returns']);
    disp(['Using bootstrap type: ' options.bootstrap_type]);
    disp(['Block size: ' num2str(options.block_size)]);
    disp(['Number of bootstrap replications: ' num2str(options.replications)]);
    disp(['Confidence level: ' num2str(options.conf_level)]);
    disp(' ');
    
    % Apply bootstrap confidence intervals with different methods
    disp('Computing bootstrap confidence intervals with different methods...');
    
    % Percentile method
    options.method = 'percentile';
    percentile_results = bootstrap_confidence_intervals(data, statistic_fn, options);
    
    % Basic method
    options.method = 'basic';
    basic_results = bootstrap_confidence_intervals(data, statistic_fn, options);
    
    % Bias-corrected method
    options.method = 'bc';
    bc_results = bootstrap_confidence_intervals(data, statistic_fn, options);
    
    % BCa method (if available, might be more complex)
    try
        options.method = 'bca';
        bca_results = bootstrap_confidence_intervals(data, statistic_fn, options);
        bca_available = true;
    catch
        bca_available = false;
        disp('BCa method not available or requires additional components.');
    end
    
    % Visualize confidence intervals with bootstrap distribution
    figure;
    
    % Use the bootstrap statistics from percentile method for distribution
    bootstrap_stats = percentile_results.bootstrap_statistics;
    original_stat = percentile_results.original_statistic;
    
    % Plot histogram of bootstrap statistics
    histogram(bootstrap_stats, 30, 'Normalization', 'probability', 'EdgeColor', 'none', 'FaceAlpha', 0.6);
    hold on;
    
    % Add vertical line for original statistic
    line([original_stat original_stat], ylim, 'Color', 'k', 'LineWidth', 2, 'DisplayName', 'Original Mean');
    
    % Add confidence interval bounds for each method
    % Percentile method
    line([percentile_results.lower percentile_results.lower], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'Percentile CI');
    line([percentile_results.upper percentile_results.upper], ylim, 'Color', 'r', 'LineWidth', 2, 'LineStyle', '--', 'HandleVisibility', 'off');
    
    % Basic method
    line([basic_results.lower basic_results.lower], ylim, 'Color', 'g', 'LineWidth', 2, 'LineStyle', ':', 'DisplayName', 'Basic CI');
    line([basic_results.upper basic_results.upper], ylim, 'Color', 'g', 'LineWidth', 2, 'LineStyle', ':', 'HandleVisibility', 'off');
    
    % BC method
    line([bc_results.lower bc_results.lower], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '-.', 'DisplayName', 'BC CI');
    line([bc_results.upper bc_results.upper], ylim, 'Color', 'b', 'LineWidth', 2, 'LineStyle', '-.', 'HandleVisibility', 'off');
    
    % BCa method (if available)
    if bca_available
        line([bca_results.lower bca_results.lower], ylim, 'Color', 'm', 'LineWidth', 2, 'LineStyle', '--', 'DisplayName', 'BCa CI');
        line([bca_results.upper bca_results.upper], ylim, 'Color', 'm', 'LineWidth', 2, 'LineStyle', '--', 'HandleVisibility', 'off');
    end
    
    hold off;
    title('Bootstrap Distribution with Confidence Intervals');
    xlabel('Mean Value');
    ylabel('Probability');
    legend('show', 'Location', 'Best');
    
    % Display results
    disp('Bootstrap Confidence Interval Results:');
    disp(['  Original Mean: ' num2str(original_stat)]);
    disp(' ');
    disp('Confidence Intervals (95%):');
    disp(['  Percentile Method: [' num2str(percentile_results.lower) ', ' num2str(percentile_results.upper) ']']);
    disp(['  Basic Method: [' num2str(basic_results.lower) ', ' num2str(basic_results.upper) ']']);
    disp(['  Bias-Corrected Method: [' num2str(bc_results.lower) ', ' num2str(bc_results.upper) ']']);
    if bca_available
        disp(['  BCa Method: [' num2str(bca_results.lower) ', ' num2str(bca_results.upper) ']']);
    end
    
    % Return results
    results = struct('original_statistic', original_stat, ...
                    'bootstrap_statistics', bootstrap_stats, ...
                    'percentile', percentile_results, ...
                    'basic', basic_results, ...
                    'bc', bc_results);
    
    if bca_available
        results.bca = bca_results;
    end
end

% Auto-execute the main function
run_bootstrap_examples();