%% Bootstrap Confidence Intervals for Time Series Parameters
% This example demonstrates how to compute confidence intervals using bootstrap
% methods in the MFE Toolbox, focusing on ARMA model parameter estimation.
% 
% The script shows how to:
% 1. Estimate parameters from an AR(1) model
% 2. Apply block bootstrap to compute confidence intervals
% 3. Apply stationary bootstrap for comparison
% 4. Visualize and interpret the results

%% Clear workspace and command window
clear;
clc;

%% Load or generate sample data
% For this example, we'll generate a simulated AR(1) process
% Generate simulated AR(1) process
T = 500;  % Sample size
phi = 0.7;  % AR coefficient (persistence parameter)
sigma = 1;  % Innovation standard deviation

% Initialize data array
data = zeros(T, 1);
% Generate random innovations
e = sigma * randn(T, 1);  

% Generate AR(1) process: y(t) = phi*y(t-1) + e(t)
for t = 2:T
    data(t) = phi * data(t-1) + e(t);
end

fprintf('Generated simulated AR(1) data with:\n');
fprintf('- Sample size: %d\n', T);
fprintf('- True AR coefficient: %.2f\n', phi);
fprintf('- Innovation std dev: %.2f\n', sigma);

%% Estimate AR(1) model parameters from the original data
model_results = estimate_ar_parameters(data);

% Extract parameters and standard errors
ar_coef = model_results.parameters(2);  % The AR coefficient (first parameter is constant)
ar_se = model_results.standardErrors(2);  % Standard error
normal_ci = [ar_coef - 1.96*ar_se, ar_coef + 1.96*ar_se];

% Display results
fprintf('\nOriginal AR(1) model estimation:\n');
fprintf('AR coefficient: %.4f (SE: %.4f)\n', ar_coef, ar_se);
fprintf('95%% Confidence interval (normal): [%.4f, %.4f]\n', normal_ci(1), normal_ci(2));

% Check model diagnostics
fprintf('\nLjung-Box test for residual autocorrelation:\n');
disp(model_results.ljungBox);

%% Compute confidence intervals using block bootstrap
% Set block bootstrap parameters
block_size = 30;  % Block size
num_bootstraps = 1000;  % Number of bootstrap replications
confidence_level = 0.95;  % 95% confidence level

% Compute block bootstrap confidence intervals
block_bs_results = compute_block_bootstrap_ci(data, block_size, num_bootstraps, confidence_level);

fprintf('\nBlock Bootstrap Results:\n');
fprintf('Block size: %d\n', block_size);
fprintf('Original AR coefficient: %.4f\n', block_bs_results.original_statistic);
fprintf('95%% CI: [%.4f, %.4f]\n', block_bs_results.lower, block_bs_results.upper);

%% Compute confidence intervals using stationary bootstrap
% Set stationary bootstrap parameters
probability = 0.1;  % Probability parameter (expected block length = 1/p = 10)

% Compute stationary bootstrap confidence intervals
stationary_bs_results = compute_stationary_bootstrap_ci(data, probability, num_bootstraps, confidence_level);

fprintf('\nStationary Bootstrap Results:\n');
fprintf('Probability parameter: %.2f (expected block length: %.1f)\n', probability, 1/probability);
fprintf('Original AR coefficient: %.4f\n', stationary_bs_results.original_statistic);
fprintf('95%% CI: [%.4f, %.4f]\n', stationary_bs_results.lower, stationary_bs_results.upper);

%% Visualization
% Plot the bootstrap distributions
figure;
histogram(block_bs_results.bootstrap_statistics, 30);
hold on;
xline(block_bs_results.original_statistic, 'r', 'Original');
xline(block_bs_results.lower, 'g--', 'Lower CI');
xline(block_bs_results.upper, 'g--', 'Upper CI');
xline(phi, 'b--', 'True Value');
title('Block Bootstrap Distribution of AR(1) Coefficient');
xlabel('AR Coefficient Value');
ylabel('Frequency');
legend('Bootstrap Distribution', 'Original Estimate', 'Lower CI', 'Upper CI', 'True Value');
hold off;

figure;
histogram(stationary_bs_results.bootstrap_statistics, 30);
hold on;
xline(stationary_bs_results.original_statistic, 'r', 'Original');
xline(stationary_bs_results.lower, 'g--', 'Lower CI');
xline(stationary_bs_results.upper, 'g--', 'Upper CI');
xline(phi, 'b--', 'True Value');
title('Stationary Bootstrap Distribution of AR(1) Coefficient');
xlabel('AR Coefficient Value');
ylabel('Frequency');
legend('Bootstrap Distribution', 'Original Estimate', 'Lower CI', 'Upper CI', 'True Value');
hold off;

% Compare confidence intervals
figure;
y = [1, 2, 3];
x = [ar_coef, ar_coef, ar_coef];
err = [
    [ar_coef - normal_ci(1), normal_ci(2) - ar_coef];
    [ar_coef - block_bs_results.lower, block_bs_results.upper - ar_coef];
    [ar_coef - stationary_bs_results.lower, stationary_bs_results.upper - ar_coef]
];

errorbar(x, y, err(:,1), err(:,2), 'horizontal', 'o');
hold on;
xline(phi, 'b--', 'True Value');
set(gca, 'YTick', 1:3);
set(gca, 'YTickLabel', {'Normal', 'Block Bootstrap', 'Stationary Bootstrap'});
title('Comparison of 95% Confidence Intervals for AR(1) Coefficient');
xlabel('AR Coefficient');
legend('Confidence Intervals', 'True Value');
grid on;
hold off;

%% Diagnostic Plots
% Plot sample autocorrelation of the original data
[acf_values, acf_se, acf_ci] = sacf(data, 20);

figure;
plot(0:20, [1; acf_values], 'b-o');
hold on;
plot(1:20, acf_ci(:,1), 'r--');
plot(1:20, acf_ci(:,2), 'r--');
title('Sample Autocorrelation Function');
xlabel('Lag');
ylabel('Autocorrelation');
legend('ACF', '95% Confidence Bounds');
grid on;

% Plot residual diagnostics
figure;
plot(model_results.residuals);
title('AR(1) Model Residuals');
xlabel('Time');
ylabel('Residual');

% Check residual autocorrelation with Ljung-Box test
ljung_results = ljungbox(model_results.residuals, 10, 1);
fprintf('\nLjung-Box test on residuals:\n');
for i = 1:length(ljung_results.lags)
    fprintf('Lag %d: Q-stat = %.4f, p-value = %.4f\n', ...
        ljung_results.lags(i), ljung_results.stats(i), ljung_results.pvals(i));
end

% Check residual autocorrelation
[residual_acf, residual_se, residual_ci] = sacf(model_results.residuals, 20);

figure;
plot(0:20, [1; residual_acf], 'b-o');
hold on;
plot(1:20, residual_ci(:,1), 'r--');
plot(1:20, residual_ci(:,2), 'r--');
title('Residual Autocorrelation Function');
xlabel('Lag');
ylabel('Autocorrelation');
legend('ACF', '95% Confidence Bounds');
grid on;

%% Summary and Interpretation
fprintf('\n----- Summary of Results -----\n');
fprintf('True AR coefficient: %.4f\n', phi);
fprintf('Original AR coefficient estimate: %.4f\n', ar_coef);
fprintf('Standard error (asymptotic): %.4f\n', ar_se);
fprintf('Normal approximation 95%% CI: [%.4f, %.4f]\n', normal_ci(1), normal_ci(2));
fprintf('Block bootstrap 95%% CI: [%.4f, %.4f]\n', block_bs_results.lower, block_bs_results.upper);
fprintf('Stationary bootstrap 95%% CI: [%.4f, %.4f]\n', stationary_bs_results.lower, stationary_bs_results.upper);

% Interpret the results
fprintf('\nInterpretation:\n');
fprintf('1. The bootstrap methods account for time series dependence when computing confidence intervals.\n');
fprintf('2. Differences between asymptotic and bootstrap CIs indicate the importance of accounting for dependency.\n');
fprintf('3. Block bootstrap uses fixed-length blocks (size=%d) while stationary bootstrap uses variable-length blocks (avg=%d).\n', block_size, round(1/probability));
fprintf('4. Bootstrap methods provide more reliable inference for time series data with temporal dependence.\n');
fprintf('5. The true parameter (%.4f) is %s by all three confidence intervals.\n', phi, ...
    (phi > normal_ci(1) && phi < normal_ci(2) && ...
     phi > block_bs_results.lower && phi < block_bs_results.upper && ...
     phi > stationary_bs_results.lower && phi < stationary_bs_results.upper) ? 'contained' : 'not contained');

%% Helper Functions

% Function to estimate AR parameters
function results = estimate_ar_parameters(data)
    % Define ARMA model options
    options = struct();
    options.p = 1;  % AR order = 1
    options.q = 0;  % MA order = 0
    
    % Estimate ARMA model
    results = armaxfilter(data, [], options);
end

% Function to extract AR coefficient for bootstrap
function coef = ar_coefficient_function(data)
    model = estimate_ar_parameters(data);
    coef = model.parameters(2);  % Extract only the AR coefficient
end

% Function to compute block bootstrap confidence intervals
function results = compute_block_bootstrap_ci(data, block_size, num_bootstraps, confidence_level)
    % Set options for bootstrap_confidence_intervals
    options = struct();
    options.bootstrap_type = 'block';
    options.block_size = block_size;
    options.replications = num_bootstraps;
    options.conf_level = confidence_level;
    options.method = 'percentile';
    
    % Compute bootstrap confidence intervals
    results = bootstrap_confidence_intervals(data, @ar_coefficient_function, options);
end

% Function to compute stationary bootstrap confidence intervals
function results = compute_stationary_bootstrap_ci(data, probability, num_bootstraps, confidence_level)
    % Set options for bootstrap_confidence_intervals
    options = struct();
    options.bootstrap_type = 'stationary';
    options.p = probability;
    options.replications = num_bootstraps;
    options.conf_level = confidence_level;
    options.method = 'percentile';
    
    % Compute bootstrap confidence intervals
    results = bootstrap_confidence_intervals(data, @ar_coefficient_function, options);
end