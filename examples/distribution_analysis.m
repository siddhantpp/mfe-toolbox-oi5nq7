%% Distribution Analysis Example for MFE Toolbox
% This script demonstrates the usage of statistical distribution functions in 
% the MFE Toolbox for analyzing financial returns data.
%
% Specifically, it focuses on:
% 1. Generalized Error Distribution (GED)
% 2. Hansen's skewed t-distribution 
% 3. Standardized Student's t-distribution
%
% We'll show parameter estimation, visualization, and applications to financial data.

%% Initialize the MFE Toolbox
% Add necessary directories to the MATLAB path
addToPath();

%% Load or Generate Financial Returns Data
% You can either load your own returns data or use the example data below
% Try to load a sample data file if it exists
try
    % Attempt to load financial returns data 
    % This could be a .mat file included with examples
    load('examples/data/financial_returns.mat');
    returns = financial_data;
    disp('Loaded example financial returns data.');
catch
    % If the example data doesn't exist, generate synthetic data
    disp('Example data not found. Generating synthetic data with fat tails.');
    % Generate returns with excess kurtosis to simulate financial returns
    rng(42); % Set random seed for reproducibility
    returns = 0.0005 + 0.01*trnd(5, 1000, 1);
end

%% Calculate Descriptive Statistics
fprintf('\n--- Descriptive Statistics of Returns ---\n');
n = length(returns);
mean_ret = mean(returns);
std_ret = std(returns);
skew_ret = sum((returns - mean_ret).^3/std_ret^3)/n;
kurt_ret = sum((returns - mean_ret).^4/std_ret^4)/n;

fprintf('Sample size: %d\n', n);
fprintf('Mean: %.6f\n', mean_ret);
fprintf('Standard deviation: %.6f\n', std_ret);
fprintf('Skewness: %.6f\n', skew_ret);
fprintf('Kurtosis: %.6f\n', kurt_ret);
fprintf('Excess kurtosis: %.6f\n', kurt_ret - 3);

%% Test for Normality using Jarque-Bera Test
jb_test = jarque_bera(returns);
fprintf('\n--- Jarque-Bera Test for Normality ---\n');
fprintf('Test statistic: %.4f\n', jb_test.statistic);
fprintf('p-value: %.6f\n', jb_test.pval);
if jb_test.pval < 0.05
    fprintf('Result: Reject normality at 5%% significance level.\n');
else
    fprintf('Result: Fail to reject normality at 5%% significance level.\n');
end

%% Create a Histogram of Returns
figure(1);
[n_hist, x_hist] = hist(returns, 50);
bar(x_hist, n_hist/sum(n_hist)/(x_hist(2)-x_hist(1)), 'FaceColor', [0.8 0.8 0.8]);
hold on;
title('Histogram of Financial Returns');
xlabel('Return');
ylabel('Density');

%% Estimate Generalized Error Distribution (GED) Parameters
fprintf('\n--- Generalized Error Distribution (GED) Estimation ---\n');
tic;
ged_params = gedfit(returns);
ged_time = toc;

fprintf('Estimated parameters:\n');
fprintf('Shape parameter (nu): %.4f (SE: %.4f)\n', ged_params.nu, ged_params.stderrors(1));
fprintf('Location (mu): %.6f (SE: %.4f)\n', ged_params.mu, ged_params.stderrors(2));
fprintf('Scale (sigma): %.6f (SE: %.4f)\n', ged_params.sigma, ged_params.stderrors(3));
fprintf('Log-likelihood: %.4f\n', ged_params.loglik);
fprintf('Estimation time: %.4f seconds\n', ged_time);

% Compute GED PDF values for plotting
x_grid = linspace(min(returns)-0.02, max(returns)+0.02, 1000);
ged_pdf = gedpdf((x_grid - ged_params.mu) / ged_params.sigma, ged_params.nu) / ged_params.sigma;

% Add GED fit to histogram
plot(x_grid, ged_pdf, 'b-', 'LineWidth', 2);

%% Estimate Hansen's Skewed t-Distribution Parameters
fprintf('\n--- Hansen''s Skewed t-Distribution Estimation ---\n');
tic;
skewt_params = skewtfit(returns);
skewt_time = toc;

fprintf('Estimated parameters:\n');
fprintf('Degrees of freedom (nu): %.4f (SE: %.4f)\n', skewt_params.nu, skewt_params.nuSE);
fprintf('Skewness (lambda): %.4f (SE: %.4f)\n', skewt_params.lambda, skewt_params.lambdaSE);
fprintf('Location (mu): %.6f (SE: %.4f)\n', skewt_params.mu, skewt_params.muSE);
fprintf('Scale (sigma): %.6f (SE: %.4f)\n', skewt_params.sigma, skewt_params.sigmaSE);
fprintf('Log-likelihood: %.4f\n', skewt_params.logL);
fprintf('Estimation time: %.4f seconds\n', skewt_time);

% Compute skewed t PDF values for plotting
x_standardized = (x_grid - skewt_params.mu) / skewt_params.sigma;
skewt_pdf = skewtpdf(x_standardized, skewt_params.nu, skewt_params.lambda) / skewt_params.sigma;

% Add skewed t fit to histogram
plot(x_grid, skewt_pdf, 'r-', 'LineWidth', 2);

%% Estimate Standardized Student's t-Distribution Parameters
fprintf('\n--- Standardized Student''s t-Distribution Estimation ---\n');
tic;
stdt_params = stdtfit(returns);
stdt_time = toc;

fprintf('Estimated parameters:\n');
fprintf('Degrees of freedom (nu): %.4f (SE: %.4f)\n', stdt_params.nu, stdt_params.nuSE);
fprintf('Location (mu): %.6f (SE: %.4f)\n', stdt_params.mu, stdt_params.muSE);
fprintf('Scale (sigma): %.6f (SE: %.4f)\n', stdt_params.sigma, stdt_params.sigmaSE);
fprintf('Log-likelihood: %.4f\n', stdt_params.logL);
fprintf('Estimation time: %.4f seconds\n', stdt_time);

% Compute standardized t PDF values for plotting
stdt_pdf = stdtpdf((x_grid - stdt_params.mu) / stdt_params.sigma, stdt_params.nu) / stdt_params.sigma;

% Add standardized t fit to histogram
plot(x_grid, stdt_pdf, 'g-', 'LineWidth', 2);

% Add normal distribution for comparison
normal_pdf = normpdf(x_grid, mean_ret, std_ret);
plot(x_grid, normal_pdf, 'k--', 'LineWidth', 1.5);

% Add legend and finalize histogram plot
legend('Empirical', 'GED', 'Skewed t', 'Standardized t', 'Normal', 'Location', 'NorthEast');
grid on;
hold off;

%% Compare Distribution Fits

% Create a new figure for detailed fit comparison
figure(2);

% Subplot 1: PDF Comparison
subplot(2,2,1);
plot(x_grid, ged_pdf, 'b-', 'LineWidth', 2);
hold on;
plot(x_grid, skewt_pdf, 'r-', 'LineWidth', 2);
plot(x_grid, stdt_pdf, 'g-', 'LineWidth', 2);
plot(x_grid, normal_pdf, 'k--', 'LineWidth', 1.5);
title('PDF Comparison');
xlabel('Return');
ylabel('Density');
legend('GED', 'Skewed t', 'Standardized t', 'Normal', 'Location', 'NorthEast');
grid on;
hold off;

% Subplot 2: CDF Comparison
subplot(2,2,2);
ged_cdf = zeros(size(x_grid));
skewt_cdf = zeros(size(x_grid));
stdt_cdf = zeros(size(x_grid));
normal_cdf = normcdf(x_grid, mean_ret, std_ret);

for i = 1:length(x_grid)
    ged_cdf(i) = gedcdf((x_grid(i) - ged_params.mu) / ged_params.sigma, ged_params.nu);
    skewt_cdf(i) = skewtcdf((x_grid(i) - skewt_params.mu) / skewt_params.sigma, skewt_params.nu, skewt_params.lambda);
    stdt_cdf(i) = stdtcdf((x_grid(i) - stdt_params.mu) / stdt_params.sigma, stdt_params.nu);
end

plot(x_grid, ged_cdf, 'b-', 'LineWidth', 2);
hold on;
plot(x_grid, skewt_cdf, 'r-', 'LineWidth', 2);
plot(x_grid, stdt_cdf, 'g-', 'LineWidth', 2);
plot(x_grid, normal_cdf, 'k--', 'LineWidth', 1.5);
title('CDF Comparison');
xlabel('Return');
ylabel('Cumulative Probability');
legend('GED', 'Skewed t', 'Standardized t', 'Normal', 'Location', 'SouthEast');
grid on;
hold off;

% Subplot 3: Left Tail Zoom
subplot(2,2,3);
left_tail_idx = find(x_grid < (mean_ret - 2.5*std_ret));
if ~isempty(left_tail_idx)
    plot(x_grid(left_tail_idx), ged_pdf(left_tail_idx), 'b-', 'LineWidth', 2);
    hold on;
    plot(x_grid(left_tail_idx), skewt_pdf(left_tail_idx), 'r-', 'LineWidth', 2);
    plot(x_grid(left_tail_idx), stdt_pdf(left_tail_idx), 'g-', 'LineWidth', 2);
    plot(x_grid(left_tail_idx), normal_pdf(left_tail_idx), 'k--', 'LineWidth', 1.5);
    title('Left Tail Comparison');
    xlabel('Return');
    ylabel('Density');
    legend('GED', 'Skewed t', 'Standardized t', 'Normal', 'Location', 'NorthEast');
    grid on;
    hold off;
end

% Subplot 4: Right Tail Zoom
subplot(2,2,4);
right_tail_idx = find(x_grid > (mean_ret + 2.5*std_ret));
if ~isempty(right_tail_idx)
    plot(x_grid(right_tail_idx), ged_pdf(right_tail_idx), 'b-', 'LineWidth', 2);
    hold on;
    plot(x_grid(right_tail_idx), skewt_pdf(right_tail_idx), 'r-', 'LineWidth', 2);
    plot(x_grid(right_tail_idx), stdt_pdf(right_tail_idx), 'g-', 'LineWidth', 2);
    plot(x_grid(right_tail_idx), normal_pdf(right_tail_idx), 'k--', 'LineWidth', 1.5);
    title('Right Tail Comparison');
    xlabel('Return');
    ylabel('Density');
    legend('GED', 'Skewed t', 'Standardized t', 'Normal', 'Location', 'NorthEast');
    grid on;
    hold off;
end

%% Generate Random Samples from Each Distribution

% Set sample size for simulation
n_sim = 10000;

% Generate random samples
ged_samples = ged_params.mu + ged_params.sigma * gedrnd(ged_params.nu, n_sim, 1);
skewt_samples = skewt_params.mu + skewt_params.sigma * skewtrnd(skewt_params.nu, skewt_params.lambda, n_sim, 1);
stdt_samples = stdt_params.mu + stdt_params.sigma * stdtrnd(n_sim, stdt_params.nu);
normal_samples = normrnd(mean_ret, std_ret, n_sim, 1);

% Create a new figure for comparing simulated distributions
figure(3);

% Plot histograms of simulated data
subplot(2,2,1);
hist(ged_samples, 50);
title('Simulated GED Returns');
xlabel('Return');
ylabel('Frequency');
grid on;

subplot(2,2,2);
hist(skewt_samples, 50);
title('Simulated Skewed t Returns');
xlabel('Return');
ylabel('Frequency');
grid on;

subplot(2,2,3);
hist(stdt_samples, 50);
title('Simulated Standardized t Returns');
xlabel('Return');
ylabel('Frequency');
grid on;

subplot(2,2,4);
hist(normal_samples, 50);
title('Simulated Normal Returns');
xlabel('Return');
ylabel('Frequency');
grid on;

%% Compare Simulated vs. Original Statistics
fprintf('\n--- Comparison of Original vs. Simulated Statistics ---\n');
fprintf('                    | Original  | GED      | Skewed t | Std t    | Normal\n');
fprintf('--------------------+-----------+----------+----------+----------+--------\n');
fprintf('Mean               | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f\n', ...
    mean_ret, mean(ged_samples), mean(skewt_samples), mean(stdt_samples), mean(normal_samples));
fprintf('Standard Deviation | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f\n', ...
    std_ret, std(ged_samples), std(skewt_samples), std(stdt_samples), std(normal_samples));
fprintf('Skewness           | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f\n', ...
    skew_ret, sum((ged_samples-mean(ged_samples)).^3)/n_sim/std(ged_samples)^3, ...
    sum((skewt_samples-mean(skewt_samples)).^3)/n_sim/std(skewt_samples)^3, ...
    sum((stdt_samples-mean(stdt_samples)).^3)/n_sim/std(stdt_samples)^3, ...
    sum((normal_samples-mean(normal_samples)).^3)/n_sim/std(normal_samples)^3);
fprintf('Kurtosis           | %8.6f | %8.6f | %8.6f | %8.6f | %8.6f\n', ...
    kurt_ret, sum((ged_samples-mean(ged_samples)).^4)/n_sim/std(ged_samples)^4, ...
    sum((skewt_samples-mean(skewt_samples)).^4)/n_sim/std(skewt_samples)^4, ...
    sum((stdt_samples-mean(stdt_samples)).^4)/n_sim/std(stdt_samples)^4, ...
    sum((normal_samples-mean(normal_samples)).^4)/n_sim/std(normal_samples)^4);

%% Calculate Value-at-Risk (VaR) for Each Distribution
alpha = [0.01, 0.025, 0.05];  % Common confidence levels for VaR
fprintf('\n--- Value-at-Risk (VaR) Comparison ---\n');
fprintf('Alpha  | Normal   | GED      | Skewed t | Std t\n');
fprintf('-------+----------+----------+----------+----------\n');

for i = 1:length(alpha)
    % Calculate VaR for each distribution
    normal_var = mean_ret + std_ret * norminv(alpha(i));
    
    % For GED: transform standardized quantile to actual quantile
    ged_std_quantile = gedinv(alpha(i), ged_params.nu);
    ged_var = ged_params.mu + ged_params.sigma * ged_std_quantile;
    
    % For skewed t: use skewtinv
    skewt_std_quantile = skewtinv(alpha(i), skewt_params.nu, skewt_params.lambda);
    skewt_var = skewt_params.mu + skewt_params.sigma * skewt_std_quantile;
    
    % For standardized t: use stdtinv
    stdt_std_quantile = stdtinv(alpha(i), stdt_params.nu);
    stdt_var = stdt_params.mu + stdt_params.sigma * stdt_std_quantile;
    
    fprintf('%.3f   | %8.6f | %8.6f | %8.6f | %8.6f\n', ...
        alpha(i), normal_var, ged_var, skewt_var, stdt_var);
end

%% Compare Model Fit Statistics
fprintf('\n--- Model Fit Comparison ---\n');
fprintf('Criterion     | GED      | Skewed t | Std t\n');
fprintf('--------------+----------+----------+----------\n');
fprintf('Log-likelihood| %8.2f | %8.2f | %8.2f\n', ...
    ged_params.loglik, skewt_params.logL, stdt_params.logL);

% Calculate AIC and BIC for each model
ged_params_count = 3;  % nu, mu, sigma
skewt_params_count = 4; % nu, lambda, mu, sigma
stdt_params_count = 3;  % nu, mu, sigma

ged_aic = -2 * ged_params.loglik + 2 * ged_params_count;
skewt_aic = -2 * skewt_params.logL + 2 * skewt_params_count;
stdt_aic = -2 * stdt_params.logL + 2 * stdt_params_count;

ged_bic = -2 * ged_params.loglik + ged_params_count * log(n);
skewt_bic = -2 * skewt_params.logL + skewt_params_count * log(n);
stdt_bic = -2 * stdt_params.logL + stdt_params_count * log(n);

fprintf('AIC           | %8.2f | %8.2f | %8.2f\n', ged_aic, skewt_aic, stdt_aic);
fprintf('BIC           | %8.2f | %8.2f | %8.2f\n', ged_bic, skewt_bic, stdt_bic);

%% Interpretation of Results

fprintf('\n--- Interpretation of Distribution Parameters ---\n');

% GED interpretation
fprintf('\nGeneralized Error Distribution (GED):\n');
fprintf('- Shape parameter (nu): %.4f\n', ged_params.nu);
if ged_params.nu < 2
    fprintf('  This indicates tails that are heavier than the normal distribution (leptokurtic).\n');
elseif ged_params.nu > 2
    fprintf('  This indicates tails that are thinner than the normal distribution (platykurtic).\n');
else
    fprintf('  This is very close to a normal distribution (nu = 2).\n');
end

% Skewed t interpretation
fprintf('\nHansen''s Skewed t-Distribution:\n');
fprintf('- Degrees of freedom (nu): %.4f\n', skewt_params.nu);
if skewt_params.nu < 5
    fprintf('  This indicates very heavy tails in the distribution.\n');
elseif skewt_params.nu < 10
    fprintf('  This indicates moderately heavy tails in the distribution.\n');
else
    fprintf('  As nu increases, the distribution approaches normality in the tails.\n');
end

fprintf('- Skewness parameter (lambda): %.4f\n', skewt_params.lambda);
if abs(skewt_params.lambda) < 0.1
    fprintf('  The distribution is approximately symmetric.\n');
elseif skewt_params.lambda < 0
    fprintf('  The distribution is skewed to the left (negative skew).\n');
else
    fprintf('  The distribution is skewed to the right (positive skew).\n');
end

% Standardized t interpretation
fprintf('\nStandardized Student''s t-Distribution:\n');
fprintf('- Degrees of freedom (nu): %.4f\n', stdt_params.nu);
if stdt_params.nu < 5
    fprintf('  This indicates very heavy tails in the distribution.\n');
elseif stdt_params.nu < 10
    fprintf('  This indicates moderately heavy tails in the distribution.\n');
else
    fprintf('  As nu increases, the distribution approaches normality in the tails.\n');
end

%% Model Selection Summary

fprintf('\n--- Distribution Selection Summary ---\n');

% Find best model based on AIC and BIC
[~, aic_best_idx] = min([ged_aic, skewt_aic, stdt_aic]);
[~, bic_best_idx] = min([ged_bic, skewt_bic, stdt_bic]);

model_names = {'GED', 'Skewed t', 'Standardized t'};
fprintf('Best model according to AIC: %s\n', model_names{aic_best_idx});
fprintf('Best model according to BIC: %s\n', model_names{bic_best_idx});

% Overall recommendation
fprintf('\nOverall recommendation for this dataset:\n');
if aic_best_idx == bic_best_idx
    fprintf('- The %s distribution provides the best fit according to both AIC and BIC.\n', model_names{aic_best_idx});
else
    fprintf('- AIC suggests the %s distribution provides the best fit.\n', model_names{aic_best_idx});
    fprintf('- BIC suggests the %s distribution provides the best fit.\n', model_names{bic_best_idx});
    fprintf('  Note: BIC penalizes model complexity more heavily than AIC.\n');
end

% Final comments on financial implications
fprintf('\nFinancial Implications:\n');
fprintf('- Accurate distribution modeling is crucial for risk measurement and asset pricing.\n');
fprintf('- Heavy-tailed distributions (lower degrees of freedom) result in higher Value-at-Risk.\n');
fprintf('- Skewness captures asymmetric return behavior important for options pricing.\n');
fprintf('- Model choice affects portfolio optimization and risk management decisions.\n');

%% End of Script
fprintf('\nEnd of Distribution Analysis Example\n\n');