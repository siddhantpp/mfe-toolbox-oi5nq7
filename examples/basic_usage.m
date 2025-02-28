%% MFE Toolbox: Basic Usage Example
% This script demonstrates the core functionalities of the MFE Toolbox for
% financial econometrics and time series analysis.

%% Initialize the MFE Toolbox
% Add all necessary directories to the MATLAB path
addToPath();

%% Load or Generate Example Financial Data
% For this example, we'll use daily financial returns
try
    % Try to load example data if available
    load example_returns.mat
    disp('Loaded example returns data.');
catch
    % If example data isn't available, generate simulated returns
    disp('Example data not found. Generating simulated returns.');
    T = 1000;
    rng(123); % For reproducibility
    
    % Generate returns with GARCH-like volatility clustering
    epsilon = randn(T, 1);
    h = ones(T, 1);
    returns = zeros(T, 1);
    
    % Simple GARCH(1,1) process
    for t = 2:T
        h(t) = 0.01 + 0.1*returns(t-1)^2 + 0.85*h(t-1);
        returns(t) = sqrt(h(t)) * epsilon(t);
    end
end

%% Display Basic Statistics
disp('Basic Statistics:');
fprintf('Mean: %.5f\n', mean(returns));
fprintf('Standard Deviation: %.5f\n', std(returns));
fprintf('Skewness: %.5f\n', skewness(returns));
fprintf('Kurtosis: %.5f\n', kurtosis(returns));

%% Visualize the Return Series
figure;
subplot(2,1,1);
plot(returns);
title('Return Series');
xlabel('Time');
ylabel('Returns');

subplot(2,1,2);
plot(abs(returns));
title('Absolute Returns (Volatility Proxy)');
xlabel('Time');
ylabel('|Returns|');

%% Compute and Plot Sample ACF and PACF
% Calculate autocorrelation and partial autocorrelation
[acf, acf_se, acf_ci] = sacf(returns, 20);
[pacf, pacf_se, pacf_ci] = spacf(returns, 20);

% Plot ACF and PACF
figure;
subplot(2,1,1);
bar(1:20, acf);
hold on;
plot([0 21], [0 0], 'r--'); % Zero line
plot([0 21], [acf_ci(1,1) acf_ci(1,1)], 'r:'); % Lower CI
plot([0 21], [acf_ci(1,2) acf_ci(1,2)], 'r:'); % Upper CI
title('Sample Autocorrelation Function (ACF)');
xlabel('Lag');
ylabel('Autocorrelation');
xlim([0 21]);
hold off;

subplot(2,1,2);
bar(1:20, pacf);
hold on;
plot([0 21], [0 0], 'r--'); % Zero line
plot([0 21], [pacf_ci(1,1) pacf_ci(1,1)], 'r:'); % Lower CI
plot([0 21], [pacf_ci(1,2) pacf_ci(1,2)], 'r:'); % Upper CI
title('Sample Partial Autocorrelation Function (PACF)');
xlabel('Lag');
ylabel('Partial Autocorrelation');
xlim([0 21]);
hold off;

%% Fit ARMA(1,1) Model to Returns
% Set up ARMA model options
arma_options = struct();
arma_options.p = 1; % AR order
arma_options.q = 1; % MA order
arma_options.distribution = 'T'; % Use t-distribution for heavy tails

% Estimate ARMA model parameters
armaResults = armaxfilter(returns, [], arma_options);

% Display estimation results
disp('ARMA(1,1) Model Estimation Results:');
disp('Parameter    Estimate    Std. Error    t-Stat    p-Value');
for i = 1:length(armaResults.parameters)
    fprintf('%-12s %10.5f %12.5f %10.4f %10.4f\n', ...
        armaResults.paramNames{i}, ...
        armaResults.parameters(i), ...
        armaResults.standardErrors(i), ...
        armaResults.tStats(i), ...
        armaResults.pValues(i));
end

% Display model fit information
fprintf('\nLog-Likelihood: %.5f\n', armaResults.LL);
fprintf('AIC: %.5f\n', armaResults.aic);
fprintf('SBIC: %.5f\n', armaResults.sbic);

%% Diagnostic Tests for ARMA Model
% Perform Ljung-Box test for autocorrelation in residuals
lb_test = ljungbox(armaResults.residuals, 10, armaResults.p + armaResults.q);

% Display Ljung-Box test results
disp('Ljung-Box Test Results:');
disp('Lag    Q-Stat    p-Value    Reject at 5%?');
for i = 1:length(lb_test.lags)
    fprintf('%-6d %10.4f %10.4f %12s\n', ...
        lb_test.lags(i), ...
        lb_test.stats(i), ...
        lb_test.pvals(i), ...
        lb_test.isRejected5pct(i) ? 'Yes' : 'No');
end

%% Fit AGARCH(1,1) Volatility Model to Return Residuals
% Set up AGARCH model options
garch_options = struct();
garch_options.p = 1; % GARCH order
garch_options.q = 1; % ARCH order
garch_options.distribution = 'T'; % Use t-distribution for heavy tails

% Estimate AGARCH model parameters
garchResults = agarchfit(armaResults.residuals, garch_options);

% Display AGARCH estimation results
disp('AGARCH(1,1) Model Estimation Results:');
disp('Parameter    Estimate    Std. Error    t-Stat    p-Value');
for i = 1:length(garchResults.parameters)
    fprintf('%-12s %10.5f %12.5f %10.4f %10.4f\n', ...
        garchResults.parameternames{i}, ...
        garchResults.parameters(i), ...
        garchResults.stderrors(i), ...
        garchResults.tstat(i), ...
        garchResults.pvalues(i));
end

% Display model diagnostics
disp('AGARCH Model Diagnostics:');
fprintf('Persistence: %.5f\n', garchResults.diagnostics.persistence);
fprintf('Unconditional Variance: %.5f\n', garchResults.diagnostics.uncvar);
fprintf('Half-life: %.5f\n', garchResults.diagnostics.halflife);

%% Generate Forecasts
% Generate ARMA forecasts for 20 periods ahead
[forecastData, forecastVar] = armafor(armaResults.parameters, returns, armaResults.p, armaResults.q, armaResults.constant, [], 20);

% Set up GARCH forecast options
forecast_options = struct();
forecast_options.simulate = true; % Use simulation-based forecasting
forecast_options.numPaths = 5000; % Number of simulation paths
forecast_options.probs = [0.025, 0.5, 0.975]; % Probability levels for quantiles

% Generate GARCH forecasts for 20 periods ahead
forecastVar = garchfor(garchResults, 20, forecast_options);

% Plot forecasts
figure;
subplot(2,1,1);
hold on;
plot(1:20, forecastData, 'b-', 'LineWidth', 2);
title('ARMA(1,1) Return Forecasts');
xlabel('Forecast Horizon');
ylabel('Forecasted Returns');
hold off;

subplot(2,1,2);
hold on;
plot(1:20, forecastVar.expectedVolatility, 'r-', 'LineWidth', 2);
plot(1:20, forecastVar.volatilityQuantiles(:,1), 'r--');
plot(1:20, forecastVar.volatilityQuantiles(:,3), 'r--');
title('AGARCH(1,1) Volatility Forecasts');
xlabel('Forecast Horizon');
ylabel('Forecasted Volatility');
legend('Point Forecast', '2.5% Quantile', '97.5% Quantile', 'Location', 'Best');
hold off;

%% Distribution Analysis using GED
% Define a range of values for demonstration
x = linspace(-5, 5, 1000)';

% Compute GED PDF for different shape parameters
ged_pdf1 = gedpdf(x, 1.0); % Heavier tails than normal
ged_pdf2 = gedpdf(x, 2.0); % Equivalent to normal distribution
ged_pdf3 = gedpdf(x, 5.0); % Thinner tails than normal

% Plot GED PDFs for comparison
figure;
plot(x, ged_pdf1, 'r-', 'LineWidth', 2);
hold on;
plot(x, ged_pdf2, 'b-', 'LineWidth', 2);
plot(x, ged_pdf3, 'g-', 'LineWidth', 2);
hold off;
title('Generalized Error Distribution (GED) PDFs');
xlabel('x');
ylabel('Probability Density');
legend('GED(\nu=1.0) - Laplace', 'GED(\nu=2.0) - Normal', 'GED(\nu=5.0) - Thin Tails');
grid on;

%% Bootstrap Analysis
% Generate bootstrap samples of the return series
block_size = 10; % Block size for block bootstrap
num_bootstrap = 1000; % Number of bootstrap samples
bootSamples = block_bootstrap(returns, block_size, num_bootstrap);

% Calculate bootstrap statistics (e.g., mean of each bootstrap sample)
bootstrap_means = squeeze(mean(bootSamples));

% Calculate bootstrap confidence intervals
ci_levels = [0.025, 0.975]; % 95% confidence interval
bootstrap_ci = quantile(bootstrap_means, ci_levels);

% Display bootstrap results
disp('Bootstrap Analysis Results:');
fprintf('Original Mean: %.5f\n', mean(returns));
fprintf('Bootstrap Mean: %.5f\n', mean(bootstrap_means));
fprintf('95%% Bootstrap CI: [%.5f, %.5f]\n', bootstrap_ci(1), bootstrap_ci(2));

% Plot bootstrap distribution
figure;
histogram(bootstrap_means, 50);
hold on;
xline(mean(returns), 'r-', 'Original Mean', 'LineWidth', 2);
xline(bootstrap_ci(1), 'b--', '2.5% Quantile', 'LineWidth', 1.5);
xline(bootstrap_ci(2), 'b--', '97.5% Quantile', 'LineWidth', 1.5);
hold off;
title('Bootstrap Distribution of Mean Returns');
xlabel('Mean Return');
ylabel('Frequency');
grid on;

%% Conclusion
% This example script has demonstrated key functionalities of the MFE Toolbox:
% 1. Time series visualization and analysis
% 2. ARMA model estimation and diagnostics
% 3. GARCH volatility modeling
% 4. Forecasting both returns and volatility
% 5. Distribution analysis with GED
% 6. Bootstrap methods for statistical inference