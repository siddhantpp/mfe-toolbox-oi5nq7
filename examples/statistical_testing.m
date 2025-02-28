%% Statistical Testing for Financial Time Series Analysis
% This script demonstrates the use of various statistical tests available in the
% MFE Toolbox for analyzing financial time series data. The tests are essential
% for model selection, specification testing, and validation in financial econometrics.
%
% The script covers the following categories of tests:
% 1. Unit root tests (ADF and Phillips-Perron tests)
% 2. Stationarity tests (KPSS test)
% 3. Autocorrelation tests (Ljung-Box test)
% 4. ARCH effects tests
% 5. Normality tests (Jarque-Bera test)
%
% Each test is demonstrated with proper configuration, execution, and interpretation
% of results in the context of financial time series analysis.
%
% MFE Toolbox Version 4.0
% Copyright: Kevin Sheppard (2009)

%% Set global parameters
SIGNIFICANCE_LEVEL = 0.05;  % Standard 5% significance level for hypothesis testing

%% Helper function for displaying test results
function display_test_results(results, test_name)
    % DISPLAY_TEST_RESULTS Helper function to display statistical test results
    % with interpretations and formatted output
    %
    % INPUTS:
    %   results   - Results structure returned by statistical test function
    %   test_name - String containing the name of the test
    
    fprintf('\n==== %s Results ====\n', test_name);
    
    % Display test statistic with formatting based on test type
    switch test_name
        case 'Augmented Dickey-Fuller Test'
            fprintf('Test statistic: %.4f\n', results.stat);
            fprintf('p-value: %.4f\n', results.pval);
            fprintf('Critical values [1%%, 5%%, 10%%]: [%.4f, %.4f, %.4f]\n', ...
                   results.crit_vals(1), results.crit_vals(2), results.crit_vals(3));
            fprintf('Regression type: %s\n', results.regression_type);
            fprintf('Number of lags: %d\n', results.lags);
            
            % Interpret the results
            if results.pval < SIGNIFICANCE_LEVEL
                fprintf('INTERPRETATION: Reject null hypothesis of unit root at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be STATIONARY.\n');
            else
                fprintf('INTERPRETATION: Cannot reject null hypothesis of unit root at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be NON-STATIONARY.\n');
            end
            
        case 'Phillips-Perron Test'
            fprintf('Z(tau) statistic: %.4f\n', results.stat_tau);
            fprintf('Z(alpha) statistic: %.4f\n', results.stat_alpha);
            fprintf('p-value: %.4f\n', results.pval);
            fprintf('Critical values [1%%, 5%%, 10%%]: [%.4f, %.4f, %.4f]\n', ...
                   results.cv_1pct, results.cv_5pct, results.cv_10pct);
            fprintf('Regression type: %s\n', results.regression_type);
            fprintf('Number of lags: %d\n', results.lags);
            
            % Interpret the results
            if results.pval < SIGNIFICANCE_LEVEL
                fprintf('INTERPRETATION: Reject null hypothesis of unit root at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be STATIONARY.\n');
            else
                fprintf('INTERPRETATION: Cannot reject null hypothesis of unit root at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be NON-STATIONARY.\n');
            end
            
        case 'KPSS Test'
            fprintf('Test statistic: %.4f\n', results.stat);
            fprintf('p-value: %.4f\n', results.pval);
            fprintf('Critical values [1%%, 2.5%%, 5%%, 10%%]: [%.4f, %.4f, %.4f, %.4f]\n', ...
                   results.cv(1), results.cv(2), results.cv(3), results.cv(4));
            fprintf('Regression type: %s\n', results.regression_type);
            fprintf('Number of lags: %d\n', results.lags);
            
            % Interpret the results (note the opposite null hypothesis compared to ADF/PP)
            if results.pval < SIGNIFICANCE_LEVEL
                fprintf('INTERPRETATION: Reject null hypothesis of stationarity at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be NON-STATIONARY.\n');
            else
                fprintf('INTERPRETATION: Cannot reject null hypothesis of stationarity at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be STATIONARY.\n');
            end
            
        case 'Ljung-Box Test'
            fprintf('Test conducted at multiple lags:\n');
            for i = 1:length(results.lags)
                fprintf('Lag %d: Q-stat = %.4f, p-value = %.4f, DoF = %d\n', ...
                       results.lags(i), results.stats(i), results.pvals(i), results.dofs(i));
            end
            
            % Interpret the results for the first few lags
            num_to_display = min(5, length(results.lags));
            fprintf('\nINTERPRETATION:\n');
            
            if any(results.pvals(1:num_to_display) < SIGNIFICANCE_LEVEL)
                fprintf('Significant autocorrelation detected in the first %d lags at %.0f%% significance.\n', ...
                       num_to_display, SIGNIFICANCE_LEVEL*100);
                fprintf('This suggests potential serial dependence in the series.\n');
            else
                fprintf('No significant autocorrelation detected in the first %d lags at %.0f%% significance.\n', ...
                       num_to_display, SIGNIFICANCE_LEVEL*100);
                fprintf('This suggests the series exhibits no significant serial dependence.\n');
            end
            
        case 'ARCH Test'
            fprintf('LM test statistic: %.4f\n', results.statistic);
            fprintf('p-value: %.4f\n', results.pval);
            fprintf('Critical values [10%%, 5%%, 1%%]: [%.4f, %.4f, %.4f]\n', ...
                   results.critical.ten, results.critical.five, results.critical.one);
            fprintf('Number of lags: %d\n', results.lags);
            
            % Interpret the results
            if results.pval < SIGNIFICANCE_LEVEL
                fprintf('INTERPRETATION: Reject null hypothesis of no ARCH effects at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series exhibits volatility clustering (ARCH effects).\n');
                fprintf('                Consider using GARCH-family models for this series.\n');
            else
                fprintf('INTERPRETATION: Cannot reject null hypothesis of no ARCH effects at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                No significant volatility clustering detected.\n');
            end
            
        case 'Jarque-Bera Test'
            fprintf('Test statistic: %.4f\n', results.statistic);
            fprintf('p-value: %.4f\n', results.pval);
            fprintf('Critical values [10%%, 5%%, 1%%]: [%.4f, %.4f, %.4f]\n', ...
                   results.crit_val(1), results.crit_val(2), results.crit_val(3));
            
            % Interpret the results
            if results.pval < SIGNIFICANCE_LEVEL
                fprintf('INTERPRETATION: Reject null hypothesis of normality at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series does not follow a normal distribution.\n');
                fprintf('                Consider using non-Gaussian error distributions in modeling.\n');
            else
                fprintf('INTERPRETATION: Cannot reject null hypothesis of normality at %.0f%% significance.\n', SIGNIFICANCE_LEVEL*100);
                fprintf('                Series appears to be normally distributed.\n');
            end
    end
end

%% Load and prepare financial data
% Load example financial data (assuming a .mat file with financial time series)
% For demonstration purposes, we'll simulate some typical financial data if file doesn't exist
try
    % Attempt to load existing data file
    load('example_financial_data.mat');
    fprintf('Loaded financial data from example_financial_data.mat\n');
catch
    % If the file doesn't exist, simulate some typical financial data
    fprintf('Data file not found. Generating simulated financial data instead.\n');
    
    % Seed for reproducibility
    rng(42);
    
    % Generate a non-stationary price series (random walk with drift)
    T = 1000;  % Number of observations
    price_series = 100 + cumsum(0.001 + 0.01*randn(T, 1));
    
    % Generate returns (stationary but with volatility clustering)
    innovations = randn(T, 1);
    volatility = ones(T, 1);
    
    % Simulate GARCH(1,1)-like volatility process
    for t = 2:T
        volatility(t) = sqrt(0.01 + 0.10*innovations(t-1)^2 + 0.85*volatility(t-1)^2);
    end
    
    % Returns with volatility clustering
    returns = 0.0005 + volatility .* innovations;
    
    % Generate an AR(1) series for testing autocorrelation
    ar_series = zeros(T, 1);
    ar_series(1) = randn(1);
    for t = 2:T
        ar_series(t) = 0.7*ar_series(t-1) + 0.1*randn(1);
    end
    
    % Pack data into a structure
    financial_data = struct('prices', price_series, ...
                           'returns', returns, ...
                           'ar_series', ar_series, ...
                           'dates', (today-T+1:today)');
    
    % Save the simulated data for future use
    save('example_financial_data.mat', 'financial_data');
    fprintf('Generated and saved simulated financial data.\n');
end

%% Visualize the data
figure(1);

% Price series
subplot(3, 1, 1);
plot(financial_data.prices);
title('Price Series');
xlabel('Time');
ylabel('Price');
grid on;

% Returns series
subplot(3, 1, 2);
plot(financial_data.returns);
title('Returns Series');
xlabel('Time');
ylabel('Returns');
grid on;

% AR(1) series
subplot(3, 1, 3);
plot(financial_data.ar_series);
title('AR(1) Series');
xlabel('Time');
ylabel('Value');
grid on;

%% Unit Root and Stationarity Tests
% These tests examine whether a time series contains a unit root (non-stationary)
% or is stationary. Stationarity is a key assumption for many time series models.

fprintf('\n\n===== UNIT ROOT AND STATIONARITY TESTS =====\n');
fprintf('Testing whether price and return series are stationary.\n');

% 1. Augmented Dickey-Fuller (ADF) Test on price series
fprintf('\nTesting price series with ADF test:\n');
% Test with constant and trend
options_adf = struct('regression_type', 'ct', 'lags', 'aic');
adf_price = adf_test(financial_data.prices, options_adf);
display_test_results(adf_price, 'Augmented Dickey-Fuller Test');

% 2. ADF Test on returns
fprintf('\nTesting returns series with ADF test:\n');
% Test with constant only
options_adf = struct('regression_type', 'c', 'lags', 'aic');
adf_returns = adf_test(financial_data.returns, options_adf);
display_test_results(adf_returns, 'Augmented Dickey-Fuller Test');

% 3. Phillips-Perron (PP) Test on price series
fprintf('\nTesting price series with Phillips-Perron test:\n');
% Test with constant and trend
options_pp = struct('regression_type', 'ct');
pp_price = pp_test(financial_data.prices, options_pp);
display_test_results(pp_price, 'Phillips-Perron Test');

% 4. PP Test on returns
fprintf('\nTesting returns series with Phillips-Perron test:\n');
% Test with constant only
options_pp = struct('regression_type', 'c');
pp_returns = pp_test(financial_data.returns, options_pp);
display_test_results(pp_returns, 'Phillips-Perron Test');

% 5. KPSS Test on price series (null hypothesis is stationarity, opposite of ADF/PP)
fprintf('\nTesting price series with KPSS test:\n');
% Test for trend stationarity
options_kpss = struct('regression_type', 'tau');
kpss_price = kpss_test(financial_data.prices, options_kpss);
display_test_results(kpss_price, 'KPSS Test');

% 6. KPSS Test on returns
fprintf('\nTesting returns series with KPSS test:\n');
% Test for level stationarity
options_kpss = struct('regression_type', 'mu');
kpss_returns = kpss_test(financial_data.returns, options_kpss);
display_test_results(kpss_returns, 'KPSS Test');

% Summarize unit root and stationarity test results
fprintf('\n===== SUMMARY OF STATIONARITY TESTING =====\n');
fprintf('Price Series:\n');
fprintf('  ADF Test: %s\n', iif(adf_price.pval < SIGNIFICANCE_LEVEL, 'Stationary', 'Non-stationary'));
fprintf('  PP Test: %s\n', iif(pp_price.pval < SIGNIFICANCE_LEVEL, 'Stationary', 'Non-stationary'));
fprintf('  KPSS Test: %s\n', iif(kpss_price.pval < SIGNIFICANCE_LEVEL, 'Non-stationary', 'Stationary'));

fprintf('\nReturns Series:\n');
fprintf('  ADF Test: %s\n', iif(adf_returns.pval < SIGNIFICANCE_LEVEL, 'Stationary', 'Non-stationary'));
fprintf('  PP Test: %s\n', iif(pp_returns.pval < SIGNIFICANCE_LEVEL, 'Stationary', 'Non-stationary'));
fprintf('  KPSS Test: %s\n', iif(kpss_returns.pval < SIGNIFICANCE_LEVEL, 'Non-stationary', 'Stationary'));

fprintf('\nConclusion: The price series is typically non-stationary (contains unit root),\n');
fprintf('while the returns series is typically stationary, which is the expected\n');
fprintf('behavior for financial time series.\n');

%% Autocorrelation Tests
% The Ljung-Box test examines whether a time series exhibits autocorrelation
% Autocorrelation can indicate market inefficiency or model misspecification

fprintf('\n\n===== AUTOCORRELATION TESTS =====\n');
fprintf('Testing for serial correlation in the time series.\n');

% 1. Ljung-Box test on returns series for linear dependence
ljung_box_lags = [5, 10, 15, 20];  % Test at multiple lags
fprintf('\nTesting returns for autocorrelation with Ljung-Box test:\n');
lb_returns = ljungbox(financial_data.returns, ljung_box_lags);
display_test_results(lb_returns, 'Ljung-Box Test');

% 2. Ljung-Box test on AR(1) series (should show autocorrelation)
fprintf('\nTesting AR(1) series for autocorrelation with Ljung-Box test:\n');
lb_ar = ljungbox(financial_data.ar_series, ljung_box_lags);
display_test_results(lb_ar, 'Ljung-Box Test');

% 3. Ljung-Box test on squared returns for nonlinear dependence (ARCH effects)
fprintf('\nTesting squared returns for autocorrelation with Ljung-Box test:\n');
lb_returns_squared = ljungbox(financial_data.returns.^2, ljung_box_lags);
display_test_results(lb_returns_squared, 'Ljung-Box Test');

% Visualize sample autocorrelation function (ACF)
figure(2);

% ACF of returns
subplot(2, 1, 1);
[acf_returns, ~, acf_ci] = sacf(financial_data.returns, 20);
hold on;
bar(1:20, acf_returns);
plot(1:20, acf_ci(:,1), 'r--');
plot(1:20, acf_ci(:,2), 'r--');
hold off;
title('Sample Autocorrelation Function (ACF) of Returns');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;

% ACF of squared returns
subplot(2, 1, 2);
[acf_returns_squared, ~, acf_squared_ci] = sacf(financial_data.returns.^2, 20);
hold on;
bar(1:20, acf_returns_squared);
plot(1:20, acf_squared_ci(:,1), 'r--');
plot(1:20, acf_squared_ci(:,2), 'r--');
hold off;
title('Sample ACF of Squared Returns (Proxy for Volatility Persistence)');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;

%% ARCH Effects Test
% Tests for time-varying volatility (volatility clustering) in returns
% Often seen in financial time series

fprintf('\n\n===== ARCH EFFECTS TEST =====\n');
fprintf('Testing for volatility clustering (ARCH effects) in the returns.\n');

% ARCH LM test on returns
arch_lags = [1, 5, 10];  % Test with different lag specifications

for i = 1:length(arch_lags)
    fprintf('\nARCH test with %d lags:\n', arch_lags(i));
    arch_results = arch_test(financial_data.returns, arch_lags(i));
    display_test_results(arch_results, 'ARCH Test');
end

% Visualize volatility clustering
figure(3);
subplot(2, 1, 1);
plot(financial_data.returns);
title('Returns Series');
xlabel('Time');
ylabel('Returns');
grid on;

subplot(2, 1, 2);
plot(abs(financial_data.returns));
title('Absolute Returns (Proxy for Volatility)');
xlabel('Time');
ylabel('Absolute Returns');
grid on;

fprintf('\nVolatility clustering is a common feature in financial returns.\n');
fprintf('If present, it suggests the need for GARCH-type models for volatility forecasting.\n');

%% Normality Test
% Tests whether returns follow a normal distribution
% Financial returns often exhibit fat tails and skewness

fprintf('\n\n===== NORMALITY TEST =====\n');
fprintf('Testing whether returns are normally distributed.\n');

% Jarque-Bera test on returns
jb_returns = jarque_bera(financial_data.returns);
display_test_results(jb_returns, 'Jarque-Bera Test');

% Visualize distribution of returns against normal
figure(4);
[hist_y, hist_x] = hist(financial_data.returns, 50);
hist_y = hist_y / (sum(hist_y) * (hist_x(2) - hist_x(1))); % Normalize to make area = 1

% Calculate normal distribution curve
x_range = linspace(min(financial_data.returns), max(financial_data.returns), 100);
mu = mean(financial_data.returns);
sigma = std(financial_data.returns);
normal_pdf = normpdf(x_range, mu, sigma);

% Plot histogram and normal PDF
subplot(2, 1, 1);
bar(hist_x, hist_y);
hold on;
plot(x_range, normal_pdf, 'r', 'LineWidth', 2);
hold off;
title('Returns Distribution vs. Normal Distribution');
xlabel('Returns');
ylabel('Density');
legend('Returns', 'Normal PDF');
grid on;

% QQ plot against normal
subplot(2, 1, 2);
qqplot(financial_data.returns);
title('Q-Q Plot of Returns vs. Standard Normal');
grid on;

fprintf('\nFinancial returns typically exhibit heavier tails than normal distribution\n');
fprintf('(excess kurtosis) and sometimes skewness. This has important implications for\n');
fprintf('risk management and option pricing. Alternative distributions such as the Student''s t,\n');
fprintf('GED, or skewed t-distribution may provide better fits for financial returns.\n');

%% Conclusion
fprintf('\n\n===== CONCLUSION =====\n');
fprintf('This script has demonstrated various statistical tests for financial time series.\n');
fprintf('Key findings from typical financial data:\n\n');

fprintf('1. Stationarity:\n');
fprintf('   - Price series is typically non-stationary (unit root present)\n');
fprintf('   - Returns series is typically stationary\n\n');

fprintf('2. Autocorrelation:\n');
fprintf('   - Returns may or may not show significant linear autocorrelation\n');
fprintf('   - Squared returns typically show strong autocorrelation (volatility clustering)\n\n');

fprintf('3. ARCH Effects:\n');
fprintf('   - Financial returns typically exhibit volatility clustering\n');
fprintf('   - This suggests the need for GARCH-type models for volatility forecasting\n\n');

fprintf('4. Normality:\n');
fprintf('   - Returns typically have heavier tails than normal distribution\n');
fprintf('   - This suggests the need for alternative distributions in modeling\n\n');

fprintf('These statistical findings inform model selection and specification in financial econometrics.\n');

%% Helper function: iif (inline if)
function result = iif(condition, true_value, false_value)
    % Simple inline if function
    if condition
        result = true_value;
    else
        result = false_value;
    end
end