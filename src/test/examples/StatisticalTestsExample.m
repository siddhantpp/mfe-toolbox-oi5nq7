%% Statistical Tests Example for Financial Time Series
% This example demonstrates how to use various statistical tests included in the
% MFE Toolbox to analyze financial time series data. The tests covered include:
%
% * Stationarity Tests:
%   - Augmented Dickey-Fuller (ADF) test
%   - Phillips-Perron (PP) test
%   - Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test
%
% * Autocorrelation Tests:
%   - Ljung-Box Q-test
%   - Lagrange Multiplier test
%
% * Normality Tests:
%   - Jarque-Bera test
%
% * Volatility Tests:
%   - ARCH test
%   - White test for heteroskedasticity
%
% * Nonlinear Dependence Tests:
%   - BDS test
%
% Each test is demonstrated with proper parameter configuration, result interpretation,
% and visualization where appropriate.

% Define formatting constants for better output readability
HEADER_FORMAT = '\n%s\n%s\n%s\n\n';
SUBHEADER_FORMAT = '\n%s\n%s\n\n';
RESULT_FORMAT = '  %-25s: %s\n';

function displayTestHeader(headerText)
    % Display a formatted header for each test section in the example
    fprintf(HEADER_FORMAT, repmat('=', 1, 80), headerText, repmat('=', 1, 80));
end

function displayTestSubheader(subheaderText)
    % Display a formatted subheader for test variants or options
    fprintf(SUBHEADER_FORMAT, subheaderText, repmat('-', 1, length(subheaderText)));
end

function runStationarityTests(returnSeries, assetName)
    % Demonstrate and compare ADF, PP, and KPSS tests for stationarity and unit roots
    
    displayTestHeader('STATIONARITY TESTS');
    
    fprintf(['Stationarity tests examine whether a time series exhibits constant statistical\n',...
             'properties over time. For financial returns, stationarity is often a key assumption\n',...
             'underlying many models. Three tests are demonstrated below:\n\n',...
             '1. ADF Test - Tests the null hypothesis of a unit root (non-stationarity)\n',...
             '2. PP Test - Similar to ADF but robust to heteroskedasticity\n',...
             '3. KPSS Test - Tests the null hypothesis of stationarity (opposite of ADF/PP)\n\n',...
             'For returns series, we typically expect stationarity.\n\n']);
    
    % Run ADF test
    displayTestSubheader('Augmented Dickey-Fuller (ADF) Test');
    options = struct('regression_type', 'c', 'lags', 'aic');
    adf_results = adf_test(returnSeries, options);
    
    fprintf('Test statistic: %.4f (p-value: %.4f)\n', adf_results.stat, adf_results.pval);
    fprintf('Critical values: [1%%: %.4f, 5%%: %.4f, 10%%: %.4f]\n', ...
        adf_results.crit_vals(1), adf_results.crit_vals(2), adf_results.crit_vals(3));
    
    if adf_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of non-stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series appears to be stationary.\n', assetName);
    else
        fprintf('Result: Cannot reject null hypothesis of non-stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series may contain a unit root.\n', assetName);
    end
    
    % Run PP test
    displayTestSubheader('Phillips-Perron (PP) Test');
    options = struct('regression_type', 'c');
    pp_results = pp_test(returnSeries, options);
    
    fprintf('Test statistic: %.4f (p-value: %.4f)\n', pp_results.stat_tau, pp_results.pval);
    fprintf('Critical values: [1%%: %.4f, 5%%: %.4f, 10%%: %.4f]\n', ...
        pp_results.cv_1pct, pp_results.cv_5pct, pp_results.cv_10pct);
    
    if pp_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of non-stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series appears to be stationary.\n', assetName);
    else
        fprintf('Result: Cannot reject null hypothesis of non-stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series may contain a unit root.\n', assetName);
    end
    
    % Run KPSS test
    displayTestSubheader('KPSS Test');
    options = struct('regression_type', 'mu');
    kpss_results = kpss_test(returnSeries, options);
    
    fprintf('Test statistic: %.4f (p-value: %.4f)\n', kpss_results.stat, kpss_results.pval);
    fprintf('Critical values: [1%%: %.4f, 2.5%%: %.4f, 5%%: %.4f, 10%%: %.4f]\n', ...
        kpss_results.cv(1), kpss_results.cv(2), kpss_results.cv(3), kpss_results.cv(4));
    
    if kpss_results.pval > 0.05
        fprintf('Result: Cannot reject null hypothesis of stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series appears to be stationary.\n', assetName);
    else
        fprintf('Result: Reject null hypothesis of stationarity at 5%% significance.\n');
        fprintf('Interpretation: The %s series may not be stationary.\n', assetName);
    end
    
    % Summary of stationarity tests
    displayTestSubheader('Summary of Stationarity Tests');
    fprintf(['For financial returns, we generally expect:\n',...
             ' - ADF/PP tests to reject the null (indicating stationarity)\n',...
             ' - KPSS test to not reject the null (also indicating stationarity)\n\n']);
    
    % Consensus interpretation
    if adf_results.pval < 0.05 && pp_results.pval < 0.05 && kpss_results.pval > 0.05
        fprintf('Overall conclusion: Strong evidence that the %s series is stationary.\n', assetName);
    elseif (adf_results.pval < 0.05 || pp_results.pval < 0.05) && kpss_results.pval > 0.05
        fprintf('Overall conclusion: Moderate evidence that the %s series is stationary.\n', assetName);
    elseif adf_results.pval < 0.05 && pp_results.pval < 0.05 && kpss_results.pval < 0.05
        fprintf(['Overall conclusion: Conflicting evidence. The %s series rejects unit root\n',...
                 'but also rejects stationarity. This can occur with series that are\n',...
                 'stationary but exhibit persistence or structural changes.\n'], assetName);
    else
        fprintf('Overall conclusion: Evidence suggests the %s series may not be stationary.\n', assetName);
    end
    
    % Visualization
    figure;
    subplot(2,1,1);
    plot(returnSeries);
    title(sprintf('%s Returns', assetName));
    xlabel('Time');
    ylabel('Returns');
    grid on;
    
    subplot(2,1,2);
    % Plot first differences to visualize stationarity transformation if needed
    plot(diff(returnSeries));
    title(sprintf('First Differences of %s Returns', assetName));
    xlabel('Time');
    ylabel('First Differences');
    grid on;
end

function runAutocorrelationTest(returnSeries, assetName)
    % Demonstrate the Ljung-Box test for autocorrelation in time series
    
    displayTestHeader('AUTOCORRELATION TESTS');
    
    fprintf(['Autocorrelation tests examine whether past values of a time series help predict\n',...
             'future values. In efficient markets, returns should not be predictable from past\n',...
             'returns, implying no significant autocorrelation. The Ljung-Box Q-test is a\n',...
             'commonly used test for detecting autocorrelation up to a specified lag.\n\n']);
    
    % Run Ljung-Box test with default lags
    displayTestSubheader('Ljung-Box Q-Test');
    max_lag = 10;
    lb_results = ljungbox(returnSeries, max_lag);
    
    fprintf('Testing for autocorrelation up to lag %d:\n\n', max_lag);
    fprintf('%-6s %-12s %-12s %-12s\n', 'Lag', 'Q-Stat', 'p-value', 'Significant at 5%%');
    
    for i = 1:length(lb_results.lags)
        fprintf('%-6d %-12.4f %-12.4f %-12s\n', ...
            lb_results.lags(i), lb_results.stats(i), lb_results.pvals(i), ...
            iif(lb_results.isRejected5pct(i), 'Yes', 'No'));
    end
    
    % Check if any lags show significant autocorrelation
    if any(lb_results.isRejected5pct)
        fprintf('\nResult: Reject null hypothesis of no autocorrelation at 5%% significance.\n');
        fprintf('Interpretation: The %s series exhibits significant autocorrelation.\n', assetName);
        
        % Identify which lags are significant
        sig_lags = lb_results.lags(lb_results.isRejected5pct);
        fprintf('Significant autocorrelation detected at lags: %s\n', mat2str(sig_lags));
    else
        fprintf('\nResult: Cannot reject null hypothesis of no autocorrelation at 5%% significance.\n');
        fprintf('Interpretation: No significant autocorrelation detected in the %s series.\n', assetName);
    end
    
    % Additional test with different lag specification
    displayTestSubheader('Ljung-Box Q-Test with Different Lag Specifications');
    fprintf(['The choice of maximum lag can affect conclusions. Here we compare results\n',...
             'using different lag specifications.\n\n']);
    
    lag_choices = [5, 10, 20];
    
    for i = 1:length(lag_choices)
        curr_lag = lag_choices(i);
        lb_results = ljungbox(returnSeries, curr_lag);
        
        % Determine overall significance for this lag specification
        any_sig = any(lb_results.isRejected5pct);
        
        fprintf('Maximum lag = %d: %s (p-min = %.4f)\n', ...
            curr_lag, ...
            iif(any_sig, 'Significant autocorrelation detected', 'No significant autocorrelation'), ...
            min(lb_results.pvals));
    end
    
    % Run LM test for autocorrelation as an alternative test
    displayTestSubheader('Lagrange Multiplier Test for Autocorrelation');
    lm_results = lmtest1(returnSeries, 10);
    
    fprintf('LM test statistic: %.4f (p-value: %.4f)\n', lm_results.stat, lm_results.pval);
    fprintf('Critical values: [10%%: %.4f, 5%%: %.4f, 1%%: %.4f]\n', ...
        lm_results.crit(1), lm_results.crit(2), lm_results.crit(3));
    
    if lm_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of no autocorrelation at 5%% significance.\n');
        fprintf('Interpretation: The %s series exhibits significant autocorrelation.\n', assetName);
    else
        fprintf('Result: Cannot reject null hypothesis of no autocorrelation at 5%% significance.\n');
        fprintf('Interpretation: No significant autocorrelation detected in the %s series.\n', assetName);
    end
    
    % Visualization
    figure;
    
    % Plot ACF
    subplot(2,1,1);
    [acf, se, ci] = sacf(returnSeries, 20);
    
    % Plot ACF
    bar(0:20, [1; acf]); % Include lag 0 (always 1)
    hold on;
    
    % Plot confidence bounds
    plot([0, 20], [1.96/sqrt(length(returnSeries)), 1.96/sqrt(length(returnSeries))], 'r--');
    plot([0, 20], [-1.96/sqrt(length(returnSeries)), -1.96/sqrt(length(returnSeries))], 'r--');
    
    title(sprintf('%s Returns - Autocorrelation Function', assetName));
    xlabel('Lag');
    ylabel('ACF');
    grid on;
    legend('ACF', '95% Confidence Bounds');
    
    % Plot Ljung-Box statistics and p-values
    subplot(2,1,2);
    lb_results = ljungbox(returnSeries, 20);
    
    yyaxis left;
    bar(lb_results.lags, lb_results.stats);
    ylabel('Ljung-Box Q Statistic');
    
    yyaxis right;
    plot(lb_results.lags, lb_results.pvals, 'r-o', 'LineWidth', 2);
    yline(0.05, 'r--', '5% Significance');
    ylabel('p-value');
    
    title(sprintf('%s Returns - Ljung-Box Q-Test', assetName));
    xlabel('Lag');
    grid on;
    legend('Q-Statistic', 'p-value', '5% Significance Level');
end

function runNormalityTest(returnSeries, assetName)
    % Demonstrate the Jarque-Bera test for normality of return distribution
    
    displayTestHeader('NORMALITY TESTS');
    
    fprintf(['Normality tests examine whether a data series follows a normal distribution. Many\n',...
             'financial models assume normality of returns, but empirical evidence often shows\n',...
             'that returns have "fat tails" and are not normally distributed. The Jarque-Bera\n',...
             'test is commonly used to test for normality based on skewness and kurtosis.\n\n']);
    
    % Calculate summary statistics
    mu = mean(returnSeries);
    sigma = std(returnSeries);
    skew = sum((returnSeries - mu).^3/sigma^3)/length(returnSeries);
    kurt = sum((returnSeries - mu).^4/sigma^4)/length(returnSeries);
    excess_kurt = kurt - 3;
    
    % Display descriptive statistics
    displayTestSubheader('Descriptive Statistics');
    fprintf('Sample size:     %d\n', length(returnSeries));
    fprintf('Mean:            %.6f\n', mu);
    fprintf('Std. Deviation:  %.6f\n', sigma);
    fprintf('Skewness:        %.6f\n', skew);
    fprintf('Kurtosis:        %.6f\n', kurt);
    fprintf('Excess Kurtosis: %.6f\n', excess_kurt);
    
    % Run Jarque-Bera test
    displayTestSubheader('Jarque-Bera Test for Normality');
    jb_results = jarque_bera(returnSeries);
    
    fprintf('Jarque-Bera statistic: %.4f (p-value: %.6f)\n', jb_results.statistic, jb_results.pval);
    fprintf('Critical values: [10%%: %.4f, 5%%: %.4f, 1%%: %.4f]\n', ...
        jb_results.crit_val(1), jb_results.crit_val(2), jb_results.crit_val(3));
    
    if jb_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of normality at 5%% significance.\n');
        if skew < 0
            skew_desc = 'negatively skewed (longer left tail)';
        elseif skew > 0
            skew_desc = 'positively skewed (longer right tail)';
        else
            skew_desc = 'approximately symmetric';
        end
        
        if excess_kurt > 0
            kurt_desc = 'leptokurtic (fat tails)';
        elseif excess_kurt < 0
            kurt_desc = 'platykurtic (thin tails)';
        else
            kurt_desc = 'mesokurtic (normal tails)';
        end
        
        fprintf(['Interpretation: The %s returns are not normally distributed. The distribution is %s\n',...
                 'and %s compared to a normal distribution.\n'], assetName, skew_desc, kurt_desc);
    else
        fprintf('Result: Cannot reject null hypothesis of normality at 5%% significance.\n');
        fprintf('Interpretation: The %s returns appear to be normally distributed.\n', assetName);
    end
    
    % Visualization
    figure;
    
    % Histogram with normal distribution overlay
    subplot(2,1,1);
    histogram(returnSeries, 50, 'Normalization', 'pdf');
    hold on;
    
    x = linspace(min(returnSeries), max(returnSeries), 1000);
    y = normpdf(x, mu, sigma);
    plot(x, y, 'r', 'LineWidth', 2);
    
    title(sprintf('%s Returns - Histogram with Normal Density', assetName));
    xlabel('Returns');
    ylabel('Probability Density');
    legend('Observed Returns', 'Normal Distribution');
    grid on;
    
    % QQ plot
    subplot(2,1,2);
    sorted_returns = sort(returnSeries);
    n = length(returnSeries);
    p = ((1:n) - 0.5)' / n;
    
    % Theoretical normal quantiles
    norm_quantiles = norminv(p, 0, 1);
    
    % Standardize observed returns
    std_returns = (sorted_returns - mu) / sigma;
    
    % Create QQ plot
    plot(norm_quantiles, std_returns, 'b.');
    hold on;
    plot([-4, 4], [-4, 4], 'r-', 'LineWidth', 1.5);
    
    title(sprintf('%s Returns - Normal QQ Plot', assetName));
    xlabel('Theoretical Quantiles');
    ylabel('Sample Quantiles');
    grid on;
    
    % Add descriptive text
    text_x = -3.5;
    text_y = 3;
    
    if excess_kurt > 0
        text(text_x, text_y, sprintf('Fat tails (Excess Kurtosis: %.2f)', excess_kurt), ...
            'FontSize', 10, 'Color', 'blue');
    elseif excess_kurt < 0
        text(text_x, text_y, sprintf('Thin tails (Excess Kurtosis: %.2f)', excess_kurt), ...
            'FontSize', 10, 'Color', 'blue');
    end
    
    if skew < -0.1
        text(text_x, text_y-0.5, sprintf('Negative Skew: %.2f', skew), ...
            'FontSize', 10, 'Color', 'blue');
    elseif skew > 0.1
        text(text_x, text_y-0.5, sprintf('Positive Skew: %.2f', skew), ...
            'FontSize', 10, 'Color', 'blue');
    end
end

function runVolatilityTests(returnSeries, assetName)
    % Demonstrate tests for volatility clustering and ARCH effects
    
    displayTestHeader('VOLATILITY CLUSTERING TESTS');
    
    fprintf(['Volatility clustering is a common feature in financial returns, where periods\n',...
             'of high volatility tend to be followed by periods of high volatility, and vice\n',...
             'versa. The ARCH test examines whether squared returns exhibit autocorrelation,\n',...
             'which would indicate the presence of volatility clustering.\n\n']);
    
    % Calculate squared returns
    squared_returns = returnSeries.^2;
    
    % Run ARCH test
    displayTestSubheader('ARCH Test for Volatility Clustering');
    arch_results = arch_test(returnSeries);
    
    fprintf('ARCH test statistic: %.4f (p-value: %.6f)\n', arch_results.statistic, arch_results.pval);
    fprintf('Critical values: [10%%: %.4f, 5%%: %.4f, 1%%: %.4f]\n', ...
        arch_results.critical.ten, arch_results.critical.five, arch_results.critical.one);
    
    if arch_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of no ARCH effects at 5%% significance.\n');
        fprintf(['Interpretation: The %s returns exhibit significant volatility clustering,\n',...
                 'suggesting the presence of ARCH effects. GARCH models may be appropriate\n',...
                 'for modeling the conditional variance of this series.\n'], assetName);
    else
        fprintf('Result: Cannot reject null hypothesis of no ARCH effects at 5%% significance.\n');
        fprintf(['Interpretation: No significant volatility clustering detected in the %s returns.\n',...
                 'GARCH modeling may not be necessary for this series.\n'], assetName);
    end
    
    % Additional test with different lag specifications
    displayTestSubheader('ARCH Test with Different Lag Specifications');
    fprintf(['Volatility persistence can occur at different time scales. We test for ARCH\n',...
             'effects using different lag specifications to capture various persistence patterns.\n\n']);
    
    lag_choices = [1, 5, 10, 20];
    
    for i = 1:length(lag_choices)
        curr_lag = lag_choices(i);
        arch_results = arch_test(returnSeries, curr_lag);
        
        fprintf('ARCH test with %d lag(s): Stat = %.4f, p-value = %.6f, Significant: %s\n', ...
            curr_lag, arch_results.statistic, arch_results.pval, ...
            iif(arch_results.pval < 0.05, 'Yes', 'No'));
    end
    
    % Run White test for heteroskedasticity 
    displayTestSubheader('White Test for Heteroskedasticity');
    
    % Need design matrix for White test
    % Create a simple AR(1) model as an example
    T = length(returnSeries);
    lagged_returns = [NaN; returnSeries(1:end-1)];
    X = [ones(T, 1), lagged_returns];
    X = X(2:end, :); % Remove first observation with NaN
    y = returnSeries(2:end);
    
    % OLS regression
    beta = (X'*X)\(X'*y);
    residuals = y - X*beta;
    
    % Run White test
    white_results = white_test(residuals, X);
    
    fprintf('White test statistic: %.4f (p-value: %.6f)\n', white_results.stat, white_results.pval);
    fprintf('Critical values: [10%%: %.4f, 5%%: %.4f, 1%%: %.4f]\n', ...
        white_results.crit(1), white_results.crit(2), white_results.crit(3));
    
    if white_results.pval < 0.05
        fprintf('Result: Reject null hypothesis of homoskedasticity at 5%% significance.\n');
        fprintf(['Interpretation: The residuals exhibit heteroskedasticity, suggesting that\n',...
                 'variance of the %s returns changes over time. Consider using models that\n',...
                 'account for time-varying volatility or robust standard errors.\n'], assetName);
    else
        fprintf('Result: Cannot reject null hypothesis of homoskedasticity at 5%% significance.\n');
        fprintf(['Interpretation: No significant heteroskedasticity detected in the residuals.\n',...
                 'Standard OLS inference may be valid for this series.\n']);
    end
    
    % Visualization
    figure;
    
    % Plot returns and squared returns
    subplot(2,1,1);
    plot(returnSeries);
    title(sprintf('%s Returns', assetName));
    xlabel('Time');
    ylabel('Returns');
    grid on;
    
    subplot(2,1,2);
    plot(squared_returns);
    title(sprintf('%s Squared Returns (Volatility Proxy)', assetName));
    xlabel('Time');
    ylabel('Squared Returns');
    grid on;
    
    % Plot autocorrelation of squared returns
    figure;
    [acf_sq, se_sq, ci_sq] = sacf(squared_returns, 20);
    
    bar(1:20, acf_sq);
    hold on;
    
    % Plot confidence bounds
    plot([0, 20], [1.96/sqrt(length(returnSeries)), 1.96/sqrt(length(returnSeries))], 'r--');
    plot([0, 20], [-1.96/sqrt(length(returnSeries)), -1.96/sqrt(length(returnSeries))], 'r--');
    
    title(sprintf('%s Squared Returns - Autocorrelation Function', assetName));
    xlabel('Lag');
    ylabel('ACF of Squared Returns');
    grid on;
    legend('ACF', '95% Confidence Bounds');
end

function runBDSTest(returnSeries, assetName)
    % Demonstrate the BDS test for nonlinear dependence in time series
    
    displayTestHeader('NONLINEAR DEPENDENCE TESTS');
    
    fprintf(['The BDS test (Brock, Dechert, Scheinkman) is a powerful test for detecting\n',...
             'nonlinear structure in time series data. It examines whether the data is\n',...
             'independent and identically distributed (i.i.d.) against an unspecified\n',...
             'alternative that could include nonlinear dependence or chaos.\n\n']);
    
    % Run BDS test with default parameters
    displayTestSubheader('BDS Test for Nonlinear Dependence');
    dimensions = 2:5;
    epsilon = 0.7 * std(returnSeries);
    
    bds_results = bds_test(returnSeries, dimensions, epsilon);
    
    fprintf('BDS test with dimensions %s and epsilon = %.4f:\n\n', mat2str(dimensions), epsilon);
    fprintf('%-10s %-15s %-15s %-15s\n', 'Dimension', 'BDS Statistic', 'p-value', 'Reject H0 at 5%%');
    
    for i = 1:length(dimensions)
        fprintf('%-10d %-15.4f %-15.6f %-15s\n', ...
            bds_results.dim(i), bds_results.stat(i), bds_results.pval(i), ...
            iif(bds_results.H(i), 'Yes', 'No'));
    end
    
    % Overall interpretation
    if any(bds_results.H)
        fprintf('\nResult: Reject null hypothesis of i.i.d. for at least one dimension.\n');
        fprintf(['Interpretation: The %s returns exhibit nonlinear dependence. This suggests\n',...
                 'that linear models may not fully capture the dynamics of this series.\n',...
                 'Consider nonlinear models or models that capture time-varying volatility.\n'], assetName);
    else
        fprintf('\nResult: Cannot reject null hypothesis of i.i.d. for any dimension.\n');
        fprintf(['Interpretation: No significant nonlinear dependence detected in the %s returns.\n',...
                 'Linear models may be sufficient for this series.\n'], assetName);
    end
    
    % Additional test with different epsilon values
    displayTestSubheader('BDS Test with Different Epsilon Values');
    fprintf(['The BDS test results can be sensitive to the choice of epsilon parameter.\n',...
             'We compare results using different epsilon values to assess robustness.\n\n']);
    
    epsilon_choices = [0.5, 0.7, 1.0, 1.5] * std(returnSeries);
    dimension = 2;  % Fixed dimension for comparison
    
    fprintf('%-15s %-15s %-15s %-15s\n', 'Epsilon', 'BDS Statistic', 'p-value', 'Reject H0 at 5%%');
    
    for i = 1:length(epsilon_choices)
        curr_epsilon = epsilon_choices(i);
        bds_results = bds_test(returnSeries, dimension, curr_epsilon);
        
        % Results for the first dimension only
        fprintf('%-15.4f %-15.4f %-15.6f %-15s\n', ...
            curr_epsilon, bds_results.stat(1), bds_results.pval(1), ...
            iif(bds_results.pval(1) < 0.05, 'Yes', 'No'));
    end
end

function result = iif(condition, trueValue, falseValue)
    % Simple if function for compact conditional expressions
    if condition
        result = trueValue;
    else
        result = falseValue;
    end
end

function runAllTests()
    % Main function that runs all statistical tests in a comprehensive example
    
    % Introduction
    fprintf('\n%s\n', repmat('*', 1, 80));
    fprintf('STATISTICAL TESTS EXAMPLE - MFE TOOLBOX\n');
    fprintf('%s\n\n', repmat('*', 1, 80));
    
    fprintf(['This example demonstrates the proper use of various statistical tests included\n',...
             'in the MFE Toolbox for financial time series analysis. Each test is presented\n',...
             'with explanations, interpretations, and visualizations where appropriate.\n\n']);
    
    % Load test data
    fprintf('Loading financial returns test data...\n');
    load financial_returns.mat;
    
    % For this example, we'll select a few representative series
    % Assuming the data contains multiple assets
    stock_idx = 1;  % e.g., S&P 500 returns
    bond_idx = 2;   % e.g., 10-year Treasury returns
    fx_idx = 3;     % e.g., EUR/USD returns
    
    % Extract return series for the selected assets
    stock_returns = returns(:, stock_idx);
    bond_returns = returns(:, bond_idx);
    fx_returns = returns(:, fx_idx);
    
    % Asset names for display
    stock_name = 'Stock';
    bond_name = 'Bond';
    fx_name = 'FX';
    
    % Run all tests for the stock returns
    fprintf('\n%s\n', repmat('#', 1, 80));
    fprintf('TESTING %s RETURNS\n', upper(stock_name));
    fprintf('%s\n', repmat('#', 1, 80));
    
    runStationarityTests(stock_returns, stock_name);
    runAutocorrelationTest(stock_returns, stock_name);
    runNormalityTest(stock_returns, stock_name);
    runVolatilityTests(stock_returns, stock_name);
    runBDSTest(stock_returns, stock_name);
    
    % Run all tests for the bond returns
    fprintf('\n%s\n', repmat('#', 1, 80));
    fprintf('TESTING %s RETURNS\n', upper(bond_name));
    fprintf('%s\n', repmat('#', 1, 80));
    
    runStationarityTests(bond_returns, bond_name);
    runAutocorrelationTest(bond_returns, bond_name);
    runNormalityTest(bond_returns, bond_name);
    runVolatilityTests(bond_returns, bond_name);
    runBDSTest(bond_returns, bond_name);
    
    % Run all tests for the FX returns
    fprintf('\n%s\n', repmat('#', 1, 80));
    fprintf('TESTING %s RETURNS\n', upper(fx_name));
    fprintf('%s\n', repmat('#', 1, 80));
    
    runStationarityTests(fx_returns, fx_name);
    runAutocorrelationTest(fx_returns, fx_name);
    runNormalityTest(fx_returns, fx_name);
    runVolatilityTests(fx_returns, fx_name);
    runBDSTest(fx_returns, fx_name);
    
    % Conclusion
    fprintf('\n%s\n', repmat('*', 1, 80));
    fprintf('CONCLUSION\n');
    fprintf('%s\n\n', repmat('*', 1, 80));
    
    fprintf(['This example has demonstrated a comprehensive approach to testing financial\n',...
             'time series data using the MFE Toolbox. Key insights to remember:\n\n',...
             '1. Stationarity is a fundamental assumption for many models and should be tested first.\n',...
             '2. Returns often exhibit stylized facts like non-normality and volatility clustering.\n',...
             '3. Using multiple tests provides more robust insights than relying on a single test.\n',...
             '4. Test results should guide model selection (e.g., GARCH for volatility clustering).\n',...
             '5. Interpreting test results requires understanding both statistical significance\n',...
             '   and economic/financial significance.\n\n',...
             'For more advanced analyses with the MFE Toolbox, try combining these tests with\n',...
             'the modeling functions like GARCH, ARMAX, realized volatility, and bootstrap methods.\n']);
end

% Call the main function to run all tests
runAllTests();