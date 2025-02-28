%% Time Series Modeling with the MFE Toolbox
% This script demonstrates a comprehensive time series modeling workflow
% using the MATLAB Financial Econometrics (MFE) Toolbox, including:
% - Model identification (ACF/PACF analysis)
% - ARMA/ARMAX model estimation
% - Model selection using information criteria
% - Diagnostic checking
% - Forecasting
% - Volatility modeling with GARCH

%% Initialize the MFE Toolbox
% Add necessary directories to the MATLAB path
try
    addToPath();
    disp('MFE Toolbox successfully initialized.');
catch ME
    error('Failed to initialize MFE Toolbox: %s', ME.message);
end

%% Load or Generate Example Data
% Try to load example financial data if available, otherwise generate synthetic data
try
    % Attempt to load example data from MAT file
    % This would typically be a file in the examples/data directory
    load('examples/data/financial_returns.mat', 'returns');
    disp('Loaded example financial returns data.');
catch
    % If data file is not available, generate synthetic financial return series
    % with realistic properties (slight autocorrelation, volatility clustering)
    disp('Example data file not found. Generating synthetic financial returns.');
    
    % Set random seed for reproducibility
    rng(42);
    
    % Number of observations
    T = 1000;
    
    % Generate return series with autocorrelation and volatility clustering
    innovations = randn(T, 1);
    volatility = ones(T, 1);
    returns = zeros(T, 1);
    
    % Parameters for data generation
    ar_coef = 0.2;        % AR(1) coefficient
    garch_alpha = 0.1;    % ARCH effect
    garch_beta = 0.85;    % GARCH effect
    garch_omega = 0.01;   % Constant term for variance
    
    % Generate time series with AR(1) structure and GARCH(1,1) innovations
    for t = 2:T
        % Compute volatility (GARCH process)
        volatility(t) = sqrt(garch_omega + garch_alpha * (returns(t-1)^2) + garch_beta * volatility(t-1)^2);
        
        % Generate return with AR(1) structure and time-varying volatility
        returns(t) = ar_coef * returns(t-1) + volatility(t) * innovations(t);
    end
    
    disp('Synthetic financial returns generated with AR(1)-GARCH(1,1) structure.');
end

%% Preliminary Data Analysis
% Plot the time series data
figure;
plot(returns);
title('Financial Returns Time Series');
xlabel('Time');
ylabel('Returns');

% Basic descriptive statistics
T = length(returns);
mean_returns = mean(returns);
std_returns = std(returns);

% Calculate skewness manually
centered_returns = returns - mean_returns;
skewness_value = sum(centered_returns.^3) / (T * std_returns^3);

% Calculate kurtosis manually
kurtosis_value = sum(centered_returns.^4) / (T * std_returns^4);

disp('Descriptive Statistics:');
disp(['Mean: ', num2str(mean_returns)]);
disp(['Standard Deviation: ', num2str(std_returns)]);
disp(['Skewness: ', num2str(skewness_value)]);
disp(['Kurtosis: ', num2str(kurtosis_value), ' (Normal = 3)']);
disp(['Excess Kurtosis: ', num2str(kurtosis_value - 3)]);

%% Calculate and Plot ACF and PACF for Model Identification
% Maximum number of lags to consider
max_lags = min(20, floor(T/4));

try
    % Calculate sample autocorrelation function (ACF)
    [acf_values, acf_se, acf_ci] = sacf(returns, max_lags);
    
    % Calculate sample partial autocorrelation function (PACF)
    [pacf_values, pacf_se, pacf_ci] = spacf(returns, max_lags);
    
    % Plot ACF
    figure;
    subplot(2, 1, 1);
    bar(1:max_lags, acf_values);
    hold on;
    plot(1:max_lags, acf_ci(:, 1), 'r--');
    plot(1:max_lags, acf_ci(:, 2), 'r--');
    title('Sample Autocorrelation Function (ACF)');
    xlabel('Lag');
    ylabel('Autocorrelation');
    grid on;
    hold off;
    
    % Plot PACF
    subplot(2, 1, 2);
    bar(1:max_lags, pacf_values);
    hold on;
    plot(1:max_lags, pacf_ci(:, 1), 'r--');
    plot(1:max_lags, pacf_ci(:, 2), 'r--');
    title('Sample Partial Autocorrelation Function (PACF)');
    xlabel('Lag');
    ylabel('Partial Autocorrelation');
    grid on;
    hold off;
    
    % Interpret ACF/PACF patterns for model identification
    disp('Model Identification:');
    disp('Based on ACF and PACF patterns:');
    
    % Check for significant autocorrelations
    significant_acf = abs(acf_values) > (1.96 / sqrt(T));
    significant_pacf = abs(pacf_values) > (1.96 / sqrt(T));
    
    if any(significant_acf)
        disp(['- ACF shows significant autocorrelation at lags: ', ...
            num2str(find(significant_acf))]);
    else
        disp('- ACF shows no significant autocorrelation');
    end
    
    if any(significant_pacf)
        disp(['- PACF shows significant partial autocorrelation at lags: ', ...
            num2str(find(significant_pacf))]);
    else
        disp('- PACF shows no significant partial autocorrelation');
    end
    
    disp('- We will try different ARMA(p,q) specifications and select based on information criteria');
catch ME
    error('Error in ACF/PACF analysis: %s', ME.message);
end

%% Estimate Multiple ARMA Models
% Define a range of ARMA orders to test
p_max = 3;  % Maximum AR order
q_max = 3;  % Maximum MA order

% Initialize arrays to store model results
num_models = (p_max + 1) * (q_max + 1) - 1;  % Excluding ARMA(0,0)
arma_models = cell(num_models, 1);
aic_values = zeros(num_models, 1);
bic_values = zeros(num_models, 1);
model_names = cell(num_models, 1);
model_orders = zeros(num_models, 2);  % Store p,q orders

% Estimate ARMA models with different orders
idx = 1;
disp('Estimating ARMA models with different orders:');
for p = 0:p_max
    for q = 0:q_max
        % Skip the case of p=0 and q=0 (no model)
        if p == 0 && q == 0
            continue;
        end
        
        % Set model options
        options = struct('p', p, 'q', q, 'distribution', 'NORMAL');
        
        % Estimate ARMA(p,q) model
        model_name = sprintf('ARMA(%d,%d)', p, q);
        disp(['  Estimating ', model_name, '...']);
        
        try
            % Use armaxfilter to estimate the model
            model = armaxfilter(returns, [], options);
            
            % Store model results
            arma_models{idx} = model;
            aic_values(idx) = model.aic;
            bic_values(idx) = model.sbic;
            model_names{idx} = model_name;
            model_orders(idx, :) = [p, q];
            
            disp(['    AIC: ', num2str(model.aic), ', BIC: ', num2str(model.sbic)]);
            idx = idx + 1;
        catch ME
            warning('Error estimating %s: %s', model_name, ME.message);
        end
    end
end

% Adjust arrays to only include successfully estimated models
num_estimated = idx - 1;
if num_estimated < 1
    error('No ARMA models could be successfully estimated.');
end

arma_models = arma_models(1:num_estimated);
aic_values = aic_values(1:num_estimated);
bic_values = bic_values(1:num_estimated);
model_names = model_names(1:num_estimated);
model_orders = model_orders(1:num_estimated, :);

%% Select the Best Model Based on Information Criteria
% Find the model with the lowest AIC and BIC
[min_aic, aic_idx] = min(aic_values);
[min_bic, bic_idx] = min(bic_values);

% AIC-selected model
aic_best_model = arma_models{aic_idx};
aic_best_name = model_names{aic_idx};
aic_best_orders = model_orders(aic_idx, :);

% BIC-selected model
bic_best_model = arma_models{bic_idx};
bic_best_name = model_names{bic_idx};
bic_best_orders = model_orders(bic_idx, :);

disp('Model Selection Results:');
disp(['Best model by AIC: ', aic_best_name, ' (AIC = ', num2str(min_aic), ')']);
disp(['Best model by BIC: ', bic_best_name, ' (BIC = ', num2str(min_bic), ')']);

% Choose the model selected by BIC (typically more parsimonious)
% BIC tends to penalize complexity more heavily, which is often preferred for forecasting
best_model = bic_best_model;
best_model_name = bic_best_name;
disp(['Selected model for further analysis: ', best_model_name]);

%% Model Diagnostics
% Extract residuals from the best model
residuals = best_model.residuals;

% Plot residuals
figure;
subplot(3, 1, 1);
plot(residuals);
title(['Residuals from ', best_model_name]);
xlabel('Time');
ylabel('Residual');
grid on;

% Plot histogram of residuals with normal distribution overlay
subplot(3, 1, 2);
[counts, edges] = histcounts(residuals, 'Normalization', 'pdf');
edges_centers = (edges(1:end-1) + edges(2:end)) / 2;
bar(edges_centers, counts);
hold on;
x_range = linspace(min(residuals), max(residuals), 100);
plot(x_range, normpdf(x_range, 0, std(residuals)), 'r-', 'LineWidth', 2);
title('Histogram of Residuals with Normal Distribution');
xlabel('Residual Value');
ylabel('Density');
legend('Residuals', 'Normal Distribution');
grid on;
hold off;

% Plot ACF of squared residuals to check for ARCH effects
try
    [acf_sq_values, acf_sq_se, acf_sq_ci] = sacf(residuals.^2, max_lags);
    subplot(3, 1, 3);
    bar(1:max_lags, acf_sq_values);
    hold on;
    plot(1:max_lags, acf_sq_ci(:, 1), 'r--');
    plot(1:max_lags, acf_sq_ci(:, 2), 'r--');
    title('ACF of Squared Residuals');
    xlabel('Lag');
    ylabel('Autocorrelation');
    grid on;
    hold off;
catch ME
    warning('Error in ACF analysis of squared residuals: %s', ME.message);
end

% Perform Ljung-Box test for residual autocorrelation
try
    lb_results = ljungbox(residuals, max_lags, best_model.p + best_model.q);
    disp('Ljung-Box Test for Residual Autocorrelation:');
    disp('  Lag   Q-Stat   p-value   Decision (5% level)');
    disp('  ---   ------   -------   ------------------');
    for i = 1:length(lb_results.lags)
        decision = lb_results.isRejected5pct(i) ? 'Reject H0' : 'Fail to reject H0';
        disp(sprintf('  %3d   %6.2f    %6.4f   %s', ...
            lb_results.lags(i), lb_results.stats(i), lb_results.pvals(i), decision));
    end
catch ME
    warning('Error in Ljung-Box test: %s', ME.message);
end

% Jarque-Bera test for normality
try
    jb_results = jarque_bera(residuals);
    disp('Jarque-Bera Test for Normality:');
    disp(['  Test Statistic: ', num2str(jb_results.statistic)]);
    disp(['  p-value: ', num2str(jb_results.pval)]);
    if jb_results.reject(2)  % Test at 5% significance level
        disp('  Decision: Reject normality (5% significance level)');
    else
        disp('  Decision: Fail to reject normality (5% significance level)');
    end
catch ME
    warning('Error in Jarque-Bera test: %s', ME.message);
end

%% Forecast Generation
% Number of steps to forecast
forecast_horizon = 20;

% Generate forecasts using the best model
% Extract model parameters
p = best_model.p;
q = best_model.q;
const = best_model.constant;
parameters = best_model.parameters;

try
    % Generate forecasts
    [forecasts, forecast_vars] = armafor(parameters, returns, p, q, const, [], forecast_horizon);
    
    % Calculate forecast confidence intervals (95%)
    conf_level = 0.95;
    z_value = norminv((1 + conf_level) / 2);  % For 95% CI, z ≈ 1.96
    forecast_std = sqrt(forecast_vars);
    lower_ci = forecasts - z_value * forecast_std;
    upper_ci = forecasts + z_value * forecast_std;
    
    % Plot the original series with forecasts and confidence intervals
    figure;
    hold on;
    % Plot original data
    plot(1:length(returns), returns, 'b-');
    % Plot forecasts
    plot(length(returns)+1:length(returns)+forecast_horizon, forecasts, 'r-', 'LineWidth', 2);
    % Plot confidence intervals
    plot(length(returns)+1:length(returns)+forecast_horizon, lower_ci, 'r:', 'LineWidth', 1.5);
    plot(length(returns)+1:length(returns)+forecast_horizon, upper_ci, 'r:', 'LineWidth', 1.5);
    % Add a vertical line at the end of the sample
    plot([length(returns) length(returns)], ylim, 'k--');
    title([best_model_name, ' Forecasts with 95% Confidence Intervals']);
    xlabel('Time');
    ylabel('Returns');
    legend('Historical Data', 'Forecasts', 'Lower 95% CI', 'Upper 95% CI', 'End of Sample');
    grid on;
    hold off;
    
    disp('Forecast Summary:');
    disp('  Horizon   Forecast   Std. Error   95% CI Lower   95% CI Upper');
    disp('  -------   --------   ----------   ------------   ------------');
    for i = 1:min(5, forecast_horizon)  % Show first 5 forecasts
        disp(sprintf('  %7d   %8.4f   %10.4f   %12.4f   %12.4f', ...
            i, forecasts(i), forecast_std(i), lower_ci(i), upper_ci(i)));
    end
    if forecast_horizon > 5
        disp('  ...');
    end
catch ME
    warning('Error in forecast generation: %s', ME.message);
end

%% Volatility Modeling
% Check if there's evidence of ARCH effects in the residuals
disp('Testing for ARCH effects in residuals:');

try
    % Calculate ACF of squared residuals again for reporting
    [acf_sq_values, ~, ~] = sacf(residuals.^2, max_lags);
    
    % Check if ACF of squared residuals shows significant autocorrelation
    has_arch_effects = any(abs(acf_sq_values) > (1.96 / sqrt(length(residuals))));
    
    if has_arch_effects
        disp('  Significant autocorrelation detected in squared residuals.');
        disp('  This suggests the presence of ARCH effects (volatility clustering).');
        disp('  Proceeding with GARCH modeling of residuals.');
        
        % Fit an asymmetric GARCH model to the residuals
        % We use AGARCH which can capture leverage effects common in financial returns
        garch_options = struct('distribution', 'T');  % Use t-distribution for fat tails
        
        disp('Fitting AGARCH(1,1) model to residuals...');
        garch_model = agarchfit(residuals, garch_options);
        
        % Display GARCH model results
        disp('AGARCH Model Results:');
        disp(['  Constant term (omega): ', num2str(garch_model.parameters(1))]);
        disp(['  ARCH effect (alpha): ', num2str(garch_model.parameters(2))]);
        disp(['  Asymmetry (gamma): ', num2str(garch_model.parameters(3))]);
        disp(['  GARCH effect (beta): ', num2str(garch_model.parameters(4))]);
        if strcmp(garch_options.distribution, 'T')
            disp(['  Degrees of Freedom (nu): ', num2str(garch_model.parameters(5))]);
        end
        disp(['  Log-likelihood: ', num2str(garch_model.LL)]);
        disp(['  AIC: ', num2str(garch_model.information_criteria.AIC)]);
        disp(['  BIC: ', num2str(garch_model.information_criteria.BIC)]);
        
        % Plot conditional variance from GARCH model
        figure;
        subplot(2, 1, 1);
        plot(residuals);
        title('Model Residuals');
        xlabel('Time');
        ylabel('Residual');
        grid on;
        
        subplot(2, 1, 2);
        plot(garch_model.ht);
        title('Conditional Variance from AGARCH Model');
        xlabel('Time');
        ylabel('Variance');
        grid on;
        
        % Check if there's asymmetry (leverage effect)
        if garch_model.parameters(3) < 0
            disp('  The negative asymmetry parameter indicates a leverage effect:');
            disp('  Negative returns tend to increase volatility more than positive returns.');
        elseif garch_model.parameters(3) > 0
            disp('  The positive asymmetry parameter indicates an inverse leverage effect:');
            disp('  Positive returns tend to increase volatility more than positive returns.');
        else
            disp('  No significant asymmetry detected in the volatility process.');
        end
        
        % Calculate persistence
        persistence = garch_model.diagnostics.persistence;
        disp(['  GARCH Persistence: ', num2str(persistence)]);
        disp(['  Volatility Half-life: ', num2str(garch_model.diagnostics.halflife), ' periods']);
    else
        disp('  No significant ARCH effects detected in the residuals.');
        disp('  GARCH modeling may not be necessary for this series.');
    end
catch ME
    warning('Error in volatility modeling: %s', ME.message);
end

%% Summary of Results
disp('========================================');
disp('SUMMARY OF TIME SERIES MODELING RESULTS');
disp('========================================');
disp(['Best ARMA Model: ', best_model_name]);
disp(['AIC: ', num2str(best_model.aic), ', BIC: ', num2str(best_model.sbic)]);
disp(' ');
disp('Parameter Estimates:');
for i = 1:length(best_model.paramNames)
    disp(['  ', best_model.paramNames{i}, ': ', num2str(best_model.parameters(i)), ...
          ' (t-stat: ', num2str(best_model.tStats(i)), ')']);
end
disp(' ');
disp('Residual Diagnostics:');
try
    disp(['  Ljung-Box Q(10): ', num2str(lb_results.stats(10)), ...
          ' (p-value: ', num2str(lb_results.pvals(10)), ')']);
catch
    disp('  Ljung-Box: Not available');
end
try
    disp(['  Jarque-Bera: ', num2str(jb_results.statistic), ...
          ' (p-value: ', num2str(jb_results.pval), ')']);
catch
    disp('  Jarque-Bera: Not available');
end
disp(' ');
disp('Forecasting:');
disp('  Generated multi-step forecasts with confidence intervals');
disp(' ');
try
    if exist('has_arch_effects', 'var') && has_arch_effects
        disp('Volatility Modeling:');
        disp('  ARCH effects detected and modeled with AGARCH(1,1)');
        disp(['  Persistence: ', num2str(persistence), ...
              ' (Half-life: ', num2str(garch_model.diagnostics.halflife), ' periods)']);
    end
catch
    % Do nothing
end
disp('========================================');