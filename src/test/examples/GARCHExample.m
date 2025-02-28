% GARCHExample.m - Demonstrates usage of GARCH volatility models in the MFE Toolbox
%
% This example demonstrates various GARCH (Generalized Autoregressive Conditional
% Heteroskedasticity) models for volatility modeling in financial time series.
% It covers model estimation, diagnostics, comparison, forecasting, and visualization.
%
% The example includes:
% 1. Basic GARCH(1,1) with normal distribution
% 2. Comparison of different error distributions
% 3. Comparison of different GARCH variants
% 4. Performance comparison with/without MEX optimization
% 5. Diagnostic testing of standardized residuals
% 6. Volatility forecasting
% 7. Results visualization
%
% This file is part of the MFE Toolbox 4.0.

%% Introduction
disp('===================================================================');
disp('MFE Toolbox Example: GARCH Volatility Modeling');
disp('===================================================================');
disp('This example demonstrates the use of GARCH-type models');
disp('for financial time series volatility modeling.');
disp(' ');

%% Load test data
% Load financial return data
try
    % Try to load test data file
    load('test_financial_returns.mat', 'returns');
    disp('Loaded test financial returns data.');
catch
    % If file not found, generate synthetic data for demonstration
    disp('Test data file not found. Generating synthetic returns data.');
    rng(42); % Set random seed for reproducibility
    T = 1000;
    returns = randn(T, 1);
    
    % Add GARCH-like volatility clustering
    volatility = ones(T, 1);
    for t = 2:T
        volatility(t) = 0.01 + 0.1*returns(t-1)^2 + 0.85*volatility(t-1);
    end
    returns = returns .* sqrt(volatility);
end

T = length(returns);
disp(['Using ' num2str(T) ' observations']);

%% Basic data analysis
figure;
subplot(3,1,1);
plot(returns);
title('Financial Returns');
xlabel('Time');
ylabel('Returns');

subplot(3,1,2);
plot(returns.^2);
title('Squared Returns (Proxy for Volatility)');
xlabel('Time');
ylabel('Squared Returns');

subplot(3,1,3);
sacf(returns.^2, 20);
title('Autocorrelation of Squared Returns');
xlabel('Lag');
ylabel('Autocorrelation');

disp('Data visualization shows characteristic volatility clustering,');
disp('indicating that GARCH models may be appropriate.');

%% Set up GARCH model options
% Basic GARCH model with normal distribution
disp(' ');
disp('Estimating GARCH models...');
options = struct();
options.p = 1;              % GARCH order
options.q = 1;              % ARCH order
options.distribution = 'NORMAL';  % Error distribution
options.display = 'off';    % Display level during optimization

%% Basic GARCH(1,1) model with normal distribution
disp('Fitting standard GARCH(1,1) with normal distribution...');

% Fit the model
garch_normal = garchfit(returns, options);

% Display results
displayModelResults(garch_normal, 'GARCH(1,1) with Normal Distribution');

%% Compare different error distributions
disp(' ');
disp('Comparing different error distributions...');

% Fit GARCH models with different error distributions
options.distribution = 'T';
garch_t = garchfit(returns, options);

options.distribution = 'GED';
garch_ged = garchfit(returns, options);

options.distribution = 'SKEWT';
garch_skewt = garchfit(returns, options);

% Compare models using information criteria
disp('Model Comparison by Error Distribution:');
disp('--------------------------------------------------');
disp('Distribution     Log-Likelihood      AIC        BIC');
disp('--------------------------------------------------');
fprintf('Normal           %12.2f   %10.2f %10.2f\n', ...
    garch_normal.LL, garch_normal.AIC, garch_normal.BIC);
fprintf('Student-t        %12.2f   %10.2f %10.2f\n', ...
    garch_t.LL, garch_t.AIC, garch_t.BIC);
fprintf('GED              %12.2f   %10.2f %10.2f\n', ...
    garch_ged.LL, garch_ged.AIC, garch_ged.BIC);
fprintf('Skewed-t         %12.2f   %10.2f %10.2f\n', ...
    garch_skewt.LL, garch_skewt.AIC, garch_skewt.BIC);
disp('--------------------------------------------------');

% Find best model based on AIC
models = {garch_normal, garch_t, garch_ged, garch_skewt};
modelNames = {'GARCH-Normal', 'GARCH-t', 'GARCH-GED', 'GARCH-Skewed-t'};
aic_values = [garch_normal.AIC, garch_t.AIC, garch_ged.AIC, garch_skewt.AIC];
[~, best_dist_idx] = min(aic_values);
disp(['Best model by AIC: ' modelNames{best_dist_idx}]);

% Use the best distribution for subsequent models
best_dist = models{best_dist_idx}.distribution;
options.distribution = best_dist;
disp(['Using ' best_dist ' distribution for subsequent models']);

%% Compare different GARCH specifications
disp(' ');
disp('Comparing different GARCH specifications...');

% Fit various GARCH-type models
garch_model = garchfit(returns, options);          % Standard GARCH
egarch_model = egarchfit(returns, 1, 1, 1, options); % EGARCH(1,1,1)
tarch_model = tarchfit(returns, options);          % TARCH/GJR
agarch_model = agarchfit(returns, options);        % AGARCH
igarch_model = igarchfit(returns, options);        % IGARCH
nagarch_model = nagarchfit(returns, options);      % NAGARCH

% Store models in cell array for comparison
garch_variants = {garch_model, egarch_model, tarch_model, agarch_model, igarch_model, nagarch_model};
variant_names = {'GARCH', 'EGARCH', 'TARCH/GJR', 'AGARCH', 'IGARCH', 'NAGARCH'};

% Compare models and find the best one
best_idx = compareModels(garch_variants, variant_names);
best_model = garch_variants{best_idx};
disp(['Best model: ' variant_names{best_idx}]);

%% Demonstrate MEX optimization
disp(' ');
disp('Demonstrating MEX optimization performance benefits...');

% Time the model estimation with and without MEX
timing_results = benchmarkMEX(returns, options);

% Display timing results
fprintf('With MEX:    %8.4f seconds\n', timing_results.with_mex);
fprintf('Without MEX: %8.4f seconds\n', timing_results.without_mex);
fprintf('Speedup:     %8.2fx faster with MEX\n', timing_results.speedup);

%% Run diagnostic tests
disp(' ');
disp('Running diagnostic tests on standardized residuals...');

% Extract standardized residuals from the best model
std_residuals = best_model.stdresid;

% Run diagnostics
diagnostics = runDiagnostics(std_residuals, variant_names{best_idx});

%% Generate forecasts
disp(' ');
disp('Generating volatility forecasts...');

% Generate 20-day ahead forecasts
forecast_horizon = 20;
forecast_options = struct('simulate', true, 'numPaths', 5000);
forecasts = garchfor(best_model, forecast_horizon, forecast_options);

% Display forecast results
disp(['Generated ' num2str(forecast_horizon) '-day ahead forecasts with simulation']);
disp('Point forecasts and 95% prediction intervals for volatility:');
disp('------------------------------------------------------------');
disp('  Horizon    Lower 2.5%    Point    Upper 97.5%');
disp('------------------------------------------------------------');
for i = 1:min(10, forecast_horizon)
    fprintf('    %3d      %8.4f    %8.4f    %8.4f\n', i, ...
        forecasts.volatilityQuantiles(i, 1), ...
        forecasts.expectedVolatility(i), ...
        forecasts.volatilityQuantiles(i, 5));
end
if forecast_horizon > 10
    disp('    ...');
end
disp('------------------------------------------------------------');

%% Visualize results
visualizeGARCH(returns, best_model, forecasts);

disp('===================================================================');
disp('GARCH Example Complete');
disp('===================================================================');


%% Helper Functions

function displayModelResults(model, modelName)
% DISPLAYMODELRESULTS Helper function to display parameter estimates and diagnostic statistics for a fitted GARCH model
%
% INPUTS:
%   model - fitted GARCH model structure
%   modelName - string with model name for display
%
% This function extracts and displays:
%   1. Parameter estimates with standard errors
%   2. Model persistence and half-life statistics
%   3. Log-likelihood and information criteria values
%   4. Convergence status information

disp('-------------------------------------------------------------------');
disp(['Model: ' modelName]);
disp('-------------------------------------------------------------------');

% Display parameter estimates
disp('Parameter Estimates:');
disp('-------------------------------------------------------------------');
disp('Parameter      Estimate      Std. Error      t-stat      p-value');
disp('-------------------------------------------------------------------');

% Handle different model types
if isfield(model, 'parameters')
    params = model.parameters;
    paramNames = model.parameternames;
    se = model.stderrors;
    tstat = model.tstat;
    pval = model.pvalues;
    
    % Display main parameters
    for i = 1:length(params)
        fprintf('%-12s  %12.6f  %12.6f  %10.4f  %10.4f\n', ...
            paramNames{i}, params(i), se(i), tstat(i), pval(i));
    end
else
    % Alternative field names for some model types
    if isfield(model, 'omega')
        fprintf('%-12s  %12.6f\n', 'omega', model.omega);
    end
    if isfield(model, 'alpha')
        for i = 1:length(model.alpha)
            fprintf('%-12s  %12.6f\n', ['alpha(' num2str(i) ')'], model.alpha(i));
        end
    end
    if isfield(model, 'gamma') && ~isscalar(model.gamma)
        for i = 1:length(model.gamma)
            fprintf('%-12s  %12.6f\n', ['gamma(' num2str(i) ')'], model.gamma(i));
        end
    end
    if isfield(model, 'gamma') && isscalar(model.gamma)
        fprintf('%-12s  %12.6f\n', 'gamma', model.gamma);
    end
    if isfield(model, 'beta')
        for i = 1:length(model.beta)
            fprintf('%-12s  %12.6f\n', ['beta(' num2str(i) ')'], model.beta(i));
        end
    end
    if isfield(model, 'nu')
        fprintf('%-12s  %12.6f\n', 'nu', model.nu);
    end
    if isfield(model, 'lambda')
        fprintf('%-12s  %12.6f\n', 'lambda', model.lambda);
    end
end

disp('-------------------------------------------------------------------');

% Display model persistence
if isfield(model, 'diagnostics') && isfield(model.diagnostics, 'persistence')
    fprintf('Model persistence: %8.4f\n', model.diagnostics.persistence);
else
    % Calculate persistence for other model types
    persistence = 0;
    if isfield(model, 'alpha') && isfield(model, 'beta')
        persistence = sum(model.alpha) + sum(model.beta);
    end
    fprintf('Model persistence: %8.4f\n', persistence);
end

% Display half-life if available
if isfield(model, 'diagnostics') && isfield(model.diagnostics, 'halflife')
    fprintf('Volatility half-life: %8.4f days\n', model.diagnostics.halflife);
end

% Display log-likelihood and information criteria
if isfield(model, 'LL')
    fprintf('Log-likelihood: %12.4f\n', model.LL);
elseif isfield(model, 'likelihood')
    fprintf('Log-likelihood: %12.4f\n', model.likelihood);
end

if isfield(model, 'AIC') && isfield(model, 'BIC')
    fprintf('AIC: %12.4f   BIC: %12.4f\n', model.AIC, model.BIC);
elseif isfield(model, 'information_criteria')
    fprintf('AIC: %12.4f   BIC: %12.4f\n', ...
        model.information_criteria.AIC, model.information_criteria.BIC);
end

disp('-------------------------------------------------------------------');
end

function bestIdx = compareModels(models, modelNames)
% COMPAREMODELS Helper function to compare multiple GARCH models using information criteria and log-likelihood
%
% INPUTS:
%   models - Cell array of fitted GARCH model structures
%   modelNames - Cell array of model names for display
%
% OUTPUTS:
%   bestIdx - Index of best model according to AIC
%
% This function extracts information criteria and log-likelihood values
% from multiple models, displays a comparison table, and identifies the best
% model according to each criterion.

% Initialize arrays for metrics
numModels = length(models);
aic = zeros(1, numModels);
bic = zeros(1, numModels);
logL = zeros(1, numModels);

% Extract metrics from each model
for i = 1:numModels
    % Handle different field names in different model types
    if isfield(models{i}, 'LL')
        logL(i) = models{i}.LL;
    elseif isfield(models{i}, 'likelihood')
        logL(i) = models{i}.likelihood;
    end
    
    if isfield(models{i}, 'AIC')
        aic(i) = models{i}.AIC;
    elseif isfield(models{i}, 'information_criteria')
        aic(i) = models{i}.information_criteria.AIC;
    end
    
    if isfield(models{i}, 'BIC')
        bic(i) = models{i}.BIC;
    elseif isfield(models{i}, 'information_criteria')
        bic(i) = models{i}.information_criteria.BIC;
    end
end

% Find best model for each criterion
[~, bestAIC] = min(aic);
[~, bestBIC] = min(bic);
[~, bestLL] = max(logL);

% Display comparison table
disp('Model Comparison:');
disp('------------------------------------------------------------------');
disp('Model         Log-Likelihood        AIC             BIC');
disp('------------------------------------------------------------------');
for i = 1:numModels
    % Add markers for best models
    aicMark = ' ';
    bicMark = ' ';
    llMark = ' ';
    
    if i == bestAIC
        aicMark = '*';
    end
    if i == bestBIC
        bicMark = '*';
    end
    if i == bestLL
        llMark = '*';
    end
    
    fprintf('%-12s  %12.2f %s  %12.2f %s  %12.2f %s\n', ...
        modelNames{i}, logL(i), llMark, aic(i), aicMark, bic(i), bicMark);
end
disp('------------------------------------------------------------------');
disp('* indicates best model according to each criterion');

% Return index of best model according to AIC
bestIdx = bestAIC;
end

function diagnostics = runDiagnostics(standardizedResiduals, modelName)
% RUNDIAGNOSTICS Helper function to run and display diagnostic tests on standardized residuals from GARCH model
%
% INPUTS:
%   standardizedResiduals - Vector of standardized residuals from a GARCH model
%   modelName - String with model name for display
%
% OUTPUTS:
%   diagnostics - Structure containing results of all diagnostic tests
%
% This function performs and reports:
%   1. Basic statistics (mean, standard deviation, skewness, kurtosis)
%   2. Ljung-Box test for autocorrelation in standardized residuals
%   3. Ljung-Box test for autocorrelation in squared standardized residuals
%   4. ARCH-LM test for remaining ARCH effects
%   5. Jarque-Bera test for normality

disp(['Diagnostic Tests for ' modelName ' Standardized Residuals:']);
disp('------------------------------------------------------------------');

% Basic statistics
mean_resid = mean(standardizedResiduals);
std_resid = std(standardizedResiduals);
skew_resid = mean((standardizedResiduals - mean_resid).^3) / std_resid^3;
kurt_resid = mean((standardizedResiduals - mean_resid).^4) / std_resid^4;

disp('Basic Statistics:');
fprintf('Mean: %8.4f (should be close to 0)\n', mean_resid);
fprintf('Std Dev: %8.4f (should be close to 1)\n', std_resid);
fprintf('Skewness: %8.4f (should be close to 0 for symmetry)\n', skew_resid);
fprintf('Kurtosis: %8.4f (should be close to 3 for normality)\n', kurt_resid);
disp(' ');

% Ljung-Box test for autocorrelation in standardized residuals
disp('Ljung-Box Test for Autocorrelation in Standardized Residuals:');
lags = [5, 10, 15, 20];
lb_results = ljungbox(standardizedResiduals, lags, 2);  % Adjust dof=2 for GARCH(1,1)

disp('Lag    Statistic    p-value    Significant at 5%?');
disp('--------------------------------------------------');
for i = 1:length(lags)
    significant = lb_results.pvals(i) < 0.05;
    sig_mark = ' ';
    if significant
        sig_mark = '*';
    end
    fprintf('%3d    %9.4f    %7.4f    %s\n', ...
        lags(i), lb_results.stats(i), lb_results.pvals(i), sig_mark);
end
disp('--------------------------------------------------');
disp('* indicates significant autocorrelation (model misspecification)');
disp(' ');

% Ljung-Box test for autocorrelation in squared standardized residuals
disp('Ljung-Box Test for ARCH Effects in Squared Standardized Residuals:');
lb_sq_results = ljungbox(standardizedResiduals.^2, lags, 2);

disp('Lag    Statistic    p-value    Significant at 5%?');
disp('--------------------------------------------------');
for i = 1:length(lags)
    significant = lb_sq_results.pvals(i) < 0.05;
    sig_mark = ' ';
    if significant
        sig_mark = '*';
    end
    fprintf('%3d    %9.4f    %7.4f    %s\n', ...
        lags(i), lb_sq_results.stats(i), lb_sq_results.pvals(i), sig_mark);
end
disp('--------------------------------------------------');
disp('* indicates remaining ARCH effects (model misspecification)');
disp(' ');

% ARCH-LM test
disp('ARCH-LM Test for Remaining ARCH Effects:');
arch_results = arch_test(standardizedResiduals, 10);

disp(['Test statistic: ' num2str(arch_results.statistic)]);
disp(['p-value: ' num2str(arch_results.pval)]);
if arch_results.pval < 0.05
    disp('Significant at 5% level - Model may be misspecified');
else
    disp('Not significant - No evidence of remaining ARCH effects');
end
disp(' ');

% Jarque-Bera test for normality
disp('Jarque-Bera Test for Normality:');
jb_results = jarque_bera(standardizedResiduals);

disp(['Test statistic: ' num2str(jb_results.statistic)]);
disp(['p-value: ' num2str(jb_results.pval)]);
if jb_results.pval < 0.05
    disp('Significant at 5% level - Standardized residuals not normally distributed');
    disp('This is expected if using t, GED, or skewed t distributions');
else
    disp('Not significant - Standardized residuals appear normally distributed');
end
disp('------------------------------------------------------------------');

% Return diagnostic results in a structure
diagnostics = struct(...
    'mean', mean_resid, ...
    'std', std_resid, ...
    'skewness', skew_resid, ...
    'kurtosis', kurt_resid, ...
    'ljung_box', lb_results, ...
    'ljung_box_squared', lb_sq_results, ...
    'arch_test', arch_results, ...
    'jarque_bera', jb_results);
end

function visualizeGARCH(returns, model, forecasts)
% VISUALIZEGARCH Helper function to create comprehensive visualization of GARCH model results
%
% INPUTS:
%   returns - Original returns series 
%   model - Fitted GARCH model structure
%   forecasts - Forecast structure from garchfor function
%
% This function creates a multi-panel figure with:
%   1. Original returns series
%   2. Conditional volatility with absolute returns overlay
%   3. Standardized residuals with significance bounds
%   4. QQ plot of standardized residuals
%   5. ACF of standardized residuals
%   6. ACF of squared standardized residuals
%   7. Volatility forecast with confidence intervals

figure('Position', [100, 100, 1200, 800]);

% 1. Original returns series
subplot(3, 3, 1);
plot(returns);
title('Financial Returns');
xlabel('Time');
ylabel('Returns');
grid on;

% 2. Conditional volatility
subplot(3, 3, 2);
hold on;
plot(sqrt(model.ht));
plot(abs(returns), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5);
title('Conditional Volatility');
xlabel('Time');
ylabel('Volatility');
legend('Model Volatility', 'Absolute Returns', 'Location', 'NorthWest');
grid on;
hold off;

% 3. Standardized residuals
subplot(3, 3, 3);
hold on;
plot(model.stdresid);
plot([1, length(returns)], [1.96, 1.96], 'r--');
plot([1, length(returns)], [-1.96, -1.96], 'r--');
title('Standardized Residuals');
xlabel('Time');
ylabel('Std. Residuals');
grid on;
hold off;

% 4. QQ plot of standardized residuals
subplot(3, 3, 4);
qqplot(model.stdresid);
title('QQ Plot of Standardized Residuals');
grid on;

% 5. ACF of standardized residuals
subplot(3, 3, 5);
[acf, ~, acf_bounds] = sacf(model.stdresid, 20);
hold on;
bar(1:20, acf);
plot([1, 20], [acf_bounds(1, 1), acf_bounds(1, 1)], 'r--');
plot([1, 20], [acf_bounds(1, 2), acf_bounds(1, 2)], 'r--');
title('ACF of Standardized Residuals');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;
hold off;

% 6. ACF of squared standardized residuals
subplot(3, 3, 6);
[acf_sq, ~, acf_sq_bounds] = sacf(model.stdresid.^2, 20);
hold on;
bar(1:20, acf_sq);
plot([1, 20], [acf_sq_bounds(1, 1), acf_sq_bounds(1, 1)], 'r--');
plot([1, 20], [acf_sq_bounds(1, 2), acf_sq_bounds(1, 2)], 'r--');
title('ACF of Squared Standardized Residuals');
xlabel('Lag');
ylabel('Autocorrelation');
grid on;
hold off;

% 7. Volatility forecast
subplot(3, 3, 7:9);
hold on;
T = length(returns);
time_axis = (1:length(forecasts.expectedVolatility)) + T;

% Plot forecast
plot(time_axis, forecasts.expectedVolatility, 'b-', 'LineWidth', 2);
plot(time_axis, forecasts.volatilityQuantiles(:, 1), 'r--'); % Lower bound (2.5%)
plot(time_axis, forecasts.volatilityQuantiles(:, 5), 'r--'); % Upper bound (97.5%)

% Plot historical volatility for context
historical_vol = sqrt(model.ht(end-30:end));
hist_time = (T-30):T;
plot(hist_time, historical_vol, 'k-');

% Fill prediction interval
x_fill = [time_axis, fliplr(time_axis)];
y_fill = [forecasts.volatilityQuantiles(:, 1)', fliplr(forecasts.volatilityQuantiles(:, 5)')];
fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

title('Volatility Forecast');
xlabel('Time');
ylabel('Volatility');
legend('Point Forecast', '95% Confidence Interval', '', 'Historical Volatility', 'Location', 'NorthWest');
grid on;
hold off;

end

function timingResults = benchmarkMEX(returns, options)
% BENCHMARKMEX Helper function to demonstrate performance benefits of MEX optimization
%
% INPUTS:
%   returns - Time series data for model estimation
%   options - Options structure for model estimation
%
% OUTPUTS:
%   timingResults - Structure with timing results:
%       with_mex - Average time with MEX optimization
%       without_mex - Average time without MEX optimization
%       speedup - Ratio of without_mex to with_mex
%
% This function runs multiple trials of GARCH estimation with and without
% MEX optimization to demonstrate the performance benefit.

% Parameters for benchmark
numRuns = 3;  % Number of runs to average

% Test with MEX enabled
options.useMEX = true;
tic;
for i = 1:numRuns
    garchfit(returns, options);
end
time_with_mex = toc / numRuns;

% Test with MEX disabled
options.useMEX = false;
tic;
for i = 1:numRuns
    garchfit(returns, options);
end
time_without_mex = toc / numRuns;

% Calculate speedup
speedup = time_without_mex / time_with_mex;

% Return results
timingResults = struct(...
    'with_mex', time_with_mex, ...
    'without_mex', time_without_mex, ...
    'speedup', speedup);
end