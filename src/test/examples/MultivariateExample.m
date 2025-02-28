%% MultivariateExample.m - Demonstration of Multivariate Time Series and Volatility Models
% This example script demonstrates the usage of multivariate time series and volatility 
% models from the MFE Toolbox, including Vector Autoregression (VAR), DCC-MVGARCH,
% and BEKK-MVGARCH models with various diagnostics, visualizations, and forecasting
% capabilities.
%
% The example showcases how to apply, interpret, and compare different multivariate 
% modeling approaches for financial time series.
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Initialization
clear;
clc;
close all;

% Global settings
PLOT_FIGURES = true;  % Set to false to suppress figures
FORECAST_HORIZON = 10;  % Number of periods to forecast

%% Load Example Data
% Load financial returns data for testing
load('../data/financial_returns.mat', 'returns', 'dates', 'assets');

% Basic data validation
returns = datacheck(returns, 'returns');
[T, K] = size(returns);

fprintf('Loaded financial returns data: %d observations for %d assets\n', T, K);
fprintf('Time period: %s to %s\n', datestr(dates(1)), datestr(dates(end)));
fprintf('Assets: %s\n', strjoin(assets, ', '));

% Display basic statistics
fprintf('\nBasic statistics for the return series:\n');
fprintf('Mean returns:\n');
disp(mean(returns));
fprintf('Return standard deviations:\n');
disp(std(returns));

% Plot return series for a few assets
if PLOT_FIGURES
    figure;
    subplot(2,1,1);
    plot(dates, returns(:,1:min(3,K)));
    title('Return Series for Selected Assets');
    legend(assets(1:min(3,K)), 'Location', 'Best');
    grid on;
    
    subplot(2,1,2);
    plot(dates, sqrt(252) * movstd(returns(:,1:min(3,K)), 21));
    title('Rolling 21-day Volatility (Annualized)');
    legend(assets(1:min(3,K)), 'Location', 'Best');
    grid on;
end

%% Vector Autoregression (VAR) Modeling
fprintf('\n========== Vector Autoregression (VAR) Modeling ==========\n');

% Select a subset of assets for VAR modeling (to keep the example manageable)
selected_indices = 1:min(4, K);
selected_data = returns(:, selected_indices);
selected_assets = assets(selected_indices);

% Estimate a VAR(2) model
var_results = example_var_model(selected_data, selected_assets);

% Display VAR results
fprintf('VAR model estimation completed. Log-likelihood: %.4f\n', var_results.logL);
fprintf('AIC: %.4f, BIC: %.4f\n', var_results.aic, var_results.sbic);

% Display a summary of the VAR coefficient matrices
fprintf('\nVAR coefficient matrices summary:\n');
disp(var_results.coefficients);

%% DCC-MVGARCH Modeling
fprintf('\n========== DCC Multivariate GARCH Modeling ==========\n');

% Estimate a DCC-GARCH model with t-distributed errors
dcc_results = example_dcc_mvgarch(selected_data, selected_assets);

% Display DCC results
fprintf('DCC-MVGARCH model estimation completed. Log-likelihood: %.4f\n', dcc_results.likelihood);
fprintf('AIC: %.4f, BIC: %.4f\n', dcc_results.aic, dcc_results.bic);

% Display correlation dynamics
fprintf('\nTime-varying correlation ranges:\n');
for i = 1:length(selected_indices)-1
    for j = i+1:length(selected_indices)
        corr_series = squeeze(dcc_results.corr(:, i, j));
        fprintf('Correlation %s-%s: Min=%.4f, Max=%.4f, Mean=%.4f\n', ...
            selected_assets{i}, selected_assets{j}, ...
            min(corr_series), max(corr_series), mean(corr_series));
    end
end

%% BEKK-MVGARCH Modeling
fprintf('\n========== BEKK Multivariate GARCH Modeling ==========\n');

% Estimate a diagonal BEKK-GARCH model
bekk_results = example_bekk_mvgarch(selected_data, selected_assets);

% Display BEKK results
fprintf('BEKK-MVGARCH model estimation completed. Log-likelihood: %.4f\n', bekk_results.likelihood);
fprintf('AIC: %.4f, BIC: %.4f\n', bekk_results.aic, bekk_results.bic);

%% Compare Multivariate Models
fprintf('\n========== Multivariate Models Comparison ==========\n');

% Compare the different multivariate models
comparison = compare_multivariate_models(var_results, dcc_results, bekk_results, selected_assets);

% Display comparison summary
fprintf('Model comparison based on fit statistics:\n');
disp(comparison.fit_stats);

fprintf('\nForecast comparison (if available):\n');
if isfield(comparison, 'forecast_stats')
    disp(comparison.forecast_stats);
else
    fprintf('Forecast comparison not available.\n');
end

function var_model_results = example_var_model(data, asset_names)
% EXAMPLE_VAR_MODEL Demonstrates Vector Autoregression (VAR) modeling for multivariate time series
%
% INPUTS:
%   data        - T x K matrix of return data
%   asset_names - Cell array of asset names
%
% OUTPUTS:
%   var_model_results - Structure with VAR model results and diagnostics

% Validate input data
data = datacheck(data, 'data');
[T, K] = size(data);

fprintf('Estimating VAR model for %d assets with %d observations...\n', K, T);

% Set up VAR model options
options = struct();
options.constant = true;  % Include constant term
options.trend = false;    % No time trend
options.criterion = 'aic'; % Use AIC for lag selection

% Specify VAR order (p=2)
p = 2;

% Estimate VAR model
model = var_model(data, p, options);

% Display model summary
fprintf('VAR(%d) model estimated with %d parameters\n', model.p, model.nparams);
fprintf('Residual covariance matrix:\n');
disp(model.sigma);

% Generate forecasts
forecasts = var_forecast(model, FORECAST_HORIZON);
fprintf('Generated %d-step ahead forecasts\n', FORECAST_HORIZON);

% Compute impulse response functions
irf = var_irf(model, 20);
fprintf('Computed impulse response functions for 20 periods\n');

% Compute forecast error variance decomposition
fevd = var_fevd(model, 20);
fprintf('Computed forecast error variance decomposition for 20 periods\n');

% Plot results if requested
if PLOT_FIGURES
    % Plot time series and forecasts
    figure;
    for i = 1:K
        subplot(K, 1, i);
        plot([data(:,i); forecasts(:,i)]);
        hold on;
        plot([T, T], ylim, 'r--');
        title(['VAR Forecast for ' asset_names{i}]);
        grid on;
        if i == K
            xlabel('Time');
        end
        ylabel('Returns');
    end
    
    % Plot impulse responses
    figure;
    for i = 1:K
        for j = 1:K
            subplot(K, K, (i-1)*K+j);
            plot(0:20, squeeze(irf(:,i,j)));
            title(['Response of ' asset_names{i} ' to ' asset_names{j}]);
            grid on;
            if i == K
                xlabel('Horizon');
            end
            if j == 1
                ylabel('Response');
            end
        end
    end
    
    % Plot variance decomposition
    figure;
    for i = 1:K
        subplot(K, 1, i);
        area(0:20, squeeze(fevd(:,i,:)));
        title(['Variance Decomposition for ' asset_names{i}]);
        grid on;
        if i == K
            xlabel('Horizon');
        end
        ylabel('Proportion');
        legend(asset_names, 'Location', 'Best');
    end
end

% Perform Ljung-Box test for residual autocorrelation
ljung_box_results = cell(K, 1);
for i = 1:K
    ljung_box_results{i} = ljungbox(model.residuals(:,i), 10, p);
    fprintf('Ljung-Box test for %s: p-value = %.4f\n', ...
        asset_names{i}, ljung_box_results{i}.pvals(end));
end

% Return results
var_model_results = model;
var_model_results.forecasts = forecasts;
var_model_results.irf = irf;
var_model_results.fevd = fevd;
var_model_results.ljung_box = ljung_box_results;
var_model_results.asset_names = asset_names;
end

function dcc_results = example_dcc_mvgarch(data, asset_names)
% EXAMPLE_DCC_MVGARCH Demonstrates Dynamic Conditional Correlation (DCC) multivariate GARCH modeling
%
% INPUTS:
%   data        - T x K matrix of return data
%   asset_names - Cell array of asset names
%
% OUTPUTS:
%   dcc_results - Structure with DCC-MVGARCH model results and diagnostics

% Validate input data
data = datacheck(data, 'data');
[T, K] = size(data);

fprintf('Estimating DCC-MVGARCH model for %d assets with %d observations...\n', K, T);

% Set up DCC model options
options = struct();
options.model = 'GARCH';        % Use standard GARCH for univariate models
options.distribution = 'T';      % Use t-distribution for errors
options.p = 1;                   % GARCH order
options.q = 1;                   % ARCH order
options.forecast = FORECAST_HORIZON; % Generate forecasts

% Estimate DCC-MVGARCH model
model = dcc_mvgarch(data, options);

% Display model summary
fprintf('DCC-MVGARCH model estimated\n');
fprintf('Estimated t-distribution degrees of freedom: %.4f\n', model.parameters.univariate{1}(end));
fprintf('DCC parameters (a,b): %.4f, %.4f\n', model.parameters.dcc(1), model.parameters.dcc(2));

% Extract time-varying correlations and volatilities
correlations = model.corr;
volatilities = sqrt(model.h);

fprintf('Conditional volatility ranges:\n');
for i = 1:K
    vol_series = volatilities(:, i);
    fprintf('%s: Min=%.4f, Max=%.4f, Mean=%.4f, Annualized Mean=%.4f\n', ...
        asset_names{i}, min(vol_series), max(vol_series), ...
        mean(vol_series), mean(vol_series)*sqrt(252));
end

% Plot results if requested
if PLOT_FIGURES
    % Plot time-varying correlations
    figure;
    subplot(2,1,1);
    hold on;
    colors = lines(K*(K-1)/2);
    counter = 1;
    legend_str = cell(K*(K-1)/2, 1);
    for i = 1:K-1
        for j = i+1:K
            corr_series = squeeze(correlations(:, i, j));
            plot(1:T, corr_series, 'Color', colors(counter,:));
            legend_str{counter} = [asset_names{i} '-' asset_names{j}];
            counter = counter + 1;
        end
    end
    title('DCC Time-Varying Correlations');
    legend(legend_str, 'Location', 'Best');
    grid on;
    
    % Plot conditional volatilities
    subplot(2,1,2);
    plot(1:T, volatilities);
    title('Conditional Volatilities');
    legend(asset_names, 'Location', 'Best');
    grid on;
    xlabel('Time');
    
    % Plot forecast of volatilities
    if isfield(model, 'forecast')
        figure;
        for i = 1:K
            subplot(K, 1, i);
            plot([volatilities(end-20:end, i); model.forecast.h(:, i)]);
            hold on;
            plot([21, 21], ylim, 'r--');
            title(['Volatility Forecast for ' asset_names{i}]);
            grid on;
            if i == K
                xlabel('Time');
            end
            ylabel('Volatility');
        end
    end
end

% Return results
dcc_results = model;
dcc_results.asset_names = asset_names;
end

function bekk_results = example_bekk_mvgarch(data, asset_names)
% EXAMPLE_BEKK_MVGARCH Demonstrates BEKK multivariate GARCH modeling for volatility spillover analysis
%
% INPUTS:
%   data        - T x K matrix of return data
%   asset_names - Cell array of asset names
%
% OUTPUTS:
%   bekk_results - Structure with BEKK-MVGARCH model results and diagnostics

% Validate input data
data = datacheck(data, 'data');
[T, K] = size(data);

fprintf('Estimating BEKK-MVGARCH model for %d assets with %d observations...\n', K, T);

% Set up BEKK model options
options = struct();
options.p = 1;                   % ARCH order
options.q = 1;                   % GARCH order
options.type = 'diagonal';       % Use diagonal BEKK for simplicity
options.distribution = 'normal'; % Use normal distribution for errors
options.forecast = FORECAST_HORIZON; % Generate forecasts

% Estimate BEKK-MVGARCH model
model = bekk_mvgarch(data, options);

% Display model summary
fprintf('Diagonal BEKK-MVGARCH model estimated\n');

% Extract parameter matrices
fprintf('BEKK parameter matrices:\n');
fprintf('C (constant):\n');
disp(model.parameters.C);

fprintf('A (ARCH):\n');
disp(model.parameters.A(:,:,1));

fprintf('B (GARCH):\n');
disp(model.parameters.B(:,:,1));

% Extract conditional covariances
H = model.H;

% Calculate conditional correlations
conditional_correlations = zeros(T, K, K);
for t = 1:T
    H_t = squeeze(H(:,:,t));
    D_t = diag(sqrt(diag(H_t)));
    D_t_inv = diag(1./diag(D_t));
    conditional_correlations(t,:,:) = D_t_inv * H_t * D_t_inv;
end

% Extract volatilities
volatilities = zeros(T, K);
for k = 1:K
    volatilities(:,k) = sqrt(H(k,k,:));
end

% Plot results if requested
if PLOT_FIGURES
    % Plot conditional variances and covariances
    figure;
    subplot(2,1,1);
    plot(1:T, volatilities.^2);
    title('BEKK Conditional Variances');
    legend(asset_names, 'Location', 'Best');
    grid on;
    
    subplot(2,1,2);
    hold on;
    colors = lines(K*(K-1)/2);
    counter = 1;
    legend_str = cell(K*(K-1)/2, 1);
    for i = 1:K-1
        for j = i+1:K
            cov_series = squeeze(H(i,j,:));
            plot(1:T, cov_series, 'Color', colors(counter,:));
            legend_str{counter} = [asset_names{i} '-' asset_names{j}];
            counter = counter + 1;
        end
    end
    title('BEKK Conditional Covariances');
    legend(legend_str, 'Location', 'Best');
    grid on;
    xlabel('Time');
    
    % Plot volatility spillover analysis
    figure;
    for i = 1:K
        for j = 1:K
            subplot(K, K, (i-1)*K+j);
            if i == j
                % Diagonal: ARCH and GARCH effects
                bar([model.parameters.A(i,i,1)^2, model.parameters.B(i,i,1)^2]);
                title([asset_names{i} ' Persistence']);
                set(gca, 'XTickLabel', {'ARCH', 'GARCH'});
            else
                % Off-diagonal: spillover effects in diagonal BEKK are products of parameters
                spillover_arch = model.parameters.A(i,i,1) * model.parameters.A(j,j,1);
                spillover_garch = model.parameters.B(i,i,1) * model.parameters.B(j,j,1);
                bar([spillover_arch, spillover_garch]);
                title(['Spillover: ' asset_names{i} ' \leftrightarrow ' asset_names{j}]);
                set(gca, 'XTickLabel', {'ARCH', 'GARCH'});
            end
            grid on;
        end
    end
    
    % Plot forecasts
    if isfield(model, 'forecast')
        figure;
        for i = 1:K
            subplot(K, 1, i);
            volatility_forecast = squeeze(sqrt(model.forecast.covariance(:,i,i)));
            plot([volatilities(end-20:end, i); volatility_forecast]);
            hold on;
            plot([21, 21], ylim, 'r--');
            title(['BEKK Volatility Forecast for ' asset_names{i}]);
            grid on;
            if i == K
                xlabel('Time');
            end
            ylabel('Volatility');
        end
    end
end

% Return results
bekk_results = model;
bekk_results.conditional_correlations = conditional_correlations;
bekk_results.volatilities = volatilities;
bekk_results.asset_names = asset_names;
end

function comparison = compare_multivariate_models(var_results, dcc_results, bekk_results, asset_names)
% COMPARE_MULTIVARIATE_MODELS Compares different multivariate volatility models based on fit statistics
% and forecasting performance
%
% INPUTS:
%   var_results  - Structure with VAR model results
%   dcc_results  - Structure with DCC-MVGARCH model results
%   bekk_results - Structure with BEKK-MVGARCH model results
%   asset_names  - Cell array of asset names
%
% OUTPUTS:
%   comparison - Structure with comparison results and performance metrics

% Initialize comparison structure
comparison = struct();
K = length(asset_names);

% Compare fit statistics
fit_stats = zeros(3, 4);
fit_stats(1, 1) = var_results.logL;
fit_stats(1, 2) = var_results.aic;
fit_stats(1, 3) = var_results.sbic;
fit_stats(1, 4) = var_results.nparams;

fit_stats(2, 1) = dcc_results.likelihood;
fit_stats(2, 2) = dcc_results.aic;
fit_stats(2, 3) = dcc_results.bic;
fit_stats(2, 4) = NaN; % DCC doesn't explicitly provide number of parameters

fit_stats(3, 1) = bekk_results.likelihood;
fit_stats(3, 2) = bekk_results.aic;
fit_stats(3, 3) = bekk_results.bic;
fit_stats(3, 4) = bekk_results.numParams;

% Create table for model fit statistics
fit_stats_table = array2table(fit_stats, ...
    'RowNames', {'VAR', 'DCC-MVGARCH', 'BEKK-MVGARCH'}, ...
    'VariableNames', {'LogLikelihood', 'AIC', 'BIC', 'NumParams'});

comparison.fit_stats = fit_stats_table;

% Compare parameter estimates
fprintf('Comparison of model features:\n');
fprintf('VAR captures linear dependencies between series\n');
fprintf('DCC allows for time-varying correlations with separate volatility processes\n');
fprintf('BEKK allows for volatility spillovers across series\n');

% Compare time-varying correlations from different models (if available)
if PLOT_FIGURES
    % Extract time series length
    T = size(dcc_results.corr, 1);
    
    % Extract DCC correlations
    dcc_corr = zeros(T, K*(K-1)/2);
    
    % Extract BEKK correlations (if available)
    bekk_corr = zeros(T, K*(K-1)/2);
    
    % Fill correlation matrices
    counter = 1;
    legend_str = cell(K*(K-1)/2, 1);
    for i = 1:K-1
        for j = i+1:K
            dcc_corr(:, counter) = squeeze(dcc_results.corr(:, i, j));
            
            if isfield(bekk_results, 'conditional_correlations')
                bekk_corr(:, counter) = squeeze(bekk_results.conditional_correlations(:, i, j));
            end
            
            legend_str{counter} = [asset_names{i} '-' asset_names{j}];
            counter = counter + 1;
        end
    end
    
    % Plot comparison of correlations
    figure;
    subplot(2,1,1);
    plot(1:T, dcc_corr);
    title('DCC Time-Varying Correlations');
    legend(legend_str, 'Location', 'Best');
    grid on;
    
    if isfield(bekk_results, 'conditional_correlations')
        subplot(2,1,2);
        plot(1:T, bekk_corr);
        title('BEKK Implied Correlations');
        legend(legend_str, 'Location', 'Best');
        grid on;
    end
    xlabel('Time');
    
    % Compare correlation dynamics: range and variability
    figure;
    boxplot([dcc_corr, bekk_corr], 'Labels', [strcat('DCC:', legend_str); strcat('BEKK:', legend_str)]);
    title('Comparison of Correlation Distributions');
    ylabel('Correlation');
    grid on;
    
    % Compare forecast performance (if forecasts are available)
    if isfield(dcc_results, 'forecast') && isfield(bekk_results, 'forecast')
        figure;
        for i = 1:K
            subplot(K,1,i);
            hold on;
            % DCC volatility forecast
            dcc_vol_forecast = sqrt(dcc_results.forecast.h(:, i));
            plot(1:FORECAST_HORIZON, dcc_vol_forecast, 'b-', 'LineWidth', 1.5);
            
            % BEKK volatility forecast
            bekk_vol_forecast = squeeze(sqrt(bekk_results.forecast.covariance(:,i,i)));
            plot(1:FORECAST_HORIZON, bekk_vol_forecast, 'r--', 'LineWidth', 1.5);
            
            title(['Volatility Forecast for ' asset_names{i}]);
            legend('DCC-MVGARCH', 'BEKK-MVGARCH');
            grid on;
            if i == K
                xlabel('Forecast Horizon');
            end
            ylabel('Volatility');
        end
    end
end

% Return comparison results
end