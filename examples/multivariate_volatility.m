%% Multivariate Volatility Modeling with MFE Toolbox
% This script demonstrates the multivariate volatility modeling capabilities
% of the MFE Toolbox (MATLAB Financial Econometrics Toolbox) with various models:
% * Constant Conditional Correlation (CCC-MVGARCH)
% * Dynamic Conditional Correlation (DCC-MVGARCH)
% * BEKK (Baba-Engle-Kraft-Kroner) MVGARCH
% * Generalized Orthogonal GARCH (GO-GARCH)
%
% The example covers model estimation, comparison, forecasting, and
% visualization for financial risk analysis and portfolio management.
%
% Version: 4.0 (2009/10/28)

%% Initialize MFE Toolbox
% Add necessary paths for all toolbox components
addToPath();

%% Load and Prepare Data
% Load example financial return data
% Note: This assumes you have a file named 'example_financial_data.mat'
% containing multivariate return series. Modify the filename as needed.
try
    load('example_financial_data.mat');
catch
    error(['Cannot find example_financial_data.mat. Please create a .mat file ',...
           'with multivariate financial returns or adjust the filename.']);
end

% Extract returns and asset names (adjust based on your data structure)
% If your data is structured differently, modify this section accordingly
if exist('returns', 'var')
    % Data already has a 'returns' variable
else
    % Try to find appropriate variables in the loaded workspace
    if exist('data', 'var')
        returns = data;
    elseif exist('r', 'var')
        returns = r;
    else
        error(['Cannot identify return data in the loaded file. ',...
               'Please ensure your .mat file contains return data.']);
    end
end

% Get dimensions of the return data
[T, K] = size(returns);

% Extract or create asset names
if exist('tickers', 'var')
    asset_names = tickers;
elseif exist('symbols', 'var')
    asset_names = symbols;
elseif exist('names', 'var')
    asset_names = names;
else
    % Create generic asset names if none are provided
    asset_names = cell(K, 1);
    for i = 1:K
        asset_names{i} = sprintf('Asset %d', i);
    end
end

%% Data Exploration and Descriptive Statistics
disp('=================================================================');
disp('Multivariate Volatility Modeling with MFE Toolbox');
disp('=================================================================');
disp(' ');
disp(['Number of assets: ', num2str(K)]);
disp(['Number of observations: ', num2str(T)]);
disp(' ');

disp('Descriptive Statistics for Return Series:');
disp('----------------------------------------');
disp('                Mean      Std Dev    Skewness   Kurtosis   Min       Max');
for i = 1:K
    stats = [mean(returns(:,i)), std(returns(:,i)), ...
             skewness(returns(:,i)), kurtosis(returns(:,i)), ...
             min(returns(:,i)), max(returns(:,i))];
    fprintf('%10s: %8.4f   %8.4f   %8.4f   %8.4f   %8.4f   %8.4f\n', ...
            asset_names{i}, stats(1), stats(2), stats(3), stats(4), stats(5), stats(6));
end

% Display unconditional correlation matrix
disp(' ');
disp('Unconditional Correlation Matrix:');
disp('--------------------------------');
unconditional_corr = corr(returns);
disp(unconditional_corr);

% Plot the return series
figure;
for i = 1:min(K, 6)  % Limit to 6 assets for clarity in plots
    subplot(min(K, 6), 1, i);
    plot(returns(:,i));
    title(asset_names{i}, 'Interpreter', 'none');
    ylabel('Returns');
    grid on;
end
xlabel('Time');
sgtitle('Return Series');

%% Set Common Estimation Options
% Define common options for all multivariate GARCH models
common_options = struct();
common_options.distribution = 'T';  % Student's t errors for fat tails
common_options.forecast = 10;       % Generate 10-step ahead forecasts

% Note on distribution options:
% - 'NORMAL': Multivariate normal distribution (default)
% - 'T': Multivariate Student's t-distribution (better for fat tails)
% - 'GED': Generalized Error Distribution 
% - 'SKEWT': Hansen's Skewed t-distribution (for asymmetric tails)

%% 1. Estimate CCC-MVGARCH Model
disp(' ');
disp('=================================================================');
disp('Estimating CCC-MVGARCH Model...');
disp('=================================================================');
disp(['The CCC-MVGARCH model (Bollerslev, 1990) uses a constant correlation']);
disp(['matrix and univariate GARCH models for each asset''s volatility.']);

% Set specific options for CCC-MVGARCH
ccc_options = common_options;
ccc_options.model = 'GARCH';  % Type of univariate GARCH process
ccc_options.p = 1;            % GARCH order
ccc_options.q = 1;            % ARCH order

try
    % Estimate model
    ccc_model = ccc_mvgarch(returns, ccc_options);
    
    % Display estimation results
    disp('CCC-MVGARCH Estimation Results:');
    disp(['Log-likelihood: ', num2str(ccc_model.likelihood, '%.4f')]);
    disp(['AIC: ', num2str(ccc_model.aic, '%.4f')]);
    disp(['BIC: ', num2str(ccc_model.bic, '%.4f')]);
    
    % Display constant correlation matrix
    disp('Constant Correlation Matrix:');
    disp(ccc_model.correlations);
catch ME
    warning('Failed to estimate CCC-MVGARCH model: %s', ME.message);
    % Create a placeholder to allow script to continue
    ccc_model = struct('likelihood', NaN, 'aic', NaN, 'bic', NaN, ...
                      'correlations', eye(K), 'h', zeros(T, K), ...
                      'forecast', struct('cov', zeros(10, K, K), 'h', zeros(10, K)));
end

%% 2. Estimate DCC-MVGARCH Model
disp(' ');
disp('=================================================================');
disp('Estimating DCC-MVGARCH Model...');
disp('=================================================================');
disp(['The DCC-MVGARCH model (Engle, 2002) allows correlations to vary over time,']);
disp(['providing more flexibility than the constant correlation model.']);

% Set specific options for DCC-MVGARCH
dcc_options = common_options;
dcc_options.model = 'GARCH';  % Type of univariate GARCH process
dcc_options.p = 1;            % GARCH order
dcc_options.q = 1;            % ARCH order
dcc_options.dccP = 1;         % DCC parameter order
dcc_options.dccQ = 1;         % DCC memory parameter order

try
    % Estimate model
    dcc_model = dcc_mvgarch(returns, dcc_options);
    
    % Display estimation results
    disp('DCC-MVGARCH Estimation Results:');
    disp(['Log-likelihood: ', num2str(dcc_model.likelihood, '%.4f')]);
    disp(['AIC: ', num2str(dcc_model.stats.aic, '%.4f')]);
    disp(['BIC: ', num2str(dcc_model.stats.bic, '%.4f')]);
    
    % Display DCC parameters
    disp('DCC Parameters (a, b):');
    disp(dcc_model.parameters.dcc);
    disp(['Sum of DCC parameters: ', num2str(sum(dcc_model.parameters.dcc), '%.4f')]);
    disp(['A value close to 1 indicates high persistence in correlations.']);
catch ME
    warning('Failed to estimate DCC-MVGARCH model: %s', ME.message);
    % Create a placeholder to allow script to continue
    dcc_model = struct('likelihood', NaN, 'stats', struct('aic', NaN, 'bic', NaN), ...
                      'parameters', struct('dcc', [0.05; 0.85]), ...
                      'corr', ones(T, K, K), 'h', zeros(T, K), ...
                      'forecast', struct('cov', zeros(10, K, K), 'h', zeros(10, K)));
end

%% 3. Estimate BEKK-MVGARCH Model
disp(' ');
disp('=================================================================');
disp('Estimating BEKK-MVGARCH Model...');
disp('=================================================================');
disp(['The BEKK-MVGARCH model (Engle & Kroner, 1995) directly models the']);
disp(['covariance matrix, ensuring positive definiteness by construction.']);

% Set specific options for BEKK-MVGARCH
bekk_options = common_options;
bekk_options.type = 'diagonal';  % 'diagonal' is more parsimonious than 'full'
bekk_options.p = 1;              % GARCH order
bekk_options.q = 1;              % ARCH order

try
    % Estimate model
    bekk_model = bekk_mvgarch(returns, bekk_options);
    
    % Display estimation results
    disp('BEKK-MVGARCH Estimation Results:');
    disp(['Log-likelihood: ', num2str(bekk_model.likelihood, '%.4f')]);
    disp(['AIC: ', num2str(bekk_model.aic, '%.4f')]);
    disp(['BIC: ', num2str(bekk_model.bic, '%.4f')]);
catch ME
    warning('Failed to estimate BEKK-MVGARCH model: %s', ME.message);
    % Create a placeholder to allow script to continue
    bekk_model = struct('likelihood', NaN, 'aic', NaN, 'bic', NaN, ...
                       'H', ones(K, K, T), ...
                       'forecast', struct('covariance', zeros(10, K, K)));
end

%% 4. Estimate GO-GARCH Model
disp(' ');
disp('=================================================================');
disp('Estimating GO-GARCH Model...');
disp('=================================================================');
disp(['The GO-GARCH model (van der Weide, 2002) uses orthogonal transformations']);
disp(['to convert correlated returns into independent factors, which are then']);
disp(['modeled with univariate GARCH processes.']);

% Set specific options for GO-GARCH
go_options = common_options;
go_options.garchType = 'GARCH';  % Type of univariate GARCH process
go_options.p = 1;                % GARCH order
go_options.q = 1;                % ARCH order
go_options.method = 'pca';       % Orthogonalization method ('pca' or 'ica')

try
    % Estimate model
    go_model = gogarch(returns, go_options);
    
    % Display estimation results
    disp('GO-GARCH Estimation Results:');
    disp(['Log-likelihood: ', num2str(go_model.logLikelihood, '%.4f')]);
    disp(['AIC: ', num2str(go_model.aic, '%.4f')]);
    disp(['BIC: ', num2str(go_model.bic, '%.4f')]);
catch ME
    warning('Failed to estimate GO-GARCH model: %s', ME.message);
    % Create a placeholder to allow script to continue
    go_model = struct('logLikelihood', NaN, 'aic', NaN, 'bic', NaN, ...
                     'covariances', ones(T, K, K), ...
                     'forecast', struct('covarianceForecasts', zeros(10, K, K)));
end

%% Model Comparison
disp(' ');
disp('=================================================================');
disp('Model Comparison:');
disp('=================================================================');
disp('Model comparison helps select the most appropriate model based on');
disp('goodness-of-fit measures like log-likelihood, AIC, and BIC.');
disp('Lower AIC and BIC values indicate better model fit.');
disp(' ');
disp('Model          Log-Likelihood     AIC           BIC');
fprintf('CCC-MVGARCH    %12.2f   %12.2f   %12.2f\n', ...
        ccc_model.likelihood, ccc_model.aic, ccc_model.bic);
fprintf('DCC-MVGARCH    %12.2f   %12.2f   %12.2f\n', ...
        dcc_model.likelihood, dcc_model.stats.aic, dcc_model.stats.bic);
fprintf('BEKK-MVGARCH   %12.2f   %12.2f   %12.2f\n', ...
        bekk_model.likelihood, bekk_model.aic, bekk_model.bic);
fprintf('GO-GARCH       %12.2f   %12.2f   %12.2f\n', ...
        go_model.logLikelihood, go_model.aic, go_model.bic);

%% Visualize Conditional Correlations
% Compare conditional correlations across different models
if K > 1
    % Select a pair of assets to visualize (first two by default)
    asset_pair = [1, 2];  % Change if needed
    pair_name = [asset_names{asset_pair(1)}, '-', asset_names{asset_pair(2)}];
    
    % Extract correlations
    ccc_corr = zeros(T, 1);
    dcc_corr = zeros(T, 1);
    bekk_corr = zeros(T, 1);
    go_corr = zeros(T, 1);
    
    % Extract time-varying correlations
    for t = 1:T
        % CCC correlation (constant across time)
        ccc_corr(t) = ccc_model.correlations(asset_pair(1), asset_pair(2));
        
        % DCC correlation
        dcc_corr(t) = dcc_model.corr(t, asset_pair(1), asset_pair(2));
        
        % BEKK correlation
        H_t = squeeze(bekk_model.H(:,:,t));
        % Safely extract correlation
        try
            D_t = diag(sqrt(diag(H_t)));
            % Check for singularity
            if any(diag(D_t) < 1e-8)
                bekk_corr(t) = bekk_corr(max(1, t-1));  % Use previous value
            else
                R_t = inv(D_t) * H_t * inv(D_t);
                bekk_corr(t) = R_t(asset_pair(1), asset_pair(2));
                % Ensure correlation is in valid range
                bekk_corr(t) = max(-1, min(1, bekk_corr(t)));
            end
        catch
            bekk_corr(t) = bekk_corr(max(1, t-1));  % Use previous value if error
        end
        
        % GO-GARCH correlation
        H_t = squeeze(go_model.covariances(t, :, :));
        % Safely extract correlation
        try
            D_t = diag(sqrt(diag(H_t)));
            % Check for singularity
            if any(diag(D_t) < 1e-8)
                go_corr(t) = go_corr(max(1, t-1));  % Use previous value
            else
                R_t = inv(D_t) * H_t * inv(D_t);
                go_corr(t) = R_t(asset_pair(1), asset_pair(2));
                % Ensure correlation is in valid range
                go_corr(t) = max(-1, min(1, go_corr(t)));
            end
        catch
            go_corr(t) = go_corr(max(1, t-1));  % Use previous value if error
        end
    end
    
    % Plot conditional correlations
    figure;
    plot(ccc_corr, 'b-', 'LineWidth', 1);
    hold on;
    plot(dcc_corr, 'r-', 'LineWidth', 1);
    plot(bekk_corr, 'g-', 'LineWidth', 1);
    plot(go_corr, 'm-', 'LineWidth', 1);
    hold off;
    title(['Conditional Correlation: ', pair_name], 'Interpreter', 'none');
    legend('CCC-MVGARCH', 'DCC-MVGARCH', 'BEKK-MVGARCH', 'GO-GARCH');
    xlabel('Time');
    ylabel('Correlation');
    ylim([-1, 1]);  % Correlations are in [-1, 1]
    grid on;
    
    % If more than 2 assets, visualize additional correlations using subplots
    if K > 2
        % Create a function to plot correlation dynamics for multiple pairs
        plot_correlation_dynamics(ccc_model, dcc_model, bekk_model, go_model, asset_names);
    end
end

%% Visualize Conditional Volatilities
% Compare volatility predictions from different models
figure;
for i = 1:min(K, 4)  % Display up to 4 assets for clarity
    subplot(min(K, 4), 1, i);
    
    % Extract conditional volatilities
    ccc_vol = sqrt(ccc_model.h(:, i));
    dcc_vol = sqrt(dcc_model.h(:, i));
    
    % Extract BEKK volatilities
    bekk_vol = zeros(T, 1);
    for t = 1:T
        bekk_vol(t) = sqrt(bekk_model.H(i, i, t));
    end
    
    % Extract GO-GARCH volatilities
    go_vol = zeros(T, 1);
    for t = 1:T
        go_vol(t) = sqrt(go_model.covariances(t, i, i));
    end
    
    plot(ccc_vol, 'b-', 'LineWidth', 1);
    hold on;
    plot(dcc_vol, 'r-', 'LineWidth', 1);
    plot(bekk_vol, 'g-', 'LineWidth', 1);
    plot(go_vol, 'm-', 'LineWidth', 1);
    hold off;
    title(['Conditional Volatility: ', asset_names{i}], 'Interpreter', 'none');
    if i == min(K, 4)  % Add legend to last subplot
        legend('CCC-MVGARCH', 'DCC-MVGARCH', 'BEKK-MVGARCH', 'GO-GARCH');
    end
    ylabel('Volatility');
    grid on;
end
xlabel('Time');

%% Forecast Comparison
% Compare volatility and correlation forecasts
disp(' ');
disp('=================================================================');
disp('Forecast Comparison');
disp('=================================================================');
disp(['Forecasting is a critical application of volatility models for risk']);
disp(['management and portfolio optimization. We compare forecasts from']);
disp(['different models to understand their predictive characteristics.']);

% Extract forecasts from each model
ccc_forecasts = ccc_model.forecast;
dcc_forecasts = dcc_model.forecast;
bekk_forecasts = bekk_model.forecast;
go_forecasts = go_model.forecast;

% Get forecast horizon
h = min([size(ccc_forecasts.h, 1), ...
         size(dcc_forecasts.h, 1), ...
         size(bekk_forecasts.covariance, 1), ...
         size(go_forecasts.covarianceForecasts, 1)]);  % Number of forecast periods

% Display forecasted volatilities for a selected asset
asset_idx = 1;  % Select the first asset (change if needed)
disp(['Volatility Forecasts for ', asset_names{asset_idx}, ':']);
disp('Day    CCC-MVGARCH    DCC-MVGARCH    BEKK-MVGARCH    GO-GARCH');
for i = 1:h
    forecast_values = [
        sqrt(ccc_forecasts.h(i, asset_idx)), ...
        sqrt(dcc_forecasts.h(i, asset_idx)), ...
        sqrt(bekk_forecasts.covariance(i, asset_idx, asset_idx)), ...
        sqrt(go_forecasts.covarianceForecasts(i, asset_idx, asset_idx))
    ];
    fprintf('%3d  %12.6f   %12.6f   %12.6f   %12.6f\n', ...
            i, forecast_values(1), forecast_values(2), ...
            forecast_values(3), forecast_values(4));
end

% Plot forecasted volatilities
figure;
plot(1:h, sqrt(ccc_forecasts.h(:, asset_idx)), 'b-o', 'LineWidth', 1.5);
hold on;
plot(1:h, sqrt(dcc_forecasts.h(:, asset_idx)), 'r-s', 'LineWidth', 1.5);
plot(1:h, sqrt(bekk_forecasts.covariance(:, asset_idx, asset_idx)), 'g-d', 'LineWidth', 1.5);
plot(1:h, sqrt(go_forecasts.covarianceForecasts(:, asset_idx, asset_idx)), 'm-^', 'LineWidth', 1.5);
hold off;
title(['Volatility Forecasts for ', asset_names{asset_idx}], 'Interpreter', 'none');
legend('CCC-MVGARCH', 'DCC-MVGARCH', 'BEKK-MVGARCH', 'GO-GARCH');
xlabel('Forecast Horizon');
ylabel('Volatility');
grid on;

% Compare correlation forecasts if K > 1
if K > 1
    pair_idx = [1, 2];  % Select a pair of assets (change if needed)
    
    % Extract correlation forecasts
    ccc_corr_forecast = zeros(h, 1);
    dcc_corr_forecast = zeros(h, 1);
    bekk_corr_forecast = zeros(h, 1);
    go_corr_forecast = zeros(h, 1);
    
    for i = 1:h
        % CCC correlation is constant
        ccc_corr_forecast(i) = ccc_model.correlations(pair_idx(1), pair_idx(2));
        
        % DCC correlation forecast
        % Safely extract correlation
        try
            H_dcc = squeeze(dcc_forecasts.cov(i, :, :));
            D_dcc = diag(sqrt(diag(H_dcc)));
            if any(diag(D_dcc) < 1e-8)
                dcc_corr_forecast(i) = 0;  % Default to zero if singular
            else
                R_dcc = inv(D_dcc) * H_dcc * inv(D_dcc);
                dcc_corr_forecast(i) = R_dcc(pair_idx(1), pair_idx(2));
                dcc_corr_forecast(i) = max(-1, min(1, dcc_corr_forecast(i)));
            end
        catch
            dcc_corr_forecast(i) = 0;  % Default to zero if error
        end
        
        % BEKK correlation forecast
        try
            H_bekk = squeeze(bekk_forecasts.covariance(i, :, :));
            D_bekk = diag(sqrt(diag(H_bekk)));
            if any(diag(D_bekk) < 1e-8)
                bekk_corr_forecast(i) = 0;  % Default to zero if singular
            else
                R_bekk = inv(D_bekk) * H_bekk * inv(D_bekk);
                bekk_corr_forecast(i) = R_bekk(pair_idx(1), pair_idx(2));
                bekk_corr_forecast(i) = max(-1, min(1, bekk_corr_forecast(i)));
            end
        catch
            bekk_corr_forecast(i) = 0;  % Default to zero if error
        end
        
        % GO-GARCH correlation forecast
        try
            H_go = squeeze(go_forecasts.covarianceForecasts(i, :, :));
            D_go = diag(sqrt(diag(H_go)));
            if any(diag(D_go) < 1e-8)
                go_corr_forecast(i) = 0;  % Default to zero if singular
            else
                R_go = inv(D_go) * H_go * inv(D_go);
                go_corr_forecast(i) = R_go(pair_idx(1), pair_idx(2));
                go_corr_forecast(i) = max(-1, min(1, go_corr_forecast(i)));
            end
        catch
            go_corr_forecast(i) = 0;  % Default to zero if error
        end
    end
    
    % Plot correlation forecasts
    figure;
    plot(1:h, ccc_corr_forecast, 'b-o', 'LineWidth', 1.5);
    hold on;
    plot(1:h, dcc_corr_forecast, 'r-s', 'LineWidth', 1.5);
    plot(1:h, bekk_corr_forecast, 'g-d', 'LineWidth', 1.5);
    plot(1:h, go_corr_forecast, 'm-^', 'LineWidth', 1.5);
    hold off;
    pair_name = [asset_names{pair_idx(1)}, '-', asset_names{pair_idx(2)}];
    title(['Correlation Forecasts: ', pair_name], 'Interpreter', 'none');
    legend('CCC-MVGARCH', 'DCC-MVGARCH', 'BEKK-MVGARCH', 'GO-GARCH');
    xlabel('Forecast Horizon');
    ylabel('Correlation');
    ylim([-1, 1]);  % Correlations are in [-1, 1]
    grid on;
end

%% Portfolio Risk Analysis
% Calculate portfolio risk measures using covariance matrices
disp(' ');
disp('=================================================================');
disp('Portfolio Risk Analysis');
disp('=================================================================');
disp(['Portfolio risk analysis shows how different volatility models']);
disp(['impact risk assessment and management decisions.']);

% Define portfolio weights (equal weighting by default)
portfolio_weights = ones(K, 1) / K;
disp(' ');
disp('Portfolio Weights:');
for i = 1:K
    fprintf('%s: %.4f\n', asset_names{i}, portfolio_weights(i));
end

% Calculate portfolio risk measures
portfolio_risk = calculate_portfolio_risk(ccc_forecasts, dcc_forecasts, ...
                                          bekk_forecasts, go_forecasts, ...
                                          portfolio_weights);

% Display portfolio volatility forecasts
disp(' ');
disp('Portfolio Volatility Forecasts:');
disp('Day    CCC-MVGARCH    DCC-MVGARCH    BEKK-MVGARCH    GO-GARCH');
for i = 1:h
    fprintf('%3d  %12.6f   %12.6f   %12.6f   %12.6f\n', ...
            i, portfolio_risk.volatility.ccc(i), ...
            portfolio_risk.volatility.dcc(i), ...
            portfolio_risk.volatility.bekk(i), ...
            portfolio_risk.volatility.go(i));
end

% Plot portfolio volatility forecasts
figure;
plot(1:h, portfolio_risk.volatility.ccc, 'b-o', 'LineWidth', 1.5);
hold on;
plot(1:h, portfolio_risk.volatility.dcc, 'r-s', 'LineWidth', 1.5);
plot(1:h, portfolio_risk.volatility.bekk, 'g-d', 'LineWidth', 1.5);
plot(1:h, portfolio_risk.volatility.go, 'm-^', 'LineWidth', 1.5);
hold off;
title('Portfolio Volatility Forecasts');
legend('CCC-MVGARCH', 'DCC-MVGARCH', 'BEKK-MVGARCH', 'GO-GARCH');
xlabel('Forecast Horizon');
ylabel('Portfolio Volatility');
grid on;

% Display diversification benefits
disp(' ');
disp('Diversification Benefits:');
disp(['The diversification benefit measures how much portfolio volatility']);
disp(['is reduced compared to a weighted average of individual volatilities.']);
disp(['Higher values indicate greater risk reduction from diversification.']);
disp(' ');
disp('Model          Diversification Benefit (%)');
if isfield(portfolio_risk, 'diversification')
    fprintf('CCC-MVGARCH    %12.2f%%\n', 100*portfolio_risk.diversification.ccc);
    fprintf('DCC-MVGARCH    %12.2f%%\n', 100*portfolio_risk.diversification.dcc);
    fprintf('BEKK-MVGARCH   %12.2f%%\n', 100*portfolio_risk.diversification.bekk);
    fprintf('GO-GARCH       %12.2f%%\n', 100*portfolio_risk.diversification.go);
end

% Display Value-at-Risk estimates
disp(' ');
disp('One-day 95% Value-at-Risk Estimates:');
disp(['VaR estimates the maximum expected loss with 95% confidence.']);
disp(['Expressed as a percentage of portfolio value.']);
disp(' ');
disp('Model          95% VaR (%)');
if isfield(portfolio_risk, 'VaR95')
    fprintf('CCC-MVGARCH    %12.2f%%\n', 100*portfolio_risk.VaR95.ccc);
    fprintf('DCC-MVGARCH    %12.2f%%\n', 100*portfolio_risk.VaR95.dcc);
    fprintf('BEKK-MVGARCH   %12.2f%%\n', 100*portfolio_risk.VaR95.bekk);
    fprintf('GO-GARCH       %12.2f%%\n', 100*portfolio_risk.VaR95.go);
end

%% Summary and Conclusions
disp(' ');
disp('=================================================================');
disp('Summary and Conclusions');
disp('=================================================================');
disp(['This example demonstrated four multivariate volatility models:']);
disp(['  1. CCC-MVGARCH: Simple, robust, but assumes constant correlations']);
disp(['  2. DCC-MVGARCH: Allows for time-varying correlations, popular in practice']);
disp(['  3. BEKK-MVGARCH: Directly models covariance matrix, guaranteed positive-definite']);
disp(['  4. GO-GARCH: Uses orthogonal factors, efficient for high dimensions']);
disp(' ');
disp(['Key considerations for model selection:']);
disp(['  - Goodness-of-fit (log-likelihood, AIC, BIC)']);
disp(['  - Computational complexity (CCC/DCC < GO-GARCH < BEKK)']);
disp(['  - Number of assets (DCC/GO-GARCH scale better to higher dimensions)']);
disp(['  - Forecast accuracy (compare out-of-sample performance)']);
disp(['  - Application needs (risk management, portfolio optimization, etc.)']);

%% Helper Function to Visualize Correlation Dynamics
function plot_correlation_dynamics(ccc_model, dcc_model, bekk_model, go_model, asset_names)
    % This function creates plots of correlation dynamics for multiple asset pairs
    % to compare how different models capture changing correlations over time
    
    % Get dimensions
    [~, K] = size(ccc_model.h);
    T = size(dcc_model.corr, 1);
    
    % Select pairs to visualize (up to 6 pairs for clarity)
    num_pairs = min(6, K*(K-1)/2);
    pair_indices = zeros(num_pairs, 2);
    pair_count = 0;
    
    for i = 1:(K-1)
        for j = (i+1):K
            pair_count = pair_count + 1;
            if pair_count <= num_pairs
                pair_indices(pair_count, :) = [i, j];
            else
                break;
            end
        end
        if pair_count >= num_pairs
            break;
        end
    end
    
    % Create subplots for each pair
    figure;
    for p = 1:num_pairs
        i = pair_indices(p, 1);
        j = pair_indices(p, 2);
        pair_name = [asset_names{i}, '-', asset_names{j}];
        
        % Extract correlations
        ccc_corr = zeros(T, 1);
        dcc_corr = zeros(T, 1);
        bekk_corr = zeros(T, 1);
        go_corr = zeros(T, 1);
        
        for t = 1:T
            % CCC (constant across time)
            ccc_corr(t) = ccc_model.correlations(i, j);
            
            % DCC
            dcc_corr(t) = dcc_model.corr(t, i, j);
            
            % BEKK
            try
                H_t = squeeze(bekk_model.H(:,:,t));
                D_t = diag(sqrt(diag(H_t)));
                % Check for singularity
                if any(diag(D_t) < 1e-8)
                    bekk_corr(t) = bekk_corr(max(1, t-1));  % Use previous value
                else
                    R_t = inv(D_t) * H_t * inv(D_t);
                    bekk_corr(t) = R_t(i, j);
                    % Ensure correlation is in valid range
                    bekk_corr(t) = max(-1, min(1, bekk_corr(t)));
                end
            catch
                bekk_corr(t) = bekk_corr(max(1, t-1));  % Use previous value if error
            end
            
            % GO-GARCH
            try
                H_t = squeeze(go_model.covariances(t, :, :));
                D_t = diag(sqrt(diag(H_t)));
                % Check for singularity
                if any(diag(D_t) < 1e-8)
                    go_corr(t) = go_corr(max(1, t-1));  % Use previous value
                else
                    R_t = inv(D_t) * H_t * inv(D_t);
                    go_corr(t) = R_t(i, j);
                    % Ensure correlation is in valid range
                    go_corr(t) = max(-1, min(1, go_corr(t)));
                end
            catch
                go_corr(t) = go_corr(max(1, t-1));  % Use previous value if error
            end
        end
        
        % Plot
        subplot(ceil(num_pairs/2), 2, p);
        plot(ccc_corr, 'b-', 'LineWidth', 1);
        hold on;
        plot(dcc_corr, 'r-', 'LineWidth', 1);
        plot(bekk_corr, 'g-', 'LineWidth', 1);
        plot(go_corr, 'm-', 'LineWidth', 1);
        hold off;
        title(['Correlation: ', pair_name], 'Interpreter', 'none');
        
        % Add legend to first subplot only
        if p == 1
            legend('CCC', 'DCC', 'BEKK', 'GO-GARCH', 'Location', 'best');
        end
        
        xlabel('Time');
        ylabel('Correlation');
        ylim([-1, 1]);  % Correlations are in [-1, 1]
        grid on;
    end
end

%% Helper Function to Calculate Portfolio Risk Measures
function risk = calculate_portfolio_risk(ccc_forecasts, dcc_forecasts, ...
                                         bekk_forecasts, go_forecasts, weights)
    % This function calculates portfolio risk measures using covariance matrices
    % from different multivariate GARCH models
    
    % Get forecast horizon (ensure consistent across models)
    h_ccc = size(ccc_forecasts.h, 1);
    h_dcc = size(dcc_forecasts.h, 1);
    h_bekk = size(bekk_forecasts.covariance, 1);
    h_go = size(go_forecasts.covarianceForecasts, 1);
    h = min([h_ccc, h_dcc, h_bekk, h_go]);
    
    % Initialize output structure
    risk = struct();
    risk.variance = struct('ccc', zeros(h, 1), 'dcc', zeros(h, 1), ...
                          'bekk', zeros(h, 1), 'go', zeros(h, 1));
    risk.volatility = struct('ccc', zeros(h, 1), 'dcc', zeros(h, 1), ...
                            'bekk', zeros(h, 1), 'go', zeros(h, 1));
    
    % Calculate portfolio variance for each model and forecast horizon
    for i = 1:h
        % Get covariance matrices, with safety checks
        try
            H_ccc = squeeze(ccc_forecasts.cov(i, :, :));
            % Ensure positive definiteness
            [V, D] = eig(H_ccc);
            d = diag(D);
            if any(d <= 1e-10)
                d(d <= 1e-10) = 1e-10;
                H_ccc = V * diag(d) * V';
                H_ccc = (H_ccc + H_ccc')/2;  % Ensure symmetry
            end
            risk.variance.ccc(i) = weights' * H_ccc * weights;
        catch
            % Fallback if covariance matrix extraction fails
            risk.variance.ccc(i) = sum(weights.^2 .* ccc_forecasts.h(i, :)');
        end
        
        try
            H_dcc = squeeze(dcc_forecasts.cov(i, :, :));
            % Ensure positive definiteness
            [V, D] = eig(H_dcc);
            d = diag(D);
            if any(d <= 1e-10)
                d(d <= 1e-10) = 1e-10;
                H_dcc = V * diag(d) * V';
                H_dcc = (H_dcc + H_dcc')/2;  % Ensure symmetry
            end
            risk.variance.dcc(i) = weights' * H_dcc * weights;
        catch
            % Fallback if covariance matrix extraction fails
            risk.variance.dcc(i) = sum(weights.^2 .* dcc_forecasts.h(i, :)');
        end
        
        try
            H_bekk = squeeze(bekk_forecasts.covariance(i, :, :));
            % Ensure positive definiteness
            [V, D] = eig(H_bekk);
            d = diag(D);
            if any(d <= 1e-10)
                d(d <= 1e-10) = 1e-10;
                H_bekk = V * diag(d) * V';
                H_bekk = (H_bekk + H_bekk')/2;  % Ensure symmetry
            end
            risk.variance.bekk(i) = weights' * H_bekk * weights;
        catch
            % Fallback if covariance matrix extraction fails
            risk.variance.bekk(i) = sum(weights.^2 .* diag(squeeze(bekk_forecasts.covariance(i, :, :))));
        end
        
        try
            H_go = squeeze(go_forecasts.covarianceForecasts(i, :, :));
            % Ensure positive definiteness
            [V, D] = eig(H_go);
            d = diag(D);
            if any(d <= 1e-10)
                d(d <= 1e-10) = 1e-10;
                H_go = V * diag(d) * V';
                H_go = (H_go + H_go')/2;  % Ensure symmetry
            end
            risk.variance.go(i) = weights' * H_go * weights;
        catch
            % Fallback if covariance matrix extraction fails
            risk.variance.go(i) = sum(weights.^2 .* diag(squeeze(go_forecasts.covarianceForecasts(i, :, :))));
        end
        
        % Calculate portfolio volatility (standard deviation)
        risk.volatility.ccc(i) = sqrt(risk.variance.ccc(i));
        risk.volatility.dcc(i) = sqrt(risk.variance.dcc(i));
        risk.volatility.bekk(i) = sqrt(risk.variance.bekk(i));
        risk.volatility.go(i) = sqrt(risk.variance.go(i));
    end
    
    % Calculate 1-day Value-at-Risk (VaR) at 95% confidence level
    % Assuming zero expected return and standard normal quantile
    alpha = 0.05;  % 5% tail probability
    z_alpha = norminv(alpha);  % Standard normal quantile
    
    risk.VaR95 = struct();
    risk.VaR95.ccc = -z_alpha * risk.volatility.ccc(1);
    risk.VaR95.dcc = -z_alpha * risk.volatility.dcc(1);
    risk.VaR95.bekk = -z_alpha * risk.volatility.bekk(1);
    risk.VaR95.go = -z_alpha * risk.volatility.go(1);
    
    % Calculate diversification benefit
    % (compared to weighted average of individual volatilities)
    risk.diversification = struct();
    
    % For each model, calculate individual asset volatilities and diversification benefit
    % Safe extraction of first forecast covariance for each model
    try
        H_ccc = squeeze(ccc_forecasts.cov(1, :, :));
        indiv_vol_ccc = sqrt(diag(H_ccc));
        weighted_avg_vol_ccc = weights' * indiv_vol_ccc;
        portfolio_vol_ccc = sqrt(weights' * H_ccc * weights);
        risk.diversification.ccc = 1 - portfolio_vol_ccc / weighted_avg_vol_ccc;
    catch
        risk.diversification.ccc = NaN;
    end
    
    try
        H_dcc = squeeze(dcc_forecasts.cov(1, :, :));
        indiv_vol_dcc = sqrt(diag(H_dcc));
        weighted_avg_vol_dcc = weights' * indiv_vol_dcc;
        portfolio_vol_dcc = sqrt(weights' * H_dcc * weights);
        risk.diversification.dcc = 1 - portfolio_vol_dcc / weighted_avg_vol_dcc;
    catch
        risk.diversification.dcc = NaN;
    end
    
    try
        H_bekk = squeeze(bekk_forecasts.covariance(1, :, :));
        indiv_vol_bekk = sqrt(diag(H_bekk));
        weighted_avg_vol_bekk = weights' * indiv_vol_bekk;
        portfolio_vol_bekk = sqrt(weights' * H_bekk * weights);
        risk.diversification.bekk = 1 - portfolio_vol_bekk / weighted_avg_vol_bekk;
    catch
        risk.diversification.bekk = NaN;
    end
    
    try
        H_go = squeeze(go_forecasts.covarianceForecasts(1, :, :));
        indiv_vol_go = sqrt(diag(H_go));
        weighted_avg_vol_go = weights' * indiv_vol_go;
        portfolio_vol_go = sqrt(weights' * H_go * weights);
        risk.diversification.go = 1 - portfolio_vol_go / weighted_avg_vol_go;
    catch
        risk.diversification.go = NaN;
    end
end