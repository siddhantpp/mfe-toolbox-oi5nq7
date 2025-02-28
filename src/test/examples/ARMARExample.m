function run_arma_example()
% RUN_ARMA_EXAMPLE Demonstrates ARMA/ARMAX time series modeling in the MFE Toolbox
%
% This example showcases the complete workflow for time series analysis using
% ARMA/ARMAX models, including:
% - Data preparation and preliminary analysis
% - Model identification using ACF and PACF
% - Model estimation with different orders
% - Model selection using information criteria and diagnostic tests
% - Forecasting with confidence intervals
% - Forecast evaluation and performance metrics
%
% The example uses financial return data and provides educational insights
% into time series modeling best practices and implementation.
%
% See also ARMAXFILTER, ARMAFOR, SACF, SPACF, AICSBIC, LJUNGBOX, ARCH_TEST

% Set random number generator for reproducibility
rng('default'); % Set random number generator for reproducibility

% Load example financial returns data
load('financial_returns.mat', 'returns', 'dates', 'assets');
disp('Loaded financial returns data:');
disp(['  - ', num2str(size(returns, 1)), ' observations']);
disp(['  - ', num2str(size(returns, 2)), ' assets']);
disp(['  - Date range: ', datestr(dates(1)), ' to ', datestr(dates(end))]);

% Select a single return series for simplicity (first asset)
selected_asset = 1;
asset_name = assets{selected_asset};
return_data = returns(:, selected_asset);

disp(['Selected asset: ', asset_name]);
disp(' ');

% Display basic statistical properties
disp('Basic statistics of the return series:');
disp(['  Mean:           ', num2str(mean(return_data))]);
disp(['  Std Deviation:  ', num2str(std(return_data))]);
disp(['  Min:            ', num2str(min(return_data))]);
disp(['  Max:            ', num2str(max(return_data))]);
disp(' ');

%% Step 1: Preliminary Analysis and Model Identification
disp('Step 1: Preliminary Analysis and Model Identification');
disp('-------------------------------------------------');

% Perform model identification based on ACF and PACF
ident_results = perform_model_identification(return_data);

% Display identification results
disp(['Suggested AR order (p): ', num2str(ident_results.suggested_p)]);
disp(['Suggested MA order (q): ', num2str(ident_results.suggested_q)]);
disp(' ');

%% Step 2: Model Estimation and Selection
disp('Step 2: Model Estimation and Selection');
disp('-------------------------------------------------');

% Set maximum orders to consider
max_ar = max(2, ident_results.suggested_p);
max_ma = max(2, ident_results.suggested_q);

disp(['Estimating ARMA models with orders up to p=', num2str(max_ar), ', q=', num2str(max_ma)]);

% Estimate multiple ARMA models with different orders
model_results = estimate_multiple_arma_models(return_data, max_ar, max_ma);

% Create and display model comparison table
create_arma_model_comparison_table(model_results);

% Select best model based on information criteria and diagnostic tests
% For demonstration, we'll choose the model with the lowest AIC
[~, best_idx] = min(model_results.aic);
best_p = model_results.p_values(best_idx);
best_q = model_results.q_values(best_idx);

disp(['Selected model: ARMA(', num2str(best_p), ',', num2str(best_q), ')']);
disp(['  AIC: ', num2str(model_results.aic(best_idx))]);
disp(['  SBIC: ', num2str(model_results.sbic(best_idx))]);
disp(['  Log-likelihood: ', num2str(model_results.logL(best_idx))]);

% Get the best model results
best_model = model_results.models{best_idx};

% Perform detailed residual diagnostics on the best model
disp(' ');
disp('Residual diagnostics for the selected model:');
diagnostic_results = perform_residual_diagnostics(best_model.residuals, best_model);

%% Step 3: Forecasting
disp(' ');
disp('Step 3: Forecasting');
disp('-------------------------------------------------');

% Set forecast horizon
forecast_horizon = 10;
disp(['Generating ', num2str(forecast_horizon), '-step ahead forecasts']);

% Generate forecasts
forecast_results = generate_arma_forecasts(best_model, return_data, forecast_horizon);

% Display forecast results
disp('Point forecasts:');
disp(forecast_results.forecasts);
disp(' ');

%% Step 4: Forecast Evaluation
disp('Step 4: Forecast Evaluation');
disp('-------------------------------------------------');

% To evaluate forecast accuracy, we'll use a hold-out sample
train_size = round(0.8 * length(return_data));
accuracy_results = evaluate_forecast_accuracy(return_data, best_model, train_size);

% Display accuracy metrics
disp('Forecast accuracy metrics:');
disp(['  Mean Error (ME):                 ', num2str(accuracy_results.me)]);
disp(['  Mean Absolute Error (MAE):       ', num2str(accuracy_results.mae)]);
disp(['  Root Mean Squared Error (RMSE):  ', num2str(accuracy_results.rmse)]);
disp(['  Mean Absolute Percentage Error:  ', num2str(accuracy_results.mape), '%']);

disp(' ');
disp('ARMA/ARMAX Example Complete');
end

function ident_results = perform_model_identification(data)
% PERFORM_MODEL_IDENTIFICATION Analyzes time series characteristics to identify
% potential ARMA model orders using ACF, PACF plots and preliminary testing
%
% INPUTS:
%   data - Time series data (numeric array)
%
% OUTPUTS:
%   ident_results - Structure with identification results and suggested orders

% Validate input data
data = datacheck(data, 'data');

% Calculate maximum lag for ACF/PACF
T = length(data);
max_lag = min(30, floor(T/4));

% Calculate sample ACF and PACF
[acf_values, acf_se, acf_ci] = sacf(data, max_lag);
[pacf_values, pacf_se, pacf_ci] = spacf(data, max_lag);

% Create a figure with ACF and PACF plots
figure;

% Plot ACF
subplot(2, 1, 1);
hold on;
stem(1:max_lag, acf_values, 'filled', 'MarkerSize', 3);
plot(1:max_lag, acf_ci(:,1), 'r--', 'LineWidth', 1);
plot(1:max_lag, acf_ci(:,2), 'r--', 'LineWidth', 1);
plot([0, max_lag+1], [0, 0], 'k-', 'LineWidth', 0.5);
title('Sample Autocorrelation Function (ACF)');
xlabel('Lag');
ylabel('Autocorrelation');
xlim([0, max_lag+1]);
hold off;

% Plot PACF
subplot(2, 1, 2);
hold on;
stem(1:max_lag, pacf_values, 'filled', 'MarkerSize', 3);
plot(1:max_lag, pacf_ci(:,1), 'r--', 'LineWidth', 1);
plot(1:max_lag, pacf_ci(:,2), 'r--', 'LineWidth', 1);
plot([0, max_lag+1], [0, 0], 'k-', 'LineWidth', 0.5);
title('Sample Partial Autocorrelation Function (PACF)');
xlabel('Lag');
ylabel('Partial Autocorrelation');
xlim([0, max_lag+1]);
hold off;

% Analyze ACF/PACF patterns to suggest model orders
% Look for significant spikes in ACF and PACF

% Count significant lags in ACF (using 95% confidence bounds)
sig_acf = abs(acf_values) > 1.96/sqrt(T);
sig_acf_count = sum(sig_acf);

% Count significant lags in PACF
sig_pacf = abs(pacf_values) > 1.96/sqrt(T);
sig_pacf_count = sum(sig_pacf);

% Determine cutoff points (where ACF/PACF becomes insignificant)
acf_cutoff = find(~sig_acf, 1, 'first');
if isempty(acf_cutoff)
    acf_cutoff = max_lag;
end

pacf_cutoff = find(~sig_pacf, 1, 'first');
if isempty(pacf_cutoff)
    pacf_cutoff = max_lag;
end

% Suggest model orders based on patterns
% AR process: PACF cuts off, ACF tails off
% MA process: ACF cuts off, PACF tails off
% ARMA process: Both ACF and PACF tail off

% Simple heuristic for initial suggestion
suggested_p = min(3, pacf_cutoff);
suggested_q = min(3, acf_cutoff);

% If both are high, suggest a low-order ARMA model as starting point
if suggested_p > 2 && suggested_q > 2
    suggested_p = min(suggested_p, 2);
    suggested_q = min(suggested_q, 2);
end

% Return results
ident_results = struct();
ident_results.acf = acf_values;
ident_results.pacf = pacf_values;
ident_results.acf_ci = acf_ci;
ident_results.pacf_ci = pacf_ci;
ident_results.sig_acf_count = sig_acf_count;
ident_results.sig_pacf_count = sig_pacf_count;
ident_results.suggested_p = suggested_p;
ident_results.suggested_q = suggested_q;
end

function model_results = estimate_multiple_arma_models(data, max_ar, max_ma)
% ESTIMATE_MULTIPLE_ARMA_MODELS Estimates multiple ARMA models with different
% orders and compares their performance using information criteria and
% diagnostic tests
%
% INPUTS:
%   data - Time series data (numeric array)
%   max_ar - Maximum AR order to consider
%   max_ma - Maximum MA order to consider
%
% OUTPUTS:
%   model_results - Structure with estimation results for all models

% Validate inputs
data = datacheck(data, 'data');

% Configure common estimation options
options = struct();
options.constant = true;           % Include constant term
options.distribution = 'normal';   % Assume normal errors for simplicity
options.stdErr = 'robust';         % Use robust standard errors

% Set maximum number of lags for diagnostic tests
T = length(data);
options.m = min(20, floor(T/4));

% Calculate total number of models to estimate
% Avoid empty model (p=0, q=0) which just has a constant
total_models = (max_ar + 1) * (max_ma + 1);
if max_ar >= 0 && max_ma >= 0
    total_models = total_models - 1;  % Subtract one for the case of p=0, q=0
end

% Initialize storage for results
models = cell(total_models, 1);
p_values = zeros(total_models, 1);
q_values = zeros(total_models, 1);
aic_values = zeros(total_models, 1);
sbic_values = zeros(total_models, 1);
logL_values = zeros(total_models, 1);
ljungbox_pvals = zeros(total_models, 1);
arch_pvals = zeros(total_models, 1);

% Counter for model index
model_idx = 0;

% Loop through AR orders
for p = 0:max_ar
    % Loop through MA orders
    for q = 0:max_ma
        % Skip if both p and q are 0 (model with only constant)
        if p == 0 && q == 0
            continue;
        end
        
        model_idx = model_idx + 1;
        
        % Update options with current AR and MA orders
        options.p = p;
        options.q = q;
        
        % Display progress
        disp(['Estimating ARMA(', num2str(p), ',', num2str(q), ') model...']);
        
        % Estimate model
        try
            model = armaxfilter(data, [], options);
            
            % Store results
            models{model_idx} = model;
            p_values(model_idx) = p;
            q_values(model_idx) = q;
            aic_values(model_idx) = model.aic;
            sbic_values(model_idx) = model.sbic;
            logL_values(model_idx) = model.logL;
            
            % Store diagnostic test results
            if ~isempty(model.ljungBox)
                % Use p-value of Ljung-Box test at lag 10 (or max available)
                lag_idx = min(10, length(model.ljungBox.pvals));
                ljungbox_pvals(model_idx) = model.ljungBox.pvals(lag_idx);
            end
            
            % Perform ARCH test on residuals
            arch_result = arch_test(model.residuals, 5);  % Test with 5 lags
            arch_pvals(model_idx) = arch_result.pval;
            
        catch ME
            % Handle estimation errors
            warning(['Error estimating ARMA(', num2str(p), ',', num2str(q), '): ', ME.message]);
            % Set invalid values for failed estimation
            models{model_idx} = [];
            aic_values(model_idx) = Inf;
            sbic_values(model_idx) = Inf;
            logL_values(model_idx) = -Inf;
            ljungbox_pvals(model_idx) = 0;
            arch_pvals(model_idx) = 0;
        end
    end
end

% Remove any empty entries (from failed estimations)
valid_idx = ~cellfun(@isempty, models);
models = models(valid_idx);
p_values = p_values(valid_idx);
q_values = q_values(valid_idx);
aic_values = aic_values(valid_idx);
sbic_values = sbic_values(valid_idx);
logL_values = logL_values(valid_idx);
ljungbox_pvals = ljungbox_pvals(valid_idx);
arch_pvals = arch_pvals(valid_idx);

% Collect results
model_results = struct();
model_results.models = models;
model_results.p_values = p_values;
model_results.q_values = q_values;
model_results.aic = aic_values;
model_results.sbic = sbic_values;
model_results.logL = logL_values;
model_results.ljungbox_pvals = ljungbox_pvals;
model_results.arch_pvals = arch_pvals;
end

function forecast_results = generate_arma_forecasts(model, data, horizon)
% GENERATE_ARMA_FORECASTS Generates and visualizes multi-step ahead forecasts
% for the selected ARMA model with confidence intervals
%
% INPUTS:
%   model - Estimated ARMA model structure from armaxfilter
%   data - Original time series data
%   horizon - Number of periods to forecast
%
% OUTPUTS:
%   forecast_results - Structure with forecast results

% Configure forecast options
method = 'exact';  % Use exact method for demonstration
nsim = 1000;       % Number of simulations if simulation method is used

% Generate forecasts using armafor
[forecasts, variances] = armafor(model.parameters, data, model.p, model.q, ...
                                model.constant, [], horizon, [], method, nsim);

% Calculate standard errors and confidence intervals
forecast_se = sqrt(variances);
alpha = 0.05;  % 95% confidence level
z_score = norminv(1 - alpha/2);  % Two-sided confidence interval
lower_ci = forecasts - z_score * forecast_se;
upper_ci = forecasts + z_score * forecast_se;

% Visualize forecasts
figure;
hold on;

% Plot historical data (last 100 observations for better visualization)
T = length(data);
plot_start = max(1, T-100);
time_idx = plot_start:T;
plot(time_idx, data(plot_start:T), 'b-', 'LineWidth', 1);

% Plot forecasts
forecast_idx = (T+1):(T+horizon);
plot(forecast_idx, forecasts, 'r-', 'LineWidth', 2);

% Plot confidence intervals
plot(forecast_idx, lower_ci, 'r--', 'LineWidth', 1);
plot(forecast_idx, upper_ci, 'r--', 'LineWidth', 1);

% Add shaded confidence interval
x_fill = [forecast_idx, fliplr(forecast_idx)];
y_fill = [lower_ci', fliplr(upper_ci')];
fill(x_fill, y_fill, 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Add vertical line separating historical data from forecasts
plot([T, T], ylim, 'k--', 'LineWidth', 1);

% Add labels and title
xlabel('Time');
ylabel('Value');
title(['ARMA(', num2str(model.p), ',', num2str(model.q), ') Forecasts with 95% Confidence Intervals']);
legend('Historical Data', 'Point Forecasts', 'Lower CI', 'Upper CI', 'Location', 'Best');
grid on;
hold off;

% Return forecast results
forecast_results = struct();
forecast_results.forecasts = forecasts;
forecast_results.variances = variances;
forecast_results.lower_ci = lower_ci;
forecast_results.upper_ci = upper_ci;
forecast_results.horizon = horizon;
end

function accuracy_results = evaluate_forecast_accuracy(data, model, train_size)
% EVALUATE_FORECAST_ACCURACY Evaluates the accuracy of ARMA forecasts by
% comparing predictions against held-out validation data
%
% INPUTS:
%   data - Complete time series data
%   model - Estimated ARMA model structure
%   train_size - Number of observations to use for training
%
% OUTPUTS:
%   accuracy_results - Structure with forecast accuracy metrics

% Split data into training and validation sets
train_data = data(1:train_size);
valid_data = data(train_size+1:end);
valid_size = length(valid_data);

% Set up options for re-estimation on training data
options = struct();
options.p = model.p;
options.q = model.q;
options.constant = model.constant;
options.distribution = model.distribution;

% Re-estimate model on training data
train_model = armaxfilter(train_data, [], options);

% Generate forecasts for validation period
[forecasts, variances] = armafor(train_model.parameters, train_data, ...
                               train_model.p, train_model.q, ...
                               train_model.constant, [], valid_size);

% Calculate accuracy metrics
errors = valid_data - forecasts;
me = mean(errors);  % Mean Error
mae = mean(abs(errors));  % Mean Absolute Error
rmse = sqrt(mean(errors.^2));  % Root Mean Squared Error
mape = mean(abs(errors ./ valid_data)) * 100;  % Mean Absolute Percentage Error

% Visualize actual vs. forecast
figure;
hold on;

% Plot actual data
plot(1:valid_size, valid_data, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Actual');

% Plot forecasts
plot(1:valid_size, forecasts, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Forecast');

% Add labels and title
xlabel('Time Steps');
ylabel('Value');
title('Forecast Evaluation: Actual vs. Predicted');
legend('Location', 'Best');
grid on;
hold off;

% Return accuracy metrics
accuracy_results = struct();
accuracy_results.me = me;
accuracy_results.mae = mae;
accuracy_results.rmse = rmse;
accuracy_results.mape = mape;
accuracy_results.errors = errors;
accuracy_results.forecasts = forecasts;
accuracy_results.actual = valid_data;
end

function diagnostic_results = perform_residual_diagnostics(residuals, model)
% PERFORM_RESIDUAL_DIAGNOSTICS Performs and visualizes comprehensive diagnostic
% tests on ARMA model residuals to validate model adequacy
%
% INPUTS:
%   residuals - Model residuals from estimation
%   model - Estimated ARMA model structure
%
% OUTPUTS:
%   diagnostic_results - Structure with diagnostic test results

% Validate inputs
residuals = datacheck(residuals, 'residuals');

% Create a new figure for diagnostic plots
figure;

% Plot 1: Residual time series
subplot(2, 2, 1);
plot(residuals);
title('Residual Time Series');
xlabel('Time');
ylabel('Residual');
grid on;

% Plot 2: ACF of residuals
subplot(2, 2, 2);
[acf_values, acf_se, acf_ci] = sacf(residuals, 20);
hold on;
stem(1:20, acf_values, 'filled', 'MarkerSize', 3);
plot(1:20, acf_ci(:,1), 'r--', 'LineWidth', 1);
plot(1:20, acf_ci(:,2), 'r--', 'LineWidth', 1);
plot([0, 21], [0, 0], 'k-', 'LineWidth', 0.5);
title('ACF of Residuals');
xlabel('Lag');
ylabel('Autocorrelation');
xlim([0, 21]);
hold off;

% Plot 3: Histogram of residuals with normal overlay
subplot(2, 2, 3);
[counts, edges] = histcounts(residuals, 'Normalization', 'pdf');
edges_centers = (edges(1:end-1) + edges(2:end))/2;
hold on;
bar(edges_centers, counts, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none');

% Overlay normal distribution
x_range = linspace(min(residuals), max(residuals), 100);
mu = mean(residuals);
sigma = std(residuals);
normal_pdf = normpdf(x_range, mu, sigma);
plot(x_range, normal_pdf, 'r-', 'LineWidth', 2);

title('Histogram of Residuals with Normal Overlay');
xlabel('Residual');
ylabel('Density');
hold off;

% Plot 4: QQ plot of residuals
subplot(2, 2, 4);
% Simple implementation of QQ plot
sorted_residuals = sort(residuals);
n = length(sorted_residuals);
p = ((1:n) - 0.5) / n;
theoretical_quantiles = norminv(p, 0, 1);
plot(theoretical_quantiles, sorted_residuals, 'b.', 'MarkerSize', 8);
hold on;
% Add reference line
min_val = min(theoretical_quantiles);
max_val = max(theoretical_quantiles);
plot([min_val, max_val], [min_val*sigma + mu, max_val*sigma + mu], 'r-', 'LineWidth', 1);
title('QQ Plot of Residuals');
xlabel('Theoretical Quantiles');
ylabel('Sample Quantiles');
grid on;
hold off;

% Perform Ljung-Box test for autocorrelation
m = min(20, floor(length(residuals)/4));
lb_test = ljungbox(residuals, m, model.p + model.q);

% Perform ARCH test for heteroskedasticity
arch_result = arch_test(residuals, 5);  % Test with 5 lags

% Display test results
disp('Residual Diagnostic Tests:');
disp('  Ljung-Box test for autocorrelation:');
disp(['    Test statistic at lag 10: ', num2str(lb_test.stats(10))]);
disp(['    p-value at lag 10: ', num2str(lb_test.pvals(10))]);
disp(['    Reject H0 (presence of autocorrelation) at 5% level: ', ...
      num2str(~lb_test.isRejected5pct(10))]);

disp('  ARCH test for heteroskedasticity:');
disp(['    Test statistic: ', num2str(arch_result.statistic)]);
disp(['    p-value: ', num2str(arch_result.pval)]);
disp(['    Reject H0 (no ARCH effects) at 5% level: ', ...
      num2str(arch_result.H0rejected.five)]);

% Calculate descriptive statistics of residuals
mean_resid = mean(residuals);
std_resid = std(residuals);
skew_resid = sum(((residuals - mean_resid)./std_resid).^3)/length(residuals);  % Compute skewness manually
kurt_resid = sum(((residuals - mean_resid)./std_resid).^4)/length(residuals);  % Compute kurtosis manually

disp('  Descriptive statistics:');
disp(['    Mean: ', num2str(mean_resid), ' (should be close to zero)']);
disp(['    Standard deviation: ', num2str(std_resid)]);
disp(['    Skewness: ', num2str(skew_resid)]);
disp(['    Kurtosis: ', num2str(kurt_resid), ' (normal = 3)']);

% Return diagnostic results
diagnostic_results = struct();
diagnostic_results.ljungbox = lb_test;
diagnostic_results.arch_test = arch_result;
diagnostic_results.mean = mean_resid;
diagnostic_results.std = std_resid;
diagnostic_results.skewness = skew_resid;
diagnostic_results.kurtosis = kurt_resid;
diagnostic_results.acf = acf_values;
diagnostic_results.acf_ci = acf_ci;
end

function create_arma_model_comparison_table(model_collection)
% CREATE_ARMA_MODEL_COMPARISON_TABLE Creates a formatted comparison table of
% multiple ARMA models, highlighting selection criteria and diagnostic test results
%
% INPUTS:
%   model_collection - Structure containing estimation results for multiple models
%
% OUTPUTS:
%   Displays formatted comparison table

% Extract model specifications and performance metrics
p_values = model_collection.p_values;
q_values = model_collection.q_values;
aic_values = model_collection.aic;
sbic_values = model_collection.sbic;
logL_values = model_collection.logL;
ljungbox_pvals = model_collection.ljungbox_pvals;
arch_pvals = model_collection.arch_pvals;

% Count number of models
num_models = length(p_values);

% Sort models by AIC (ascending)
[sorted_aic, sort_idx] = sort(aic_values);
sorted_p = p_values(sort_idx);
sorted_q = q_values(sort_idx);
sorted_sbic = sbic_values(sort_idx);
sorted_logL = logL_values(sort_idx);
sorted_ljungbox = ljungbox_pvals(sort_idx);
sorted_arch = arch_pvals(sort_idx);

% Find best models according to different criteria
[~, best_aic_idx] = min(aic_values);
[~, best_sbic_idx] = min(sbic_values);
[~, best_logL_idx] = max(logL_values);

% Create table header
disp(' ');
disp('Model Comparison Table (sorted by AIC)');
disp('-----------------------------------------------------------------------');
disp('                 |                  |     Diagnostic Tests    |');
disp(' Model     |     Information Criteria     |  (p-values > 0.05 = good)  |');
disp('-----------------------------------------------------------------------');
disp(' ARMA(p,q) |   AIC    |   SBIC   |   LogL   | Ljung-Box | ARCH Test |');
disp('-----------------------------------------------------------------------');

% Display model comparison table
for i = 1:min(num_models, 10)  % Show top 10 models at most
    % Format model specification
    model_spec = sprintf('ARMA(%d,%d)', sorted_p(i), sorted_q(i));
    
    % Format information criteria and log-likelihood
    aic_str = sprintf('%8.2f', sorted_aic(i));
    sbic_str = sprintf('%8.2f', sorted_sbic(i));
    logL_str = sprintf('%8.2f', sorted_logL(i));
    
    % Format diagnostic test p-values
    lb_str = sprintf('%8.3f', sorted_ljungbox(i));
    arch_str = sprintf('%8.3f', sorted_arch(i));
    
    % Add highlighting for best models
    if sort_idx(i) == best_aic_idx
        aic_str = ['*' aic_str '*'];
    end
    if sort_idx(i) == best_sbic_idx
        sbic_str = ['*' sbic_str '*'];
    end
    if sort_idx(i) == best_logL_idx
        logL_str = ['*' logL_str '*'];
    end
    
    % Display row
    disp([' ', model_spec, repmat(' ', 1, 11-length(model_spec)), ...
          '| ', aic_str, ' | ', sbic_str, ' | ', logL_str, ...
          ' | ', lb_str, ' | ', arch_str, ' |']);
end

disp('-----------------------------------------------------------------------');
disp('* Indicates best model according to each criterion');
disp('For diagnostic tests, p-values > 0.05 indicate model adequacy');
disp('  (no significant autocorrelation or ARCH effects)');
disp(' ');
end