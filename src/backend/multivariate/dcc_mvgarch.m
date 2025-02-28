function model = dcc_mvgarch(data, options)
% DCC_MVGARCH Estimates a Dynamic Conditional Correlation (DCC) Multivariate GARCH model
%
% USAGE:
%   MODEL = dcc_mvgarch(DATA)
%   MODEL = dcc_mvgarch(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T x K matrix of financial time series data
%             (T observations, K series)
%   OPTIONS  - [OPTIONAL] Structure with estimation options:
%              options.p - Scalar integer, GARCH order [1]
%              options.q - Scalar integer, ARCH order [1]
%              options.model - String specifying GARCH model type ['GARCH']
%                              (supported: 'GARCH','EGARCH','GJR','TARCH')
%              options.distribution - String specifying error distribution ['NORMAL']
%                                   (supported: 'NORMAL','T','GED','SKEWT')
%              options.forecast - Integer, forecast horizon [0]
%              options.method - String, estimation method ['2STAGE']
%                              (supported: '2STAGE','JOINT')
%              options.optimizer - String, optimization method for DCC parameters ['FMINCON']
%              options.startvalues - Vector, starting values for DCC parameters [auto]
%              options.dccP - Scalar integer, DCC correlation parameter order [1]
%              options.dccQ - Scalar integer, DCC memory parameter order [1]
%              options.constraints - Structure with constraints for estimation
%                                  options.constraints.a - Upper bound for DCC parameters [1]
%                                  options.constraints.b - Upper bound for DCC parameters [1]
%                                  options.constraints.sum - Upper bound for sum of a and b [0.999]
%              options.optimoptions - Options structure for optimization function
%
% OUTPUTS:
%   MODEL    - Structure with the following fields:
%              model.parameters - Structure with estimated parameters
%                                model.parameters.univariate - K x 1 cell of univariate GARCH parameters
%                                model.parameters.dcc - Vector of DCC parameters
%              model.likelihood - Overall log-likelihood at the optimum
%              model.corr - T x K x K array of time-varying correlation matrices
%              model.cov - T x K x K array of time-varying covariance matrices
%              model.h - T x K matrix of conditional variances
%              model.std_residuals - T x K matrix of standardized residuals
%              model.residuals - T x K matrix of residuals
%              model.data - T x K matrix of original data
%              model.stats - Structure with estimation statistics
%                           model.stats.aic - Akaike Information Criterion
%                           model.stats.bic - Bayesian Information Criterion
%              model.forecast - Structure with forecast results (if requested)
%                             model.forecast.h - h x K matrix of variance forecasts
%                             model.forecast.corr - h x K x K array of correlation forecasts
%                             model.forecast.cov - h x K x K array of covariance forecasts
%
% COMMENTS:
%   Implements a DCC-MVGARCH model with various univariate GARCH specifications.
%   The model uses a two-stage estimation procedure by default:
%   1. Estimate individual univariate GARCH models for each series
%   2. Estimate DCC parameters conditional on the univariate estimates
%
%   DCC model:
%   R_t = diag(Q_t)^(-1/2) * Q_t * diag(Q_t)^(-1/2)
%   Q_t = (1-a-b)*Q_bar + a*(ε_{t-1}*ε_{t-1}') + b*Q_{t-1}
%   where:
%   - R_t is the time-varying correlation matrix
%   - Q_t is a quasi-correlation matrix
%   - Q_bar is the unconditional correlation of standardized residuals
%   - a, b are the DCC parameters
%
% EXAMPLES:
%   % Estimate a DCC-GARCH(1,1) model with normal errors
%   model = dcc_mvgarch(returns);
%
%   % Estimate a DCC-EGARCH(1,1) model with t-distributed errors and 5-day forecast
%   options = struct('model', 'EGARCH', 'distribution', 'T', 'forecast', 5);
%   model = dcc_mvgarch(returns, options);
%
% See also GARCH, EGARCH, GJR, TARCH, ARMAXFILTER

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Check for MEX availability
persistent MEX_AVAILABLE;
if isempty(MEX_AVAILABLE)
    MEX_AVAILABLE = (exist('composite_likelihood', 'file') == 3);
end

% Minimum value for variance to prevent numerical issues
MIN_VARIANCE = 1e-12;

%% Input validation
% Check input data
data = datacheck(data, 'data');

% Get dimensions of data
[T, K] = size(data);

% Check if data is column format (for multiple series)
if K == 1
    error('At least two series are required for multivariate GARCH');
end

%% Process options
if nargin < 2 || isempty(options)
    options = struct();
end

% Set default options if not provided
if ~isfield(options, 'p')
    options.p = 1;
end

if ~isfield(options, 'q')
    options.q = 1;
end

if ~isfield(options, 'model')
    options.model = 'GARCH';
end

if ~isfield(options, 'distribution')
    options.distribution = 'NORMAL';
end

if ~isfield(options, 'forecast')
    options.forecast = 0;
end

if ~isfield(options, 'method')
    options.method = '2STAGE';
end

if ~isfield(options, 'optimizer')
    options.optimizer = 'FMINCON';
end

if ~isfield(options, 'dccP')
    options.dccP = 1;
end

if ~isfield(options, 'dccQ')
    options.dccQ = 1;
end

% Set up constraints for DCC parameters
if ~isfield(options, 'constraints')
    options.constraints = struct();
end

if ~isfield(options.constraints, 'a')
    options.constraints.a = 1;
end

if ~isfield(options.constraints, 'b')
    options.constraints.b = 1;
end

if ~isfield(options.constraints, 'sum')
    options.constraints.sum = 0.999;
end

% Optimization options for DCC parameter estimation
if ~isfield(options, 'optimoptions')
    options.optimoptions = optimoptions('fmincon', 'Display', 'iter');
end

% Allocate output arrays
h = zeros(T, K);              % Conditional variances
std_residuals = zeros(T, K);  % Standardized residuals
univariate_parameters = cell(K, 1);  % Store parameters for each series

%% Step 1: Estimate univariate GARCH models
% Set up options for univariate GARCH estimation
univariate_options = struct();
univariate_options.p = options.p;
univariate_options.q = options.q;
univariate_options.model = options.model;
univariate_options.distribution = options.distribution;

for k = 1:K
    % Extract data for current series
    series_data = data(:, k);
    
    % Initialize starting values for GARCH parameters
    start_params = garchinit(series_data, univariate_options);
    
    % Set up optimization options
    if strcmpi(options.model, 'EGARCH')
        % EGARCH doesn't need parameter constraints in the same way
        uni_constraints = [];
    else
        % Parameter constraints for other models to ensure stationarity
        A = []; b = []; Aeq = []; beq = [];
        lb = zeros(size(start_params)); % Lower bounds (non-negative parameters)
        ub = Inf(size(start_params));   % Upper bounds (infinity)
        uni_constraints = struct('A', A, 'b', b, 'Aeq', Aeq, 'beq', beq, 'lb', lb, 'ub', ub);
    end
    
    % Optimize GARCH parameters for this series
    [parameters, ~, h(:, k), ~, ~, ~, ~] = garch(series_data, univariate_options.p, ...
                                               univariate_options.q, univariate_options, ...
                                               uni_constraints, start_params);
    
    % Store parameters for this univariate series
    univariate_parameters{k} = parameters;
    
    % Calculate standardized residuals
    std_residuals(:, k) = series_data ./ sqrt(max(h(:, k), MIN_VARIANCE));
end

%% Step 2: Estimate DCC parameters
% Calculate unconditional correlation matrix from standardized residuals
Q_bar = std_residuals' * std_residuals / T;
% Normalize to ensure it's a valid correlation matrix
Q_bar = normalize_correlation(Q_bar);

% Initial values for DCC parameters (a and b)
if isfield(options, 'startvalues') && ~isempty(options.startvalues)
    dcc_start = options.startvalues;
else
    % Default starting values
    a = 0.01 * ones(options.dccP, 1);
    b = 0.97 * ones(options.dccQ, 1);
    dcc_start = [a; b];
end

% Set up constraints for DCC parameters
% a, b must be positive; a + b < 1 for stationarity
A = ones(1, options.dccP + options.dccQ);  % Sum of a and b
b_constraint = options.constraints.sum;    % Typically 0.999
lb = zeros(options.dccP + options.dccQ, 1); % Lower bounds (non-negative parameters)
ub = [options.constraints.a * ones(options.dccP, 1); options.constraints.b * ones(options.dccQ, 1)]; % Upper bounds

dcc_constraints = struct('A', A, 'b', b_constraint, 'Aeq', [], 'beq', [], 'lb', lb, 'ub', ub);

% Objective function for DCC optimization
dcc_objective = @(params) dcc_likelihood(params, std_residuals, Q_bar, options);

% Estimate DCC parameters
[dcc_parameters, neg_likelihood, ~, ~, ~, ~, ~] = ...
    fmincon(dcc_objective, dcc_start, dcc_constraints.A, dcc_constraints.b, ...
            dcc_constraints.Aeq, dcc_constraints.beq, dcc_constraints.lb, ...
            dcc_constraints.ub, [], options.optimoptions);

% Convert negative log-likelihood to log-likelihood
likelihood = -neg_likelihood;

% Extract DCC parameters
a_params = dcc_parameters(1:options.dccP);
b_params = dcc_parameters(options.dccP+1:options.dccP+options.dccQ);

%% Compute time-varying correlation matrices
% Initialize correlation and covariance arrays
corr_matrices = zeros(T, K, K);
cov_matrices = zeros(T, K, K);

% Initialize Q matrix with unconditional correlation
Qt = Q_bar;

% Apply DCC recursion to compute time-varying correlations
for t = 2:T
    % Previous standardized residuals outer product
    epsilon_t_minus_1 = std_residuals(t-1, :)';
    outer_prod = epsilon_t_minus_1 * epsilon_t_minus_1';
    
    % DCC recursion: Qt = (1-sum(a)-sum(b))*Q_bar + sum(a_i*outer_prod) + sum(b_j*Qt_prev)
    Qt_new = (1 - sum(a_params) - sum(b_params)) * Q_bar;
    
    % Add ARCH effects (a_i terms)
    for i = 1:options.dccP
        if t > i
            Qt_new = Qt_new + a_params(i) * outer_prod;
        end
    end
    
    % Add GARCH effects (b_j terms)
    for j = 1:options.dccQ
        if t > j
            Qt_new = Qt_new + b_params(j) * Qt;
        end
    end
    
    % Update Qt for next iteration
    Qt = Qt_new;
    
    % Normalize to get correlation matrix
    Rt = normalize_correlation(Qt);
    
    % Store correlation matrix
    corr_matrices(t, :, :) = Rt;
    
    % Compute covariance matrix from correlation and univariate variances
    % Create diagonal matrix of standard deviations
    D_t = diag(sqrt(h(t, :)));
    
    % H_t = D_t * R_t * D_t
    cov_matrices(t, :, :) = D_t * Rt * D_t;
end

% Fill first time point with unconditional correlation
corr_matrices(1, :, :) = Q_bar;
D_1 = diag(sqrt(h(1, :)));
cov_matrices(1, :, :) = D_1 * Q_bar * D_1;

%% Compute model statistics
% Calculate information criteria
n_univariate_params = 0;
for k = 1:K
    n_univariate_params = n_univariate_params + length(univariate_parameters{k});
end
n_dcc_params = length(dcc_parameters);
total_params = n_univariate_params + n_dcc_params;

% Compute AIC and BIC
ic = aicsbic(likelihood, total_params, T);

%% Generate forecasts if requested
if options.forecast > 0
    forecast = dcc_forecast(struct('parameters', struct('univariate', {univariate_parameters}, ...
                                                   'dcc', dcc_parameters), ...
                               'h', h, ...
                               'corr', corr_matrices, ...
                               'data', data, ...
                               'options', options), ...
                          options.forecast);
else
    forecast = [];
end

%% Assemble output structure
model = struct();
model.parameters = struct('univariate', {univariate_parameters}, 'dcc', dcc_parameters);
model.likelihood = likelihood;
model.corr = corr_matrices;
model.cov = cov_matrices;
model.h = h;
model.std_residuals = std_residuals;
model.residuals = data;
model.data = data;
model.stats = struct('aic', ic.aic, 'bic', ic.sbic);
model.options = options;
model.Q_bar = Q_bar;

% Add forecast if generated
if options.forecast > 0
    model.forecast = forecast;
end

% Validate the model
model.validation = validate_dcc_model(model);

end

%--------------------------------------------------------------------------
function neg_likelihood = dcc_likelihood(parameters, std_residuals, Q_bar, options)
% DCC_LIKELIHOOD Computes negative log-likelihood for DCC parameters in MGARCH models
%
% USAGE:
%   NEG_LIKELIHOOD = dcc_likelihood(PARAMETERS, STD_RESIDUALS, Q_BAR, OPTIONS)
%
% INPUTS:
%   PARAMETERS    - Vector of DCC parameters [a; b]
%   STD_RESIDUALS - T x K matrix of standardized residuals
%   Q_BAR         - K x K unconditional correlation matrix
%   OPTIONS       - Structure with options (same as for dcc_mvgarch)
%
% OUTPUTS:
%   NEG_LIKELIHOOD - Negative log-likelihood value (for minimization)
%
% COMMENTS:
%   This function computes the negative log-likelihood for DCC parameters,
%   conditioning on univariate GARCH parameters. Used internally by dcc_mvgarch.

% Check if MEX implementation is available for likelihood calculation
persistent MEX_AVAILABLE;
if isempty(MEX_AVAILABLE)
    MEX_AVAILABLE = (exist('composite_likelihood', 'file') == 3);
end

% Minimum value for variance to prevent numerical issues
MIN_VARIANCE = 1e-12;

% Extract DCC parameters
dcc_p = options.dccP;
dcc_q = options.dccQ;
a_params = parameters(1:dcc_p);
b_params = parameters(dcc_p+1:dcc_p+dcc_q);

% Check stationarity constraint
if sum(a_params) + sum(b_params) >= 1
    neg_likelihood = 1e10;  % Large value to penalize invalid parameters
    return;
end

% Get dimensions
[T, K] = size(std_residuals);

% Initialize log-likelihood
log_likelihood = 0;

% Initialize arrays
Qt = Q_bar;  % Initialize with unconditional correlation
Rt = zeros(K, K);  % Correlation matrix
log_det_Rt = zeros(T, 1);
quad_form = zeros(T, 1);

% Distribution constants
if strcmp(options.distribution, 'T')
    % For multivariate t-distribution, we need the degrees of freedom
    if isfield(options, 'nu')
        nu = options.nu;
    else
        % Default to a moderate value if not specified
        nu = 8;
    end
    halfnuplus = (nu + K) / 2;
    halfnu = nu / 2;
    log_gamma_ratio = gammaln(halfnuplus) - gammaln(halfnu);
    log_constant = log_gamma_ratio - (K/2) * log(pi * (nu - 2));
end

% Check if we can use MEX implementation for speed
use_mex = MEX_AVAILABLE && K > 2 && strcmp(options.distribution, 'NORMAL');

if use_mex
    % Prepare data for MEX function
    % This would use the composite_likelihood MEX function
    % Since we don't have access to the MEX implementation details, 
    % we fall back to MATLAB implementation
    use_mex = false;
end

if ~use_mex
    % MATLAB implementation
    for t = 2:T
        % Previous standardized residuals
        epsilon_t_minus_1 = std_residuals(t-1, :)';
        outer_prod = epsilon_t_minus_1 * epsilon_t_minus_1';
        
        % DCC recursion: Qt = (1-sum(a)-sum(b))*Q_bar + sum(a_i*outer_prod) + sum(b_j*Qt_prev)
        Qt_new = (1 - sum(a_params) - sum(b_params)) * Q_bar;
        
        % Add ARCH effects (a_i terms)
        for i = 1:dcc_p
            if t > i
                Qt_new = Qt_new + a_params(i) * outer_prod;
            end
        end
        
        % Add GARCH effects (b_j terms)
        for j = 1:dcc_q
            if t > j
                Qt_new = Qt_new + b_params(j) * Qt;
            end
        end
        
        % Update Qt for next iteration
        Qt = Qt_new;
        
        % Normalize to get correlation matrix
        Rt = normalize_correlation(Qt);
        
        % Current standardized residuals
        epsilon_t = std_residuals(t, :)';
        
        % Calculate log determinant of correlation matrix
        log_det_Rt(t) = log(det(Rt));
        
        % Calculate quadratic form: ε'_t * R^(-1)_t * ε_t
        quad_form(t) = epsilon_t' * inv(Rt) * epsilon_t;
    end
    
    % Compute log-likelihood based on distribution type
    if strcmp(options.distribution, 'NORMAL')
        % Multivariate normal log-likelihood
        % l_t = -0.5*K*log(2π) - 0.5*log|R_t| - 0.5*ε'_t*R^(-1)_t*ε_t
        log_likelihood = -0.5 * K * log(2*pi) * T - 0.5 * sum(log_det_Rt) - 0.5 * sum(quad_form);
    elseif strcmp(options.distribution, 'T')
        % Multivariate t-distribution log-likelihood
        % l_t = log_constant - 0.5*log|R_t| - (ν+K)/2 * log[1 + ε'_t*R^(-1)_t*ε_t / (ν-2)]
        log_likelihood_parts = log_constant - 0.5 * log_det_Rt - ...
            halfnuplus * log(1 + quad_form / (nu - 2));
        log_likelihood = sum(log_likelihood_parts);
    elseif strcmp(options.distribution, 'GED')
        % For GED, we'd need to implement multivariate GED log-likelihood
        % As an approximation, use normal for now
        warning('Multivariate GED not fully implemented, using normal approximation.');
        log_likelihood = -0.5 * K * log(2*pi) * T - 0.5 * sum(log_det_Rt) - 0.5 * sum(quad_form);
    elseif strcmp(options.distribution, 'SKEWT')
        % For skewed t, we'd need to implement multivariate skewed t log-likelihood
        % As an approximation, use normal for now
        warning('Multivariate skewed t not fully implemented, using normal approximation.');
        log_likelihood = -0.5 * K * log(2*pi) * T - 0.5 * sum(log_det_Rt) - 0.5 * sum(quad_form);
    else
        error('Unsupported distribution type: %s', options.distribution);
    end
end

% Return negative log-likelihood for minimization
neg_likelihood = -log_likelihood;

end

%--------------------------------------------------------------------------
function forecast = dcc_forecast(model, horizon)
% DCC_FORECAST Generates forecasts for DCC-MVGARCH models
%
% USAGE:
%   FORECAST = dcc_forecast(MODEL, HORIZON)
%
% INPUTS:
%   MODEL   - Structure produced by dcc_mvgarch
%   HORIZON - Forecast horizon (number of steps ahead)
%
% OUTPUTS:
%   FORECAST - Structure with the following fields:
%              forecast.h - HORIZON x K matrix of conditional variance forecasts
%              forecast.corr - HORIZON x K x K array of correlation matrix forecasts
%              forecast.cov - HORIZON x K x K array of covariance matrix forecasts
%
% COMMENTS:
%   This function generates multi-step forecasts for DCC-MVGARCH models.
%   The forecast includes conditional variances, correlation matrices, and
%   covariance matrices for each forecast horizon.

% Extract dimensions
T = size(model.data, 1);
K = size(model.data, 2);

% Extract parameters
univariate_params = model.parameters.univariate;
dcc_params = model.parameters.dcc;

% Extract DCC model parameters
dcc_p = length(model.parameters.dcc) / 2;
dcc_q = length(model.parameters.dcc) / 2;
a_params = dcc_params(1:dcc_p);
b_params = dcc_params(dcc_p+1:dcc_p+dcc_q);

% Extract unconditional correlation matrix
Q_bar = model.Q_bar;

% Extract last observed values
last_std_resid = model.std_residuals(end, :)';
last_h = model.h(end, :);
last_Qt = reshape(model.corr(end, :, :), K, K);

% Initialize forecast arrays
h_forecast = zeros(horizon, K);
corr_forecast = zeros(horizon, K, K);
cov_forecast = zeros(horizon, K, K);

% Forecast univariate GARCH processes
for k = 1:K
    % Get model type and parameters for this series
    params = univariate_params{k};
    model_type = model.options.model;
    p = model.options.p;
    q = model.options.q;
    
    % Generate variance forecasts
    h_forecast(:, k) = forecast_univariate_garch(params, last_h(k), ...
                                               model.data(:, k), model_type, p, q, horizon);
end

% Forecast DCC correlation dynamics
Qt_forecast = last_Qt;
for h = 1:horizon
    if h == 1
        % For one-step ahead, use last standardized residuals
        epsilon_outer = last_std_resid * last_std_resid';
    else
        % For multi-step ahead, the expected outer product is the unconditional correlation
        epsilon_outer = Q_bar;
    end
    
    % DCC recursion for forecast
    Qt_next = (1 - sum(a_params) - sum(b_params)) * Q_bar + ...
              sum(a_params) * epsilon_outer + ...
              sum(b_params) * Qt_forecast;
    
    % Normalize to get correlation matrix
    Rt_next = normalize_correlation(Qt_next);
    
    % Store forecasts
    corr_forecast(h, :, :) = Rt_next;
    
    % Create diagonal matrix of forecasted standard deviations
    D_next = diag(sqrt(h_forecast(h, :)));
    
    % H_t+h = D_t+h * R_t+h * D_t+h
    cov_forecast(h, :, :) = D_next * Rt_next * D_next;
    
    % Update Qt for next iteration
    Qt_forecast = Qt_next;
end

% Assemble output structure
forecast = struct();
forecast.h = h_forecast;
forecast.corr = corr_forecast;
forecast.cov = cov_forecast;
forecast.horizon = horizon;

end

%--------------------------------------------------------------------------
function h_forecast = forecast_univariate_garch(params, last_h, data, model_type, p, q, horizon)
% Helper function that generates forecasts for a univariate GARCH process
%
% INPUTS:
%   params      - Estimated parameters for the univariate GARCH model
%   last_h      - Last observed conditional variance
%   data        - Time series data for the univariate series
%   model_type  - String, type of GARCH model ('GARCH', 'EGARCH', etc.)
%   p           - GARCH order
%   q           - ARCH order
%   horizon     - Forecast horizon
%
% OUTPUTS:
%   h_forecast - HORIZON x 1 vector of conditional variance forecasts

% Initialize forecast array
h_forecast = zeros(horizon, 1);

% Extract parameters based on model type
if strcmp(model_type, 'GARCH')
    omega = params(1);
    alpha = params(2:q+1);
    beta = params(q+2:q+p+1);
    
    % Calculate unconditional variance
    uncond_var = omega / (1 - sum(alpha) - sum(beta));
    
    % Set up multi-step forecast
    for h = 1:horizon
        if h == 1
            % One-step ahead forecast
            last_resid_sq = data(end)^2;
            h_forecast(h) = omega + sum(alpha) * last_resid_sq + sum(beta) * last_h;
        else
            % Multi-step ahead forecast (using previous forecast)
            persistence = sum(alpha) + sum(beta);
            h_forecast(h) = omega + persistence * h_forecast(h-1);
            
            % For long horizons, the forecast approaches the unconditional variance
            if persistence < 0.999
                % Apply mean reverting correction
                h_forecast(h) = persistence^(h-1) * h_forecast(1) + (1 - persistence^(h-1)) * uncond_var;
            end
        end
    end
elseif strcmp(model_type, 'EGARCH')
    % EGARCH forecasting logic
    % For simplicity, using a simple persistence approach
    omega = params(1);
    alpha = params(2:q+1);
    gamma = params(q+2:2*q+1);
    beta = params(2*q+2:2*q+p+1);
    
    % Log variance dynamics for EGARCH
    log_last_h = log(last_h);
    log_h_forecast = zeros(horizon, 1);
    
    % Calculate unconditional log variance (approximate)
    uncond_log_var = omega / (1 - sum(beta));
    
    for h = 1:horizon
        if h == 1
            % One-step ahead uses observed data
            last_std_resid = data(end) / sqrt(last_h);
            g_z = alpha(1) * (abs(last_std_resid) - sqrt(2/pi)) + gamma(1) * last_std_resid;
            log_h_forecast(h) = omega + g_z + beta(1) * log_last_h;
        else
            % Multi-step forecast
            if h <= length(beta)
                log_h_forecast(h) = omega + sum(beta(1:min(h-1,length(beta)))) * log_h_forecast(1:min(h-1,length(beta)));
            else
                % Long-horizon forecast: mean reversion to unconditional log variance
                persistence = sum(beta);
                log_h_forecast(h) = persistence * log_h_forecast(h-1) + (1 - persistence) * uncond_log_var;
            end
        end
    end
    
    % Convert log variance to variance
    h_forecast = exp(log_h_forecast);
    
elseif strcmp(model_type, 'GJR') || strcmp(model_type, 'TARCH')
    % GJR/TARCH parameters
    omega = params(1);
    alpha = params(2:q+1);
    gamma = params(q+2:2*q+1);
    beta = params(2*q+2:2*q+p+1);
    
    % Determine if last shock was negative for the asymmetry term
    last_resid = data(end);
    last_resid_sq = last_resid^2;
    negative_shock = (last_resid < 0);
    
    % Effective alpha for negative shocks includes gamma
    alpha_effective = alpha;
    if negative_shock
        alpha_effective = alpha + gamma;
    end
    
    % Unconditional variance (approximation)
    uncond_var = omega / (1 - sum(alpha) - 0.5 * sum(gamma) - sum(beta));
    
    for h = 1:horizon
        if h == 1
            % One-step ahead forecast
            h_forecast(h) = omega + sum(alpha_effective) * last_resid_sq + sum(beta) * last_h;
        else
            % Multi-step ahead forecast
            % For GJR, expected future asymmetric term is 0.5*gamma (equal prob of positive/negative)
            persistence = sum(alpha) + 0.5 * sum(gamma) + sum(beta);
            h_forecast(h) = omega + persistence * h_forecast(h-1);
            
            % Mean-reversion for long horizons
            if persistence < 0.999
                h_forecast(h) = persistence^(h-1) * h_forecast(1) + (1 - persistence^(h-1)) * uncond_var;
            end
        end
    end
else
    % Default to simple exponential smoothing for unimplemented models
    h_forecast = last_h * ones(horizon, 1);
end

end

%--------------------------------------------------------------------------
function R = normalize_correlation(Q)
% NORMALIZE_CORRELATION Normalizes a quasi-correlation matrix to a proper correlation matrix
%
% USAGE:
%   R = normalize_correlation(Q)
%
% INPUTS:
%   Q - K x K matrix (quasi-correlation matrix)
%
% OUTPUTS:
%   R - K x K matrix (normalized correlation matrix with unit diagonal)
%
% COMMENTS:
%   This function normalizes a quasi-correlation matrix Q to ensure it is a 
%   valid correlation matrix with unit diagonal elements.
%   R_ij = Q_ij / sqrt(Q_ii * Q_jj)

% Get dimensions
K = size(Q, 1);

% Extract diagonal elements
q_diag = diag(Q);

% Create diagonal matrix of inverse square roots
D_inv = diag(1 ./ sqrt(q_diag));

% Normalize: R = D^(-1/2) * Q * D^(-1/2)
R = D_inv * Q * D_inv;

% Ensure diagonal elements are exactly 1 (correcting any numerical issues)
R(1:K+1:end) = 1;

% Ensure correlations are in [-1, 1] range due to numerical issues
R = min(max(R, -1), 1);

% Ensure the matrix is symmetric
R = (R + R') / 2;

end

%--------------------------------------------------------------------------
function validation = validate_dcc_model(model)
% VALIDATE_DCC_MODEL Validates estimated DCC-MVGARCH model for numerical stability and validity
%
% USAGE:
%   VALIDATION = validate_dcc_model(MODEL)
%
% INPUTS:
%   MODEL - Structure produced by dcc_mvgarch
%
% OUTPUTS:
%   VALIDATION - Structure containing validation results with fields:
%                validation.stationary - Boolean, true if model is stationary
%                validation.posdef - Boolean, true if all correlation matrices are positive definite
%                validation.valid_corr - Boolean, true if all correlations are valid
%                validation.issues - Cell array of identified issues
%
% COMMENTS:
%   This function performs diagnostic checks on an estimated DCC-MVGARCH model
%   to ensure numerical stability, parameter validity, and correlation constraints.

% Get dimensions
T = size(model.data, 1);
K = size(model.data, 2);

% Extract DCC parameters
dcc_params = model.parameters.dcc;
dcc_p = length(dcc_params) / 2;
a_params = dcc_params(1:dcc_p);
b_params = dcc_params(dcc_p+1:end);

% Initialize validation structure
validation = struct();
validation.issues = {};

% Check stationarity condition: a + b < 1
validation.stationary = (sum(a_params) + sum(b_params) < 1);
if ~validation.stationary
    validation.issues{end+1} = 'DCC model is not stationary: a + b >= 1';
end

% Check if all correlation matrices are valid
is_valid_corr = true;
is_posdef = true;

for t = 1:T
    % Extract correlation matrix for time t
    Rt = reshape(model.corr(t, :, :), K, K);
    
    % Check diagonal elements (should be 1)
    diag_Rt = diag(Rt);
    if any(abs(diag_Rt - 1) > 1e-6)
        is_valid_corr = false;
        validation.issues{end+1} = 'Some correlation matrices have diagonal elements not equal to 1';
        break;
    end
    
    % Check off-diagonal elements (should be in [-1, 1])
    if any(Rt(:) < -1-1e-6) || any(Rt(:) > 1+1e-6)
        is_valid_corr = false;
        validation.issues{end+1} = 'Some correlation coefficients are outside the [-1, 1] range';
        break;
    end
    
    % Check symmetry
    if norm(Rt - Rt', 'inf') > 1e-6
        is_valid_corr = false;
        validation.issues{end+1} = 'Some correlation matrices are not symmetric';
        break;
    end
    
    % Check positive definiteness (simplistic approach)
    % A more robust approach would use eigenvalues or Cholesky decomposition
    try
        chol(Rt);
    catch
        is_posdef = false;
        validation.issues{end+1} = 'Some correlation matrices are not positive definite';
        break;
    end
end

validation.valid_corr = is_valid_corr;
validation.posdef = is_posdef;

% Validate univariate GARCH models
is_uni_stationary = true;
for k = 1:K
    params = model.parameters.univariate{k};
    model_type = model.options.model;
    
    if strcmp(model_type, 'GARCH')
        p = model.options.p;
        q = model.options.q;
        
        % Extract GARCH parameters
        omega = params(1);
        alpha = params(2:q+1);
        beta = params(q+2:q+p+1);
        
        % Check stationarity condition
        if sum(alpha) + sum(beta) >= 1
            is_uni_stationary = false;
            validation.issues{end+1} = sprintf('Univariate GARCH model for series %d is not stationary', k);
        end
        
        % Check positivity of parameters
        if omega <= 0 || any(alpha < 0) || any(beta < 0)
            validation.issues{end+1} = sprintf('Univariate GARCH model for series %d has negative parameters', k);
        end
    elseif strcmp(model_type, 'EGARCH')
        % EGARCH has different stationarity conditions
        % Not implementing all checks for all models
    end
end

validation.univariate_stationary = is_uni_stationary;

% Overall model validity
validation.valid = validation.stationary && validation.posdef && ...
                   validation.valid_corr && validation.univariate_stationary;

end