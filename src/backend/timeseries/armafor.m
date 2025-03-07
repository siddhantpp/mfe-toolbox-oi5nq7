function [forecasts, variances, Paths] = armafor(parameters, data, p, q, constant, x, T, shocks, method, nsim, error_dist, dist_params)
% ARMAFOR Generates multi-step ahead forecasts for ARMA/ARMAX models.
%
% USAGE:
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS, METHOD)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS, METHOD, NSIM)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS, METHOD, NSIM, ERROR_DIST)
%   [FORECASTS, VARIANCES] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS, METHOD, NSIM, ERROR_DIST, DIST_PARAMS)
%   [FORECASTS, VARIANCES, PATHS] = armafor(PARAMETERS, DATA, P, Q, CONSTANT, X, T, SHOCKS, METHOD, NSIM, ERROR_DIST, DIST_PARAMS)
%
% INPUTS:
%   PARAMETERS - Vector of model parameters with the structure:
%                [ar(1),...,ar(p),ma(1),...,ma(q),x(1),...,x(k)]
%                where k is the number of exogenous variables
%   DATA       - Vector of data with the most recent observation last
%   P          - Positive integer representing the AR order
%   Q          - Positive integer representing the MA order
%   CONSTANT   - Boolean indicating whether the model includes a constant
%   X          - [OPTIONAL] Matrix (T x k) of forecasted exogenous variables
%                for the forecasting horizon, or empty
%   T          - [OPTIONAL] Number of steps to forecast (default: 1)
%   SHOCKS     - [OPTIONAL] Vector of shocks to use for initial forecast periods
%                If not provided, uses zeros (default: [])
%   METHOD     - [OPTIONAL] Method to use for generating the forecasts
%                'exact' - Use the exact method which uses expected values 
%                'simulation' - Use simulation-based method (default: 'exact')
%   NSIM       - [OPTIONAL] Number of simulations if METHOD='simulation' (default: 1000)
%   ERROR_DIST - [OPTIONAL] Error distribution for simulations (default: 'normal')
%                Supported options: 'normal', 'student', 'ged', 'skewt'
%   DIST_PARAMS - [OPTIONAL] Structure with distribution parameters:
%                 .nu - Degrees of freedom for 'student', 'ged', or 'skewt'
%                 .lambda - Skewness parameter for 'skewt'
%
% OUTPUTS:
%   FORECASTS - T x 1 vector of point forecasts
%   VARIANCES - T x 1 vector of forecast error variances
%   PATHS     - [OPTIONAL] T x nsim matrix of simulated forecast paths
%               (Only returned when METHOD = 'simulation')
%
% COMMENTS:
%   When using the 'exact' method, forecasts are computed recursively using
%   conditional expectations. For the 'simulation' method, forecasts are
%   generated by simulating multiple paths with random innovations drawn from
%   the specified error distribution.
%
%   The autoregressive order p can be zero, which gives an MA(q) model,
%   and the moving average order q can be zero, which gives an AR(p) model.
%
%   The model can include a constant term as well as exogenous variables.
%   For exogenous variables, the corresponding future values must be provided
%   in the X input.
%
% EXAMPLES:
%   % Generate forecasts for an AR(2) model
%   forecasts = armafor([0.5, 0.2], y, 2, 0, true);
%
%   % Generate forecasts for an ARMA(1,1) model with a constant
%   [forecasts, variances] = armafor([0.8, 0.3], y, 1, 1, true);
%
%   % Generate forecasts for an ARMAX(1,1) model with exogenous variables
%   [forecasts, variances] = armafor([0.8, 0.3, 0.5], y, 1, 1, true, x_future);
%
%   % Simulate 5000 forecast paths for a 10-period horizon with t-distributed errors
%   [forecasts, variances, paths] = armafor([0.8, 0.3], y, 1, 1, true, [], 10, [], 'simulation', 5000, 'student', struct('nu', 5));
%
% See also ARMAXFILTER, ARMAXERRORS

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input Validation
if nargin < 5
    error('At least 5 inputs are required (PARAMETERS, DATA, P, Q, CONSTANT)');
end

% Validate required inputs
parameters = datacheck(parameters, 'parameters');
parameters = columncheck(parameters, 'parameters');
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Validate AR and MA order parameters
p_options.isNonNegative = true;
p_options.isInteger = true;
p_options.isscalar = true;
p = parametercheck(p, 'p', p_options);

q_options.isNonNegative = true;
q_options.isInteger = true;
q_options.isscalar = true;
q = parametercheck(q, 'q', q_options);

% Process optional inputs and set defaults
if nargin < 6 || isempty(x)
    x = [];
    k = 0;
else
    x = datacheck(x, 'x');
    if size(x, 2) > 1
        % If x has multiple columns, ensure it's properly formatted
        k = size(x, 2);
    else
        x = columncheck(x, 'x');
        k = 1;
    end
end

% Determine if we have the right number of parameters
expected_params = p + q + k + constant;
if length(parameters) ~= expected_params
    error('Length of PARAMETERS must be p + q + k + constant = %d. Found %d parameters.', expected_params, length(parameters));
end

if nargin < 7 || isempty(T)
    T = 1;
else
    T_options.isscalar = true;
    T_options.isInteger = true;
    T_options.isPositive = true;
    T = parametercheck(T, 'T', T_options);
end

if nargin < 8 || isempty(shocks)
    shocks = [];
else
    shocks = datacheck(shocks, 'shocks');
    shocks = columncheck(shocks, 'shocks');
    if length(shocks) > q
        error('Length of SHOCKS must be <= q = %d', q);
    end
end

if nargin < 9 || isempty(method)
    method = 'exact';
elseif ~ischar(method)
    error('METHOD must be a string');
else
    valid_methods = {'exact', 'simulation'};
    if ~ismember(lower(method), valid_methods)
        error('METHOD must be one of: ''exact'', ''simulation''');
    end
    method = lower(method);
end

if nargin < 10 || isempty(nsim)
    nsim = 1000;
elseif strcmp(method, 'simulation')
    nsim_options.isscalar = true;
    nsim_options.isInteger = true;
    nsim_options.isPositive = true;
    nsim = parametercheck(nsim, 'nsim', nsim_options);
end

if nargin < 11 || isempty(error_dist)
    error_dist = 'normal';
elseif ~ischar(error_dist)
    error('ERROR_DIST must be a string');
else
    valid_dists = {'normal', 'student', 'ged', 'skewt'};
    if ~ismember(lower(error_dist), valid_dists)
        error('ERROR_DIST must be one of: ''normal'', ''student'', ''ged'', ''skewt''');
    end
    error_dist = lower(error_dist);
end

if nargin < 12
    dist_params = struct();
end

% Check that we have enough historical data
if length(data) < max(p, q)
    error('DATA must contain at least max(p,q) = %d observations', max(p, q));
end

% Check that we have sufficient future exogenous data if needed
if ~isempty(x) && size(x, 1) < T
    error('X must have at least T = %d rows to forecast %d periods ahead', T, T);
end

% Extract model parameters
if constant
    constant_value = parameters(1);
    ar_params = parameters(1 + (1:p));
    ma_params = parameters(1 + p + (1:q));
    if k > 0
        x_params = parameters(1 + p + q + (1:k));
    else
        x_params = [];
    end
else
    constant_value = 0;
    ar_params = parameters(1:p);
    ma_params = parameters(p + (1:q));
    if k > 0
        x_params = parameters(p + q + (1:k));
    else
        x_params = [];
    end
end

% Compute residuals from historical data for use in forecasting
residuals = armaxerrors(data, ar_params, ma_params, x_params, [], constant_value);

%% Generate Forecasts
if strcmp(method, 'exact')
    % Use exact method for forecasting
    [forecasts, variances] = compute_exact_forecasts(ar_params, ma_params, x_params, data, residuals, x, constant, constant_value, T);
    Paths = [];  % No paths for exact method
elseif strcmp(method, 'simulation')
    % Use simulation-based forecasting
    [forecasts, variances, Paths] = simulate_forecast_paths(ar_params, ma_params, x_params, data, residuals, x, constant, constant_value, T, nsim, error_dist, dist_params);
end

% Only return Paths if explicitly requested
if nargout < 3
    Paths = [];
end

end

%% Helper Functions
function [forecasts, variances] = compute_exact_forecasts(ar_params, ma_params, x_params, data, residuals, x_future, has_constant, constant_value, forecast_horizon)
% Compute exact forecasts and forecast error variances for the specified ARMA/ARMAX model

% Initialize arrays for forecasts and variances
forecasts = zeros(forecast_horizon, 1);
variances = zeros(forecast_horizon, 1);

% Get dimensions
n = length(data);
p = length(ar_params);
q = length(ma_params);
if ~isempty(x_params)
    k = length(x_params);
else
    k = 0;
end

% Initialize array for residual coefficients (for variance computation)
res_coef = ones(forecast_horizon, 1);

% Set up extended data array to hold actual data and forecasts
extended_data = [data; zeros(forecast_horizon, 1)];

% For each forecast period
for t = 1:forecast_horizon
    % Start with constant if present
    forecast_value = 0;
    if has_constant
        forecast_value = constant_value;
    end
    
    % Add AR component - use actual data when available, forecasts otherwise
    for i = 1:p
        if (n + t - i) <= n
            % Using historical data
            forecast_value = forecast_value + ar_params(i) * data(n + t - i);
        else
            % Using previous forecasts
            forecast_value = forecast_value + ar_params(i) * extended_data(n + t - i);
        end
    end
    
    % Add MA component - uses historical residuals, zeros for future periods
    for i = 1:q
        if (n + t - i) <= n
            % Using historical residuals
            forecast_value = forecast_value + ma_params(i) * residuals(n + t - i);
        end
        % Future residuals are set to their expected value (zero)
    end
    
    % Add exogenous component if present
    if k > 0 && ~isempty(x_future)
        for j = 1:k
            forecast_value = forecast_value + x_params(j) * x_future(t, j);
        end
    end
    
    % Store forecast
    forecasts(t) = forecast_value;
    extended_data(n + t) = forecast_value;
    
    % Compute variance coefficients for MA polynomial
    % For h-step ahead forecasts, the variance contribution is:
    % σ² * (1 + θ₁² + θ₂² + ... + θ_{h-1}²)
    if t == 1
        % For 1-step ahead, variance is just σ²
        variances(t) = 1;
    else
        % Compute MA coefficients for error accumulation
        % This is effectively calculating the h-step ahead MA representation
        % through recursion on the ARMA model
        for s = 1:(t-1)
            coef = 0;
            for i = 1:min(s, p)
                if (s - i) < t
                    coef = coef + ar_params(i) * res_coef(s - i + 1);
                end
            end
            
            if s <= q
                coef = coef + ma_params(s);
            end
            
            res_coef(s + 1) = coef;
        end
        
        % Compute the variance as σ² times the sum of squared MA coefficients
        variances(t) = sum(res_coef(1:t).^2);
    end
end

end

function [forecasts, variances, Paths] = simulate_forecast_paths(ar_params, ma_params, x_params, data, residuals, x_future, has_constant, constant_value, forecast_horizon, num_simulations, error_distribution, dist_parameters)
% Generate forecast paths through simulation for the specified ARMA/ARMAX model

% Get dimensions
n = length(data);
p = length(ar_params);
q = length(ma_params);
if ~isempty(x_params)
    k = length(x_params);
else
    k = 0;
end

% Initialize simulation paths array
Paths = zeros(forecast_horizon, num_simulations);

% Generate random innovations based on the specified distribution
switch error_distribution
    case 'normal'
        % Standard normal innovations
        innovations = randn(forecast_horizon, num_simulations);
        
    case 'student'
        % Check for degrees of freedom parameter
        if ~isfield(dist_parameters, 'nu')
            error('Degrees of freedom (nu) must be provided for Student''s t distribution');
        end
        nu = dist_parameters.nu;
        
        % Generate standardized Student's t innovations
        innovations = stdtrnd([forecast_horizon, num_simulations], nu);
        
    case 'ged'
        % Check for shape parameter
        if ~isfield(dist_parameters, 'nu')
            error('Shape parameter (nu) must be provided for GED distribution');
        end
        nu = dist_parameters.nu;
        
        % Generate GED innovations
        innovations = gedrnd(nu, forecast_horizon, num_simulations);
        
    case 'skewt'
        % Check for required parameters
        if ~isfield(dist_parameters, 'nu') || ~isfield(dist_parameters, 'lambda')
            error('Both degrees of freedom (nu) and skewness (lambda) must be provided for skewed t distribution');
        end
        nu = dist_parameters.nu;
        lambda = dist_parameters.lambda;
        
        % Generate skewed t innovations
        innovations = skewtrnd(nu, lambda, forecast_horizon, num_simulations);
        
    otherwise
        error('Unsupported error distribution: %s', error_distribution);
end

% Scale innovations by the model's residual standard deviation
% Note: innovations are already standardized with unit variance
std_resid = std(residuals);
innovations = innovations * std_resid;

% For each simulation
for sim = 1:num_simulations
    % Set up extended data array for this simulation
    extended_data = [data; zeros(forecast_horizon, 1)];
    
    % Set up extended residuals array
    extended_residuals = [residuals; zeros(forecast_horizon, 1)];
    
    % For each forecast period
    for t = 1:forecast_horizon
        % Start with constant if present
        forecast_value = 0;
        if has_constant
            forecast_value = constant_value;
        end
        
        % Add AR component
        for i = 1:p
            if (n + t - i) <= n
                % Using historical data
                forecast_value = forecast_value + ar_params(i) * data(n + t - i);
            else
                % Using previous forecasts in this simulation
                forecast_value = forecast_value + ar_params(i) * extended_data(n + t - i);
            end
        end
        
        % Add MA component
        for i = 1:q
            if (n + t - i) <= n
                % Using historical residuals
                forecast_value = forecast_value + ma_params(i) * residuals(n + t - i);
            else
                % Using simulated residuals from this path
                forecast_value = forecast_value + ma_params(i) * extended_residuals(n + t - i);
            end
        end
        
        % Add exogenous component if present
        if k > 0 && ~isempty(x_future)
            for j = 1:k
                forecast_value = forecast_value + x_params(j) * x_future(t, j);
            end
        end
        
        % Add random innovation for this period
        current_innovation = innovations(t, sim);
        forecast_value = forecast_value + current_innovation;
        
        % Store forecast and residual
        extended_data(n + t) = forecast_value;
        extended_residuals(n + t) = current_innovation;
        
        % Store in the paths matrix
        Paths(t, sim) = forecast_value;
    end
end

% Compute point forecasts as the mean across simulations
forecasts = mean(Paths, 2);

% Compute variances as the variance across simulations
variances = var(Paths, 0, 2);  % 0 means normalize by N not N-1

end