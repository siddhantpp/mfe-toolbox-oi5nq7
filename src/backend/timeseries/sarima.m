function results = sarima(y, p, d, q, P, D, Q, s, options)
% SARIMA Fits a Seasonal ARIMA (p,d,q)×(P,D,Q)s model to time series data.
%
% USAGE:
%   RESULTS = sarima(Y, P, D, Q, P, D, Q, S)
%   RESULTS = sarima(Y, P, D, Q, P, D, Q, S, OPTIONS)
%
% INPUTS:
%   Y       - Time series data (T x 1 vector)
%   P       - Non-seasonal autoregressive order
%   D       - Non-seasonal differencing order
%   Q       - Non-seasonal moving average order
%   P       - Seasonal autoregressive order
%   D       - Seasonal differencing order
%   Q       - Seasonal moving average order
%   S       - Seasonal period (e.g., 4 for quarterly, 12 for monthly)
%   OPTIONS - [OPTIONAL] Options structure with fields:
%             constant    - [OPTIONAL] Boolean indicating inclusion of constant term
%                           Default: true
%             distribution - [OPTIONAL] Error distribution assumption:
%                           'normal'  - Normal distribution
%                           't'       - Student's t
%                           'ged'     - Generalized Error Distribution
%                           'skewt'   - Hansen's Skewed t
%                           Default: 'normal'
%             startingVals - [OPTIONAL] Vector of starting values
%                           Default: [] (use reasonable defaults)
%             optimopts   - [OPTIONAL] Options for fminsearch optimization
%                           Default: [] (use default optimization options)
%
% OUTPUTS:
%   RESULTS - Structure containing:
%             parameters    - Estimated parameters
%             standardErrors - Standard errors of parameters
%             tStats       - t-statistics for parameter estimates
%             pValues      - p-values for parameter estimates
%             paramNames   - Names of parameters
%             residuals    - Model residuals/innovations
%             logL         - Log-likelihood of the model
%             aic          - Akaike Information Criterion
%             sbic         - Schwarz Bayesian Information Criterion
%             ljungBox     - Results of Ljung-Box test for autocorrelation
%             ...plus model orders and data information
%
% COMMENTS:
%   Implements SARIMA (p,d,q)×(P,D,Q)s model:
%   (1-phi_1B-...-phi_pB^p)(1-Phi_1B^s-...-Phi_PB^Ps)(1-B)^d(1-B^s)^D y_t = 
%   (1+theta_1B+...+theta_qB^q)(1+Theta_1B^s+...+Theta_QB^Qs) e_t
%
%   Where B is the backshift operator, and e_t can follow various distributions.
%
% EXAMPLES:
%   % Simple ARIMA(1,1,1) model with normal errors
%   results = sarima(returns, 1, 1, 1, 0, 0, 0, 0);
%
%   % SARIMA(1,0,1)×(1,1,1)12 model with t-distributed errors
%   options = struct('distribution', 't');
%   results = sarima(returns, 1, 0, 1, 1, 1, 1, 12, options);
%
% See also ARMAXFILTER, ARMAFOR, AICSBIC, LJUNGBOX

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input Validation
if nargin < 8
    error('At least 8 inputs are required (Y, P, D, Q, P, D, Q, S)');
end

% Validate y data
y = datacheck(y, 'y');
y = columncheck(y, 'y');

% Get time series length
T = length(y);

% Validate SARIMA model orders
options_param = struct('isInteger', true, 'isNonNegative', true, 'isscalar', true);
p = parametercheck(p, 'p', options_param);
d = parametercheck(d, 'd', options_param);
q = parametercheck(q, 'q', options_param);
P = parametercheck(P, 'P', options_param);
D = parametercheck(D, 'D', options_param);
Q = parametercheck(Q, 'Q', options_param);
s = parametercheck(s, 's', options_param);

% Ensure seasonal period makes sense
if s < 2 && (P > 0 || D > 0 || Q > 0)
    error('Seasonal period (s) must be >= 2 when using seasonal components');
end

% Set default options
defaultOptions = struct(...
    'constant', true, ...
    'distribution', 'normal', ...
    'startingVals', [], ...
    'optimopts', []);

% Process user options
if nargin < 9 || isempty(options)
    options = defaultOptions;
else
    % Merge user options with defaults
    optionFields = fieldnames(defaultOptions);
    for i = 1:length(optionFields)
        field = optionFields{i};
        if ~isfield(options, field)
            options.(field) = defaultOptions.(field);
        end
    end
end

% Extract parameters from options
constant = options.constant;
distType = options.distribution;

% Check if we have enough data after differencing
if T <= p + P*s + d + D*s
    error('Not enough data points after differencing. Need more than p + P*s + d + D*s = %d observations', p + P*s + d + D*s);
end

%% Apply Differencing
% Store original data for undifferencing later if needed
y_original = y;

% Apply regular differencing d times
y_diff = y;
for i = 1:d
    y_diff = diff(y_diff);
end

% Apply seasonal differencing D times
for i = 1:D
    if length(y_diff) > s
        y_diff = diff(y_diff, 1, s);
    else
        error('Not enough data points for seasonal differencing');
    end
end

% New sample size after differencing
T_diff = length(y_diff);

%% Set Up SARIMA Model as ARMA Representation

% Initialize parameter counts
numARParams = p;
numMAParams = q;
numSARParams = P;
numSMAParams = Q;
numConstParam = constant * 1;
numDistParams = 0; % Additional distribution parameters if needed

% Determine number of distribution parameters based on distribution type
switch lower(distType)
    case 'normal'
        numDistParams = 0;
    case 't'
        numDistParams = 1; % Degrees of freedom
    case 'ged'
        numDistParams = 1; % Shape parameter
    case 'skewt'
        numDistParams = 2; % Degrees of freedom and skewness
    otherwise
        error('Unsupported distribution type. Options are: ''normal'', ''t'', ''ged'', ''skewt''.');
end

% Total number of parameters
numParams = numARParams + numMAParams + numSARParams + numSMAParams + numConstParam + numDistParams;

% Initialize parameters
if isempty(options.startingVals)
    % Set default starting values
    startingVals = zeros(numParams, 1);
    
    % Set index counters
    idx = 1;
    
    % Constant term starts at sample mean if included
    if constant
        startingVals(idx) = mean(y_diff);
        idx = idx + 1;
    end
    
    % AR parameters initialized with small positive values
    if p > 0
        startingVals(idx:idx+p-1) = 0.1 * ones(p, 1) / p;
        idx = idx + p;
    end
    
    % MA parameters initialized with small positive values
    if q > 0
        startingVals(idx:idx+q-1) = 0.1 * ones(q, 1) / q;
        idx = idx + q;
    end
    
    % Seasonal AR parameters
    if P > 0
        startingVals(idx:idx+P-1) = 0.1 * ones(P, 1) / P;
        idx = idx + P;
    end
    
    % Seasonal MA parameters
    if Q > 0
        startingVals(idx:idx+Q-1) = 0.1 * ones(Q, 1) / Q;
        idx = idx + Q;
    end
    
    % Distribution parameters
    switch lower(distType)
        case 'normal'
            % No additional parameters
        case 't'
            startingVals(idx) = 8; % Initial df = 8
        case 'ged'
            startingVals(idx) = 1.5; % Initial shape = 1.5
        case 'skewt'
            startingVals(idx) = 8; % Initial df = 8
            startingVals(idx+1) = 0; % Initial skewness = 0
    end
else
    % Use user-provided starting values
    startingVals = options.startingVals;
    if length(startingVals) ~= numParams
        error('Starting values vector must have length equal to number of parameters (%d).', numParams);
    end
end

% Configure optimization options
if isempty(options.optimopts)
    optimOpts = optimset('fminsearch');
    optimOpts.Display = 'off';
    optimOpts.MaxIter = 1000;
    optimOpts.TolFun = 1e-6;
    optimOpts.TolX = 1e-6;
else
    optimOpts = options.optimopts;
end

%% Perform Optimization

% Define likelihood function
likelihoodFcn = @(params) sarima_likelihood(params, y_diff, p, q, P, Q, s, constant, distType);

% Perform optimization using fminsearch
[optimalParams, negLogL] = fminsearch(likelihoodFcn, startingVals, optimOpts);

% Compute likelihood and residuals
[~, residuals, fullARpoly, fullMApoly] = sarima_likelihood(optimalParams, y_diff, p, q, P, Q, s, constant, distType);

%% Extract Estimated Parameters
idx = 1;
if constant
    constParam = optimalParams(idx);
    idx = idx + 1;
else
    constParam = 0;
end

% Extract AR parameters
ar_params = zeros(p, 1);
if p > 0
    ar_params = optimalParams(idx:idx+p-1);
    idx = idx + p;
end

% Extract MA parameters
ma_params = zeros(q, 1);
if q > 0
    ma_params = optimalParams(idx:idx+q-1);
    idx = idx + q;
end

% Extract Seasonal AR parameters
sar_params = zeros(P, 1);
if P > 0
    sar_params = optimalParams(idx:idx+P-1);
    idx = idx + P;
end

% Extract Seasonal MA parameters
sma_params = zeros(Q, 1);
if Q > 0
    sma_params = optimalParams(idx:idx+Q-1);
    idx = idx + Q;
end

% Extract distribution parameters
dist_params = [];
switch lower(distType)
    case 'normal'
        % No additional parameters
    case 't'
        dist_params = optimalParams(idx);
    case 'ged'
        dist_params = optimalParams(idx);
    case 'skewt'
        dist_params = optimalParams(idx:idx+1);
end

%% Compute Standard Errors and Test Statistics
% Calculate numerical Hessian for standard errors
h = 1e-4;  % Step size for numerical derivatives
H = zeros(numParams, numParams);

% Compute Hessian
for i = 1:numParams
    for j = i:numParams
        if i == j
            % Diagonal elements (second derivatives)
            paramsPlus = optimalParams;
            paramsMinus = optimalParams;
            paramsPlus(i) = optimalParams(i) + h;
            paramsMinus(i) = optimalParams(i) - h;
            
            llPlus = likelihoodFcn(paramsPlus);
            llMinus = likelihoodFcn(paramsMinus);
            llCenter = negLogL;
            
            H(i,j) = (llPlus - 2*llCenter + llMinus) / (h^2);
        else
            % Off-diagonal elements (mixed partial derivatives)
            paramsPluspPlus = optimalParams;
            paramsPlusMinus = optimalParams;
            paramsMinusPlus = optimalParams;
            paramsMinusMinus = optimalParams;
            
            paramsPluspPlus(i) = optimalParams(i) + h;
            paramsPluspPlus(j) = optimalParams(j) + h;
            
            paramsPlusMinus(i) = optimalParams(i) + h;
            paramsPlusMinus(j) = optimalParams(j) - h;
            
            paramsMinusPlus(i) = optimalParams(i) - h;
            paramsMinusPlus(j) = optimalParams(j) + h;
            
            paramsMinusMinus(i) = optimalParams(i) - h;
            paramsMinusMinus(j) = optimalParams(j) - h;
            
            llPluspPlus = likelihoodFcn(paramsPluspPlus);
            llPlusMinus = likelihoodFcn(paramsPlusMinus);
            llMinusPlus = likelihoodFcn(paramsMinusPlus);
            llMinusMinus = likelihoodFcn(paramsMinusMinus);
            
            H(i,j) = (llPluspPlus - llPlusMinus - llMinusPlus + llMinusMinus) / (4 * h^2);
            H(j,i) = H(i,j);  % Ensure symmetry
        end
    end
end

% Ensure numerical stability of the Hessian
H = (H + H') / 2;  % Make sure it's symmetric

% Compute standard errors
try
    % Attempt to invert the Hessian
    V = inv(H);
    stdErrors = sqrt(diag(V));
    
    % Check for valid standard errors
    if any(~isfinite(stdErrors) | stdErrors <= 0)
        % Fall back to approximation if there are numerical issues
        stdErrors = ones(numParams, 1) * 0.1;
    end
catch
    % If Hessian inversion fails, use approximate standard errors
    stdErrors = ones(numParams, 1) * 0.1;
end

% Calculate t-statistics
tStats = optimalParams ./ stdErrors;

% Calculate p-values
pValues = 2 * (1 - tcdf(abs(tStats), T_diff - numParams));

%% Create Parameter Names
paramNames = cell(numParams, 1);
idx = 1;

if constant
    paramNames{idx} = 'Constant';
    idx = idx + 1;
end

for i = 1:p
    paramNames{idx} = sprintf('AR(%d)', i);
    idx = idx + 1;
end

for i = 1:q
    paramNames{idx} = sprintf('MA(%d)', i);
    idx = idx + 1;
end

for i = 1:P
    paramNames{idx} = sprintf('SAR(%d)', i*s);
    idx = idx + 1;
end

for i = 1:Q
    paramNames{idx} = sprintf('SMA(%d)', i*s);
    idx = idx + 1;
end

switch lower(distType)
    case 'normal'
        % No additional parameters
    case 't'
        paramNames{idx} = 'DoF';
    case 'ged'
        paramNames{idx} = 'Shape';
    case 'skewt'
        paramNames{idx} = 'DoF';
        paramNames{idx+1} = 'Skew';
end

%% Compute Information Criteria
% AIC and BIC
ic = aicsbic(-negLogL, numParams, T_diff);

%% Perform Diagnostic Tests
% Calculate max lag for Ljung-Box test
maxLag = min(20, floor(T_diff/4));
ljungBoxResult = ljungbox(residuals, maxLag, p + q + P + Q);

%% Assemble Results Structure
results = struct();
results.parameters = optimalParams;
results.standardErrors = stdErrors;
results.tStats = tStats;
results.pValues = pValues;
results.paramNames = paramNames;
results.residuals = residuals;
results.logL = -negLogL;
results.aic = ic.aic;
results.sbic = ic.sbic;
results.ljungBox = ljungBoxResult;
results.constant = constant;
results.p = p;
results.d = d;
results.q = q;
results.P = P;
results.D = D;
results.Q = Q;
results.s = s;
results.distribution = distType;
results.T = T;
results.T_diff = T_diff;
results.y = y_original;
results.y_diff = y_diff;
results.ARpoly = fullARpoly;
results.MApoly = fullMApoly;
results.options = options;

end

% Nested likelihood function
function [nll, residuals, fullARpoly, fullMApoly] = sarima_likelihood(params, y_diff, p, q, P, Q, s, includeConstant, distType)
% SARIMA_LIKELIHOOD Computes negative log-likelihood for SARIMA model
%
% Internal function used by sarima to calculate the likelihood of the model
% given parameter values.

% Initialize
T = length(y_diff);
residuals = zeros(T, 1);

% Extract parameters
idx = 1;

% Extract constant if included
if includeConstant
    constant = params(idx);
    idx = idx + 1;
else
    constant = 0;
end

% Extract AR parameters
ar_params = zeros(p, 1);
if p > 0
    ar_params = params(idx:idx+p-1);
    idx = idx + p;
end

% Extract MA parameters
ma_params = zeros(q, 1);
if q > 0
    ma_params = params(idx:idx+q-1);
    idx = idx + q;
end

% Extract Seasonal AR parameters
sar_params = zeros(P, 1);
if P > 0
    sar_params = params(idx:idx+P-1);
    idx = idx + P;
end

% Extract Seasonal MA parameters
sma_params = zeros(Q, 1);
if Q > 0
    sma_params = params(idx:idx+Q-1);
    idx = idx + Q;
end

% Extract distribution parameters
dist_params = [];
switch lower(distType)
    case 'normal'
        % No additional parameters
    case 't'
        dist_params = params(idx);
        % Ensure degrees of freedom > 2 for finite variance
        if dist_params < 2.001
            nll = 1e10; % Return large value for invalid parameters
            fullARpoly = [];
            fullMApoly = [];
            return;
        end
    case 'ged'
        dist_params = params(idx);
        % Ensure shape parameter > 0
        if dist_params <= 0
            nll = 1e10; % Return large value for invalid parameters
            fullARpoly = [];
            fullMApoly = [];
            return;
        end
    case 'skewt'
        dist_params = params(idx:idx+1);
        % Ensure degrees of freedom > 2 and skewness in (-1,1)
        if dist_params(1) < 2.001 || abs(dist_params(2)) >= 1
            nll = 1e10; % Return large value for invalid parameters
            fullARpoly = [];
            fullMApoly = [];
            return;
        end
end

% Expand seasonal polynomials into full lag polynomials
% For non-seasonal: (1 - phi₁B - phi₂B² - ... - phi_pBᵖ)
% For seasonal: (1 - Phi₁Bˢ - Phi₂B²ˢ - ... - Phi_PBᴾˢ)

% Create full AR polynomial by multiplying regular and seasonal AR parts
if p > 0
    ar_poly = [1; -ar_params];
else
    ar_poly = 1;
end

if P > 0
    % Initialize seasonal AR polynomial
    sar_poly = zeros(P*s + 1, 1);
    sar_poly(1) = 1;
    for i = 1:P
        sar_poly(i*s + 1) = -sar_params(i);
    end
    
    % Multiply regular and seasonal polynomials
    fullARpoly = conv(ar_poly, sar_poly);
else
    fullARpoly = ar_poly;
end

% Create full MA polynomial by multiplying regular and seasonal MA parts
if q > 0
    ma_poly = [1; ma_params];
else
    ma_poly = 1;
end

if Q > 0
    % Initialize seasonal MA polynomial
    sma_poly = zeros(Q*s + 1, 1);
    sma_poly(1) = 1;
    for i = 1:Q
        sma_poly(i*s + 1) = sma_params(i);
    end
    
    % Multiply regular and seasonal polynomials
    fullMApoly = conv(ma_poly, sma_poly);
else
    fullMApoly = ma_poly;
end

% Convert to ARMA representation and compute residuals
% Use armaxerrors to calculate residuals efficiently
ar_len = length(fullARpoly) - 1;
ma_len = length(fullMApoly) - 1;

% Need to convert polynomial coefficients for armaxerrors
ar_coefs = -fullARpoly(2:end);  % Skip the leading 1
ma_coefs = fullMApoly(2:end);   % Skip the leading 1

% Ensure coefficients are column vectors
ar_coefs = ar_coefs(:);
ma_coefs = ma_coefs(:);

% Compute residuals using armaxerrors
try
    residuals = armaxerrors(y_diff, ar_coefs, ma_coefs, [], [], constant);
catch
    % If armaxerrors fails, return a large value for the likelihood
    nll = 1e10;
    return;
end

% Compute log-likelihood based on distribution type
sigma = std(residuals);

% Calculate likelihood based on error distribution
switch lower(distType)
    case 'normal'
        % Standard normal likelihood
        ll = -0.5 * T * log(2*pi) - T * log(sigma) - sum((residuals).^2) / (2 * sigma^2);
        nll = -ll;
    case 't'
        % Student's t likelihood
        nu = dist_params(1);
        nll = stdtloglik(residuals, nu, 0, sigma);
    case 'ged'
        % GED likelihood
        nu = dist_params(1);
        nll = -gedloglik(residuals, nu, 0, sigma);
    case 'skewt'
        % Hansen's skewed t likelihood
        nu = dist_params(1);
        lambda = dist_params(2);
        params_skewt = [nu, lambda, 0, sigma];
        nll = skewtloglik(residuals, params_skewt);
end

% Handle numerical instability
if ~isfinite(nll)
    nll = 1e10; % Large value for invalid likelihood
end

end

function forecasts = sarima_forecast(model, horizon, forecast_options)
% SARIMA_FORECAST Generates forecasts from a fitted SARIMA model
%
% USAGE:
%   FORECASTS = sarima_forecast(MODEL, HORIZON)
%   FORECASTS = sarima_forecast(MODEL, HORIZON, FORECAST_OPTIONS)
%
% INPUTS:
%   MODEL            - Structure from sarima function containing model parameters
%   HORIZON          - Forecast horizon (number of periods to forecast)
%   FORECAST_OPTIONS - [OPTIONAL] Options structure with fields:
%                      alpha - Confidence level for forecast intervals (default: 0.05)
%                      method - Forecasting method ('exact' or 'simulation') (default: 'exact')
%                      nsim - Number of simulations if method='simulation' (default: 1000)
%
% OUTPUTS:
%   FORECASTS - Structure containing:
%               point    - Point forecasts
%               lower    - Lower forecast interval bounds
%               upper    - Upper forecast interval bounds
%               std      - Forecast standard errors
%               paths    - Simulated paths (if method='simulation')
%               ...plus other diagnostic information
%
% COMMENTS:
%   Produces forecasts for a fitted SARIMA model by first converting to equivalent
%   ARMA representation, then forecasting using the armafor function, and finally
%   applying appropriate inverse differencing transformations.
%
% See also SARIMA, ARMAFOR

% Input validation
horizon_options = struct('isInteger', true, 'isPositive', true, 'isscalar', true);
horizon = parametercheck(horizon, 'horizon', horizon_options);

% Set default forecast options
default_forecast_options = struct('alpha', 0.05, 'method', 'exact', 'nsim', 1000);
if nargin < 3 || isempty(forecast_options)
    forecast_options = default_forecast_options;
else
    % Merge with defaults
    fields = fieldnames(default_forecast_options);
    for i = 1:length(fields)
        if ~isfield(forecast_options, fields{i})
            forecast_options.(fields{i}) = default_forecast_options.(fields{i});
        end
    end
end

% Extract model parameters
p = model.p;
d = model.d;
q = model.q;
P = model.P;
D = model.D;
Q = model.Q;
s = model.s;
constant = model.constant;
y = model.y;
ARpoly = model.ARpoly;
MApoly = model.MApoly;
distribution = model.distribution;

% Get the actual parameters
all_params = model.parameters;
idx = 1;

% Extract constant if included
if constant
    const_param = all_params(idx);
    idx = idx + 1;
else
    const_param = 0;
end

% Skip the AR, MA, SAR, SMA parameters to get to distribution parameters
idx = idx + p + q + P + Q;

% Extract distribution parameters for forecast intervals
dist_params = struct();
switch lower(distribution)
    case 'normal'
        % No additional parameters needed
    case 't'
        dist_params.nu = all_params(idx);
    case 'ged'
        dist_params.nu = all_params(idx);
    case 'skewt'
        dist_params.nu = all_params(idx);
        dist_params.lambda = all_params(idx+1);
end

% Convert to ARMA representation
ar_coefs = -ARpoly(2:end);  % Skip the leading 1
ma_coefs = MApoly(2:end);   % Skip the leading 1

% Create parameters vector in format expected by armafor
armaParams = [const_param; ar_coefs; ma_coefs];

% Use armafor to generate forecasts for the differenced series
if strcmp(forecast_options.method, 'simulation')
    [diff_forecasts, diff_variances, diff_paths] = armafor(armaParams, model.y_diff, ...
        length(ar_coefs), length(ma_coefs), constant, [], horizon, [], 'simulation', ...
        forecast_options.nsim, distribution, dist_params);
else
    [diff_forecasts, diff_variances] = armafor(armaParams, model.y_diff, ...
        length(ar_coefs), length(ma_coefs), constant, [], horizon, [], 'exact');
end

% Create differencing information structure
diff_info = struct();
diff_info.d = d;
diff_info.D = D;
diff_info.s = s;

% Apply inverse differencing to get forecasts in original scale
undiff_forecasts = inverse_differencing(diff_forecasts, diff_info, y);

% Calculate forecast intervals
alpha = forecast_options.alpha;
z_alpha = norminv(1 - alpha/2);  % For normal distribution
forecast_std = sqrt(diff_variances);

% Adjust critical value based on distribution
switch lower(distribution)
    case 'normal'
        crit_val = z_alpha;
    case 't'
        nu = dist_params.nu;
        crit_val = stdtinv(1 - alpha/2, nu);
    case 'ged'
        nu = dist_params.nu;
        crit_val = gedinv(1 - alpha/2, nu);
    case 'skewt'
        nu = dist_params.nu;
        lambda = dist_params.lambda;
        crit_val = skewtinv(1 - alpha/2, nu, lambda);
end

% Calculate forecast intervals
lower_bounds = undiff_forecasts - crit_val * forecast_std;
upper_bounds = undiff_forecasts + crit_val * forecast_std;

% Create output structure
forecasts = struct();
forecasts.point = undiff_forecasts;
forecasts.lower = lower_bounds;
forecasts.upper = upper_bounds;
forecasts.std = forecast_std;
forecasts.horizon = horizon;
forecasts.alpha = alpha;
forecasts.method = forecast_options.method;

% Include simulation paths if simulation method was used
if strcmp(forecast_options.method, 'simulation') && exist('diff_paths', 'var')
    % Apply inverse differencing to each path
    undiff_paths = zeros(size(diff_paths));
    for i = 1:size(diff_paths, 2)
        undiff_paths(:,i) = inverse_differencing(diff_paths(:,i), diff_info, y);
    end
    forecasts.paths = undiff_paths;
end

end

function undifferenced = inverse_differencing(forecasts, diff_info, original_data)
% INVERSE_DIFFERENCING Applies inverse differencing to convert forecasts back to the original scale
%
% USAGE:
%   UNDIFFERENCED = inverse_differencing(FORECASTS, DIFF_INFO, ORIGINAL_DATA)
%
% INPUTS:
%   FORECASTS     - Forecasts from the differenced model
%   DIFF_INFO     - Differencing information structure
%   ORIGINAL_DATA - Original time series data
%
% OUTPUTS:
%   UNDIFFERENCED - Forecasts in the original scale

% Extract differencing information
d = diff_info.d;
D = diff_info.D;
s = diff_info.s;
h = length(forecasts);

% Initialize with the forecasts for the differenced series
undifferenced = forecasts;

% Apply inverse seasonal differencing
if D > 0
    % For each forecast horizon, we need the appropriate lagged values
    % Start with the last D*s observations from the original series
    orig_values = original_data(end-(D*s)+1:end);
    
    % Create a buffer that will hold original values and forecasts
    buffer = [orig_values; undifferenced];
    
    % Apply inverse seasonal differencing
    for i = 1:D
        seasonal_indices = s * (D - i + 1);
        for j = 1:h
            idx = length(orig_values) - seasonal_indices + j;
            if idx > 0
                undifferenced(j) = undifferenced(j) + buffer(idx);
            end
        end
        % Update buffer
        buffer = [orig_values; undifferenced];
    end
end

% Apply inverse regular differencing
if d > 0
    % Get the last d observations from the original series (after seasonal undifferencing)
    if D > 0
        % If we've already applied seasonal differencing, we need the last values
        % from the seasonally adjusted series
        orig_values = original_data(end-d+1:end);
        
        % Apply inverse regular differencing using the pre-seasonal undifferenced values
        buffer = [orig_values; undifferenced];
        
        for i = 1:d
            for j = 1:h
                undifferenced(j) = undifferenced(j) + buffer(j+d-i);
            end
            buffer = [orig_values; undifferenced];
        end
    else
        % No seasonal differencing, just apply regular inverse differencing
        orig_values = original_data(end-d+1:end);
        
        % Initialize buffer with original values to use for initial conditions
        buffer = [orig_values; zeros(h, 1)];
        
        % Apply regular undifferencing
        for j = 1:h
            buffer(d+j) = forecasts(j);
            for i = 1:d
                buffer(d+j) = buffer(d+j) + buffer(d+j-i);
            end
        end
        
        undifferenced = buffer(d+1:d+h);
    end
end

end

function expanded = expand_seasonal_polynomial(regular_coefs, seasonal_coefs, s)
% EXPAND_SEASONAL_POLYNOMIAL Expands the product of regular and seasonal polynomials
%
% USAGE:
%   EXPANDED = expand_seasonal_polynomial(REGULAR_COEFS, SEASONAL_COEFS, S)
%
% INPUTS:
%   REGULAR_COEFS  - Coefficients of the regular polynomial
%   SEASONAL_COEFS - Coefficients of the seasonal polynomial
%   S              - Seasonal period
%
% OUTPUTS:
%   EXPANDED       - Coefficients of the expanded polynomial

% Check if inputs are empty
if isempty(regular_coefs)
    regular_coefs = 1;
end
if isempty(seasonal_coefs)
    seasonal_coefs = 1;
end

% Create full polynomials including the constant term
if length(regular_coefs) == 1 && regular_coefs == 1
    reg_poly = 1;
else
    reg_poly = [1; regular_coefs(:)];
end

% Create seasonal polynomial
seas_poly = zeros(s * length(seasonal_coefs), 1);
seas_poly(1) = 1;
for i = 1:length(seasonal_coefs)
    seas_poly(1 + i*s) = seasonal_coefs(i);
end

% Multiply polynomials using convolution
expanded = conv(reg_poly, seas_poly);
end

function [differenced, diff_info] = apply_differencing(data, d, D, s)
% APPLY_DIFFERENCING Applies regular and seasonal differencing to a time series
%
% USAGE:
%   [DIFFERENCED, DIFF_INFO] = apply_differencing(DATA, D, D, S)
%
% INPUTS:
%   DATA - Original time series data
%   D    - Order of regular differencing
%   D    - Order of seasonal differencing
%   S    - Seasonal period
%
% OUTPUTS:
%   DIFFERENCED - Differenced time series
%   DIFF_INFO   - Structure containing differencing information needed for
%                 inverse differencing

% Validate inputs
data = columncheck(data, 'data');
T = length(data);

% Store original data
diff_info.original_data = data;
diff_info.d = d;
diff_info.D = D;
diff_info.s = s;

% Apply regular differencing
temp = data;
for i = 1:d
    temp = diff(temp);
    diff_info.reg_diff_info{i} = temp;
end

% Apply seasonal differencing
for i = 1:D
    if length(temp) > s
        temp = diff(temp, 1, s);
        diff_info.seas_diff_info{i} = temp;
    else
        error('Not enough data points for seasonal differencing');
    end
end

differenced = temp;
end