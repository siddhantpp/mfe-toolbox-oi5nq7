function results = adf_test(y, options)
% ADF_TEST Augmented Dickey-Fuller test for unit roots in time series
%
% USAGE:
%   RESULTS = adf_test(Y)
%   RESULTS = adf_test(Y, OPTIONS)
%
% INPUTS:
%   Y          - Time series data (T x 1 vector)
%   OPTIONS    - [OPTIONAL] Structure with fields:
%                'regression_type': Type of regression
%                   'n'  - No constant or trend
%                   'c'  - Constant only (default)
%                   'ct' - Constant and trend
%                'lags': Number of lags to include in test
%                   p    - Fixed number of lags where p is a non-negative integer
%                   'aic'- Determine lag using Akaike Information Criterion
%                   'bic'- Determine lag using Bayesian Information Criterion
%                   Default: 0 (standard Dickey-Fuller test)
%                'max_lags': Maximum lags to consider when using 'aic' or 'bic'
%                   Default: min(floor(T/3), 8)
%
% OUTPUTS:
%   RESULTS    - Structure with fields:
%                'stat'            - ADF test statistic
%                'pval'            - Approximate p-value
%                'crit_vals'       - Critical values [1%, 5%, 10%, 90%]
%                'lags'            - Number of lags used in test
%                'regression_type' - Type of regression used
%                'y'               - Original time series data
%                'T'               - Sample size
%                'optimal_lag'     - Optimal lag (if using 'aic' or 'bic')
%                'ic_values'       - Information criterion values (if using 'aic' or 'bic')
%
% COMMENTS:
%   Implements the Augmented Dickey-Fuller test for unit roots in time series data.
%   The null hypothesis is that the time series contains a unit root (non-stationary).
%   Rejection of the null hypothesis supports stationarity.
%
%   The ADF test regression is:
%   Δy_t = α + βt + ρy_{t-1} + δ_1Δy_{t-1} + ... + δ_pΔy_{t-p} + ε_t
%
%   where:
%   - α is the constant term (included if regression_type is 'c' or 'ct')
%   - βt is the deterministic trend (included if regression_type is 'ct')
%   - ρ is the coefficient on the lagged level (test focuses on this coefficient)
%   - δ_i are the coefficients on the lagged differences
%
%   The test statistic is the t-statistic for ρ, which follows a non-standard
%   distribution under the null hypothesis.
%
%   Critical values are from MacKinnon (1996, 2010).
%
% EXAMPLES:
%   % Basic test with default settings (constant, no lags)
%   results = adf_test(y);
%
%   % Test with constant and trend, 2 lags
%   options = struct('regression_type', 'ct', 'lags', 2);
%   results = adf_test(y, options);
%
%   % Test with automatic lag selection using AIC
%   options = struct('lags', 'aic');
%   results = adf_test(y, options);
%
% REFERENCES:
%   Dickey, D.A., Fuller, W.A. (1979). "Distribution of the estimators for
%       autoregressive time series with a unit root". Journal of the American
%       Statistical Association, 74 (366): 427–431.
%   MacKinnon, J.G. (1996). "Numerical distribution functions for unit root and
%       cointegration tests". Journal of Applied Econometrics, 11, 601-618.
%   MacKinnon, J.G. (2010). "Critical Values for Cointegration Tests". 
%       Queen's Economics Department Working Paper No. 1227.
%
% See also KPSSTEST, PPTEST, STATTEST

% Copyright: MFE Toolbox Version 4.0
% Revision: 4.0    Date: 2009/10/28

% Define critical values for ADF test with no constant
global adf_n_cvs
if isempty(adf_n_cvs)
    % Critical values at 1%, 5%, 10%, and 90% for ADF test with no constant
    adf_n_cvs = [-2.56, -1.94, -1.62, 0.91];
end

% Define critical values for ADF test with constant
global adf_c_cvs
if isempty(adf_c_cvs)
    % Critical values at 1%, 5%, 10%, and 90% for ADF test with constant
    adf_c_cvs = [-3.43, -2.86, -2.57, 0.22];
end

% Define critical values for ADF test with constant and trend
global adf_ct_cvs
if isempty(adf_ct_cvs)
    % Critical values at 1%, 5%, 10%, and 90% for ADF test with constant and trend
    adf_ct_cvs = [-3.96, -3.41, -3.13, -1.25];
end

% Validate input time series
y = columncheck(y, 'y');
y = datacheck(y, 'y');

% Get sample size
T = size(y, 1);

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Process regression type option
if ~isfield(options, 'regression_type') || isempty(options.regression_type)
    options.regression_type = 'c'; % Default is constant only
else
    valid_types = {'n', 'c', 'ct'};
    if ~ismember(options.regression_type, valid_types)
        error('Invalid regression_type. Must be ''n'', ''c'', or ''ct''.');
    end
end

% Process lags option
if ~isfield(options, 'lags') || isempty(options.lags)
    options.lags = 0; % Default is standard Dickey-Fuller test (no lags)
end

% Set default max_lags if not provided
if ~isfield(options, 'max_lags') || isempty(options.max_lags)
    options.max_lags = min(floor(T/3), 8);
end

% Extract options for clarity
regression_type = options.regression_type;
lags = options.lags;
max_lags = options.max_lags;

% Validate max_lags
if ~isscalar(max_lags) || max_lags < 0 || max_lags ~= floor(max_lags)
    error('max_lags must be a non-negative integer.');
end

% Handle automatic lag selection
if ischar(lags)
    if strcmpi(lags, 'aic') || strcmpi(lags, 'bic')
        % Store original lags type for results
        lag_type = lags;
        % Compute optimal lag using information criterion
        [lags, ic_values] = compute_ic_lag(y, lag_type, regression_type, max_lags);
    else
        error('If lags is a string, it must be ''aic'' or ''bic''.');
    end
else
    % Validate lags as non-negative integer
    if ~isscalar(lags) || lags < 0 || lags ~= floor(lags)
        error('lags must be a non-negative integer, ''aic'', or ''bic''.');
    end
    lag_type = 'fixed';
    ic_values = [];
end

% Compute first difference of y
dy = diff(y);
T_eff = T - 1; % Effective sample size after differencing

% Prepare lagged level (y_{t-1})
y_lag = y(1:end-1);

% Initialize regressors matrix
if lags == 0
    % No lagged differences
    dy_lags = [];
else
    % Create matrix of lagged differences
    dy_lags = zeros(T_eff-lags, lags);
    for i = 1:lags
        dy_lags(:,i) = dy(lags+1-i:T_eff-i);
    end
end

% Set up dependent variable (trimmed to match regressors)
if lags == 0
    dy_dep = dy;
else
    dy_dep = dy(lags+1:end);
end

% Determine regression variables based on regression type
switch regression_type
    case 'n' % No constant or trend
        if lags == 0
            X = y_lag;
        else
            X = [y_lag(lags+1:end), dy_lags];
        end
    
    case 'c' % Constant only
        if lags == 0
            X = [ones(T_eff,1), y_lag];
        else
            X = [ones(T_eff-lags,1), y_lag(lags+1:end), dy_lags];
        end
    
    case 'ct' % Constant and trend
        if lags == 0
            X = [ones(T_eff,1), (1:T_eff)', y_lag];
        else
            X = [ones(T_eff-lags,1), (1:T_eff-lags)', y_lag(lags+1:end), dy_lags];
        end
end

% Estimate regression using OLS
[beta, ~, ~, ~, stats] = regress(dy_dep, X);

% Extract coefficient on lagged level and its standard error
switch regression_type
    case 'n' % No constant or trend
        coef_idx = 1;
    case 'c' % Constant only
        coef_idx = 2;
    case 'ct' % Constant and trend
        coef_idx = 3;
end

% Calculate ADF test statistic (t-statistic for coefficient on y_{t-1})
adf_stat = stats(3); % t-statistic is in stats(3)

% Determine critical values based on regression type
switch regression_type
    case 'n'
        crit_vals = adf_n_cvs;
    case 'c'
        crit_vals = adf_c_cvs;
    case 'ct'
        crit_vals = adf_ct_cvs;
end

% Calculate approximate p-value using interpolation or extrapolation
if adf_stat <= crit_vals(1)
    % Below 1% critical value
    p_value = 0.01 * adf_stat / crit_vals(1);
elseif adf_stat <= crit_vals(2)
    % Between 1% and 5% critical values
    p_value = 0.01 + 0.04 * (adf_stat - crit_vals(1)) / (crit_vals(2) - crit_vals(1));
elseif adf_stat <= crit_vals(3)
    % Between 5% and 10% critical values
    p_value = 0.05 + 0.05 * (adf_stat - crit_vals(2)) / (crit_vals(3) - crit_vals(2));
elseif adf_stat <= crit_vals(4)
    % Between 10% and 90% critical values
    p_value = 0.10 + 0.80 * (adf_stat - crit_vals(3)) / (crit_vals(4) - crit_vals(3));
else
    % Above 90% critical value
    p_value = 0.90 + 0.10 * (adf_stat - crit_vals(4)) / (1 - crit_vals(4));
    p_value = min(p_value, 1.0); % Cap at 1.0
end

% Create and populate results structure
results = struct();
results.stat = adf_stat;
results.pval = p_value;
results.crit_vals = crit_vals;
results.lags = lags;
results.regression_type = regression_type;
results.y = y;
results.T = T;

% Include optimal lag information if using AIC or BIC
if exist('lag_type', 'var') && (strcmpi(lag_type, 'aic') || strcmpi(lag_type, 'bic'))
    results.lag_selection_method = lag_type;
    results.optimal_lag = lags;
    results.ic_values = ic_values;
end

end

function [optimal_lag, ic_values] = compute_ic_lag(y, ic_type, regression_type, max_lags)
% COMPUTE_IC_LAG Determine optimal lag length using information criteria
%
% INPUTS:
%   y              - Time series data
%   ic_type        - Information criterion to use ('aic' or 'bic')
%   regression_type - Type of regression ('n', 'c', or 'ct')
%   max_lags       - Maximum number of lags to consider
%
% OUTPUTS:
%   optimal_lag    - Optimal lag length according to specified criterion
%   ic_values      - Vector of information criterion values for each lag

% Get sample size
T = length(y);

% Set default maximum lag if not provided
if nargin < 4 || isempty(max_lags)
    max_lags = min(floor(T/3), 8);
end

% Initialize storage for information criterion values
ic_values = zeros(max_lags+1, 1);

% Compute first difference of y
dy = diff(y);
T_eff = T - 1; % Effective sample size after differencing

% Loop through potential lags (including 0)
for lag = 0:max_lags
    % Prepare lagged level (y_{t-1})
    y_lag = y(1:end-1);
    
    % Initialize regressors matrix
    if lag == 0
        % No lagged differences
        dy_lags = [];
    else
        % Create matrix of lagged differences
        dy_lags = zeros(T_eff-lag, lag);
        for i = 1:lag
            dy_lags(:,i) = dy(lag+1-i:T_eff-i);
        end
    end
    
    % Set up dependent variable (trimmed to match regressors)
    if lag == 0
        dy_dep = dy;
    else
        dy_dep = dy(lag+1:end);
    end
    
    % Determine regression variables based on regression type
    switch regression_type
        case 'n' % No constant or trend
            if lag == 0
                X = y_lag;
                num_params = 1 + lag; % y_{t-1} + lagged differences
            else
                X = [y_lag(lag+1:end), dy_lags];
                num_params = 1 + lag; % y_{t-1} + lagged differences
            end
        
        case 'c' % Constant only
            if lag == 0
                X = [ones(T_eff,1), y_lag];
                num_params = 2 + lag; % constant + y_{t-1} + lagged differences
            else
                X = [ones(T_eff-lag,1), y_lag(lag+1:end), dy_lags];
                num_params = 2 + lag; % constant + y_{t-1} + lagged differences
            end
        
        case 'ct' % Constant and trend
            if lag == 0
                X = [ones(T_eff,1), (1:T_eff)', y_lag];
                num_params = 3 + lag; % constant + trend + y_{t-1} + lagged differences
            else
                X = [ones(T_eff-lag,1), (1:T_eff-lag)', y_lag(lag+1:end), dy_lags];
                num_params = 3 + lag; % constant + trend + y_{t-1} + lagged differences
            end
    end
    
    % Estimate regression using OLS
    [beta, ~, resid] = regress(dy_dep, X);
    
    % Compute residual sum of squares
    RSS = sum(resid.^2);
    
    % Compute information criterion
    T_reg = length(dy_dep); % Number of observations in regression
    
    if strcmpi(ic_type, 'aic')
        % Akaike Information Criterion
        ic_values(lag+1) = log(RSS/T_reg) + 2 * num_params / T_reg;
    else % BIC
        % Bayesian Information Criterion
        ic_values(lag+1) = log(RSS/T_reg) + num_params * log(T_reg) / T_reg;
    end
end

% Find lag with minimum information criterion value
[~, min_idx] = min(ic_values);
optimal_lag = min_idx - 1; % Convert back to lag (0-based indexing)

end