% Critical values for Phillips-Perron test
% These are for the three regression types: no constant, constant only, and constant and trend
% Each row represents critical values at [1%, 5%, 10%, 90%] significance levels

% Critical values for regression with no constant (n)
pp_n_cvs = [-2.58, -1.95, -1.62, 1.62];

% Critical values for regression with constant (c)
pp_c_cvs = [-3.43, -2.86, -2.57, 0.26];

% Critical values for regression with constant and trend (ct)
pp_ct_cvs = [-3.96, -3.41, -3.13, -1.14];

function results = pp_test(y, options)
% PP_TEST Conducts the Phillips-Perron test for unit roots in time series data
%
% The Phillips-Perron test is a non-parametric unit root test that accounts for
% serial correlation and heteroskedasticity in the error term by using Newey-West
% robust standard errors. It is an extension of the Augmented Dickey-Fuller test
% and is particularly useful for financial time series with complex error structures.
%
% USAGE:
%   results = pp_test(y)
%   results = pp_test(y, options)
%
% INPUTS:
%   y             - T by 1 vector of time series data to test for unit roots
%   options       - [OPTIONAL] Structure with the following fields:
%                   options.regression_type - String specifying the regression type:
%                       'n' - No constant (demeaning) or trend           [y(t) = rho*y(t-1) + e(t)]
%                       'c' - Constant only (default)                   [y(t) = mu + rho*y(t-1) + e(t)]
%                       'ct' - Constant and trend                       [y(t) = mu + beta*t + rho*y(t-1) + e(t)]
%                   options.lags - Integer specifying the lag order for Newey-West
%                       correction. If empty or not provided, automatically set to
%                       floor(4*(T/100)^0.25)
%
% OUTPUTS:
%   results       - Structure containing test results with fields:
%                   results.stat_alpha - The Z(alpha) Phillips-Perron statistic
%                   results.stat_tau - The Z(tau) Phillips-Perron statistic (t-stat)
%                   results.pval - Approximate p-value for the test
%                   results.cv_1pct - 1% critical value
%                   results.cv_5pct - 5% critical value 
%                   results.cv_10pct - 10% critical value
%                   results.regression_type - Regression type used in test
%                   results.lags - Number of lags used in Newey-West correction
%                   results.nobs - Number of observations used in the test
%
% REFERENCES:
%   Phillips, P. C. B., and P. Perron (1988). "Testing for a Unit Root in
%   Time Series Regression." Biometrika, 75(2), 335-346.
%
% EXAMPLES:
%   % Test for unit root in returns data with default options (constant)
%   results = pp_test(returns);
%
%   % Test for unit root with constant and trend
%   options.regression_type = 'ct';
%   results = pp_test(returns, options);
%
%   % Test with specified lag order for Newey-West correction
%   options.regression_type = 'c';
%   options.lags = 10;
%   results = pp_test(returns, options);
%
% See also ADF_TEST, KPSS_TEST, NWSE, PARAMETERCHECK, COLUMNCHECK, DATACHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Access global critical values
global pp_n_cvs pp_c_cvs pp_ct_cvs;

% Step 1: Input validation
y = columncheck(y, 'y');  % Ensure y is a column vector
y = datacheck(y, 'y');    % Comprehensive validation of y data

% Step 2: Process options
if nargin < 2
    options = struct();
end

% Set default regression type to 'c' (constant only) if not specified
if ~isfield(options, 'regression_type') || isempty(options.regression_type)
    options.regression_type = 'c';
else
    % Validate regression_type
    if ~ischar(options.regression_type) || ~any(strcmp(options.regression_type, {'n', 'c', 'ct'}))
        error('options.regression_type must be one of ''n'', ''c'', or ''ct''');
    end
end

% Get sample size
T = size(y, 1);

% Set default lag order if not specified
if ~isfield(options, 'lags') || isempty(options.lags)
    % Common rule of thumb for lag selection in Newey-West: floor(4*(T/100)^0.25)
    options.lags = floor(4 * (T/100)^0.25);
else
    % Validate lags using parametercheck
    lag_options = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
    options.lags = parametercheck(options.lags, 'options.lags', lag_options);
end

% Step 3: Prepare data for Phillips-Perron test
% Compute first differences
dy = diff(y);
% Prepare lagged series (y_{t-1})
y_lag = y(1:end-1);

% Step 4: Prepare regression variables based on regression type
if strcmp(options.regression_type, 'n')
    % No constant model: dy(t) = (rho-1)*y(t-1) + e(t)
    X = y_lag;
    regression_type_index = 1;
elseif strcmp(options.regression_type, 'c')
    % Constant model: dy(t) = mu + (rho-1)*y(t-1) + e(t)
    X = [ones(T-1, 1), y_lag];
    regression_type_index = 2;
elseif strcmp(options.regression_type, 'ct')
    % Constant and trend model: dy(t) = mu + beta*t + (rho-1)*y(t-1) + e(t)
    X = [ones(T-1, 1), (1:T-1)', y_lag];
    regression_type_index = 3;
end

% Step 5: Run OLS regression using regress
% Estimate the regression coefficients
[beta, ~, residuals] = regress(dy, X);

% Extract relevant coefficient and its position
if strcmp(options.regression_type, 'n')
    rho_hat = beta(1);
    rho_index = 1;
elseif strcmp(options.regression_type, 'c')
    rho_hat = beta(2);
    rho_index = 2;
elseif strcmp(options.regression_type, 'ct')
    rho_hat = beta(3);
    rho_index = 3;
end

% Step 6: Calculate the uncorrected test statistic
% Compute OLS standard error for coefficient (without Newey-West correction)
SSR = sum(residuals.^2);
df = T - 1 - size(X, 2) + 1;  % Degrees of freedom
sigma2_ols = SSR / df;  % OLS estimate of residual variance

% Calculate OLS standard error for rho
XX_inv = inv(X' * X);
se_rho_ols = sqrt(sigma2_ols * XX_inv(rho_index, rho_index));

% Uncorrected t-statistic for rho
t_stat_ols = rho_hat / se_rho_ols;

% Step 7: Calculate Newey-West robust standard errors
se_nw = nwse(X, residuals, options.lags);
se_rho_nw = se_nw(rho_index);

% Step 8: Calculate Phillips-Perron test statistics with corrections

% Calculate the sample variance of the residuals (short-run variance)
s2 = SSR / (T-1);

% Calculate the long-run variance using Newey-West kernel
s2_nw = s2;
for l = 1:options.lags
    weight = 1 - l/(options.lags+1);  % Bartlett kernel weight
    gamma_l = 0;
    for t = (l+1):(T-1)
        gamma_l = gamma_l + residuals(t) * residuals(t-l);
    end
    gamma_l = gamma_l / (T-1);
    s2_nw = s2_nw + 2 * weight * gamma_l;
end

% Calculate the correction factors for Phillips-Perron test
lambda2 = s2_nw;
lambda2_0 = s2;
correction_factor = (T-1) * (lambda2 - lambda2_0) / (2 * sum(y_lag.^2));

% Calculate Z(alpha) and Z(tau) statistics
% Z(alpha) = T * (rho_hat - 1) - correction_factor
Z_alpha = (T-1) * (rho_hat - 1) - correction_factor;

% Z(tau) = t_stat_ols * sqrt(lambda2_0/lambda2) - correction_factor / (sqrt(lambda2) * sqrt(sum(y_lag.^2)/(T-1)))
Z_tau = t_stat_ols * sqrt(lambda2_0/lambda2) - correction_factor / (sqrt(lambda2) * sqrt(sum(y_lag.^2)/(T-1)));

% Step 9: Determine critical values based on regression type
if strcmp(options.regression_type, 'n')
    cv = pp_n_cvs;
elseif strcmp(options.regression_type, 'c')
    cv = pp_c_cvs;
elseif strcmp(options.regression_type, 'ct')
    cv = pp_ct_cvs;
end

% Step 10: Calculate p-value (approximate using interpolation/extrapolation)
if Z_tau <= cv(3)  % If stat is less than or equal to 10% critical value
    % Interpolate between known critical values
    if Z_tau <= cv(1)  % Less than 1% critical value
        p_value = 0.01;
    elseif Z_tau <= cv(2)  % Between 1% and 5% critical values
        p_value = 0.01 + 0.04 * (Z_tau - cv(1)) / (cv(2) - cv(1));
    else  % Between 5% and 10% critical values
        p_value = 0.05 + 0.05 * (Z_tau - cv(2)) / (cv(3) - cv(2));
    end
else  % Greater than 10% critical value
    % Extrapolate for p-values > 10%
    p_value = 0.10 + 0.80 * (Z_tau - cv(3)) / (cv(4) - cv(3));
    p_value = min(p_value, 1);  % Cap at 1.0
end

% Step 11: Assemble results structure
results = struct();
results.stat_alpha = Z_alpha;
results.stat_tau = Z_tau;
results.pval = p_value;
results.cv_1pct = cv(1);
results.cv_5pct = cv(2);
results.cv_10pct = cv(3);
results.regression_type = options.regression_type;
results.lags = options.lags;
results.nobs = T;

end