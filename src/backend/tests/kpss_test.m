function results = kpss_test(y, options)
% KPSS_TEST Performs the Kwiatkowski-Phillips-Schmidt-Shin test for stationarity
%
% USAGE:
%   RESULTS = kpss_test(Y)
%   RESULTS = kpss_test(Y, OPTIONS)
%
% INPUTS:
%   Y            - Time series data (T by 1 vector)
%   OPTIONS      - Optional inputs (structure)
%     .regression_type - String indicating the type of stationarity test:
%                       'mu'  - Level stationarity (default)
%                       'tau' - Trend stationarity
%     .lags           - Integer value indicating the number of lags to use in
%                       Newey-West estimator. If empty or not provided, will
%                       be set automatically as floor(4*(T/100)^0.25)
%
% OUTPUTS:
%   RESULTS      - Results structure with fields:
%     .stat       - KPSS test statistic
%     .pval       - Approximate p-value
%     .cv         - Critical values at 1%, 2.5%, 5%, and 10% significance levels
%     .lags       - Number of lags used in test
%     .regression_type - Type of regression used in test
%     .significance_levels - Significance levels for critical values
%
% COMMENTS:
%   The KPSS test tests the null hypothesis that a time series is stationary
%   around a deterministic level or trend. This is opposite of unit root tests
%   (ADF, PP) which test the null of non-stationarity.
%
%   Two regression types are supported:
%   1. 'mu' - Level stationarity (constant only)
%   2. 'tau' - Trend stationarity (constant and trend)
%
%   The test statistic is calculated as:
%   KPSS = sum(S_t^2) / (T^2 * lrv)
%   where S_t is the partial sum of residuals and lrv is the long-run variance.
%
% EXAMPLES:
%   % Test for level stationarity with default options
%   results = kpss_test(returns);
%
%   % Test for trend stationarity with 8 lags
%   options.regression_type = 'tau';
%   options.lags = 8;
%   results = kpss_test(prices, options);
%
% REFERENCES:
%   Kwiatkowski, D., Phillips, P.C.B., Schmidt, P., and Shin, Y. (1992).
%   "Testing the Null Hypothesis of Stationarity against the Alternative of a
%   Unit Root." Journal of Econometrics, 54, 159-178.
%
% See also ADF_TEST, PP_TEST, NWSE

% Critical values from Kwiatkowski et al. (1992), Table 1
global kpss_mu_cvs kpss_tau_cvs;

% Critical values for level stationarity (mu)
kpss_mu_cvs = [0.739, 0.574, 0.463, 0.347];  % 1%, 2.5%, 5%, 10%

% Critical values for trend stationarity (tau)
kpss_tau_cvs = [0.216, 0.176, 0.146, 0.119]; % 1%, 2.5%, 5%, 10%

% Step 1: Validate input time series
y = columncheck(y, 'y');
y = datacheck(y, 'y');

% Step 2: Process options
if nargin < 2
    options = [];
end

% Set default options
if ~isfield(options, 'regression_type') || isempty(options.regression_type)
    options.regression_type = 'mu';
else
    if ~ismember(options.regression_type, {'mu', 'tau'})
        error('regression_type must be either ''mu'' or ''tau''');
    end
end

% Get sample size
T = size(y, 1);

% Set default lag length if not specified
if ~isfield(options, 'lags') || isempty(options.lags)
    options.lags = floor(4 * (T/100)^0.25);
else
    % Validate lag parameter
    lag_options = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
    options.lags = parametercheck(options.lags, 'lags', lag_options);
end

% Check if lag length is zero
if options.lags == 0
    warning('KPSS_TEST:ZeroLag', 'Lag length is zero. Long-run variance estimation may not account for autocorrelation.');
end

% Step 3: Detrend the series based on regression_type
if strcmp(options.regression_type, 'mu')
    % Level stationarity - regress on constant only
    X = ones(T, 1);
    beta = regress(y, X);
    residuals = y - X * beta;
    cvs = kpss_mu_cvs;
else
    % Trend stationarity - regress on constant and trend
    t = linspace(1, T, T)';
    X = [ones(T, 1), t];
    beta = regress(y, X);
    residuals = y - X * beta;
    cvs = kpss_tau_cvs;
end

% Step 4: Calculate partial sums of residuals
S = cumsum(residuals);

% Step 5: Calculate long-run variance using Newey-West
% Create X matrix for nwse (intercept only for long-run variance)
X_nw = ones(T, 1);
lrv = nwse(X_nw, residuals, options.lags)^2;

% Step 6: Compute KPSS test statistic
% KPSS = sum(S_t^2) / (T^2 * lrv)
kpss_stat = sum(S.^2) / (T^2 * lrv);

% Step 7: Determine p-value
significance_levels = [0.01, 0.025, 0.05, 0.10];
pvalue = NaN; % Initialize pvalue

if kpss_stat > cvs(1)
    pvalue = 0.01;  % Strong evidence against null of stationarity
elseif kpss_stat < cvs(4)
    pvalue = 0.10;  % Weak evidence against null of stationarity
else
    % Interpolate p-value from critical values
    for i = 1:length(cvs)-1
        if kpss_stat <= cvs(i) && kpss_stat > cvs(i+1)
            % Linear interpolation
            p1 = significance_levels(i);
            p2 = significance_levels(i+1);
            c1 = cvs(i);
            c2 = cvs(i+1);
            pvalue = p1 + (p2 - p1) * (kpss_stat - c1) / (c2 - c1);
            break;
        end
    end
end

% If pvalue is still NaN, something went wrong with the interpolation
if isnan(pvalue)
    warning('KPSS_TEST:PValueInterpolation', 'Could not interpolate p-value. Check test statistic and critical values.');
    pvalue = NaN;
end

% Step 8: Assemble results structure
results = struct();
results.stat = kpss_stat;
results.pval = pvalue;
results.cv = cvs;
results.lags = options.lags;
results.regression_type = options.regression_type;
results.significance_levels = significance_levels;
end