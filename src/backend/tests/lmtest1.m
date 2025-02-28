function results = lmtest1(data, lags, regressors)
% LMTEST1 Lagrange Multiplier test for autocorrelation in time series residuals
%
% USAGE:
%   RESULTS = lmtest1(DATA)
%   RESULTS = lmtest1(DATA, LAGS)
%   RESULTS = lmtest1(DATA, LAGS, REGRESSORS)
%
% INPUTS:
%   DATA       - T by 1 vector of time series residuals to test for autocorrelation
%   LAGS       - [OPTIONAL] Positive integer indicating the number of lags to include in the test.
%                Default is min(10, T/5) where T is the length of DATA
%   REGRESSORS - [OPTIONAL] T by K matrix of regressors to include in the auxiliary regression
%
% OUTPUTS:
%   RESULTS    - Structure with the following fields:
%       stat       - LM test statistic (T*R^2)
%       pval       - P-value of the test
%       dof        - Degrees of freedom (equal to number of lags)
%       crit       - Critical values at [0.10, 0.05, 0.01]
%       sig        - Logical vector indicating if null is rejected at [0.10, 0.05, 0.01]
%
% COMMENTS:
%   Tests the null hypothesis of no autocorrelation up to order LAGS against 
%   the alternative of autocorrelation. The test is based on an auxiliary regression 
%   of the residuals on their own lags (and optional regressors).
%
%   Under the null, the test statistic is asymptotically distributed as Chi^2
%   with degrees of freedom equal to the number of lags.
%
% EXAMPLES:
%   % Test for autocorrelation in GARCH model residuals up to order 10
%   results = lmtest1(residuals, 10)
%
%   % Test for autocorrelation with additional regressors
%   results = lmtest1(residuals, 5, exog_variables)
%
% See also LJUNGBOX
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Determine sample size
T = length(data);

% Set default lags if not provided
if nargin < 2 || isempty(lags)
    lags = min(10, floor(T/5));
end

% Validate lags parameter
options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
lags = parametercheck(lags, 'lags', options);

% Enforce maximum number of lags constraint
max_lags = T - 1;
if lags >= max_lags
    error('The number of lags must be less than the sample size minus one.');
end

% Initialize regressors if not provided
if nargin < 3
    regressors = [];
end

% Validate regressors if provided
if ~isempty(regressors)
    regressors = datacheck(regressors, 'regressors');
    if size(regressors, 1) ~= T
        error('REGRESSORS must have the same number of rows as DATA.');
    end
end

% Compute LM test statistics
lm_results = compute_lm_statistic(data, lags, regressors);

% Critical values at 10%, 5%, and 1% significance levels
% These are well-known percentiles of the chi-square distribution
% that can be hard-coded based on degrees of freedom
switch lm_results.dof
    case 1
        crit = [2.706 3.841 6.635];
    case 2
        crit = [4.605 5.991 9.210];
    case 3
        crit = [6.251 7.815 11.345];
    case 4
        crit = [7.779 9.488 13.277];
    case 5
        crit = [9.236 11.070 15.086];
    case 6
        crit = [10.645 12.592 16.812];
    case 7
        crit = [12.017 14.067 18.475];
    case 8
        crit = [13.362 15.507 20.090];
    case 9
        crit = [14.684 16.919 21.666];
    case 10
        crit = [15.987 18.307 23.209];
    otherwise
        % For lags > 10, provide a reasonable approximation
        crit = zeros(1, 3);
        % Using chi-square distribution properties
        % For large dof, chi-square approaches normal distribution with
        % mean = dof and variance = 2*dof
        crit(1) = lm_results.dof + 1.282 * sqrt(2*lm_results.dof); % 10%
        crit(2) = lm_results.dof + 1.645 * sqrt(2*lm_results.dof); % 5%
        crit(3) = lm_results.dof + 2.326 * sqrt(2*lm_results.dof); % 1%
end

% Determine if null hypothesis is rejected at each significance level
sig = lm_results.stat >= crit;

% Construct results structure
results = struct('stat', lm_results.stat, ...
                'pval', lm_results.pval, ...
                'dof', lm_results.dof, ...
                'crit', crit, ...
                'sig', sig);
end

function lm_results = compute_lm_statistic(residuals, lags, exog_regressors)
% Internal function to compute the Lagrange Multiplier test statistic

% Determine effective sample size
T = length(residuals);
effective_T = T - lags;

% Create matrix of lagged residuals for auxiliary regression
X_lags = zeros(effective_T, lags);
for lag = 1:lags
    X_lags(:, lag) = residuals(lags+1-lag:T-lag);
end

% Current residuals (dependent variable in auxiliary regression)
y = residuals(lags+1:end);

% Set up design matrix
if isempty(exog_regressors)
    X = [ones(effective_T, 1) X_lags];
else
    X = [ones(effective_T, 1) X_lags exog_regressors(lags+1:end, :)];
end

% Perform auxiliary regression
beta = (X'*X)\(X'*y);
yhat = X*beta;
e = y - yhat;

% Calculate R^2
SSR = e'*e;  % Sum of squared residuals
SST = (y - mean(y))'*(y - mean(y));  % Total sum of squares
R2 = 1 - SSR/SST;

% Compute LM statistic
lm_stat = effective_T * R2;

% Degrees of freedom = number of lags
dof = lags;

% P-value from chi-square distribution
pval = 1 - chi2cdf(lm_stat, dof);

% Return results
lm_results = struct('stat', lm_stat, ...
                   'pval', pval, ...
                   'dof', dof);
end