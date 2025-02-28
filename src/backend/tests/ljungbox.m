function results = ljungbox(data, lags, dofsAdjust)
% LJUNGBOX Performs the Ljung-Box Q-test for autocorrelation in time series residuals
%
% USAGE:
%   [RESULTS] = ljungbox(DATA)
%   [RESULTS] = ljungbox(DATA, LAGS)
%   [RESULTS] = ljungbox(DATA, LAGS, DOFSADJUST)
%
% INPUTS:
%   DATA       - T by 1 vector of data (typically model residuals)
%   LAGS       - [OPTIONAL] Maximum lag to include in the test, or vector of lags.
%                Default is min(10, T/5) where T is the sample size.
%   DOFSADJUST - [OPTIONAL] Degrees of freedom adjustment for the test, typically
%                set to the number of parameters estimated in the model.
%                Default is 0 (no adjustment).
%
% OUTPUTS:
%   RESULTS    - Structure containing the test results with fields:
%                  'stats'    - Ljung-Box Q-statistics for each lag
%                  'pvals'    - p-values for the Q-statistics
%                  'isRejected10pct' - Logical indicating rejection at 10% level
%                  'isRejected5pct'  - Logical indicating rejection at 5% level
%                  'isRejected1pct'  - Logical indicating rejection at 1% level
%                  'lags'     - The lags used in the test
%                  'dofs'     - Degrees of freedom for each lag
%                  'dofsAdjust' - The degrees of freedom adjustment used
%                  'T'        - The sample size
%
% COMMENTS:
%   The Ljung-Box Q-test examines whether the autocorrelations of the residuals
%   from a fitted model differ significantly from zero. The test statistic is:
%
%   Q = T*(T+2)*sum((ρ_k^2)/(T-k)) for k=1 to m
%
%   Where T is the sample size, ρ_k is the sample autocorrelation at lag k,
%   and m is the maximum lag being considered.
%
%   Under the null hypothesis of no autocorrelation, Q follows a chi-square
%   distribution with degrees of freedom equal to m-dofsAdjust, where dofsAdjust
%   is the number of parameters estimated in the model.
%
%   Rejection of the null hypothesis indicates that the model does not 
%   adequately capture the dependency structure in the data.
%
% EXAMPLES:
%   % Basic usage with default lags
%   results = ljungbox(residuals);
%   
%   % Specify maximum lag
%   results = ljungbox(residuals, 15);
%   
%   % Adjust degrees of freedom for ARMA(1,1) model
%   results = ljungbox(residuals, 15, 2);
%
% See also sacf, datacheck, columncheck, parametercheck, chi2cdf

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Get data length
T = length(data);

% Process lags parameter
if nargin < 2 || isempty(lags)
    % Default lags: min(10, T/5)
    lags = min(10, floor(T/5));
end

% Validate lags parameter
lagOptions = struct('isInteger', true, 'isPositive', true);
lags = parametercheck(lags, 'lags', lagOptions);

% If scalar, expand to vector 1:lags
if isscalar(lags)
    lags = (1:lags)';
else
    % Ensure column vector format
    lags = columncheck(lags, 'lags');
    % Sort lags in ascending order
    lags = sort(lags);
end

% Process dofsAdjust parameter
if nargin < 3 || isempty(dofsAdjust)
    dofsAdjust = 0;
else
    % Validate dofsAdjust parameter
    dofsAdjustOptions = struct('isInteger', true, 'isNonNegative', true, 'isscalar', true);
    dofsAdjust = parametercheck(dofsAdjust, 'dofsAdjust', dofsAdjustOptions);
    
    % Ensure dofsAdjust does not exceed maximum lag
    if dofsAdjust >= max(lags)
        error('dofsAdjust must be less than the maximum lag');
    end
end

% Compute sample autocorrelations for lags 1 to max(lags)
acf = sacf(data, (1:max(lags))');

% Calculate the Ljung-Box Q-statistic for each specified lag
qStats = zeros(length(lags), 1);
for i = 1:length(lags)
    m = lags(i);
    
    % Get the first m autocorrelations
    autocorrs = acf(1:m);
    
    % Calculate the Q-statistic using the first m autocorrelations
    sumTerm = sum((autocorrs.^2) ./ (T - (1:m)'));
    qStats(i) = T * (T + 2) * sumTerm;
end

% Calculate degrees of freedom for chi-square distribution
dofs = lags - dofsAdjust;
% Ensure all dofs are positive
dofs = max(dofs, 1);

% Calculate p-values using chi-square distribution
pValues = 1 - chi2cdf(qStats, dofs);

% Determine if null hypothesis is rejected at different significance levels
isRejected10pct = pValues < 0.10;
isRejected5pct = pValues < 0.05;
isRejected1pct = pValues < 0.01;

% Create and return results structure
results = struct(...
    'stats', qStats, ...
    'pvals', pValues, ...
    'isRejected10pct', isRejected10pct, ...
    'isRejected5pct', isRejected5pct, ...
    'isRejected1pct', isRejected1pct, ...
    'lags', lags, ...
    'dofs', dofs, ...
    'dofsAdjust', dofsAdjust, ...
    'T', T ...
);
end