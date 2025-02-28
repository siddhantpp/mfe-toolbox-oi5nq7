function results = jump_test(returns, options)
% JUMP_TEST Implements statistical tests for identifying jumps in high-frequency financial time series
%
% The function uses the ratio of realized volatility (RV) to bipower variation (BV) to identify
% the presence and significance of price jumps in financial assets. Since BV is robust to jumps
% while RV captures both continuous and jump variations, their ratio provides a statistical
% test for jump detection.
%
% USAGE:
%   RESULTS = jump_test(RETURNS)
%   RESULTS = jump_test(RETURNS, OPTIONS)
%
% INPUTS:
%   RETURNS - T×N matrix of high-frequency returns where T is the number
%             of observations and N is the number of assets
%   OPTIONS - [Optional] Structure containing options for the jump test:
%      OPTIONS.alpha      - [Optional] Significance level for the test (default: 0.05)
%      OPTIONS.bvOptions  - [Optional] Options to pass to bv_compute
%      OPTIONS.rvOptions  - [Optional] Options to pass to rv_compute
%
% OUTPUTS:
%   RESULTS - Structure containing jump test results with fields:
%      .zStatistic     - Z-statistic for the jump test
%      .pValue         - p-value for the test
%      .criticalValues - Critical values at standard significance levels (0.01, 0.05, 0.10)
%      .jumpDetected   - Logical array indicating whether jumps are detected at various
%                        significance levels (0.01, 0.05, 0.10)
%      .jumpComponent  - Estimated jump component (when detected)
%      .contComponent  - Estimated continuous component
%      .rv             - Realized volatility estimates
%      .bv             - Bipower variation estimates
%      .ratio          - RV/BV ratio
%
% COMMENTS:
%   The jump test is based on the theoretical result that the ratio of RV to BV
%   converges to 1 in the absence of jumps. A test statistic is computed and
%   compared against critical values to determine if jumps are present.
%
%   The test assumes that the returns are from a continuous-time stochastic process
%   that may include jumps, sampled at high frequency.
%
% REFERENCES:
%   Barndorff-Nielsen, O.E. and Shephard, N. (2004), "Power and bipower variation
%   with stochastic volatility and jumps", Journal of Financial Econometrics, 2, 1-48.
%
%   Barndorff-Nielsen, O.E. and Shephard, N. (2006), "Econometrics of testing for jumps
%   in financial economics using bipower variation", Journal of Financial Econometrics, 4, 1-30.
%
% EXAMPLES:
%   % Simple jump test on a vector of 5-minute returns
%   results = jump_test(fiveminreturns);
%
%   % Jump test with custom options
%   options.alpha = 0.01;  % 1% significance level
%   results = jump_test(fiveminreturns, options);
%
%   % Jump test with custom BV options
%   options.bvOptions.scaleFactor = 0.8;
%   results = jump_test(fiveminreturns, options);
%
% See also RV_COMPUTE, BV_COMPUTE, STDTCDF

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

%% Input validation
if nargin < 1
    error('At least one input argument (RETURNS) is required.');
end

%% Process options
if nargin < 2 || isempty(options)
    options = struct();
end

% Set default options
if ~isfield(options, 'alpha')
    options.alpha = 0.05;
else
    alphaOptions = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.alpha = parametercheck(options.alpha, 'alpha', alphaOptions);
end

if ~isfield(options, 'rvOptions')
    options.rvOptions = struct();
end

if ~isfield(options, 'bvOptions')
    options.bvOptions = struct();
end

%% Validate returns
returns = datacheck(returns, 'returns');
returns = columncheck(returns, 'returns');

% Get dimensions after validation
[T, N] = size(returns);

% Check for sufficient observations
if T < 2
    error('At least 2 observations are required to compute bipower variation.');
end

%% Compute Realized Volatility and Bipower Variation
rv = rv_compute(returns, options.rvOptions);
bv = bv_compute(returns, options.bvOptions);

% Ensure rv and bv are row vectors
rv = reshape(rv, 1, N);
bv = reshape(bv, 1, N);

%% Compute the ratio statistic
% Check for potential numerical issues
if any(bv <= 0)
    warning('Bipower variation contains zero or negative values. Results may be unreliable.');
    % Replace problematic values with small positive numbers to avoid division by zero
    bv(bv <= 0) = eps;
end

% Ratio of RV to BV (should be 1 in absence of jumps)
ratio = rv ./ bv;

%% Calculate test statistic based on Barndorff-Nielsen and Shephard (2006)
% Compute the ratio test statistic (should be close to 1 under null of no jumps)
ratio_minus_1 = ratio - 1;

% Calculate asymptotic variance (theta)
theta = (pi^2/4 + pi - 5);

% Compute the standardized test statistic
z_stat = ratio_minus_1 .* (sqrt(T) / sqrt(theta));

%% Compute p-values and critical values
% For a one-sided test (we're looking for jumps which increase RV relative to BV)
% Use the standardized t-distribution with high df to approximate normal
dof = 30; % High value approximates normal distribution
p_value = 1 - stdtcdf(z_stat, dof);

% Standard significance levels
significance_levels = [0.01, 0.05, 0.10];
critical_values = zeros(length(significance_levels), N);

% Compute critical values for each significance level
for i = 1:length(significance_levels)
    % One-sided critical value for the upper tail
    cv = -stdtcdf(1 - significance_levels(i), dof);
    critical_values(i, :) = repmat(cv, 1, N);
end

% Determine if jumps are detected at each significance level
jump_detected = false(length(significance_levels), N);
for i = 1:length(significance_levels)
    jump_detected(i, :) = z_stat > critical_values(i, :);
end

%% Compute jump and continuous components
% Jump component: max(0, 1 - BV/RV)
jump_component = max(0, 1 - bv ./ rv);

% Continuous component: min(1, BV/RV) * RV
continuous_component = min(1, bv ./ rv) .* rv;

%% Return results
results = struct();
results.zStatistic = z_stat;
results.pValue = p_value;
results.criticalValues = critical_values;
results.significanceLevels = significance_levels;
results.jumpDetected = jump_detected;
results.jumpComponent = jump_component;
results.contComponent = continuous_component;
results.rv = rv;
results.bv = bv;
results.ratio = ratio;

end