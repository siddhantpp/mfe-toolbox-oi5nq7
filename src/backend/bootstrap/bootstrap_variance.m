function results = bootstrap_variance(data, statistic_fn, options)
% BOOTSTRAP_VARIANCE Computes robust variance estimates using bootstrap resampling
%
% USAGE:
%   RESULTS = bootstrap_variance(DATA, STATISTIC_FN, OPTIONS)
%
% INPUTS:
%   DATA         - T by K matrix of time series data
%   STATISTIC_FN - Function handle for the statistic to be bootstrapped
%                  Must accept a T by K matrix and return a scalar result
%   OPTIONS      - Structure with bootstrap parameters:
%                  OPTIONS.bootstrap_type: String, either 'block' or 'stationary'
%                      'block': Uses circular block bootstrap
%                      'stationary': Uses stationary bootstrap
%                      Default: 'block'
%                  OPTIONS.block_size: Integer, block length for block bootstrap
%                      Required when bootstrap_type = 'block'
%                  OPTIONS.p: Scalar between 0 and 1, probability parameter for stationary bootstrap
%                      Required when bootstrap_type = 'stationary'
%                  OPTIONS.replications: Positive integer, number of bootstrap replications
%                      Default: 1000
%                  OPTIONS.conf_level: Confidence level for intervals (between 0 and 1)
%                      Default: 0.95
%
% OUTPUTS:
%   RESULTS      - Structure with fields:
%                  RESULTS.variance      - Bootstrap variance estimate of the statistic
%                  RESULTS.std_error     - Standard error of the statistic
%                  RESULTS.conf_lower    - Lower confidence bound for the statistic
%                  RESULTS.conf_upper    - Upper confidence bound for the statistic
%                  RESULTS.bootstrap_stats - All bootstrap statistics
%                  RESULTS.mean          - Mean of bootstrap statistics
%                  RESULTS.median        - Median of bootstrap statistics
%                  RESULTS.min           - Minimum of bootstrap statistics
%                  RESULTS.max           - Maximum of bootstrap statistics
%                  RESULTS.q25           - 25th percentile of bootstrap statistics
%                  RESULTS.q75           - 75th percentile of bootstrap statistics
%
% COMMENTS:
%   Bootstrap variance estimation is a resampling technique particularly useful
%   for time series data with temporal dependencies. This function supports two
%   bootstrap methods suitable for dependent data:
%
%   1. Block Bootstrap: Resamples blocks of fixed length to preserve temporal
%      structure within each block. Appropriate when dependency decays with
%      distance between observations.
%
%   2. Stationary Bootstrap: Uses random block lengths from a geometric
%      distribution with parameter p to ensure the resampled series remains
%      stationary. The expected block length is 1/p.
%
%   The variance estimate is calculated from the variation among bootstrap
%   statistics, providing a robust estimate that accounts for temporal
%   dependencies in the data, making it particularly suitable for financial 
%   time series with autocorrelation and heteroskedasticity.
%
% EXAMPLES:
%   % Estimate variance of mean return using block bootstrap
%   returns = randn(100, 1);
%   options.bootstrap_type = 'block';
%   options.block_size = 10;
%   options.replications = 1000;
%   results = bootstrap_variance(returns, @mean, options);
%
%   % Estimate variance of Sharpe ratio using stationary bootstrap
%   options.bootstrap_type = 'stationary';
%   options.p = 0.1;
%   sharpe = @(x) mean(x) / std(x);
%   results = bootstrap_variance(returns, sharpe, options);
%
% See also block_bootstrap, stationary_bootstrap, datacheck, parametercheck
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate inputs
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Validate statistic_fn
if ~isa(statistic_fn, 'function_handle')
    error('STATISTIC_FN must be a function handle.');
end

% Parse options with defaults
if nargin < 3 || isempty(options)
    options = struct();
end

% Default bootstrap type is block bootstrap
if ~isfield(options, 'bootstrap_type') || isempty(options.bootstrap_type)
    options.bootstrap_type = 'block';
else
    % Validate bootstrap_type
    if ~ischar(options.bootstrap_type) || ...
            (~strcmpi(options.bootstrap_type, 'block') && ~strcmpi(options.bootstrap_type, 'stationary'))
        error('OPTIONS.bootstrap_type must be either ''block'' or ''stationary''.');
    end
end

% Set default number of replications
if ~isfield(options, 'replications') || isempty(options.replications)
    options.replications = 1000;
else
    % Validate replications parameter
    options_check = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    options.replications = parametercheck(options.replications, 'options.replications', options_check);
end

% Set default confidence level
if ~isfield(options, 'conf_level') || isempty(options.conf_level)
    options.conf_level = 0.95;
else
    % Validate confidence level parameter
    options_check = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.conf_level = parametercheck(options.conf_level, 'options.conf_level', options_check);
end

% Get sample size
[T, ~] = size(data);

% Check block size or p parameter based on bootstrap type
if strcmpi(options.bootstrap_type, 'block')
    % Block bootstrap requires block_size
    if ~isfield(options, 'block_size') || isempty(options.block_size)
        % Default block size based on sample size
        options.block_size = max(floor(1.75 * T^(1/3)), 2);
    else
        % Validate block_size parameter
        options_check = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
        options.block_size = parametercheck(options.block_size, 'options.block_size', options_check);
    end
else
    % Stationary bootstrap requires p parameter
    if ~isfield(options, 'p') || isempty(options.p)
        % Default p based on sample size (expected block length = 1/p)
        options.p = min(1/max(floor(1.75 * T^(1/3)), 2), 1);
    else
        % Validate p parameter
        options_check = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
        options.p = parametercheck(options.p, 'options.p', options_check);
    end
end

% Generate bootstrap samples
if strcmpi(options.bootstrap_type, 'block')
    % Use block bootstrap
    bs_data = block_bootstrap(data, options.block_size, options.replications);
else
    % Use stationary bootstrap
    bs_data = stationary_bootstrap(data, options.p, options.replications);
end

% Apply the statistic function to each bootstrap sample
bs_stats = zeros(options.replications, 1);
for i = 1:options.replications
    sample = bs_data(:, :, i);
    stat_value = statistic_fn(sample);
    
    % Ensure the statistic is scalar
    if ~isscalar(stat_value)
        error('STATISTIC_FN must return a scalar value.');
    end
    
    bs_stats(i) = stat_value;
end

% Compute the variance and standard error of the bootstrap statistics
variance_estimate = var(bs_stats);
std_error = sqrt(variance_estimate);

% Compute confidence intervals for the statistic
alpha = 1 - options.conf_level;
conf_lower = prctile(bs_stats, alpha/2 * 100);
conf_upper = prctile(bs_stats, (1 - alpha/2) * 100);

% Construct results structure
results = struct();
results.variance = variance_estimate;
results.std_error = std_error;
results.conf_lower = conf_lower;
results.conf_upper = conf_upper;
results.bootstrap_stats = bs_stats;

% Add additional bootstrap statistics
results.mean = mean(bs_stats);
results.median = prctile(bs_stats, 50);
results.min = min(bs_stats);
results.max = max(bs_stats);
results.q25 = prctile(bs_stats, 25);
results.q75 = prctile(bs_stats, 75);

end