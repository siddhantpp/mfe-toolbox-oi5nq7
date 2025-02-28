function results = bootstrap_confidence_intervals(data, statistic_fn, options)
% BOOTSTRAP_CONFIDENCE_INTERVALS Computes various types of bootstrap confidence intervals
% for statistics derived from time series data, supporting both block and stationary
% bootstrap methods to account for temporal dependence.
%
% USAGE:
%   RESULTS = bootstrap_confidence_intervals(DATA, STATISTIC_FN, OPTIONS)
%
% INPUTS:
%   DATA         - T by N matrix of time series data
%   STATISTIC_FN - Function handle of the form @(x) that calculates the statistic
%                  of interest from the data x. Must return a scalar value.
%   OPTIONS      - Structure with fields:
%                  bootstrap_type: 'block' or 'stationary' (default: 'block')
%                  block_size: Block size for block bootstrap (default: automatic)
%                  p: Probability parameter for stationary bootstrap (default: automatic)
%                  replications: Number of bootstrap samples (default: 1000)
%                  conf_level: Confidence level (default: 0.95)
%                  method: Confidence interval method (default: 'percentile')
%                           'percentile', 'basic', 'studentized', 'bc', 'bca'
%
% OUTPUTS:
%   RESULTS      - Structure with fields:
%                  original_statistic: Value of statistic on original data
%                  conf_level: Confidence level used
%                  lower: Lower bound of confidence interval
%                  upper: Upper bound of confidence interval
%                  bootstrap_statistics: Vector of bootstrap statistics
%                  method: Method used for confidence interval
%                  bootstrap_options: Options used for bootstrap
%
% COMMENTS:
%   This function implements several methods for computing bootstrap confidence
%   intervals for financial time series with temporal dependence:
%
%   - 'percentile': Direct percentiles of bootstrap distribution
%   - 'basic': Symmetric around original estimate based on bootstrap distribution
%   - 'studentized': Accounts for variability in bootstrap estimates (computationally intensive)
%   - 'bc': Bias-corrected percentiles, adjusts for median bias in bootstrap distribution
%   - 'bca': Bias-corrected and accelerated, adjusts for both bias and skewness
%
% EXAMPLES:
%   % Compute 95% confidence interval for mean return using block bootstrap
%   options = struct('bootstrap_type', 'block', 'block_size', 10, 'method', 'percentile');
%   results = bootstrap_confidence_intervals(returns, @mean, options);
%
%   % Compute 90% confidence interval for Sharpe ratio using stationary bootstrap
%   sharpe = @(x) mean(x) / std(x);
%   options = struct('bootstrap_type', 'stationary', 'p', 0.1, 'conf_level', 0.90, 'method', 'bca');
%   results = bootstrap_confidence_intervals(returns, sharpe, options);
%
% See also block_bootstrap, stationary_bootstrap, parametercheck, datacheck, columncheck

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Validate statistic_fn is a function handle
if ~isa(statistic_fn, 'function_handle')
    error('STATISTIC_FN must be a function handle.');
end

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Default bootstrap type
if ~isfield(options, 'bootstrap_type') || isempty(options.bootstrap_type)
    options.bootstrap_type = 'block';
else
    if ~ismember(lower(options.bootstrap_type), {'block', 'stationary'})
        error('OPTIONS.bootstrap_type must be either ''block'' or ''stationary''.');
    end
    options.bootstrap_type = lower(options.bootstrap_type);
end

% Default number of bootstrap replications
if ~isfield(options, 'replications') || isempty(options.replications)
    options.replications = 1000;
else
    opt_struct = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    options.replications = parametercheck(options.replications, 'options.replications', opt_struct);
end

% Default confidence level
if ~isfield(options, 'conf_level') || isempty(options.conf_level)
    options.conf_level = 0.95;
else
    opt_struct = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.conf_level = parametercheck(options.conf_level, 'options.conf_level', opt_struct);
end

% Default confidence interval method
if ~isfield(options, 'method') || isempty(options.method)
    options.method = 'percentile';
else
    valid_methods = {'percentile', 'basic', 'studentized', 'bc', 'bca'};
    if ~ismember(lower(options.method), valid_methods)
        error(['OPTIONS.method must be one of: ', strjoin(valid_methods, ', ')]);
    end
    options.method = lower(options.method);
end

% Set block size or probability parameter based on bootstrap type
[T, ~] = size(data);
if strcmp(options.bootstrap_type, 'block')
    % Block bootstrap parameters
    if ~isfield(options, 'block_size') || isempty(options.block_size)
        % Default block size - a heuristic commonly used is T^(1/3)
        options.block_size = ceil(T^(1/3));
    else
        opt_struct = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
        options.block_size = parametercheck(options.block_size, 'options.block_size', opt_struct);
        if options.block_size >= T
            error('Block size must be less than the sample size.');
        end
    end
else 
    % Stationary bootstrap parameters
    if ~isfield(options, 'p') || isempty(options.p)
        % Default p parameter - inverse of optimal block size
        options.p = 1 / ceil(T^(1/3));
    else
        opt_struct = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
        options.p = parametercheck(options.p, 'options.p', opt_struct);
    end
end

% Calculate the original statistic
original_stat = statistic_fn(data);

% Check that the statistic returns a scalar
if ~isscalar(original_stat)
    error('STATISTIC_FN must return a scalar value.');
end

% Generate bootstrap samples
if strcmp(options.bootstrap_type, 'block')
    bs_data = block_bootstrap(data, options.block_size, options.replications);
else
    bs_data = stationary_bootstrap(data, options.p, options.replications);
end

% Calculate bootstrap statistics
bs_stats = zeros(options.replications, 1);
for b = 1:options.replications
    bs_sample = bs_data(:, :, b);
    bs_stats(b) = statistic_fn(bs_sample);
end

% Create results structure with bootstrap statistics
results = struct(...
    'original_statistic', original_stat, ...
    'conf_level', options.conf_level, ...
    'bootstrap_statistics', bs_stats, ...
    'method', options.method, ...
    'bootstrap_options', options ...
);

% Calculate alpha for confidence interval (e.g., 0.05 for 95% confidence)
alpha = 1 - options.conf_level;

% Calculate confidence intervals based on the specified method
switch options.method
    case 'percentile'
        % Percentile method: direct percentiles of bootstrap distribution
        probs = [alpha/2, 1-alpha/2];
        quantiles = prctile(bs_stats, 100 * probs);
        results.lower = quantiles(1);
        results.upper = quantiles(2);
        
    case 'basic'
        % Basic method: symmetric around original estimate
        probs = [alpha/2, 1-alpha/2];
        quantiles = prctile(bs_stats, 100 * probs);
        results.lower = 2 * original_stat - quantiles(2);
        results.upper = 2 * original_stat - quantiles(1);
        
    case 'studentized'
        % Studentized method: accounts for variability in bootstrap estimates
        % Estimate standard errors using a nested bootstrap
        bs_std_errs = zeros(options.replications, 1);
        
        % Number of nested bootstrap samples (reduced for performance)
        nested_reps = min(25, max(10, floor(options.replications / 20)));
        
        for b = 1:options.replications
            bs_sample = bs_data(:, :, b);
            
            % Generate nested bootstrap samples
            if strcmp(options.bootstrap_type, 'block')
                nested_data = block_bootstrap(bs_sample, options.block_size, nested_reps);
            else
                nested_data = stationary_bootstrap(bs_sample, options.p, nested_reps);
            end
            
            % Calculate statistics on nested samples
            nested_stats = zeros(nested_reps, 1);
            for j = 1:nested_reps
                nested_stats(j) = statistic_fn(nested_data(:, :, j));
            end
            
            % Compute standard error (standard deviation of nested statistics)
            bs_std_errs(b) = std(nested_stats);
        end
        
        % Calculate studentized statistics
        t_stats = (bs_stats - original_stat) ./ bs_std_errs;
        
        % Get critical values from studentized statistics
        t_stats_sorted = sort(t_stats);
        idx_lower = floor((alpha/2) * options.replications) + 1;
        idx_upper = ceil((1-alpha/2) * options.replications);
        
        % Ensure indices are within bounds
        idx_lower = max(1, idx_lower);
        idx_upper = min(options.replications, idx_upper);
        
        t_lower = t_stats_sorted(idx_lower);
        t_upper = t_stats_sorted(idx_upper);
        
        % Estimate standard error of original statistic from bootstrap
        se_original = std(bs_stats);
        
        % Compute confidence interval
        results.lower = original_stat - t_upper * se_original;
        results.upper = original_stat - t_lower * se_original;
        
    case 'bc'
        % Bias-corrected method
        % Calculate bias correction factor z0
        proportion_below = mean(bs_stats < original_stat);
        
        % Handle edge cases
        if proportion_below == 0
            proportion_below = 1 / (2 * options.replications);
        elseif proportion_below == 1
            proportion_below = 1 - 1 / (2 * options.replications);
        end
        
        z0 = norminv(proportion_below);
        
        % Calculate adjusted percentiles
        z_alpha = norminv(alpha/2);
        z_1_alpha = norminv(1-alpha/2);
        
        % Bias-corrected percentiles
        p_lower = normcdf(2*z0 + z_alpha);
        p_upper = normcdf(2*z0 + z_1_alpha);
        
        % Get confidence interval from adjusted percentiles
        results.lower = prctile(bs_stats, 100 * p_lower);
        results.upper = prctile(bs_stats, 100 * p_upper);
        
    case 'bca'
        % Bias-corrected and accelerated method
        % Calculate bias correction factor z0
        proportion_below = mean(bs_stats < original_stat);
        
        % Handle edge cases
        if proportion_below == 0
            proportion_below = 1 / (2 * options.replications);
        elseif proportion_below == 1
            proportion_below = 1 - 1 / (2 * options.replications);
        end
        
        z0 = norminv(proportion_below);
        
        % Calculate acceleration factor using jackknife
        n = size(data, 1);
        jack_stats = zeros(n, 1);
        
        for i = 1:n
            jack_sample = data;
            jack_sample(i, :) = []; % Remove i-th observation
            jack_stats(i) = statistic_fn(jack_sample);
        end
        
        % Calculate acceleration factor
        jack_mean = mean(jack_stats);
        num = sum((jack_mean - jack_stats).^3);
        denom = 6 * (sum((jack_mean - jack_stats).^2))^(3/2);
        
        % Avoid division by zero
        if abs(denom) < 1e-10
            a = 0;
        else
            a = num / denom;
        end
        
        % Calculate adjusted percentiles
        z_alpha = norminv(alpha/2);
        z_1_alpha = norminv(1-alpha/2);
        
        % BCa percentiles
        p_lower = normcdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)));
        p_upper = normcdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)));
        
        % Get confidence interval from adjusted percentiles
        results.lower = prctile(bs_stats, 100 * p_lower);
        results.upper = prctile(bs_stats, 100 * p_upper);
end

end