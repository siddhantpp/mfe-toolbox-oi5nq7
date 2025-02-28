function results = bds_test(data, dimensions, epsilon, options)
% BDS_TEST Performs the BDS test for independence in time series data
%
% The BDS (Brock-Dechert-Scheinkman) test is a powerful non-parametric test
% for detecting non-linear serial dependence in time series. It examines 
% the spatial correlation of points in reconstructed phase space to detect 
% various types of dependencies that linear models cannot capture.
%
% USAGE:
%   results = bds_test(data)
%   results = bds_test(data, dimensions)
%   results = bds_test(data, dimensions, epsilon)
%   results = bds_test(data, dimensions, epsilon, options)
%
% INPUTS:
%   data        - T×1 vector of time series data
%   dimensions  - [Optional] Vector of embedding dimensions to test (default: 2:5)
%   epsilon     - [Optional] Distance threshold (default: 0.7*std(data))
%   options     - [Optional] Structure with options:
%                   options.alpha     - Significance level for critical values (default: 0.05)
%
% OUTPUTS:
%   results     - Structure with the following fields:
%                   results.stat      - BDS test statistics for each dimension
%                   results.pval      - Asymptotic p-values
%                   results.cv        - Critical values at alpha significance
%                   results.dim       - Embedding dimensions used
%                   results.epsilon   - Epsilon value used
%                   results.nobs      - Number of observations in original series
%                   results.H         - Binary rejection indicators (1: reject independence)
%                   results.message   - Interpretation of test results
%
% REFERENCES:
%   Brock, W. A., W. D. Dechert, and J. A. Scheinkman (1987) "A Test for
%   Independence Based on the Correlation Dimension"
%
% EXAMPLES:
%   % Generate an AR(1) process
%   y = filter(1, [1 -0.5], randn(1000, 1));
%   
%   % Basic BDS test with default parameters
%   results = bds_test(y);
%   
%   % BDS test with custom embedding dimensions
%   results = bds_test(y, [2 3 4 6 8]);
%   
%   % BDS test with custom epsilon
%   results = bds_test(y, 2:5, 0.5*std(y));
%
% See also columncheck, parametercheck, datacheck, normcdf, pdist

% Copyright: MFE Toolbox
% Version 4.0 (October 28, 2009)

% Input validation
data = columncheck(data, 'data');
data = datacheck(data, 'data');

% Set default dimensions if not provided
if nargin < 2 || isempty(dimensions)
    dimensions = 2:5;
else
    % Validate dimensions
    dimensions_options = struct('isInteger', true, 'isPositive', true);
    dimensions = parametercheck(dimensions, 'dimensions', dimensions_options);
end

% Sort dimensions in ascending order
dimensions = sort(dimensions);

% Set default epsilon if not provided (0.7 * standard deviation is common)
if nargin < 3 || isempty(epsilon)
    epsilon = 0.7 * std(data);
else
    % Validate epsilon
    epsilon_options = struct('isscalar', true, 'isPositive', true);
    epsilon = parametercheck(epsilon, 'epsilon', epsilon_options);
end

% Set default options if not provided
if nargin < 4
    options = struct();
end

% Extract options with defaults
if ~isfield(options, 'alpha')
    options.alpha = 0.05;
else
    alpha_options = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.alpha = parametercheck(options.alpha, 'options.alpha', alpha_options);
end

% Check that we have enough data for the maximum dimension
T = length(data);
max_dim = max(dimensions);
if T <= max_dim
    error('BDS test requires more observations than the maximum embedding dimension');
end

% Initialize results matrices
num_dim = length(dimensions);
bds_stat = zeros(num_dim, 1);
pval = zeros(num_dim, 1);
H = zeros(num_dim, 1);
cval = zeros(num_dim, 1);

% Calculate the correlation integral for dimension 1
c1 = correlation_integral(data, 1, epsilon);

% For each dimension, compute BDS statistic
for i = 1:num_dim
    m = dimensions(i);
    
    % Check sufficient sample size for this dimension
    Tm = T - m + 1;
    if Tm < 100
        warning('BDS:SmallSample', ...
                'Sample size may be too small for reliable asymptotic inference with dimension %d', m);
    end
    
    % Calculate the correlation integral for dimension m
    cm = correlation_integral(data, m, epsilon);
    
    % Calculate the BDS statistic
    % Under the null hypothesis of i.i.d., C(m,ε) should equal [C(1,ε)]^m
    c1_m = c1^m;
    
    % Calculate the asymptotic variance components according to BDS formula
    % Calculate K(m)
    k = 0;
    for j = 1:m-1
        k = k + (m-j) * c1^(2*(m-j));
    end
    
    % Compute the variance
    v_m = 4 * (k + (c1^(2*m)) + ((m-1)^2 * c1^(2*m)) - (m^2 * c1^(2*m-2)));
    
    % Compute the standard error
    std_err = sqrt(v_m / Tm);
    
    % Compute the BDS statistic
    bds_stat(i) = (cm - c1_m) / std_err;
    
    % Compute p-value using normal approximation (two-sided test)
    pval(i) = 2 * (1 - normcdf(abs(bds_stat(i))));
    
    % Compute critical value at alpha significance
    cval(i) = norminv(1-options.alpha/2);
    
    % Determine if null hypothesis is rejected
    H(i) = abs(bds_stat(i)) > cval(i);
end

% Organize the results
results = struct();
results.stat = bds_stat;
results.pval = pval;
results.cv = cval;
results.H = H;
results.dim = dimensions;
results.epsilon = epsilon;
results.nobs = T;
results.alpha = options.alpha;

% Add interpretation message
if any(H)
    results.message = 'The null hypothesis of independence is rejected for at least one embedding dimension.';
else
    results.message = 'The null hypothesis of independence is not rejected for any embedding dimension.';
end

end

% =========================================================================
function c_integral = correlation_integral(data, dimension, epsilon)
% CORRELATION_INTEGRAL Computes the correlation integral for the BDS test
%
% This function calculates the correlation integral C(m,ε) for a given
% embedding dimension m and distance threshold ε. The correlation integral
% measures the probability that any two m-dimensional points in the
% embedded series are within distance ε of each other.
%
% INPUTS:
%   data      - T×1 vector of time series data
%   dimension - Embedding dimension (m)
%   epsilon   - Distance threshold (ε)
%
% OUTPUTS:
%   c_integral - The correlation integral value C(m,ε)

% Number of observations
T = length(data);

% Number of points in embedded space
n = T - dimension + 1;

% Create the embedded matrix (n × dimension) where each row is an m-history
embedded = zeros(n, dimension);
for i = 1:dimension
    embedded(:, i) = data(i:i+n-1);
end

% Compute pairwise distances using the Chebyshev (max) distance
% This is equivalent to the sup norm used in the BDS test
distances = squareform(pdist(embedded, 'chebychev'));

% Count pairs of points with distance less than epsilon
indicator = distances < epsilon;

% Remove the diagonal (distance to self is always 0)
indicator = indicator - diag(diag(indicator));

% The correlation integral is the fraction of pairs with distance < epsilon
c_integral = sum(sum(indicator)) / (n * (n - 1));
end