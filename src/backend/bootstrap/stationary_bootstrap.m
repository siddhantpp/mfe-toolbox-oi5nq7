function [bsdata] = stationary_bootstrap(data, p, B, circular)
% STATIONARY_BOOTSTRAP Implements the stationary bootstrap for time series data
%
% USAGE:
%   [BSDATA] = stationary_bootstrap(DATA, P, B)
%   [BSDATA] = stationary_bootstrap(DATA, P, B, CIRCULAR)
%
% INPUTS:
%   DATA      - T by N matrix of data to be bootstrapped
%   P         - Probability of starting a new block, between 0 and 1
%   B         - Number of bootstrap samples to generate
%   CIRCULAR  - Boolean indicating whether to use circular blocks (default = true)
%
% OUTPUTS:
%   BSDATA    - T by N by B array of bootstrapped data with dimensions:
%               T = number of observations in original series
%               N = number of variables
%               B = number of bootstrap samples
%
% COMMENTS:
%   Implements the stationary bootstrap of Politis & Romano (1994), which
%   generates bootstrap samples by randomly selecting blocks of varying length
%   according to a geometric distribution with parameter p. Unlike the fixed-length
%   block bootstrap, this method ensures the resampled series is stationary,
%   making it particularly suitable for financial time series with
%   heteroskedasticity and temporal dependence.
%
%   The expected block length is 1/p.
%
%   If CIRCULAR is true (default), then blocks that extend beyond the end of the
%   sample wrap around to the beginning, maintaining the original time series
%   dependence structure.
%
% REFERENCES:
%   [1] Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap.
%       Journal of the American Statistical Association, 89(428), 1303-1313.
%
% See also bootstrap, block_bootstrap

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Validate p parameter (must be between 0 and 1)
options.lowerBound = 0;
options.upperBound = 1;
p = parametercheck(p, 'p', options);

% Validate B parameter (must be a positive integer)
options = struct();
options.isInteger = true;
options.isPositive = true;
B = parametercheck(B, 'B', options);

% Set default for circular if not provided
if nargin < 4 || isempty(circular)
    circular = true;
end

% Get dimensions of the data
[T, N] = size(data);

% Initialize output array
bsdata = zeros(T, N, B);

% Generate bootstrap samples
for b = 1:B
    % Generate indices for this bootstrap sample
    indices = zeros(T, 1);
    
    % Generate first index randomly
    indices(1) = randi(T);
    
    % Generate remaining indices
    for t = 2:T
        % With probability p, start a new block
        if rand(1) < p
            indices(t) = randi(T);
        else
            % Continue the current block
            if circular
                % Wrap around to beginning if needed
                indices(t) = mod(indices(t-1), T) + 1;
            else
                % If at the end of data and not circular, start a new block
                if indices(t-1) == T
                    indices(t) = randi(T);
                else
                    indices(t) = indices(t-1) + 1;
                end
            end
        end
    end
    
    % Create bootstrap sample using the generated indices
    bsdata(:, :, b) = data(indices, :);
end

end