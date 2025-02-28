function bsdata = block_bootstrap(data, block_length, B, circular)
% BLOCK_BOOTSTRAP Implements block bootstrap resampling for dependent time series data
%
% USAGE:
%   BSDATA = block_bootstrap(DATA, BLOCK_LENGTH, B)
%   BSDATA = block_bootstrap(DATA, BLOCK_LENGTH, B, CIRCULAR)
%
% INPUTS:
%   DATA         - T by N matrix of data to be resampled
%   BLOCK_LENGTH - Positive integer containing the length of blocks
%   B            - Positive integer containing number of bootstrap samples
%   CIRCULAR     - [Optional] Boolean indicating whether to use circular 
%                  block bootstrap (true) or truncated block bootstrap (false).
%                  Default is true.
%
% OUTPUTS:
%   BSDATA       - T by N by B 3-dimensional array of bootstrapped data
%
% COMMENTS:
%   Block bootstrap is designed for dependent data where the dependency
%   decreases with the distance between observations. Block bootstrap
%   samples blocks of length BLOCK_LENGTH from the original data to form 
%   a bootstrap sample that preserves the temporal dependence structure 
%   within each block.
%
%   If CIRCULAR is true (default), the method uses circular block bootstrap
%   where blocks can wrap around the end of the data. If false, blocks that
%   would extend beyond the end of the data are truncated.
%
% EXAMPLES:
%   % Generate 1000 bootstrap samples with block length 10
%   returns = randn(100, 1);
%   bs_returns = block_bootstrap(returns, 10, 1000);
%
%   % Compute bootstrap confidence interval for the mean
%   bs_means = squeeze(mean(bs_returns));
%   ci = prctile(bs_means, [2.5 97.5]);
%
% See also stationary_bootstrap, datacheck, parametercheck, columncheck
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate data using datacheck
data = datacheck(data, 'data');

% Get dimensions of the input data
[T, N] = size(data);

% If data is a vector, use columncheck to ensure it's a column vector
if min(T, N) == 1
    data = columncheck(data, 'data');
    [T, N] = size(data);  % Update dimensions after column check
end

% Validate block_length parameter
options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
block_length = parametercheck(block_length, 'block_length', options);

% Validate B parameter (number of bootstrap samples)
B = parametercheck(B, 'B', options);

% Set default for circular parameter if not provided
if nargin < 4 || isempty(circular)
    circular = true;
elseif ~islogical(circular) && ~(isnumeric(circular) && (circular == 0 || circular == 1))
    error('CIRCULAR must be a logical value (true or false).');
end

% Ensure block_length is less than the sample size
if block_length >= T
    error('BLOCK_LENGTH must be less than the length of the data.');
end

% Initialize the output array
bsdata = zeros(T, N, B);

% Calculate the number of blocks needed to cover the sample length
num_blocks = ceil(T / block_length);

% Generate bootstrap samples
for b = 1:B
    % Generate random block starting positions
    start_indices = randi(T, num_blocks, 1);
    
    % Initialize index for the bootstrap sample
    bs_idx = 1;
    
    % Create bootstrap sample by assembling blocks
    for i = 1:num_blocks
        % Skip if we've already filled the bootstrap sample
        if bs_idx > T
            break;
        end
        
        % Starting position for this block
        start_idx = start_indices(i);
        
        % Calculate the remaining space in the bootstrap sample
        remaining = T - bs_idx + 1;
        
        % Determine the actual block size (may be smaller than block_length if we're near the end)
        actual_block_size = min(block_length, remaining);
        
        % Handle circular blocks
        if circular
            % For circular blocks, wrap around the end of the data
            indices = mod((start_idx:start_idx+actual_block_size-1)-1, T) + 1;
        else
            % For non-circular blocks, truncate at the end of the data
            end_idx = start_idx + actual_block_size - 1;
            if end_idx > T
                end_idx = T;
                actual_block_size = end_idx - start_idx + 1;
            end
            indices = start_idx:end_idx;
        end
        
        % Add the block to the bootstrap sample
        bsdata(bs_idx:bs_idx+actual_block_size-1, :, b) = data(indices, :);
        
        % Update index
        bs_idx = bs_idx + actual_block_size;
    end
end