function bv = bv_compute(returns, options)
% BV_COMPUTE Computes bipower variation (BV) from high-frequency financial return data.
%
% The bipower variation is a jump-robust estimator of integrated variance that uses
% products of adjacent absolute returns. It is particularly useful in high-frequency
% financial econometrics for disentangling the continuous and jump components of 
% price processes.
%
% USAGE:
%   BV = bv_compute(RETURNS)
%   BV = bv_compute(RETURNS, OPTIONS)
%
% INPUTS:
%   RETURNS - T×N matrix of high-frequency returns where T is the number
%             of observations and N is the number of assets
%   OPTIONS - [Optional] Structure containing options for the BV computation
%      OPTIONS.scaleFactor - [Optional] Custom scaling factor to replace π/2
%                           Default: pi/2 (theoretical scaling factor)
%
% OUTPUTS:
%   BV - N×1 vector of bipower variation estimates for each asset
%
% COMMENTS:
%   The mathematical formula for bipower variation is:
%       BV = (π/2) * Σ(|r_t| * |r_{t-1}|)
%
%   The bipower variation converges to the integrated variance in the absence of
%   jumps and provides a consistent estimator of the integrated variance even
%   when jumps are present in the price process.
%
%   This implementation is fully vectorized for efficiency with multiple assets.
%
% EXAMPLES:
%   % Compute BV for a single series of returns
%   bv = bv_compute(returns);
%
%   % Compute BV with custom scaling factor
%   options.scaleFactor = 0.8;
%   bv = bv_compute(returns, options);
%
% See also RV_COMPUTE, JUMP_TEST, REALIZED_SPECTRUM

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Input validation
if nargin < 1
    error('At least one input argument is required.');
end

% Set default options if not provided
if nargin < 2
    options = struct();
end

% Validate returns using datacheck utility
returns = datacheck(returns, 'returns');

% Ensure returns are column vectors if a vector is provided
returns = columncheck(returns, 'returns');

% Get dimensions
[T, N] = size(returns);

% Check if we have sufficient observations
if T < 2
    error('At least 2 observations are required to compute bipower variation.');
end

% Get scaling factor (default is π/2)
if isfield(options, 'scaleFactor')
    scaleFactor = parametercheck(options.scaleFactor, 'options.scaleFactor', ...
        struct('isPositive', true, 'isscalar', true));
else
    scaleFactor = pi/2;
end

% Compute absolute returns
absReturns = abs(returns);

% Create the product of adjacent absolute returns
% This gives a (T-1) × N matrix of products
adjacentProducts = absReturns(1:T-1, :) .* absReturns(2:T, :);

% Sum the products for each asset and apply scaling factor
bv = scaleFactor * sum(adjacentProducts, 1)'; % Transpose to get N×1 vector

% If only one asset and only one output requested, return a scalar instead of a 1×1 vector
if N == 1 && nargout <= 1
    bv = bv(1);
end

end