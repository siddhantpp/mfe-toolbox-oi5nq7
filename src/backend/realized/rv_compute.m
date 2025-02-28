function rv = rv_compute(returns, options)
% RV_COMPUTE Computes realized volatility (variance) from high-frequency financial return data
%
% The function implements the standard quadratic variation estimator as the sum of squared 
% intraday returns, providing a measure of integrated variance over a specified time period.
%
% USAGE:
%   RV = rv_compute(RETURNS)
%   RV = rv_compute(RETURNS, OPTIONS)
%
% INPUTS:
%   RETURNS - An m by n matrix of high-frequency returns where:
%             m is the number of observations (intraday returns)
%             n is the number of assets or time series
%             
%   OPTIONS - [Optional] A structure with the following fields:
%             scale      - [Optional] Scaling factor for annualization or other adjustment.
%                          Default is 1.
%             adjustSize - [Optional] Logical flag to adjust for the total number of
%                          observations. Default is false.
%             weights    - [Optional] Vector of weights for weighted realized volatility.
%                          Default is equal weighting (vector of ones).
%             method     - [Optional] Method to compute RV: 'standard' (default) or 'subsample'.
%             subSample  - [Optional] Subsampling period when method is 'subsample'. Default is 5.
%             jackknife  - [Optional] Logical flag to use jackknife bias correction. Default is false.
%                          Only applicable when method is 'standard'.
%
% OUTPUTS:
%   RV      - Realized volatility (variance) estimate based on the sum of squared returns.
%             1 by n vector where n is the number of assets or time series.
%
% COMMENTS:
%   The realized volatility is computed as the sum of squared returns, which
%   provides a consistent estimator of integrated variance under suitable 
%   assumptions including no jumps and no microstructure noise.
%
%   For high-frequency data with microstructure noise, the 'subsample' method can
%   provide more robust estimates by averaging multiple subsampled realized volatility
%   estimates.
%
%   When jackknife is true, a jackknife bias correction is applied to reduce the bias
%   from microstructure noise. This is a simple bias correction method and more 
%   sophisticated approaches are available in the rv_kernel function.
%
% EXAMPLES:
%   % Compute realized volatility from a vector of 5-minute returns
%   rv = rv_compute(fiveminreturns);
%
%   % Compute realized volatility with scaling and subsampling
%   options.scale = 252;  % Annual scaling for daily RV
%   options.method = 'subsample';
%   options.subSample = 3;
%   rv = rv_compute(fiveminreturns, options);
%
%   % Compute jackknife bias-corrected realized volatility
%   options.jackknife = true;
%   rv = rv_compute(fiveminreturns, options);
%
% See also BV_COMPUTE, RV_KERNEL, REALIZED_SPECTRUM, JUMP_TEST

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 10/28/2009

%% Input validation
if nargin < 1
    error('At least one input argument (RETURNS) is required.');
end

% Verify returns data is valid
returns = datacheck(returns, 'returns');

% Get dimensions
[m, n] = size(returns);

%% Process options
if nargin < 2 || isempty(options)
    options = struct();
end

% Default options
if ~isfield(options, 'scale')
    options.scale = 1;
else
    % Check if scale is valid
    options.scale = parametercheck(options.scale, 'scale', struct('isscalar', true));
end

if ~isfield(options, 'adjustSize')
    options.adjustSize = false;
else
    if ~islogical(options.adjustSize) && ~(options.adjustSize == 0 || options.adjustSize == 1)
        error('OPTIONS.adjustSize must be a logical value.');
    end
end

if ~isfield(options, 'weights')
    options.weights = ones(m, 1);
else
    if length(options.weights) ~= m
        error('OPTIONS.weights must have the same length as the number of observations in RETURNS.');
    end
    options.weights = columncheck(options.weights, 'weights');
    options.weights = datacheck(options.weights, 'weights');
end

if ~isfield(options, 'method')
    options.method = 'standard';
else
    if ~ischar(options.method) || ~ismember(options.method, {'standard', 'subsample'})
        error('OPTIONS.method must be either ''standard'' or ''subsample''.');
    end
end

if ~isfield(options, 'subSample')
    options.subSample = 5;
else
    paramOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    options.subSample = parametercheck(options.subSample, 'subSample', paramOptions);
    if options.subSample > m/2
        warning('Subsampling period is very large relative to sample size. Results may be unreliable.');
    end
end

if ~isfield(options, 'jackknife')
    options.jackknife = false;
else
    if ~islogical(options.jackknife) && ~(options.jackknife == 0 || options.jackknife == 1)
        error('OPTIONS.jackknife must be a logical value.');
    end
    if options.jackknife && strcmp(options.method, 'subsample')
        warning('Jackknife bias correction is not applied when using subsampling method.');
        options.jackknife = false;
    end
end

%% Compute realized volatility
rv = zeros(1, n);

% Process each column (asset) separately
for i = 1:n
    % Extract returns for the current asset
    assetReturns = returns(:, i);
    
    switch options.method
        case 'standard'
            % Standard realized volatility: weighted sum of squared returns
            squaredReturns = assetReturns.^2;
            rv(i) = sum(options.weights .* squaredReturns);
            
            % Apply size adjustment if requested
            if options.adjustSize
                rv(i) = rv(i) * (m / sum(options.weights));
            end
            
            % Apply jackknife bias correction if requested
            if options.jackknife && m > 2
                % Jackknife bias correction based on first-order autocorrelation
                % This is a simple implementation to reduce bias from microstructure noise
                autocov1 = sum(assetReturns(1:end-1) .* assetReturns(2:end)) / (m - 1);
                biasCorrection = -2 * autocov1 * (m - 1) / m;
                rv(i) = rv(i) + biasCorrection;
                
                % Ensure non-negative result
                rv(i) = max(rv(i), 0);
            end
            
        case 'subsample'
            % Subsampling method to mitigate microstructure noise
            k = options.subSample;
            if k >= m
                warning('Subsampling period exceeds or equals sample size. Using standard method.');
                squaredReturns = assetReturns.^2;
                rv(i) = sum(options.weights .* squaredReturns);
            else
                % Create k subsamples and average RV estimates
                subRV = zeros(k, 1);
                for j = 1:k
                    % Select every k-th observation starting from j
                    idx = j:k:m;
                    subReturns = assetReturns(idx);
                    subWeights = options.weights(idx);
                    
                    % Compute RV for this subsample and scale by k to account for frequency
                    subRV(j) = sum(subWeights .* subReturns.^2) * k;
                    
                    % Apply size adjustment if requested
                    if options.adjustSize
                        subRV(j) = subRV(j) * (length(idx) / sum(subWeights));
                    end
                end
                
                % Average the subsampled RV estimates
                rv(i) = mean(subRV);
            end
    end
    
    % Apply scaling factor
    rv(i) = rv(i) * options.scale;
end

% Ensure output is a row vector for consistency
rv = reshape(rv, 1, n);

end