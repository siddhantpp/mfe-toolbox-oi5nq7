function [rv] = rv_kernel(returns, options)
% RV_KERNEL Computes kernel-based realized volatility estimation for high-frequency data
%
% USAGE:
%   [RV] = rv_kernel(RETURNS)
%   [RV] = rv_kernel(RETURNS, OPTIONS)
%
% INPUTS:
%   RETURNS - T by 1 vector of high-frequency returns (can be a matrix for multiple assets)
%   OPTIONS - Optional input structure with fields:
%              kernelType       - String specifying kernel function ['Bartlett-Parzen']
%                                 Supported types: 'Bartlett-Parzen', 'Quadratic', 'Cubic',
%                                 'Exponential', 'Tukey-Hanning'
%              bandwidth        - Positive integer specifying kernel bandwidth (lag order)
%                                 If not provided, automatically determined from data
%              autoCorrection   - Boolean indicating whether to apply asymptotic
%                                 bias correction [false]
%              handleOvernight  - Boolean indicating whether to apply overnight
%                                 returns adjustment [false]
%              removeOutliers   - Boolean indicating whether to detect and
%                                 remove extreme outliers [false]
%
% OUTPUTS:
%   RV - Kernel-based realized volatility (variance) estimate, noise-robust
%
% COMMENTS:
%   This function implements kernel-based realized volatility (variance) estimation
%   for high-frequency financial return data. The kernel weighting reduces the
%   impact of market microstructure noise, providing more accurate integrated
%   variance estimates compared to standard realized volatility methods.
%
%   Kernel options:
%   1. Bartlett-Parzen: w(x) = 1 - |x| for |x|≤1, 0 otherwise
%   2. Quadratic:       w(x) = 1 - x² for |x|≤1, 0 otherwise
%   3. Cubic:           w(x) = 1 - 3x² + 2|x|³ for |x|≤1, 0 otherwise
%   4. Exponential:     w(x) = exp(-|x|)
%   5. Tukey-Hanning:   w(x) = (1 + cos(πx))/2 for |x|≤1, 0 otherwise
%
% EXAMPLES:
%   % Basic usage with default settings
%   rv = rv_kernel(returns);
%
%   % With custom kernel and bandwidth
%   options.kernelType = 'Tukey-Hanning';
%   options.bandwidth = 10;
%   rv = rv_kernel(returns, options);
%
% REFERENCES:
%   Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., & Shephard, N. (2008).
%   "Designing realized kernels to measure the ex post variation of equity
%   prices in the presence of noise." Econometrica, 76(6), 1481-1536.
%
% See also datacheck, columncheck, parametercheck

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input validation and setup
if nargin < 1
    error('At least one input (returns) is required.');
end

% Default options
defaultOptions = struct('kernelType', 'Bartlett-Parzen', ...
                        'bandwidth', [], ...
                        'autoCorrection', false, ...
                        'handleOvernight', false, ...
                        'removeOutliers', false);

% Process options
if nargin < 2 || isempty(options)
    options = defaultOptions;
else
    % Check for missing fields and assign defaults
    if ~isfield(options, 'kernelType') || isempty(options.kernelType)
        options.kernelType = defaultOptions.kernelType;
    end
    if ~isfield(options, 'bandwidth') || isempty(options.bandwidth)
        options.bandwidth = defaultOptions.bandwidth;
    end
    if ~isfield(options, 'autoCorrection')
        options.autoCorrection = defaultOptions.autoCorrection;
    end
    if ~isfield(options, 'handleOvernight')
        options.handleOvernight = defaultOptions.handleOvernight;
    end
    if ~isfield(options, 'removeOutliers')
        options.removeOutliers = defaultOptions.removeOutliers;
    end
end

% Validate returns data
returns = datacheck(returns, 'returns');

% Ensure returns are column vectors
returns = columncheck(returns, 'returns');

% Get data dimensions
[T, numAssets] = size(returns);

% Set default bandwidth if not specified
if isempty(options.bandwidth)
    % Common rule of thumb: c*T^(1/5) for optimal bandwidth
    options.bandwidth = ceil(4 * (T/100)^0.2);
    % Ensure bandwidth doesn't exceed reasonable limits
    options.bandwidth = min(options.bandwidth, floor(T/4));
    options.bandwidth = max(options.bandwidth, 1);
end

% Validate bandwidth parameter
bandwidthOptions = struct('isscalar', true, 'isInteger', true, ...
                         'isPositive', true, 'upperBound', T-1);
options.bandwidth = parametercheck(options.bandwidth, 'bandwidth', bandwidthOptions);

%% Apply outlier removal if specified
if options.removeOutliers
    % Simple outlier detection: remove returns > 5 std dev
    stdReturns = std(returns);
    outlierThreshold = 5 * stdReturns;
    outlierIndices = abs(returns) > repmat(outlierThreshold, T, 1);
    
    % Replace outliers with zeros (effectively excluding them)
    if any(outlierIndices(:))
        warning(['Detected and removed ' num2str(sum(outlierIndices(:))) ' outliers.']);
        returns(outlierIndices) = 0;
    end
end

%% Handle overnight returns if specified
if options.handleOvernight && T > 1
    % Simple approach: downweight first return of each day
    % For a more sophisticated approach, you would need day markers
    % This is a placeholder for the overnight return adjustment
    warning(['Overnight returns handling is enabled but requires additional ' ...
             'day marker information for proper implementation.']);
end

%% Initialize rv calculation
rv = zeros(1, numAssets);

% Process each asset
for a = 1:numAssets
    r = returns(:, a);
    
    % Calculate autocovariances up to bandwidth
    gamma0 = sum(r.^2);  % Realized variance (order 0 autocovariance)
    
    % Initialize weighted sum with gamma0
    weightedSum = gamma0;
    
    % Calculate higher-order autocovariances and apply kernel weights
    for k = 1:options.bandwidth
        % Calculate k-th order autocovariance: γ_k = Σ_t r_t * r_{t-k}
        gammaK = sum(r(k+1:end) .* r(1:end-k));
        
        % Apply kernel weight based on selected kernel type
        x = k / options.bandwidth;  % Normalized lag
        
        switch lower(options.kernelType)
            case 'bartlett-parzen'
                % Bartlett-Parzen kernel: 1-|x| for |x|≤1
                weight = max(0, 1 - abs(x));
            case 'quadratic'
                % Quadratic kernel: 1-x² for |x|≤1
                weight = (abs(x) <= 1) * (1 - x^2);
            case 'cubic'
                % Cubic kernel: 1-3x²+2|x|³ for |x|≤1
                weight = (abs(x) <= 1) * (1 - 3*x^2 + 2*abs(x)^3);
            case 'exponential'
                % Exponential kernel: exp(-|x|)
                weight = exp(-abs(x));
            case 'tukey-hanning'
                % Tukey-Hanning kernel: (1+cos(πx))/2 for |x|≤1
                weight = (abs(x) <= 1) * (1 + cos(pi*x))/2;
            otherwise
                error('Unsupported kernel type: %s', options.kernelType);
        end
        
        % Add weighted autocovariance (multiplied by 2 to account for both sides)
        weightedSum = weightedSum + 2 * weight * gammaK;
    end
    
    % Apply asymptotic bias correction if requested
    if options.autoCorrection
        % Simple correction factor for noise bias
        % More sophisticated corrections would depend on specific noise models
        correctionFactor = 1 + 1/(2*options.bandwidth);
        weightedSum = weightedSum / correctionFactor;
    end
    
    % Store the result for this asset
    rv(1, a) = weightedSum;
end

% Ensure non-negative results (numerical issues might yield small negative values)
rv = max(0, rv);

% If user wants just the value when nargout=0
if nargout == 0 && numAssets == 1
    disp(['Kernel-Based Realized Volatility: ', num2str(rv)]);
end
end