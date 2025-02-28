function backCastValue = backcast(data, options)
% BACKCAST Computes initial variance (backcast) values for GARCH model estimation
%
% USAGE:
%   BACKCASTVALUE = backcast(DATA)
%   BACKCASTVALUE = backcast(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - Time series data for which to compute the backcast, T by 1 column vector
%              or T by N matrix of N series
%   OPTIONS  - [OPTIONAL] Struct with fields:
%              'type' - String indicating the backcast method:
%                  'default' - Sample variance (default if not specified)
%                  'EWMA'    - Exponentially Weighted Moving Average
%                  'decay'   - Decay factor applied to weighted sum of squared observations
%                  'fixed'   - Fixed value specified in 'value' field
%              'value' - For fixed type, the value to use as backcast
%              'lambda' - For EWMA, the decay factor (default = 0.94)
%              'decay' - For decay method, the decay factor (default = 0.7)
%
% OUTPUTS:
%   BACKCASTVALUE - Initial variance estimate (backcast value) for GARCH models.
%                   Scalar for vector input, N by 1 vector for matrix input.
%
% COMMENTS:
%   Backcasting is a technique used to initialize variance estimates in GARCH models.
%   It provides robust starting values for volatility recursion, essential for 
%   convergence of GARCH parameter estimation algorithms.
%
%   Different methods are available:
%   1. Default: Uses the sample variance of the data
%   2. EWMA: Applies exponentially weighted moving average to squared data
%   3. Decay: Applies decay factor to weighted sum of squared observations
%   4. Fixed: Uses a user-specified fixed value
%
% EXAMPLES:
%   % Default method (sample variance)
%   backcastValue = backcast(returns);
%
%   % EWMA method with custom lambda
%   options = struct('type', 'EWMA', 'lambda', 0.95);
%   backcastValue = backcast(returns, options);
%
%   % Fixed value method
%   options = struct('type', 'fixed', 'value', 0.001);
%   backcastValue = backcast(returns, options);
%
% See also GARCH, APARCH, EGARCH, GJR, TARCH

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Input validation
% Validate data format (column vector or matrix)
data = columncheck(data, 'data');

% Validate numerical properties
data = datacheck(data, 'data');

% Get dimensions of data
[T, numSeries] = size(data);

% Check if options provided, create default if not
if nargin < 2 || isempty(options)
    options = struct('type', 'default');
end

% Ensure options is a structure
if ~isstruct(options)
    error('OPTIONS must be a structure.');
end

% Initialize backcast value(s)
backCastValue = zeros(numSeries, 1);

% Apply backcasting method to each series
for i = 1:numSeries
    seriesData = data(:, i);
    
    % Determine which backcasting method to use
    if ~isfield(options, 'type')
        backcastType = 'default';
    else
        backcastType = lower(options.type);
    end
    
    switch backcastType
        case 'default'
            % Method 1: Default - Use sample variance
            % Using unbiased estimator (divided by T-1)
            backCastValue(i) = var(seriesData);
            
        case 'ewma'
            % Method 2: EWMA - Exponentially Weighted Moving Average
            % Set default lambda if not provided
            if ~isfield(options, 'lambda')
                lambda = 0.94; % Default RiskMetrics value
            else
                lambda = options.lambda;
                if lambda <= 0 || lambda >= 1
                    error('LAMBDA must be between 0 and 1 for EWMA backcast method.');
                end
            end
            
            % Apply EWMA to squared returns
            squaredData = seriesData.^2;
            weights = (1-lambda) * lambda.^(T-1:-1:0)';
            backCastValue(i) = weights' * squaredData;
            
        case 'decay'
            % Method 3: Decay - Apply decay factor to squared observations
            % Set default decay factor if not provided
            if ~isfield(options, 'decay')
                decayFactor = 0.7; % Default decay factor
            else
                decayFactor = options.decay;
                if decayFactor <= 0 || decayFactor >= 1
                    error('DECAY must be between 0 and 1 for decay backcast method.');
                end
            end
            
            % Apply decay to squared returns
            squaredData = seriesData.^2;
            weights = decayFactor.^((T-1):-1:0)';
            weights = weights / sum(weights); % Normalize weights
            backCastValue(i) = weights' * squaredData;
            
        case 'fixed'
            % Method 4: Fixed - Use specified fixed value
            if ~isfield(options, 'value')
                error('OPTIONS.value must be provided when using fixed backcast method.');
            end
            
            fixedValue = options.value;
            if ~isnumeric(fixedValue) || ~isscalar(fixedValue) || fixedValue <= 0
                error('OPTIONS.value must be a positive scalar for fixed backcast method.');
            end
            
            backCastValue(i) = fixedValue;
            
        otherwise
            error('Unknown backcast type. Supported types are: default, EWMA, decay, fixed.');
    end
    
    % Ensure backcast value is positive (for numerical stability)
    if backCastValue(i) <= 0
        % Fall back to absolute mean if variance is zero or negative
        backCastValue(i) = mean(abs(seriesData))^2;
        
        % If still zero or negative, use small positive value
        if backCastValue(i) <= 0
            backCastValue(i) = 1e-6;
        end
    end
end

% If input was a vector, return scalar
if numSeries == 1
    backCastValue = backCastValue(1);
end

end