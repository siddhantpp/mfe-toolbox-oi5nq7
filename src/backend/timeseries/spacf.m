function [pacf, varargout] = spacf(data, varargin)
% SPACF Computes the sample partial autocorrelation function for time series data
%
% USAGE:
%   [PACF] = spacf(DATA)
%   [PACF] = spacf(DATA, LAGS)
%   [PACF] = spacf(DATA, LAGS, OPTIONS)
%   [PACF, SE] = spacf(DATA, ...)
%   [PACF, SE, CI] = spacf(DATA, ...)
%
% INPUTS:
%   DATA       - T by 1 vector of data
%   LAGS       - [OPTIONAL] Scalar integer or N by 1 vector of lags to include in the PACF.
%                Default is 1:min(20,floor(T/4)).
%   OPTIONS    - [OPTIONAL] Options structure with fields:
%                  'alpha'  - [OPTIONAL] Scalar in (0,1) for the confidence interval.
%                             Default is 0.05 (95% confidence).
%                  'demean' - [OPTIONAL] Logical indicating whether to demean the data.
%                             Default is true.
%
% OUTPUTS:
%   PACF       - N by 1 vector of partial autocorrelations corresponding to LAGS
%   SE         - [OPTIONAL] N by 1 vector of standard errors for the PACF
%   CI         - [OPTIONAL] N by 2 matrix of confidence intervals for the PACF
%
% COMMENTS:
%   The sample partial autocorrelation function (PACF) measures the correlation 
%   between observations separated by a lag k after removing the effects of 
%   observations at intermediate lags (1 to k-1).
%
%   PACF at lag k is the partial correlation between y_t and y_{t-k} after
%   controlling for y_{t-1}, y_{t-2}, ..., y_{t-k+1}.
%
%   For a stationary process, the sample PACF at lag k is the last coefficient 
%   in an AR(k) model. The PACF is computed using the Yule-Walker equations.
%
%   For model identification:
%   - In an AR(p) process, PACF cuts off after lag p
%   - In an MA(q) process, PACF tails off to zero
%   - In an ARMA(p,q) process, PACF tails off to zero
%
%   The standard error for PACF is approximately 1/sqrt(T) for lag > p in an AR(p) process.
%   
%   The confidence intervals are calculated using the normal approximation:
%   CI = PACF ± z_(α/2) * SE
%   where z_(α/2) is the (1-α/2) quantile of the standard normal distribution.
%
% EXAMPLES:
%   % Basic usage with default lags
%   pacf = spacf(returns);
%   
%   % With specific lags
%   pacf = spacf(returns, [1 2 3 4 5]);
%   
%   % With standard errors
%   [pacf, se] = spacf(returns);
%   
%   % With confidence intervals (99% confidence)
%   [pacf, se, ci] = spacf(returns, [], struct('alpha', 0.01));
%
%   % Without demeaning the data
%   pacf = spacf(returns, [], struct('demean', false));
%
% See also columncheck, datacheck, parametercheck, sacf

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Get data length
T = length(data);

% Process lags parameter
if nargin > 1 && ~isempty(varargin{1})
    lags = varargin{1};
    % Validate lags
    lagOptions = struct('isInteger', true, 'isNonNegative', true);
    lags = parametercheck(lags, 'lags', lagOptions);
    
    % If scalar, expand to vector 1:lags
    if isscalar(lags)
        lags = (1:lags)';
    else
        % Ensure column vector format
        lags = columncheck(lags, 'lags');
    end
else
    % Default lags: 1:min(20,floor(T/4))
    maxLag = min(20, floor(T/4));
    lags = (1:maxLag)';
end

% Process options parameter
options = struct('alpha', 0.05, 'demean', true);
if nargin > 2 && ~isempty(varargin{2})
    userOptions = varargin{2};
    
    % Process alpha option
    if isfield(userOptions, 'alpha')
        alphaOptions = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
        options.alpha = parametercheck(userOptions.alpha, 'alpha', alphaOptions);
    end
    
    % Process demean option
    if isfield(userOptions, 'demean')
        options.demean = userOptions.demean;
        if ~islogical(options.demean) && ~(options.demean==0 || options.demean==1)
            error('demean must be a logical value (true/false)');
        end
    end
end

% Center data if required
if options.demean
    data = data - mean(data);
end

% Determine the maximum lag needed
maxLag = max(lags);

% Calculate sample autocorrelation function up to maximum lag
acf_values = sacf(data, 0:maxLag, struct('demean', false));

% Prepare output arrays for the requested lags
pacf = zeros(length(lags), 1);

% Compute PACF for each requested lag
for i = 1:length(lags)
    k = lags(i);
    
    if k == 0
        % PACF at lag 0 is 1 by definition
        pacf(i) = 1;
    elseif k == 1
        % PACF at lag 1 is the same as ACF at lag 1
        pacf(i) = acf_values(2);  % acf_values(1) is lag 0 (always 1)
    else
        % For lags > 1, solve Yule-Walker equations to get AR coefficients
        % Create Toeplitz matrix of autocorrelations for lags 0 to k-1
        R = toeplitz(acf_values(1:k));  % k x k Toeplitz matrix
        
        % Right-hand side vector: autocorrelations at lags 1 to k
        b = acf_values(2:k+1);
        
        % Solve Yule-Walker equations: R*phi = b
        phi = R \ b;
        
        % The PACF at lag k is the last coefficient (phi_k)
        pacf(i) = phi(end);
    end
end

% Calculate standard errors and confidence intervals if requested
if nargout > 1
    % Standard error calculation for PACF
    % For lags > p in an AR(p) process, SE is approximately 1/sqrt(T)
    se = ones(length(lags), 1) / sqrt(T);
    varargout{1} = se;
    
    % Calculate confidence intervals if requested
    if nargout > 2
        z = norminv(1 - options.alpha/2);  % Two-sided confidence interval
        ci = [pacf - z*se, pacf + z*se];
        varargout{2} = ci;
    end
end
end