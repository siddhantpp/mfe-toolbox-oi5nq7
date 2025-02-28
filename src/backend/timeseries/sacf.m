function [acf, varargout] = sacf(data, varargin)
% SACF Computes the sample autocorrelation function for time series data
%
% USAGE:
%   [ACF] = sacf(DATA)
%   [ACF] = sacf(DATA, LAGS)
%   [ACF] = sacf(DATA, LAGS, OPTIONS)
%   [ACF, SE] = sacf(DATA, ...)
%   [ACF, SE, CI] = sacf(DATA, ...)
%
% INPUTS:
%   DATA       - T by 1 vector of data
%   LAGS       - [OPTIONAL] Scalar integer or N by 1 vector of lags to include in the ACF.
%                Default is 1:min(20,floor(T/4)).
%   OPTIONS    - [OPTIONAL] Options structure with fields:
%                  'alpha'  - [OPTIONAL] Scalar in (0,1) for the confidence interval.
%                             Default is 0.05 (95% confidence).
%                  'demean' - [OPTIONAL] Logical indicating whether to demean the data.
%                             Default is true.
%
% OUTPUTS:
%   ACF        - N by 1 vector of autocorrelations corresponding to LAGS
%   SE         - [OPTIONAL] N by 1 vector of standard errors for the ACF
%   CI         - [OPTIONAL] N by 2 matrix of confidence intervals for the ACF
%
% COMMENTS:
%   The sample autocorrelation function (ACF) is a fundamental tool for identifying 
%   patterns in time series data. It measures the correlation between observations
%   separated by different time lags. For financial time series, the ACF helps identify
%   serial dependence and is crucial for model identification and diagnostic checking.
%
%   The ACF at lag k is calculated as:
%   ACF(k) = Cov(x_t, x_{t-k}) / Var(x_t)
%
%   For a stationary process, the sample ACF is approximately normally distributed 
%   with a standard error of 1/sqrt(T) where T is the sample size.
%
%   The confidence intervals are calculated using the normal approximation:
%   CI = ACF ± z_(α/2) * SE
%   where z_(α/2) is the (1-α/2) quantile of the standard normal distribution.
%
% EXAMPLES:
%   % Basic usage with default lags
%   acf = sacf(returns);
%   
%   % With specific lags
%   acf = sacf(returns, [1 2 3 4 5]);
%   
%   % With standard errors
%   [acf, se] = sacf(returns);
%   
%   % With confidence intervals (99% confidence)
%   [acf, se, ci] = sacf(returns, [], struct('alpha', 0.01));
%
%   % Without demeaning the data
%   acf = sacf(returns, [], struct('demean', false));
%
% See also columncheck, datacheck, parametercheck, spacf

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

% Prepare output arrays
acf = zeros(length(lags), 1);

% Calculate sample variance (denominator for all autocorrelations)
% Using the unbiased estimator with T as the normalization factor
variance = sum(data.^2) / T;

% Calculate autocorrelations for each lag
for i = 1:length(lags)
    lag = lags(i);
    if lag == 0
        acf(i) = 1;  % Autocorrelation at lag 0 is always 1
    elseif lag >= T
        acf(i) = 0;  % Set autocorrelation to 0 if lag exceeds sample size
    else
        % Calculate autocovariance for the lag
        covariance = sum(data(lag+1:T) .* data(1:T-lag)) / T;
        
        % Normalize by variance to get autocorrelation
        acf(i) = covariance / variance;
    end
end

% Calculate standard errors and confidence intervals if requested
if nargout > 1
    % Standard error calculation for ACF
    % For a stationary process, standard error is approximately 1/sqrt(T)
    se = ones(length(lags), 1) / sqrt(T);
    varargout{1} = se;
    
    % Calculate confidence intervals if requested
    if nargout > 2
        z = norminv(1 - options.alpha/2);  % Two-sided confidence interval
        ci = [acf - z*se, acf + z*se];
        varargout{2} = ci;
    end
end
end