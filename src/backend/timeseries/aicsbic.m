function ic = aicsbic(logL, k, T)
% AICSBIC Calculates AIC and SBIC information criteria for time series model selection
%
% USAGE:
%   ic = aicsbic(logL, k, T)
%
% INPUTS:
%   logL  - Log-likelihood value(s) of the model(s)
%   k     - Number of parameters in the model
%   T     - Sample size used for estimation
%
% OUTPUTS:
%   ic    - Structure with fields:
%           ic.aic  - Akaike Information Criterion
%           ic.sbic - Schwarz Bayesian Information Criterion (also known as BIC)
%
% COMMENTS:
%   Lower values of both criteria indicate better models, with SBIC penalizing
%   model complexity more heavily than AIC. Both criteria are calculated as:
%   
%   AIC  = -2*logL + 2*k
%   SBIC = -2*logL + k*log(T)
%
%   The function supports vectorized operation:
%   - If logL is a vector, both k and T should be scalars, and ic will contain
%     vectors of information criteria for multiple models with the same parameter
%     count and sample size.
%   - If logL, k, and T are all vectors of the same length, the function calculates
%     criteria for multiple models with different parameter counts and sample sizes.
%
% See also ARMAXFILTER, PARAMETERCHECK

% Check number of inputs
if nargin ~= 3
    error('Three inputs required: logL, k, and T.');
end

% Validate logL (log-likelihood) using parametercheck to ensure it's numeric, non-empty, and finite
logL = parametercheck(logL, 'logL');

% Validate k (number of parameters) using parametercheck to ensure it's a positive integer
options = struct('isPositive', true, 'isInteger', true);
k = parametercheck(k, 'k', options);

% Validate T (sample size) using parametercheck to ensure it's a positive integer
options = struct('isPositive', true, 'isInteger', true);
T = parametercheck(T, 'T', options);

% Calculate AIC = -2*logL + 2*k
ic.aic = -2 * logL + 2 * k;

% Calculate SBIC = -2*logL + k*log(T)
ic.sbic = -2 * logL + k * log(T);

end