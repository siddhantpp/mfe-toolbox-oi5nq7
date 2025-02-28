function [ll] = stdtloglik(data, nu, mu, sigma)
% STDTLOGLIK Computes the negative log-likelihood for standardized Student's t
%
% USAGE:
%   [LL] = stdtloglik(DATA, NU, MU, SIGMA)
%
% INPUTS:
%   DATA   - A column vector (or row vector which will be transformed to column) 
%            of data to be evaluated under the standardized t distribution
%   NU     - Degrees of freedom parameter, must be > 2 for standardized t
%   MU     - Location parameter (mean)
%   SIGMA  - Scale parameter (standard deviation-like), must be positive
%
% OUTPUTS:
%   LL     - Negative log-likelihood value, suitable for minimization in estimation
%
% COMMENTS:
%   Calculates the log-likelihood for the standardized Student's t distribution.
%   The standardized t has mean 0 and variance 1 when NU > 2.
%   
%   For numerical stability, this function can use two mathematically equivalent
%   forms for the log-likelihood calculation.
%
% REFERENCES:
%   [1] Bollerslev, T. (1987). A conditionally heteroskedastic time series model for 
%       speculative prices and rates of return. The Review of Economics and Statistics.
%
% See also STDTFIT, STDTRND, PARAMETERCHECK, DATACHECK, COLUMNCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
if nargin ~= 4
    error('Four inputs required: DATA, NU, MU, SIGMA');
end

% Parameter validation
options.isscalar = true;
options.isPositive = true;
options.lowerBound = 2;
nu = parametercheck(nu, 'NU', options);

% Check location (mu) parameter
mu = parametercheck(mu, 'MU');

% Check scale (sigma) parameter
options.isPositive = true;
sigma = parametercheck(sigma, 'SIGMA', options);

% Data validation
data = datacheck(data, 'DATA');
data = columncheck(data, 'DATA');

% Initialize log-likelihood to -Inf (will be updated if parameters are valid)
ll = -inf;

% Ensure degrees of freedom > 2 for a standardized t-distribution
if nu <= 2
    return;
end

% Ensure scale parameter is positive
if any(sigma <= 0)
    return;
end

% Handle scalar mu/sigma with vector data through broadcasting
T = size(data, 1);
if isscalar(mu) && T > 1
    mu = mu * ones(T, 1);
end
if isscalar(sigma) && T > 1
    sigma = sigma * ones(T, 1);
end

% Standardize the data
z = (data - mu)./sigma;

% Compute constants for the log-likelihood
halfnuplus1 = (nu + 1) / 2;
halfnu = nu / 2;
logconstant = log(gamma(halfnuplus1)) - log(gamma(halfnu)) - 0.5*log(pi*nu);

% Compute the log-likelihood using the t-distribution formula
% LL = sum(logconstant - halfnuplus1 * log(1 + (z.^2)/nu) - log(sigma));

% Alternative implementation using the beta function for enhanced numerical stability
% Using the relationship: Γ((ν+1)/2) / (Γ(ν/2) * sqrt(πν)) = 1 / (sqrt(ν) * B(1/2, ν/2))
logconstant_alt = -log(sqrt(nu) * beta(0.5, halfnu));
ll = sum(logconstant_alt - halfnuplus1 * log(1 + (z.^2)/nu) - log(sigma));

% Convert to negative log-likelihood for minimization purposes
ll = -ll;
end