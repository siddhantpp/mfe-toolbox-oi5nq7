function loglik = gedloglik(data, nu, mu, sigma)
% GEDLOGLIK Computes the log-likelihood for the Generalized Error Distribution (GED)
%
% USAGE:
%   [LOGLIK] = gedloglik(DATA, NU, MU, SIGMA)
%
% INPUTS:
%   DATA  - A vector of data
%   NU    - Shape parameter, also known as the degrees of freedom
%   MU    - Location parameter (mean)
%   SIGMA - Scale parameter (standard deviation-like)
%
% OUTPUTS:
%   LOGLIK - The log-likelihood of the data given the GED parameters
%
% COMMENTS:
%   Compute log-likelihood for the Generalized Error Distribution.
%   Used primarily in maximum likelihood estimation (gedfit) to optimize
%   distribution parameters.
%
%   The GED probability density function is:
%   f(x) = [nu/(2*sigma*gamma(1/nu))] * exp(-(1/2)*|[(x-mu)/sigma]|^nu)
%
%   Where:
%   - nu > 0 is the shape parameter
%   - mu is the location parameter
%   - sigma > 0 is the scale parameter
%
%   When nu = 2, the GED is equivalent to the normal distribution.
%   When nu < 2, the distribution has heavier tails than the normal.
%   When nu > 2, the distribution has lighter tails than the normal.
%
% EXAMPLES:
%   % Compute log-likelihood for data with NU=1.5, MU=0, SIGMA=1
%   data = randn(100, 1);
%   ll = gedloglik(data, 1.5, 0, 1);
%
% REFERENCES:
%   [1] Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns:
%       A new approach. Econometrica, 59, 347-370.
%
% See also GEDCDF, GEDFIT, GEDINV, GEDPDF, GEDRND, GEDSTAT

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate parameters
nu = parametercheck(nu, 'nu');
mu = parametercheck(mu, 'mu');
sigma = parametercheck(sigma, 'sigma');

% Validate data
data = datacheck(data, 'data');

% Check that NU (shape parameter) is positive
if nu <= 0
    loglik = -inf;
    return;
end

% Check that SIGMA (scale parameter) is positive
if any(sigma <= 0)
    loglik = -inf;
    return;
end

% Handle dimensions for vector data with scalar parameters
T = size(data, 1);
if isscalar(mu) && T > 1
    mu = repmat(mu, T, 1);
end

if isscalar(sigma) && T > 1
    sigma = repmat(sigma, T, 1);
end

% Standardize the data
z = abs((data - mu) ./ sigma);

% Compute log-likelihood using GED formula
% Formula: ln(f(x)) = ln(nu/(2*sigma*gamma(1/nu))) - (1/2)*|(x-mu)/sigma|^nu
const = log(nu) - log(2) - log(sigma) - log(gamma(1/nu));
loglik = sum(const - 0.5 * (z.^nu));

end