function r = gedrnd(nu, n, m, mu, sigma)
% GEDRND Generates random numbers from the Generalized Error Distribution (GED).
%
% USAGE:
%   R = gedrnd(NU, N, M)
%   R = gedrnd(NU, N, M, MU, SIGMA)
%
% INPUTS:
%   NU    - Shape parameter controlling tail thickness (NU > 0)
%           NU < 2 implies tails thicker than normal
%           NU = 2 implies normal distribution
%           NU > 2 implies thinner tails than normal
%   N     - Number of rows in the output matrix
%   M     - Number of columns in the output matrix
%   MU    - Location parameter (Optional, Default = 0)
%   SIGMA - Scale parameter (Optional, Default = 1, SIGMA > 0)
%
% OUTPUTS:
%   R     - N x M matrix of random numbers from the GED(NU, MU, SIGMA) distribution
%
% COMMENTS:
%   This function generates random numbers from the Generalized Error Distribution
%   using the inverse cumulative distribution function (inverse CDF) method.
%   The procedure is:
%   1. Generate uniform random numbers
%   2. Transform these to GED random variables using the inverse CDF
%   3. Apply location-scale transformation if needed
%
% EXAMPLES:
%   r = gedrnd(1.5, 100, 1)
%   r = gedrnd(1.5, 100, 1, 0, 2)
%
% See also GEDPDF, GEDCDF, GEDINV, GEDFIT

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate the shape parameter (nu)
options.isPositive = true;
nu = parametercheck(nu, 'NU', options);

% Handle optional inputs and defaults
switch nargin
    case 3
        % Default values: mu = 0, sigma = 1
        mu = 0;
        sigma = 1;
    case 4
        % Default value: sigma = 1
        sigma = 1;
    case 5
        % All parameters provided
    otherwise
        error('3 to 5 input arguments required.');
end

% Validate scale parameter (sigma)
sigma = parametercheck(sigma, 'SIGMA', options);

% Validate input size parameters
options.isInteger = true;
options.isPositive = true;
n = parametercheck(n, 'N', options);
m = parametercheck(m, 'M', options);

% Additional check to ensure n and m are scalar
if ~isscalar(n) || ~isscalar(m)
    error('N and M must be scalar values.');
end

% Generate uniform random numbers between 0 and 1
u = rand(n, m);

% Transform uniform random numbers to GED distributed random variables
% using the inverse CDF method
r = gedinv(u, nu, mu, sigma);

end