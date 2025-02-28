function p = gedcdf(x, nu, mu, sigma)
% GEDCDF Computes the cumulative distribution function (CDF) of the Generalized Error Distribution.
%
% USAGE:
%   P = gedcdf(X, NU)
%   P = gedcdf(X, NU, MU, SIGMA)
%
% INPUTS:
%   X     - Values at which to evaluate the CDF. Either a column vector or a row vector, 
%           which will be converted to a column vector
%   NU    - Shape parameter controlling tail thickness (NU > 0)
%           NU < 2 implies tails thicker than normal
%           NU = 2 implies normal distribution
%           NU > 2 implies thinner tails than normal
%   MU    - Location parameter (Optional, Default = 0)
%   SIGMA - Scale parameter (Optional, Default = 1, SIGMA > 0)
%
% OUTPUTS:
%   P     - CDF values corresponding to each value in X
%
% COMMENTS:
%   The CDF of the Generalized Error Distribution is defined as:
%   For z >= 0: P = 0.5 + 0.5 * gammainc((abs(z)/lambda)^nu / 2, 1/nu)
%   For z < 0:  P = 0.5 - 0.5 * gammainc((abs(z)/lambda)^nu / 2, 1/nu)
%   where z = (x - mu) / (lambda * sigma), and
%   lambda = sqrt(gamma(3/nu)/gamma(1/nu))
%
% EXAMPLES:
%   p = gedcdf(0, 1.5)
%   p = gedcdf([-1 0 1]', 1.5, 0, 2)
%
% See also GEDPDF, GEDINV, GEDRND, GEDFIT, GAMMAINCCORE

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Validate the shape parameter (nu)
options.isPositive = true;
nu = parametercheck(nu, 'NU', options);

% Handle optional inputs and defaults
switch nargin
    case 2
        % Default values: mu = 0, sigma = 1
        mu = 0;
        sigma = 1;
    case 3
        % Default value: sigma = 1
        sigma = 1;
    case 4
        % All parameters provided
    otherwise
        error('2 to 4 input arguments required.');
end

% Validate scale parameter (sigma)
sigma = parametercheck(sigma, 'SIGMA', options);

% Validate input values (x)
x = datacheck(x, 'X');

% Ensure x is a column vector
x = columncheck(x, 'X');

% Calculate normalization parameter lambda
lambda = sqrt(gamma(3/nu)/gamma(1/nu));

% Standardize the input values
z = (x - mu) / (lambda * sigma);

% Initialize CDF values
p = zeros(size(z));

% Calculate the CDF for each value
% For z >= 0: P = 0.5 + 0.5 * gammainc((abs(z)/lambda)^nu / 2, 1/nu)
% For z < 0:  P = 0.5 - 0.5 * gammainc((abs(z)/lambda)^nu / 2, 1/nu)
% Note: We've already standardized z with lambda, so we simplify to (abs(z))^nu
posZ = z >= 0;
p(posZ) = 0.5 + 0.5 * gammainc((abs(z(posZ))).^nu / 2, 1/nu);
p(~posZ) = 0.5 - 0.5 * gammainc((abs(z(~posZ))).^nu / 2, 1/nu);

% Handle boundary cases
% p should be in [0,1] due to the definition of a CDF
p = max(0, min(1, p));
end