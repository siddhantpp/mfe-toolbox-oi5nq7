function r = skewtrnd(nu, lambda, n, m)
% SKEWTRND Generates random numbers from Hansen's skewed t-distribution.
%
% USAGE:
%   R = skewtrnd(NU, LAMBDA)
%   R = skewtrnd(NU, LAMBDA, N)
%   R = skewtrnd(NU, LAMBDA, N, M)
%
% INPUTS:
%   NU      - Degrees of freedom parameter, must be > 2
%   LAMBDA  - Skewness parameter, must be in range [-1, 1]
%   N       - Number of rows in output (default: 1)
%   M       - Number of columns in output (default: 1)
%
% OUTPUTS:
%   R       - n x m matrix of random numbers from Hansen's skewed t-distribution
%
% COMMENTS:
%   Hansen's skewed t-distribution extends the Student's t-distribution by
%   incorporating a skewness parameter. This function generates random samples
%   from this distribution using the inverse CDF method.
%
%   The implementation follows Hansen (1994) and requires NU > 2 for a
%   well-defined variance.
%
% REFERENCES:
%   Hansen, B. E. (1994). Autoregressive conditional density estimation.
%   International Economic Review, 35(3), 705-730.
%
% See also: skewtcdf, skewtpdf, skewtinv, stdtrnd

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate degrees of freedom parameter
options.lowerBound = 2;  % NU must be > 2
nu = parametercheck(nu, 'nu', options);

% Step 2: Validate skewness parameter
options.lowerBound = -1;
options.upperBound = 1;
lambda = parametercheck(lambda, 'lambda', options);

% Step 3: Handle optional size parameters
if nargin < 3
    n = 1;
end
if nargin < 4
    m = 1;
end

% Step 4: Validate size parameters
n = datacheck(n, 'n');
m = datacheck(m, 'm');
options.isInteger = true;
options.isPositive = true;
n = parametercheck(n, 'n', options);
m = parametercheck(m, 'm', options);

% Step 5: Generate uniform random numbers
u = rand(n, m);

% Step 6: Transform uniform random numbers to skewed t using inverse CDF
r = skewtinv(u, nu, lambda);

end