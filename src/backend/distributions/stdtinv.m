function x = stdtinv(p, nu)
% STDTINV Computes the inverse CDF (quantile function) of the standardized Student's t-distribution.
%
% USAGE:
%   X = stdtinv(P, NU)
%
% INPUTS:
%   P     - Vector or matrix of probabilities at which to evaluate the inverse CDF, 0 <= p <= 1
%   NU    - Degrees of freedom, scalar value > 2
%
% OUTPUTS:
%   X     - Quantile values corresponding to probabilities in P
%
% COMMENTS:
%   The standardized Student's t-distribution has mean 0 and variance 1,
%   which requires NU > 2. For NU <= 2, variance is undefined or infinite.
%
%   For a probability P, this function computes the value X such that P(T <= X) = P
%   where T follows a standardized Student's t-distribution with NU degrees of freedom.
%
%   This implementation exploits the symmetry of the t-distribution and uses
%   the relationship between the t-distribution and the incomplete beta function
%   for efficient and numerically stable computation.
%
% EXAMPLES:
%   x = stdtinv(0.975, 5)  % 95% confidence interval upper bound for standardized t(5)
%   x = stdtinv([0.025 0.5 0.975], 10)  % Multiple quantiles
%
% See also: stdtcdf, stdtpdf, stdtrnd, betaincinv

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate degrees of freedom parameter
options.isscalar = true;
options.lowerBound = 2;  % Require nu > 2 for standardized t
nu = parametercheck(nu, 'nu', options);

% Step 2: Validate input probabilities
options = [];
options.lowerBound = 0;
options.upperBound = 1;
p = parametercheck(p, 'p', options);
p = datacheck(p, 'p');

% Step 3: Ensure p is column vector format if it's a vector
p = columncheck(p, 'p');

% Step 4: Initialize output array with same size as p
x = zeros(size(p));

% Step 5: Handle special cases

% For p = 0, quantile is -Inf
x(p == 0) = -Inf;

% For p = 1, quantile is Inf
x(p == 1) = Inf;

% For p = 0.5, quantile is 0 (by symmetry of the standardized t-distribution)
x(p == 0.5) = 0;

% Step 6: Compute quantiles for non-special cases
% Identify regular cases (neither 0, 0.5, nor 1)
regular_cases = (p > 0 & p < 1 & p ~= 0.5);

if any(regular_cases)
    p_reg = p(regular_cases);
    
    % For all regular cases, use the formula:
    % t = sgn(p - 1/2) * sqrt(nu * (1/B - 1))
    % where B = betaincinv(2*min(p,1-p), 1/2, nu/2)
    
    % Compute sign factor based on whether p > 0.5 or p < 0.5
    sign_factor = sign(p_reg - 0.5);
    
    % Compute the minimum of p and 1-p for each value
    p_min = min(p_reg, 1 - p_reg);
    
    % Compute the inverse incomplete beta function
    beta_arg = 2 * p_min;
    beta_inv = betaincinv(beta_arg, 0.5, nu/2);
    
    % Compute t-distribution quantile
    % For a regular t-distribution
    t_quantile = sign_factor .* sqrt(nu * (1./beta_inv - 1));
    
    % Convert to standardized t-distribution
    % If X follows regular t(nu), then X * sqrt((nu-2)/nu) follows standardized t
    x_reg = t_quantile .* sqrt((nu-2)/nu);
    
    % Assign to output
    x(regular_cases) = x_reg;
end

% Alternative implementation using numerical inversion with fzero
% This can be enabled as a backup or for verification/testing
% for idx = find(regular_cases)'
%     target_p = p(idx);
%     % Initial guess
%     x0 = sign(target_p - 0.5) * 1;
%     % Use anonymous function for stdtcdf with fixed nu
%     tcdf_fun = @(x) stdtcdf(x, nu) - target_p;
%     % Find the root using fzero
%     x(idx) = fzero(tcdf_fun, x0);
% end

end