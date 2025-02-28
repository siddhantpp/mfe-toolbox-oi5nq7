function p = stdtcdf(x, nu)
% STDTCDF Computes the CDF of the standardized Student's t-distribution.
%
% USAGE:
%   P = stdtcdf(X, NU)
%
% INPUTS:
%   X     - Vector or matrix of points at which to evaluate the CDF
%   NU    - Degrees of freedom, scalar value > 2
%
% OUTPUTS:
%   P     - CDF values evaluated at points in X
%
% COMMENTS:
%   The standardized Student's t-distribution has mean 0 and variance 1,
%   which requires NU > 2. For NU <= 2, variance is undefined or infinite.
%
%   This implementation is optimized for numerical stability and uses the
%   relationship between the t-distribution CDF and the incomplete beta function.
%   It is particularly useful for financial applications including tail risk 
%   modeling and quantile-based risk measures.
%
% EXAMPLES:
%   p = stdtcdf(1.96, 5)  % Probability that a standardized t(5) is less than 1.96
%   p = stdtcdf([-1 0 1], 10)  % CDF at multiple points
%
% See also: stdtpdf, stdtrnd, stdtinv, betainc

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate degrees of freedom parameter
options.isscalar = true;
options.lowerBound = 2;  % Require nu > 2 for standardized t
nu = parametercheck(nu, 'nu', options);

% Step 2: Validate input data
x = datacheck(x, 'x');

% Step 3: Ensure x is column vector format if it's a vector
x = columncheck(x, 'x');

% Step 4: Initialize output array
p = zeros(size(x));

% Step 5: Handle special cases
% For x = 0, CDF = 0.5 (by symmetry)
p(x == 0) = 0.5;

% For extreme large positive values, CDF ≈ 1
extreme_pos = (x > 1e10);
p(extreme_pos) = 1;

% For extreme large negative values, CDF ≈ 0
extreme_neg = (x < -1e10);
p(extreme_neg) = 0;

% Step 6: Calculate CDF for remaining values
% Identify non-extreme, non-zero values
regular_vals = ~(extreme_pos | extreme_neg | (x == 0));

if any(regular_vals)
    % For the standardized t-distribution (mean=0, var=1), we need to adjust input values
    % to equivalent values in a regular t-distribution (mean=0, var=nu/(nu-2))
    % If Y follows a standardized t-distribution, then X = Y / sqrt((nu-2)/nu) follows 
    % a regular t-distribution with nu degrees of freedom.
    % To compute P(Y <= y), we need to compute P(X <= y/sqrt((nu-2)/nu))
    x_reg = x(regular_vals) / sqrt((nu-2)/nu);
    
    % Calculate the CDF using the relationship with the incomplete beta function
    % Separate negative and positive values for better numerical stability
    neg_idx = (x_reg < 0);
    pos_idx = ~neg_idx;
    
    % For negative x
    if any(neg_idx)
        x_neg = x_reg(neg_idx);
        p_neg = 0.5 * betainc(nu./(nu + x_neg.^2), nu/2, 0.5);
        p(regular_vals(neg_idx)) = p_neg;
    end
    
    % For positive x
    if any(pos_idx)
        x_pos = x_reg(pos_idx);
        p_pos = 1 - 0.5 * betainc(nu./(nu + x_pos.^2), nu/2, 0.5);
        p(regular_vals(pos_idx)) = p_pos;
    end
end
end