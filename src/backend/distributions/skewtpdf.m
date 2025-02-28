function pdf = skewtpdf(x, nu, lambda)
% SKEWTPDF Computes the probability density function of Hansen's skewed t-distribution
%
% USAGE:
%   PDF = skewtpdf(X, NU, LAMBDA)
%
% INPUTS:
%   X      - Values at which to evaluate the PDF
%   NU     - Degrees of freedom parameter, NU > 2 
%   LAMBDA - Skewness parameter, -1 <= LAMBDA <= 1
%
% OUTPUTS:
%   PDF    - Probability density function values
%
% COMMENTS:
%   Hansen's skewed t-distribution extends the Student's t-distribution with
%   a skewness parameter, making it particularly useful for modeling financial
%   returns that exhibit asymmetry.
%
%   The PDF is computed using two different formulations for the left and right
%   tails of the distribution, separated by a threshold value that depends on
%   the skewness parameter.
%
%   For reference, see Hansen's 1994 paper:
%   "Autoregressive Conditional Density Estimation" International Economic Review
%
% REFERENCES:
%   Hansen, B. E. (1994). "Autoregressive Conditional Density Estimation".
%   International Economic Review, 35(3), 705-730.
%
% See also STDTPDF, STDTRND, SKEWTRND

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Handle the case of empty input
if isempty(x)
    pdf = x;
    return;
end

% Validate degrees of freedom parameter (nu > 2)
options.lowerBound = 2;
nu = parametercheck(nu, 'nu', options);

% Validate skewness parameter (lambda in [-1,1])
options.lowerBound = -1;
options.upperBound = 1;
lambda = parametercheck(lambda, 'lambda', options);

% Ensure x is in column format
x = columncheck(x, 'x');

% Check data dimensions for compatibility
x = datacheck(x, 'x');

% Get sizes of inputs
[T, k] = size(x);
[nuRows, nuCols] = size(nu);
[lambdaRows, lambdaCols] = size(lambda);

% If nu is scalar, broadcast to match x size
if nuRows == 1 && nuCols == 1
    nu = nu * ones(T, k);
end

% If lambda is scalar, broadcast to match x size
if lambdaRows == 1 && lambdaCols == 1
    lambda = lambda * ones(T, k);
end

% Check that dimensions match after broadcasting
if ~isequal(size(x), size(nu)) || ~isequal(size(x), size(lambda))
    error('Inputs x, nu, and lambda must have compatible dimensions');
end

% Calculate the constant c
c = gamma((nu + 1) / 2) ./ (sqrt(pi * (nu - 2)) .* gamma(nu / 2));

% Calculate parameter a
a = 4 * lambda .* c .* ((nu - 2) ./ nu);

% Calculate parameter b
b = sqrt(1 + 3 * lambda.^2 - a.^2);

% Calculate threshold parameter (-a/b)
threshold = -a ./ b;

% Initialize PDF array with same size as x
pdf = zeros(size(x));

% Create mask for left tail (x < threshold)
leftMask = (x < threshold);

% Calculate PDF for left tail
if any(leftMask(:))
    z = (b(leftMask) .* x(leftMask) + a(leftMask)) ./ (1 - lambda(leftMask));
    pdf(leftMask) = b(leftMask) .* c(leftMask) .* ...
                    (1 + 1./(nu(leftMask) - 2) .* z.^2).^(-(nu(leftMask) + 1) / 2);
end

% Calculate PDF for right tail
rightMask = ~leftMask;
if any(rightMask(:))
    z = (b(rightMask) .* x(rightMask) + a(rightMask)) ./ (1 + lambda(rightMask));
    pdf(rightMask) = b(rightMask) .* c(rightMask) .* ...
                     (1 + 1./(nu(rightMask) - 2) .* z.^2).^(-(nu(rightMask) + 1) / 2);
end

% Handle any non-finite results due to numerical issues
pdf(~isfinite(pdf)) = 0;

% Ensure non-negative values (PDF must be non-negative)
pdf = max(pdf, 0);

end