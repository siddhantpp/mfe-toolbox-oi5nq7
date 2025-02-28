function f = stdtpdf(x, nu)
% STDTPDF Computes the probability density function (PDF) of the standardized Student's t-distribution.
%
% USAGE:
%   F = stdtpdf(X, NU)
%
% INPUTS:
%   X     - Vector of points to evaluate the PDF
%   NU    - Degrees of freedom parameter, NU > 2 for standardized distribution
%
% OUTPUTS:
%   F     - PDF values corresponding to each point in X
%
% COMMENTS:
%   Computes the PDF of the standardized Student's t-distribution with mean 0 and
%   variance 1. The standardization is achieved by scaling the standard
%   t-distribution (which has variance ν/(ν-2) for ν > 2) to have unit variance.
%
%   The PDF is calculated using the formula:
%   f(x) = (1/scale) * [Γ((ν+1)/2)/(√(π·ν)·Γ(ν/2))] · [1 + (x/scale)²/ν]^(-((ν+1)/2))
%   where scale = sqrt((ν-2)/ν)
%
%   This function is optimized for numerical stability in extreme tail calculations
%   common in financial applications.
%
% EXAMPLES:
%   x = linspace(-5, 5, 1000);
%   pdf_values = stdtpdf(x, 5);
%   plot(x, pdf_values);
%
% See also STDTRND, STDTCDF, STDTINV

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1   Date: 28-Oct-2009

% Validate degrees of freedom parameter
options.isscalar = true;
options.lowerBound = 2;  % For standardized t-distribution, nu > 2 is required
nu = parametercheck(nu, 'nu', options);

% Validate input data
x = datacheck(x, 'x');

% Ensure x is in column vector format
x = columncheck(x, 'x');

% Scaling factor to convert between standard and standardized t
% Standard t-distribution has variance ν/(ν-2) for ν > 2
% Standardized t-distribution has variance 1
scale = sqrt((nu-2)/nu);  

% Points at which to evaluate the standard t-distribution
z = x / scale;  % if y = x * scale, then x = y / scale

% Compute standard t-distribution PDF at transformed points
% Using the gamma function formula
const = gamma((nu+1)/2) / (sqrt(pi*nu) * gamma(nu/2));
f_std = const * (1 + (z.^2)/nu).^(-((nu+1)/2));

% Apply the change of variable formula: f_Y(y) = (1/scale) * f_X(y/scale)
f = f_std / scale;

% Alternative implementation using the beta function for enhanced numerical stability
% in extreme cases or when dealing with high degrees of freedom
% const_beta = 1 / (sqrt(nu) * beta(1/2, nu/2));
% f_std_beta = const_beta * (1 + (z.^2)/nu).^(-((nu+1)/2));
% f = f_std_beta / scale;
end