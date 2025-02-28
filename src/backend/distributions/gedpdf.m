function pdf = gedpdf(x, nu)
% GEDPDF Computes the probability density function of the Generalized Error Distribution
%
% USAGE:
%   PDF = gedpdf(X, NU)
%
% INPUTS:
%   X     - Points at which to compute the PDF, scalar or column vector
%   NU    - Shape parameter of the GED distribution, NU > 0
%           Controls the tail thickness of the distribution:
%           - NU < 2: Fat tails (leptokurtic)
%           - NU = 2: Normal distribution
%           - NU > 2: Thin tails (platykurtic)
%
% OUTPUTS:
%   PDF   - Probability density function values, vector with same dimensions as X
%
% EXAMPLES:
%   % Compute the PDF of standard GED with shape=1.5 at points -3:0.1:3
%   x = (-3:0.1:3)';
%   pdf = gedpdf(x, 1.5);
%   plot(x, pdf);
%   title('GED PDF with \nu = 1.5');
%
%   % Compare different shape parameters
%   x = (-5:0.1:5)';
%   pdf1 = gedpdf(x, 1);   % Laplace distribution
%   pdf2 = gedpdf(x, 2);   % Normal distribution
%   pdf3 = gedpdf(x, 5);   % Thin-tailed distribution
%   plot(x, [pdf1, pdf2, pdf3]);
%   legend('\nu = 1', '\nu = 2', '\nu = 5');
%   title('GED PDF with Different Shape Parameters');
%
% COMMENTS:
%   The Generalized Error Distribution (GED) includes the following special cases:
%   - NU = 1: Laplace (double exponential) distribution
%   - NU = 2: Normal (Gaussian) distribution
%   - NU < 2: Leptokurtic (fat-tailed) distribution
%   - NU > 2: Platykurtic (thin-tailed) distribution
%
%   The PDF of the standardized GED with parameter NU is given by:
%   f(x; nu) = [nu / (2 * lambda * Γ(1/nu))] * exp[-(1/2) * |x/lambda|^nu]
%   where lambda = sqrt[Γ(3/nu) / Γ(1/nu)] and Γ is the gamma function.
%
%   The distribution has zero mean and unit variance.
%
% REFERENCES:
%   [1] Box, G. E. P., Tiao, G. C. (1973) "Bayesian Inference in Statistical Analysis", 
%       Addison-Wesley, Reading, MA.
%   [2] Nelson, D. B. (1991) "Conditional Heteroskedasticity in Asset Returns: 
%       A New Approach", Econometrica, 59(2), 347-370.
%
% See also gedcdf, gedrnd, gedfit, gedlike, stdtpdf, normpdf, tpdf

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate shape parameter (nu)
options.isscalar = true;    % Ensure nu is a scalar
options.isPositive = true;  % Ensure nu > 0
nu = parametercheck(nu, 'nu', options);

% Step 2: Validate input data (x)
x = datacheck(x, 'x');      % Ensure x is numeric and free of NaN/Inf
x = columncheck(x, 'x');    % Ensure x is a column vector

% Step 3: Calculate normalization constants
% Scale parameter (lambda) to ensure unit variance
lambda = sqrt(gamma(3/nu) / gamma(1/nu));
% Normalization constant to ensure the PDF integrates to 1
c = nu / (2 * lambda * gamma(1/nu));

% Step 4: Compute the PDF
% The main formula for GED PDF
pdf = c * exp(-0.5 * abs(x/lambda).^nu);

end