function p = skewtcdf(x, nu, lambda)
% SKEWTCDF Computes the CDF of Hansen's skewed t-distribution.
%
% USAGE:
%   P = skewtcdf(X, NU, LAMBDA)
%
% INPUTS:
%   X       - Points at which to evaluate the CDF
%   NU      - Degrees of freedom parameter, must be > 2
%   LAMBDA  - Skewness parameter, must be in range [-1, 1]
%
% OUTPUTS:
%   P       - CDF values evaluated at points in X
%
% COMMENTS:
%   Hansen's skewed t-distribution extends the Student's t-distribution by
%   incorporating a skewness parameter. This distribution is useful for
%   modeling asymmetric financial returns and risk measures.
%
%   The implementation follows Hansen (1994) and requires NU > 2 for a
%   well-defined variance.
%
% REFERENCES:
%   Hansen, B. E. (1994). Autoregressive conditional density estimation.
%   International Economic Review, 35(3), 705-730.
%
% See also: stdtcdf, skewtpdf, skewtinv, skewtrnd

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Handle special case of empty input
if isempty(x)
    p = x;
    return;
end

% Validate parameters
options.lowerBound = 2;  % NU must be > 2
nu = parametercheck(nu, 'nu', options);

options.lowerBound = -1;
options.upperBound = 1;
lambda = parametercheck(lambda, 'lambda', options);

% Ensure x is in column format
x = columncheck(x, 'x');

% Validate input data
x = datacheck(x, 'x');

% Initialize output array
p = zeros(size(x));

% Process based on scalar/vector inputs
if isscalar(nu) && isscalar(lambda)
    % Most common case: scalar parameters, calculate constants once
    c = gamma((nu+1)/2) / (gamma(nu/2) * sqrt(pi*(nu-2)));
    a = 4 * lambda * c * (nu-2) / nu;
    b = sqrt(1 + 3*lambda^2 - a^2);
    
    % Threshold
    threshold = -a/b;
    
    % Below threshold
    below_idx = (x < threshold);
    if any(below_idx)
        denominator = sqrt(1-lambda^2);
        transformed_x = (-a-b*x(below_idx))/denominator;
        p(below_idx) = (1-lambda)^2 .* stdtcdf(transformed_x, nu);
    end
    
    % At or above threshold
    above_idx = ~below_idx;
    if any(above_idx)
        denominator = sqrt(1+lambda^2);
        transformed_x = (a+b*x(above_idx))/denominator;
        
        term1 = 0.5 + (1-lambda)^2/2;
        term2 = (1+lambda)^2/2 .* (stdtcdf(transformed_x, nu) - 0.5);
        p(above_idx) = term1 + term2;
    end
else
    % Handle case where some parameters are non-scalar
    % We'll iterate through each element
    
    % If x is scalar but parameters are not, expand x
    if isscalar(x) && (~isscalar(nu) || ~isscalar(lambda))
        if ~isscalar(nu)
            x = x * ones(size(nu));
        else
            x = x * ones(size(lambda));
        end
    end
    
    % Initialize output with the same size as x
    p = zeros(size(x));
    
    % Iterate through each element
    for i = 1:numel(x)
        % Get the current parameter values
        if isscalar(nu)
            nu_i = nu;
        else
            nu_i = nu(i);
        end
        
        if isscalar(lambda)
            lambda_i = lambda;
        else
            lambda_i = lambda(i);
        end
        
        % Calculate constants
        c = gamma((nu_i+1)/2) / (gamma(nu_i/2) * sqrt(pi*(nu_i-2)));
        a = 4 * lambda_i * c * (nu_i-2) / nu_i;
        b = sqrt(1 + 3*lambda_i^2 - a^2);
        
        % Threshold
        threshold = -a/b;
        
        % Calculate CDF based on region
        if x(i) < threshold
            denominator = sqrt(1-lambda_i^2);
            transformed_x = (-a-b*x(i))/denominator;
            p(i) = (1-lambda_i)^2 * stdtcdf(transformed_x, nu_i);
        else
            denominator = sqrt(1+lambda_i^2);
            transformed_x = (a+b*x(i))/denominator;
            
            term1 = 0.5 + (1-lambda_i)^2/2;
            term2 = (1+lambda_i)^2/2 * (stdtcdf(transformed_x, nu_i) - 0.5);
            p(i) = term1 + term2;
        end
    end
end

% Handle any numerical issues (ensure p is in [0,1])
p = max(0, min(1, p));

end