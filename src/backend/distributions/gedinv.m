function x = gedinv(p, nu, mu, sigma)
% GEDINV Computes the inverse cumulative distribution function (quantile function) of the Generalized Error Distribution.
%
% USAGE:
%   X = gedinv(P, NU)
%   X = gedinv(P, NU, MU, SIGMA)
%
% INPUTS:
%   P     - Probability values at which to evaluate the inverse CDF (0 ≤ P ≤ 1).
%           Either a column vector or a row vector, which will be converted to 
%           a column vector.
%   NU    - Shape parameter controlling tail thickness (NU > 0)
%           NU < 2 implies tails thicker than normal
%           NU = 2 implies normal distribution
%           NU > 2 implies thinner tails than normal
%   MU    - Location parameter (Optional, Default = 0)
%   SIGMA - Scale parameter (Optional, Default = 1, SIGMA > 0)
%
% OUTPUTS:
%   X     - Quantile values corresponding to each probability in P
%
% COMMENTS:
%   The inverse CDF computes quantiles (values x such that P(X ≤ x) = p) for
%   the Generalized Error Distribution. For extreme probability values near 0 or 1,
%   numerical techniques are employed to ensure accuracy.
%
%   The function utilizes numerical inversion to find points where the CDF equals
%   the input probability values. For p=0.5, it exploits the symmetry of the GED
%   distribution to return mu directly.
%
% EXAMPLES:
%   x = gedinv(0.95, 1.5)
%   x = gedinv([0.025 0.5 0.975]', 1.5, 0, 2)
%
% See also GEDPDF, GEDCDF, GEDRND, GEDFIT

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

% Validate probability values (p)
p = datacheck(p, 'P');

% Add additional probability validation (0 ≤ p ≤ 1)
if any(p < 0 | p > 1)
    error('All values in P must be between 0 and 1 inclusive.');
end

% Ensure p is a column vector
p = columncheck(p, 'P');

% Calculate normalization parameter lambda
lambda = sqrt(gamma(3/nu)/gamma(1/nu));

% Initialize quantile values with same size as p
x = zeros(size(p));

% Handle boundary cases
% For p=0, the quantile is -Infinity
% For p=1, the quantile is +Infinity
x(p == 0) = -Inf;
x(p == 1) = Inf;

% For p=0.5, exploit symmetry of the GED distribution
% The median is at 0 for the standardized GED
x(p == 0.5) = 0;

% For remaining probabilities, implement numerical inversion
remainingIdx = (p > 0 & p < 1 & p ~= 0.5);
if any(remainingIdx)
    pVals = p(remainingIdx);
    xVals = zeros(size(pVals));
    
    % Process each probability value individually using numerical inversion
    for i = 1:length(pVals)
        pVal = pVals(i);
        
        % Define the objective function: find x where gedcdf(x, nu) = pVal
        objectiveFunc = @(x) gedcdf(x, nu) - pVal;
        
        % For numerical stability, use different approaches based on the probability region
        if pVal < 0.5
            % For left tail probabilities
            % Try to establish a bracket where the objective function changes sign
            
            % Start with a reasonable negative value
            leftPoint = -20;
            % And a point that should be above the quantile (p=0.5 corresponds to x=0)
            rightPoint = 0;
            
            % Check if our bracket actually contains the root
            fLeft = objectiveFunc(leftPoint);
            fRight = objectiveFunc(rightPoint);
            
            % If the bracket doesn't contain the root, expand the left boundary
            while sign(fLeft) == sign(fRight) && leftPoint > -1000
                leftPoint = leftPoint * 2;
                fLeft = objectiveFunc(leftPoint);
            end
            
            % If a valid bracket is found, use it
            if sign(fLeft) ~= sign(fRight)
                xVal = fzero(objectiveFunc, [leftPoint, rightPoint]);
            else
                % Otherwise, just try with the leftPoint as initial guess
                xVal = fzero(objectiveFunc, leftPoint);
            end
        else
            % For right tail probabilities (p > 0.5)
            % Similar approach but with positive values
            
            % Start from median (x=0) and a reasonable positive value
            leftPoint = 0;
            rightPoint = 20;
            
            % Check if our bracket contains the root
            fLeft = objectiveFunc(leftPoint);
            fRight = objectiveFunc(rightPoint);
            
            % If the bracket doesn't contain the root, expand the right boundary
            while sign(fLeft) == sign(fRight) && rightPoint < 1000
                rightPoint = rightPoint * 2;
                fRight = objectiveFunc(rightPoint);
            end
            
            % If a valid bracket is found, use it
            if sign(fLeft) ~= sign(fRight)
                xVal = fzero(objectiveFunc, [leftPoint, rightPoint]);
            else
                % Otherwise, just try with the rightPoint as initial guess
                xVal = fzero(objectiveFunc, rightPoint);
            end
        end
        
        % Store the result
        xVals(i) = xVal;
    end
    
    % Store the computed values
    x(remainingIdx) = xVals;
end

% Apply location-scale transformation to all quantiles
% This adjusts the standardized quantiles for mu and sigma
x = mu + sigma * lambda * x;

end