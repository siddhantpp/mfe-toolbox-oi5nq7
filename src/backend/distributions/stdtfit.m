function parameters = stdtfit(data, options)
% STDTFIT Estimates parameters of the standardized Student's t-distribution using maximum likelihood
%
% USAGE:
%   parameters = stdtfit(data)
%   parameters = stdtfit(data, options)
%
% INPUTS:
%   data        - Vector of data to fit
%   options     - Optional structure with fields:
%                 options.startingVal - Starting value for degrees of freedom (nu), default = 8
%                 options.display     - Display optimization progress ('iter', 'final', 'off'), default = 'off'
%
% OUTPUTS:
%   parameters  - Structure with fields:
%                 parameters.nu       - Degrees of freedom estimate
%                 parameters.nuSE     - Standard error of nu
%                 parameters.mu       - Location parameter (mean)
%                 parameters.muSE     - Standard error of mu
%                 parameters.sigma    - Scale parameter (standard deviation)
%                 parameters.sigmaSE  - Standard error of sigma
%                 parameters.logL     - Log-likelihood
%                 parameters.AIC      - Akaike information criterion
%                 parameters.BIC      - Bayesian information criterion
%                 parameters.convCode - Convergence code (1=converged)
%                 parameters.optOutput - Optimization details
%
% COMMENTS:
%   Estimates the parameters of the standardized Student's t-distribution
%   using maximum likelihood estimation. The standardized t-distribution
%   has heavier tails than the normal distribution and is often more
%   appropriate for modeling financial returns.
%
%   The degrees of freedom (nu) parameter determines the tail thickness,
%   with smaller values corresponding to heavier tails. The function
%   requires nu > 2 to ensure finite variance.
%
% EXAMPLES:
%   % Generate data from a t distribution
%   rng(123);
%   data = trnd(5, 1000, 1);
%   
%   % Fit standardized t distribution
%   parameters = stdtfit(data);
%   
%   % Display results
%   disp(['Estimated degrees of freedom: ' num2str(parameters.nu)]);
%   disp(['Log-likelihood: ' num2str(parameters.logL)]);
%
% REFERENCES:
%   [1] Bollerslev, T. (1987). A conditionally heteroskedastic time series model for 
%       speculative prices and rates of return. The Review of Economics and Statistics.
%
% See also STDTLOGLIK, PARAMETERCHECK, DATACHECK, COLUMNCHECK, AICSBIC

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate input data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Step 2: Process and validate optional parameters
if nargin < 2
    options = [];
end

% Default options
defaultOptions.startingVal = 8;  % Default starting value for nu
defaultOptions.display = 'off';  % Default display setting

% Override defaults with provided options
if ~isempty(options)
    fieldNames = fieldnames(defaultOptions);
    for i = 1:length(fieldNames)
        if ~isfield(options, fieldNames{i})
            options.(fieldNames{i}) = defaultOptions.(fieldNames{i});
        end
    end
else
    options = defaultOptions;
end

% Validate starting value using parametercheck
optionsCheck.isscalar = true;
optionsCheck.isPositive = true;
startingVal = parametercheck(options.startingVal, 'options.startingVal', optionsCheck);

% Ensure starting value is appropriate for standardized t (nu > 2)
if startingVal <= 2
    startingVal = 8;  % Reset to default if below minimum threshold
    warning('Starting value for degrees of freedom must be > 2. Using default value of 8.');
end

% Step 3: Initialize location (mu) and scale (sigma) parameters
mu = mean(data);
sigma = std(data);

% Step 4: Define bounds for degrees of freedom parameter
% nu > 2 ensures existence of variance
lowerBound = 2.0001;  % Slightly above 2 to ensure variance exists
upperBound = 100;     % Upper bound for degrees of freedom

% Step 5: Create wrapper function that calls stdtloglik
% Note: stdtloglik already returns negative log-likelihood for minimization
negLogLikFun = @(nu) stdtloglik(data, nu, mu, sigma);

% Step 6: Configure optimization options
optOptions = optimset('TolFun', 1e-8, 'TolX', 1e-8, 'Display', options.display, ...
                      'MaxIter', 1000, 'MaxFunEvals', 1000);

% Step 7: Use fminbnd to find optimal degrees of freedom that minimizes negative log-likelihood
[nu, fval, exitflag, optOutput] = fminbnd(negLogLikFun, lowerBound, upperBound, optOptions);

% Step 8: Calculate standard errors using numerical approximation of the Hessian
h = 1e-4 * nu;  % Step size for numerical approximation
nuPlus = nu + h;
nuMinus = max(nu - h, lowerBound);  % Ensure nu doesn't go below lower bound

% Compute log-likelihood at optimal point and nearby points for Hessian approximation
negLogLOptimal = fval;  % This is already the negative log-likelihood
negLogLPlus = stdtloglik(data, nuPlus, mu, sigma);
negLogLMinus = stdtloglik(data, nuMinus, mu, sigma);

% Second derivative approximation for numerical Hessian (of negative log-likelihood)
d2L_dnu2 = (negLogLPlus - 2*negLogLOptimal + negLogLMinus) / (h^2);

% Check for numerical issues in Hessian calculation
if d2L_dnu2 <= 1e-6  % Use a small positive threshold instead of exactly 0
    % If Hessian of negative log-likelihood is not sufficiently positive definite
    nuSE = NaN;
    warning('Hessian approximation is not reliably positive definite. Standard error for nu not available.');
else
    % Standard error of nu from information matrix (inverse of Hessian of negative log-likelihood)
    nuSE = sqrt(1/d2L_dnu2);
end

% Standard errors for mu and sigma using asymptotic theory
% For standardized t distribution with nu > 2
T = size(data,1);
muSE = sigma/sqrt(T);  % Approximation assuming finite 4th moment
sigmaSE = sigma/sqrt(2*T);  % Approximation assuming finite 4th moment

% Step 9: Compute final log-likelihood at optimal parameter values
logL = -negLogLOptimal;  % Convert back to positive log-likelihood for reporting

% Step 10: Compute information criteria
k = 3;  % Number of parameters (nu, mu, sigma)
IC = aicsbic(logL, k, T);

% Step 11: Construct return structure
parameters = struct();
parameters.nu = nu;
parameters.nuSE = nuSE;
parameters.mu = mu;
parameters.muSE = muSE;
parameters.sigma = sigma;
parameters.sigmaSE = sigmaSE;
parameters.logL = logL;
parameters.AIC = IC.aic;
parameters.BIC = IC.sbic;
parameters.convCode = exitflag;
parameters.optOutput = optOutput;

% Step 12: Check for convergence issues
if exitflag ~= 1
    warning('Optimization may not have converged. Check parameters.convCode for details.');
end

end