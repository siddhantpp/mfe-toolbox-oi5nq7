function results = skewtfit(data, options)
% SKEWTFIT Estimates parameters of Hansen's skewed t-distribution using maximum likelihood.
%
% USAGE:
%   RESULTS = skewtfit(DATA)
%   RESULTS = skewtfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA    - Vector of data to fit the distribution to
%   OPTIONS - Optional structure with fields:
%             'startingVals' - [nu, lambda, mu, sigma] starting values for parameters
%             'display'      - Level of display: 'off' (default), 'iter', 'final'
%             'maxIter'      - Maximum number of iterations [default: 500]
%             'tolerance'    - Optimization tolerance [default: 1e-6]
%
% OUTPUTS:
%   RESULTS - Structure with fields:
%             'nu'          - Estimated degrees of freedom parameter
%             'lambda'      - Estimated skewness parameter
%             'mu'          - Estimated location parameter
%             'sigma'       - Estimated scale parameter
%             'nuSE'        - Standard error for nu
%             'lambdaSE'    - Standard error for lambda
%             'muSE'        - Standard error for mu
%             'sigmaSE'     - Standard error for sigma
%             'logL'        - Log-likelihood at optimum
%             'aic'         - Akaike Information Criterion
%             'bic'         - Bayesian Information Criterion
%             'parameters'  - Vector of parameters [nu, lambda, mu, sigma]
%             'vcv'         - Variance-covariance matrix of parameters
%             'convergence' - Boolean indicating if the optimization converged
%             'iterations'  - Number of iterations until convergence
%             'exitflag'    - Exit flag from optimization routine
%             'message'     - Exit message from optimization routine
%
% COMMENTS:
%   Estimates parameters of Hansen's skewed t-distribution using maximum likelihood.
%   The skewed t-distribution extends the Student's t with a skewness parameter
%   to model asymmetric financial returns that exhibit heavy tails.
%
%   For reference, see Hansen's 1994 paper:
%   "Autoregressive Conditional Density Estimation" International Economic Review
%
% EXAMPLES:
%   % Generate data from a skewed t-distribution and estimate parameters
%   data = randn(1000, 1);
%   results = skewtfit(data);
%
%   % Provide starting values and control optimization
%   options.startingVals = [5, 0.2, 0, 1];
%   options.display = 'iter';
%   results = skewtfit(data, options);
%
% REFERENCES:
%   Hansen, B. E. (1994). "Autoregressive Conditional Density Estimation".
%   International Economic Review, 35(3), 705-730.
%
% See also SKEWTLOGLIK, SKEWTPDF, PARAMETERCHECK, DATACHECK, COLUMNCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Parse options
if nargin < 2
    options = [];
end

% Set default options
defaultOptions.startingVals = [];
defaultOptions.display = 'off';
defaultOptions.maxIter = 500;
defaultOptions.tolerance = 1e-6;

% Merge provided options with defaults
if isempty(options)
    options = defaultOptions;
else
    % Only add fields that don't exist
    fieldNames = fieldnames(defaultOptions);
    for i = 1:length(fieldNames)
        if ~isfield(options, fieldNames{i})
            options.(fieldNames{i}) = defaultOptions.(fieldNames{i});
        end
    end
end

% Get data dimensions
T = length(data);

% Set up default starting values if none provided
if isempty(options.startingVals)
    % Initial estimates
    mu = mean(data);
    sigma = std(data);
    lambda = 0;  % Start with symmetric case
    nu = 5;      % Moderate degrees of freedom
    startingVals = [nu, lambda, mu, sigma];
else
    startingVals = options.startingVals;
    if length(startingVals) ~= 4
        error('Starting values must contain 4 elements: [nu, lambda, mu, sigma]');
    end
    % Extract parameters
    nu = startingVals(1);
    lambda = startingVals(2);
    mu = startingVals(3);
    sigma = startingVals(4);
end

% Validate starting values
% nu > 2 for valid skewed t
optionsCheck = struct('lowerBound', 2);
nu = parametercheck(nu, 'nu', optionsCheck);

% lambda in [-1, 1]
optionsCheck = struct('lowerBound', -1, 'upperBound', 1);
lambda = parametercheck(lambda, 'lambda', optionsCheck);

% mu is unrestricted
mu = parametercheck(mu, 'mu');

% sigma > 0
optionsCheck = struct('isPositive', true);
sigma = parametercheck(sigma, 'sigma', optionsCheck);

% Update starting values after validation
startingVals = [nu, lambda, mu, sigma];

% Set up parameter constraints for optimization
% A * x ≤ b (linear inequality constraints)
A = [];
b = [];

% Aeq * x = beq (linear equality constraints)
Aeq = [];
beq = [];

% Lower bounds: nu > 2, lambda > -1, mu unrestricted, sigma > 0
lb = [2, -1, -Inf, 1e-6];
% Upper bounds: nu unlimited, lambda < 1, mu unrestricted, sigma unlimited
ub = [Inf, 1, Inf, Inf];

% Nonlinear constraints
nonlcon = [];

% Set up optimization options
optimOptions = optimset('fmincon');
optimOptions = optimset(optimOptions, 'Display', options.display);
optimOptions = optimset(optimOptions, 'MaxIter', options.maxIter);
optimOptions = optimset(optimOptions, 'TolFun', options.tolerance);
optimOptions = optimset(optimOptions, 'TolX', options.tolerance);
optimOptions = optimset(optimOptions, 'MaxFunEvals', options.maxIter*5);
optimOptions = optimset(optimOptions, 'Algorithm', 'interior-point');
optimOptions = optimset(optimOptions, 'Diagnostics', 'off');
optimOptions = optimset(optimOptions, 'LargeScale', 'off');
optimOptions = optimset(optimOptions, 'GradObj', 'off');
optimOptions = optimset(optimOptions, 'Hessian', 'off');

% Define the objective function (use a nested function for better readability)
objectiveFunction = @(p) skewtfit_likelihood(p, data);

% Run the optimization
[parameters, fval, exitflag, output, ~, ~, hessian] = fmincon(objectiveFunction, startingVals, A, b, Aeq, beq, lb, ub, nonlcon, optimOptions);

% Extract optimized parameters
nu = parameters(1);
lambda = parameters(2);
mu = parameters(3);
sigma = parameters(4);

% Compute the log likelihood at the optimum
logLikelihood = -fval;

% Compute AIC and BIC (4 parameters estimated)
aic = -2 * logLikelihood + 2 * 4;
bic = -2 * logLikelihood + log(T) * 4;

% Calculate standard errors from the Hessian
if ~isempty(hessian) && all(isfinite(hessian(:))) && (rcond(hessian) > eps)
    % Try to invert the Hessian to get variance-covariance matrix
    vcv = inv(hessian);
    
    % Extract standard errors (diagonal elements of sqrt(vcv))
    if all(diag(vcv) > 0)
        nuSE = sqrt(vcv(1,1));
        lambdaSE = sqrt(vcv(2,2));
        muSE = sqrt(vcv(3,3));
        sigmaSE = sqrt(vcv(4,4));
    else
        % Non-positive variances indicate issues with estimation
        vcv = NaN(4,4);
        nuSE = NaN;
        lambdaSE = NaN;
        muSE = NaN;
        sigmaSE = NaN;
    end
else
    % If Hessian is empty, contains non-finite values, or is ill-conditioned
    vcv = NaN(4,4);
    nuSE = NaN;
    lambdaSE = NaN;
    muSE = NaN;
    sigmaSE = NaN;
end

% Validate final parameters
optionsCheck = struct('lowerBound', 2);
nu = parametercheck(nu, 'nu', optionsCheck);

optionsCheck = struct('lowerBound', -1, 'upperBound', 1);
lambda = parametercheck(lambda, 'lambda', optionsCheck);

mu = parametercheck(mu, 'mu');

optionsCheck = struct('isPositive', true);
sigma = parametercheck(sigma, 'sigma', optionsCheck);

% Check convergence status
convergence = (exitflag > 0);

% Prepare results structure
results = struct();
results.nu = nu;
results.lambda = lambda;
results.mu = mu;
results.sigma = sigma;
results.nuSE = nuSE;
results.lambdaSE = lambdaSE;
results.muSE = muSE;
results.sigmaSE = sigmaSE;
results.logL = logLikelihood;
results.aic = aic;
results.bic = bic;
results.parameters = parameters;
results.vcv = vcv;
results.iterations = output.iterations;
results.convergence = convergence;
results.exitflag = exitflag;
results.message = output.message;

% If distribution didn't converge, add warning
if ~convergence
    warning('Distribution parameter estimation did not converge. Results may be unreliable.');
end
end

% Helper function to compute negative log-likelihood for optimization
function nlogL = skewtfit_likelihood(parameters, data)
% Extract parameters
nu = parameters(1);
lambda = parameters(2);
mu = parameters(3);
sigma = parameters(4);

% Compute negative log-likelihood
[nlogL, ~] = skewtloglik(data, [nu, lambda, mu, sigma]);

% Ensure it's a valid number for optimization
if ~isfinite(nlogL) || ~isreal(nlogL)
    nlogL = 1e10;  % Large penalty for invalid parameters
end
end