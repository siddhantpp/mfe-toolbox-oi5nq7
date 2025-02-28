function parameters = nagarchfit(data, options)
% NAGARCHFIT Estimates parameters of a NAGARCH (Nonlinear Asymmetric GARCH) model
%
% USAGE:
%   [PARAMETERS] = nagarchfit(DATA)
%   [PARAMETERS] = nagarchfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA       - Vector of mean zero residuals
%   OPTIONS    - [OPTIONAL] Options structure with fields:
%                'p' - Positive integer, the order of the symmetric GARCH component (Default: 1)
%                'q' - Positive integer, the order of the ARCH component (Default: 1)
%                'error_type' - String, specifying the error distribution:
%                    'NORMAL' - Gaussian innovations (Default)
%                    'T' - Student's t innovations
%                    'GED' - Generalized Error Distribution innovations
%                    'SKEWT' - Hansen's skewed t innovations
%                'compute_std_errors' - Boolean, indicating whether to compute
%                    standard errors (Default: true)
%                'starting_values' - Vector of initial parameter values
%                    [omega, alpha(1),...,alpha(q), gamma, beta(1),...,beta(p), [dist_params]]
%                'max_iter' - Maximum number of iterations in optimization (Default: 1000)
%                'display' - String, display option during optimization: 'off', 'iter', or 'final'
%                    (Default: 'off')
%
% OUTPUTS:
%   PARAMETERS - Structure containing parameter estimates and related information:
%                .parameters - Vector of parameters
%                    [omega, alpha(1),...,alpha(q), gamma, beta(1),...,beta(p), [dist_params]]
%                .LL - Log-likelihood at optimum
%                .LLs - Vector of log-likelihoods
%                .std_errors - Vector of standard errors
%                .tstat - Vector of t-statistics
%                .e - The residuals (innovations)
%                .stdresid - Standardized residuals (e/sqrt(ht))
%                .ht - Conditional variances
%                .model_type - String indicating model type ('NAGARCH')
%                .error_type - String indicating error distribution
%                .convergence - Integer, convergence indicator
%                .nu - Degrees of freedom (when applicable)
%                .lambda - Skewness parameter (when applicable)
%
% COMMENTS:
%   The NAGARCH (Nonlinear Asymmetric GARCH) model extends standard GARCH by incorporating
%   nonlinear asymmetry in the conditional variance equation to better capture leverage effects
%   in financial time series. The variance equation has the form:
%
%   h_t = omega + sum(alpha(i) * h_{t-i} * (e_{t-i}/sqrt(h_{t-i}) - gamma)^2) + sum(beta(j) * h_{t-j})
%
%   This function performs maximum likelihood estimation to find optimal parameters and
%   provides comprehensive diagnostics including conditional variances, standardized residuals,
%   and statistical tests.
%
% EXAMPLES:
%   % Fit a NAGARCH(1,1) model with normal errors
%   parameters = nagarchfit(data);
%
%   % Fit a NAGARCH(1,1) model with t-distributed errors
%   options = struct('error_type', 'T');
%   parameters = nagarchfit(data, options);
%
%   % Fit a NAGARCH(2,1) model with GED errors and display iteration info
%   options = struct('p', 2, 'error_type', 'GED', 'display', 'iter');
%   parameters = nagarchfit(data, options);
%
% See also GARCHLIKELIHOOD, GARCHINIT, BACKCAST, GARCHCORE, PARAMETERCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Default options
DEFAULT_ERROR_TYPE = 'NORMAL';
DEFAULT_STATIONARITY_BOUND = 0.998;

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Process options
if nargin < 2
    options = [];
end

% Set default options if not provided
if isempty(options)
    options = struct();
end

% Process error type
if ~isfield(options, 'error_type')
    options.error_type = DEFAULT_ERROR_TYPE;
else
    options.error_type = upper(options.error_type);
end

% Process ARCH/GARCH orders
if ~isfield(options, 'p')
    options.p = 1;
else
    options.p = parametercheck(options.p, 'options.p', struct('isscalar', true, 'isInteger', true, 'isPositive', true));
end

if ~isfield(options, 'q')
    options.q = 1;
else
    options.q = parametercheck(options.q, 'options.q', struct('isscalar', true, 'isInteger', true, 'isPositive', true));
end

% Extract key values
p = options.p;
q = options.q;
T = length(data);

% Set up model type for NAGARCH and map error_type to distribution for other functions
options.model = 'NAGARCH';
options.distribution = options.error_type;

% Initialize parameters using garchinit
if ~isfield(options, 'starting_values')
    initOptions = options;
    startingParams = garchinit(data, initOptions);
else
    startingParams = options.starting_values;
    parametercheck(startingParams, 'options.starting_values');
end

% Set up parameter constraints
% For NAGARCH, the constraints are:
% 1. omega > 0
% 2. alpha(i) >= 0 for all i
% 3. beta(j) >= 0 for all j
% 4. sum(alpha(i)*(1+gamma^2)) + sum(beta(j)) < stationarity_bound

% Set lower and upper bounds
% Lower: omega > 0, alpha(i) >= 0, gamma unconstrained, beta(j) >= 0, [dist params bounds]
LB = zeros(length(startingParams), 1);
LB(1 + q + 1) = -Inf;  % No constraint on gamma (can be negative)

% Upper: Unrestricted except for distribution parameters
UB = Inf(length(startingParams), 1);

% Add constraints for distribution parameters
switch options.error_type
    case 'T'
        % nu > 2 for t-distribution
        LB(end) = 2.01;
    case 'GED'
        % nu > 0 for GED
        LB(end) = 0.01;
    case 'SKEWT'
        % nu > 2, -1 < lambda < 1 for skewed t
        LB(end-1) = 2.01;
        LB(end) = -0.99;
        UB(end) = 0.99;
end

% Configure optimization options
optimOptions = optimset('fmincon');
optimOptions = optimset(optimOptions, 'TolFun', 1e-8, 'TolX', 1e-8);
if isfield(options, 'display')
    optimOptions = optimset(optimOptions, 'Display', options.display);
else
    optimOptions = optimset(optimOptions, 'Display', 'off');
end
if isfield(options, 'max_iter')
    optimOptions = optimset(optimOptions, 'MaxIter', options.max_iter);
    optimOptions = optimset(optimOptions, 'MaxFunEvals', options.max_iter*4);
else
    optimOptions = optimset(optimOptions, 'MaxIter', 1000);
    optimOptions = optimset(optimOptions, 'MaxFunEvals', 4000);
end

% Create wrapper function for garchlikelihood
% Add stationarity constraint to options
options.constrainStationarity = true;
nagarchFitFcn = @(params) nagarch_likelihood_wrapper(params, data, options);

% Run optimization
[parameters, fval, exitflag, output, ~, grad, hessian] = ...
    fmincon(nagarchFitFcn, startingParams, [], [], [], [], LB, UB, [], optimOptions);

% Calculate final log-likelihood
loglikelihood = -fval;

% Calculate model diagnostics
diagnostics = compute_nagarch_diagnostics(parameters, data, options);
ht = diagnostics.ht;
e = data;
stdresid = e ./ sqrt(ht);

% Compute standard errors if requested
if ~isfield(options, 'compute_std_errors') || options.compute_std_errors
    % Compute numerical Hessian if not provided or not valid
    if isempty(hessian) || any(any(~isfinite(hessian)))
        % Compute numerical Hessian
        H = compute_numerical_hessian(parameters, data, options);
    else
        H = hessian;
    end
    
    % Check if Hessian is valid
    if all(all(isfinite(H))) && rcond(H) > 1e-12
        % Compute variance-covariance matrix
        vcv = inv(H);
        
        % Extract standard errors
        std_errors = sqrt(diag(vcv));
        
        % Compute t-statistics
        tstat = parameters ./ std_errors;
    else
        % Hessian is not valid, use NaN for standard errors
        std_errors = nan(size(parameters));
        tstat = nan(size(parameters));
    end
else
    % No standard errors requested
    std_errors = nan(size(parameters));
    tstat = nan(size(parameters));
end

% Re-estimate distribution parameters if applicable
nu = [];
lambda = [];
switch options.error_type
    case 'T'
        tFit = stdtfit(stdresid);
        nu = tFit.nu;
    case 'GED'
        gedFit = gedfit(stdresid);
        nu = gedFit.nu;
    case 'SKEWT'
        skewtFit = skewtfit(stdresid);
        nu = skewtFit.nu;
        lambda = skewtFit.lambda;
end

% Prepare output structure
output_params = struct();
output_params.parameters = parameters;
output_params.LL = loglikelihood;
output_params.LLs = diagnostics.LLs;
output_params.std_errors = std_errors;
output_params.tstat = tstat;
output_params.e = e;
output_params.stdresid = stdresid;
output_params.ht = ht;
output_params.model_type = 'NAGARCH';
output_params.error_type = options.error_type;
output_params.convergence = exitflag;

% Add distribution parameters if applicable
switch options.error_type
    case 'T'
        output_params.nu = nu;
    case 'GED'
        output_params.nu = nu;
    case 'SKEWT'
        output_params.nu = nu;
        output_params.lambda = lambda;
end

end

function [negativeLogLik] = nagarch_likelihood_wrapper(parameters, data, options)
% Helper function that wraps the garchlikelihood function specifically for NAGARCH optimization
%
% Ensures the model type is set correctly and passes parameters to the main likelihood function

% Ensure model type is set correctly
options.model = 'NAGARCH';

% Call the standard garchlikelihood function
negativeLogLik = garchlikelihood(parameters, data, options);
end

function diagnostics = compute_nagarch_diagnostics(parameters, data, options)
% Helper function that computes conditional variances and residuals for NAGARCH model
%
% Generates conditional variances, standardized residuals, and log-likelihood series

% Extract model orders
p = options.p;
q = options.q;
T = length(data);

% Generate conditional variances using garchcore
ht = garchcore(parameters, data, options);

% Calculate standardized residuals
e = data;
stdresid = e ./ sqrt(ht);

% Compute log-likelihood series for individual observations
LLs = zeros(T, 1);
for t = 1:T
    % Standardized innovation
    z = e(t) / sqrt(ht(t));
    
    % Base log-likelihood component (log Jacobian term from variance transformation)
    LLs(t) = -0.5 * log(ht(t));
    
    % Distribution-specific likelihood contribution
    switch options.error_type
        case 'NORMAL'
            % Normal: -0.5*log(2*pi) - 0.5*z^2
            LLs(t) = LLs(t) - 0.5*log(2*pi) - 0.5*z^2;
        case 'T'
            % Student's t
            nu = parameters(end);
            LLs(t) = LLs(t) - stdtloglik(z, nu, 0, 1);
        case 'GED'
            % GED
            nu = parameters(end);
            LLs(t) = LLs(t) + gedloglik(z, nu, 0, 1);
        case 'SKEWT'
            % Skewed t
            nu = parameters(end-1);
            lambda = parameters(end);
            [~, ll_vector] = skewtloglik(z, [nu, lambda, 0, 1]);
            LLs(t) = LLs(t) + ll_vector;
    end
end

% Return diagnostics
diagnostics = struct('ht', ht, 'LLs', LLs);
end

function H = compute_numerical_hessian(parameters, data, options)
% Compute numerical approximation of Hessian for standard error calculation
%
% Uses finite difference method to approximate the Hessian matrix of the likelihood function

n = length(parameters);
H = zeros(n, n);
f0 = nagarch_likelihood_wrapper(parameters, data, options);
h = 1e-4 * max(abs(parameters), 1e-2);

% Compute diagonal elements
for i = 1:n
    ei = zeros(n, 1);
    ei(i) = 1;
    
    % Forward difference
    fp = nagarch_likelihood_wrapper(parameters + h(i).*ei, data, options);
    
    % Backward difference
    fm = nagarch_likelihood_wrapper(parameters - h(i).*ei, data, options);
    
    % Central difference for diagonal elements
    H(i,i) = (fp - 2*f0 + fm) / (h(i)^2);
end

% Compute off-diagonal elements
for i = 1:n-1
    ei = zeros(n, 1);
    ei(i) = 1;
    
    for j = i+1:n
        ej = zeros(n, 1);
        ej(j) = 1;
        
        % Mixed partial derivative approximation
        fpp = nagarch_likelihood_wrapper(parameters + h(i).*ei + h(j).*ej, data, options);
        fpm = nagarch_likelihood_wrapper(parameters + h(i).*ei - h(j).*ej, data, options);
        fmp = nagarch_likelihood_wrapper(parameters - h(i).*ei + h(j).*ej, data, options);
        fmm = nagarch_likelihood_wrapper(parameters - h(i).*ei - h(j).*ej, data, options);
        
        H(i,j) = (fpp - fpm - fmp + fmm) / (4 * h(i) * h(j));
        H(j,i) = H(i,j);  % Symmetry
    end
end
end