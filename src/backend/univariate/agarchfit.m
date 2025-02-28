function [model] = agarchfit(data, options)
% AGARCHFIT Fits an Asymmetric GARCH (AGARCH) volatility model to univariate time series data
%
% USAGE:
%   [MODEL] = agarchfit(DATA)
%   [MODEL] = agarchfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA      - T by 1 vector of mean zero residuals
%   OPTIONS   - [OPTIONAL] Structure of model options. Fields include:
%               'p'            - Order of the symmetric ARCH effects [default = 1]
%               'q'            - Order of the GARCH effects [default = 1]
%               'distribution' - String specifying the error distribution:
%                                'NORMAL' - Gaussian distribution (default)
%                                'T'      - Student's t-distribution
%                                'GED'    - Generalized Error Distribution
%                                'SKEWT'  - Hansen's Skewed t-distribution
%               'useMEX'       - Boolean flag indicating whether to use MEX implementation if available [default = true]
%               'startingvals' - Vector of starting values for the AGARCH parameters [optional]
%                                Format: [omega alpha(1) ... alpha(p) gamma beta(1) ... beta(q) [dist_params]]
%               'backcast'     - Scalar or structure for initializing the variance recursion [optional]
%                                If a scalar, used directly; if a structure, passed to backcast()
%               'optimoptions' - Options passed to the optimizer (fmincon)
%                   'Algorithm'    - Algorithm used in the optimization ['interior-point']
%                   'Display'      - Level of display during optimization ['off']
%                   'MaxFunEvals'  - Maximum number of objective function evaluations [1000]
%                   'MaxIter'      - Maximum number of iterations [500]
%                   'TolFun'       - Tolerance for termination based on function value [1e-6]
%                   'TolX'         - Tolerance for termination based on parameter values [1e-6]
%
% OUTPUTS:
%   MODEL     - Structure containing estimation results with fields:
%               'distribution'   - String indicating error distribution used ('NORMAL', 'T', 'GED', 'SKEWT')
%               'p'              - Order of the symmetric ARCH effects
%               'q'              - Order of the GARCH effects
%               'parameters'     - Vector of estimated model parameters
%                                  [omega alpha(1) ... alpha(p) gamma beta(1) ... beta(q) [dist_params]]
%               'parameternames' - Cell array of parameter names
%               'LL'             - Log-likelihood value at optimum
%               'ht'             - Vector of conditional variances
%               'stdresid'       - Standardized residuals
%               'stderrors'      - Vector of standard errors for parameters
%               'tstat'          - Vector of t-statistics for parameters
%               'pvalues'        - Vector of p-values for parameters
%               'diagnostics'    - Structure of diagnostic tests
%                   'persistence'- Model persistence measure
%                   'uncvar'     - Unconditional variance
%                   'halflife'   - Volatility half-life
%               'dist_parameters'- Structure of distribution parameters
%               'information_criteria'
%                   'AIC'        - Akaike Information Criterion
%                   'BIC'        - Bayesian (Schwarz) Information Criterion
%               'convergence'    - Structure containing convergence information
%               'usedMEX'        - Boolean indicating whether MEX implementation was used
%
% COMMENTS:
%   The Asymmetric GARCH (AGARCH) model is a generalization of the standard GARCH
%   model that introduces asymmetry in the volatility process through a shift
%   parameter (gamma). The model is specified as:
%
%   h(t) = omega + sum(alpha(i)*(e(t-i) - gamma*sqrt(h(t-i)))^2, i=1,...,p) + sum(beta(j)*h(t-j), j=1,...,q)
%
%   where h(t) is the conditional variance at time t, e(t) are the innovations,
%   omega > 0 is the constant term, alpha(i) >= 0 are the ARCH parameters,
%   gamma is the asymmetry parameter, and beta(j) >= 0 are the GARCH parameters.
%
%   For improved performance, the function will use a MEX implementation (agarch_core)
%   if available and not disabled by setting options.useMEX = false.
%
% EXAMPLES:
%   % Fit an AGARCH(1,1) model with normal innovations
%   model = agarchfit(returns);
%
%   % Fit an AGARCH(1,1) model with Student's t innovations and display optimization progress
%   options = struct('distribution', 'T', 'optimoptions', struct('Display', 'iter'));
%   model = agarchfit(returns, options);
%
%   % Fit an AGARCH(2,1) model with GED innovations
%   options = struct('p', 2, 'distribution', 'GED');
%   model = agarchfit(returns, options);
%
% See also GARCHFIT, EGARCHFIT, TARCHFIT, GARCHINIT, GARCHLIKELIHOOD, BACKCAST

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default model parameters
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    options.p = parametercheck(options.p, 'p', struct('isscalar', true, 'isInteger', true, 'isPositive', true));
end

if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    options.q = parametercheck(options.q, 'q', struct('isscalar', true, 'isInteger', true, 'isPositive', true));
end

% Default error distribution
if ~isfield(options, 'distribution') || isempty(options.distribution)
    options.distribution = 'NORMAL';
else
    options.distribution = upper(options.distribution);
    valid_distributions = {'NORMAL', 'T', 'GED', 'SKEWT'};
    if ~ismember(options.distribution, valid_distributions)
        error('Invalid distribution specified. Must be one of: NORMAL, T, GED, SKEWT');
    end
end

% Default MEX usage
if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

% Default optimization options
if ~isfield(options, 'optimoptions') || isempty(options.optimoptions)
    options.optimoptions = struct();
end

if ~isfield(options.optimoptions, 'Algorithm') || isempty(options.optimoptions.Algorithm)
    options.optimoptions.Algorithm = 'interior-point';
end

if ~isfield(options.optimoptions, 'Display') || isempty(options.optimoptions.Display)
    options.optimoptions.Display = 'off';
end

if ~isfield(options.optimoptions, 'MaxFunEvals') || isempty(options.optimoptions.MaxFunEvals)
    options.optimoptions.MaxFunEvals = 1000;
end

if ~isfield(options.optimoptions, 'MaxIter') || isempty(options.optimoptions.MaxIter)
    options.optimoptions.MaxIter = 500;
end

if ~isfield(options.optimoptions, 'TolFun') || isempty(options.optimoptions.TolFun)
    options.optimoptions.TolFun = 1e-6;
end

if ~isfield(options.optimoptions, 'TolX') || isempty(options.optimoptions.TolX)
    options.optimoptions.TolX = 1e-6;
end

% Extract model orders
p = options.p;
q = options.q;

% Set up specific model specification for AGARCH
agarch_options = struct('model', 'AGARCH', 'p', p, 'q', q, 'distribution', options.distribution);

% Initialize parameters using garchinit if starting values not provided
if ~isfield(options, 'startingvals') || isempty(options.startingvals)
    parameters = garchinit(data, agarch_options);
else
    parameters = options.startingvals;
    % Validate parameter length
    min_param_length = 1 + p + 1 + q; % omega, alpha(1...p), gamma, beta(1...q)
    
    % Add distribution parameters if needed
    if strcmp(options.distribution, 'T') || strcmp(options.distribution, 'GED')
        min_param_length = min_param_length + 1; % Add one parameter (nu)
    elseif strcmp(options.distribution, 'SKEWT')
        min_param_length = min_param_length + 2; % Add two parameters (nu, lambda)
    end
    
    if length(parameters) < min_param_length
        error('Insufficient starting values provided. Need at least %d parameters.', min_param_length);
    end
end

% Generate backcast value for conditional variance
if ~isfield(options, 'backcast') || isempty(options.backcast)
    backcastValue = backcast(data);
else
    if isstruct(options.backcast)
        backcastValue = backcast(data, options.backcast);
    else
        backcastValue = options.backcast;
    end
end

% Set up constraints for optimization
% For AGARCH, we need:
% 1. omega > 0
% 2. alpha_i >= 0 for all i
% 3. beta_j >= 0 for all j
% 4. sum(alpha) + sum(beta) < 1 (stationarity)

% Number of parameters (excluding distribution parameters)
num_base_params = 1 + p + 1 + q; % omega, alpha(1...p), gamma, beta(1...q)

% Lower bounds
lb = zeros(length(parameters), 1);
lb(1) = 1e-6; % omega > 0
% alpha bounds
for i = 2:(p+1)
    lb(i) = 0; % alpha >= 0
end
% gamma is unrestricted, can be negative
% beta bounds
for i = (p+3):(p+2+q)
    lb(i) = 0; % beta >= 0
end

% Upper bounds
ub = inf(length(parameters), 1);

% Set up options for GARCH likelihood computation
likelihood_options = struct('model', 'AGARCH', 'p', p, 'q', q, 'distribution', options.distribution, 'useMEX', options.useMEX);

% Add backcast information
likelihood_options.backcast = backcastValue;

% Create a wrapper function for the likelihood
likelihood_wrapper = @(params) agarch_likelihood_wrapper(params, data, likelihood_options);

% Configure optimization options
optim_options = optimset('fmincon');
optim_field_names = fieldnames(options.optimoptions);
for i = 1:length(optim_field_names)
    field_name = optim_field_names{i};
    optim_options = optimset(optim_options, field_name, options.optimoptions.(field_name));
end

% Perform constrained optimization
[parameters, fval, exitflag, output] = fmincon(likelihood_wrapper, parameters, [], [], [], [], lb, ub, [], optim_options);

% Compute conditional variances
% Check if MEX implementation is available and should be used
usedMEX = false;
if options.useMEX && exist('agarch_core', 'file') == 3 % 3 = MEX file
    try
        % Extract parameters for MEX call
        omega = parameters(1);
        alpha = parameters(2:(p+1));
        gamma = parameters(p+2);
        beta = parameters((p+3):(p+2+q));
        
        % Call MEX function to compute variance
        ht = agarch_core(data, [omega; alpha; gamma; beta], backcastValue, p, q, length(data));
        usedMEX = true;
    catch
        % If MEX call fails, fall back to MATLAB implementation
        warning('MEX implementation failed, falling back to MATLAB implementation.');
        ht = compute_agarch_variance(data, parameters, backcastValue, p, q);
    end
else
    % Use MATLAB implementation
    ht = compute_agarch_variance(data, parameters, backcastValue, p, q);
end

% Compute standardized residuals
stdresid = data ./ sqrt(ht);

% Fit distribution to standardized residuals and get distribution parameters
dist_parameters = struct();
switch options.distribution
    case 'NORMAL'
        % No additional parameters for normal distribution
        
    case 'T'
        % Student's t with nu parameter
        t_fit = stdtfit(stdresid);
        dist_parameters.nu = t_fit.nu;
        dist_parameters.nuSE = t_fit.nuSE;
        
    case 'GED'
        % GED with nu parameter
        ged_fit = gedfit(stdresid);
        dist_parameters.nu = ged_fit.nu;
        dist_parameters.nuSE = ged_fit.stderrors(1);
        
    case 'SKEWT'
        % Skewed t with nu and lambda parameters
        skewt_fit = skewtfit(stdresid);
        dist_parameters.nu = skewt_fit.nu;
        dist_parameters.lambda = skewt_fit.lambda;
        dist_parameters.nuSE = skewt_fit.nuSE;
        dist_parameters.lambdaSE = skewt_fit.lambdaSE;
end

% Compute log-likelihood at optimum
logL = -fval; % Convert from negative log-likelihood to log-likelihood

% Compute AIC and BIC
num_estimated_params = length(parameters);
T = length(data);
ic = aicsbic(logL, num_estimated_params, T);

% Compute model diagnostics
omega = parameters(1);
alpha = parameters(2:(p+1));
gamma = parameters(p+2);
beta = parameters((p+3):(p+2+q));

% Persistence measure
persistence = sum(alpha) + sum(beta);

% Unconditional variance (if stationarity condition met)
if persistence < 1
    uncvar = omega / (1 - persistence);
else
    uncvar = NaN;
end

% Volatility half-life (if persistence < 1)
if persistence < 1
    halflife = log(0.5) / log(persistence);
else
    halflife = Inf;
end

% Prepare parameter names
parameternames = cell(length(parameters), 1);
parameternames{1} = 'omega';
for i = 1:p
    parameternames{i+1} = sprintf('alpha(%d)', i);
end
parameternames{p+2} = 'gamma';
for i = 1:q
    parameternames{p+2+i} = sprintf('beta(%d)', i);
end

% Add distribution parameter names
if strcmp(options.distribution, 'T') || strcmp(options.distribution, 'GED')
    if length(parameternames) >= p+q+3
        parameternames{p+q+3} = 'nu';
    end
elseif strcmp(options.distribution, 'SKEWT')
    if length(parameternames) >= p+q+3
        parameternames{p+q+3} = 'nu';
    end
    if length(parameternames) >= p+q+4
        parameternames{p+q+4} = 'lambda';
    end
end

% Calculate standard errors (this would require Hessian)
% For simplicity, we'll use numerical approximation of the Hessian
stderrors = zeros(length(parameters), 1);
try
    % Compute numerical approximation of Hessian using finite differences
    h = 1e-4 * max(abs(parameters), 1);
    hessian = zeros(length(parameters));
    
    % Calculate diagonal elements
    for i = 1:length(parameters)
        % Create parameter vectors with small perturbations
        params_plus = parameters;
        params_plus(i) = params_plus(i) + h(i);
        params_minus = parameters;
        params_minus(i) = params_minus(i) - h(i);
        
        % Compute likelihood at perturbed points
        f_plus = likelihood_wrapper(params_plus);
        f_minus = likelihood_wrapper(params_minus);
        f_center = fval;
        
        % Second derivative approximation
        hessian(i,i) = (f_plus - 2*f_center + f_minus) / (h(i)^2);
    end
    
    % Calculate off-diagonal elements (if needed for better accuracy)
    for i = 1:length(parameters)-1
        for j = i+1:length(parameters)
            % Create parameter vectors with small perturbations
            params_plus_plus = parameters;
            params_plus_plus(i) = params_plus_plus(i) + h(i);
            params_plus_plus(j) = params_plus_plus(j) + h(j);
            
            params_minus_minus = parameters;
            params_minus_minus(i) = params_minus_minus(i) - h(i);
            params_minus_minus(j) = params_minus_minus(j) - h(j);
            
            params_plus_minus = parameters;
            params_plus_minus(i) = params_plus_minus(i) + h(i);
            params_plus_minus(j) = params_plus_minus(j) - h(j);
            
            params_minus_plus = parameters;
            params_minus_plus(i) = params_minus_plus(i) - h(i);
            params_minus_plus(j) = params_minus_plus(j) + h(j);
            
            % Compute likelihood at perturbed points
            f_plus_plus = likelihood_wrapper(params_plus_plus);
            f_minus_minus = likelihood_wrapper(params_minus_minus);
            f_plus_minus = likelihood_wrapper(params_plus_minus);
            f_minus_plus = likelihood_wrapper(params_minus_plus);
            
            % Cross-derivative approximation
            hessian(i,j) = (f_plus_plus + f_minus_minus - f_plus_minus - f_minus_plus) / (4 * h(i) * h(j));
            hessian(j,i) = hessian(i,j); % Symmetric
        end
    end
    
    % Compute inverse of hessian for covariance matrix
    covariance = inv(hessian);
    
    % Extract standard errors
    stderrors = sqrt(diag(covariance));
catch
    warning('Standard error calculation failed. Standard errors set to NaN.');
    stderrors = NaN(length(parameters), 1);
end

% Calculate t-statistics and p-values
tstat = parameters ./ stderrors;
pvalues = 2 * (1 - tcdf(abs(tstat), T - length(parameters)));

% Structure the results
model = struct();
model.distribution = options.distribution;
model.p = p;
model.q = q;
model.parameters = parameters;
model.parameternames = parameternames;
model.LL = logL;
model.ht = ht;
model.stdresid = stdresid;
model.stderrors = stderrors;
model.tstat = tstat;
model.pvalues = pvalues;
model.diagnostics = struct('persistence', persistence, 'uncvar', uncvar, 'halflife', halflife);
model.dist_parameters = dist_parameters;
model.information_criteria = struct('AIC', ic.aic, 'BIC', ic.sbic);
model.convergence = struct('exitflag', exitflag, 'message', output.message);
model.usedMEX = usedMEX;

end

function negLogL = agarch_likelihood_wrapper(parameters, data, options)
% AGARCH_LIKELIHOOD_WRAPPER Internal helper function that wraps the garchlikelihood function for use with optimization routines.
%
% INPUTS:
%   parameters - Vector of model parameters
%   data       - Time series data (T x 1)
%   options    - Options structure for garchlikelihood
%
% OUTPUTS:
%   negLogL    - Negative log-likelihood value

options.parameters = parameters;
negLogL = garchlikelihood(parameters, data, options);
end

function ht = compute_agarch_variance(data, parameters, backcastValue, p, q)
% COMPUTE_AGARCH_VARIANCE Computes conditional variances using the AGARCH model specification.
%
% INPUTS:
%   data           - Time series data (T x 1)
%   parameters     - Vector of model parameters [omega, alpha(1...p), gamma, beta(1...q)]
%   backcastValue  - Initial value for variance recursion
%   p              - Order of the ARCH component
%   q              - Order of the GARCH component
%
% OUTPUTS:
%   ht             - Conditional variance series (T x 1)

% Extract parameters
omega = parameters(1);
alpha = parameters(2:(p+1));
gamma = parameters(p+2);
beta = parameters((p+3):(p+2+q));

% Data length
T = length(data);

% Initialize variance series with backcast value
ht = ones(T, 1) * backcastValue;

% Minimum variance for numerical stability
MIN_VARIANCE = 1e-10;

% AGARCH recursion
for t = 2:T
    ht(t) = omega;
    
    % Add ARCH terms with asymmetry
    for i = 1:p
        if t-i > 0
            % (e_{t-i} - gamma * sqrt(h_{t-i}))^2
            shock = data(t-i) - gamma * sqrt(ht(t-i));
            ht(t) = ht(t) + alpha(i) * shock^2;
        else
            % Use backcast for pre-sample values
            ht(t) = ht(t) + alpha(i) * backcastValue;
        end
    end
    
    % Add GARCH terms
    for j = 1:q
        if t-j > 0
            ht(t) = ht(t) + beta(j) * ht(t-j);
        else
            ht(t) = ht(t) + beta(j) * backcastValue;
        end
    end
    
    % Ensure variance is positive
    ht(t) = max(ht(t), MIN_VARIANCE);
end
end