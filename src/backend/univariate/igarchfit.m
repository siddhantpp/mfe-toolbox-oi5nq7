function parameters = igarchfit(data, options)
% IGARCHFIT Estimates parameters of the Integrated GARCH (IGARCH) model
% with unit persistence constraint (sum of alpha and beta coefficients equals 1)
% for financial time series volatility modeling.
%
% USAGE:
%   [PARAMETERS] = igarchfit(DATA)
%   [PARAMETERS] = igarchfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - Vector of mean zero residuals (T×1)
%   OPTIONS  - [OPTIONAL] Options structure with fields:
%              'p'            - Non-negative integer for GARCH order (Default: 1)
%              'q'            - Non-negative integer for ARCH order (Default: 1)
%              'distribution' - String specifying the error distribution:
%                 'NORMAL' (Default) - Gaussian errors
%                 'T'               - Student's t distribution
%                 'GED'             - Generalized Error Distribution
%                 'SKEWT'           - Hansen's Skewed t distribution
%              'useMEX'       - Boolean, whether to use MEX optimization if available
%                                (Default: true)
%              'startingvals' - Vector of starting values (optional)
%              'optimizer'    - String specifying optimizer (default='FMINCON')
%              'optimoptions' - Options for the optimizer
%
% OUTPUTS:
%   PARAMETERS - Structure with fields:
%                'omega'       - Constant parameter in variance equation
%                'alpha'       - ARCH coefficients (q×1 vector)
%                'beta'        - GARCH coefficients (p×1 vector)
%                'distribution'- Distribution used in estimation
%                'LL'          - Log-likelihood at optimum
%                'ht'          - Conditional variances (T×1 vector)
%                'stdresid'    - Standardized residuals
%                'stderrors'   - Standard errors of parameters
%                'p'           - GARCH order
%                'q'           - ARCH order
%                'T'           - Sample size
%                'diagnostics' - Structure with goodness-of-fit statistics
%                                - AIC, BIC and other criteria
%                'optimization'- Optimization information
%                                - exitflag, iterations, convergence status
%                And possibly distribution-specific parameters:
%                'nu'          - Degrees of freedom (T, GED, SKEWT distributions)
%                'lambda'      - Skewness parameter (SKEWT distribution)
%
% COMMENTS:
%   IGARCH models estimate the conditional variance with a unit persistence
%   constraint: sum(alpha) + sum(beta) = 1. This means the unconditional
%   variance doesn't exist, creating long-memory in the variance process.
%   
%   The variance equation for IGARCH(p,q) is:
%   h(t) = omega + sum(alpha_i * e(t-i)^2) + sum(beta_j * h(t-j))
%   
%   Subject to: sum(alpha_i) + sum(beta_j) = 1
%
% EXAMPLES:
%   % Estimate a standard IGARCH(1,1) model with normal errors
%   parameters = igarchfit(data)
%
%   % Estimate an IGARCH(1,1) model with t-distributed errors 
%   options.distribution = 'T';
%   parameters = igarchfit(data, options)
%
%   % Estimate an IGARCH(2,1) model
%   options.p = 2;
%   parameters = igarchfit(data, options)
%
% See also GARCHFIT, EGARCHFIT, GJRFIT, GARCHCORE, GARCHLIKELIHOOD

% Define global constants for minimum parameter values
MIN_ALPHA = 1e-8;
MIN_OMEGA = 1e-8;

%% Step 1: Validate input data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

%% Step 2: Process options structure
if nargin < 2
    options = [];
end

% Default options
if isempty(options)
    options = struct();
end

% Default model order (p)
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    validate_opts = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
    options.p = parametercheck(options.p, 'options.p', validate_opts);
end

% Default model order (q)
if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    validate_opts = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
    options.q = parametercheck(options.q, 'options.q', validate_opts);
end

% Default distribution
if ~isfield(options, 'distribution') || isempty(options.distribution)
    options.distribution = 'NORMAL';
else
    options.distribution = upper(options.distribution);
    valid_distributions = {'NORMAL', 'T', 'GED', 'SKEWT'};
    if ~ismember(options.distribution, valid_distributions)
        error('Invalid distribution. Must be one of: NORMAL, T, GED, SKEWT');
    end
end

% Default MEX usage
if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

% Extract model orders
p = options.p;
q = options.q;

%% Step 3: Calculate initial backcast value for variance recursion
if ~isfield(options, 'backcast') || isempty(options.backcast)
    backcastValue = backcast(data);
else
    if isstruct(options.backcast)
        backcastValue = backcast(data, options.backcast);
    else
        backcastValue = options.backcast;
    end
end

%% Step 4: Initialize parameters
% For IGARCH, the constraint is sum(alpha) + sum(beta) = 1
% We only estimate omega and alpha parameters, and derive beta from the constraint

% Initialize using GARCH initialization and then modify
init_options = options;
init_options.model = 'IGARCH';
initial_params = garchinit(data, init_options);

% Extract omega
omega = initial_params(1);

% For IGARCH, we need to make sure the alpha parameters sum to less than 1
% We'll optimize alpha values and derive beta from the constraint
alpha = initial_params(2:q+1);

% Ensure sum(alpha) < 1 for IGARCH constraint
if sum(alpha) >= 0.999
    alpha = alpha * 0.8 / sum(alpha);  % Scale down alpha to leave room for beta
end

% Initial parameter vector for optimization (omega and alpha parameters only)
parameters_vec = [omega; alpha];

%% Step 5: Configure optimization
% Set up optimization options
if ~isfield(options, 'optimoptions') || isempty(options.optimoptions)
    % Default optimization options
    optimOptions = optimset('fmincon');
    optimOptions = optimset(optimOptions, 'Display', 'off', 'TolFun', 1e-6, 'TolX', 1e-6, ...
        'Algorithm', 'interior-point', 'MaxIter', 1000, 'MaxFunEvals', 1000);
else
    optimOptions = options.optimoptions;
end

% Lower bounds: omega > 0, alpha > 0
lb = [MIN_OMEGA; MIN_ALPHA * ones(q, 1)];

% Upper bounds: omega < Inf, alpha < 1 (to ensure beta is positive)
ub = [Inf; 0.999 * ones(q, 1)];

% Linear inequality constraints: sum(alpha) <= 0.999 to ensure beta is positive
A = [0, ones(1, q)];
b = 0.999;

%% Step 6: Define the objective function for optimization
objective_function = @(params) igarch_likelihood_wrapper(params, data, options, backcastValue);

%% Step 7: Run constrained optimization
[final_params, fval, exitflag, output] = fmincon(objective_function, parameters_vec, A, b, [], [], lb, ub, [], optimOptions);

%% Step 8: Extract optimized parameters
omega = final_params(1);
alpha = final_params(2:end);

% Apply IGARCH constraint to derive beta coefficients
beta_sum = 1 - sum(alpha);
if p == 1
    beta = beta_sum;
else
    % For p>1, distribute beta_sum with exponential decay
    beta = zeros(p, 1);
    decay_factor = 0.8;
    for i = 1:p
        beta(i) = beta_sum * decay_factor^(i-1);
    end
    beta = beta / sum(beta) * beta_sum;  % Normalize to ensure sum(beta) = beta_sum
end

%% Step 9: Calculate conditional variances using final parameters
% Construct full parameter vector for variance computation
full_params = [omega; alpha; beta];

% Options for garchcore
garch_options = struct('model', 'IGARCH', 'p', p, 'q', q, 'useMEX', options.useMEX, 'backcast', backcastValue);

% Compute conditional variances
ht = garchcore(full_params, data, garch_options);

%% Step 10: Compute standardized residuals
std_residuals = data ./ sqrt(ht);

%% Step 11: Estimate distribution parameters if not normal
dist_params = [];
switch options.distribution
    case 'T'
        % Estimate t-distribution parameters
        t_results = stdtfit(std_residuals);
        nu = t_results.nu;
        dist_params = nu;
        full_params = [full_params; nu];
        
    case 'GED'
        % Estimate GED parameters
        ged_results = gedfit(std_residuals);
        nu = ged_results.nu;
        dist_params = nu;
        full_params = [full_params; nu];
        
    case 'SKEWT'
        % Estimate skewed t-distribution parameters
        skewt_results = skewtfit(std_residuals);
        nu = skewt_results.nu;
        lambda = skewt_results.lambda;
        dist_params = [nu; lambda];
        full_params = [full_params; nu; lambda];
end

%% Step 12: Calculate final log-likelihood with all parameters
garch_options.distribution = options.distribution;
logL = -garchlikelihood(full_params, data, garch_options);

%% Step 13: Compute standard errors
se = compute_standard_errors(final_params, data, options, backcastValue, logL);

%% Step 14: Calculate information criteria
T = length(data);
n_params = length(final_params) + length(dist_params);
ic = aicsbic(logL, n_params, T);

%% Step 15: Construct the output structure
parameters = struct();

% Model parameters
parameters.omega = omega;
parameters.alpha = alpha;
parameters.beta = beta;

% Distribution parameters
if strcmp(options.distribution, 'T') || strcmp(options.distribution, 'GED')
    parameters.nu = dist_params;
elseif strcmp(options.distribution, 'SKEWT')
    parameters.nu = dist_params(1);
    parameters.lambda = dist_params(2);
end

% Model information
parameters.p = p;
parameters.q = q;
parameters.T = T;
parameters.distribution = options.distribution;
parameters.LL = logL;
parameters.ht = ht;
parameters.stdresid = std_residuals;

% Standard errors
parameters.stderrors.omega = se.omega;
parameters.stderrors.alpha = se.alpha;
parameters.stderrors.beta = se.beta;

% Diagnostics
parameters.diagnostics.AIC = ic.aic;
parameters.diagnostics.BIC = ic.sbic;
parameters.diagnostics.HQIC = -2*logL + 2*n_params*log(log(T));
parameters.diagnostics.logL = logL;

% Optimization information
parameters.optimization.exitflag = exitflag;
parameters.optimization.iterations = output.iterations;
parameters.optimization.converged = (exitflag > 0);
parameters.optimization.message = output.message;

%% Step 16: Check for convergence issues
if exitflag <= 0
    warning('MFE:Convergence', 'IGARCH estimation may not have converged. Exitflag = %d', exitflag);
end

% Warning for small omega
if omega < MIN_OMEGA * 10
    warning('MFE:SmallOmega', 'Estimated omega is very small (%g). Model may be misspecified.', omega);
end

end

%% Helper function to wrap the likelihood function with the IGARCH constraint
function nll = igarch_likelihood_wrapper(parameters, data, options, backcast)
    % This function wraps garchlikelihood with the IGARCH constraint applied
    
    % Extract parameters
    omega = parameters(1);
    alpha = parameters(2:end);
    
    % Get model orders
    p = options.p;
    q = options.q;
    
    % Apply IGARCH constraint to get beta coefficients
    beta_sum = 1 - sum(alpha);
    
    % Beta values are derived from the constraint
    if p == 1
        beta = beta_sum;
    else
        decay_factor = 0.8;
        beta = zeros(p, 1);
        for i = 1:p
            beta(i) = beta_sum * decay_factor^(i-1);
        end
        beta = beta / sum(beta) * beta_sum;
    end
    
    % Construct full parameter vector
    full_params = [omega; alpha; beta];
    
    % Add distribution parameters (using default values for optimization)
    switch options.distribution
        case 'T'
            full_params = [full_params; 8];  % Default degrees of freedom
        case 'GED'
            full_params = [full_params; 1.5];  % Default shape parameter
        case 'SKEWT'
            full_params = [full_params; 8; 0];  % Default nu and lambda
    end
    
    % Configure options for garchlikelihood
    garch_options = struct('model', 'IGARCH', 'p', p, 'q', q, ...
        'distribution', options.distribution, 'useMEX', options.useMEX, ...
        'backcast', backcast);
    
    % Calculate negative log-likelihood
    nll = garchlikelihood(full_params, data, garch_options);
    
    % Add penalty for parameters near bounds
    if any(alpha < MIN_ALPHA) || omega < MIN_OMEGA
        nll = nll + 1e6;
    end
end

%% Helper function to enforce the IGARCH constraint
function [constrained_params, beta] = enforce_igarch_constraint(parameters, p, q)
    % Extract parameters
    omega = parameters(1);
    alpha = parameters(2:q+1);
    
    % Apply IGARCH constraint to get beta
    beta_sum = 1 - sum(alpha);
    
    % Ensure beta_sum is positive (necessary for valid model)
    if beta_sum <= 0
        % Scale down alpha to ensure at least some persistence through beta
        alpha = alpha * 0.9 / sum(alpha);
        beta_sum = 1 - sum(alpha);
    end
    
    % For p=1, beta is simply beta_sum
    if p == 1
        beta = beta_sum;
    else
        % For p>1, distribute beta_sum with exponential decay
        decay_factor = 0.8;
        beta = zeros(p, 1);
        for i = 1:p
            beta(i) = beta_sum * decay_factor^(i-1);
        end
        beta = beta / sum(beta) * beta_sum;
    end
    
    % Ensure minimum values for numerical stability
    alpha = max(alpha, MIN_ALPHA);
    
    % Construct constrained parameter vector
    constrained_params = [omega; alpha; beta];
end

%% Helper function to compute standard errors
function se = compute_standard_errors(parameters, data, options, backcast, logL)
    % Compute numerical approximation of the Hessian for standard errors
    
    % Step size for finite differencing
    h = max(abs(parameters), 1e-2) * 1e-4;
    
    % Number of parameters being optimized
    n_params = length(parameters);
    
    % Initialize Hessian matrix
    Hessian = zeros(n_params, n_params);
    
    % Function handle for negative log-likelihood
    obj_fun = @(params) igarch_likelihood_wrapper(params, data, options, backcast);
    
    % Optimal function value
    f0 = obj_fun(parameters);
    
    try
        % Compute diagonal elements of Hessian
        for i = 1:n_params
            % Perturb i-th parameter positively
            theta_plus = parameters;
            theta_plus(i) = theta_plus(i) + h(i);
            f_plus = obj_fun(theta_plus);
            
            % Perturb i-th parameter negatively (ensuring it stays valid)
            theta_minus = parameters;
            theta_minus(i) = max(theta_minus(i) - h(i), MIN_OMEGA/10);
            f_minus = obj_fun(theta_minus);
            
            % Second derivative approximation
            Hessian(i,i) = (f_plus - 2*f0 + f_minus) / (h(i)^2);
        end
        
        % Compute off-diagonal elements
        for i = 1:n_params-1
            for j = i+1:n_params
                % Perturb both i and j positively
                theta_pp = parameters;
                theta_pp(i) = theta_pp(i) + h(i);
                theta_pp(j) = theta_pp(j) + h(j);
                f_pp = obj_fun(theta_pp);
                
                % Perturb i positively, j negatively
                theta_pm = parameters;
                theta_pm(i) = theta_pm(i) + h(i);
                theta_pm(j) = max(theta_pm(j) - h(j), MIN_OMEGA/10);
                f_pm = obj_fun(theta_pm);
                
                % Perturb i negatively, j positively
                theta_mp = parameters;
                theta_mp(i) = max(theta_mp(i) - h(i), MIN_OMEGA/10);
                theta_mp(j) = theta_mp(j) + h(j);
                f_mp = obj_fun(theta_mp);
                
                % Perturb both i and j negatively
                theta_mm = parameters;
                theta_mm(i) = max(theta_mm(i) - h(i), MIN_OMEGA/10);
                theta_mm(j) = max(theta_mm(j) - h(j), MIN_OMEGA/10);
                f_mm = obj_fun(theta_mm);
                
                % Mixed partial derivative approximation
                Hessian(i,j) = (f_pp - f_pm - f_mp + f_mm) / (4 * h(i) * h(j));
                Hessian(j,i) = Hessian(i,j);  % Symmetry
            end
        end
        
        % Ensure Hessian is positive definite
        [V, D] = eig(Hessian);
        d = diag(D);
        if any(d <= 0)
            d(d <= 0) = 1e-6;
            Hessian = V * diag(d) * V';
        end
        
        % Compute covariance matrix by inverting the Hessian
        cov_matrix = inv(Hessian);
        
        % Extract standard errors (square root of diagonal elements)
        se.omega = sqrt(cov_matrix(1,1));
        
        % Alpha standard errors
        q = options.q;
        se.alpha = zeros(q, 1);
        for i = 1:q
            if i+1 <= n_params
                se.alpha(i) = sqrt(max(cov_matrix(i+1, i+1), 0));
            else
                se.alpha(i) = NaN;
            end
        end
        
        % Compute variance of sum(alpha) for delta method
        alpha_indices = 2:n_params;
        var_sum_alpha = 0;
        for i = alpha_indices
            for j = alpha_indices
                var_sum_alpha = var_sum_alpha + cov_matrix(i,j);
            end
        end
        
        % Beta standard errors
        % For IGARCH these are derived using delta method since beta is a function of alpha
        p = options.p;
        se.beta = zeros(p, 1);
        
        % For p=1, beta = 1-sum(alpha), so Var(beta) = Var(sum(alpha))
        if p == 1
            se.beta(1) = sqrt(var_sum_alpha);
        else
            % For multiple beta, calculate SE based on their relative values
            beta_sum = 1 - sum(parameters(2:end));
            
            if p == 1
                beta = beta_sum;
            else
                decay_factor = 0.8;
                beta = zeros(p, 1);
                for i = 1:p
                    beta(i) = beta_sum * decay_factor^(i-1);
                end
                beta = beta / sum(beta) * beta_sum;
            end
            
            for i = 1:p
                se.beta(i) = sqrt(var_sum_alpha) * beta(i) / sum(beta);
            end
        end
    catch
        % If standard error calculation fails, set all to NaN
        se.omega = NaN;
        se.alpha = NaN(options.q, 1);
        se.beta = NaN(options.p, 1);
        warning('MFE:StandardErrors', 'Standard error calculation failed. Results may be unreliable.');
    end
end