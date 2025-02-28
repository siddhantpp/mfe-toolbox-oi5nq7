function [negativeLogLik] = garchlikelihood(parameters, data, options)
% GARCHLIKELIHOOD Computes the log-likelihood for GARCH model estimation
%
% USAGE:
%   [NEGATIVELOGLIK] = garchlikelihood(PARAMETERS, DATA, OPTIONS)
%
% INPUTS:
%   PARAMETERS - Vector of parameters governing the variance process and
%                potentially the error distribution.
%                Format depends on model type and distribution:
%                GARCH: [omega, alpha(1),...,alpha(q), beta(1),...,beta(p), [dist_params]]
%                GJR/TARCH: [omega, alpha(1),...,alpha(q), gamma(1),...,gamma(q), beta(1),...,beta(p), [dist_params]]
%                EGARCH: [omega, alpha(1),...,alpha(q), gamma(1),...,gamma(q), beta(1),...,beta(p), [dist_params]]
%                AGARCH: [omega, alpha(1),...,alpha(q), gamma, beta(1),...,beta(p), [dist_params]]
%                IGARCH: [omega, alpha(1),...,alpha(q-1), [dist_params]]
%                NAGARCH: [omega, alpha(1),...,alpha(q), gamma, beta(1),...,beta(p), [dist_params]]
%                
%                Where [dist_params] depends on the error distribution:
%                NORMAL: No additional parameters
%                T: [nu] (degrees of freedom, nu > 2)
%                GED: [nu] (shape parameter, nu > 0)
%                SKEWT: [nu, lambda] (nu > 2, -1 < lambda < 1)
%
%   DATA       - Zero mean residuals, T by 1 column vector
%   OPTIONS    - Structure with model specification fields:
%                model - String specifying the model type: 'GARCH' (default),
%                        'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'NAGARCH', 'IGARCH'
%                p     - Positive integer for the GARCH order (default = 1)
%                q     - Positive integer for the ARCH order (default = 1)
%                distribution - String for error distribution: 
%                             'NORMAL' (default), 'T', 'GED', 'SKEWT'
%                useMEX - Boolean, whether to use MEX implementations if available
%                         (default = true)
%                constrainStationarity - Boolean, whether to enforce stationarity
%                                     constraints (default = true)
%                backcast - Options for initial variance values (see backcast.m)
%
% OUTPUTS:
%   NEGATIVELOGLIK - Negative log-likelihood value for minimization algorithms
%
% COMMENTS:
%   This function computes the log-likelihood for GARCH model parameter 
%   estimation. The negative log-likelihood is returned for compatibility 
%   with minimization algorithms.
%
%   The log-likelihood is computed based on the conditional variance from 
%   garchcore() and the specified error distribution.
%
%   For likelihood computation, the standardized residuals are calculated as:
%   z_t = data_t / sqrt(h_t)
%
%   The total log-likelihood is the sum of the individual contributions plus
%   a log-Jacobian term for the variance transformation.
%
% EXAMPLES:
%   % Compute log-likelihood for a standard GARCH(1,1) with normal errors
%   ll = garchlikelihood([0.01; 0.1; 0.8], residuals, struct('model','GARCH'));
%
%   % Compute log-likelihood for GJR-GARCH with t-distribution
%   options = struct('model','GJR','distribution','T');
%   ll = garchlikelihood([0.01; 0.05; 0.1; 0.8; 8], residuals, options);
%
% See also GARCH, EGARCH, GJR, TARCH, AGARCH, IGARCH, NAGARCH, GARCHCORE, GARCHINIT

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Minimum likelihood value for numerical stability
MIN_LIKELIHOOD = -1e6;

%% Input validation
% Check parameters
parametercheck(parameters, 'parameters');

% Check data
data = datacheck(data, 'data');

% Ensure data is a column vector
data = columncheck(data, 'data');

%% Process options
if nargin < 3 || isempty(options)
    options = struct();
end

% Set default model type if not specified
if ~isfield(options, 'model') || isempty(options.model)
    options.model = 'GARCH';
else
    % Convert model type to uppercase
    options.model = upper(options.model);
end

% Set default GARCH order (p) if not specified
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    % Validate p (must be positive integer)
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.p, 'options.p', opts);
end

% Set default ARCH order (q) if not specified
if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    % Validate q (must be positive integer)
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.q, 'options.q', opts);
end

% Set default distribution if not specified
if ~isfield(options, 'distribution') || isempty(options.distribution)
    options.distribution = 'NORMAL';
else
    % Convert distribution type to uppercase
    options.distribution = upper(options.distribution);
end

% Set default for stationarity constraint
if ~isfield(options, 'constrainStationarity') || isempty(options.constrainStationarity)
    options.constrainStationarity = true;
end

%% Extract model specifications
modelType = options.model;
p = options.p;
q = options.q;
distributionType = options.distribution;

%% Compute conditional variances using garchcore
ht = garchcore(parameters, data, options);

%% Check for invalid variances (numerical problems)
if any(ht <= 0) || any(~isfinite(ht))
    negativeLogLik = -MIN_LIKELIHOOD;  % Return a large positive value for minimization
    return;
end

%% Standardize residuals
std_residuals = data ./ sqrt(ht);

%% Compute log-likelihood based on distribution type
switch distributionType
    case 'NORMAL'
        % Compute log-likelihood for standard normal distribution
        ll = compute_gaussian_likelihood(std_residuals);
        
    case 'T'
        % For Student's t-distribution
        % Extract degrees of freedom parameter
        nu = parameters(end);
        
        % Check if degrees of freedom is in valid range
        if nu <= 2
            negativeLogLik = -MIN_LIKELIHOOD;
            return;
        end
        
        % Compute log-likelihood for standardized t-distribution
        ll = -stdtloglik(std_residuals, nu, 0, 1);
        
    case 'GED'
        % For Generalized Error Distribution
        % Extract shape parameter
        nu = parameters(end);
        
        % Check if shape parameter is in valid range
        if nu <= 0
            negativeLogLik = -MIN_LIKELIHOOD;
            return;
        end
        
        % Compute log-likelihood for GED
        ll = gedloglik(std_residuals, nu, 0, 1);
        
    case 'SKEWT'
        % For Hansen's skewed t-distribution
        % Extract degrees of freedom and skewness parameters
        nu = parameters(end-1);
        lambda = parameters(end);
        
        % Check if parameters are in valid range
        if nu <= 2 || abs(lambda) >= 1
            negativeLogLik = -MIN_LIKELIHOOD;
            return;
        end
        
        % Compute log-likelihood for skewed t-distribution
        [~, ll_vector] = skewtloglik(std_residuals, [nu, lambda, 0, 1]);
        ll = sum(ll_vector);
        
    otherwise
        error('Unknown distribution type: %s. Supported distributions are NORMAL, T, GED, and SKEWT.', distributionType);
end

%% Add log-Jacobian term from variance transformation
logJacobianTerm = -0.5 * sum(log(ht));
ll = ll + logJacobianTerm;

%% Apply parameter constraints if enabled
if options.constrainStationarity
    constraintPenalty = apply_parameter_constraints(parameters, modelType, p, q);
    ll = ll + constraintPenalty;
end

%% Return negative log-likelihood for minimization
% Apply minimum likelihood threshold for numerical stability
if ~isfinite(ll) || ll < MIN_LIKELIHOOD
    negativeLogLik = -MIN_LIKELIHOOD;
else
    negativeLogLik = -ll;
end

end

%% Helper function to compute Gaussian log-likelihood
function ll = compute_gaussian_likelihood(std_residuals)
% Compute log-likelihood for standard normal distribution

% Log-likelihood for standard normal: -0.5*ln(2π) - 0.5*z^2
constant = -0.5 * log(2*pi);
ll = sum(constant - 0.5 * std_residuals.^2);

end

%% Helper function to apply parameter constraints
function constraintPenalty = apply_parameter_constraints(parameters, modelType, p, q)
% Apply constraints to ensure valid parameter space
% Returns 0 if constraints are satisfied, negative values otherwise

% Initialize with no penalty
constraintPenalty = 0;

% Extract basic parameters based on model type
switch modelType
    case 'GARCH'
        % Standard GARCH model
        % parameters = [omega, alpha_1,...,alpha_q, beta_1,...,beta_p, [dist_params]]
        offset = 0;
        omega = parameters(offset + 1);
        offset = offset + 1;
        
        alpha = parameters(offset + 1:offset + q);
        offset = offset + q;
        
        beta = parameters(offset + 1:offset + p);
        
        % Check stationarity constraint: sum(alpha) + sum(beta) < 1
        stationarity = sum(alpha) + sum(beta);
        if stationarity >= 1
            constraintPenalty = -1000 * (stationarity - 0.999)^2;
        end
        
        % Check positivity constraints
        if omega <= 0 || any(alpha < 0) || any(beta < 0)
            constraintPenalty = constraintPenalty - 1000;
        end
        
    case {'GJR', 'TARCH'}
        % Threshold ARCH / GJR-GARCH model
        % parameters = [omega, alpha_1,...,alpha_q, gamma_1,...,gamma_q, beta_1,...,beta_p, [dist_params]]
        offset = 0;
        omega = parameters(offset + 1);
        offset = offset + 1;
        
        alpha = parameters(offset + 1:offset + q);
        offset = offset + q;
        
        gamma = parameters(offset + 1:offset + q);
        offset = offset + q;
        
        beta = parameters(offset + 1:offset + p);
        
        % Check stationarity constraint: sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
        stationarity = sum(alpha) + 0.5*sum(gamma) + sum(beta);
        if stationarity >= 1
            constraintPenalty = -1000 * (stationarity - 0.999)^2;
        end
        
        % Check positivity constraints
        if omega <= 0 || any(alpha < 0) || any(beta < 0)
            constraintPenalty = constraintPenalty - 1000;
        end
        
    case 'EGARCH'
        % Exponential GARCH model
        % parameters = [omega, alpha_1,...,alpha_q, gamma_1,...,gamma_q, beta_1,...,beta_p, [dist_params]]
        offset = 0;
        offset = offset + 1; % Skip omega
        
        offset = offset + q; % Skip alpha
        
        offset = offset + q; % Skip gamma
        
        beta = parameters(offset + 1:offset + p);
        
        % Check stability constraint: all roots of AR polynomial outside unit circle
        % For EGARCH, this is typically |sum(beta)| < 1
        if sum(abs(beta)) >= 1
            constraintPenalty = -1000 * (sum(abs(beta)) - 0.999)^2;
        end
        
    case 'AGARCH'
        % Asymmetric GARCH model
        % parameters = [omega, alpha_1,...,alpha_q, gamma, beta_1,...,beta_p, [dist_params]]
        offset = 0;
        omega = parameters(offset + 1);
        offset = offset + 1;
        
        alpha = parameters(offset + 1:offset + q);
        offset = offset + q;
        
        offset = offset + 1; % Skip gamma
        
        beta = parameters(offset + 1:offset + p);
        
        % Check stationarity constraint: sum(alpha) + sum(beta) < 1
        stationarity = sum(alpha) + sum(beta);
        if stationarity >= 1
            constraintPenalty = -1000 * (stationarity - 0.999)^2;
        end
        
        % Check positivity constraints
        if omega <= 0 || any(alpha < 0) || any(beta < 0)
            constraintPenalty = constraintPenalty - 1000;
        end
        
    case 'IGARCH'
        % Integrated GARCH model
        % parameters = [omega, alpha_1,...,alpha_(q-1), [dist_params]]
        % Note: Last alpha and beta are constrained by IGARCH condition
        offset = 0;
        omega = parameters(offset + 1);
        offset = offset + 1;
        
        alpha = parameters(offset + 1:offset + q - 1);
        
        % For IGARCH, we need sum(alpha) < 1 and omega > 0
        if sum(alpha) >= 1 || any(alpha < 0) || omega <= 0
            constraintPenalty = -1000;
        end
        
    case 'NAGARCH'
        % Nonlinear Asymmetric GARCH model
        % parameters = [omega, alpha_1,...,alpha_q, gamma, beta_1,...,beta_p, [dist_params]]
        offset = 0;
        omega = parameters(offset + 1);
        offset = offset + 1;
        
        alpha = parameters(offset + 1:offset + q);
        offset = offset + q;
        
        gamma = parameters(offset + 1); % Single asymmetry parameter
        offset = offset + 1;
        
        beta = parameters(offset + 1:offset + p);
        
        % Check stationarity constraint for NAGARCH
        % For numerical stability, use individual terms
        alpha_term = 0;
        for i=1:length(alpha)
            alpha_term = alpha_term + alpha(i)*(1 + gamma^2);
        end
        
        stationarity = alpha_term + sum(beta);
        if stationarity >= 1
            constraintPenalty = -1000 * (stationarity - 0.999)^2;
        end
        
        % Check positivity constraints
        if omega <= 0 || any(alpha < 0) || any(beta < 0)
            constraintPenalty = constraintPenalty - 1000;
        end
        
    otherwise
        error('Unknown model type: %s', modelType);
end

end