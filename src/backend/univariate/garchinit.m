function parameters = garchinit(data, options)
% GARCHINIT Initializes parameters for GARCH model estimation
%
% USAGE:
%   [PARAMETERS] = garchinit(DATA)
%   [PARAMETERS] = garchinit(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - Vector of mean zero residuals
%   OPTIONS  - [OPTIONAL] Options structure with fields:
%              'model' - Type of variance model:
%                        'GARCH' - Standard GARCH (Default)
%                        'EGARCH' - Exponential GARCH
%                        'GJR' or 'TARCH' - GJR or Threshold ARCH  
%                        'AGARCH' - Asymmetric GARCH
%                        'NAGARCH' - Nonlinear Asymmetric GARCH
%                        'IGARCH' - Integrated GARCH
%              'p' - Order of symmetric innovations [1]
%              'q' - Order of lagged conditional variance [1]
%              'distribution' - Distribution of innovations:
%                        'NORMAL' - Gaussian distribution (Default)
%                        'T' - Student's t distribution
%                        'GED' - Generalized Error Distribution
%                        'SKEWT' - Hansen's Skewed t distribution
%
% OUTPUTS:
%   PARAMETERS - Initial parameter values for the selected GARCH model
%                The structure depends on the model type and distribution
%
% COMMENTS:
%   This function provides intelligent starting values for GARCH model estimation
%   based on the characteristics of the data and model specification. Good starting
%   values are critical for convergence of maximum likelihood estimation in
%   volatility modeling.
%
%   The default model is a GARCH(1,1) with normal innovations.
%
% EXAMPLES:
%   % Initialize parameters for a standard GARCH(1,1) model
%   parameters = garchinit(data);
%
%   % Initialize parameters for a GJR-GARCH(1,1) with t-distribution
%   options = struct('model', 'GJR', 'distribution', 'T');
%   parameters = garchinit(data, options);
%
%   % Initialize parameters for an EGARCH(2,1) model
%   options = struct('model', 'EGARCH', 'p', 2);
%   parameters = garchinit(data, options);
%
% See also GARCH, EGARCH, TARCH, APARCH, AGARCH, IGARCH, BACKCAST

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Define constants for minimum values and defaults
MIN_OMEGA = 1e-6;  % Minimum value for omega parameter
DEFAULT_P = 1;     % Default ARCH order
DEFAULT_Q = 1;     % Default GARCH order

% Step 1: Validate input data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Step 2: Process options
if nargin < 2 || isempty(options)
    options = struct('model', 'GARCH', 'p', DEFAULT_P, 'q', DEFAULT_Q, 'distribution', 'NORMAL');
end

% Step 3: Validate and extract model type
if ~isfield(options, 'model')
    options.model = 'GARCH';
end
modelType = upper(options.model);

% Step 4: Validate and extract model orders p and q
if ~isfield(options, 'p')
    p = DEFAULT_P;
else
    p = options.p;
    if p < 0 || floor(p) ~= p
        error('P must be a non-negative integer.');
    end
end

if ~isfield(options, 'q')
    q = DEFAULT_Q;
else
    q = options.q;
    if q < 0 || floor(q) ~= q
        error('Q must be a non-negative integer.');
    end
end

% Step 5: Validate and extract distribution type
if ~isfield(options, 'distribution')
    options.distribution = 'NORMAL';
end
distributionType = upper(options.distribution);

% Step 6: Initialize parameters based on the model type
switch modelType
    case 'GARCH'
        parameters = initialize_garch_parameters(data, p, q);
    case {'GJR', 'TARCH'}
        parameters = initialize_tarch_parameters(data, p, q);
    case 'EGARCH'
        parameters = initialize_egarch_parameters(data, p, q);
    case 'AGARCH'
        % For AGARCH, start with GARCH parameters then modify
        base_params = initialize_garch_parameters(data, p, q);
        
        % Extract components
        omega = base_params(1);
        alpha = base_params(2:(p+1));
        beta = base_params((p+2):end);
        
        % Add gamma parameters for asymmetry (one per alpha)
        gamma = zeros(p, 1);
        for i = 1:p
            % Calculate average asymmetry in the data
            neg_returns = data(data < 0);
            pos_returns = data(data > 0);
            
            if mean(neg_returns.^2) > mean(pos_returns.^2)
                gamma_total = 0.1;  % Positive for leverage effect
            else
                gamma_total = 0.05;  % Small value if no clear leverage
            end
            
            gamma(i) = gamma_total / p * (1.3^(-(i-1)));
        end
        
        % Assemble parameters
        parameters = [omega; alpha; gamma; beta];
        
    case 'IGARCH'
        % For IGARCH, beta is constrained so sum(alpha) + sum(beta) = 1
        uncond_var = var(data);
        
        % Small omega for IGARCH
        omega = max(0.01 * uncond_var, MIN_OMEGA);
        
        % Initialize alpha (IGARCH typically has smaller alpha than GARCH)
        alpha = zeros(p, 1);
        if p > 0
            alpha_total = 0.2;
            for i = 1:p
                alpha(i) = alpha_total / p * (1.5^(-(i-1)));
            end
            alpha = alpha / sum(alpha) * alpha_total;
        end
        
        % For IGARCH, we only return omega and alpha as parameters
        parameters = [omega; alpha];
        
    case 'NAGARCH'
        % For NAGARCH, start with GARCH parameters then modify
        base_params = initialize_garch_parameters(data, p, q);
        
        % Extract components
        omega = base_params(1);
        alpha = base_params(2:(p+1));
        beta = base_params((p+2):end);
        
        % Add gamma parameters for nonlinear asymmetry (one per alpha)
        gamma = zeros(p, 1);
        for i = 1:p
            % Calculate average asymmetry in the data
            neg_returns = data(data < 0);
            pos_returns = data(data > 0);
            
            if mean(neg_returns.^2) > mean(pos_returns.^2)
                gamma_total = 0.1;  % Positive for leverage effect
            else
                gamma_total = 0.05;  % Small value if no clear leverage
            end
            
            gamma(i) = gamma_total / p * (1.3^(-(i-1)));
        end
        
        % Assemble parameters
        parameters = [omega; alpha; gamma; beta];
        
    otherwise
        error('Unsupported model type: %s', modelType);
end

% Step 7: Add distribution parameters if needed
distParams = initialize_distribution_parameters(distributionType);
if ~isempty(distParams)
    parameters = [parameters; distParams];
end

end

function parameters = initialize_garch_parameters(data, p, q)
% Helper function that initializes parameters for standard GARCH(p,q) model
%
% INPUTS:
%   data - Time series data (residuals)
%   p    - ARCH order
%   q    - GARCH order
%
% OUTPUTS:
%   parameters - Initial parameter vector [omega; alpha_1,...,alpha_p; beta_1,...,beta_q]

% Define minimum omega value
MIN_OMEGA = 1e-6;

% Calculate the unconditional variance for scaling
uncond_var = var(data);

% Initialize omega (constant term) - small fraction of unconditional variance
omega = max(0.05 * uncond_var, MIN_OMEGA);

% Initialize alpha coefficients (ARCH terms) - small decreasing values
alpha = zeros(p, 1);
if p > 0
    total_alpha = 0.15;  % Total weight for all alpha parameters
    for i = 1:p
        alpha(i) = total_alpha / p * (1.5^(-(i-1)));
    end
    % Normalize to ensure sum equals target
    if sum(alpha) > 0
        alpha = alpha / sum(alpha) * total_alpha;
    end
end

% Initialize beta coefficients (GARCH terms) - larger decreasing values
beta = zeros(q, 1);
if q > 0
    total_beta = 0.75;  % Total weight for all beta parameters
    for i = 1:q
        beta(i) = total_beta / q * (1.2^(-(i-1)));
    end
    % Normalize to ensure sum equals target
    if sum(beta) > 0
        beta = beta / sum(beta) * total_beta;
    end
end

% Combine all parameters
parameters = [omega; alpha; beta];

% Ensure stationarity condition: sum(alpha) + sum(beta) < 1
if sum(alpha) + sum(beta) >= 0.999
    scale_factor = 0.99 / (sum(alpha) + sum(beta));
    alpha = alpha * scale_factor;
    beta = beta * scale_factor;
    parameters = [omega; alpha; beta];
end

end

function parameters = initialize_egarch_parameters(data, p, q)
% Helper function that initializes parameters for EGARCH model
%
% INPUTS:
%   data - Time series data (residuals)
%   p    - ARCH order
%   q    - GARCH order
%
% OUTPUTS:
%   parameters - Initial parameter vector [omega; alpha_1,...,alpha_p; gamma_1,...,gamma_p; beta_1,...,beta_q]

% Calculate log of unconditional variance for omega initialization
uncond_var = var(data);
log_var = log(uncond_var);

% Initialize omega based on the portion of log variance not explained by persistence
omega = log_var * (1 - 0.75);

% Initialize alpha coefficients (symmetric effect)
alpha = zeros(p, 1);
if p > 0
    alpha_total = 0.1;
    for i = 1:p
        alpha(i) = alpha_total / p * (1.5^(-(i-1)));
    end
    if sum(alpha) > 0
        alpha = alpha / sum(alpha) * alpha_total;
    end
end

% Initialize gamma coefficients (asymmetric effect)
% Typically negative for EGARCH to capture leverage effect
gamma = zeros(p, 1);
if p > 0
    % Calculate if negative returns have higher volatility than positive returns
    neg_returns = data(data < 0);
    pos_returns = data(data > 0);
    
    if ~isempty(neg_returns) && ~isempty(pos_returns) && mean(neg_returns.^2) > mean(pos_returns.^2)
        gamma_total = -0.1;  % Negative for leverage effect
    else
        gamma_total = 0.05;  % Small positive if no clear leverage
    end
    
    for i = 1:p
        gamma(i) = gamma_total / p * (1.3^(-(i-1)));
    end
end

% Initialize beta coefficients (persistence)
beta = zeros(q, 1);
if q > 0
    beta_total = 0.75;  % High persistence typical in financial data
    for i = 1:q
        beta(i) = beta_total / q * (1.2^(-(i-1)));
    end
    if sum(beta) > 0
        beta = beta / sum(beta) * beta_total;
    end
end

% Combine all parameters
parameters = [omega; alpha; gamma; beta];

end

function parameters = initialize_tarch_parameters(data, p, q)
% Helper function that initializes parameters for Threshold ARCH (TARCH) model
%
% INPUTS:
%   data - Time series data (residuals)
%   p    - ARCH order
%   q    - GARCH order
%
% OUTPUTS:
%   parameters - Initial parameter vector [omega; alpha_1,...,alpha_p; gamma_1,...,gamma_p; beta_1,...,beta_q]

% Define minimum omega value
MIN_OMEGA = 1e-6;

% Calculate the unconditional variance for scaling
uncond_var = var(data);

% Initialize omega (constant term)
omega = max(0.05 * uncond_var, MIN_OMEGA);

% Initialize alpha coefficients (ARCH terms)
alpha = zeros(p, 1);
if p > 0
    alpha_total = 0.1;
    for i = 1:p
        alpha(i) = alpha_total / p * (1.5^(-(i-1)));
    end
    if sum(alpha) > 0
        alpha = alpha / sum(alpha) * alpha_total;
    end
end

% Initialize gamma coefficients (threshold/asymmetric terms)
gamma = zeros(p, 1);
if p > 0
    % Calculate if negative returns have higher volatility (leverage effect)
    neg_returns = data(data < 0);
    pos_returns = data(data > 0);
    
    if ~isempty(neg_returns) && ~isempty(pos_returns) && mean(neg_returns.^2) > mean(pos_returns.^2)
        gamma_total = 0.1;  % Positive for leverage effect in TARCH
    else
        gamma_total = 0.05;  % Small value if no clear leverage
    end
    
    for i = 1:p
        gamma(i) = gamma_total / p * (1.3^(-(i-1)));
    end
end

% Initialize beta coefficients (GARCH terms)
beta = zeros(q, 1);
if q > 0
    beta_total = 0.75;
    for i = 1:q
        beta(i) = beta_total / q * (1.2^(-(i-1)));
    end
    if sum(beta) > 0
        beta = beta / sum(beta) * beta_total;
    end
end

% Combine all parameters
parameters = [omega; alpha; gamma; beta];

% Ensure stationarity: alpha + 0.5*gamma + beta < 1
if sum(alpha) + 0.5*sum(gamma) + sum(beta) >= 0.999
    scale_factor = 0.99 / (sum(alpha) + 0.5*sum(gamma) + sum(beta));
    alpha = alpha * scale_factor;
    gamma = gamma * scale_factor;
    beta = beta * scale_factor;
    parameters = [omega; alpha; gamma; beta];
end

end

function parameters = initialize_distribution_parameters(distributionType)
% Helper function that initializes parameters for error distributions
%
% INPUTS:
%   distributionType - String indicating distribution type: 'NORMAL', 'T', 'GED', or 'SKEWT'
%
% OUTPUTS:
%   parameters - Initial parameter vector for the specified distribution

switch upper(distributionType)
    case 'NORMAL'
        % Normal distribution has no additional parameters
        parameters = [];
        
    case 'T'
        % Student's t-distribution has one parameter: degrees of freedom
        % Start with moderate tail thickness (ν = 8)
        parameters = 8;
        
    case 'GED'
        % Generalized Error Distribution has one parameter: shape parameter
        % Start with shape=1.5 (between normal and Laplace)
        parameters = 1.5;
        
    case 'SKEWT'
        % Hansen's Skewed t-distribution has two parameters:
        % degrees of freedom (ν) and skewness parameter (λ)
        nu = 8;        % degrees of freedom
        lambda = 0.0;  % start with symmetric case
        parameters = [nu; lambda];
        
    otherwise
        error('Unsupported distribution type: %s', distributionType);
end

end