function [parameters] = egarchfit(data, p, o, q, options)
% EGARCHFIT Estimates an Exponential GARCH (EGARCH) model for volatility modeling
%
% USAGE:
%   [PARAMETERS] = egarchfit(DATA, P, O, Q)
%   [PARAMETERS] = egarchfit(DATA, P, O, Q, OPTIONS)
%
% INPUTS:
%   DATA     - A column (or row) vector of mean zero residuals (e.g., from ARMAX)
%   P        - Positive integer representing the ARCH order
%   O        - Positive integer representing the leverage order (typically equals P)
%   Q        - Positive integer representing the GARCH order
%   OPTIONS  - [OPTIONAL] Structure with fields:
%              'distribution' - String specifying the error distribution:
%                   'normal'  - Gaussian distribution with mean 0, variance 1 (default)
%                   't'       - Student's t distribution with mean 0, variance 1
%                   'ged'     - Generalized Error Distribution with mean 0, variance 1
%                   'skewt'   - Hansen's Skewed t-distribution with mean 0, variance 1
%              'startingvals' - Vector of starting values for optimization
%              'forecast'     - Number of periods to forecast
%              'constrainpersistence' - Flag to enforce stationarity (default: 1)
%              'optimoptions' - Options to pass to the optimizer (fmincon)
%              'backcast'     - Structure for variance backcasting:
%                   'type'   - 'default', 'EWMA', 'decay', or 'fixed'
%                   'value'  - For 'fixed' backcasting
%                   'lambda' - For 'EWMA' backcasting
%                   'decay'  - For 'decay' backcasting
%
% OUTPUTS:
%   PARAMETERS - Structure containing:
%      .distribution    - Error distribution name
%      .ll              - Log-likelihood at optimal parameters
%      .parameters      - Vector of estimated parameters: 
%                        [omega alpha(1:p) gamma(1:o) beta(1:q) [nu lambda]]
%      .stderrors       - Vector of standard errors
%      .tstat           - Vector of t-statistics
%      .pvalues         - Vector of p-values based on normal distribution
%      .ht              - Conditional variance series
%      .loght           - Log of the conditional variances
%      .AIC             - Akaike information criterion
%      .SBIC            - Schwarz Bayesian information criterion
%      .persistence     - Sum of beta parameters (GARCH persistence)
%      .diagnostics     - Structure with fields:
%                        .stderrors  - Source of standard errors
%                        .z          - Standardized residuals
%                        .zabs       - Absolute standardized residuals
%      .parameterCount  - Number of model parameters
%
% COMMENTS:
%   This function estimates Exponential GARCH models which capture asymmetric
%   volatility responses. The EGARCH model is specified as:
%
%   log(sigma²_t) = omega + sum_(i=1)^p [alpha_i(|z_(t-i)| - E[|z|]) + gamma_i*z_(t-i)] + sum_(j=1)^q beta_j*log(sigma²_(t-j))
%
%   where z_t = data_t/sqrt(sigma²_t)
%   
%   EGARCH models have several advantages:
%   1. Variance is guaranteed positive through log transformation
%   2. Asymmetric effects are captured by gamma parameters
%   3. No constraints on parameters are needed to ensure positivity
%
%   The function uses MEX-accelerated computations for high performance when
%   available. This implements high-speed recursive calculation of the 
%   conditional variances and their derivatives.
%
% EXAMPLES:
%   % Standard EGARCH(1,1,1) with normal errors
%   parameters = egarchfit(returns, 1, 1, 1);
%
%   % EGARCH(1,1,1) with t distribution
%   options = struct('distribution', 't');
%   parameters = egarchfit(returns, 1, 1, 1, options);
%
%   % EGARCH(2,2,1) with custom starting values and GED distribution
%   options = struct('distribution', 'ged', 'startingvals', [0.05, 0.1, 0.05, -0.08, -0.05, 0.85, 1.5]);
%   parameters = egarchfit(returns, 2, 2, 1, options);
%
% See also EGARCHSIM, GARCHFIT, TARCHFIT, APARCHFIT

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
[data, p, o, q, options] = validateInputs(data, p, o, q, nargin < 5 ? struct() : options);

% Initialize starting parameter values if not provided
if ~isfield(options, 'startingvals') || isempty(options.startingvals)
    startingValues = initializeEGARCHParameters(data, p, o, q);
else
    startingValues = options.startingvals;
end

% Setup optimization options
if isfield(options, 'optimoptions')
    optimOptions = options.optimoptions;
else
    optimOptions = optimset('fmincon');
    optimOptions = optimset(optimOptions, 'Display', 'iter', 'Diagnostics', 'on', ...
        'Algorithm', 'interior-point', 'TolFun', 1e-6, 'TolX', 1e-6, 'MaxIter', 1000);
end

% Add constraints if requested
if ~isfield(options, 'constrainpersistence') || options.constrainpersistence
    % Nonlinear constraints for stationarity
    nonlcon = @(params) computeEGARCHConstraints(params, p, o, q);
else
    nonlcon = [];
end

% Set up lower and upper bounds
% Order: omega, alpha(1:p), gamma(1:o), beta(1:q), distParams
lowerBound = [-Inf * ones(1 + p + o + q, 1); zeros(getNumDistParams(options.distribution), 1)];
upperBound = [Inf * ones(1 + p + o + q, 1); Inf * ones(getNumDistParams(options.distribution), 1)];

% Further constrain distribution parameters based on distribution type
if strcmp(options.distribution, 't')
    % nu > 2 for Student's t
    lowerBound(end) = 2.00001;
elseif strcmp(options.distribution, 'ged')
    % nu > 1 for GED
    lowerBound(end) = 1.00001;
elseif strcmp(options.distribution, 'skewt')
    % nu > 2 for skewed t
    lowerBound(end-1) = 2.00001;
    % -1 < lambda < 1 for skewed t
    lowerBound(end) = -0.9999;
    upperBound(end) = 0.9999;
end

% Get initial variance (backcast) value for recursion
if isfield(options, 'backcast')
    backcastOptions = options.backcast;
else
    backcastOptions = struct('type', 'default');
end
backcastValue = backcast(data, backcastOptions);

% Estimate parameters using maximum likelihood
[fminconParams, fval, exitflag, output, lambda, grad, hessian] = fmincon(...
    @(params) egarchLikelihood(params, data, p, o, q, options), ...
    startingValues, [], [], [], [], lowerBound, upperBound, nonlcon, optimOptions);

% Calculate standard errors using inverse Hessian
paramCount = length(fminconParams);
if exitflag > 0 && ~any(isnan(hessian(:))) && all(isfinite(hessian(:)))
    invHessian = inv(hessian);
    if all(diag(invHessian) > 0)
        stderrors = sqrt(diag(invHessian));
        SEType = 'Hessian';
    else
        % Fall back to numerical computation if Hessian is not positive definite
        [~, ~, ~, variance] = egarchLikelihood(fminconParams, data, p, o, q, options);
        scores = zeros(length(data), paramCount);
        h = 1e-4 * max(abs(fminconParams), 1);
        
        for i = 1:paramCount
            paramsPlusH = fminconParams;
            paramsPlusH(i) = paramsPlusH(i) + h(i);
            [~, ~, ~, variancePlus] = egarchLikelihood(paramsPlusH, data, p, o, q, options);
            scores(:, i) = (log(variancePlus) - log(variance)) / h(i);
        end
        
        varianceMatrix = scores' * scores;
        stderrors = sqrt(diag(inv(varianceMatrix)));
        SEType = 'Numerical';
    end
else
    % Unable to compute reliable standard errors
    stderrors = NaN(paramCount, 1);
    SEType = 'None';
end

% Compute test statistics
tstat = fminconParams ./ stderrors;
pvalues = 2 * (1 - normcdf(abs(tstat)));

% Extract final parameter values and compute diagnostics
omega = fminconParams(1);
alpha = fminconParams(2:(p+1));
gamma = fminconParams((p+2):(p+o+1));
beta = fminconParams((p+o+2):(p+o+q+1));

% Calculate model persistence (sum of beta)
persistence = sum(beta);

% Recalculate conditional variance for final parameters
[negLogL, ~, ht, loght] = egarchLikelihood(fminconParams, data, p, o, q, options);
logL = -negLogL;

% Compute information criteria
T = length(data);
ic = aicsbic(logL, paramCount, T);

% Extract distribution parameters
if strcmp(options.distribution, 'normal')
    distParams = [];
    distParamNames = {};
elseif strcmp(options.distribution, 't') || strcmp(options.distribution, 'ged')
    distParams = fminconParams(end);
    distParamNames = {'nu'};
elseif strcmp(options.distribution, 'skewt')
    distParams = fminconParams((end-1):end);
    distParamNames = {'nu', 'lambda'};
end

% Calculate standardized residuals
z = data ./ sqrt(ht);
zabs = abs(z);

% Prepare output structure
parameters = struct(...
    'distribution', options.distribution, ...
    'll', logL, ...
    'parameters', fminconParams, ...
    'stderrors', stderrors, ...
    'tstat', tstat, ...
    'pvalues', pvalues, ...
    'ht', ht, ...
    'loght', loght, ...
    'AIC', ic.aic, ...
    'SBIC', ic.sbic, ...
    'persistence', persistence, ...
    'parameterCount', paramCount, ...
    'omega', omega, ...
    'alpha', alpha, ...
    'gamma', gamma, ...
    'beta', beta, ...
    'optim', struct('exitflag', exitflag, 'output', output), ...
    'diagnostics', struct('stderrors', SEType, 'z', z, 'zabs', zabs) ...
);

% Add distribution parameters to output if applicable
if ~isempty(distParams)
    for i = 1:length(distParamNames)
        parameters.(distParamNames{i}) = distParams(i);
    end
end

end

%% Helper functions
function [data, p, o, q, options] = validateInputs(data, p, o, q, options)
% Validates input parameters for EGARCH estimation

% Validate data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Validate ARCH order (p)
options_p = struct('isPositive', true, 'isInteger', true);
p = parametercheck(p, 'p', options_p);

% Validate leverage order (o)
options_o = struct('isPositive', true, 'isInteger', true);
o = parametercheck(o, 'o', options_o);

% Validate GARCH order (q)
options_q = struct('isPositive', true, 'isInteger', true);
q = parametercheck(q, 'q', options_q);

% Set default options if not provided
if ~isfield(options, 'distribution')
    options.distribution = 'normal';
end

% Validate distribution option
validDist = {'normal', 't', 'ged', 'skewt'};
if ~ismember(lower(options.distribution), validDist)
    error('Invalid distribution option. Must be one of: normal, t, ged, skewt');
end
options.distribution = lower(options.distribution);

% Set default constraint option if not provided
if ~isfield(options, 'constrainpersistence')
    options.constrainpersistence = 1;
end

end

function [negLL, LLs, ht, loght] = egarchLikelihood(parameters, data, p, o, q, options)
% Computes negative log-likelihood for EGARCH model
% This is the objective function for optimization

% Extract parameters
omega = parameters(1);
alpha = parameters(2:(p+1));
gamma = parameters((p+2):(p+o+1));
beta = parameters((p+o+2):(p+o+q+1));

% Extract distribution parameters based on distribution type
if strcmp(options.distribution, 'normal')
    distParams = [];
elseif strcmp(options.distribution, 't') || strcmp(options.distribution, 'ged')
    distParams = parameters(end);
elseif strcmp(options.distribution, 'skewt')
    distParams = parameters((end-1):end);
end

% Get data length
T = length(data);

% Compute initial log-variance value (backcast)
if isfield(options, 'backcast')
    backcastOptions = options.backcast;
else
    backcastOptions = struct('type', 'default');
end
backcastValue = backcast(data, backcastOptions);

% Set up distribution type for MEX function
if strcmp(options.distribution, 'normal')
    distType = 1;
elseif strcmp(options.distribution, 't')
    distType = 2;
elseif strcmp(options.distribution, 'ged')
    distType = 3;
elseif strcmp(options.distribution, 'skewt')
    distType = 4;
end

% Call MEX function for high-performance computation
try
    % Check if MEX file exists in the path
    [ht, loght, ll] = egarch_core(data, parameters, p, o, q, backcastValue, 1, distType, distParams);
    LLs = ll; % MEX function computes total likelihood directly
    negLL = -ll;
catch
    % If MEX fails, fall back to MATLAB implementation
    
    % Initialize variance and log-variance arrays
    ht = zeros(T, 1);
    loght = zeros(T, 1);
    
    % Set initial values for recursion
    maxPOQ = max([p, o, q]);
    for i = 1:maxPOQ
        loght(i) = log(backcastValue);
        ht(i) = backcastValue;
    end
    
    % EGARCH recursion
    for t = (maxPOQ + 1):T
        % Log-variance equation
        loght(t) = omega;
        
        % ARCH and asymmetry terms
        for i = 1:p
            z = data(t-i) / sqrt(ht(t-i));
            % E[|z|] = sqrt(2/pi) ≈ 0.7979 for normal distribution
            loght(t) = loght(t) + alpha(i) * (abs(z) - sqrt(2/pi)) + gamma(min(i,length(gamma))) * z;
        end
        
        % GARCH terms
        for j = 1:q
            loght(t) = loght(t) + beta(j) * loght(t-j);
        end
        
        % Compute variance from log-variance
        ht(t) = exp(loght(t));
    end
    
    % Compute log-likelihood based on distribution
    LLs = zeros(T, 1);
    
    switch options.distribution
        case 'normal'
            % Standard normal log-likelihood
            LLs = -0.5 * log(2*pi) - 0.5 * log(ht) - 0.5 * (data.^2) ./ ht;
            
        case 't'
            % Student's t log-likelihood
            nu = distParams;
            const = gamma((nu+1)/2) / (gamma(nu/2) * sqrt(pi*(nu-2)));
            LLs = log(const) - 0.5*log(ht) - ((nu+1)/2) * log(1 + (data.^2) ./ (ht * (nu-2)));
            
        case 'ged'
            % GED log-likelihood (use the gedloglik function)
            for t = 1:T
                LLs(t) = gedloglik(data(t), distParams, 0, sqrt(ht(t)));
            end
            
        case 'skewt'
            % Skewed t log-likelihood (use the skewtloglik function)
            % We need to compute individual observations since skewtloglik expects a vector
            nu = distParams(1);
            lambda = distParams(2);
            for t = 1:T
                [~, logL] = skewtloglik(data(t), [nu, lambda, 0, sqrt(ht(t))]);
                LLs(t) = logL;
            end
    end
    
    % Sum log-likelihood (excluding initial observations)
    ll = sum(LLs(maxPOQ+1:end));
    negLL = -ll;
end

% Apply constraints for numerical stability
if isinf(negLL) || isnan(negLL)
    negLL = 1e8;
end

% Add penalty for non-stationarity if constrainpersistence is enabled
if isfield(options, 'constrainpersistence') && options.constrainpersistence
    persistence = sum(beta);
    if persistence >= 1
        negLL = negLL + 1e8 * (persistence - 0.999)^2;
    end
end

end

function [c, ceq] = computeEGARCHConstraints(parameters, p, o, q)
% Nonlinear constraint function for EGARCH model
% Ensures stationarity (sum of beta < 1)

% Extract beta parameters
beta = parameters((p+o+2):(p+o+q+1));

% Compute constraint: sum(beta) < 1 for stationarity
c = sum(beta) - 0.9999; % slightly less than 1 for numerical stability
ceq = [];

end

function startingValues = initializeEGARCHParameters(data, p, o, q)
% Initialize parameter values for EGARCH estimation

% Get data characteristics
datavar = var(data);
absmean = mean(abs(data));
T = length(data);

% Initialize parameter vector
% Order: omega, alpha(1:p), gamma(1:o), beta(1:q), distParams
numParams = 1 + p + o + q;

% Set initial parameters
startingValues = zeros(numParams, 1);

% Set omega (constant term) to log of unconditional variance
startingValues(1) = log(datavar) * 0.1;

% Set alpha (ARCH terms) to small positive values with decaying pattern
for i = 1:p
    startingValues(i+1) = 0.05 / i;
end

% Set gamma (leverage terms) to small negative values (typically negative for financial data)
for i = 1:o
    startingValues(p+i+1) = -0.05 / i;
end

% Set beta (GARCH terms) to ensure persistence < 1 with decaying pattern
targetPersistence = 0.85; % Target total persistence
for i = 1:q
    startingValues(p+o+i+1) = targetPersistence / q;
end

end

function numParams = getNumDistParams(distribution)
% Returns the number of parameters for a given distribution
if strcmp(distribution, 'normal')
    numParams = 0;
elseif strcmp(distribution, 't') || strcmp(distribution, 'ged')
    numParams = 1;
elseif strcmp(distribution, 'skewt')
    numParams = 2;
else
    numParams = 0;
end
end