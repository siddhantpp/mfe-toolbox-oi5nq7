function results = armaxfilter(y, x, options)
% ARMAXFILTER Estimate ARMAX model parameters with comprehensive diagnostics
%
% USAGE:
%   RESULTS = armaxfilter(Y)
%   RESULTS = armaxfilter(Y, X)
%   RESULTS = armaxfilter(Y, X, OPTIONS)
%
% INPUTS:
%   Y       - Time series data (T x 1 vector)
%   X       - [OPTIONAL] Exogenous variables (T x r matrix)
%             Default: [] (no exogenous variables)
%   OPTIONS - [OPTIONAL] Options structure with fields:
%             constant    - [OPTIONAL] Boolean indicating inclusion of constant term
%                           Default: true
%             p           - [OPTIONAL] AR order 
%                           Default: 1
%             q           - [OPTIONAL] MA order
%                           Default: 1
%             m           - [OPTIONAL] Maximum lag for diagnostics
%                           Default: min(20, T/4)
%             stdErr      - [OPTIONAL] Method for computing standard errors:
%                           'robust'  - Robust standard errors
%                           'hessian' - Hessian-based standard errors
%                           Default: 'robust'
%             startingVals - [OPTIONAL] Vector of starting values
%                           Default: [] (use reasonable defaults)
%             distribution - [OPTIONAL] Error distribution assumption:
%                           'normal'  - Normal distribution
%                           't'       - Student's t
%                           'ged'     - Generalized Error Distribution
%                           'skewt'   - Hansen's Skewed t
%                           Default: 'normal'
%             optimopts   - [OPTIONAL] Options for fminsearch optimization
%                           Default: [] (use default optimization options)
%
% OUTPUTS:
%   RESULTS - Structure containing:
%             parameters    - Estimated parameters
%             standardErrors - Standard errors of parameters
%             tStats       - t-statistics for parameter estimates
%             pValues      - p-values for parameter estimates
%             paramNames   - Names of parameters
%             residuals    - Model residuals/innovations
%             logL         - Log-likelihood of the model
%             aic          - Akaike Information Criterion
%             sbic         - Schwarz Bayesian Information Criterion
%             ljungBox     - Results of Ljung-Box test for autocorrelation
%             lmTest       - Results of LM test for autocorrelation
%             constant     - Boolean indicating if constant term was included
%             p            - AR order
%             q            - MA order
%             r            - Number of exogenous variables
%             distribution - Distribution assumption used
%             T            - Sample size
%             y            - Original time series data
%             x            - Original exogenous variables
%
% COMMENTS:
%   Implements ARMAX(p,q,r) model estimation:
%   y(t) = c + a(1)*y(t-1) + ... + a(p)*y(t-p) + 
%          b(1)*e(t-1) + ... + b(q)*e(t-q) +
%          d(1)*x(1,t) + ... + d(r)*x(r,t) + e(t)
%
%   Where e(t) can follow various distributions (normal, t, GED, skewed-t).
%   Uses MEX-optimized armaxerrors function for efficient residual computation.
%
% EXAMPLES:
%   % Simple AR(1) model with normal errors
%   results = armaxfilter(returns);
%
%   % ARMA(2,1) model with t-distributed errors
%   options = struct('p', 2, 'q', 1, 'distribution', 't');
%   results = armaxfilter(returns, [], options);
%
%   % ARMAX(1,1,2) model with GED errors and exogenous variables
%   options = struct('distribution', 'ged');
%   results = armaxfilter(returns, exog_vars, options);
%
% See also AICSBIC, LJUNGBOX, LMTEST1, ARMAXERRORS, STDTLOGLIK, GEDLOGLIK, SKEWTLOGLIK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation
if nargin < 1
    error('At least one input (time series data) is required.');
end

% Validate y data
y = datacheck(y, 'y');
y = columncheck(y, 'y');

% Get time series length
T = length(y);

% Handle optional x (exogenous variables)
if nargin < 2 || isempty(x)
    x = [];
    r = 0;
else
    x = datacheck(x, 'x');
    % If x is a column vector, convert to matrix form
    if size(x, 2) == 1
        x = x';
    end
    
    % Check x dimension compatibility with y
    if size(x, 1) ~= T
        error('Exogenous variables must have the same number of rows as the time series data.');
    end
    
    % Get number of exogenous variables
    r = size(x, 2);
end

% Set default options
defaultOptions = struct(...
    'constant', true, ...
    'p', 1, ...
    'q', 1, ...
    'm', min(20, floor(T/4)), ...
    'stdErr', 'robust', ...
    'startingVals', [], ...
    'distribution', 'normal', ...
    'optimopts', []);

% Process user options
if nargin < 3 || isempty(options)
    options = defaultOptions;
else
    % Merge user options with defaults
    optionFields = fieldnames(defaultOptions);
    for i = 1:length(optionFields)
        field = optionFields{i};
        if ~isfield(options, field)
            options.(field) = defaultOptions.(field);
        end
    end
end

% Extract parameters from options
p = options.p;
q = options.q;
constant = options.constant;
distType = options.distribution;
m = options.m;
stdErrType = options.stdErr;

% Validate model orders
options_param = struct('isInteger', true, 'isNonNegative', true, 'isscalar', true);
p = parametercheck(p, 'p', options_param);
q = parametercheck(q, 'q', options_param);

% Initialize parameter counts
numARParams = p;
numMAParams = q;
numXParams = r;
numConstParam = constant * 1;
numDistParams = 0; % Additional distribution parameters if needed

% Determine number of distribution parameters based on distribution type
switch lower(distType)
    case 'normal'
        numDistParams = 0;
    case 't'
        numDistParams = 1; % Degrees of freedom
    case 'ged'
        numDistParams = 1; % Shape parameter
    case 'skewt'
        numDistParams = 2; % Degrees of freedom and skewness
    otherwise
        error('Unsupported distribution type. Options are: ''normal'', ''t'', ''ged'', ''skewt''.');
end

% Total number of parameters
numParams = numARParams + numMAParams + numXParams + numConstParam + numDistParams;

% Initialize parameters
if isempty(options.startingVals)
    % Set default starting values
    startingVals = zeros(numParams, 1);
    
    % Set index counters
    idx = 1;
    
    % Constant term starts at sample mean if included
    if constant
        startingVals(idx) = mean(y);
        idx = idx + 1;
    end
    
    % AR parameters initialized with small positive values
    if p > 0
        startingVals(idx:idx+p-1) = 0.1 * ones(p, 1) / p;
        idx = idx + p;
    end
    
    % MA parameters initialized with small positive values
    if q > 0
        startingVals(idx:idx+q-1) = 0.1 * ones(q, 1) / q;
        idx = idx + q;
    end
    
    % Exogenous variable parameters (initialized via OLS if possible)
    if r > 0
        if p == 0 && q == 0 && constant
            % Simple OLS for exogenous variables
            beta = (x'*x)\(x'*(y - startingVals(1)));
            startingVals(idx:idx+r-1) = beta;
        else
            % Default initialization
            startingVals(idx:idx+r-1) = zeros(r, 1);
        end
        idx = idx + r;
    end
    
    % Distribution parameters
    switch lower(distType)
        case 'normal'
            % No additional parameters
        case 't'
            startingVals(idx) = 8; % Initial df = 8
        case 'ged'
            startingVals(idx) = 1.5; % Initial shape = 1.5
        case 'skewt'
            startingVals(idx) = 8; % Initial df = 8
            startingVals(idx+1) = 0; % Initial skewness = 0
    end
else
    % Use user-provided starting values
    startingVals = options.startingVals;
    if length(startingVals) ~= numParams
        error('Starting values vector must have length equal to number of parameters (%d).', numParams);
    end
end

% Configure optimization options
if isempty(options.optimopts)
    optimOpts = optimset('fminsearch');
    optimOpts.Display = 'off';
    optimOpts.MaxIter = 1000;
    optimOpts.TolFun = 1e-6;
    optimOpts.TolX = 1e-6;
else
    optimOpts = options.optimopts;
end

% Define function handles for optimization
objectiveFcn = @(params) armax_likelihood(params);

% Perform optimization using fminsearch
[optimalParams, optimalLL] = fminsearch(objectiveFcn, startingVals, optimOpts);

% Get residuals using optimized parameters
[~, errors] = armax_likelihood(optimalParams);

% Calculate standard errors of parameter estimates
stdErrors = compute_parameter_standard_errors(optimalParams, errors, strcmp(stdErrType, 'robust'));

% Create parameter names
paramNames = cell(numParams, 1);
idx = 1;

if constant
    paramNames{idx} = 'Constant';
    idx = idx + 1;
end

for i = 1:p
    paramNames{idx} = sprintf('AR(%d)', i);
    idx = idx + 1;
end

for i = 1:q
    paramNames{idx} = sprintf('MA(%d)', i);
    idx = idx + 1;
end

for i = 1:r
    paramNames{idx} = sprintf('Exog(%d)', i);
    idx = idx + 1;
end

switch lower(distType)
    case 'normal'
        % No additional parameters
    case 't'
        paramNames{idx} = 'DoF';
    case 'ged'
        paramNames{idx} = 'Shape';
    case 'skewt'
        paramNames{idx} = 'DoF';
        paramNames{idx+1} = 'Skew';
end

% Calculate t-statistics
tStats = optimalParams ./ stdErrors;

% Calculate p-values (using two-sided test)
pValues = 2 * (1 - tcdf(abs(tStats), T - numParams));

% Calculate information criteria
ic = aicsbic(-optimalLL, numParams, T);

% Perform diagnostic tests
ljungBoxResult = ljungbox(errors, m, p + q);
lmTestResult = lmtest1(errors, m);

% Assemble results structure
results = struct();
results.parameters = optimalParams;
results.standardErrors = stdErrors;
results.tStats = tStats;
results.pValues = pValues;
results.paramNames = paramNames;
results.residuals = errors;
results.logL = -optimalLL;
results.aic = ic.aic;
results.sbic = ic.sbic;
results.ljungBox = ljungBoxResult;
results.lmTest = lmTestResult;
results.constant = constant;
results.p = p;
results.q = q;
results.r = r;
results.distribution = distType;
results.T = T;
results.y = y;
results.x = x;
results.options = options;

% Nested function to compute negative log-likelihood
function [nll, errors] = armax_likelihood(params)
    % Extract parameters
    idx = 1;
    
    % Extract constant term if included
    if constant
        const = params(idx);
        idx = idx + 1;
    else
        const = 0;
    end
    
    % Extract AR parameters
    ar_params = zeros(p, 1);
    if p > 0
        ar_params = params(idx:idx+p-1);
        idx = idx + p;
    end
    
    % Extract MA parameters
    ma_params = zeros(q, 1);
    if q > 0
        ma_params = params(idx:idx+q-1);
        idx = idx + q;
    end
    
    % Extract exogenous variable parameters
    x_params = [];
    if r > 0
        x_params = params(idx:idx+r-1);
        idx = idx + r;
    end
    
    % Extract distribution parameters
    dist_params = [];
    switch lower(distType)
        case 'normal'
            % No additional parameters
        case 't'
            dist_params = params(idx);
            % Ensure degrees of freedom > 2 for finite variance
            if dist_params < 2.001
                nll = 1e10; % Return large value for invalid parameters
                errors = zeros(T, 1);
                return;
            end
        case 'ged'
            dist_params = params(idx);
            % Ensure shape parameter > 0
            if dist_params <= 0
                nll = 1e10; % Return large value for invalid parameters
                errors = zeros(T, 1);
                return;
            end
        case 'skewt'
            dist_params = params(idx:idx+1);
            % Ensure degrees of freedom > 2 and skewness in (-1,1)
            if dist_params(1) < 2.001 || abs(dist_params(2)) >= 1
                nll = 1e10; % Return large value for invalid parameters
                errors = zeros(T, 1);
                return;
            end
    end
    
    % Compute model residuals using MEX-optimized function
    % Call MEX implementation: armaxerrors(y, ar_params, ma_params, x, x_params, const)
    if r > 0 && ~isempty(x_params)
        errors = armaxerrors(y, ar_params, ma_params, x, x_params, const);
    else
        errors = armaxerrors(y, ar_params, ma_params, [], [], const);
    end
    
    % Compute log-likelihood based on distribution type
    sigma = std(errors);
    
    % Calculate likelihood based on error distribution
    switch lower(distType)
        case 'normal'
            % Standard normal likelihood
            ll = -0.5 * T * log(2*pi) - T * log(sigma) - sum((errors).^2) / (2 * sigma^2);
            nll = -ll;
        case 't'
            % Student's t likelihood
            nu = dist_params(1);
            nll = stdtloglik(errors, nu, 0, sigma);
        case 'ged'
            % GED likelihood
            nu = dist_params(1);
            nll = -gedloglik(errors, nu, 0, sigma);
        case 'skewt'
            % Hansen's skewed t likelihood
            nu = dist_params(1);
            lambda = dist_params(2);
            params_skewt = [nu, lambda, 0, sigma];
            nll = skewtloglik(errors, params_skewt);
    end
    
    % Handle numerical instability
    if ~isfinite(nll)
        nll = 1e10; % Large value for invalid likelihood
    end
end

% Nested function to compute parameter standard errors
function stdErrors = compute_parameter_standard_errors(params, residuals, robust)
    % Total number of parameters
    numParams = length(params);
    
    % Small adjustment for numerical derivatives
    h = 1e-5;
    
    % Initialize Hessian approximation matrix
    H = zeros(numParams, numParams);
    
    % Compute numerical Hessian via finite differences
    for i = 1:numParams
        for j = i:numParams  % Only compute upper triangle due to symmetry
            if i == j
                % Diagonal elements (second derivatives)
                paramsPlus = params;
                paramsMinus = params;
                paramsPlus(i) = params(i) + h;
                paramsMinus(i) = params(i) - h;
                
                llPlus = armax_likelihood(paramsPlus);
                llMinus = armax_likelihood(paramsMinus);
                llCenter = armax_likelihood(params);
                
                H(i,j) = (llPlus - 2*llCenter + llMinus) / (h^2);
            else
                % Off-diagonal elements (mixed partial derivatives)
                paramsPluspPlus = params;
                paramsPlusMinus = params;
                paramsMinusPlus = params;
                paramsMinusMinus = params;
                
                paramsPluspPlus(i) = params(i) + h;
                paramsPluspPlus(j) = params(j) + h;
                
                paramsPlusMinus(i) = params(i) + h;
                paramsPlusMinus(j) = params(j) - h;
                
                paramsMinusPlus(i) = params(i) - h;
                paramsMinusPlus(j) = params(j) + h;
                
                paramsMinusMinus(i) = params(i) - h;
                paramsMinusMinus(j) = params(j) - h;
                
                llPluspPlus = armax_likelihood(paramsPluspPlus);
                llPlusMinus = armax_likelihood(paramsPlusMinus);
                llMinusPlus = armax_likelihood(paramsMinusPlus);
                llMinusMinus = armax_likelihood(paramsMinusMinus);
                
                H(i,j) = (llPluspPlus - llPlusMinus - llMinusPlus + llMinusMinus) / (4 * h^2);
                H(j,i) = H(i,j);  % Ensure symmetry
            end
        end
    end
    
    % Ensure the Hessian is symmetric (numerical stability)
    H = (H + H') / 2;
    
    % Compute the covariance matrix from the Hessian
    try
        % Attempt to invert the Hessian
        V = inv(H);
    catch
        % If Hessian is singular, try regularization
        V = inv(H + eye(numParams) * 1e-8);
    end
    
    % If robust standard errors are requested, apply White's heteroskedasticity correction
    if robust
        % Compute scores for each observation
        scores = zeros(T, numParams);
        
        for i = 1:numParams
            params_plus = params;
            params_minus = params;
            params_plus(i) = params(i) + h;
            params_minus(i) = params(i) - h;
            
            [~, e_plus] = armax_likelihood(params_plus);
            [~, e_minus] = armax_likelihood(params_minus);
            
            % Compute score contribution for each observation
            score_i = (e_plus.^2 - e_minus.^2) / (2*h);
            scores(:, i) = score_i;
        end
        
        % Compute robust covariance matrix (sandwich estimator)
        S = scores' * scores;
        V = inv(H) * S * inv(H);
    end
    
    % Extract standard errors from the diagonal of the covariance matrix
    stdErrors = sqrt(diag(V));
    
    % Apply degree-of-freedom correction for small sample
    stdErrors = stdErrors * sqrt(T / (T - numParams));
end

end