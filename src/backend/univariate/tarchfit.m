function parameters = tarchfit(data, options)
% TARCHFIT Fits a Threshold ARCH (TARCH) model via maximum likelihood
%
% USAGE:
%   [PARAMETERS] = tarchfit(DATA)
%   [PARAMETERS] = tarchfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA     - T by 1 vector of mean zero financial data (typically returns)
%   OPTIONS  - [OPTIONAL] Structure with fields:
%              'p'           - Positive integer for ARCH order [default = 1]
%              'q'           - Positive integer for GARCH order [default = 1]
%              'distribution' - String specifying the error distribution:
%                              'NORMAL' - Gaussian distribution (default)
%                              'T'      - Student's t-distribution
%                              'GED'    - Generalized Error Distribution
%                              'SKEWT'  - Hansen's skewed t-distribution
%              'startingvals' - Vector of starting values for parameters
%              'useMEX'       - Boolean indicating whether to use the MEX implementation
%                              [default = true]
%              'optimoptions'- Options structure for fmincon optimization
%              'backcast'    - Scalar or structure for initial variance (see backcast.m)
%              'constrainStationarity' - Boolean to enforce stationarity [default = true]
%
% OUTPUTS:
%   PARAMETERS - Structure containing:
%               .omega       - TARCH constant term
%               .alpha       - ARCH coefficient(s)
%               .gamma       - Threshold (asymmetry) coefficient(s)
%               .beta        - GARCH coefficient(s)
%               .nu          - Degree of freedom parameter (for T, GED, SKEWT)
%               .lambda      - Skewness parameter (for SKEWT only)
%               .p           - ARCH order
%               .q           - GARCH order
%               .likelihood  - Log-likelihood at optimum
%               .ht          - Conditional variances
%               .stdresid    - Standardized residuals
%               .model       - String indicating 'TARCH'
%               .distribution - String indicating error distribution
%               .aic         - Akaike Information Criterion
%               .bic         - Bayesian (Schwarz) Information Criterion
%               .vcv         - Parameter covariance matrix
%               .stderrors   - Standard errors of parameters
%               .tstat       - t-statistics of parameter estimates
%               .pvalues     - p-values for parameter significance
%               .LjungBox    - Ljung-Box test on standardized residuals
%               .LBsquared   - Ljung-Box test on squared standardized residuals
%
% COMMENTS:
%   TARCH (Threshold ARCH) models the conditional variance as:
%   h_t = omega + sum(alpha_i * e_{t-i}^2) + sum(gamma_i * e_{t-i}^2 * I[e_{t-i}<0]) + sum(beta_j * h_{t-j})
%   
%   where I[e_{t-i}<0] is an indicator function that takes value 1 when e_{t-i}<0
%   and 0 otherwise. This captures the asymmetric response of volatility to positive
%   and negative shocks (leverage effect), which is common in financial returns.
%
%   The model estimation supports four error distributions: normal, Student's t,
%   GED (Generalized Error Distribution), and Hansen's skewed t-distribution.
%
%   For computational efficiency, a MEX implementation is used when available.
%
% EXAMPLES:
%   % Standard TARCH(1,1) with normal errors
%   params = tarchfit(returns);
%
%   % TARCH(1,1) with t-distribution errors
%   options = struct('distribution', 'T');
%   params = tarchfit(returns, options);
%
%   % TARCH(2,1) with skewed t-distribution
%   options = struct('p', 2, 'distribution', 'SKEWT');
%   params = tarchfit(returns, options);
%
% See also GARCHFIT, EGARCHFIT, GARCHCORE, GARCHLIKELIHOOD, TARCH_CORE

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

%% Input validation
% Check data
data = datacheck(data, 'data');
data = columncheck(data, 'data');
T = length(data);

% Set default options
if nargin < 2
    options = [];
end

%% Process options
% Default options
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    % Validate p (must be a positive integer)
    optsValid = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.p, 'options.p', optsValid);
end

if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    % Validate q (must be a positive integer)
    optsValid = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.q, 'options.q', optsValid);
end

if ~isfield(options, 'distribution') || isempty(options.distribution)
    options.distribution = 'NORMAL';
else
    % Validate distribution (must be one of the supported types)
    validDist = {'NORMAL', 'T', 'GED', 'SKEWT'};
    options.distribution = upper(options.distribution);
    if ~ismember(options.distribution, validDist)
        error('Invalid distribution type. Must be one of: NORMAL, T, GED, SKEWT');
    end
end

if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

if ~isfield(options, 'constrainStationarity') || isempty(options.constrainStationarity)
    options.constrainStationarity = true;
end

% Set model type to TARCH for initialization and likelihood computation
options.model = 'TARCH';

% Extract parameters
p = options.p;
q = options.q;
distribution = options.distribution;

%% Parameter initialization
% Get intelligent starting values
if isfield(options, 'startingvals') && ~isempty(options.startingvals)
    startingVals = options.startingvals;
    
    % Validate starting values
    parametercheck(startingVals, 'options.startingvals');
    
    % Check parameter count
    expectedLength = 1 + p + p + q; % omega, alpha(p), gamma(p), beta(q)
    
    % Add distribution parameters if needed
    if strcmp(distribution, 'T') || strcmp(distribution, 'GED')
        expectedLength = expectedLength + 1; % Add shape parameter
    elseif strcmp(distribution, 'SKEWT')
        expectedLength = expectedLength + 2; % Add shape and skewness parameters
    end
    
    if length(startingVals) ~= expectedLength
        error(['Starting values vector must have ', num2str(expectedLength), ' elements.']);
    end
else
    % Use garchinit to get intelligent starting values
    startingVals = garchinit(data, options);
end

%% Set up parameter constraints
% Calculate number of parameters in volatility model
numVolParams = 1 + p + p + q; % omega, alpha(p), gamma(p), beta(q)

% Calculate number of distribution parameters
if strcmp(distribution, 'NORMAL')
    numDistParams = 0;
elseif strcmp(distribution, 'T') || strcmp(distribution, 'GED')
    numDistParams = 1;
elseif strcmp(distribution, 'SKEWT')
    numDistParams = 2;
end

numParams = numVolParams + numDistParams;

% Set parameter bounds
LB = zeros(numParams, 1); % All parameters >= 0
UB = inf(numParams, 1);   % No upper bound for most parameters

% TARCH stationarity constraint: sum(alpha) + 0.5*sum(gamma) + sum(beta) < 1
% Linear constraint: A*x <= b
A = zeros(1, numParams);
A(2:(1+p)) = 1;           % alpha coefficients
A((2+p):(1+p+p)) = 0.5;   % gamma coefficients (weighted by 0.5)
A((2+p+p):(1+p+p+q)) = 1; % beta coefficients
b = 0.999;                % slightly less than 1 for numerical stability

% Distribution parameter constraints
if strcmp(distribution, 'T')
    % t distribution: nu > 2 for finite variance
    LB(numVolParams+1) = 2.01;
    UB(numVolParams+1) = 100;
elseif strcmp(distribution, 'GED')
    % GED: nu > 0
    LB(numVolParams+1) = 0.1;
    UB(numVolParams+1) = 10;
elseif strcmp(distribution, 'SKEWT')
    % Skewed t: nu > 2, -1 < lambda < 1
    LB(numVolParams+1) = 2.01;
    UB(numVolParams+1) = 100;
    LB(numVolParams+2) = -0.99;
    UB(numVolParams+2) = 0.99;
end

%% Set up optimization options
% Default optimization options
optOptions = optimset('fmincon');
optOptions = optimset(optOptions, 'TolFun', 1e-6, 'TolX', 1e-6);
optOptions = optimset(optOptions, 'MaxFunEvals', 1000*numParams);
optOptions = optimset(optOptions, 'MaxIter', 1000);
optOptions = optimset(optOptions, 'Display', 'off');
optOptions = optimset(optOptions, 'Algorithm', 'interior-point');

% Override with user-provided options if available
if isfield(options, 'optimoptions') && ~isempty(options.optimoptions)
    userFields = fieldnames(options.optimoptions);
    for i = 1:length(userFields)
        optOptions = optimset(optOptions, userFields{i}, options.optimoptions.(userFields{i}));
    end
end

%% Prepare likelihood function
% Configure likelihood options
likelihoodOptions = struct('p', p, 'q', q, 'distribution', distribution, ...
                          'model', 'TARCH', 'useMEX', options.useMEX, ...
                          'constrainStationarity', options.constrainStationarity);

% Pass backcast options if provided
if isfield(options, 'backcast')
    likelihoodOptions.backcast = options.backcast;
end

% Create objective function for minimization
objFun = @(params) garchlikelihood(params, data, likelihoodOptions);

%% Perform constrained optimization
[params, fval, exitflag, output, ~, ~, hessian] = ...
    fmincon(objFun, startingVals, A, b, [], [], LB, UB, [], optOptions);

% Check for convergence issues
if exitflag <= 0
    warning('TARCH estimation failed to converge. Try different starting values or adjust optimization options.');
end

%% Extract parameters
% Extract volatility model parameters
parameters = struct();
parameters.omega = params(1);
parameters.alpha = params(2:(1+p));
parameters.gamma = params((2+p):(1+p+p));
parameters.beta = params((2+p+p):(1+p+p+q));
parameters.p = p;
parameters.q = q;

% Extract distribution parameters
if strcmp(distribution, 'T')
    parameters.nu = params(numVolParams+1);
elseif strcmp(distribution, 'GED')
    parameters.nu = params(numVolParams+1);
elseif strcmp(distribution, 'SKEWT')
    parameters.nu = params(numVolParams+1);
    parameters.lambda = params(numVolParams+2);
end

%% Compute conditional variances
% Configure options for variance computation
htOptions = struct('model', 'TARCH', 'p', p, 'q', q, 'useMEX', options.useMEX);

% Pass backcast options if provided
if isfield(options, 'backcast')
    htOptions.backcast = options.backcast;
end

% Compute variances using appropriate method
if options.useMEX && exist('tarch_core', 'file') == 3
    % Use MEX implementation if available
    backcastValue = backcast(data);
    ht = tarch_core(data, params, backcastValue, p, q, T);
else
    % Use MATLAB implementation
    ht = garchcore(params, data, htOptions);
end

%% Calculate standardized residuals and likelihood
stdresid = data ./ sqrt(ht);
likelihood = -fval; % Negative of negative log-likelihood

%% Calculate parameter standard errors
% Only calculate if Hessian is valid and well-conditioned
if all(isfinite(hessian(:))) && rcond(hessian) > sqrt(eps)
    % Compute variance-covariance matrix
    vcv = inv(hessian);
    
    % Extract standard errors
    stderrors = sqrt(diag(vcv));
    
    % Calculate t-statistics
    tstats = params ./ stderrors;
    
    % Calculate p-values (two-sided)
    pvalues = 2 * (1 - tcdf(abs(tstats), T - numParams));
else
    warning('Hessian matrix is ill-conditioned. Standard errors may be unreliable.');
    vcv = NaN(numParams);
    stderrors = NaN(numParams, 1);
    tstats = NaN(numParams, 1);
    pvalues = NaN(numParams, 1);
end

%% Compute information criteria
ic = aicsbic(likelihood, numParams, T);
aic = ic.aic;
bic = ic.sbic;

%% Perform diagnostic tests
% Ljung-Box test on standardized residuals (for autocorrelation)
maxLag = min(20, floor(T/4));
LB = ljungbox(stdresid, maxLag, p+q);

% Ljung-Box test on squared standardized residuals (for ARCH effects)
LB2 = ljungbox(stdresid.^2, maxLag, p+q);

%% Assemble parameter structure for return
% Store model specifications
parameters.model = 'TARCH';
parameters.distribution = distribution;

% Store results
parameters.likelihood = likelihood;
parameters.ht = ht;
parameters.stdresid = stdresid;
parameters.aic = aic;
parameters.bic = bic;
parameters.vcv = vcv;
parameters.stderrors = stderrors;
parameters.tstat = tstats;
parameters.pvalues = pvalues;
parameters.exitflag = exitflag;
parameters.optimization = output;
parameters.LjungBox = LB;
parameters.LBsquared = LB2;

% Calculate basic statistics of standardized residuals
parameters.residual_stats = struct();
parameters.residual_stats.mean = mean(stdresid);
parameters.residual_stats.variance = var(stdresid);
parameters.residual_stats.skewness = mean((stdresid-mean(stdresid)).^3) / var(stdresid)^(3/2);
parameters.residual_stats.kurtosis = mean((stdresid-mean(stdresid)).^4) / var(stdresid)^2;

% Fit distribution to standardized residuals
if strcmp(distribution, 'T')
    dist_fit = stdtfit(stdresid);
    parameters.dist_fit = dist_fit;
elseif strcmp(distribution, 'GED')
    dist_fit = gedfit(stdresid);
    parameters.dist_fit = dist_fit;
elseif strcmp(distribution, 'SKEWT')
    dist_fit = skewtfit(stdresid);
    parameters.dist_fit = dist_fit;
end

end