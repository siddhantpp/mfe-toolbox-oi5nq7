function model = gogarch(data, options)
% GOGARCH Estimates a Generalized Orthogonal GARCH (GO-GARCH) model for multivariate volatility.
%
% The GO-GARCH model transforms correlated returns into orthogonal factors using PCA,
% fits univariate GARCH models to these factors, and transforms back to obtain
% time-varying covariance matrices.
%
% USAGE:
%   MODEL = gogarch(DATA, OPTIONS)
%
% INPUTS:
%   DATA        - T by K matrix of mean zero returns/residuals
%   OPTIONS     - Structure with estimation options
%     fields:
%       'garchType'     - Type of univariate GARCH for factors:
%                         'GARCH' (default), 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'
%       'p'             - Positive integer for GARCH order (default = 1)
%       'q'             - Positive integer for ARCH order (default = 1)
%       'distribution'  - Error distribution: 'NORMAL' (default), 'T', 'GED', 'SKEWT'
%       'forecast'      - Number of periods to forecast (default = 0)
%       'method'        - Estimation method: 'pca' (default) or 'ica'
%       'useMEX'        - Boolean, whether to use MEX implementation if available (default = true)
%       'maxIterations' - Maximum iterations for optimization (default = 1000)
%       'tolerance'     - Convergence tolerance (default = 1e-6)
%       'algorithm'     - Optimization algorithm for fmincon (default = 'interior-point')
%
% OUTPUTS:
%   MODEL      - Structure with the following fields:
%     'parameters'     - Structure with estimated parameters
%     'mixingMatrix'   - K by K orthogonal transformation (mixing) matrix
%     'factorModels'   - Cell array of univariate GARCH models for each factor
%     'factors'        - T by K matrix of orthogonal factors
%     'factorVariances'- T by K matrix of factor conditional variances
%     'covariances'    - T by K by K array of conditional covariance matrices
%     'logLikelihood'  - Log-likelihood of the model
%     'aic'            - Akaike Information Criterion
%     'bic'            - Bayesian Information Criterion
%     'stats'          - Structure with goodness-of-fit statistics
%     'forecast'       - Structure with forecasted covariances (if requested)
%
% REFERENCES:
%   van der Weide, R. (2002). "GO-GARCH: A Multivariate Generalized 
%   Orthogonal GARCH Model." Journal of Applied Econometrics, 17, 549-564.
%
% See also GARCH, GARCHCORE, GARCHFOR, PCACOV

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Minimum variance level for numerical stability
MIN_VARIANCE = 1e-12;

%% Input validation
% Check data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Get data dimensions
[T, K] = size(data);

% Set default options if not provided
if nargin < 2 || isempty(options)
    options = struct();
end

% Default GARCH options for factors
if ~isfield(options, 'garchType') || isempty(options.garchType)
    options.garchType = 'GARCH';
end

% Default GARCH orders
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.p, 'options.p', opts);
end

if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.q, 'options.q', opts);
end

% Default distribution
if ~isfield(options, 'distribution') || isempty(options.distribution)
    options.distribution = 'NORMAL';
end

% Default forecast horizon
if ~isfield(options, 'forecast') || isempty(options.forecast)
    options.forecast = 0;
else
    opts = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
    parametercheck(options.forecast, 'options.forecast', opts);
end

% Default estimation method
if ~isfield(options, 'method') || isempty(options.method)
    options.method = 'pca';
end

% Default for MEX usage
if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

% Default optimization options
if ~isfield(options, 'maxIterations') || isempty(options.maxIterations)
    options.maxIterations = 1000;
end

if ~isfield(options, 'tolerance') || isempty(options.tolerance)
    options.tolerance = 1e-6;
end

if ~isfield(options, 'algorithm') || isempty(options.algorithm)
    options.algorithm = 'interior-point';
end

%% Step 1: Compute unconditional covariance matrix
unconditionalCov = cov(data);

%% Step 2: Perform orthogonal decomposition using PCA
if strcmpi(options.method, 'pca')
    % Use Principal Component Analysis
    [eigenvectors, eigenvalues] = pcacov(unconditionalCov);
    
    % Normalize the eigenvectors to ensure orthogonality
    % The mixing matrix transforms data to orthogonal factors
    mixingMatrix = eigenvectors';
    
    % Alternative normalization if needed:
    % D = diag(sqrt(eigenvalues));
    % mixingMatrix = D * eigenvectors';
elseif strcmpi(options.method, 'ica')
    % Placeholder for ICA implementation
    % This would require an ICA implementation not provided in the imports
    warning('ICA method not implemented. Using PCA instead.');
    [eigenvectors, ~] = pcacov(unconditionalCov);
    mixingMatrix = eigenvectors';
else
    error('Unknown method: %s. Supported methods are ''pca'' and ''ica''', options.method);
end

%% Step 3: Transform data to orthogonal factors
factors = data * mixingMatrix';

%% Step 4: Estimate univariate GARCH models for each factor
factorModels = cell(K, 1);
factorVariances = zeros(T, K);
stdResiduals = zeros(T, K);

% Setup univariate GARCH options
garchOptions = struct();
garchOptions.model = options.garchType;
garchOptions.p = options.p;
garchOptions.q = options.q;
garchOptions.distribution = options.distribution;
garchOptions.useMEX = options.useMEX;

% GARCH estimation options for fmincon
optimOptions = optimoptions('fmincon', 'Algorithm', options.algorithm, ...
    'MaxIterations', options.maxIterations, ...
    'TolFun', options.tolerance, ...
    'TolX', options.tolerance, ...
    'Display', 'off');

% Estimate GARCH model for each orthogonal factor
for k = 1:K
    factorData = factors(:, k);
    
    % Set up and estimate univariate GARCH model
    try
        % For simple estimation, use direct calls to garchcore
        initialParams = garch_initial_params(factorData, garchOptions);
        
        % Define objective function for likelihood maximization
        objFunction = @(params) garchlikelihood(params, factorData, garchOptions);
        
        % Set up constraints for fmincon
        A = [];
        b = [];
        Aeq = [];
        beq = [];
        lb = [];
        ub = [];
        
        % Add stationarity constraints based on model type
        constraints = @(params) garch_constraints(params, garchOptions);
        
        % Estimate parameters using fmincon
        [estParams, fval] = fmincon(objFunction, initialParams, A, b, Aeq, beq, lb, ub, constraints, optimOptions);
        
        % Compute conditional variances using estimated parameters
        htFactor = garchcore(estParams, factorData, garchOptions);
        
        % Store results
        factorModel = struct();
        factorModel.parameters = estParams;
        factorModel.garchType = options.garchType;
        factorModel.p = options.p;
        factorModel.q = options.q;
        factorModel.distribution = options.distribution;
        factorModel.ht = htFactor;
        factorModel.stdResiduals = factorData ./ sqrt(htFactor);
        factorModel.logLikelihood = -fval; % Negative because garchlikelihood returns negative logL
        
        % Add model selection criteria
        paramCount = length(estParams);
        ic = aicsbic(factorModel.logLikelihood, paramCount, T);
        factorModel.aic = ic.aic;
        factorModel.bic = ic.sbic;
        
        % Store the model and variances
        factorModels{k} = factorModel;
        factorVariances(:, k) = htFactor;
        stdResiduals(:, k) = factorModel.stdResiduals;
    catch ME
        warning('Failed to estimate GARCH model for factor %d: %s', k, ME.message);
        % Fallback to simple exponential smoothing if GARCH fails
        alpha = 0.94;
        htFactor = zeros(T, 1);
        htFactor(1) = factorData(1)^2;
        for t = 2:T
            htFactor(t) = alpha * htFactor(t-1) + (1-alpha) * factorData(t-1)^2;
        end
        
        % Create a minimal factor model
        factorModel = struct();
        factorModel.parameters = [alpha];
        factorModel.garchType = 'EWMA';
        factorModel.ht = htFactor;
        factorModel.stdResiduals = factorData ./ sqrt(htFactor);
        
        % Store the model and variances
        factorModels{k} = factorModel;
        factorVariances(:, k) = htFactor;
        stdResiduals(:, k) = factorModel.stdResiduals;
    end
end

%% Step 5: Transform factor variances to multivariate covariance matrices
covariances = transform_factors_to_covariance(factorVariances, mixingMatrix);

%% Step 6: Compute model log-likelihood and fit statistics
logLikelihood = compute_gogarch_likelihood(data, covariances, options);

% Compute total number of parameters
totalParams = sum(cellfun(@(x) length(x.parameters), factorModels)) + K*K; % Factor parameters + mixing matrix
ic = aicsbic(logLikelihood, totalParams, T);

%% Step 7: Create output structure
model = struct();

% Core parameters
model.parameters = struct();
model.parameters.garchType = options.garchType;
model.parameters.p = options.p;
model.parameters.q = options.q;
model.parameters.distribution = options.distribution;
model.parameters.method = options.method;

% Model components
model.mixingMatrix = mixingMatrix;
model.factorModels = factorModels;
model.factors = factors;
model.factorVariances = factorVariances;
model.covariances = covariances;

% Fit statistics
model.logLikelihood = logLikelihood;
model.aic = ic.aic;
model.bic = ic.sbic;
model.numParameters = totalParams;

% Validation statistics
model.stats = validate_gogarch_model(model);

%% Step 8: Generate forecasts if requested
if options.forecast > 0
    model.forecast = gogarch_forecast(model, options.forecast);
end

end

%% Helper function to transform factor variances to covariance matrices
function covMatrices = transform_factors_to_covariance(factorVariances, mixingMatrix)
% TRANSFORM_FACTORS_TO_COVARIANCE Transforms factor variances to multivariate covariance matrices
%
% INPUTS:
%   factorVariances - T by K matrix of factor conditional variances
%   mixingMatrix    - K by K orthogonal transformation matrix
%
% OUTPUTS:
%   covMatrices     - T by K by K array of conditional covariance matrices

% Get dimensions
[T, K] = size(factorVariances);

% Initialize storage for covariance matrices
covMatrices = zeros(T, K, K);

% For each time point
for t = 1:T
    % Create diagonal matrix of factor variances
    D_t = diag(factorVariances(t, :));
    
    % Transform factor variance matrix to observation covariance
    % H_t = A^(-1) * D_t * (A^(-1))'
    % Since A is orthogonal, A^(-1) = A'
    H_t = mixingMatrix' * D_t * mixingMatrix;
    
    % Ensure the covariance matrix is positive definite and symmetric
    H_t = (H_t + H_t') / 2;
    
    % Apply minimum variance threshold for numerical stability
    for i = 1:K
        if H_t(i, i) < MIN_VARIANCE
            H_t(i, i) = MIN_VARIANCE;
        end
    end
    
    % Store the covariance matrix
    covMatrices(t, :, :) = H_t;
end

end

%% Helper function to compute GO-GARCH log-likelihood
function logL = compute_gogarch_likelihood(data, covMatrices, options)
% COMPUTE_GOGARCH_LIKELIHOOD Computes log-likelihood for the GO-GARCH model
%
% INPUTS:
%   data         - T by K matrix of returns/residuals
%   covMatrices  - T by K by K array of conditional covariance matrices
%   options      - Structure with model options
%
% OUTPUTS:
%   logL         - Log-likelihood value

% Get dimensions
[T, K] = size(data);

% Initialize log-likelihood
logL = 0;

% For each time point
for t = 1:T
    % Extract the covariance matrix for this time point
    H_t = squeeze(covMatrices(t, :, :));
    
    % Extract the data point
    x_t = data(t, :)';
    
    % For numerical stability, ensure H_t is positive definite
    H_t = (H_t + H_t') / 2;
    
    % Apply minimum variance threshold
    for i = 1:K
        if H_t(i, i) < MIN_VARIANCE
            H_t(i, i) = MIN_VARIANCE;
        end
    end
    
    % Compute log determinant and inverse of H_t
    try
        logDetH = log(det(H_t));
        invH = inv(H_t);
    catch
        % If matrix is ill-conditioned, apply regularization
        H_t = H_t + MIN_VARIANCE * eye(K);
        logDetH = log(det(H_t));
        invH = inv(H_t);
    end
    
    % Compute quadratic form: x_t' * invH * x_t
    quadForm = x_t' * invH * x_t;
    
    % Compute likelihood contribution based on distribution
    if strcmpi(options.distribution, 'NORMAL')
        % Multivariate normal: -0.5 * (K*log(2π) + logDetH + quadForm)
        logL_t = -0.5 * (K * log(2*pi) + logDetH + quadForm);
    elseif strcmpi(options.distribution, 'T')
        % Would need to implement multivariate t distribution likelihood
        % For now, use normal approximation
        warning('Multivariate T distribution not fully implemented for GO-GARCH likelihood. Using normal approximation.');
        logL_t = -0.5 * (K * log(2*pi) + logDetH + quadForm);
    else
        % Default to normal for other distributions
        logL_t = -0.5 * (K * log(2*pi) + logDetH + quadForm);
    end
    
    % Accumulate log-likelihood
    logL = logL + logL_t;
end

end

%% Helper function to generate GO-GARCH forecasts
function forecast = gogarch_forecast(model, horizon)
% GOGARCH_FORECAST Generates forecasts for the GO-GARCH model
%
% INPUTS:
%   model   - GO-GARCH model structure
%   horizon - Number of periods to forecast
%
% OUTPUTS:
%   forecast - Structure with forecast results

% Get dimensions
K = size(model.mixingMatrix, 1);

% Initialize forecast arrays
factorVarForecast = zeros(horizon, K);
covMatricesForecast = zeros(horizon, K, K);

% Generate forecasts for each factor GARCH model
for k = 1:K
    factorModel = model.factorModels{k};
    
    % Create a GARCH model structure compatible with garchfor
    garchModelStruct = struct();
    garchModelStruct.parameters = factorModel.parameters;
    garchModelStruct.modelType = factorModel.garchType;
    garchModelStruct.p = factorModel.p;
    garchModelStruct.q = factorModel.q;
    garchModelStruct.data = model.factors(:, k);
    garchModelStruct.residuals = model.factors(:, k);
    garchModelStruct.ht = factorModel.ht;
    garchModelStruct.distribution = factorModel.distribution;
    
    if isfield(factorModel, 'distParams')
        garchModelStruct.distParams = factorModel.distParams;
    end
    
    % Generate forecasts for this factor
    try
        factorForecast = garchfor(garchModelStruct, horizon);
        factorVarForecast(:, k) = factorForecast.expectedVariances;
    catch ME
        warning('Error forecasting factor %d: %s. Using last variance value.', k, ME.message);
        % Fallback: use last variance value
        factorVarForecast(:, k) = repmat(factorModel.ht(end), horizon, 1);
    end
end

% Transform factor variance forecasts to covariance matrices
covMatricesForecast = transform_factors_to_covariance(factorVarForecast, model.mixingMatrix);

% Create forecast structure
forecast = struct();
forecast.factorVarianceForecasts = factorVarForecast;
forecast.covarianceForecasts = covMatricesForecast;
forecast.horizon = horizon;

end

%% Helper function to validate GO-GARCH model
function validationResults = validate_gogarch_model(model)
% VALIDATE_GOGARCH_MODEL Validates the estimated GO-GARCH model
%
% INPUTS:
%   model - GO-GARCH model structure
%
% OUTPUTS:
%   validationResults - Structure with validation statistics

% Initialize validation results
validationResults = struct();

% Check orthogonality of mixing matrix
mixingMatrixOrthogonality = model.mixingMatrix * model.mixingMatrix';
validationResults.orthogonalityError = norm(mixingMatrixOrthogonality - eye(size(mixingMatrixOrthogonality))) / numel(mixingMatrixOrthogonality);

% Check factor GARCH model stationarity
K = length(model.factorModels);
stationarity = zeros(K, 1);
persistence = zeros(K, 1);

for k = 1:K
    factorModel = model.factorModels{k};
    garchType = factorModel.garchType;
    params = factorModel.parameters;
    
    % Calculate persistence based on model type
    if strcmpi(garchType, 'GARCH')
        p = factorModel.p;
        q = factorModel.q;
        
        % For GARCH: persistence = sum(alpha) + sum(beta)
        alpha = params(2:(q+1));
        beta = params((q+2):(q+p+1));
        persist = sum(alpha) + sum(beta);
        
    elseif strcmpi(garchType, 'IGARCH')
        % IGARCH is integrated, so persistence = 1 by definition
        persist = 1;
    elseif any(strcmpi(garchType, {'GJR', 'TARCH'}))
        p = factorModel.p;
        q = factorModel.q;
        
        % For GJR/TARCH: persistence = sum(alpha) + 0.5*sum(gamma) + sum(beta)
        alpha = params(2:(q+1));
        gamma = params((q+2):(2*q+1));
        beta = params((2*q+2):(2*q+p+1));
        persist = sum(alpha) + 0.5*sum(gamma) + sum(beta);
    else
        % For other models, use a generic approximation
        persist = NaN;  % Could be computed for specific models
    end
    
    persistence(k) = persist;
    stationarity(k) = persist < 1;
end

validationResults.factorStationarity = stationarity;
validationResults.factorPersistence = persistence;

% Check positive definiteness of covariance matrices
T = size(model.covariances, 1);
pdViolations = 0;

for t = 1:T
    H_t = squeeze(model.covariances(t, :, :));
    
    % Check eigenvalues to determine positive definiteness
    try
        e = eig(H_t);
        if any(e <= 0)
            pdViolations = pdViolations + 1;
        end
    catch
        pdViolations = pdViolations + 1;
    end
end

validationResults.posDefViolations = pdViolations;
validationResults.posDefViolationRate = pdViolations / T;

% Overall model validity
validationResults.isValid = (validationResults.orthogonalityError < 0.01) && ...
                           (all(validationResults.factorStationarity)) && ...
                           (validationResults.posDefViolationRate < 0.01);

end

%% Helper function to generate initial parameters for univariate GARCH models
function params = garch_initial_params(data, garchOptions)
% Helper function to generate initial parameters for univariate GARCH models

% Compute variance for scaling
dataVar = var(data);

% Default parameters for different GARCH models
switch upper(garchOptions.model)
    case 'GARCH'
        % For GARCH(p,q): [omega, alpha_1,...,alpha_q, beta_1,...,beta_p]
        p = garchOptions.p;
        q = garchOptions.q;
        
        % Initial omega (scaled by data variance)
        omega = 0.05 * dataVar;
        
        % Initial alpha parameters
        alpha = 0.1 / q * ones(q, 1);
        
        % Initial beta parameters
        beta = 0.8 / p * ones(p, 1);
        
        % Combine parameters
        params = [omega; alpha; beta];
        
    case {'GJR', 'TARCH'}
        % For GJR/TARCH: [omega, alpha_1,...,alpha_q, gamma_1,...,gamma_q, beta_1,...,beta_p]
        p = garchOptions.p;
        q = garchOptions.q;
        
        omega = 0.05 * dataVar;
        alpha = 0.05 / q * ones(q, 1);
        gamma = 0.05 / q * ones(q, 1);  % Asymmetry parameters
        beta = 0.8 / p * ones(p, 1);
        
        params = [omega; alpha; gamma; beta];
        
    case 'EGARCH'
        % For EGARCH: [omega, alpha_1,...,alpha_q, gamma_1,...,gamma_q, beta_1,...,beta_p]
        p = garchOptions.p;
        q = garchOptions.q;
        
        omega = log(0.05 * dataVar);
        alpha = 0.1 / q * ones(q, 1);
        gamma = -0.05 / q * ones(q, 1);  % Negative for leverage effect
        beta = 0.9 / p * ones(p, 1);
        
        params = [omega; alpha; gamma; beta];
        
    case 'AGARCH'
        % For AGARCH: [omega, alpha_1,...,alpha_q, gamma, beta_1,...,beta_p]
        p = garchOptions.p;
        q = garchOptions.q;
        
        omega = 0.05 * dataVar;
        alpha = 0.1 / q * ones(q, 1);
        gamma = 0.1;  % Single asymmetry parameter
        beta = 0.8 / p * ones(p, 1);
        
        params = [omega; alpha; gamma; beta];
        
    case 'IGARCH'
        % For IGARCH: [omega, alpha_1,...,alpha_{q-1}]
        % Beta parameters are constrained by sum(alpha) + sum(beta) = 1
        q = garchOptions.q;
        
        omega = 0.01 * dataVar;
        alpha = 0.1 / q * ones(q-1, 1);  // One fewer alpha than usual
        
        params = [omega; alpha];
        
    otherwise
        % Default to GARCH(1,1)
        params = [0.05*dataVar; 0.1; 0.8];
end

% Append distribution parameters if needed
if strcmpi(garchOptions.distribution, 'T')
    params = [params; 8];  % Degrees of freedom for t
elseif strcmpi(garchOptions.distribution, 'GED')
    params = [params; 1.5];  % Shape parameter for GED
elseif strcmpi(garchOptions.distribution, 'SKEWT')
    params = [params; 8; 0];  % Degrees of freedom and skewness for skewed t
end

end

%% Helper function for GARCH parameter constraints
function [c, ceq] = garch_constraints(params, garchOptions)
% Helper function for GARCH parameter constraints
% c <= 0 are inequality constraints
% ceq = 0 are equality constraints

% Initialize constraints
c = [];
ceq = [];

% Different constraints based on model type
switch upper(garchOptions.model)
    case 'GARCH'
        p = garchOptions.p;
        q = garchOptions.q;
        
        % Extract parameters
        omega = params(1);
        alpha = params(2:(q+1));
        beta = params((q+2):(q+p+1));
        
        % Constraints:
        % 1. omega > 0
        c(1) = -omega;
        
        % 2. alpha_i >= 0
        for i = 1:q
            c(1+i) = -alpha(i);
        end
        
        % 3. beta_j >= 0
        for j = 1:p
            c(1+q+j) = -beta(j);
        end
        
        % 4. sum(alpha) + sum(beta) < 1 (stationarity)
        c(1+q+p+1) = sum(alpha) + sum(beta) - 0.9999;
        
    case {'GJR', 'TARCH'}
        p = garchOptions.p;
        q = garchOptions.q;
        
        % Extract parameters
        omega = params(1);
        alpha = params(2:(q+1));
        gamma = params((q+2):(2*q+1));
        beta = params((2*q+2):(2*q+p+1));
        
        % Constraints:
        % 1. omega > 0
        c(1) = -omega;
        
        % 2. alpha_i >= 0
        for i = 1:q
            c(1+i) = -alpha(i);
        end
        
        % 3. beta_j >= 0
        for j = 1:p
            c(1+q+j) = -beta(j);
        end
        
        % 4. alpha_i + gamma_i/2 >= 0 (non-negative news impact for negative shocks)
        for i = 1:q
            c(1+q+p+i) = -(alpha(i) + gamma(i)/2);
        end
        
        % 5. sum(alpha) + sum(gamma)/2 + sum(beta) < 1 (stationarity)
        c(1+q+p+q+1) = sum(alpha) + sum(gamma)/2 + sum(beta) - 0.9999;
        
    case 'EGARCH'
        p = garchOptions.p;
        
        % For EGARCH, the main constraint is on beta parameters
        % Extract beta parameters
        beta = params((2*q+2):(2*q+p+1));
        
        % Constraint: sum(|beta|) < 1 (stationarity)
        c(1) = sum(abs(beta)) - 0.9999;
        
    case 'IGARCH'
        q = garchOptions.q;
        
        % Extract parameters
        omega = params(1);
        alpha = params(2:q);
        
        % Constraints:
        % 1. omega > 0
        c(1) = -omega;
        
        % 2. alpha_i >= 0
        for i = 1:(q-1)
            c(1+i) = -alpha(i);
        end
        
        % 3. sum(alpha) < 1
        c(1+q) = sum(alpha) - 0.9999;
        
    otherwise
        % Default constraints for other models
        c(1) = -params(1);  % omega > 0
end

% Add constraints for distribution parameters if present
if strcmpi(garchOptions.distribution, 'T')
    % For t-distribution: nu > 2
    nu = params(end);
    c(end+1) = 2.001 - nu;
elseif strcmpi(garchOptions.distribution, 'GED')
    % For GED: nu > 0
    nu = params(end);
    c(end+1) = -nu;
elseif strcmpi(garchOptions.distribution, 'SKEWT')
    % For skewed t: nu > 2, -1 < lambda < 1
    nu = params(end-1);
    lambda = params(end);
    c(end+1) = 2.001 - nu;
    c(end+2) = abs(lambda) - 0.9999;
end

end