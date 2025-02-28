% TESTDATAGENERATOR Data generation script for MFE Toolbox test datasets
%
% This script creates various test datasets with known statistical properties
% for validating the MFE Toolbox functionality. It generates controlled data samples
% for testing statistical distributions, time series models, volatility models,
% high-frequency data analysis, and cross-sectional methods.
%
% The generated datasets have known properties (parameters, moments, etc.) that
% can be used to verify the correctness of estimation and analysis methods in the
% MFE Toolbox.
%
% USAGE:
%   TestDataGenerator                   % Run this script to display help
%   generateAllTestData()               % Generate all standard test datasets
%   generateAllTestData(true)           % Regenerate all datasets, overwriting existing files
%   data = generateFinancialReturns(1000, 5) % Generate specific test data
%
% FUNCTIONS:
%   generateFinancialReturns    - Synthetic financial return series
%   generateVolatilitySeries    - GARCH volatility series with known parameters
%   generateDistributionSamples - Random samples from various distributions
%   generateHighFrequencyData   - Synthetic high-frequency financial data
%   generateCrossSectionalData  - Cross-sectional data with factor structure
%   generateMacroeconomicData   - Synthetic macroeconomic time series
%   generateSimulatedData       - Generic simulated data with specified properties
%   generateAllTestData         - Generate all standard test datasets
%   saveTestData                - Save generated test data to MAT files
%
% EXAMPLES:
%   % Generate and save all test datasets
%   generateAllTestData();
%
%   % Generate a specific dataset
%   returns = generateFinancialReturns(1000, 3);
%
%   % Generate volatility series with GARCH(1,1)
%   volData = generateVolatilitySeries(1000, 'GARCH', struct('p',1, 'q',1));
%
%   % Save custom dataset
%   saveTestData(returns, 'my_test_data.mat');
%
% See also GARCH, APARCH, EGARCH, ARMAFOR, ARMAXFILTER

% Define global constants
global DATA_PATH RANDOM_SEED
DATA_PATH = '../data/';
RANDOM_SEED = '20090101';

% Ensure test data directory exists
if ~exist(DATA_PATH, 'dir')
    mkdir(DATA_PATH);
end

% If this script is run directly, display help information
if ~nargout
    help TestDataGenerator
end

function returns = generateFinancialReturns(numObservations, numSeries, parameters)
% GENERATEFINANCIALRETURNS Generates synthetic financial return series with specified statistical properties
%
% USAGE:
%   RETURNS = generateFinancialReturns(NUMOBSERVATIONS, NUMSERIES)
%   RETURNS = generateFinancialReturns(NUMOBSERVATIONS, NUMSERIES, PARAMETERS)
%
% INPUTS:
%   NUMOBSERVATIONS - Number of observations (time periods) to generate
%   NUMSERIES       - Number of return series to generate
%   PARAMETERS      - [OPTIONAL] Structure with fields:
%                     .distribution - String: 'normal', 't', 'ged', 'skewt' (default: 'normal')
%                     .distParams   - Parameters for the distribution:
%                                     't': degrees of freedom (default: 5)
%                                     'ged': shape parameter (default: 1.5)
%                                     'skewt': [nu lambda] (default: [5 -0.2])
%                     .arCoeffs     - AR coefficients (default: [])
%                     .maCoeffs     - MA coefficients (default: [])
%                     .garchParams  - GARCH parameters [omega alpha beta] (default: [0.01 0.1 0.85])
%                     .correlation  - Target correlation matrix (default: eye(numSeries))
%                     .mean         - Mean returns (default: zeros(numSeries,1))
%                     .scale        - Volatility scaling (default: ones(numSeries,1))
%
% OUTPUT:
%   RETURNS - numObservations × numSeries matrix of simulated financial returns
%
% COMMENTS:
%   This function generates financial return series with realistic statistical properties
%   including distributional characteristics (fat tails, skewness), autocorrelation,
%   volatility clustering, and cross-sectional correlation.
%
% EXAMPLES:
%   % Generate 1000 observations of 3 uncorrelated normal return series
%   returns = generateFinancialReturns(1000, 3);
%
%   % Generate returns with t-distributed innovations
%   params = struct('distribution', 't', 'distParams', 4);
%   returns = generateFinancialReturns(1000, 3, params);
%
%   % Generate returns with GARCH(1,1) volatility and AR(1) dynamics
%   params = struct('arCoeffs', 0.2, 'garchParams', [0.01 0.1 0.8]);
%   returns = generateFinancialReturns(1000, 2, params);

global RANDOM_SEED;

% Validate input parameters
options.isInteger = true;
options.isPositive = true;
numObservations = parametercheck(numObservations, 'numObservations', options);
numSeries = parametercheck(numSeries, 'numSeries', options);

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set default distribution if not provided
if ~isfield(parameters, 'distribution')
    parameters.distribution = 'normal';
end

% Process distribution parameters
distType = lower(parameters.distribution);
switch distType
    case 'normal'
        % Normal distribution (no additional parameters needed)
        distParams = [];
    case 't'
        % Student's t distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 5; % Default degrees of freedom
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'ged'
        % GED distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 1.5; % Default shape parameter
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'skewt'
        % Skewed t distribution
        if ~isfield(parameters, 'distParams') || length(parameters.distParams) < 2
            parameters.distParams = [5, -0.2]; % Default: DoF=5, lambda=-0.2 (negative skew)
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams(1), 'distParams(1)', options); % nu > 0
        if abs(distParams(2)) >= 1
            error('Skewness parameter lambda must be in range (-1,1)');
        end
    otherwise
        error('Unsupported distribution type: %s', distType);
end

% Set default time series parameters
if ~isfield(parameters, 'arCoeffs')
    parameters.arCoeffs = []; % Default: no AR component
end

if ~isfield(parameters, 'maCoeffs')
    parameters.maCoeffs = []; % Default: no MA component
end

if ~isfield(parameters, 'garchParams')
    % Default GARCH(1,1) parameters: omega=0.01, alpha=0.1, beta=0.85
    parameters.garchParams = [0.01, 0.1, 0.85];
end

% Process correlation structure
if ~isfield(parameters, 'correlation')
    % Default: identity matrix (uncorrelated series)
    parameters.correlation = eye(numSeries);
else
    % Validate correlation matrix
    corrMatrix = parameters.correlation;
    if size(corrMatrix, 1) ~= numSeries || size(corrMatrix, 2) ~= numSeries
        error('Correlation matrix must be numSeries × numSeries');
    end
    % Check if valid correlation matrix (symmetric, ones on diagonal, values in [-1,1])
    if ~isequal(corrMatrix, corrMatrix')
        error('Correlation matrix must be symmetric');
    end
    if any(diag(corrMatrix) ~= 1)
        error('Correlation matrix must have ones on the diagonal');
    end
    if any(abs(corrMatrix(:)) > 1)
        error('Correlation matrix must have values in range [-1,1]');
    end
    % Ensure positive semidefinite
    [~, p] = chol(corrMatrix);
    if p > 0
        warning('Correlation matrix is not positive semidefinite. Adjusting to nearest valid matrix.');
        % Adjust to nearest positive semidefinite matrix
        [V, D] = eig(corrMatrix);
        D = max(0, D); % Force eigenvalues to be non-negative
        corrMatrix = V * D * V';
        % Rescale to ensure ones on diagonal
        d = sqrt(diag(corrMatrix));
        corrMatrix = corrMatrix ./ (d * d');
        parameters.correlation = corrMatrix;
    end
end

% Process mean and scaling
if ~isfield(parameters, 'mean')
    parameters.mean = zeros(numSeries, 1); % Default: zero mean
else
    if length(parameters.mean) ~= numSeries
        error('Mean vector must have length equal to numSeries');
    end
    parameters.mean = parameters.mean(:); % Ensure column vector
end

if ~isfield(parameters, 'scale')
    parameters.scale = ones(numSeries, 1); % Default: unit scale
else
    if length(parameters.scale) ~= numSeries
        error('Scale vector must have length equal to numSeries');
    end
    options.isPositive = true;
    parametercheck(parameters.scale, 'parameters.scale', options);
    parameters.scale = parameters.scale(:); % Ensure column vector
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Step 1: Generate base innovations
innovations = generateBaseInnovations(numObservations, numSeries, distType, distParams);

% Step 2: Apply correlation structure
% Use Cholesky decomposition to induce correlation
C = chol(parameters.correlation, 'lower');
innovations = innovations * C';

% Step 3: Generate time series with specified AR/MA components
returns = zeros(numObservations, numSeries);
for i = 1:numSeries
    seriesInnovations = innovations(:, i);
    
    % Generate base returns (without volatility clustering)
    if ~isempty(parameters.arCoeffs) || ~isempty(parameters.maCoeffs)
        % Use ARMA process if AR or MA coefficients specified
        p = length(parameters.arCoeffs);
        q = length(parameters.maCoeffs);
        
        % Need to simulate with some initial values
        burnin = max(100, 10*(p+q));
        arma_params = [parameters.arCoeffs(:); parameters.maCoeffs(:)];
        
        % Generate ARMA process using armafor function
        includeConstant = false; % No constant term in test data
        [baseReturns, ~] = armafor(arma_params, [zeros(p+q, 1); seriesInnovations(1:burnin)], ...
            p, q, includeConstant, [], numObservations+burnin);
        
        % Keep only the requested number of observations
        baseReturns = baseReturns(end-numObservations+1:end);
    else
        % If no ARMA components, base returns are the innovations
        baseReturns = seriesInnovations;
    end
    
    % Apply GARCH volatility if specified
    if ~isempty(parameters.garchParams)
        garchParams = parameters.garchParams;
        
        % Create GARCH model structure
        p = 1; % Default GARCH order
        q = 1; % Default ARCH order
        if length(garchParams) >= 3
            omega = garchParams(1);
            alpha = garchParams(2);
            beta = garchParams(3);
        else
            % Use default values if incomplete
            omega = 0.01;
            alpha = 0.1;
            beta = 0.85;
        end
        
        % Create residuals with GARCH volatility
        burnin = 500; % Burn-in period to stabilize volatility
        
        % Initialize variance series
        initialVariance = omega / (1 - alpha - beta); % Unconditional variance
        
        % Generate extended series with burn-in
        extendedBaseReturns = [randn(burnin, 1); baseReturns];
        ht = zeros(burnin + numObservations, 1);
        ht(1) = initialVariance;
        
        % GARCH recursion
        for t = 2:length(ht)
            ht(t) = omega + alpha * extendedBaseReturns(t-1)^2 + beta * ht(t-1);
        end
        
        % Apply volatility to returns
        volReturns = zeros(size(extendedBaseReturns));
        volReturns = extendedBaseReturns .* sqrt(ht);
        
        % Remove burn-in period
        garchReturns = volReturns(burnin+1:end);
    else
        % If no GARCH, use base returns
        garchReturns = baseReturns;
    end
    
    % Scale and add mean
    returns(:, i) = parameters.mean(i) + parameters.scale(i) * garchReturns;
end

% Return the generated financial returns
end

function innovations = generateBaseInnovations(numObservations, numSeries, distType, distParams)
% Helper function to generate random innovations from specified distribution

switch distType
    case 'normal'
        % Standard normal innovations
        innovations = randn(numObservations, numSeries);
        
    case 't'
        % Standardized Student's t innovations
        nu = distParams;
        innovations = stdtrnd([numObservations, numSeries], nu);
        
    case 'ged'
        % Generalized Error Distribution innovations
        nu = distParams;
        innovations = zeros(numObservations, numSeries);
        for i = 1:numSeries
            innovations(:, i) = gedrnd(nu, numObservations, 1);
        end
        
    case 'skewt'
        % Hansen's skewed t innovations
        nu = distParams(1);
        lambda = distParams(2);
        innovations = zeros(numObservations, numSeries);
        for i = 1:numSeries
            innovations(:, i) = skewtrnd(nu, lambda, numObservations, 1);
        end
end
end

function volData = generateVolatilitySeries(numObservations, modelType, parameters)
% GENERATEVOLATILITYSERIES Generates synthetic volatility series with known GARCH parameters
%
% USAGE:
%   VOLDATA = generateVolatilitySeries(NUMOBSERVATIONS, MODELTYPE)
%   VOLDATA = generateVolatilitySeries(NUMOBSERVATIONS, MODELTYPE, PARAMETERS)
%
% INPUTS:
%   NUMOBSERVATIONS - Number of observations to generate
%   MODELTYPE       - String specifying the volatility model type:
%                     'GARCH', 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'
%   PARAMETERS      - [OPTIONAL] Structure with fields:
%                     .p          - GARCH order (default: 1)
%                     .q          - ARCH order (default: 1)
%                     .omega      - Constant term (default: model-specific)
%                     .alpha      - ARCH parameter(s) (default: model-specific)
%                     .beta       - GARCH parameter(s) (default: model-specific)
%                     .gamma      - Asymmetry parameter(s) (default: model-specific)
%                     .distribution - Innovation distribution: 'normal', 't', 'ged', 'skewt'
%                                     (default: 'normal')
%                     .distParams   - Distribution parameters (default: distribution-specific)
%
% OUTPUT:
%   VOLDATA - Structure containing:
%             .returns - Generated return series
%             .ht      - True conditional variances
%             .parameters - True model parameters
%             .model   - Model specification
%
% COMMENTS:
%   This function generates synthetic return series with volatility clustering according
%   to a specified GARCH-family model. Both the returns and the true conditional
%   variances are provided, which is useful for testing volatility model estimation.
%
% EXAMPLES:
%   % Generate GARCH(1,1) volatility series
%   volData = generateVolatilitySeries(1000, 'GARCH');
%
%   % Generate GJR-GARCH with Student's t innovations
%   params = struct('p', 1, 'q', 1, 'distribution', 't', 'distParams', 5);
%   volData = generateVolatilitySeries(1000, 'GJR', params);
%
%   % Generate EGARCH with custom parameters
%   params = struct('omega', -0.1, 'alpha', 0.15, 'beta', 0.96, 'gamma', -0.08);
%   volData = generateVolatilitySeries(2000, 'EGARCH', params);

global RANDOM_SEED;

% Validate input parameters
options.isInteger = true;
options.isPositive = true;
numObservations = parametercheck(numObservations, 'numObservations', options);

% Validate model type
modelType = upper(modelType);
validModels = {'GARCH', 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'};
if ~ismember(modelType, validModels)
    error('Invalid model type. Supported models: %s', strjoin(validModels, ', '));
end

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set model orders
if ~isfield(parameters, 'p')
    parameters.p = 1; % Default GARCH order
else
    options.isInteger = true;
    options.isPositive = true;
    parameters.p = parametercheck(parameters.p, 'p', options);
end

if ~isfield(parameters, 'q')
    parameters.q = 1; % Default ARCH order
else
    options.isInteger = true;
    options.isPositive = true;
    parameters.q = parametercheck(parameters.q, 'q', options);
end

% Set distribution type and parameters
if ~isfield(parameters, 'distribution')
    parameters.distribution = 'normal';
end
distType = lower(parameters.distribution);

% Process distribution parameters
switch distType
    case 'normal'
        % Normal distribution (no additional parameters needed)
        distParams = [];
    case 't'
        % Student's t distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 5; % Default degrees of freedom
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'ged'
        % GED distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 1.5; % Default shape parameter
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'skewt'
        % Skewed t distribution
        if ~isfield(parameters, 'distParams') || length(parameters.distParams) < 2
            parameters.distParams = [5, -0.2]; % Default: DoF=5, lambda=-0.2 (negative skew)
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams(1), 'distParams(1)', options); % nu > 0
        if abs(distParams(2)) >= 1
            error('Skewness parameter lambda must be in range (-1,1)');
        end
    otherwise
        error('Unsupported distribution type: %s', distType);
end

% Set default model parameters based on model type
p = parameters.p;
q = parameters.q;

switch modelType
    case 'GARCH'
        % Standard GARCH(p,q) model
        if ~isfield(parameters, 'omega')
            parameters.omega = 0.01; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.1 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'beta')
            parameters.beta = 0.85 * ones(p, 1); % Default GARCH parameter(s)
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        end
        
        % Check stationarity constraint: sum(alpha) + sum(beta) < 1
        if sum(parameters.alpha) + sum(parameters.beta) >= 1
            warning('Sum of alpha and beta is >= 1. Model may not be stationary.');
        end
        
        % Prepare parameter vector [omega, alpha(1:q), beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.beta(:)];
        
    case {'GJR', 'TARCH'}
        % GJR/TARCH model with asymmetric effects
        if ~isfield(parameters, 'omega')
            parameters.omega = 0.01; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.05 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'gamma')
            parameters.gamma = 0.1 * ones(q, 1); % Default asymmetry parameter(s)
        elseif length(parameters.gamma) ~= q
            error('Length of gamma must match q');
        end
        
        if ~isfield(parameters, 'beta')
            parameters.beta = 0.8 * ones(p, 1); % Default GARCH parameter(s)
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        end
        
        % Check stationarity constraint: sum(alpha + gamma/2) + sum(beta) < 1
        if sum(parameters.alpha) + sum(parameters.gamma)/2 + sum(parameters.beta) >= 1
            warning('Sum of alpha + gamma/2 and beta is >= 1. Model may not be stationary.');
        end
        
        % Prepare parameter vector [omega, alpha(1:q), gamma(1:q), beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.gamma(:); parameters.beta(:)];
        
    case 'EGARCH'
        % Exponential GARCH model
        if ~isfield(parameters, 'omega')
            parameters.omega = -0.1; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.15 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'gamma')
            parameters.gamma = -0.1 * ones(q, 1); % Default asymmetry parameter(s)
        elseif length(parameters.gamma) ~= q
            error('Length of gamma must match q');
        end
        
        if ~isfield(parameters, 'beta')
            parameters.beta = 0.95 * ones(p, 1); % Default GARCH parameter(s)
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        end
        
        % Check stationarity constraint: sum(beta) < 1
        if sum(parameters.beta) >= 1
            warning('Sum of beta is >= 1. Model may not be stationary.');
        end
        
        % Prepare parameter vector [omega, alpha(1:q), gamma(1:q), beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.gamma(:); parameters.beta(:)];
        
    case 'AGARCH'
        % Asymmetric GARCH model
        if ~isfield(parameters, 'omega')
            parameters.omega = 0.01; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.1 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'gamma')
            parameters.gamma = 0.05; % Default asymmetry parameter (scalar)
        end
        
        if ~isfield(parameters, 'beta')
            parameters.beta = 0.85 * ones(p, 1); % Default GARCH parameter(s)
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        end
        
        % Prepare parameter vector [omega, alpha(1:q), gamma, beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.gamma; parameters.beta(:)];
        
    case 'IGARCH'
        % Integrated GARCH model (sum(alpha) + sum(beta) = 1)
        if ~isfield(parameters, 'omega')
            parameters.omega = 0.01; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.15 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'beta')
            % In IGARCH, sum(alpha) + sum(beta) = 1
            alphaPart = sum(parameters.alpha);
            if alphaPart >= 1
                warning('Sum of alpha is >= 1. Adjusting to enforce IGARCH constraint.');
                parameters.alpha = parameters.alpha / (alphaPart * 1.1); % Scale down alpha
                alphaPart = sum(parameters.alpha);
            end
            
            betaSum = 1 - alphaPart;
            parameters.beta = betaSum * ones(p, 1) / p; % Distribute evenly among betas
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        else
            % Adjust beta to ensure IGARCH constraint
            betaSum = 1 - sum(parameters.alpha);
            if betaSum <= 0
                warning('Sum of alpha is >= 1. Adjusting to enforce IGARCH constraint.');
                parameters.alpha = parameters.alpha * 0.9; % Scale down alpha
                betaSum = 1 - sum(parameters.alpha);
            end
            betaRatio = sum(parameters.beta);
            if betaRatio > 0
                parameters.beta = parameters.beta * (betaSum / betaRatio);
            else
                parameters.beta = betaSum * ones(p, 1) / p;
            end
        end
        
        % Prepare parameter vector [omega, alpha(1:q), beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.beta(:)];
        
    case 'NAGARCH'
        % Nonlinear Asymmetric GARCH model
        if ~isfield(parameters, 'omega')
            parameters.omega = 0.01; % Default constant term
        end
        
        if ~isfield(parameters, 'alpha')
            parameters.alpha = 0.05 * ones(q, 1); % Default ARCH parameter(s)
        elseif length(parameters.alpha) ~= q
            error('Length of alpha must match q');
        end
        
        if ~isfield(parameters, 'gamma')
            parameters.gamma = 0.5; % Default asymmetry parameter (scalar)
        end
        
        if ~isfield(parameters, 'beta')
            parameters.beta = 0.9 * ones(p, 1); % Default GARCH parameter(s)
        elseif length(parameters.beta) ~= p
            error('Length of beta must match p');
        end
        
        % Prepare parameter vector [omega, alpha(1:q), gamma, beta(1:p)]
        paramVector = [parameters.omega; parameters.alpha(:); parameters.gamma; parameters.beta(:)];
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Create model structure for garchfor function
garchModel = struct();
garchModel.parameters = paramVector;
garchModel.modelType = modelType;
garchModel.p = p;
garchModel.q = q;
garchModel.distribution = parameters.distribution;
if isfield(parameters, 'distParams')
    garchModel.distParams = parameters.distParams;
end

% Generate burnin period to stabilize the volatility process
burnin = 1000;
totalObs = burnin + numObservations;

% Generate innovations based on the specified distribution
switch distType
    case 'normal'
        % Standard normal innovations
        innovations = randn(totalObs, 1);
    case 't'
        % Standardized Student's t innovations
        innovations = stdtrnd([totalObs, 1], distParams);
    case 'ged'
        % Generalized Error Distribution innovations
        innovations = gedrnd(distParams, totalObs, 1);
    case 'skewt'
        % Hansen's skewed t innovations
        nu = distParams(1);
        lambda = distParams(2);
        innovations = skewtrnd(nu, lambda, totalObs, 1);
end

% Generate the volatility series based on the model
ht = zeros(totalObs, 1);

% Set initial variance
if strcmp(modelType, 'GARCH') || strcmp(modelType, 'IGARCH')
    omega = parameters.omega;
    alpha_sum = sum(parameters.alpha);
    beta_sum = sum(parameters.beta);
    
    if strcmp(modelType, 'GARCH') && (alpha_sum + beta_sum < 1)
        unconditional_variance = omega / (1 - alpha_sum - beta_sum);
    else
        % For IGARCH or non-stationary GARCH
        unconditional_variance = omega * 10; % Just use a larger value
    end
    
    ht(1) = unconditional_variance;
else
    % For other models, use a reasonable starting value
    ht(1) = parameters.omega * 10;
end

% Create placeholder for previous residuals (for MA component)
residuals = zeros(totalObs, 1);

% Generate the variance and return series using the appropriate model recursion
switch modelType
    case 'GARCH'
        % Standard GARCH(p,q) model
        omega = parameters.omega;
        alpha = parameters.alpha;
        beta = parameters.beta;
        
        for t = 2:totalObs
            % GARCH recursion
            ht(t) = omega;
            
            % ARCH terms
            for i = 1:min(t-1, q)
                ht(t) = ht(t) + alpha(i) * residuals(t-i)^2;
            end
            
            % GARCH terms
            for j = 1:min(t-1, p)
                ht(t) = ht(t) + beta(j) * ht(t-j);
            end
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
        
    case {'GJR', 'TARCH'}
        % GJR/TARCH model with asymmetric effects
        omega = parameters.omega;
        alpha = parameters.alpha;
        gamma = parameters.gamma;
        beta = parameters.beta;
        
        for t = 2:totalObs
            % GJR recursion
            ht(t) = omega;
            
            % ARCH terms with asymmetry
            for i = 1:min(t-1, q)
                arch_term = alpha(i) * residuals(t-i)^2;
                if residuals(t-i) < 0
                    arch_term = arch_term + gamma(i) * residuals(t-i)^2;
                end
                ht(t) = ht(t) + arch_term;
            end
            
            % GARCH terms
            for j = 1:min(t-1, p)
                ht(t) = ht(t) + beta(j) * ht(t-j);
            end
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
        
    case 'EGARCH'
        % Exponential GARCH model
        omega = parameters.omega;
        alpha = parameters.alpha;
        gamma = parameters.gamma;
        beta = parameters.beta;
        
        % Expected value of |z_t| for standard normal = sqrt(2/pi)
        absZexpected = sqrt(2/pi);
        
        for t = 2:totalObs
            % EGARCH recursion on log variance
            log_ht = omega;
            
            % ARCH terms with asymmetry
            for i = 1:min(t-1, q)
                if ht(t-i) > 0 % Ensure valid variance for standardization
                    z_t = residuals(t-i) / sqrt(ht(t-i));
                    log_ht = log_ht + alpha(i) * (abs(z_t) - absZexpected);
                    log_ht = log_ht + gamma(i) * z_t;
                end
            end
            
            % GARCH terms
            for j = 1:min(t-1, p)
                log_ht = log_ht + beta(j) * log(max(ht(t-j), 1e-6));
            end
            
            % Convert from log variance to variance
            ht(t) = exp(log_ht);
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
        
    case 'AGARCH'
        % Asymmetric GARCH model
        omega = parameters.omega;
        alpha = parameters.alpha;
        gamma = parameters.gamma;
        beta = parameters.beta;
        
        for t = 2:totalObs
            % AGARCH recursion
            ht(t) = omega;
            
            % ARCH terms with asymmetry
            for i = 1:min(t-1, q)
                ht(t) = ht(t) + alpha(i) * (residuals(t-i) - gamma)^2;
            end
            
            % GARCH terms
            for j = 1:min(t-1, p)
                ht(t) = ht(t) + beta(j) * ht(t-j);
            end
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
        
    case 'IGARCH'
        % Integrated GARCH model
        omega = parameters.omega;
        alpha = parameters.alpha;
        beta = parameters.beta;
        
        for t = 2:totalObs
            % IGARCH recursion
            ht(t) = omega;
            
            % ARCH terms
            for i = 1:min(t-1, q)
                ht(t) = ht(t) + alpha(i) * residuals(t-i)^2;
            end
            
            % GARCH terms
            for j = 1:min(t-1, p)
                ht(t) = ht(t) + beta(j) * ht(t-j);
            end
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
        
    case 'NAGARCH'
        % Nonlinear Asymmetric GARCH model
        omega = parameters.omega;
        alpha = parameters.alpha;
        gamma = parameters.gamma;
        beta = parameters.beta;
        
        for t = 2:totalObs
            % NAGARCH recursion
            ht(t) = omega;
            
            % GARCH terms
            for j = 1:min(t-1, p)
                ht(t) = ht(t) + beta(j) * ht(t-j);
            end
            
            % ARCH terms with nonlinear asymmetry
            for i = 1:min(t-1, q)
                if ht(t-i) > 0 % Ensure valid variance for standardization
                    z_t = residuals(t-i) / sqrt(ht(t-i));
                    ht(t) = ht(t) + alpha(i) * ht(t-i) * (z_t - gamma)^2;
                end
            end
            
            % Generate return with this variance
            residuals(t) = sqrt(ht(t)) * innovations(t);
        end
end

% Remove burn-in period
returns = residuals(burnin+1:end);
variances = ht(burnin+1:end);

% Ensure all variances are positive (numerical stability)
variances = max(variances, 1e-6);

% Create model object for the output
model = struct('modelType', modelType, 'p', p, 'q', q, ...
    'distribution', parameters.distribution);
if isfield(parameters, 'distParams')
    model.distParams = parameters.distParams;
end

% Create output structure
volData = struct();
volData.returns = returns;
volData.ht = variances;
volData.parameters = parameters;
volData.model = model;

end

function samples = generateDistributionSamples(distributionType, numSamples, parameters)
% GENERATEDISTRIBUTIONSAMPLES Generates random samples from various statistical distributions
%
% USAGE:
%   SAMPLES = generateDistributionSamples(DISTRIBUTIONTYPE, NUMSAMPLES)
%   SAMPLES = generateDistributionSamples(DISTRIBUTIONTYPE, NUMSAMPLES, PARAMETERS)
%
% INPUTS:
%   DISTRIBUTIONTYPE - String specifying the distribution:
%                      'normal', 't', 'ged', 'skewt'
%   NUMSAMPLES       - Number of random samples to generate
%   PARAMETERS       - [OPTIONAL] Structure with fields specific to distribution:
%                      For 'normal': .mean, .variance
%                      For 't': .nu (degrees of freedom)
%                      For 'ged': .nu (shape parameter)
%                      For 'skewt': .nu (degrees of freedom), .lambda (skewness)
%
% OUTPUT:
%   SAMPLES - Structure containing:
%             .data         - Generated random samples
%             .parameters   - True distribution parameters
%             .distribution - Distribution name
%             .moments      - Theoretical moments (mean, variance, skewness, kurtosis)
%
% COMMENTS:
%   This function generates random samples from various statistical distributions
%   commonly used in financial modeling. The generated samples can be used to test
%   distribution fitting functions, risk measures, and other statistical tools.
%
% EXAMPLES:
%   % Generate normal samples
%   normalSamples = generateDistributionSamples('normal', 1000);
%
%   % Generate Student's t samples with 5 degrees of freedom
%   params = struct('nu', 5);
%   tSamples = generateDistributionSamples('t', 1000, params);
%
%   % Generate skewed t samples
%   params = struct('nu', 5, 'lambda', -0.3);
%   skewSamples = generateDistributionSamples('skewt', 1000, params);

global RANDOM_SEED;

% Validate distribution type
distributionType = lower(distributionType);
validDists = {'normal', 't', 'ged', 'skewt'};
if ~ismember(distributionType, validDists)
    error('Invalid distribution type. Supported types: %s', strjoin(validDists, ', '));
end

% Validate number of samples
options.isInteger = true;
options.isPositive = true;
numSamples = parametercheck(numSamples, 'numSamples', options);

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Initialize output structure
samples = struct();
samples.distribution = distributionType;

% Process parameters and generate samples based on distribution type
switch distributionType
    case 'normal'
        % Normal distribution
        if ~isfield(parameters, 'mean')
            parameters.mean = 0;
        end
        
        if ~isfield(parameters, 'variance')
            parameters.variance = 1;
        else
            options.isPositive = true;
            parametercheck(parameters.variance, 'variance', options);
        end
        
        % Generate samples
        samples.data = parameters.mean + sqrt(parameters.variance) * randn(numSamples, 1);
        samples.parameters = parameters;
        
        % Theoretical moments
        samples.moments.mean = parameters.mean;
        samples.moments.variance = parameters.variance;
        samples.moments.skewness = 0;
        samples.moments.kurtosis = 3; % Excess kurtosis = 0
        
    case 't'
        % Student's t distribution
        if ~isfield(parameters, 'nu')
            parameters.nu = 5; % Default degrees of freedom
        else
            options.isPositive = true;
            parametercheck(parameters.nu, 'nu', options);
            
            % DoF must be > 2 for standardized t-distribution
            if parameters.nu <= 2
                error('Degrees of freedom (nu) must be > 2 for a well-defined variance');
            end
        end
        
        if ~isfield(parameters, 'mean')
            parameters.mean = 0;
        end
        
        if ~isfield(parameters, 'variance')
            parameters.variance = 1;
        else
            options.isPositive = true;
            parametercheck(parameters.variance, 'variance', options);
        end
        
        % Generate standardized t samples
        std_t_samples = stdtrnd([numSamples, 1], parameters.nu);
        
        % Apply location and scale
        samples.data = parameters.mean + sqrt(parameters.variance) * std_t_samples;
        samples.parameters = parameters;
        
        % Theoretical moments
        samples.moments.mean = parameters.mean;
        samples.moments.variance = parameters.variance;
        samples.moments.skewness = 0; % t-distribution is symmetric
        
        % Excess kurtosis = 6/(nu-4) for nu > 4, infinite otherwise
        if parameters.nu > 4
            samples.moments.kurtosis = 3 + 6/(parameters.nu - 4);
        else
            samples.moments.kurtosis = Inf;
        end
        
    case 'ged'
        % Generalized Error Distribution
        if ~isfield(parameters, 'nu')
            parameters.nu = 1.5; % Default shape parameter
        else
            options.isPositive = true;
            parametercheck(parameters.nu, 'nu', options);
        end
        
        if ~isfield(parameters, 'mean')
            parameters.mean = 0;
        end
        
        if ~isfield(parameters, 'variance')
            parameters.variance = 1;
        else
            options.isPositive = true;
            parametercheck(parameters.variance, 'variance', options);
        end
        
        % Generate GED samples
        ged_samples = gedrnd(parameters.nu, numSamples, 1);
        
        % Apply location and scale
        samples.data = parameters.mean + sqrt(parameters.variance) * ged_samples;
        samples.parameters = parameters;
        
        % Theoretical moments
        samples.moments.mean = parameters.mean;
        samples.moments.variance = parameters.variance;
        samples.moments.skewness = 0; % GED is symmetric
        
        % Excess kurtosis = gamma(5/nu)*gamma(1/nu)/(gamma(3/nu)^2) - 3
        nu = parameters.nu;
        samples.moments.kurtosis = 3 + (gamma(5/nu)*gamma(1/nu))/(gamma(3/nu)^2) - 3;
        
    case 'skewt'
        % Hansen's skewed t distribution
        if ~isfield(parameters, 'nu')
            parameters.nu = 5; % Default degrees of freedom
        else
            options.isPositive = true;
            parametercheck(parameters.nu, 'nu', options);
            
            % DoF must be > 2 for standardized t-distribution
            if parameters.nu <= 2
                error('Degrees of freedom (nu) must be > 2 for a well-defined variance');
            end
        end
        
        if ~isfield(parameters, 'lambda')
            parameters.lambda = -0.2; % Default negative skewness
        else
            if abs(parameters.lambda) >= 1
                error('Skewness parameter lambda must be in range (-1,1)');
            end
        end
        
        if ~isfield(parameters, 'mean')
            parameters.mean = 0;
        end
        
        if ~isfield(parameters, 'variance')
            parameters.variance = 1;
        else
            options.isPositive = true;
            parametercheck(parameters.variance, 'variance', options);
        end
        
        % Generate skewed t samples
        skewt_samples = skewtrnd(parameters.nu, parameters.lambda, numSamples, 1);
        
        % Apply location and scale
        samples.data = parameters.mean + sqrt(parameters.variance) * skewt_samples;
        samples.parameters = parameters;
        
        % Theoretical moments
        % These formulas are approximations based on Hansen's skewed-t
        nu = parameters.nu;
        lambda = parameters.lambda;
        
        % Compute Hansen's skewed t moments
        a = 4*lambda*((nu-2)/(nu-1))*(gamma((nu+1)/2)/gamma(nu/2)/sqrt(pi));
        b = sqrt(1 + 3*lambda^2 - a^2);
        
        % Mean and variance for the standardized skewed-t
        samples.moments.mean = parameters.mean;
        samples.moments.variance = parameters.variance;
        
        % Skewness and kurtosis depend on the distribution parameters
        % Note: These are approximations, exact formulas are complex
        % Skewness is proportional to lambda
        samples.moments.skewness = 4*lambda*b^3/(1-2*lambda^2)^(3/2);
        
        % Kurtosis increases as nu decreases and lambda increases in magnitude
        if nu > 4
            base_kurt = 3 + 6/(nu-4); % Base kurtosis from regular t
        else
            base_kurt = Inf;
        end
        samples.moments.kurtosis = base_kurt * (1 + 0.5*lambda^2);
end

% Compute sample moments
sampleMean = mean(samples.data);
sampleVar = var(samples.data);
sampleSkew = skewness(samples.data);
sampleKurt = kurtosis(samples.data);

% Add sample moments to output
samples.sampleMoments = struct(...
    'mean', sampleMean, ...
    'variance', sampleVar, ...
    'skewness', sampleSkew, ...
    'kurtosis', sampleKurt);

end

function hfData = generateHighFrequencyData(numDays, observationsPerDay, parameters)
% GENERATEHIGHFREQUENCYDATA Generates synthetic high-frequency financial data for realized volatility testing
%
% USAGE:
%   HFDATA = generateHighFrequencyData(NUMDAYS, OBSERVATIONSPERDAY)
%   HFDATA = generateHighFrequencyData(NUMDAYS, OBSERVATIONSPERDAY, PARAMETERS)
%
% INPUTS:
%   NUMDAYS           - Number of days to simulate
%   OBSERVATIONSPERDAY - Number of intraday observations per day
%   PARAMETERS        - [OPTIONAL] Structure with fields:
%                       .garchParams - [omega alpha beta] for daily GARCH process (default: [1e-5 0.1 0.85])
%                       .jumpIntensity - Average number of jumps per day (default: 1)
%                       .jumpSize - Average absolute jump size (default: 0.5%)
%                       .diurnalFactor - Strength of U-shaped intraday pattern (default: 0.5)
%                       .noiseLevel - Magnitude of microstructure noise (default: 1e-4)
%                       .distribution - Innovation distribution: 'normal', 't', 'ged', 'skewt'
%                                       (default: 'normal')
%                       .distParams - Distribution parameters (default: distribution-specific)
%
% OUTPUT:
%   HFDATA - Structure containing:
%            .returns      - High-frequency return series
%            .prices       - Simulated price path
%            .timestamps   - Time points for each observation
%            .dailyVolatility - True daily integrated volatility
%            .jumpTimes    - Times of price jumps
%            .jumpSizes    - Sizes of price jumps
%            .parameters   - Simulation parameters
%
% COMMENTS:
%   This function generates high-frequency financial return data with realistic
%   features including:
%   - Daily volatility clustering (GARCH process)
%   - Intraday U-shaped volatility pattern
%   - Price jumps
%   - Microstructure noise
%   - Non-Gaussian return distributions
%
%   The data is suitable for testing realized volatility estimators, jump
%   detection methods, and other high-frequency financial econometric techniques.
%
% EXAMPLES:
%   % Generate 5-minute data for 30 days
%   hfData = generateHighFrequencyData(30, 78);
%
%   % Generate 1-minute data with increased jumps and t-distributed returns
%   params = struct('jumpIntensity', 3, 'distribution', 't', 'distParams', 4);
%   hfData = generateHighFrequencyData(10, 390, params);

global RANDOM_SEED;

% Validate input parameters
options.isInteger = true;
options.isPositive = true;
numDays = parametercheck(numDays, 'numDays', options);
observationsPerDay = parametercheck(observationsPerDay, 'observationsPerDay', options);

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set GARCH parameters for daily volatility
if ~isfield(parameters, 'garchParams')
    % Default GARCH(1,1) parameters: omega=1e-5, alpha=0.1, beta=0.85
    parameters.garchParams = [1e-5, 0.1, 0.85];
end

% Set jump parameters
if ~isfield(parameters, 'jumpIntensity')
    parameters.jumpIntensity = 1; % Average 1 jump per day
else
    options.isNonNegative = true;
    parametercheck(parameters.jumpIntensity, 'jumpIntensity', options);
end

if ~isfield(parameters, 'jumpSize')
    parameters.jumpSize = 0.005; % 0.5% average jump size
else
    options.isPositive = true;
    parametercheck(parameters.jumpSize, 'jumpSize', options);
end

% Set diurnal pattern strength
if ~isfield(parameters, 'diurnalFactor')
    parameters.diurnalFactor = 0.5; % Moderate U-shaped pattern
else
    options.isNonNegative = true;
    parametercheck(parameters.diurnalFactor, 'diurnalFactor', options);
end

% Set microstructure noise level
if ~isfield(parameters, 'noiseLevel')
    parameters.noiseLevel = 1e-4; % Typical level for liquid assets
else
    options.isNonNegative = true;
    parametercheck(parameters.noiseLevel, 'noiseLevel', options);
end

% Set distribution type and parameters
if ~isfield(parameters, 'distribution')
    parameters.distribution = 'normal';
end
distType = lower(parameters.distribution);

% Process distribution parameters
switch distType
    case 'normal'
        % Normal distribution (no additional parameters needed)
        distParams = [];
    case 't'
        % Student's t distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 5; % Default degrees of freedom
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'ged'
        % GED distribution
        if ~isfield(parameters, 'distParams')
            parameters.distParams = 1.5; % Default shape parameter
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams, 'distParams', options);
    case 'skewt'
        % Skewed t distribution
        if ~isfield(parameters, 'distParams') || length(parameters.distParams) < 2
            parameters.distParams = [5, -0.2]; % Default: DoF=5, lambda=-0.2 (negative skew)
        end
        distParams = parameters.distParams;
        options.isPositive = true;
        parametercheck(distParams(1), 'distParams(1)', options); % nu > 0
        if abs(distParams(2)) >= 1
            error('Skewness parameter lambda must be in range (-1,1)');
        end
    otherwise
        error('Unsupported distribution type: %s', distType);
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Calculate total number of observations
totalObs = numDays * observationsPerDay;

% Generate timestamp array (assuming trading day from 9:30 to 16:00)
timeIncrementPerDay = 6.5 / observationsPerDay; % 6.5 hours of trading = 390 minutes
timestamps = zeros(totalObs, 1);
for day = 1:numDays
    dayStart = (day - 1) * observationsPerDay + 1;
    dayEnd = day * observationsPerDay;
    dayTimestamps = (0:observationsPerDay-1)' * timeIncrementPerDay + (day-1) * 24;
    timestamps(dayStart:dayEnd) = dayTimestamps;
end

% Generate daily volatility using GARCH process
garchParams = parameters.garchParams;
omega = garchParams(1);
alpha = garchParams(2);
beta = garchParams(3);

% Simulate daily volatility
dailyVol = zeros(numDays + 1, 1); % +1 for burn-in
dailyVol(1) = omega / (1 - alpha - beta); % Start with unconditional variance

% Generate daily volatility using GARCH recursion
dailyReturns = zeros(numDays + 1, 1);
for t = 2:numDays+1
    % GARCH recursion
    dailyVol(t) = omega + alpha * dailyReturns(t-1)^2 + beta * dailyVol(t-1);
    
    % Generate daily return (not used, just for GARCH process)
    dailyReturns(t) = sqrt(dailyVol(t)) * randn();
end

% Remove burn-in period
dailyVol = dailyVol(2:end);

% Generate intraday diurnal pattern (U-shaped volatility)
diurnalPattern = zeros(observationsPerDay, 1);
for i = 1:observationsPerDay
    % Normalized time in [0, 1] for the trading day
    t_norm = (i - 1) / (observationsPerDay - 1);
    
    % U-shaped pattern: high at open and close, lower in the middle
    diurnalPattern(i) = 1 + parameters.diurnalFactor * (1 - 4 * (t_norm - 0.5)^2);
end

% Replicate diurnal pattern for all days
fullDiurnalPattern = repmat(diurnalPattern, numDays, 1);

% Generate high-frequency returns
hfReturns = zeros(totalObs, 1);
trueVol = zeros(totalObs, 1);
microstructureNoise = zeros(totalObs, 1);

% Generate innovations based on the specified distribution
switch distType
    case 'normal'
        % Standard normal innovations
        innovations = randn(totalObs, 1);
    case 't'
        % Standardized Student's t innovations
        innovations = stdtrnd([totalObs, 1], distParams);
    case 'ged'
        % Generalized Error Distribution innovations
        innovations = gedrnd(distParams, totalObs, 1);
    case 'skewt'
        % Hansen's skewed t innovations
        nu = distParams(1);
        lambda = distParams(2);
        innovations = skewtrnd(nu, lambda, totalObs, 1);
end

% Scale innovations by local volatility (daily GARCH * intraday pattern)
for day = 1:numDays
    dayStart = (day - 1) * observationsPerDay + 1;
    dayEnd = day * observationsPerDay;
    
    % Scale factor for this day's volatility
    dayScale = sqrt(dailyVol(day) / observationsPerDay);
    
    % Apply daily volatility and intraday pattern
    for i = dayStart:dayEnd
        localVol = dayScale * fullDiurnalPattern(i);
        trueVol(i) = localVol^2; % Store true local variance
        hfReturns(i) = localVol * innovations(i);
        
        % Add microstructure noise
        microstructureNoise(i) = parameters.noiseLevel * randn();
        hfReturns(i) = hfReturns(i) + microstructureNoise(i);
    end
end

% Generate price jumps
% Expected number of jumps
expectedJumps = numDays * parameters.jumpIntensity;
actualNumJumps = poissrnd(expectedJumps);

% Generate jump times and sizes
jumpTimes = sort(ceil(rand(actualNumJumps, 1) * totalObs));
jumpSizes = parameters.jumpSize * randn(actualNumJumps, 1);

% Add jumps to returns
for i = 1:actualNumJumps
    if jumpTimes(i) <= totalObs
        hfReturns(jumpTimes(i)) = hfReturns(jumpTimes(i)) + jumpSizes(i);
    end
end

% Convert returns to price path (starting at 100)
prices = 100 * cumprod(1 + hfReturns);

% Compute true daily integrated volatility (sum of squared intraday true volatilities)
dailyIntVolatility = zeros(numDays, 1);
for day = 1:numDays
    dayStart = (day - 1) * observationsPerDay + 1;
    dayEnd = day * observationsPerDay;
    dailyIntVolatility(day) = sum(trueVol(dayStart:dayEnd));
end

% Create output structure
hfData = struct();
hfData.returns = hfReturns;
hfData.prices = prices;
hfData.timestamps = timestamps;
hfData.dailyVolatility = dailyIntVolatility;
hfData.jumpTimes = jumpTimes;
hfData.jumpSizes = jumpSizes;
hfData.parameters = parameters;
hfData.trueLocalVolatility = trueVol;
hfData.microstructureNoise = microstructureNoise;

end

function csData = generateCrossSectionalData(numAssets, numPeriods, numFactors, parameters)
% GENERATECROSSSECTIONALDATA Generates cross-sectional financial data with known factor structure
%
% USAGE:
%   CSDATA = generateCrossSectionalData(NUMASSETS, NUMPERIODS, NUMFACTORS)
%   CSDATA = generateCrossSectionalData(NUMASSETS, NUMPERIODS, NUMFACTORS, PARAMETERS)
%
% INPUTS:
%   NUMASSETS   - Number of assets (e.g., stocks)
%   NUMPERIODS  - Number of time periods
%   NUMFACTORS  - Number of risk factors
%   PARAMETERS  - [OPTIONAL] Structure with fields:
%                 .factorCorr   - Factor correlation matrix (default: identity)
%                 .factorVol    - Factor volatilities (default: ones)
%                 .idioVol      - Idiosyncratic volatility (default: 0.2)
%                 .meanReturn   - Mean asset returns (default: 0.001 daily)
%                 .factorPremia - Risk premia for factors (default: [0.05, 0.03, 0.02, ...]/sqrt(252))
%                 .betaRange    - Range for factor loadings [min, max] (default: [-1, 1])
%                 .characteristics - Number of firm characteristics (default: 5)
%                 .charCorr     - Correlation between characteristics and betas
%                                 (default: 0.7 for first factor, decreasing for others)
%
% OUTPUT:
%   CSDATA - Structure containing:
%            .returns      - Asset returns, numPeriods × numAssets
%            .factors      - Factor returns, numPeriods × numFactors
%            .betas        - Factor loadings, numAssets × numFactors
%            .idioReturns  - Idiosyncratic returns, numPeriods × numAssets
%            .characteristics - Firm characteristics, numAssets × numCharacteristics
%            .factorPremia - True risk premia for factors
%            .parameters   - Simulation parameters
%
% COMMENTS:
%   This function generates cross-sectional financial data with a known factor structure
%   suitable for testing asset pricing models, cross-sectional regressions, portfolio
%   sorting techniques, and factor models.
%
%   The data generation follows a standard factor model:
%   r_it = alpha_i + sum_j(beta_ij * f_jt) + e_it
%
%   Firm characteristics are generated with specified correlation to factor loadings,
%   allowing testing of characteristic-based versus risk-based pricing.
%
% EXAMPLES:
%   % Generate data for 100 assets over 60 months with 3 factors
%   csData = generateCrossSectionalData(100, 60, 3);
%
%   % Generate data with custom parameters
%   params = struct('idioVol', 0.3, 'factorPremia', [0.08, 0.05]/sqrt(252));
%   csData = generateCrossSectionalData(500, 120, 2, params);

global RANDOM_SEED;

% Validate input parameters
options.isInteger = true;
options.isPositive = true;
numAssets = parametercheck(numAssets, 'numAssets', options);
numPeriods = parametercheck(numPeriods, 'numPeriods', options);
numFactors = parametercheck(numFactors, 'numFactors', options);

if numFactors > numAssets
    warning('Number of factors exceeds number of assets. This may result in singularity issues.');
end

% Set default parameters if not provided
if nargin < 4 || isempty(parameters)
    parameters = struct();
end

% Set factor correlation structure
if ~isfield(parameters, 'factorCorr')
    % Default: identity matrix (uncorrelated factors)
    parameters.factorCorr = eye(numFactors);
else
    % Validate correlation matrix
    corrMatrix = parameters.factorCorr;
    if size(corrMatrix, 1) ~= numFactors || size(corrMatrix, 2) ~= numFactors
        error('Factor correlation matrix must be numFactors × numFactors');
    end
    % Check if valid correlation matrix
    if ~isequal(corrMatrix, corrMatrix')
        error('Factor correlation matrix must be symmetric');
    end
    if any(diag(corrMatrix) ~= 1)
        error('Factor correlation matrix must have ones on the diagonal');
    end
    if any(abs(corrMatrix(:)) > 1)
        error('Factor correlation matrix must have values in range [-1,1]');
    end
    % Ensure positive semidefinite
    [~, p] = chol(corrMatrix);
    if p > 0
        warning('Factor correlation matrix is not positive semidefinite. Adjusting to nearest valid matrix.');
        % Adjust to nearest positive semidefinite matrix
        [V, D] = eig(corrMatrix);
        D = max(0, D); % Force eigenvalues to be non-negative
        corrMatrix = V * D * V';
        % Rescale to ensure ones on diagonal
        d = sqrt(diag(corrMatrix));
        corrMatrix = corrMatrix ./ (d * d');
        parameters.factorCorr = corrMatrix;
    end
end

% Set factor volatilities
if ~isfield(parameters, 'factorVol')
    % Default: unit volatility for all factors
    parameters.factorVol = ones(numFactors, 1);
else
    if length(parameters.factorVol) ~= numFactors
        error('Factor volatilities must have length equal to numFactors');
    end
    options.isPositive = true;
    parametercheck(parameters.factorVol, 'factorVol', options);
    parameters.factorVol = parameters.factorVol(:); % Ensure column vector
end

% Set idiosyncratic volatility
if ~isfield(parameters, 'idioVol')
    parameters.idioVol = 0.2; % Default: 20% idiosyncratic volatility
else
    options.isPositive = true;
    parametercheck(parameters.idioVol, 'idioVol', options);
end

% Set mean return
if ~isfield(parameters, 'meanReturn')
    parameters.meanReturn = 0.001; % Default: 0.1% daily mean return
end

% Set factor risk premia
if ~isfield(parameters, 'factorPremia')
    % Default factor risk premia, declining for higher factors
    % Scaled for daily returns if 252 trading days per year
    baseValues = [0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.006, 0.005, 0.004, 0.003];
    baseValues = baseValues / sqrt(252); % Convert annual to daily
    
    if numFactors <= length(baseValues)
        parameters.factorPremia = baseValues(1:numFactors)';
    else
        % For many factors, extend with small values
        parameters.factorPremia = [baseValues'; 0.002 * ones(numFactors - length(baseValues), 1) / sqrt(252)];
    end
else
    if length(parameters.factorPremia) ~= numFactors
        error('Factor risk premia must have length equal to numFactors');
    end
    parameters.factorPremia = parameters.factorPremia(:); % Ensure column vector
end

% Set beta range
if ~isfield(parameters, 'betaRange')
    parameters.betaRange = [-1, 2]; % Default beta range
else
    if length(parameters.betaRange) ~= 2
        error('Beta range must be a 2-element vector [min, max]');
    end
    if parameters.betaRange(1) >= parameters.betaRange(2)
        error('Beta range [min, max] must have min < max');
    end
end

% Set number of firm characteristics
if ~isfield(parameters, 'characteristics')
    parameters.characteristics = 5; % Default: 5 firm characteristics
else
    options.isInteger = true;
    options.isPositive = true;
    parametercheck(parameters.characteristics, 'characteristics', options);
end
numCharacteristics = parameters.characteristics;

% Set correlation between characteristics and betas
if ~isfield(parameters, 'charCorr')
    % Default: high correlation with first factor, lower for others
    if numFactors == 1
        parameters.charCorr = 0.7;
    else
        % Declining correlation pattern
        parameters.charCorr = 0.7 ./ (1:numFactors)';
    end
else
    if length(parameters.charCorr) ~= numFactors
        error('Characteristic correlation must have length equal to numFactors');
    end
    if any(abs(parameters.charCorr) > 1)
        error('Characteristic correlations must be in range [-1,1]');
    end
    parameters.charCorr = parameters.charCorr(:); % Ensure column vector
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Step 1: Generate factor returns with desired correlation structure
% Use Cholesky decomposition to induce correlation
C = chol(parameters.factorCorr, 'lower');
rawFactors = randn(numPeriods, numFactors);
factors = rawFactors * C';

% Scale factors by their volatilities and add risk premia
for j = 1:numFactors
    factors(:, j) = parameters.meanReturn + parameters.factorPremia(j) + ...
        parameters.factorVol(j) * factors(:, j);
end

% Step 2: Generate factor loadings (betas)
betaMin = parameters.betaRange(1);
betaMax = parameters.betaRange(2);
betas = betaMin + (betaMax - betaMin) * rand(numAssets, numFactors);

% Step 3: Generate firm characteristics correlated with factor loadings
characteristics = randn(numAssets, numCharacteristics);

% Create correlation between characteristics and factor loadings
% The first numFactors characteristics are correlated with corresponding betas
for j = 1:min(numFactors, numCharacteristics)
    % Standardize beta_j to have unit variance
    std_beta_j = (betas(:, j) - mean(betas(:, j))) / std(betas(:, j));
    
    % Generate correlated characteristic
    rho = parameters.charCorr(j);
    characteristics(:, j) = rho * std_beta_j + sqrt(1 - rho^2) * characteristics(:, j);
end

% Step 4: Generate idiosyncratic returns
idioReturns = parameters.idioVol * randn(numPeriods, numAssets);

% Step 5: Generate asset returns
returns = zeros(numPeriods, numAssets);
for i = 1:numAssets
    % Factor component of returns
    factorComponent = factors * betas(i, :)';
    
    % Add idiosyncratic component
    returns(:, i) = factorComponent + idioReturns(:, i);
end

% Create output structure
csData = struct();
csData.returns = returns;
csData.factors = factors;
csData.betas = betas;
csData.idioReturns = idioReturns;
csData.characteristics = characteristics;
csData.factorPremia = parameters.factorPremia;
csData.parameters = parameters;

end

function macroData = generateMacroeconomicData(numObservations, numSeries, parameters)
% GENERATEMACROECONOMICDATA Generates synthetic macroeconomic time series for VAR/VECM testing
%
% USAGE:
%   MACRODATA = generateMacroeconomicData(NUMOBSERVATIONS, NUMSERIES)
%   MACRODATA = generateMacroeconomicData(NUMOBSERVATIONS, NUMSERIES, PARAMETERS)
%
% INPUTS:
%   NUMOBSERVATIONS - Number of observations to generate
%   NUMSERIES       - Number of time series to generate
%   PARAMETERS      - [OPTIONAL] Structure with fields:
%                     .cointegration - Boolean or integer indicating number of cointegration relations
%                                     (default: true for 1 relation if numSeries>1)
%                     .unitRoots     - Boolean or vector indicating which series have unit roots
%                                     (default: true for all series)
%                     .seasonality   - Boolean or structure with seasonal parameters
%                                     (default: false)
%                     .structural    - Boolean indicating whether to include structural breaks
%                                     (default: false)
%                     .breakPoints   - Vector of break points if structural=true
%                                     (default: [0.33, 0.67] * numObservations)
%                     .varOrder      - Order p of the VAR process (default: 2)
%                     .correlation   - Target correlation matrix (default: AR(1) correlation structure)
%                     .trendStrength - Strength of deterministic trends (default: 0.001)
%
% OUTPUT:
%   MACRODATA - Structure containing:
%              .data        - Generated macroeconomic time series (numObservations × numSeries)
%              .integrated  - Boolean vector indicating which series are integrated
%              .cointRank   - Number of cointegration relations
%              .cointVectors - Cointegration vectors if applicable
%              .varCoefficients - VAR/VECM coefficient matrices
%              .breakPoints - Structural break points if applicable
%              .seasonal    - Seasonal component if applicable
%              .parameters  - Original parameters structure
%
% COMMENTS:
%   This function generates macroeconomic time series with realistic features
%   including unit roots, cointegration, seasonality, structural breaks, and
%   VAR dynamics. The generated data is suitable for testing VAR/VECM models,
%   unit root tests, and cointegration tests.
%
% EXAMPLES:
%   % Generate basic macroeconomic series
%   macroData = generateMacroeconomicData(200, 3);
%
%   % Generate series with seasonal component
%   params = struct('seasonality', true);
%   macroData = generateMacroeconomicData(300, 2, params);
%
%   % Generate series with structural breaks
%   params = struct('structural', true, 'breakPoints', [100, 200]);
%   macroData = generateMacroeconomicData(300, 4, params);

global RANDOM_SEED;

% Validate input parameters
options.isInteger = true;
options.isPositive = true;
numObservations = parametercheck(numObservations, 'numObservations', options);
numSeries = parametercheck(numSeries, 'numSeries', options);

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set cointegration flag/rank
if ~isfield(parameters, 'cointegration')
    % Default: one cointegration relation if multiple series
    parameters.cointegration = (numSeries > 1);
elseif isnumeric(parameters.cointegration)
    % Ensure cointegration rank is valid
    cointRank = parameters.cointegration;
    if cointRank < 0 || cointRank >= numSeries
        error('Cointegration rank must be between 0 and numSeries-1');
    end
end

% Convert boolean cointegration flag to rank
if islogical(parameters.cointegration)
    if parameters.cointegration
        cointRank = 1; % Default: one cointegration relation
    else
        cointRank = 0; % No cointegration
    end
else
    cointRank = parameters.cointegration;
end

% Set unit root flags
if ~isfield(parameters, 'unitRoots')
    % Default: all series have unit roots
    parameters.unitRoots = true;
end

if islogical(parameters.unitRoots) && length(parameters.unitRoots) == 1
    % Single boolean value - apply to all series
    hasUnitRoot = repmat(parameters.unitRoots, numSeries, 1);
else
    % Vector of booleans or indices
    if length(parameters.unitRoots) ~= numSeries
        error('Unit roots specification must be a scalar or a vector of length numSeries');
    end
    hasUnitRoot = logical(parameters.unitRoots(:));
end

% If we have cointegration, ensure we have enough unit roots
if cointRank > 0 && sum(hasUnitRoot) < cointRank + 1
    warning('Increasing number of unit roots to support cointegration rank %d', cointRank);
    hasUnitRoot(1:(cointRank+1)) = true;
end

% Set seasonality parameters
if ~isfield(parameters, 'seasonality')
    parameters.seasonality = false;
end

if isstruct(parameters.seasonality)
    % Seasonality structure provided
    seasonality = parameters.seasonality;
    if ~isfield(seasonality, 'frequency')
        seasonality.frequency = 4; % Default: quarterly data
    end
    if ~isfield(seasonality, 'amplitude')
        seasonality.amplitude = 0.1; % Default amplitude
    end
elseif parameters.seasonality
    % Boolean true - use defaults
    seasonality = struct('frequency', 4, 'amplitude', 0.1);
else
    % No seasonality
    seasonality = [];
end

% Set structural break parameters
if ~isfield(parameters, 'structural')
    parameters.structural = false;
end

if parameters.structural
    if ~isfield(parameters, 'breakPoints')
        % Default: breaks at 1/3 and 2/3 of the sample
        parameters.breakPoints = round([0.33, 0.67] * numObservations);
    else
        % Check validity of break points
        breakPoints = parameters.breakPoints;
        if any(breakPoints < 1) || any(breakPoints >= numObservations)
            error('Break points must be between 1 and numObservations-1');
        end
        parameters.breakPoints = sort(unique(breakPoints));
    end
    
    structural = true;
    breakPoints = parameters.breakPoints;
else
    structural = false;
    breakPoints = [];
end

% Set VAR order
if ~isfield(parameters, 'varOrder')
    parameters.varOrder = 2; % Default: VAR(2)
else
    options.isInteger = true;
    options.isPositive = true;
    parametercheck(parameters.varOrder, 'varOrder', options);
end
varOrder = parameters.varOrder;

% Set correlation structure
if ~isfield(parameters, 'correlation')
    % Default: AR(1)-type correlation with coefficient 0.5
    corrCoef = 0.5;
    parameters.correlation = zeros(numSeries, numSeries);
    for i = 1:numSeries
        for j = 1:numSeries
            parameters.correlation(i, j) = corrCoef^abs(i-j);
        end
    end
else
    % Validate correlation matrix
    corrMatrix = parameters.correlation;
    if size(corrMatrix, 1) ~= numSeries || size(corrMatrix, 2) ~= numSeries
        error('Correlation matrix must be numSeries × numSeries');
    end
    % Check if valid correlation matrix
    if ~isequal(corrMatrix, corrMatrix')
        error('Correlation matrix must be symmetric');
    end
    if any(diag(corrMatrix) ~= 1)
        error('Correlation matrix must have ones on the diagonal');
    end
    if any(abs(corrMatrix(:)) > 1)
        error('Correlation matrix must have values in range [-1,1]');
    end
    
    % Ensure positive semidefinite
    [~, p] = chol(corrMatrix);
    if p > 0
        warning('Correlation matrix is not positive semidefinite. Adjusting to nearest valid matrix.');
        % Adjust to nearest positive semidefinite matrix
        [V, D] = eig(corrMatrix);
        D = max(0, D); % Force eigenvalues to be non-negative
        corrMatrix = V * D * V';
        % Rescale to ensure ones on diagonal
        d = sqrt(diag(corrMatrix));
        corrMatrix = corrMatrix ./ (d * d');
        parameters.correlation = corrMatrix;
    end
end

% Set trend strength
if ~isfield(parameters, 'trendStrength')
    parameters.trendStrength = 0.001; % Default: weak trend
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Step 1: Generate initial series with correlation structure
% Use Cholesky decomposition for correlation
C = chol(parameters.correlation, 'lower');
innovations = randn(numObservations + varOrder, numSeries);
correlated_innovations = innovations * C';

% Step 2: Generate VAR/VECM process
% For cointegrated series, we'll generate a VECM, then convert to levels
if cointRank > 0
    % Generate VECM process
    % We need to create coefficient matrices that respect the cointegration rank
    
    % First, create cointegration vectors (alpha and beta)
    beta = randn(numSeries, cointRank); % Cointegration vectors
    alpha = randn(numSeries, cointRank) * 0.1; % Adjustment speeds (small for stability)
    
    % For series without unit roots, set corresponding alpha to zero
    alpha(~hasUnitRoot, :) = 0;
    
    % Create short-run dynamics coefficient matrix (for lagged differences)
    shortRunCoef = zeros(numSeries, numSeries, varOrder-1);
    for lag = 1:(varOrder-1)
        shortRunCoef(:,:,lag) = 0.2 * eye(numSeries) / lag; 
    end
    
    % Initialize the data array
    data = zeros(numObservations + varOrder, numSeries);
    
    % Set initial conditions
    data(1:varOrder, :) = randn(varOrder, numSeries);
    
    % Generate VECM process
    for t = (varOrder+1):(numObservations + varOrder)
        % Error correction term
        errCorrection = alpha * (beta' * data(t-1, :)');
        
        % Lagged differences
        laggedDiff = zeros(numSeries, 1);
        for lag = 1:(varOrder-1)
            laggedDiff = laggedDiff + shortRunCoef(:,:,lag) * (data(t-lag, :) - data(t-lag-1, :))';
        end
        
        % Combined dynamic
        data(t, :) = data(t-1, :) + (errCorrection + laggedDiff + correlated_innovations(t, :)')';
    end
    
    % Store the cointegration vectors for output
    cointVectors = beta;
    varCoefficients = struct('alpha', alpha, 'beta', beta, 'shortRun', shortRunCoef);
    
else
    % Generate VAR process without cointegration
    % Create VAR coefficient matrices
    varCoef = zeros(numSeries, numSeries, varOrder);
    
    % For unit root series, set own lag coefficient close to 1
    for i = 1:numSeries
        if hasUnitRoot(i)
            varCoef(i, i, 1) = 0.98; % Near unit root
        else
            varCoef(i, i, 1) = 0.5; % Stationary
        end
    end
    
    % Add cross-variable effects (small values for stability)
    for lag = 1:varOrder
        for i = 1:numSeries
            for j = 1:numSeries
                if i ~= j
                    varCoef(i, j, lag) = 0.05 / (lag * (1 + abs(i-j)));
                elseif lag > 1
                    varCoef(i, i, lag) = 0.1 / lag; 
                end
            end
        end
    end
    
    % Initialize the data array
    data = zeros(numObservations + varOrder, numSeries);
    
    % Set initial conditions
    data(1:varOrder, :) = randn(varOrder, numSeries);
    
    % Generate VAR process
    for t = (varOrder+1):(numObservations + varOrder)
        % Current value is sum of lagged effects plus innovation
        current = zeros(1, numSeries);
        
        for lag = 1:varOrder
            current = current + (data(t-lag, :) * varCoef(:,:,lag)');
        end
        
        % Add innovation
        data(t, :) = current + correlated_innovations(t, :);
    end
    
    % No cointegration vectors
    cointVectors = [];
    varCoefficients = varCoef;
end

% Remove burn-in period (keep only the last numObservations rows)
data = data((varOrder+1):end, :);

% Step 3: Add deterministic components (trends, seasonality, breaks)

% Add deterministic trend
trend = (1:numObservations)' * parameters.trendStrength;
for i = 1:numSeries
    % Apply trend to each series with random magnitude and direction
    data(:, i) = data(:, i) + trend * randn() * (0.5 + 0.5 * hasUnitRoot(i));
end

% Add seasonality if requested
if ~isempty(seasonality)
    freq = seasonality.frequency;
    amp = seasonality.amplitude;
    
    % Create seasonal pattern
    seasonalPattern = zeros(numObservations, numSeries);
    for i = 1:numSeries
        % Create unique seasonal pattern for each series
        phase = 2 * pi * rand(); % Random phase shift
        for t = 1:numObservations
            % Seasonal component: sin wave with amplitude and frequency
            seasonalPattern(t, i) = amp * sin(2 * pi * t / freq + phase);
        end
    end
    
    % Add seasonal pattern to data
    data = data + seasonalPattern;
else
    seasonalPattern = [];
end

% Add structural breaks if requested
if structural
    numBreaks = length(breakPoints);
    
    for b = 1:numBreaks
        breakPoint = breakPoints(b);
        
        % Create shift vector (0 before break, 1 after break)
        shiftVector = zeros(numObservations, 1);
        shiftVector((breakPoint+1):end) = 1;
        
        % Apply break to each series with random magnitude
        for i = 1:numSeries
            breakMag = 0.2 * randn(); % Random break magnitude
            data(:, i) = data(:, i) + shiftVector * breakMag;
        end
    end
end

% Create output structure
macroData = struct();
macroData.data = data;
macroData.integrated = hasUnitRoot;
macroData.cointRank = cointRank;
macroData.cointVectors = cointVectors;
macroData.varCoefficients = varCoefficients;
macroData.breakPoints = breakPoints;
macroData.seasonal = seasonalPattern;
macroData.parameters = parameters;

end

function simData = generateSimulatedData(properties, numObservations)
% GENERATESIMULATEDDATA Generates generic simulated data with specified properties
%
% USAGE:
%   SIMDATA = generateSimulatedData(PROPERTIES, NUMOBSERVATIONS)
%
% INPUTS:
%   PROPERTIES     - Structure with desired data properties:
%                    .mean      - Desired mean (default: 0)
%                    .variance  - Desired variance (default: 1)
%                    .skewness  - Desired skewness (default: 0)
%                    .kurtosis  - Desired kurtosis (default: 3)
%                    .autocorr  - Desired autocorrelation structure as vector of
%                                 lag coefficients (default: [])
%                    .garch     - GARCH parameters [omega alpha beta] (default: [])
%                    .heavyTail - Boolean for heavy-tailed distribution (default: false)
%                    .multivariate - Number of series for multivariate data (default: 1)
%                    .correlation  - Correlation matrix for multivariate data
%   NUMOBSERVATIONS - Number of observations to generate
%
% OUTPUT:
%   SIMDATA - Structure containing:
%             .data         - Generated data
%             .properties   - Actual properties of the generated data
%             .targetProps  - Target properties specified in input
%
% COMMENTS:
%   This function generates simulated data with specified statistical properties
%   including mean, variance, higher moments, autocorrelation, and heteroskedasticity.
%   It provides flexible data generation for testing statistical methods under
%   controlled conditions.
%
% EXAMPLES:
%   % Generate data with specified moments
%   props = struct('mean', 0.1, 'variance', 2, 'skewness', -0.5, 'kurtosis', 5);
%   simData = generateSimulatedData(props, 1000);
%
%   % Generate data with autocorrelation
%   props = struct('autocorr', [0.8, 0.6, 0.4, 0.2]);
%   simData = generateSimulatedData(props, 500);
%
%   % Generate multivariate data
%   props = struct('multivariate', 3, 'correlation', [1, 0.7, 0.3; 0.7, 1, 0.5; 0.3, 0.5, 1]);
%   simData = generateSimulatedData(props, 200);

global RANDOM_SEED;

% Validate number of observations
options.isInteger = true;
options.isPositive = true;
numObservations = parametercheck(numObservations, 'numObservations', options);

% Set default properties if not provided
if isempty(properties) || ~isstruct(properties)
    properties = struct();
end

% Set default mean
if ~isfield(properties, 'mean')
    properties.mean = 0;
end

% Set default variance
if ~isfield(properties, 'variance')
    properties.variance = 1;
else
    options.isPositive = true;
    parametercheck(properties.variance, 'variance', options);
end

% Set default skewness
if ~isfield(properties, 'skewness')
    properties.skewness = 0;
end

% Set default kurtosis (normal distribution has kurtosis = 3)
if ~isfield(properties, 'kurtosis')
    properties.kurtosis = 3;
else
    if properties.kurtosis < 1
        error('Kurtosis must be at least 1');
    end
end

% Set default autocorrelation
if ~isfield(properties, 'autocorr')
    properties.autocorr = [];
end

% Set default GARCH parameters
if ~isfield(properties, 'garch')
    properties.garch = [];
end

% Set default heavy tail flag
if ~isfield(properties, 'heavyTail')
    properties.heavyTail = false;
end

% Set default multivariate flag
if ~isfield(properties, 'multivariate')
    properties.multivariate = 1; % Default: univariate
else
    options.isInteger = true;
    options.isPositive = true;
    parametercheck(properties.multivariate, 'multivariate', options);
end
numSeries = properties.multivariate;

% Set correlation matrix for multivariate data
if numSeries > 1
    if ~isfield(properties, 'correlation')
        % Default correlation matrix: identity (uncorrelated series)
        properties.correlation = eye(numSeries);
    else
        % Validate correlation matrix
        corrMatrix = properties.correlation;
        if size(corrMatrix, 1) ~= numSeries || size(corrMatrix, 2) ~= numSeries
            error('Correlation matrix must be numSeries × numSeries');
        end
        if ~isequal(corrMatrix, corrMatrix')
            error('Correlation matrix must be symmetric');
        end
        if any(diag(corrMatrix) ~= 1)
            error('Correlation matrix must have ones on the diagonal');
        end
        if any(abs(corrMatrix(:)) > 1)
            error('Correlation matrix must have values in range [-1,1]');
        end
        
        % Ensure positive semidefinite
        [~, p] = chol(corrMatrix);
        if p > 0
            warning('Correlation matrix is not positive semidefinite. Adjusting to nearest valid matrix.');
            % Adjust to nearest positive semidefinite matrix
            [V, D] = eig(corrMatrix);
            D = max(0, D); % Force eigenvalues to be non-negative
            corrMatrix = V * D * V';
            % Rescale to ensure ones on diagonal
            d = sqrt(diag(corrMatrix));
            corrMatrix = corrMatrix ./ (d * d');
            properties.correlation = corrMatrix;
        end
    end
end

% Initialize random number generator for reproducibility
rng(RANDOM_SEED);

% Step 1: Generate base innovations with desired moments (univariate case)
if numSeries == 1
    % Generate data with specified moments using Fleishman's method
    % (for univariate data)
    
    % For heavy-tailed data without specific kurtosis, use t-distribution
    if properties.heavyTail && ~isfield(properties, 'kurtosis')
        % Use t-distribution with 5 degrees of freedom
        innovations = stdtrnd([numObservations, 1], 5);
    else
        % Use Fleishman's method for polynomial transformation of normal data
        % to achieve desired skewness and kurtosis
        [a, b, c, d] = fleishman_coeffs(properties.skewness, properties.kurtosis);
        
        % Generate standard normal data
        z = randn(numObservations, 1);
        
        % Apply polynomial transformation
        innovations = a + b*z + c*z.^2 + d*z.^3;
        
        % Standardize to have zero mean and unit variance
        innovations = (innovations - mean(innovations)) / std(innovations);
    end
    
    % Apply autocorrelation if specified
    if ~isempty(properties.autocorr)
        p = length(properties.autocorr);
        if any(abs(properties.autocorr) >= 1)
            error('Autocorrelation coefficients must be in range (-1,1)');
        end
        
        % Generate AR(p) process
        ar_data = zeros(numObservations + p, 1);
        ar_data(1:p) = innovations(1:p);
        
        for t = (p+1):(numObservations + p)
            ar_value = 0;
            for i = 1:p
                ar_value = ar_value + properties.autocorr(i) * ar_data(t-i);
            end
            ar_data(t) = ar_value + innovations(t-p);
        end
        
        % Use only the last numObservations values
        data = ar_data((p+1):end);
        
        % Restandardize
        data = (data - mean(data)) / std(data);
    else
        data = innovations;
    end
    
    % Apply GARCH effect if specified
    if ~isempty(properties.garch)
        if length(properties.garch) >= 3
            omega = properties.garch(1);
            alpha = properties.garch(2);
            beta = properties.garch(3);
            
            % Check stationarity
            if alpha + beta >= 1
                warning('GARCH parameters are non-stationary (alpha + beta >= 1)');
            end
            
            % Initialize variance series
            h = zeros(numObservations, 1);
            h(1) = omega / (1 - alpha - beta); % Unconditional variance
            
            % Generate GARCH process
            garch_data = zeros(numObservations, 1);
            for t = 1:numObservations
                if t > 1
                    h(t) = omega + alpha * garch_data(t-1)^2 + beta * h(t-1);
                end
                garch_data(t) = sqrt(h(t)) * data(t);
            end
            
            % Use the GARCH process data
            data = garch_data;
        else
            warning('GARCH parameters should be [omega, alpha, beta]. Ignoring GARCH effect.');
        end
    end
    
    % Apply desired mean and variance
    data = properties.mean + sqrt(properties.variance) * data;
    
else
    % Multivariate case: Generate correlated data using Cholesky decomposition
    
    % Generate independent standardized innovations
    if properties.heavyTail
        % Use t-distribution for heavy tails
        innovations = stdtrnd([numObservations, numSeries], 5);
    else
        innovations = randn(numObservations, numSeries);
    end
    
    % Apply correlation structure
    C = chol(properties.correlation, 'lower');
    correlated_innovations = innovations * C';
    
    % Apply autocorrelation if specified (to each series independently)
    if ~isempty(properties.autocorr)
        p = length(properties.autocorr);
        if any(abs(properties.autocorr) >= 1)
            error('Autocorrelation coefficients must be in range (-1,1)');
        end
        
        data = zeros(numObservations, numSeries);
        
        for series = 1:numSeries
            % Generate AR(p) process for this series
            ar_data = zeros(numObservations + p, 1);
            ar_data(1:p) = correlated_innovations(1:p, series);
            
            for t = (p+1):(numObservations + p)
                ar_value = 0;
                for i = 1:p
                    ar_value = ar_value + properties.autocorr(i) * ar_data(t-i);
                end
                ar_data(t) = ar_value + correlated_innovations(t-p, series);
            end
            
            % Use only the last numObservations values
            data(:, series) = ar_data((p+1):end);
        end
        
        % Restandardize each series
        for series = 1:numSeries
            data(:, series) = (data(:, series) - mean(data(:, series))) / std(data(:, series));
        end
    else
        data = correlated_innovations;
    end
    
    % Apply GARCH effect if specified (to each series independently)
    if ~isempty(properties.garch)
        if length(properties.garch) >= 3
            omega = properties.garch(1);
            alpha = properties.garch(2);
            beta = properties.garch(3);
            
            % Initialize output data matrix
            garch_data = zeros(numObservations, numSeries);
            
            for series = 1:numSeries
                % Initialize variance series
                h = zeros(numObservations, 1);
                h(1) = omega / (1 - alpha - beta); % Unconditional variance
                
                % Generate GARCH process for this series
                for t = 1:numObservations
                    if t > 1
                        h(t) = omega + alpha * garch_data(t-1, series)^2 + beta * h(t-1);
                    end
                    garch_data(t, series) = sqrt(h(t)) * data(t, series);
                end
            end
            
            % Use the GARCH process data
            data = garch_data;
        else
            warning('GARCH parameters should be [omega, alpha, beta]. Ignoring GARCH effect.');
        end
    end
    
    % Apply desired mean and variance to each series
    if isscalar(properties.mean)
        means = repmat(properties.mean, 1, numSeries);
    else
        means = properties.mean;
    end
    
    if isscalar(properties.variance)
        variances = repmat(properties.variance, 1, numSeries);
    else
        variances = properties.variance;
    end
    
    % Apply means and variances
    for series = 1:numSeries
        data(:, series) = means(series) + sqrt(variances(series)) * data(:, series);
    end
end

% Calculate achieved properties
achievedProps = struct();

if numSeries == 1
    achievedProps.mean = mean(data);
    achievedProps.variance = var(data);
    achievedProps.skewness = skewness(data);
    achievedProps.kurtosis = kurtosis(data);
    
    % Calculate autocorrelation if requested
    if ~isempty(properties.autocorr)
        p = length(properties.autocorr);
        achievedProps.autocorr = zeros(p, 1);
        for lag = 1:p
            autocorr_lag = corrcoef(data(1:end-lag), data(lag+1:end));
            achievedProps.autocorr(lag) = autocorr_lag(1, 2);
        end
    end
else
    achievedProps.mean = mean(data);
    achievedProps.variance = var(data);
    
    % Calculate correlation matrix
    achievedProps.correlation = corrcoef(data);
end

% Create output structure
simData = struct();
simData.data = data;
simData.properties = achievedProps;
simData.targetProps = properties;

end

function [a, b, c, d] = fleishman_coeffs(skewness, kurtosis)
% Helper function for Fleishman's method (polynomial transformation to achieve desired moments)
% Solves for Fleishman's polynomial coefficients to achieve desired skewness and kurtosis
% Polynomial form: Y = a + b*Z + c*Z^2 + d*Z^3 where Z is standard normal

% Ensure kurtosis is valid (>= skewness^2 + 1)
min_kurtosis = skewness^2 + 1;
if kurtosis < min_kurtosis
    warning('Kurtosis must be at least skewness^2 + 1. Adjusting kurtosis.');
    kurtosis = min_kurtosis + 0.01;
end

% Initial guess for coefficients
if skewness == 0
    % Symmetric case
    b_guess = 1;
    c_guess = 0;
    d_guess = (kurtosis - 3) / 6;
else
    % Asymmetric case - use approximation for initial guess
    b_guess = 1;
    c_guess = skewness / 2;
    d_guess = (kurtosis - 3) / 24;
end

% Objective function for optimization
function f = objective(x)
    b = x(1);
    c = x(2);
    d = x(3);
    
    % Constraint: E[Y] = 0
    eq1 = c;
    
    % Constraint: Var[Y] = 1
    eq2 = b^2 + 6*b*d + 2*c^2 + 15*d^2 - 1;
    
    % Constraint: Skewness[Y] = target skewness
    eq3 = 2*c*(b^2 + 24*b*d + 105*d^2 + 2) + (8*d + 4*b*d)*(b^2 + 24*b*d + 105*d^2 + 2) - skewness;
    
    % Constraint: Kurtosis[Y] = target kurtosis
    eq4 = 24*c^4 + 24*(b*d + c^2 + d^2*105)^2 + 12*c^2*(b^2 + 6*b*d + 2*c^2 + 15*d^2) + ...
          4*d*(b^3 + 36*b^2*d + 270*b*d^2 + 4*b*c^2 + 24*c^2*d + 105*d^3) - kurtosis;
    
    f = eq1^2 + eq2^2 + eq3^2 + eq4^2;
end

% Use optimization to find coefficients
options = optimset('Display', 'off', 'MaxFunEvals', 1000, 'MaxIter', 500);
sol = fminsearch(@objective, [b_guess, c_guess, d_guess], options);

% Extract solution
b = sol(1);
c = sol(2);
d = sol(3);
a = -c; % Ensures E[Y] = 0

end

function skew = skewness(x)
% Compute the skewness of the data
x = x(:);
n = length(x);
x_mean = mean(x);
x_std = std(x);
skew = (1/n) * sum(((x - x_mean) / x_std).^3);
end

function kurt = kurtosis(x)
% Compute the kurtosis of the data
x = x(:);
n = length(x);
x_mean = mean(x);
x_std = std(x);
kurt = (1/n) * sum(((x - x_mean) / x_std).^4);
end

function success = generateAllTestData(overwrite)
% GENERATEALLTESTDATA Generates all standard test datasets needed by the MFE Toolbox test suite
%
% USAGE:
%   SUCCESS = generateAllTestData()
%   SUCCESS = generateAllTestData(OVERWRITE)
%
% INPUTS:
%   OVERWRITE - [OPTIONAL] Boolean flag indicating whether to overwrite existing data files
%              (default: false)
%
% OUTPUT:
%   SUCCESS - Boolean indicating whether all datasets were successfully generated
%
% COMMENTS:
%   This function generates all standard test datasets needed by the MFE Toolbox test suite.
%   Each dataset is saved in a separate MAT file in the data directory.
%
%   The generated datasets include:
%   - financial_returns.mat: Synthetic financial return series
%   - high_frequency_data.mat: High-frequency financial data
%   - cross_sectional_data.mat: Cross-sectional financial data
%   - macroeconomic_data.mat: Synthetic macroeconomic time series
%   - known_distributions.mat: Samples from various distributions
%   - vol_data.mat: Volatility series with GARCH parameters
%   - simulated_data.mat: Generic simulated data with specified properties
%
% EXAMPLES:
%   % Generate all test datasets (don't overwrite existing files)
%   success = generateAllTestData();
%
%   % Regenerate all test datasets, overwriting existing files
%   success = generateAllTestData(true);

% Global variables
global DATA_PATH RANDOM_SEED

% Default overwrite flag
if nargin < 1 || isempty(overwrite)
    overwrite = false;
end

% Ensure the data directory exists
if ~exist(DATA_PATH, 'dir')
    mkdir(DATA_PATH);
end

% Set random seed for reproducibility
rng(RANDOM_SEED);

% Initialize success flag
success = true;

try
    % Generate financial returns data
    disp('Generating financial returns data...');
    % Generate 5 series of 1000 observations with different distributions
    financial_returns = struct();
    
    % Normal returns
    financial_returns.normal = generateFinancialReturns(1000, 1, struct('distribution', 'normal'));
    
    % Student's t returns
    financial_returns.t = generateFinancialReturns(1000, 1, struct('distribution', 't', 'distParams', 5));
    
    % GED returns
    financial_returns.ged = generateFinancialReturns(1000, 1, struct('distribution', 'ged', 'distParams', 1.5));
    
    % Skewed t returns
    financial_returns.skewt = generateFinancialReturns(1000, 1, struct('distribution', 'skewt', 'distParams', [5, -0.2]));
    
    % Returns with GARCH volatility
    financial_returns.garch = generateFinancialReturns(1000, 1, struct('garchParams', [0.01, 0.1, 0.85]));
    
    % Correlated returns
    corr_matrix = [1.0, 0.7, 0.5; 0.7, 1.0, 0.3; 0.5, 0.3, 1.0];
    financial_returns.correlated = generateFinancialReturns(1000, 3, struct('correlation', corr_matrix));
    
    % Save financial returns data
    saveTestData(financial_returns, 'financial_returns.mat', overwrite);
    
    % Generate high-frequency data
    disp('Generating high-frequency data...');
    % 5-minute data for 20 trading days
    hf_data = generateHighFrequencyData(20, 78, struct('jumpIntensity', 2));
    saveTestData(hf_data, 'high_frequency_data.mat', overwrite);
    
    % Generate cross-sectional data
    disp('Generating cross-sectional data...');
    % 100 assets, 60 months, 3 factors
    cs_data = generateCrossSectionalData(100, 60, 3);
    saveTestData(cs_data, 'cross_sectional_data.mat', overwrite);
    
    % Generate macroeconomic data
    disp('Generating macroeconomic data...');
    % 200 observations, 4 series, with cointegration and seasonality
    macro_params = struct('cointegration', true, 'seasonality', true, 'structural', true);
    macro_data = generateMacroeconomicData(200, 4, macro_params);
    saveTestData(macro_data, 'macroeconomic_data.mat', overwrite);
    
    % Generate samples from known distributions
    disp('Generating known distribution samples...');
    % Generate 5000 samples from each distribution
    dist_samples = struct();
    dist_samples.normal = generateDistributionSamples('normal', 5000);
    dist_samples.t = generateDistributionSamples('t', 5000, struct('nu', 5));
    dist_samples.ged = generateDistributionSamples('ged', 5000, struct('nu', 1.5));
    dist_samples.skewt = generateDistributionSamples('skewt', 5000, struct('nu', 5, 'lambda', -0.2));
    saveTestData(dist_samples, 'known_distributions.mat', overwrite);
    
    % Generate volatility data
    disp('Generating volatility data...');
    % Generate various GARCH-type volatility models
    vol_data = struct();
    vol_data.garch = generateVolatilitySeries(1000, 'GARCH');
    vol_data.egarch = generateVolatilitySeries(1000, 'EGARCH');
    vol_data.gjr = generateVolatilitySeries(1000, 'GJR');
    vol_data.tarch = generateVolatilitySeries(1000, 'TARCH');
    saveTestData(vol_data, 'vol_data.mat', overwrite);
    
    % Generate simulated data with specific properties
    disp('Generating simulated data...');
    sim_data = struct();
    
    % Data with high kurtosis
    props1 = struct('kurtosis', 8, 'skewness', 0);
    sim_data.heavy_tailed = generateSimulatedData(props1, 1000);
    
    % Data with negative skewness
    props2 = struct('skewness', -0.5);
    sim_data.skewed = generateSimulatedData(props2, 1000);
    
    % Data with autocorrelation
    props3 = struct('autocorr', [0.8, 0.6, 0.4, 0.2]);
    sim_data.autocorrelated = generateSimulatedData(props3, 1000);
    
    % Data with GARCH effects
    props4 = struct('garch', [0.01, 0.1, 0.85]);
    sim_data.garch = generateSimulatedData(props4, 1000);
    
    % Multivariate data
    props5 = struct('multivariate', 3, 'correlation', [1, 0.7, 0.3; 0.7, 1, 0.5; 0.3, 0.5, 1]);
    sim_data.multivariate = generateSimulatedData(props5, 1000);
    
    saveTestData(sim_data, 'simulated_data.mat', overwrite);
    
    disp('All test datasets successfully generated.');
catch e
    disp(['Error generating test data: ' e.message]);
    success = false;
end

end

function success = saveTestData(data, fileName, overwrite)
% SAVETESTDATA Saves generated test data to a MAT file
%
% USAGE:
%   SUCCESS = saveTestData(DATA, FILENAME)
%   SUCCESS = saveTestData(DATA, FILENAME, OVERWRITE)
%
% INPUTS:
%   DATA      - Data structure to save
%   FILENAME  - Name of the MAT file to create
%   OVERWRITE - [OPTIONAL] Boolean flag indicating whether to overwrite existing file
%               (default: false)
%
% OUTPUT:
%   SUCCESS - Boolean indicating whether the data was successfully saved
%
% COMMENTS:
%   This function saves a data structure to a MAT file in the data directory.
%   It adds metadata including the creation date and the random seed used.
%
% EXAMPLES:
%   % Save data without overwriting
%   saveTestData(returns, 'test_returns.mat');
%
%   % Save data and overwrite existing file
%   saveTestData(volatility, 'test_vol.mat', true);

% Validate inputs
datacheck(data, 'data');
if ~ischar(fileName)
    error('FILENAME must be a string');
end

% Default overwrite flag
if nargin < 3 || isempty(overwrite)
    overwrite = false;
end

% Global variables
global DATA_PATH RANDOM_SEED

% Create full file path
fullPath = fullfile(DATA_PATH, fileName);

% Check if file exists
fileExists = exist(fullPath, 'file') == 2;
if fileExists && ~overwrite
    warning('File %s already exists. Use overwrite=true to replace.', fullPath);
    success = false;
    return;
end

% Add metadata to the data structure
if isstruct(data)
    % Add creation date
    data.metadata.created = datestr(now);
    data.metadata.seed = RANDOM_SEED;
else
    % Wrap non-struct data in a struct
    origData = data;
    data = struct();
    data.data = origData;
    data.metadata.created = datestr(now);
    data.metadata.seed = RANDOM_SEED;
end

% Save the data
try
    save(fullPath, '-struct', 'data');
    disp(['Successfully saved ' fullPath]);
    success = true;
catch e
    warning('Failed to save file: %s', e.message);
    success = false;
end

end