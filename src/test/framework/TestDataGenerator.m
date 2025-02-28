function [varargout] = TestDataGenerator(functionName, varargin)
% TESTDATAGENERATOR A utility class for generating test data sets with known statistical properties for the MFE Toolbox test framework
%
% This utility generates test data with controlled statistical properties for validating
% MFE Toolbox components. Functions create financial time series, distribution samples, 
% volatility series, and other financial econometric datasets with controlled parameters
% to enable comprehensive validation.
%
% USAGE:
%   returns = TestDataGenerator('generateFinancialReturns', numObservations, numSeries, parameters)
%   volData = TestDataGenerator('generateVolatilitySeries', numObservations, modelType, parameters)
%   distData = TestDataGenerator('generateDistributionSamples', distributionType, numSamples, parameters)
%   hfData = TestDataGenerator('generateHighFrequencyData', numDays, observationsPerDay, parameters)
%   csData = TestDataGenerator('generateCrossSectionalData', numAssets, numPeriods, numFactors, parameters)
%   success = TestDataGenerator('saveTestData', data, fileName, options)
%   data = TestDataGenerator('loadTestData', fileName)
%   success = TestDataGenerator('generateAllTestData', overwrite)
%   tsData = TestDataGenerator('generateTimeSeriesData', parameters)
%   bootstrapData = TestDataGenerator('generateBootstrapTestData', parameters)
%
% See help for each specific function for detailed documentation on inputs and outputs
%
% Examples:
%   % Generate financial returns with GARCH(1,1) volatility
%   params = struct('mean', 0, 'variance', 1, 'garch', struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85));
%   returns = TestDataGenerator('generateFinancialReturns', 1000, 1, params);
%
%   % Generate volatility series with a GJR-GARCH model
%   params = struct('omega', 0.03, 'alpha', 0.05, 'gamma', 0.1, 'beta', 0.85);
%   volData = TestDataGenerator('generateVolatilitySeries', 1000, 'GJR', params);
%
%   % Generate all standard test datasets
%   TestDataGenerator('generateAllTestData', true);
%
% See also GEDRND, STDTRND, SKEWTRND, ARMAFOR, GARCHFOR, DATACHECK, PARAMETERCHECK

% Global path for test data storage
global TEST_DATA_PATH;
if isempty(TEST_DATA_PATH)
    TEST_DATA_PATH = '../data/';
end

% Input validation
if nargin < 1
    error('Function name must be provided as first argument');
end

% Dispatch to appropriate function based on function name
switch lower(functionName)
    case 'generatefinancialreturns'
        [varargout{1:nargout}] = generateFinancialReturns(varargin{:});
    case 'generatevolatilityseries'
        [varargout{1:nargout}] = generateVolatilitySeries(varargin{:});
    case 'generatedistributionsamples'
        [varargout{1:nargout}] = generateDistributionSamples(varargin{:});
    case 'generatehighfrequencydata'
        [varargout{1:nargout}] = generateHighFrequencyData(varargin{:});
    case 'generatecrosssectionaldata'
        [varargout{1:nargout}] = generateCrossSectionalData(varargin{:});
    case 'savetestdata'
        [varargout{1:nargout}] = saveTestData(varargin{:});
    case 'loadtestdata'
        [varargout{1:nargout}] = loadTestData(varargin{:});
    case 'generatealltestdata'
        [varargout{1:nargout}] = generateAllTestData(varargin{:});
    case 'generatetimeseriesdata'
        [varargout{1:nargout}] = generateTimeSeriesData(varargin{:});
    case 'generatebootstraptestdata'
        [varargout{1:nargout}] = generateBootstrapTestData(varargin{:});
    otherwise
        error('Unknown function: %s', functionName);
end
end

%% Financial Returns Generation Function
function returns = generateFinancialReturns(numObservations, numSeries, parameters)
% GENERATEFINANCIALRETURNS Generates synthetic financial return series with specified statistical properties
%
% USAGE:
%   RETURNS = generateFinancialReturns(NUMOBSERVATIONS, NUMSERIES, PARAMETERS)
%
% INPUTS:
%   NUMOBSERVATIONS - Number of observations to generate
%   NUMSERIES - Number of return series to generate
%   PARAMETERS - Structure with fields:
%       mean - Mean return (scalar or vector of length NUMSERIES) [default: 0]
%       variance - Return variance (scalar or vector) [default: 1]
%       distribution - String: 'normal', 't', 'ged', 'skewt' [default: 'normal']
%       distParams - Distribution parameters:
%          For 't': degrees of freedom [default: 5]
%          For 'ged': shape parameter [default: 1.5]
%          For 'skewt': [degrees of freedom, skewness] [default: [5, -0.1]]
%       acf - Autocorrelation structure (vector of coefficients) [default: []]
%       crossCorr - Cross-correlation matrix (NUMSERIES × NUMSERIES) [default: identity]
%       garch - Structure with GARCH parameters for volatility clustering:
%          modelType - 'GARCH', 'EGARCH', 'GJR', etc. [default: 'GARCH'] 
%          p - GARCH order [default: 1]
%          q - ARCH order [default: 1]
%          omega, alpha, beta - Model parameters [defaults: 0.05, 0.1, 0.85]
%          (additional parameters for asymmetric models like 'gamma')
%       arParameters - Vector of AR coefficients [default: []]
%       maParameters - Vector of MA coefficients [default: []]
%
% OUTPUTS:
%   RETURNS - Matrix of generated returns (NUMOBSERVATIONS × NUMSERIES)

% Validate inputs
parametercheck(numObservations, 'numObservations', struct('isInteger', true, 'isPositive', true));
parametercheck(numSeries, 'numSeries', struct('isInteger', true, 'isPositive', true));

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set default mean
if ~isfield(parameters, 'mean') || isempty(parameters.mean)
    parameters.mean = zeros(1, numSeries);
elseif isscalar(parameters.mean)
    parameters.mean = repmat(parameters.mean, 1, numSeries);
elseif length(parameters.mean) ~= numSeries
    error('Length of mean vector must match numSeries');
end

% Set default variance
if ~isfield(parameters, 'variance') || isempty(parameters.variance)
    parameters.variance = ones(1, numSeries);
elseif isscalar(parameters.variance)
    parameters.variance = repmat(parameters.variance, 1, numSeries);
elseif length(parameters.variance) ~= numSeries
    error('Length of variance vector must match numSeries');
end

% Set default distribution
if ~isfield(parameters, 'distribution') || isempty(parameters.distribution)
    parameters.distribution = 'normal';
end

% Set default distribution parameters
if ~isfield(parameters, 'distParams') || isempty(parameters.distParams)
    switch lower(parameters.distribution)
        case 't'
            parameters.distParams = 5;
        case 'ged'
            parameters.distParams = 1.5;
        case 'skewt'
            parameters.distParams = [5, -0.1];
    end
end

% Set default cross-correlation
if ~isfield(parameters, 'crossCorr') || isempty(parameters.crossCorr)
    parameters.crossCorr = eye(numSeries);
elseif size(parameters.crossCorr, 1) ~= numSeries || size(parameters.crossCorr, 2) ~= numSeries
    error('Cross-correlation matrix must be numSeries × numSeries');
end

% Set default AR/MA parameters
if ~isfield(parameters, 'arParameters') || isempty(parameters.arParameters)
    parameters.arParameters = [];
end

if ~isfield(parameters, 'maParameters') || isempty(parameters.maParameters)
    parameters.maParameters = [];
end

% Generate base innovations
innovations = zeros(numObservations, numSeries);
switch lower(parameters.distribution)
    case 'normal'
        innovations = randn(numObservations, numSeries);
    case 't'
        nu = parameters.distParams;
        innovations = stdtrnd([numObservations, numSeries], nu);
    case 'ged'
        nu = parameters.distParams;
        for i = 1:numSeries
            innovations(:,i) = gedrnd(nu, numObservations, 1);
        end
    case 'skewt'
        nu = parameters.distParams(1);
        lambda = parameters.distParams(2);
        for i = 1:numSeries
            innovations(:,i) = skewtrnd(nu, lambda, numObservations, 1);
        end
    otherwise
        error('Unsupported distribution: %s', parameters.distribution);
end

% Apply cross-correlation structure
cholCorr = chol(parameters.crossCorr, 'lower');
correlatedInnovations = innovations * cholCorr';

% Initialize returns matrix
returns = zeros(numObservations, numSeries);

% Apply GARCH volatility structure if specified
if isfield(parameters, 'garch') && ~isempty(parameters.garch)
    for i = 1:numSeries
        % Extract or use default GARCH parameters
        garchParams = parameters.garch;
        if ~isfield(garchParams, 'modelType') || isempty(garchParams.modelType)
            garchParams.modelType = 'GARCH';
        end
        if ~isfield(garchParams, 'p') || isempty(garchParams.p)
            garchParams.p = 1;
        end
        if ~isfield(garchParams, 'q') || isempty(garchParams.q)
            garchParams.q = 1;
        end
        if ~isfield(garchParams, 'omega') || isempty(garchParams.omega)
            garchParams.omega = 0.05;
        end
        if ~isfield(garchParams, 'alpha') || isempty(garchParams.alpha)
            garchParams.alpha = 0.1;
        end
        if ~isfield(garchParams, 'beta') || isempty(garchParams.beta)
            garchParams.beta = 0.85;
        end
        
        % Build GARCH model
        garchModel = struct();
        garchModel.modelType = garchParams.modelType;
        garchModel.p = garchParams.p;
        garchModel.q = garchParams.q;
        
        % Create parameter vector based on model type
        switch upper(garchParams.modelType)
            case 'GARCH'
                garchModel.parameters = [garchParams.omega; garchParams.alpha; garchParams.beta];
            case {'GJR', 'TARCH', 'EGARCH'}
                if ~isfield(garchParams, 'gamma') || isempty(garchParams.gamma)
                    garchParams.gamma = 0.1;
                end
                garchModel.parameters = [garchParams.omega; garchParams.alpha; garchParams.gamma; garchParams.beta];
            case {'AGARCH', 'NAGARCH'}
                if ~isfield(garchParams, 'gamma') || isempty(garchParams.gamma)
                    garchParams.gamma = 0.1;
                end
                garchModel.parameters = [garchParams.omega; garchParams.alpha; garchParams.gamma; garchParams.beta];
            case 'IGARCH'
                % Normalize parameters to ensure sum(alpha) + sum(beta) = 1
                totalParam = garchParams.alpha + garchParams.beta;
                alpha = garchParams.alpha / totalParam;
                beta = garchParams.beta / totalParam;
                garchModel.parameters = [garchParams.omega; alpha; beta];
        end
        
        % Set data and distribution
        garchModel.data = correlatedInnovations(:,i);
        garchModel.distribution = parameters.distribution;
        garchModel.distParams = parameters.distParams;
        
        % Generate preliminary ht and residuals to complete the model
        tempHt = garchfor(garchModel, 0).expectedVariances; % Just get initial values
        garchModel.ht = [tempHt; ones(numObservations-length(tempHt), 1) * tempHt(end)];
        garchModel.residuals = correlatedInnovations(:,i);
        
        % Generate volatility forecast for full series
        forecast = garchfor(garchModel, numObservations);
        
        % Apply volatility to returns
        volatility = sqrt(forecast.expectedVariances);
        returns(:,i) = correlatedInnovations(:,i) .* volatility;
    end
else
    % No GARCH structure, just apply variance scaling
    for i = 1:numSeries
        returns(:,i) = correlatedInnovations(:,i) * sqrt(parameters.variance(i));
    end
end

% Apply time series structure (AR/MA components) if specified
if ~isempty(parameters.arParameters) || ~isempty(parameters.maParameters)
    p = length(parameters.arParameters);
    q = length(parameters.maParameters);

    for i = 1:numSeries
        % Use armafor to apply AR/MA structure
        hasConstant = isfield(parameters, 'constant') && ~isempty(parameters.constant);
        if hasConstant
            constant = parameters.constant;
        else
            constant = 0;
        end
        
        % Set up parameters vector for armafor
        armaParams = [];
        if hasConstant
            armaParams = [constant; parameters.arParameters(:); parameters.maParameters(:)];
        else
            armaParams = [parameters.arParameters(:); parameters.maParameters(:)];
        end
        
        % Get ARMA forecasts with existing returns as innovations
        [filtered, ~] = armafor(armaParams, returns(:,i), p, q, hasConstant);
        returns(:,i) = filtered;
    end
end

% Add mean
for i = 1:numSeries
    returns(:,i) = returns(:,i) + parameters.mean(i);
end
end

%% Volatility Series Generation Function
function volData = generateVolatilitySeries(numObservations, modelType, parameters)
% GENERATEVOLATILITYSERIES Generates synthetic volatility series with known parameters for testing GARCH-type models
%
% USAGE:
%   VOLDATA = generateVolatilitySeries(NUMOBSERVATIONS, MODELTYPE, PARAMETERS)
%
% INPUTS:
%   NUMOBSERVATIONS - Number of observations to generate
%   MODELTYPE - String specifying model type: 'GARCH', 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'
%   PARAMETERS - Structure with fields:
%       omega - GARCH constant term [default: 0.05]
%       alpha - ARCH parameter(s) [default: 0.1]
%       beta - GARCH parameter(s) [default: 0.85]
%       gamma - Asymmetry parameter(s) (for asymmetric models) [default: 0.1]
%       p - GARCH order [default: 1]
%       q - ARCH order [default: 1]
%       distribution - Innovation distribution: 'normal', 't', 'ged', 'skewt' [default: 'normal']
%       distParams - Distribution parameters (model-specific)
%
% OUTPUTS:
%   VOLDATA - Structure containing returns, conditional variances, residuals, and true model parameters

% Validate inputs
parametercheck(numObservations, 'numObservations', struct('isInteger', true, 'isPositive', true));
modelType = upper(modelType);
validatestring(modelType, {'GARCH', 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'});

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set default GARCH parameters
if ~isfield(parameters, 'p') || isempty(parameters.p)
    parameters.p = 1;
end

if ~isfield(parameters, 'q') || isempty(parameters.q)
    parameters.q = 1;
end

if ~isfield(parameters, 'omega') || isempty(parameters.omega)
    parameters.omega = 0.05;
end

if ~isfield(parameters, 'alpha') || isempty(parameters.alpha)
    if parameters.q == 1
        parameters.alpha = 0.1;
    else
        parameters.alpha = repmat(0.1/parameters.q, parameters.q, 1);
    end
end

if ~isfield(parameters, 'beta') || isempty(parameters.beta)
    if parameters.p == 1
        parameters.beta = 0.85;
    else
        parameters.beta = repmat(0.85/parameters.p, parameters.p, 1);
    end
end

% Add asymmetry parameter for models that require it
if ismember(modelType, {'GJR', 'TARCH', 'EGARCH', 'AGARCH', 'NAGARCH'})
    if ~isfield(parameters, 'gamma') || isempty(parameters.gamma)
        if ismember(modelType, {'GJR', 'TARCH', 'EGARCH'})
            parameters.gamma = repmat(0.1, parameters.q, 1);
        else
            parameters.gamma = 0.1;
        end
    end
end

% Set innovation distribution
if ~isfield(parameters, 'distribution') || isempty(parameters.distribution)
    parameters.distribution = 'normal';
end

% Set distribution parameters
if ~isfield(parameters, 'distParams') || isempty(parameters.distParams)
    switch lower(parameters.distribution)
        case 't'
            parameters.distParams = 5;
        case 'ged'
            parameters.distParams = 1.5;
        case 'skewt'
            parameters.distParams = [5, -0.1];
    end
end

% Build parameter vector based on model type
switch modelType
    case 'GARCH'
        garchParams = [parameters.omega; parameters.alpha(:); parameters.beta(:)];
    case {'GJR', 'TARCH'}
        garchParams = [parameters.omega; parameters.alpha(:); parameters.gamma(:); parameters.beta(:)];
    case 'EGARCH'
        garchParams = [parameters.omega; parameters.alpha(:); parameters.gamma(:); parameters.beta(:)];
    case {'AGARCH', 'NAGARCH'}
        garchParams = [parameters.omega; parameters.alpha(:); parameters.gamma; parameters.beta(:)];
    case 'IGARCH'
        % Enforce constraint sum(alpha) + sum(beta) = 1
        totalAlpha = sum(parameters.alpha(:));
        totalBeta = sum(parameters.beta(:));
        scaleFactor = 1 / (totalAlpha + totalBeta);
        alphaScaled = parameters.alpha(:) * scaleFactor;
        betaScaled = parameters.beta(:) * scaleFactor;
        garchParams = [parameters.omega; alphaScaled; betaScaled];
        parameters.alpha = alphaScaled;
        parameters.beta = betaScaled;
end

% Generate innovations based on distribution
innovations = zeros(numObservations, 1);
switch lower(parameters.distribution)
    case 'normal'
        innovations = randn(numObservations, 1);
    case 't'
        innovations = stdtrnd(numObservations, parameters.distParams);
    case 'ged'
        innovations = gedrnd(parameters.distParams, numObservations, 1);
    case 'skewt'
        nu = parameters.distParams(1);
        lambda = parameters.distParams(2);
        innovations = skewtrnd(nu, lambda, numObservations, 1);
end

% Initialize arrays for returns and variances
returns = zeros(numObservations, 1);
ht = zeros(numObservations, 1);

% Set initial variance (backcast)
ht(1) = var(innovations);
returns(1) = sqrt(ht(1)) * innovations(1);

% Generate GARCH process
for t = 2:numObservations
    % Calculate conditional variance based on model type
    switch modelType
        case 'GARCH'
            ht(t) = parameters.omega;
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                ht(t) = ht(t) + parameters.alpha(alpha_idx) * returns(t-i)^2;
            end
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                ht(t) = ht(t) + parameters.beta(beta_idx) * ht(t-j);
            end
            
        case {'GJR', 'TARCH'}
            ht(t) = parameters.omega;
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                gamma_idx = min(i, length(parameters.gamma));
                archEffect = parameters.alpha(alpha_idx) * returns(t-i)^2;
                if returns(t-i) < 0
                    archEffect = archEffect + parameters.gamma(gamma_idx) * returns(t-i)^2;
                end
                ht(t) = ht(t) + archEffect;
            end
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                ht(t) = ht(t) + parameters.beta(beta_idx) * ht(t-j);
            end
            
        case 'EGARCH'
            logHt = parameters.omega;
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                gamma_idx = min(i, length(parameters.gamma));
                stdResid = returns(t-i) / sqrt(ht(t-i));
                absStdResid = abs(stdResid);
                logHt = logHt + parameters.alpha(alpha_idx) * (absStdResid - sqrt(2/pi));
                logHt = logHt + parameters.gamma(gamma_idx) * stdResid;
            end
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                logHt = logHt + parameters.beta(beta_idx) * log(max(ht(t-j), 1e-6));
            end
            ht(t) = exp(logHt);
            
        case 'AGARCH'
            ht(t) = parameters.omega;
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                ht(t) = ht(t) + parameters.alpha(alpha_idx) * (returns(t-i) - parameters.gamma)^2;
            end
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                ht(t) = ht(t) + parameters.beta(beta_idx) * ht(t-j);
            end
            
        case 'NAGARCH'
            ht(t) = parameters.omega;
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                ht(t) = ht(t) + parameters.beta(beta_idx) * ht(t-j);
            end
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                stdResid = returns(t-i) / sqrt(ht(t-i));
                ht(t) = ht(t) + parameters.alpha(alpha_idx) * ht(t-i) * (stdResid - parameters.gamma)^2;
            end
            
        case 'IGARCH'
            ht(t) = parameters.omega;
            for i = 1:min(t-1, parameters.q)
                alpha_idx = min(i, length(parameters.alpha));
                ht(t) = ht(t) + parameters.alpha(alpha_idx) * returns(t-i)^2;
            end
            for j = 1:min(t-1, parameters.p)
                beta_idx = min(j, length(parameters.beta));
                ht(t) = ht(t) + parameters.beta(beta_idx) * ht(t-j);
            end
    end
    
    % Ensure positive variance
    ht(t) = max(ht(t), 1e-6);
    
    % Generate return
    returns(t) = sqrt(ht(t)) * innovations(t);
end

% Build output structure
volData = struct();
volData.returns = returns;
volData.ht = ht;
volData.residuals = returns ./ sqrt(ht);
volData.parameters = parameters;
volData.modelType = modelType;
volData.p = parameters.p;
volData.q = parameters.q;
volData.distribution = parameters.distribution;
volData.distParams = parameters.distParams;
volData.innovations = innovations;
volData.paramVector = garchParams;

% Create model structure for direct use with garchfor
volData.model = struct();
volData.model.parameters = garchParams;
volData.model.modelType = modelType;
volData.model.p = parameters.p;
volData.model.q = parameters.q;
volData.model.data = returns;
volData.model.residuals = volData.residuals;
volData.model.ht = ht;
volData.model.distribution = parameters.distribution;
volData.model.distParams = parameters.distParams;
end

%% Distribution Samples Generation Function
function distData = generateDistributionSamples(distributionType, numSamples, parameters)
% GENERATEDISTRIBUTIONSAMPLES Generates random samples from various statistical distributions with known parameters
%
% USAGE:
%   DISTDATA = generateDistributionSamples(DISTRIBUTIONTYPE, NUMSAMPLES, PARAMETERS)
%
% INPUTS:
%   DISTRIBUTIONTYPE - String indicating distribution type: 'normal', 't', 'ged', 'skewt'
%   NUMSAMPLES - Number of random samples to generate
%   PARAMETERS - Structure with distribution-specific parameters:
%                For 'normal': mu [default: 0], sigma [default: 1]
%                For 't': nu (degrees of freedom) [default: 5]
%                For 'ged': nu (shape parameter) [default: 1.5]
%                For 'skewt': nu (degrees of freedom) [default: 5], 
%                            lambda (skewness) [default: -0.1]
%
% OUTPUTS:
%   DISTDATA - Structure containing samples, parameters, and theoretical/empirical moments

% Validate inputs
parametercheck(numSamples, 'numSamples', struct('isInteger', true, 'isPositive', true));
distributionType = lower(distributionType);
validatestring(distributionType, {'normal', 't', 'ged', 'skewt'});

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Generate samples based on distribution type
switch distributionType
    case 'normal'
        % Set default parameters
        if ~isfield(parameters, 'mu') || isempty(parameters.mu)
            parameters.mu = 0;
        end
        if ~isfield(parameters, 'sigma') || isempty(parameters.sigma)
            parameters.sigma = 1;
        end
        
        % Generate normal samples
        samples = parameters.mu + parameters.sigma * randn(numSamples, 1);
        
        % Compute theoretical moments
        theoreticalMoments = struct();
        theoreticalMoments.mean = parameters.mu;
        theoreticalMoments.variance = parameters.sigma^2;
        theoreticalMoments.skewness = 0;
        theoreticalMoments.kurtosis = 3;
        
    case 't'
        % Set default parameters
        if ~isfield(parameters, 'nu') || isempty(parameters.nu)
            parameters.nu = 5;
        end
        parametercheck(parameters.nu, 'nu', struct('isPositive', true));
        
        % Set location and scale if provided
        if ~isfield(parameters, 'mu')
            parameters.mu = 0;
        end
        if ~isfield(parameters, 'sigma')
            parameters.sigma = 1;
        end
        
        % Generate Student's t samples
        if parameters.nu > 2
            samples = stdtrnd(numSamples, parameters.nu);
        else
            samples = stdtrnd(numSamples, parameters.nu);
        end
        
        % Apply location and scale
        samples = parameters.mu + parameters.sigma * samples;
        
        % Compute theoretical moments
        theoreticalMoments = struct();
        theoreticalMoments.mean = parameters.mu;
        
        if parameters.nu > 2
            theoreticalMoments.variance = parameters.sigma^2 * parameters.nu/(parameters.nu-2);
        else
            theoreticalMoments.variance = Inf;
        end
        
        if parameters.nu > 3
            theoreticalMoments.skewness = 0;
        else
            theoreticalMoments.skewness = NaN;
        end
        
        if parameters.nu > 4
            theoreticalMoments.kurtosis = 3 + 6/(parameters.nu-4);
        else
            theoreticalMoments.kurtosis = Inf;
        end
        
    case 'ged'
        % Set default parameters
        if ~isfield(parameters, 'nu') || isempty(parameters.nu)
            parameters.nu = 1.5;
        end
        parametercheck(parameters.nu, 'nu', struct('isPositive', true));
        
        % Set location and scale if provided
        if ~isfield(parameters, 'mu')
            parameters.mu = 0;
        end
        if ~isfield(parameters, 'sigma')
            parameters.sigma = 1;
        end
        
        % Generate GED samples
        samples = gedrnd(parameters.nu, numSamples, 1);
        
        % Apply location and scale
        samples = parameters.mu + parameters.sigma * samples;
        
        % Compute theoretical moments
        theoreticalMoments = struct();
        theoreticalMoments.mean = parameters.mu;
        theoreticalMoments.variance = parameters.sigma^2;
        theoreticalMoments.skewness = 0;
        
        % Kurtosis for GED
        theoreticalMoments.kurtosis = gamma(5/parameters.nu) * gamma(1/parameters.nu) / (gamma(3/parameters.nu)^2);
        
    case 'skewt'
        % Set default parameters
        if ~isfield(parameters, 'nu') || isempty(parameters.nu)
            parameters.nu = 5;
        end
        if ~isfield(parameters, 'lambda') || isempty(parameters.lambda)
            parameters.lambda = -0.1;
        end
        
        parametercheck(parameters.nu, 'nu', struct('isPositive', true));
        parametercheck(parameters.lambda, 'lambda', struct('lowerBound', -1, 'upperBound', 1));
        
        % Set location and scale if provided
        if ~isfield(parameters, 'mu')
            parameters.mu = 0;
        end
        if ~isfield(parameters, 'sigma')
            parameters.sigma = 1;
        end
        
        % Generate skewed t samples
        samples = skewtrnd(parameters.nu, parameters.lambda, numSamples, 1);
        
        % Apply location and scale
        samples = parameters.mu + parameters.sigma * samples;
        
        % Compute theoretical moments (approximations for skewed t)
        theoreticalMoments = struct();
        theoreticalMoments.mean = parameters.mu;
        
        % Compute approximate moments
        nu = parameters.nu;
        lambda = parameters.lambda;
        
        if nu > 2
            theoreticalMoments.variance = parameters.sigma^2;
        else
            theoreticalMoments.variance = Inf;
        end
        
        if nu > 3
            % Approximate skewness
            theoreticalMoments.skewness = 4 * lambda * (nu-2) / (nu-1) * 0.5;
        else
            theoreticalMoments.skewness = NaN;
        end
        
        if nu > 4
            % Approximate kurtosis
            base_kurtosis = 3 + 6/(nu-4);
            lambda_effect = 1 + lambda^2;
            theoreticalMoments.kurtosis = base_kurtosis * lambda_effect;
        else
            theoreticalMoments.kurtosis = Inf;
        end
end

% Compute empirical moments
empiricalMoments = struct();
empiricalMoments.mean = mean(samples);
empiricalMoments.variance = var(samples);
empiricalMoments.skewness = computeSkewness(samples);
empiricalMoments.kurtosis = computeKurtosis(samples);

% Create output structure
distData = struct();
distData.samples = samples;
distData.parameters = parameters;
distData.distributionType = distributionType;
distData.theoreticalMoments = theoreticalMoments;
distData.empiricalMoments = empiricalMoments;
end

%% High-Frequency Data Generation Function
function hfData = generateHighFrequencyData(numDays, observationsPerDay, parameters)
% GENERATEHIGHFREQUENCYDATA Generates synthetic high-frequency financial data for testing realized volatility measures
%
% USAGE:
%   HFDATA = generateHighFrequencyData(NUMDAYS, OBSERVATIONSPERDAY, PARAMETERS)
%
% INPUTS:
%   NUMDAYS - Number of days to simulate
%   OBSERVATIONSPERDAY - Number of intraday observations per day
%   PARAMETERS - Structure with fields for volatility model, intraday pattern,
%                jump process, microstructure noise, and distribution

% Validate inputs
parametercheck(numDays, 'numDays', struct('isInteger', true, 'isPositive', true));
parametercheck(observationsPerDay, 'observationsPerDay', struct('isInteger', true, 'isPositive', true));

% Set default parameters if not provided
if nargin < 3 || isempty(parameters)
    parameters = struct();
end

% Set default volatility model
if ~isfield(parameters, 'volatilityModel') || isempty(parameters.volatilityModel)
    parameters.volatilityModel = 'constant';
end

% Set default volatility parameters
if ~isfield(parameters, 'volatilityParams') || isempty(parameters.volatilityParams)
    switch lower(parameters.volatilityModel)
        case 'constant'
            parameters.volatilityParams = struct('sigma', 0.01);
        case 'garch'
            parameters.volatilityParams = struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
        case 'sinusoidal'
            parameters.volatilityParams = struct('amplitude', 0.005, 'period', 20, 'phase', 0);
    end
end

% Set default intraday pattern
if ~isfield(parameters, 'intradayPattern') || isempty(parameters.intradayPattern)
    parameters.intradayPattern = 'flat';
end

% Set default jump process
if ~isfield(parameters, 'jumpProcess') || isempty(parameters.jumpProcess)
    parameters.jumpProcess = 'none';
end

% Set default microstructure noise
if ~isfield(parameters, 'microstructure') || isempty(parameters.microstructure)
    parameters.microstructure = 'none';
end

% Set default distribution
if ~isfield(parameters, 'distribution') || isempty(parameters.distribution)
    parameters.distribution = 'normal';
end

% Total number of observations
totalObs = numDays * observationsPerDay;

% Generate day and intraday indices
dayIndex = ceil((1:totalObs) / observationsPerDay)';
intraDayIndex = repmat((1:observationsPerDay)', numDays, 1);

% Generate timestamps
timeStep = 1/observationsPerDay;
timestamps = ((1:totalObs) - 1) * timeStep;

% 1. Generate daily volatility process
dailyVolatility = generateDailyVolatility(numDays, parameters);

% 2. Generate intraday volatility pattern
intradayVolatility = generateIntradayPattern(observationsPerDay, parameters);

% 3. Combine daily and intraday volatility
totalVolatility = zeros(totalObs, 1);
for i = 1:totalObs
    day = dayIndex(i);
    intraday = intraDayIndex(i);
    totalVolatility(i) = dailyVolatility(day) * intradayVolatility(intraday);
end

% 4. Generate base returns
innovations = generateDistributionForHF(totalObs, parameters);
baseReturns = innovations .* totalVolatility;

% 5. Add jumps if specified
jumpReturns = zeros(totalObs, 1);
jumpTimes = [];
jumpSizes = [];
if strcmpi(parameters.jumpProcess, 'poisson')
    if ~isfield(parameters, 'jumpParams') || isempty(parameters.jumpParams)
        parameters.jumpParams = struct('intensity', 0.1, 'jumpSize', [0, 0.01]);
    end
    [jumpReturns, jumpTimes, jumpSizes] = addJumps(totalObs, numDays, parameters.jumpParams);
end

% True returns (without microstructure noise)
trueReturns = baseReturns + jumpReturns;

% 6. Add microstructure noise if specified
observedReturns = trueReturns;
noise = zeros(totalObs, 1);
if ~strcmpi(parameters.microstructure, 'none')
    if ~isfield(parameters, 'microstructureParams') || isempty(parameters.microstructureParams)
        switch lower(parameters.microstructure)
            case 'additive'
                parameters.microstructureParams = struct('std', 0.0005);
            case 'bid-ask'
                parameters.microstructureParams = struct('spread', 0.001);
        end
    end
    [observedReturns, noise] = addMicrostructureNoise(trueReturns, parameters.microstructure, parameters.microstructureParams);
end

% Generate price path from returns (starting at 100)
prices = 100 * cumprod(1 + observedReturns);
prices = [100; prices];

% Create output structure
hfData = struct();
hfData.returns = observedReturns;
hfData.timestamps = timestamps;
hfData.dayIndex = dayIndex;
hfData.intraDayIndex = intraDayIndex;
hfData.dailyVolatility = dailyVolatility;
hfData.intradayVolatility = intradayVolatility;
hfData.totalVolatility = totalVolatility;
hfData.parameters = parameters;
hfData.jumpTimes = jumpTimes;
hfData.jumpSizes = jumpSizes;
hfData.trueReturns = trueReturns;
hfData.noise = noise;
hfData.prices = prices;
end

%% Cross-Sectional Data Generation Function
function csData = generateCrossSectionalData(numAssets, numPeriods, numFactors, parameters)
% GENERATECROSSSECTIONALDATA Generates cross-sectional financial data for testing cross-sectional analysis tools
%
% USAGE:
%   CSDATA = generateCrossSectionalData(NUMASSETS, NUMPERIODS, NUMFACTORS, PARAMETERS)
%
% INPUTS:
%   NUMASSETS - Number of assets in cross-section
%   NUMPERIODS - Number of time periods
%   NUMFACTORS - Number of factors in the factor model
%   PARAMETERS - Structure with fields for factor means, covariance, loadings, idiosyncratic
%                volatility, and firm characteristics

% Validate inputs
parametercheck(numAssets, 'numAssets', struct('isInteger', true, 'isPositive', true));
parametercheck(numPeriods, 'numPeriods', struct('isInteger', true, 'isPositive', true));
parametercheck(numFactors, 'numFactors', struct('isInteger', true, 'isPositive', true));

% Set default parameters
if nargin < 4 || isempty(parameters)
    parameters = struct();
end

% Set default factor means
if ~isfield(parameters, 'factorMean') || isempty(parameters.factorMean)
    parameters.factorMean = zeros(numFactors, 1);
elseif length(parameters.factorMean) ~= numFactors
    error('factorMean must be a vector of length numFactors');
end

% Set default factor covariance
if ~isfield(parameters, 'factorCov') || isempty(parameters.factorCov)
    parameters.factorCov = eye(numFactors);
elseif size(parameters.factorCov, 1) ~= numFactors || size(parameters.factorCov, 2) ~= numFactors
    error('factorCov must be a matrix of size numFactors x numFactors');
end

% Set default factor model
if ~isfield(parameters, 'factorModel') || isempty(parameters.factorModel)
    parameters.factorModel = 'normal';
end

% Set default loading distribution
if ~isfield(parameters, 'loadingDist') || isempty(parameters.loadingDist)
    parameters.loadingDist = 'normal';
end

% Set default loading parameters
if ~isfield(parameters, 'loadingParams') || isempty(parameters.loadingParams)
    switch lower(parameters.loadingDist)
        case 'normal'
            parameters.loadingParams = struct('mean', 1.0, 'std', 0.3);
        case 'uniform'
            parameters.loadingParams = struct('min', 0.5, 'max', 1.5);
    end
end

% Set default idiosyncratic volatility
if ~isfield(parameters, 'idioVol') || isempty(parameters.idioVol)
    parameters.idioVol = 0.02;
end

% Set default idiosyncratic correlation
if ~isfield(parameters, 'idioCorr') || isempty(parameters.idioCorr)
    parameters.idioCorr = 0;
end

% Handle characteristics if specified
characteristics = [];
if isfield(parameters, 'characteristics') && ~isempty(parameters.characteristics)
    if ~isfield(parameters.characteristics, 'numChars') || isempty(parameters.characteristics.numChars)
        parameters.characteristics.numChars = 0;
    end
    
    numChars = parameters.characteristics.numChars;
    if numChars > 0
        % Set default characteristics parameters
        if ~isfield(parameters.characteristics, 'mean') || isempty(parameters.characteristics.mean)
            parameters.characteristics.mean = zeros(numChars, 1);
        end
        
        if ~isfield(parameters.characteristics, 'correlation') || isempty(parameters.characteristics.correlation)
            parameters.characteristics.correlation = eye(numChars);
        end
        
        if ~isfield(parameters.characteristics, 'loadingRelevance') || isempty(parameters.characteristics.loadingRelevance)
            parameters.characteristics.loadingRelevance = zeros(numChars, numFactors);
        end
        
        % Generate characteristics
        characteristics = mvnrnd(parameters.characteristics.mean(:)', ...
                                parameters.characteristics.correlation, numAssets);
    end
end

% 1. Generate factor returns
factors = generateFactorReturnsForCS(numPeriods, numFactors, parameters);

% 2. Generate factor loadings
loadings = generateFactorLoadingsForCS(numAssets, numFactors, parameters, characteristics);

% 3. Generate idiosyncratic returns
idioReturns = generateIdiosyncraticReturnsForCS(numPeriods, numAssets, parameters);

% 4. Combine to create asset returns
% r = beta * f' + epsilon
assetReturns = (loadings * factors')' + idioReturns;

% 5. Generate risk premiums
if isfield(parameters, 'riskPremium') && ~isempty(parameters.riskPremium)
    riskPremium = parameters.riskPremium(:);
else
    % Default risk premium proportional to factor volatility
    factorVol = sqrt(diag(cov(factors)));
    riskPremium = 0.5 * factorVol;
end

% Create output structure
csData = struct();
csData.returns = assetReturns;
csData.factors = factors;
csData.loadings = loadings;
csData.idioReturns = idioReturns;
csData.parameters = parameters;
csData.trueRiskPremium = riskPremium;
csData.trueBetas = loadings;

% Add characteristics if generated
if ~isempty(characteristics)
    csData.characteristics = characteristics;
end
end

%% Data Storage Functions
function success = saveTestData(data, fileName, options)
% SAVETESTDATA Saves generated test data to MAT files in the test data directory
%
% USAGE:
%   SUCCESS = saveTestData(DATA, FILENAME, OPTIONS)
%
% INPUTS:
%   DATA - Structure containing data to save
%   FILENAME - Name of the file to save (with or without .mat extension)
%   OPTIONS - [OPTIONAL] Structure with fields for overwrite, metadata, compression options

% Validate inputs
if ~isstruct(data)
    error('DATA must be a structure');
end

if ~ischar(fileName)
    error('FILENAME must be a string');
end

% Set default options
if nargin < 3 || isempty(options)
    options = struct();
end

if ~isfield(options, 'overwrite') || isempty(options.overwrite)
    options.overwrite = false;
end

if ~isfield(options, 'addMetadata') || isempty(options.addMetadata)
    options.addMetadata = true;
end

if ~isfield(options, 'compress') || isempty(options.compress)
    options.compress = true;
end

% Get global data path
global TEST_DATA_PATH;
if isempty(TEST_DATA_PATH)
    TEST_DATA_PATH = '../data/';
end

% Ensure file has .mat extension
if ~endsWith(fileName, '.mat')
    fileName = [fileName, '.mat'];
end

% Construct full file path
filePath = fullfile(TEST_DATA_PATH, fileName);

% Check if directory exists, create if not
if ~exist(TEST_DATA_PATH, 'dir')
    [status, msg] = mkdir(TEST_DATA_PATH);
    if ~status
        error('Failed to create data directory: %s', msg);
    end
end

% Check if file exists and handle overwrite
if exist(filePath, 'file') && ~options.overwrite
    warning('File %s already exists. Use options.overwrite=true to overwrite.', filePath);
    success = false;
    return;
end

% Add metadata if requested
if options.addMetadata
    % Add generation timestamp
    data.metadata = struct();
    data.metadata.generationTime = datestr(now);
    data.metadata.generationDate = date;
    
    % Add version information
    if isfield(data, 'parameters')
        data.metadata.parameters = data.parameters;
    end
end

% Save the data
try
    if options.compress
        save(filePath, '-struct', 'data', '-v7.3');
    else
        save(filePath, '-struct', 'data', '-v7.3', '-nocompression');
    end
    success = true;
catch ME
    warning('Failed to save file: %s', ME.message);
    success = false;
end
end

function data = loadTestData(fileName)
% LOADTESTDATA Loads test data from MAT files in the test data directory
%
% USAGE:
%   DATA = loadTestData(FILENAME)
%
% INPUTS:
%   FILENAME - Name of the file to load (with or without .mat extension)
%
% OUTPUTS:
%   DATA - Loaded data structure

% Validate input
if ~ischar(fileName)
    error('FILENAME must be a string');
end

% Get global data path
global TEST_DATA_PATH;
if isempty(TEST_DATA_PATH)
    TEST_DATA_PATH = '../data/';
end

% Ensure file has .mat extension
if ~endsWith(fileName, '.mat')
    fileName = [fileName, '.mat'];
end

% Construct full file path
filePath = fullfile(TEST_DATA_PATH, fileName);

% Check if file exists
if ~exist(filePath, 'file')
    error('File %s does not exist', filePath);
end

% Load the data
try
    data = load(filePath);
catch ME
    error('Failed to load file: %s', ME.message);
end

% Verify data integrity
if ~isstruct(data)
    warning('Loaded data is not a structure. This may not be a valid test data file.');
end
end

%% Main Test Data Generation Function
function success = generateAllTestData(overwrite)
% GENERATEALLTESTDATA Generates all standard test datasets for the MFE Toolbox test suite
%
% USAGE:
%   SUCCESS = generateAllTestData(OVERWRITE)
%
% INPUTS:
%   OVERWRITE - [OPTIONAL] Boolean indicating whether to overwrite existing files (default: false)
%
% OUTPUTS:
%   SUCCESS - Boolean indicating if all data was generated successfully

% Set default overwrite flag
if nargin < 1 || isempty(overwrite)
    overwrite = false;
end

% Set random seed for reproducibility
rng(20231201, 'twister');

% Initialize success flag
success = true;

try
    % 1. Generate financial returns data
    disp('Generating financial returns test data...');
    
    % Set parameters for financial returns
    returnParams = struct();
    returnParams.mean = [0, 0.0005, 0.001];
    returnParams.variance = [1, 0.5, 2];
    returnParams.distribution = 't';
    returnParams.distParams = 5;
    returnParams.acf = [0.1, 0.05];
    returnParams.crossCorr = [1, 0.3, 0.2; 0.3, 1, 0.5; 0.2, 0.5, 1];
    
    % GARCH parameters for volatility clustering
    returnParams.garch = struct();
    returnParams.garch.modelType = 'GARCH';
    returnParams.garch.p = 1;
    returnParams.garch.q = 1;
    returnParams.garch.omega = 0.05;
    returnParams.garch.alpha = 0.1;
    returnParams.garch.beta = 0.85;
    
    % Generate returns
    returns = generateFinancialReturns(1000, 3, returnParams);
    
    % Save financial returns data
    financialData = struct('returns', returns, 'parameters', returnParams);
    if ~saveTestData(financialData, 'financial_returns.mat', struct('overwrite', overwrite))
        warning('Failed to save financial returns data');
        success = false;
    end
    
    % 2. Generate volatility model data
    disp('Generating volatility model test data...');
    
    % Generate different GARCH model types
    volData = struct();
    
    % GARCH(1,1)
    garchParams = struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
    volData.garch = generateVolatilitySeries(1000, 'GARCH', garchParams);
    
    % GJR-GARCH(1,1)
    gjrParams = struct('omega', 0.05, 'alpha', 0.05, 'gamma', 0.1, 'beta', 0.85);
    volData.gjr = generateVolatilitySeries(1000, 'GJR', gjrParams);
    
    % EGARCH(1,1)
    egarchParams = struct('omega', -0.1, 'alpha', 0.1, 'gamma', -0.05, 'beta', 0.95);
    volData.egarch = generateVolatilitySeries(1000, 'EGARCH', egarchParams);
    
    % Save volatility model data
    if ~saveTestData(volData, 'volatility_models.mat', struct('overwrite', overwrite))
        warning('Failed to save volatility model data');
        success = false;
    end
    
    % 3. Generate distribution samples data
    disp('Generating distribution samples test data...');
    
    distData = struct();
    
    % Normal distribution
    normalParams = struct('mu', 0, 'sigma', 1);
    distData.normal = generateDistributionSamples('normal', 1000, normalParams);
    
    % Student's t distribution
    tParams = struct('nu', 5);
    distData.t = generateDistributionSamples('t', 1000, tParams);
    
    % GED distribution
    gedParams = struct('nu', 1.5);
    distData.ged = generateDistributionSamples('ged', 1000, gedParams);
    
    % Skewed t distribution
    skewtParams = struct('nu', 5, 'lambda', -0.2);
    distData.skewt = generateDistributionSamples('skewt', 1000, skewtParams);
    
    % Save distribution samples data
    if ~saveTestData(distData, 'known_distributions.mat', struct('overwrite', overwrite))
        warning('Failed to save distribution samples data');
        success = false;
    end
    
    % 4. Generate high-frequency data
    disp('Generating high-frequency data...');
    
    hfParams = struct();
    hfParams.volatilityModel = 'garch';
    hfParams.volatilityParams = struct('modelType', 'GARCH', 'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
    hfParams.intradayPattern = 'U-shape';
    hfParams.jumpProcess = 'poisson';
    hfParams.jumpParams = struct('intensity', 0.1, 'jumpSize', [0, 0.005]);
    hfParams.microstructure = 'additive';
    hfParams.microstructureParams = struct('std', 0.0002);
    
    hfData = generateHighFrequencyData(5, 78, hfParams);
    
    if ~saveTestData(hfData, 'high_frequency_data.mat', struct('overwrite', overwrite))
        warning('Failed to save high-frequency data');
        success = false;
    end
    
    % 5. Generate cross-sectional data
    disp('Generating cross-sectional data...');
    
    csParams = struct();
    csParams.factorMean = [0.001, 0.0005, 0.0008];
    csParams.factorCov = [0.04, 0.01, 0.005; 0.01, 0.02, 0.003; 0.005, 0.003, 0.03];
    csParams.factorModel = 'normal';
    csParams.loadingDist = 'normal';
    csParams.loadingParams = struct('mean', 1.0, 'std', 0.3);
    csParams.idioVol = 0.02;
    csParams.idioCorr = 0.1;
    
    % Add firm characteristics
    csParams.characteristics = struct();
    csParams.characteristics.numChars = 3;
    csParams.characteristics.mean = [0, 0, 0];
    csParams.characteristics.correlation = [1, 0.2, -0.1; 0.2, 1, 0.3; -0.1, 0.3, 1];
    csParams.characteristics.loadingRelevance = [0.2, 0.1, 0; 0, 0.2, 0.1; 0.1, 0, 0.2];
    
    csData = generateCrossSectionalData(100, 60, 3, csParams);
    
    if ~saveTestData(csData, 'cross_sectional_data.mat', struct('overwrite', overwrite))
        warning('Failed to save cross-sectional data');
        success = false;
    end
    
    % 6. Generate time series data
    disp('Generating time series test data...');
    
    tsParams = struct();
    tsParams.ar = [0.8, -0.2];
    tsParams.ma = [0.3, 0.1];
    tsParams.constant = 0.001;
    tsParams.exogenousVars = true;
    tsParams.exogenousCoef = [0.5, -0.3];
    tsParams.numExoVars = 2;
    tsParams.distribution = 'normal';
    
    tsData = generateTimeSeriesData(tsParams);
    
    if ~saveTestData(tsData, 'time_series_data.mat', struct('overwrite', overwrite))
        warning('Failed to save time series data');
        success = false;
    end
    
    % 7. Generate bootstrap test data
    disp('Generating bootstrap test data...');
    
    bootParams = struct();
    bootParams.timeSeriesLength = 500;
    bootParams.blockSizes = [5, 10, 20];
    bootParams.dependenceType = 'ar';
    bootParams.dependenceParams = struct('ar', 0.7);
    
    bootData = generateBootstrapTestData(bootParams);
    
    if ~saveTestData(bootData, 'bootstrap_data.mat', struct('overwrite', overwrite))
        warning('Failed to save bootstrap test data');
        success = false;
    end
    
    % Success message
    if success
        disp('All test data generated successfully!');
    else
        disp('Test data generation completed with some errors.');
    end
    
catch ME
    warning('Error during test data generation: %s', ME.message);
    success = false;
end
end

%% Time Series Data Generation Function
function tsData = generateTimeSeriesData(parameters)
% GENERATETIMESERIESDATA Generates test data specifically for ARMA/ARMAX model testing
%
% USAGE:
%   TSDATA = generateTimeSeriesData(PARAMETERS)
%
% INPUTS:
%   PARAMETERS - Structure with fields for AR parameters, MA parameters, constant term,
%                exogenous variables, distribution, and structural breaks
%
% OUTPUTS:
%   TSDATA - Structure containing time series data with known ARMA properties

% Set default parameters if not provided
if nargin < 1 || isempty(parameters)
    parameters = struct();
end

% Set number of observations
if ~isfield(parameters, 'numObs') || isempty(parameters.numObs)
    parameters.numObs = 1000;
end
numObs = parameters.numObs;

% Set AR parameters
if ~isfield(parameters, 'ar') || isempty(parameters.ar)
    parameters.ar = [];
end
arParams = parameters.ar(:);
arOrder = length(arParams);

% Set MA parameters
if ~isfield(parameters, 'ma') || isempty(parameters.ma)
    parameters.ma = [];
end
maParams = parameters.ma(:);
maOrder = length(maParams);

% Set constant term
if ~isfield(parameters, 'constant') || isempty(parameters.constant)
    parameters.constant = 0;
end
constant = parameters.constant;

% Generate exogenous variables if requested
x = [];
exogenousCoef = [];
if isfield(parameters, 'exogenousVars') && parameters.exogenousVars
    % Check for exogenous coefficients
    if ~isfield(parameters, 'exogenousCoef') || isempty(parameters.exogenousCoef)
        error('exogenousCoef must be provided when exogenousVars=true');
    end
    exogenousCoef = parameters.exogenousCoef(:);
    
    % Set number of exogenous variables
    if ~isfield(parameters, 'numExoVars') || isempty(parameters.numExoVars)
        parameters.numExoVars = length(exogenousCoef);
    end
    numExoVars = parameters.numExoVars;
    
    % Set AR coefficient for generating exogenous variables
    if ~isfield(parameters, 'exogenousAR') || isempty(parameters.exogenousAR)
        parameters.exogenousAR = 0.7;
    end
    exogenousAR = parameters.exogenousAR;
    
    % Generate exogenous variables with AR(1) process
    x = zeros(numObs, numExoVars);
    for i = 1:numExoVars
        % Initialize with standard normal
        x(1, i) = randn();
        
        % Generate AR(1) process
        for t = 2:numObs
            x(t, i) = exogenousAR * x(t-1, i) + 0.5 * randn();
        end
        
        % Standardize to have unit variance
        x(:, i) = x(:, i) / std(x(:, i));
    end
end

% Set distribution
if ~isfield(parameters, 'distribution') || isempty(parameters.distribution)
    parameters.distribution = 'normal';
end

% Generate innovations based on distribution
innovations = zeros(numObs, 1);
switch lower(parameters.distribution)
    case 'normal'
        innovations = randn(numObs, 1);
    case 't'
        if ~isfield(parameters, 'distParams') || isempty(parameters.distParams)
            parameters.distParams = 5;
        end
        innovations = stdtrnd(numObs, parameters.distParams);
    case 'ged'
        if ~isfield(parameters, 'distParams') || isempty(parameters.distParams)
            parameters.distParams = 1.5;
        end
        innovations = gedrnd(parameters.distParams, numObs, 1);
    case 'skewt'
        if ~isfield(parameters, 'distParams') || isempty(parameters.distParams)
            parameters.distParams = [5, -0.1];
        end
        nu = parameters.distParams(1);
        lambda = parameters.distParams(2);
        innovations = skewtrnd(nu, lambda, numObs, 1);
end

% Initialize y series
y = zeros(numObs, 1);

% Check for structural breaks
hasBreaks = isfield(parameters, 'breaks') && ~isempty(parameters.breaks) && ...
           isfield(parameters.breaks, 'locations') && ~isempty(parameters.breaks.locations);

if hasBreaks
    % Validate break specification
    if ~isfield(parameters.breaks, 'parameters') || isempty(parameters.breaks.parameters)
        error('parameters.breaks.parameters must be provided for structural breaks');
    end
    
    breakLocations = parameters.breaks.locations;
    breakParameters = parameters.breaks.parameters;
    
    if length(breakLocations) + 1 ~= length(breakParameters)
        error('Number of parameter regimes must be one more than number of breaks');
    end
    
    % Generate data for each regime
    regimes = [1, breakLocations, numObs+1];
    for r = 1:(length(regimes)-1)
        startIdx = regimes(r);
        endIdx = regimes(r+1) - 1;
        
        % Get parameters for this regime
        regimeParams = breakParameters{r};
        
        % Generate data for this regime
        segmentLength = endIdx - startIdx + 1;
        segment = generateARMASegment(segmentLength, regimeParams.ar, regimeParams.ma, ...
                                    regimeParams.constant, innovations(startIdx:endIdx), ...
                                    isfield(parameters, 'exogenousVars') && parameters.exogenousVars, ...
                                    x(startIdx:endIdx,:), regimeParams.exogenousCoef);
        
        % Insert into y series
        y(startIdx:endIdx) = segment;
    end
else
    % No breaks, generate data for the whole sample
    y = generateARMASegment(numObs, arParams, maParams, constant, innovations, ...
                          isfield(parameters, 'exogenousVars') && parameters.exogenousVars, ...
                          x, exogenousCoef);
end

% Build output structure
tsData = struct();
tsData.y = y;
if ~isempty(x)
    tsData.x = x;
end
tsData.innovations = innovations;
tsData.parameters = parameters;
if hasBreaks
    tsData.breaks = parameters.breaks;
end

% Add true parameters for reference
tsData.trueParameters = struct();
tsData.trueParameters.constant = constant;
tsData.trueParameters.ar = arParams;
tsData.trueParameters.ma = maParams;
if ~isempty(x)
    tsData.trueParameters.exogenousCoef = exogenousCoef;
end
end

%% Bootstrap Test Data Generation Function
function bootData = generateBootstrapTestData(parameters)
% GENERATEBOOTSTRAPTESTDATA Generates test data specifically for bootstrap method validation
%
% USAGE:
%   BOOTDATA = generateBootstrapTestData(PARAMETERS)
%
% INPUTS:
%   PARAMETERS - Structure with fields for time series length, block sizes,
%                dependence type, and other bootstrap-specific parameters
%
% OUTPUTS:
%   BOOTDATA - Structure containing data for bootstrap method validation

% Set default parameters if not provided
if nargin < 1 || isempty(parameters)
    parameters = struct();
end

% Set time series length
if ~isfield(parameters, 'timeSeriesLength') || isempty(parameters.timeSeriesLength)
    parameters.timeSeriesLength = 500;
end
T = parameters.timeSeriesLength;

% Set block sizes
if ~isfield(parameters, 'blockSizes') || isempty(parameters.blockSizes)
    parameters.blockSizes = [5, 10, 20];
end
blockSizes = parameters.blockSizes;

% Set dependence type
if ~isfield(parameters, 'dependenceType') || isempty(parameters.dependenceType)
    parameters.dependenceType = 'ar';
end
dependenceType = parameters.dependenceType;

% Set dependence parameters
if ~isfield(parameters, 'dependenceParams') || isempty(parameters.dependenceParams)
    switch lower(dependenceType)
        case 'none'
            parameters.dependenceParams = struct();
        case 'ar'
            parameters.dependenceParams = struct('ar', 0.7);
        case 'garch'
            parameters.dependenceParams = struct('omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
        case 'nonlinear'
            parameters.dependenceParams = struct('ar', 0.3, 'nonlinear', 0.5);
    end
end

% Initialize data structure
data = zeros(T, 4);
dependenceProperties = cell(4, 1);

% Generate independent series
data(:, 1) = randn(T, 1);
dependenceProperties{1} = struct('type', 'none', 'params', []);

% Generate series based on dependence type
switch lower(dependenceType)
    case 'ar'
        % AR(1) process
        arCoef = parameters.dependenceParams.ar;
        data(1, 2) = randn();
        for t = 2:T
            data(t, 2) = arCoef * data(t-1, 2) + randn();
        end
        dependenceProperties{2} = struct('type', 'AR', 'params', arCoef);
        
        % Optimal block size for AR process
        optimalBlockSize = ceil(1.5 * (2 * arCoef / (1 - arCoef^2))^(2/3) * T^(1/3));
        
    case 'garch'
        % GARCH process
        omega = parameters.dependenceParams.omega;
        alpha = parameters.dependenceParams.alpha;
        beta = parameters.dependenceParams.beta;
        
        % Initialize
        variance = omega / (1 - alpha - beta);  % Unconditional variance
        data(1, 2) = sqrt(variance) * randn();
        
        % Generate GARCH(1,1) process
        for t = 2:T
            variance = omega + alpha * data(t-1, 2)^2 + beta * variance;
            data(t, 2) = sqrt(variance) * randn();
        end
        dependenceProperties{2} = struct('type', 'GARCH', 'params', [omega, alpha, beta]);
        
        % Optimal block size for GARCH - approximate
        effectiveAR = alpha + beta;
        optimalBlockSize = ceil(1.5 * (2 * effectiveAR / (1 - effectiveAR^2))^(2/3) * T^(1/3));
        
    case 'nonlinear'
        % Nonlinear AR process
        arCoef = parameters.dependenceParams.ar;
        nonlinCoef = parameters.dependenceParams.nonlinear;
        
        data(1, 2) = randn();
        for t = 2:T
            % Nonlinear AR: y_t = ar*y_{t-1} + nonlinear*y_{t-1}^2 + e_t
            data(t, 2) = arCoef * data(t-1, 2) + nonlinCoef * sign(data(t-1, 2)) * data(t-1, 2)^2 + randn();
        end
        dependenceProperties{2} = struct('type', 'NonlinearAR', 'params', [arCoef, nonlinCoef]);
        
        % Optimal block size - use similar approach as AR
        optimalBlockSize = ceil(2 * (2 * arCoef / (1 - arCoef^2))^(2/3) * T^(1/3));
        
    otherwise
        % Default to independent series
        data(:, 2) = randn(T, 1);
        dependenceProperties{2} = struct('type', 'none', 'params', []);
        optimalBlockSize = 1;
end

% Generate series with heteroskedasticity if requested
if isfield(parameters, 'heteroskedastic') && parameters.heteroskedastic
    volatility = 0.5 + 1.5 * abs(sin(2*pi*(1:T)'/T));
    data(:, 3) = volatility .* randn(T, 1);
    dependenceProperties{3} = struct('type', 'Heteroskedastic', 'params', []);
else
    data(:, 3) = data(:, 2);
    dependenceProperties{3} = dependenceProperties{2};
end

% Generate series with jumps if requested
if isfield(parameters, 'jumps') && parameters.jumps
    if ~isfield(parameters, 'jumpParams') || isempty(parameters.jumpParams)
        parameters.jumpParams = struct('intensity', 0.05, 'jumpSize', [0, 0.02]);
    end
    [jumpReturns, jumpTimes, jumpSizes] = addJumps(T, T/250, parameters.jumpParams);
    data(:, 4) = data(:, 2) + jumpReturns;
    dependenceProperties{4} = struct('type', 'Jumps', 'params', parameters.jumpParams, ...
                                   'jumpTimes', jumpTimes, 'jumpSizes', jumpSizes);
else
    data(:, 4) = data(:, 2);
    dependenceProperties{4} = dependenceProperties{2};
end

% Generate specific series for block bootstrap testing
blockBootstrapSeries = cell(length(blockSizes), 1);
for i = 1:length(blockSizes)
    blockSize = blockSizes(i);
    
    % Generate blocks
    numBlocks = ceil(T / blockSize);
    blocks = randn(blockSize, numBlocks);
    
    % Add correlation within blocks but independence between blocks
    for b = 1:numBlocks
        for t = 2:blockSize
            blocks(t, b) = 0.6 * blocks(t-1, b) + 0.4 * randn();
        end
    end
    
    % Assemble series
    series = reshape(blocks(:, 1:numBlocks), [], 1);
    blockBootstrapSeries{i} = series(1:T);
end

% Generate series for stationary bootstrap testing
stationarySeries = struct();
stationarySeries.weak = generateARSeries(T, 0.3);
stationarySeries.medium = generateARSeries(T, 0.6);
stationarySeries.strong = generateARSeries(T, 0.8);

% Create output structure
bootData = struct();
bootData.data = data;
bootData.trueBlockSizes = optimalBlockSize;
bootData.parameters = parameters;
bootData.dependenceProperties = dependenceProperties;
bootData.blockBootstrapSeries = blockBootstrapSeries;
bootData.blockSizes = blockSizes;
bootData.stationarySeries = stationarySeries;

% Generate non-stationary series if requested
if isfield(parameters, 'stationary') && ~parameters.stationary
    bootData.nonstationarySeries = cumsum(randn(T, 1));
end
end

%% Helper Functions

function dailyVol = generateDailyVolatility(numDays, parameters)
% Generates daily volatility process based on specified model
switch lower(parameters.volatilityModel)
    case 'constant'
        sigma = parameters.volatilityParams.sigma;
        dailyVol = repmat(sigma, numDays, 1);
        
    case 'garch'
        % Generate volatility using GARCH model
        garchParams = parameters.volatilityParams;
        volData = generateVolatilitySeries(numDays, garchParams.modelType, garchParams);
        dailyVol = sqrt(volData.ht);
        
    case 'sinusoidal'
        % Generate sinusoidal volatility pattern
        amplitude = parameters.volatilityParams.amplitude;
        period = parameters.volatilityParams.period;
        phase = parameters.volatilityParams.phase;
        
        baseVol = 0.01;
        t = (1:numDays)';
        dailyVol = baseVol + amplitude * sin(2*pi*t/period + phase);
        
    otherwise
        error('Unknown volatility model: %s', parameters.volatilityModel);
end
end

function intradayVol = generateIntradayPattern(obsPerDay, parameters)
% Generates intraday volatility pattern
switch lower(parameters.intradayPattern)
    case 'flat'
        intradayVol = ones(obsPerDay, 1);
        
    case 'u-shape'
        t = linspace(0, 1, obsPerDay)';
        
        % Parameters for U-shape
        if isfield(parameters, 'uShapeParams') && ~isempty(parameters.uShapeParams)
            a = parameters.uShapeParams.a;
            b = parameters.uShapeParams.b;
            c = parameters.uShapeParams.c;
        else
            a = 3;   % Opening volatility
            b = 3;   % Closing volatility
            c = 0.5; % Mid-day volatility
        end
        
        intradayVol = a*(1-t).^2 + c + b*t.^2;
        intradayVol = intradayVol / mean(intradayVol);
        
    case 'custom'
        if ~isfield(parameters, 'intradayShape') || isempty(parameters.intradayShape)
            error('Custom intraday pattern requires intradayShape parameter');
        end
        
        intradayVol = parameters.intradayShape(:);
        intradayVol = intradayVol / mean(intradayVol);
        
    otherwise
        error('Unknown intraday pattern: %s', parameters.intradayPattern);
end
end

function [jumpReturns, jumpTimes, jumpSizes] = addJumps(totalObs, numDays, jumpParams)
% Add jumps based on Poisson process
intensity = jumpParams.intensity;
jumpSizeParams = jumpParams.jumpSize;

expectedJumps = intensity * numDays;
numJumps = poissrnd(expectedJumps);

jumpReturns = zeros(totalObs, 1);

if numJumps > 0
    jumpTimes = randi(totalObs, numJumps, 1);
    
    jumpMean = jumpSizeParams(1);
    jumpStd = jumpSizeParams(2);
    jumpSizes = jumpMean + jumpStd * randn(numJumps, 1);
    
    for i = 1:numJumps
        jumpReturns(jumpTimes(i)) = jumpSizes(i);
    end
else
    jumpTimes = [];
    jumpSizes = [];
end
end

function [observedReturns, noise] = addMicrostructureNoise(trueReturns, noiseType, noiseParams)
% Add microstructure noise to returns
switch lower(noiseType)
    case 'additive'
        std = noiseParams.std;
        noise = std * randn(size(trueReturns));
        observedReturns = trueReturns + noise;
        
    case 'bid-ask'
        spread = noiseParams.spread;
        halfSpread = spread / 2;
        
        bidAskIndicator = sign(randn(size(trueReturns)));
        noise = halfSpread * (bidAskIndicator - [0; bidAskIndicator(1:end-1)]);
        observedReturns = trueReturns + noise;
        
    otherwise
        error('Unknown microstructure noise type: %s', noiseType);
end
end

function innovations = generateDistributionForHF(n, parameters)
% Generate innovations from specified distribution
switch lower(parameters.distribution)
    case 'normal'
        innovations = randn(n, 1);
        
    case 't'
        if isfield(parameters, 'distParams') && isfield(parameters.distParams, 'nu')
            nu = parameters.distParams.nu;
        else
            nu = 5;
        end
        innovations = stdtrnd(n, nu);
        
    case 'ged'
        if isfield(parameters, 'distParams') && isfield(parameters.distParams, 'nu')
            nu = parameters.distParams.nu;
        else
            nu = 1.5;
        end
        innovations = gedrnd(nu, n, 1);
        
    case 'skewt'
        if isfield(parameters, 'distParams')
            if isfield(parameters.distParams, 'nu') && isfield(parameters.distParams, 'lambda')
                nu = parameters.distParams.nu;
                lambda = parameters.distParams.lambda;
            else
                nu = 5;
                lambda = -0.1;
            end
        else
            nu = 5;
            lambda = -0.1;
        end
        innovations = skewtrnd(nu, lambda, n, 1);
        
    otherwise
        innovations = randn(n, 1);
end
end

function factors = generateFactorReturnsForCS(numPeriods, numFactors, parameters)
% Generate factor returns for cross-sectional data
switch lower(parameters.factorModel)
    case 'normal'
        factors = mvnrnd(parameters.factorMean, parameters.factorCov, numPeriods);
        
    case 'ar'
        factors = zeros(numPeriods, numFactors);
        
        if ~isfield(parameters.factorParams, 'ar')
            parameters.factorParams.ar = 0.3 * ones(numFactors, 1);
        end
        
        if ~isfield(parameters.factorParams, 'innov_cov')
            parameters.factorParams.innov_cov = parameters.factorCov;
        end
        
        factors(1,:) = mvnrnd(parameters.factorMean, parameters.factorCov, 1);
        
        for t = 2:numPeriods
            condMean = parameters.factorMean(:)' + ...
                       parameters.factorParams.ar(:)' .* (factors(t-1,:) - parameters.factorMean(:)');
            innovation = mvnrnd(zeros(1, numFactors), parameters.factorParams.innov_cov, 1);
            factors(t,:) = condMean + innovation;
        end
        
    case 'garch'
        factors = zeros(numPeriods, numFactors);
        
        for f = 1:numFactors
            if ~isfield(parameters.factorParams, 'garch')
                garchParams = struct('modelType', 'GARCH', 'p', 1, 'q', 1, ...
                                   'omega', 0.05, 'alpha', 0.1, 'beta', 0.85);
            else
                garchParams = parameters.factorParams.garch;
            end
            
            volData = generateVolatilitySeries(numPeriods, garchParams.modelType, garchParams);
            factors(:,f) = volData.returns + parameters.factorMean(f);
        end
        
        % If factor correlation is desired, transform using Cholesky
        if ~all(all(parameters.factorCov == diag(diag(parameters.factorCov))))
            factors = (factors - mean(factors)) ./ std(factors);
            chol_factor = chol(parameters.factorCov, 'lower');
            factors = factors * chol_factor';
            factors = factors + repmat(parameters.factorMean(:)', numPeriods, 1);
        end
        
    otherwise
        error('Unknown factor model: %s', parameters.factorModel);
end
end

function loadings = generateFactorLoadingsForCS(numAssets, numFactors, parameters, characteristics)
% Generate factor loadings
loadings = zeros(numAssets, numFactors);

switch lower(parameters.loadingDist)
    case 'normal'
        mean = parameters.loadingParams.mean;
        std = parameters.loadingParams.std;
        
        for f = 1:numFactors
            loadings(:,f) = mean + std * randn(numAssets, 1);
        end
        
    case 'uniform'
        min_val = parameters.loadingParams.min;
        max_val = parameters.loadingParams.max;
        
        for f = 1:numFactors
            loadings(:,f) = min_val + (max_val - min_val) * rand(numAssets, 1);
        end
        
    case 'custom'
        if ~isfield(parameters.loadingParams, 'loadings')
            error('Custom loading distribution requires loadingParams.loadings matrix');
        end
        
        loadings = parameters.loadingParams.loadings;
end

% Modify loadings based on characteristics if provided
if ~isempty(characteristics) && isfield(parameters.characteristics, 'loadingRelevance')
    [~, numChars] = size(characteristics);
    
    for f = 1:numFactors
        for c = 1:numChars
            effect = parameters.characteristics.loadingRelevance(c, f) * characteristics(:, c);
            loadings(:, f) = loadings(:, f) + effect;
        end
    end
end
end

function idioReturns = generateIdiosyncraticReturnsForCS(numPeriods, numAssets, parameters)
% Generate idiosyncratic returns
idioVol = parameters.idioVol;
idioCorr = parameters.idioCorr;

if isscalar(idioVol)
    idioVol = repmat(idioVol, numAssets, 1);
end

if isscalar(idioCorr) && idioCorr == 0
    idioCov = diag(idioVol.^2);
else
    if isscalar(idioCorr)
        corrMatrix = idioCorr * ones(numAssets) + (1 - idioCorr) * eye(numAssets);
    else
        corrMatrix = idioCorr;
    end
    
    idioCov = diag(idioVol) * corrMatrix * diag(idioVol);
end

idioReturns = mvnrnd(zeros(numAssets, 1), idioCov, numPeriods);
end

function segment = generateARMASegment(numObs, arParams, maParams, constant, innovations, hasExo, exoVars, exoCoef)
% Generate ARMA segment with given parameters
p = length(arParams);
q = length(maParams);
segment = zeros(numObs, 1);

% Initialize with innovations
for t = 1:numObs
    % Start with constant
    segment(t) = constant;
    
    % Add AR component
    for i = 1:min(t-1, p)
        segment(t) = segment(t) + arParams(i) * segment(t-i);
    end
    
    % Add MA component
    for j = 1:min(t-1, q)
        segment(t) = segment(t) + maParams(j) * innovations(t-j);
    end
    
    % Add exogenous component
    if hasExo && ~isempty(exoVars) && ~isempty(exoCoef)
        segment(t) = segment(t) + exoVars(t,:) * exoCoef;
    end
    
    % Add current innovation
    segment(t) = segment(t) + innovations(t);
end
end

function series = generateARSeries(T, arCoef)
% Generate AR(1) series with given coefficient
series = zeros(T, 1);
series(1) = randn();
for t = 2:T
    series(t) = arCoef * series(t-1) + randn();
end
end

function skew = computeSkewness(x)
% Compute sample skewness
n = length(x);
x = x - mean(x);
s = std(x);
skew = sum(x.^3)/(n * s^3);
end

function kurt = computeKurtosis(x)
% Compute sample kurtosis
n = length(x);
x = x - mean(x);
v = var(x);
kurt = sum(x.^4)/(n * v^2);
end