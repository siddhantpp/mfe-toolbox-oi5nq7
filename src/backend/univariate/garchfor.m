function forecast = garchfor(garchModel, numPeriods, options)
% GARCHFOR Generates volatility forecasts for univariate GARCH-type models
%
% USAGE:
%   FORECAST = garchfor(GARCHMODEL, NUMPERIODS)
%   FORECAST = garchfor(GARCHMODEL, NUMPERIODS, OPTIONS)
%
% INPUTS:
%   GARCHMODEL - Structure containing GARCH model specification with fields:
%                parameters - Vector of model parameters
%                modelType  - String specifying model type: 'GARCH', 'EGARCH', 'GJR', 
%                            'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'
%                p          - GARCH order (positive integer)
%                q          - ARCH order (positive integer)
%                data       - Time series data used for estimation
%                residuals  - Model residuals
%                ht         - Conditional variances from the model
%                distribution - Distribution used: 'normal', 't', 'ged', 'skewt'
%                distParams   - Distribution parameters (e.g., nu for t, nu for GED, 
%                              or [nu lambda] for skewt)
%
%   NUMPERIODS - Number of periods to forecast ahead
%
%   OPTIONS    - [OPTIONAL] Structure containing forecasting options:
%                simulate     - Boolean, whether to use simulation [default: false]
%                numPaths     - Number of simulation paths [default: 1000]
%                seed         - Random seed for reproducibility [default: none]
%                probs        - Vector of probability levels for quantiles 
%                              [default: [0.01 0.05 0.5 0.95 0.99]]
%
% OUTPUTS:
%   FORECAST   - Structure containing forecast results with fields:
%                expectedVariances - Point forecasts of conditional variances
%                expectedReturns   - Point forecasts of returns (zero for 
%                                   zero-mean models)
%                expectedVolatility - Point forecasts of volatility (sqrt of variance)
%                If simulation is used, also includes:
%                varPaths    - Matrix of simulated variance paths
%                returnPaths - Matrix of simulated return paths
%                volatilityPaths - Matrix of simulated volatility paths
%                varQuantiles - Quantiles of forecasted variances
%                returnQuantiles - Quantiles of forecasted returns
%                volatilityQuantiles - Quantiles of forecasted volatilities
%                varMean     - Mean of simulated variances
%                returnMean  - Mean of simulated returns
%                volatilityMean - Mean of simulated volatilities
%                probLevels  - Probability levels used for quantiles
%
% COMMENTS:
%   This function provides h-step ahead volatility forecasts for various GARCH-type
%   models including standard GARCH, EGARCH, TARCH/GJR, AGARCH, IGARCH, and NAGARCH.
%   
%   Two forecasting methods are available:
%   1. Deterministic forecasting: Computes conditional expectations of future variances
%      based on model parameters and last observed variances.
%   2. Simulation-based forecasting: Generates multiple paths using random innovations
%      from the specified distribution and computes statistics from these paths.
%
%   Different error distributions are supported:
%   - Normal: Standard normal distribution
%   - Student's t: Heavy-tailed symmetric distribution
%   - GED: Generalized Error Distribution
%   - Skewed t: Hansen's skewed t-distribution
%
% EXAMPLES:
%   % Generate 10-period ahead deterministic forecast
%   forecast = garchfor(garchModel, 10);
%
%   % Generate 20-period ahead forecast with 5000 simulation paths
%   options = struct('simulate', true, 'numPaths', 5000);
%   forecast = garchfor(garchModel, 20, options);
%
%   % Generate forecast with custom quantile probabilities and fixed random seed
%   options = struct('simulate', true, 'probs', [0.025 0.25 0.5 0.75 0.975], 'seed', 123);
%   forecast = garchfor(garchModel, 15, options);
%
% See also GARCH, GARCHCORE, EGARCH, GJR, TARCH, AGARCH, IGARCH, NAGARCH

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Minimum variance level for numerical stability
MIN_VARIANCE = 1e-12;

%% Input validation

% Validate garchModel structure
parametercheck(garchModel, 'garchModel');
if ~isstruct(garchModel)
    error('GARCHMODEL must be a structure.');
end

% Check required fields in garchModel
requiredFields = {'parameters', 'modelType', 'p', 'q', 'data', 'residuals', 'ht'};
for i = 1:length(requiredFields)
    if ~isfield(garchModel, requiredFields{i})
        error(['GARCHMODEL is missing required field: ' requiredFields{i}]);
    end
end

% Validate numPeriods
opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
parametercheck(numPeriods, 'numPeriods', opts);

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Set default simulation flag
if ~isfield(options, 'simulate') || isempty(options.simulate)
    options.simulate = false;
end

% Set default number of simulation paths
if ~isfield(options, 'numPaths') || isempty(options.numPaths)
    options.numPaths = 1000;
else
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.numPaths, 'options.numPaths', opts);
end

% Set default probability levels for quantiles
if ~isfield(options, 'probs') || isempty(options.probs)
    options.probs = [0.01 0.05 0.5 0.95 0.99];
else
    opts = struct('lowerBound', 0, 'upperBound', 1);
    parametercheck(options.probs, 'options.probs', opts);
end

% Set random seed if provided
if isfield(options, 'seed') && ~isempty(options.seed)
    rng(options.seed);
end

%% Extract model parameters and data

% Extract model information
parameters = garchModel.parameters;
modelType = upper(garchModel.modelType);
p = garchModel.p;
q = garchModel.q;
residuals = garchModel.residuals;
ht = garchModel.ht;

% Ensure residuals are column vector
residuals = columncheck(residuals, 'residuals');
datacheck(residuals, 'residuals');

% Validate the model type
validModels = {'GARCH', 'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'IGARCH', 'NAGARCH'};
if ~ismember(modelType, validModels)
    error('Unknown model type: %s', modelType);
end

% Extract distribution information
if isfield(garchModel, 'distribution')
    distribution = lower(garchModel.distribution);
else
    distribution = 'normal';  % Default to normal distribution
end

% Validate distribution type
validDistributions = {'normal', 't', 'ged', 'skewt'};
if ~ismember(distribution, validDistributions)
    error('Unknown distribution type: %s', distribution);
end

% Extract distribution parameters
if isfield(garchModel, 'distParams')
    distParams = garchModel.distParams;
else
    distParams = [];
end

% Validate distribution parameters based on distribution type
if strcmp(distribution, 't') || strcmp(distribution, 'ged')
    if isempty(distParams)
        error('Distribution parameters must be provided for %s distribution.', distribution);
    end
    opts = struct('isscalar', true, 'isPositive', true);
    parametercheck(distParams, 'distParams', opts);
elseif strcmp(distribution, 'skewt')
    if isempty(distParams) || length(distParams) < 2
        error('Distribution parameters (nu and lambda) must be provided for skewt distribution.');
    end
    opts = struct('isPositive', true);
    parametercheck(distParams(1), 'distParams(1)', opts);
    opts = struct('lowerBound', -1, 'upperBound', 1);
    parametercheck(distParams(2), 'distParams(2)', opts);
end

% Extract the last returns and variances for forecasting
maxLag = max(p, q);
lastVariances = ht(end-maxLag+1:end);
lastReturns = residuals(end-maxLag+1:end);

%% Generate forecasts

% Initialize the forecast structure
forecast = struct();

% Compute deterministic (point) forecasts
expectedVariances = compute_deterministic_forecast(parameters, lastVariances, lastReturns, ...
    modelType, p, q, numPeriods);

% Store deterministic forecasts
forecast.expectedVariances = expectedVariances;
forecast.expectedReturns = zeros(numPeriods, 1);  % Expected returns are zero for zero-mean models
forecast.expectedVolatility = sqrt(expectedVariances);  % Volatility forecasts (standard deviation)

% If simulation is requested, compute simulated paths and statistics
if options.simulate
    % Generate multiple simulation paths
    simResults = simulate_garch_paths(parameters, lastVariances, lastReturns, ...
        modelType, p, q, numPeriods, options.numPaths, distribution, distParams);
    
    % Store simulation results
    forecast.varPaths = simResults.varPaths;
    forecast.returnPaths = simResults.returnPaths;
    forecast.volatilityPaths = sqrt(simResults.varPaths);  % Volatility paths
    
    % Compute and store statistics from simulation
    stats = compute_forecast_statistics(simResults.varPaths, simResults.returnPaths, options.probs);
    
    % Add statistics to the forecast structure
    forecast.varMean = stats.varMean;
    forecast.returnMean = stats.returnMean;
    forecast.volatilityMean = sqrt(stats.varMean);  % Mean volatility
    
    forecast.varMedian = stats.varMedian;
    forecast.returnMedian = stats.returnMedian;
    forecast.volatilityMedian = sqrt(stats.varMedian);  % Median volatility
    
    forecast.varQuantiles = stats.varQuantiles;
    forecast.returnQuantiles = stats.returnQuantiles;
    forecast.volatilityQuantiles = sqrt(stats.varQuantiles);  % Volatility quantiles
    
    % Add the probability levels for reference
    forecast.probLevels = options.probs;
end

end

%% Helper function for deterministic forecasting
function expectedVariances = compute_deterministic_forecast(parameters, lastVariances, lastReturns, ...
    modelType, p, q, numPeriods)
% Computes deterministic volatility forecasts for GARCH-type models

% Initialize array for forecasted variances
expectedVariances = zeros(numPeriods, 1);

% Extract model parameters based on model type
switch modelType
    case 'GARCH'
        % Extract parameters: omega, alpha(1:q), beta(1:p)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        beta = parameters(q+2:q+p+1);
        
        % Precompute unconditional variance for long-horizon forecasts
        persistenceLevel = sum(alpha) + sum(beta);
        if persistenceLevel < 1
            uncondVariance = omega / (1 - persistenceLevel);
        else
            uncondVariance = NaN;  % Undefined for non-stationary model
        end
        
        % Generate multi-step ahead forecasts
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    nextVar = nextVar + alpha(i) * lastReturns(end-i+1)^2;
                end
                for j = 1:p
                    nextVar = nextVar + beta(j) * lastVariances(end-j+1);
                end
            else
                % Multi-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    if i < h
                        % For longer horizons, use expected variance (since E[e_t^2] = h_t)
                        nextVar = nextVar + alpha(i) * expectedVariances(h-i);
                    else
                        % For near-term forecasts, use last observed squared returns
                        nextVar = nextVar + alpha(i) * lastReturns(end-i+h)^2;
                    end
                end
                
                for j = 1:p
                    if j < h
                        % For longer horizons, use previously forecasted variances
                        nextVar = nextVar + beta(j) * expectedVariances(h-j);
                    else
                        % For near-term forecasts, use last observed variances
                        nextVar = nextVar + beta(j) * lastVariances(end-j+h);
                    end
                end
                
                % For very long horizons, approach unconditional variance if available
                if h > 100 && ~isnan(uncondVariance)
                    nextVar = 0.99 * nextVar + 0.01 * uncondVariance;
                end
            end
            
            % Store the forecasted variance with minimum threshold
            expectedVariances(h) = max(nextVar, MIN_VARIANCE);
        end
        
    case {'GJR', 'TARCH'}
        % Extract parameters: omega, alpha(1:q), gamma(1:q), beta(1:p)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        gamma = parameters(q+2:2*q+1);
        beta = parameters(2*q+2:2*q+p+1);
        
        % Generate multi-step ahead forecasts
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    archEffect = alpha(i) * lastReturns(end-i+1)^2;
                    % Add asymmetry effect if return is negative
                    if lastReturns(end-i+1) < 0
                        archEffect = archEffect + gamma(i) * lastReturns(end-i+1)^2;
                    end
                    nextVar = nextVar + archEffect;
                end
                for j = 1:p
                    nextVar = nextVar + beta(j) * lastVariances(end-j+1);
                end
            else
                % Multi-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    if i < h
                        % For longer horizons, use expected variance with asymmetry adjustment
                        % E[alpha*e_t^2 + gamma*e_t^2*I(e_t<0)] = alpha*h_t + gamma*h_t*0.5
                        % since P(e_t<0) = 0.5 for zero-mean symmetric distributions
                        nextVar = nextVar + (alpha(i) + 0.5*gamma(i)) * expectedVariances(h-i);
                    else
                        % For near-term forecasts, use last observed returns
                        archEffect = alpha(i) * lastReturns(end-i+h)^2;
                        if lastReturns(end-i+h) < 0
                            archEffect = archEffect + gamma(i) * lastReturns(end-i+h)^2;
                        end
                        nextVar = nextVar + archEffect;
                    end
                end
                
                for j = 1:p
                    if j < h
                        nextVar = nextVar + beta(j) * expectedVariances(h-j);
                    else
                        nextVar = nextVar + beta(j) * lastVariances(end-j+h);
                    end
                end
            end
            
            % Store the forecasted variance
            expectedVariances(h) = max(nextVar, MIN_VARIANCE);
        end
        
    case 'EGARCH'
        % Extract parameters: omega, alpha(1:q), gamma(1:q), beta(1:p)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        gamma = parameters(q+2:2*q+1);
        beta = parameters(2*q+2:2*q+p+1);
        
        % Expected value of |z_t| for standard normal = sqrt(2/pi)
        absZexpected = sqrt(2/pi);
        
        % Compute log of last variances
        lastLogVars = log(lastVariances);
        
        % Generate multi-step ahead forecasts for log-variance
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast for log-variance
                logVar = omega;
                for i = 1:q
                    stdResid = lastReturns(end-i+1) / sqrt(lastVariances(end-i+1));
                    absStdResid = abs(stdResid);
                    
                    % Add size effect
                    logVar = logVar + alpha(i) * (absStdResid - absZexpected);
                    
                    % Add sign effect
                    logVar = logVar + gamma(i) * stdResid;
                end
                for j = 1:p
                    logVar = logVar + beta(j) * lastLogVars(end-j+1);
                end
            else
                % Multi-step ahead forecast
                logVar = omega;
                
                % For multi-step forecasts, size and sign effects have zero expectation
                % The size effect E[|z_t| - E[|z_t|]] = 0
                % The sign effect E[z_t] = 0 for symmetric distributions
                
                % Add persistence from past log-variances
                for j = 1:p
                    if j < h
                        logVar = logVar + beta(j) * log(expectedVariances(h-j));
                    else
                        logVar = logVar + beta(j) * lastLogVars(end-j+h);
                    end
                end
            end
            
            % Convert to variance by exponentiating
            expectedVariances(h) = max(exp(logVar), MIN_VARIANCE);
        end
        
    case 'AGARCH'
        % Extract parameters: omega, alpha(1:q), gamma, beta(1:p)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        gamma = parameters(q+2);
        beta = parameters(q+3:q+p+2);
        
        % Generate multi-step ahead forecasts
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    % Asymmetric news impact
                    asymNews = (lastReturns(end-i+1) - gamma)^2;
                    nextVar = nextVar + alpha(i) * asymNews;
                end
                for j = 1:p
                    nextVar = nextVar + beta(j) * lastVariances(end-j+1);
                end
            else
                % Multi-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    if i < h
                        % For longer horizons, use expected variance with asymmetry adjustment
                        % E[(e_t - gamma)^2] = E[e_t^2 - 2*gamma*e_t + gamma^2]
                        % = h_t + gamma^2 (since E[e_t] = 0)
                        nextVar = nextVar + alpha(i) * (expectedVariances(h-i) + gamma^2);
                    else
                        asymNews = (lastReturns(end-i+h) - gamma)^2;
                        nextVar = nextVar + alpha(i) * asymNews;
                    end
                end
                
                for j = 1:p
                    if j < h
                        nextVar = nextVar + beta(j) * expectedVariances(h-j);
                    else
                        nextVar = nextVar + beta(j) * lastVariances(end-j+h);
                    end
                end
            end
            
            % Store the forecasted variance
            expectedVariances(h) = max(nextVar, MIN_VARIANCE);
        end
        
    case 'IGARCH'
        % Extract parameters: omega, alpha(1:q), beta(1:p)
        % In IGARCH, sum(alpha) + sum(beta) = 1 (this should be enforced during estimation)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        beta = parameters(q+2:q+p+1);
        
        % For IGARCH, long-term forecasts increase linearly with horizon for large h
        
        % Generate multi-step ahead forecasts
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    nextVar = nextVar + alpha(i) * lastReturns(end-i+1)^2;
                end
                for j = 1:p
                    nextVar = nextVar + beta(j) * lastVariances(end-j+1);
                end
            else
                % Multi-step ahead forecast
                nextVar = omega;
                for i = 1:q
                    if i < h
                        nextVar = nextVar + alpha(i) * expectedVariances(h-i);
                    else
                        nextVar = nextVar + alpha(i) * lastReturns(end-i+h)^2;
                    end
                end
                
                for j = 1:p
                    if j < h
                        nextVar = nextVar + beta(j) * expectedVariances(h-j);
                    else
                        nextVar = nextVar + beta(j) * lastVariances(end-j+h);
                    end
                end
                
                % For very long horizons, linear growth can be approximated
                if h > 100
                    nextVar = expectedVariances(1) + omega * (h - 1);
                end
            end
            
            % Store the forecasted variance
            expectedVariances(h) = max(nextVar, MIN_VARIANCE);
        end
        
    case 'NAGARCH'
        % Extract parameters: omega, alpha(1:q), gamma, beta(1:p)
        omega = parameters(1);
        alpha = parameters(2:q+1);
        gamma = parameters(q+2);
        beta = parameters(q+3:q+p+2);
        
        % Generate multi-step ahead forecasts
        for h = 1:numPeriods
            if h == 1
                % One-step ahead forecast
                nextVar = omega;
                for j = 1:p
                    nextVar = nextVar + beta(j) * lastVariances(end-j+1);
                end
                
                for i = 1:q
                    stdResid = lastReturns(end-i+1) / sqrt(lastVariances(end-i+1));
                    nonlinAsym = (stdResid - gamma)^2;
                    nextVar = nextVar + alpha(i) * lastVariances(end-i+1) * nonlinAsym;
                end
            else
                % Multi-step ahead forecast
                nextVar = omega;
                
                % Add GARCH components first
                for j = 1:p
                    if j < h
                        nextVar = nextVar + beta(j) * expectedVariances(h-j);
                    else
                        nextVar = nextVar + beta(j) * lastVariances(end-j+h);
                    end
                end
                
                % Add nonlinear ARCH components
                for i = 1:q
                    if i < h
                        % For longer horizons, use expected value
                        % E[(z_t - gamma)^2] = E[z_t^2 - 2*gamma*z_t + gamma^2]
                        % = 1 + gamma^2 (since E[z_t^2] = 1 and E[z_t] = 0)
                        expectedNonlinAsym = 1 + gamma^2;
                        nextVar = nextVar + alpha(i) * expectedVariances(h-i) * expectedNonlinAsym;
                    else
                        stdResid = lastReturns(end-i+h) / sqrt(lastVariances(end-i+h));
                        nonlinAsym = (stdResid - gamma)^2;
                        nextVar = nextVar + alpha(i) * lastVariances(end-i+h) * nonlinAsym;
                    end
                end
            end
            
            % Store the forecasted variance
            expectedVariances(h) = max(nextVar, MIN_VARIANCE);
        end
        
    otherwise
        error('Unknown model type: %s', modelType);
end

end

%% Helper function for simulation-based forecasting
function simResults = simulate_garch_paths(parameters, lastVariances, lastReturns, ...
    modelType, p, q, numPeriods, numPaths, distribution, distParams)
% Simulates multiple paths of returns and variances for a GARCH-type model

% Initialize arrays for simulated paths
varPaths = zeros(numPeriods, numPaths);
returnPaths = zeros(numPeriods, numPaths);

% Generate random innovations based on the specified distribution
switch distribution
    case 'normal'
        % Standard normal innovations
        innovations = randn(numPeriods, numPaths);
        
    case 't'
        % Student's t innovations (nu degrees of freedom, standardized)
        nu = distParams;
        innovations = stdtrnd([numPeriods, numPaths], nu);
        
    case 'ged'
        % Generalized Error Distribution innovations
        nu = distParams;
        innovations = gedrnd(nu, numPeriods, numPaths);
        
    case 'skewt'
        % Hansen's skewed t innovations
        nu = distParams(1);
        lambda = distParams(2);
        innovations = skewtrnd(nu, lambda, numPeriods, numPaths);
        
    otherwise
        error('Unknown distribution type: %s', distribution);
end

% Get the number of lags needed
maxLag = max(p, q);

% Simulate each path
for path = 1:numPaths
    % Initialize history with the last observed values
    varHistory = lastVariances;
    returnHistory = lastReturns;
    
    % Generate time series for this path
    for t = 1:numPeriods
        % Compute conditional variance for this time step based on model type
        if strcmp(modelType, 'GARCH')
            % Standard GARCH
            omega = parameters(1);
            alpha = parameters(2:q+1);
            beta = parameters(q+2:q+p+1);
            
            % Compute variance using GARCH recursion
            nextVar = omega;
            for i = 1:q
                nextVar = nextVar + alpha(i) * returnHistory(end-i+1)^2;
            end
            for j = 1:p
                nextVar = nextVar + beta(j) * varHistory(end-j+1);
            end
            
        elseif strcmp(modelType, 'EGARCH')
            % EGARCH model
            omega = parameters(1);
            alpha = parameters(2:q+1);
            gamma = parameters(q+2:2*q+1);
            beta = parameters(2*q+2:2*q+p+1);
            
            % Expected value of |z_t| for standard normal = sqrt(2/pi)
            absZexpected = sqrt(2/pi);
            
            % Compute log-variance
            logVar = omega;
            for i = 1:q
                stdResid = returnHistory(end-i+1) / sqrt(varHistory(end-i+1));
                absStdResid = abs(stdResid);
                
                % Add size effect
                logVar = logVar + alpha(i) * (absStdResid - absZexpected);
                
                % Add sign effect
                logVar = logVar + gamma(i) * stdResid;
            end
            
            for j = 1:p
                logVar = logVar + beta(j) * log(varHistory(end-j+1));
            end
            
            nextVar = exp(logVar);
            
        elseif strcmp(modelType, 'GJR') || strcmp(modelType, 'TARCH')
            % GJR/TARCH model
            omega = parameters(1);
            alpha = parameters(2:q+1);
            gamma = parameters(q+2:2*q+1);
            beta = parameters(2*q+2:2*q+p+1);
            
            nextVar = omega;
            for i = 1:q
                archEffect = alpha(i) * returnHistory(end-i+1)^2;
                if returnHistory(end-i+1) < 0
                    archEffect = archEffect + gamma(i) * returnHistory(end-i+1)^2;
                end
                nextVar = nextVar + archEffect;
            end
            
            for j = 1:p
                nextVar = nextVar + beta(j) * varHistory(end-j+1);
            end
            
        elseif strcmp(modelType, 'AGARCH')
            % AGARCH model
            omega = parameters(1);
            alpha = parameters(2:q+1);
            gamma = parameters(q+2);
            beta = parameters(q+3:q+p+2);
            
            nextVar = omega;
            for i = 1:q
                asymNews = (returnHistory(end-i+1) - gamma)^2;
                nextVar = nextVar + alpha(i) * asymNews;
            end
            
            for j = 1:p
                nextVar = nextVar + beta(j) * varHistory(end-j+1);
            end
            
        elseif strcmp(modelType, 'IGARCH')
            % IGARCH model
            omega = parameters(1);
            alpha = parameters(2:q+1);
            beta = parameters(q+2:q+p+1);
            
            nextVar = omega;
            for i = 1:q
                nextVar = nextVar + alpha(i) * returnHistory(end-i+1)^2;
            end
            
            for j = 1:p
                nextVar = nextVar + beta(j) * varHistory(end-j+1);
            end
            
        elseif strcmp(modelType, 'NAGARCH')
            % NAGARCH model
            omega = parameters(1);
            alpha = parameters(2:q+1);
            gamma = parameters(q+2);
            beta = parameters(q+3:q+p+2);
            
            nextVar = omega;
            for j = 1:p
                nextVar = nextVar + beta(j) * varHistory(end-j+1);
            end
            
            for i = 1:q
                stdResid = returnHistory(end-i+1) / sqrt(varHistory(end-i+1));
                nonlinAsym = (stdResid - gamma)^2;
                nextVar = nextVar + alpha(i) * varHistory(end-i+1) * nonlinAsym;
            end
            
        else
            error('Unknown model type: %s', modelType);
        end
        
        % Ensure minimum variance
        nextVar = max(nextVar, MIN_VARIANCE);
        
        % Generate return with the computed variance
        nextReturn = sqrt(nextVar) * innovations(t, path);
        
        % Store the simulated values for this path
        varPaths(t, path) = nextVar;
        returnPaths(t, path) = nextReturn;
        
        % Update histories for next iteration
        varHistory = [varHistory(2:end); nextVar];
        returnHistory = [returnHistory(2:end); nextReturn];
    end
end

% Return simulation results
simResults = struct('varPaths', varPaths, 'returnPaths', returnPaths);

end

%% Helper function for computing forecast statistics
function stats = compute_forecast_statistics(varPaths, returnPaths, probs)
% Computes summary statistics from simulated GARCH paths

% Get dimensions
[numPeriods, numPaths] = size(varPaths);

% Compute mean forecasts across paths
varMean = mean(varPaths, 2);
returnMean = mean(returnPaths, 2);

% Compute median forecasts
varMedian = median(varPaths, 2);
returnMedian = median(returnPaths, 2);

% Compute quantiles for each forecast horizon
numProbs = length(probs);
varQuantiles = zeros(numPeriods, numProbs);
returnQuantiles = zeros(numPeriods, numProbs);

for t = 1:numPeriods
    varQuantiles(t, :) = quantile(varPaths(t, :), probs);
    returnQuantiles(t, :) = quantile(returnPaths(t, :), probs);
end

% Return statistics structure
stats = struct('varMean', varMean, 'returnMean', returnMean, ...
               'varMedian', varMedian, 'returnMedian', returnMedian, ...
               'varQuantiles', varQuantiles, 'returnQuantiles', returnQuantiles);

end