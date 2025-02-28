function Results = ccc_mvgarch(data, options)
% CCC_MVGARCH Constant Conditional Correlation Multivariate GARCH model estimation
%
% USAGE:
%   Results = ccc_mvgarch(data)
%   Results = ccc_mvgarch(data, options)
%
% INPUTS:
%   data    - A T by K matrix of zero mean residuals where T is the number
%             of observations and K is the number of series
%   options - Structure with the following fields:
%             model      - Type of univariate GARCH process:
%                          'GARCH' (default)
%                          'EGARCH'
%                          'GJR' or 'TARCH'
%                          'AGARCH'
%                          'NAGARCH'
%                          'IGARCH'
%             univariate - Structure containing univariate GARCH parameters:
%                          p         - Order of symmetric innovations [1]
%                          q         - Order of lagged conditional variance [1]
%                          distribution - Innovation distribution
%                                      'NORMAL' (default)
%                                      'T' (t-distribution)
%                                      'GED' (Generalized Error Distribution)
%                                      'SKEWT' (Skewed t-distribution)
%             modelP     - Optional output of estimated univariate models 
%                          (will suppress estimation of univariate models)
%             forecast   - Number of periods to forecast, if any [0]
%             algorithm  - Optimization algorithm to use:
%                          'interior-point' (default)
%                          'sqp'
%                          'active-set'
%                          'trust-region'
%             useMEX     - Boolean flag to use MEX implementations [true]
%
% OUTPUTS:
%   Results - A structure with the following fields:
%             parameters  - A K by sum(p,q,1) matrix of univariate parameters
%             correlations - K by K correlation matrix  
%             Ht          - A T by K by K array of conditional covariances
%             Dt          - A T by K matrix of conditional standard deviations
%             R           - K by K constant correlation matrix
%             loglikelihood - The log likelihood at the optimum
%             aic         - Akaike Information Criterion
%             bic         - Bayes Information Criterion
%             diagnostics - Structure containing diagnostic tests
%             forecast    - Structure containing forecasted values (if requested)
%
% COMMENTS:
%   The Constant Conditional Correlation (CCC) GARCH model of Bollerslev (1990)
%   decomposes the conditional covariance matrix into conditional standard deviations
%   and a constant correlation matrix. Each univariate volatility process is estimated
%   separately, and then the constant correlation matrix is computed from the
%   standardized residuals.
%
%   The conditional covariance takes the form:
%   H_t = D_t * R * D_t
%
%   where:
%   - H_t is the conditional covariance matrix at time t
%   - D_t is a diagonal matrix of conditional standard deviations
%   - R is the constant correlation matrix
%
%   The implementation uses a two-stage estimation approach:
%   1. Estimate univariate GARCH models for each series
%   2. Compute the constant correlation matrix from standardized residuals
%
% EXAMPLES:
%   % Estimate a CCC-MVGARCH model with default options
%   results = ccc_mvgarch(returns);
%
%   % Estimate with t-distributed errors and forecast 10 periods ahead
%   options = struct('univariate', struct('distribution', 'T'), 'forecast', 10);
%   results = ccc_mvgarch(returns, options);
%
%   % Use GJR-GARCH for univariate models
%   options = struct('model', 'GJR');
%   results = ccc_mvgarch(returns, options);
%
% REFERENCES:
%   Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange
%   rates: A multivariate generalized ARCH model. Review of Economics and
%   Statistics, 72, 498-505.
%
% See also GARCH, EGARCH, GJR, TARCH, AGARCH, IGARCH

% Copyright: 
% Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Input validation and processing
data = datacheck(data, 'data');

% Get dimensions of data
[T, K] = size(data);

% Process options
if nargin < 2 || isempty(options)
    options = struct();
end

% Set default model type if not specified
if ~isfield(options, 'model') || isempty(options.model)
    options.model = 'GARCH';
else
    options.model = upper(options.model);
end

% Process univariate GARCH options
if ~isfield(options, 'univariate') || isempty(options.univariate)
    options.univariate = struct();
end

% Set default distribution if not specified
if ~isfield(options.univariate, 'distribution') || isempty(options.univariate.distribution)
    options.univariate.distribution = 'NORMAL';
else
    options.univariate.distribution = upper(options.univariate.distribution);
end

% Set default GARCH order (p) if not specified
if ~isfield(options.univariate, 'p') || isempty(options.univariate.p)
    options.univariate.p = 1;
end

% Set default ARCH order (q) if not specified
if ~isfield(options.univariate, 'q') || isempty(options.univariate.q)
    options.univariate.q = 1;
end

% Set default for MEX usage if not specified
if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

% Set default forecast horizon
if ~isfield(options, 'forecast') || isempty(options.forecast)
    options.forecast = 0;
else
    % Verify forecast is non-negative integer
    options.forecast = max(0, round(options.forecast));
end

% Stage 1: Estimate univariate GARCH models for each series
if isfield(options, 'modelP') && ~isempty(options.modelP)
    modelP = options.modelP;
    model_types = cell(K, 1);
    
    if isfield(options, 'model_types') && ~isempty(options.model_types)
        model_types = options.model_types;
    else
        % Default all to the specified model
        for k = 1:K
            model_types{k} = options.model;
        end
    end
    
    useProvidedModels = true;
else
    % No pre-estimated models, need to estimate
    useProvidedModels = false;
    modelP = cell(K, 1);
    model_types = cell(K, 1);
    
    % Configure optimization options
    if isfield(options, 'algorithm')
        optim_options = optimset('Algorithm', options.algorithm, 'Display', 'off');
    else
        optim_options = optimset('Algorithm', 'interior-point', 'Display', 'off');
    end
end

% Initialize arrays for standardized residuals and conditional variances
std_resid = zeros(T, K);
h = zeros(T, K);

% Process each series
for k = 1:K
    % Extract current series
    series_k = data(:, k);
    
    % Set up GARCH options
    garch_options = options.univariate;
    garch_options.model = options.model;
    garch_options.useMEX = options.useMEX;
    
    if ~useProvidedModels
        % Estimate GARCH model if not provided
        
        % Initialize parameters
        starting_params = garchinit(series_k, garch_options);
        
        % Optimize to find GARCH parameters
        [parameters, ~, exitflag] = fmincon(@(p) garchlikelihood(p, series_k, garch_options), ...
            starting_params, [], [], [], [], [], [], [], optim_options);
        
        if exitflag <= 0
            warning('CCC_MVGARCH:Optimization', 'Optimization for series %d did not converge.', k);
        end
        
        % Store parameters
        modelP{k} = parameters;
        model_types{k} = options.model;
    end
    
    % Compute conditional variances using estimated/provided parameters
    h(:, k) = garchcore(modelP{k}, series_k, garch_options);
    
    % Compute standardized residuals
    std_resid(:, k) = series_k ./ sqrt(h(:, k));
end

% Stage 2: Compute constant correlation matrix
R = compute_correlation_matrix(std_resid);

% Compute log-likelihood
ll_options = struct();
ll_options.distribution = options.univariate.distribution;
ll_options.useMEX = options.useMEX;

% Add distribution parameters if available
if strcmpi(ll_options.distribution, 'T') && isfield(options.univariate, 'nu')
    ll_options.nu = options.univariate.nu;
elseif strcmpi(ll_options.distribution, 'GED') && isfield(options.univariate, 'nu')
    ll_options.nu = options.univariate.nu;
elseif strcmpi(ll_options.distribution, 'SKEWT') && isfield(options.univariate, 'skewt_params')
    ll_options.skewt_params = options.univariate.skewt_params;
end

% Calculate log-likelihood with constant correlation matrix
loglikelihood = -ccc_mvgarch_likelihood(std_resid, R, ll_options);

% Calculate information criteria
numParams = 0;
for k = 1:K
    % Count parameters in univariate models
    numParams = numParams + length(modelP{k});
end
% Add parameters for correlation matrix (K*(K-1)/2 free parameters)
numParams = numParams + K*(K-1)/2;

aic = -2*loglikelihood + 2*numParams;
bic = -2*loglikelihood + log(T)*numParams;

% Calculate conditional covariances
Ht = zeros(T, K, K);
for t = 1:T
    % Create diagonal matrix of conditional standard deviations
    Dt = diag(sqrt(h(t, :)));
    
    % Calculate conditional covariance: H_t = D_t * R * D_t
    Ht(t, :, :) = Dt * R * Dt;
end

% Prepare results structure
Results = struct();
Results.parameters = modelP;
Results.model_types = model_types;
Results.correlations = R;
Results.Ht = Ht;
Results.Dt = sqrt(h);
Results.R = R;
Results.loglikelihood = loglikelihood;
Results.aic = aic;
Results.bic = bic;
Results.std_residuals = std_resid;
Results.univariate_models = modelP;

% Perform model validation and diagnostics
Results.diagnostics = validate_ccc_model(Results);

% Generate forecasts if requested
if options.forecast > 0
    Results.forecast = ccc_mvgarch_forecast(Results, options.forecast);
end

end

%--------------------------------------------------------------------------
function [negLogL] = ccc_mvgarch_likelihood(std_residuals, R, options)
% CCC_MVGARCH_LIKELIHOOD Computes the log-likelihood for CCC-MVGARCH model
%
% USAGE:
%   [NEGLOGL] = ccc_mvgarch_likelihood(STD_RESIDUALS, R, OPTIONS)
%
% INPUTS:
%   STD_RESIDUALS - T by K matrix of standardized residuals
%   R             - K by K correlation matrix
%   OPTIONS       - Structure with the following fields:
%                   distribution - Error distribution: 'NORMAL', 'T', 'GED', 'SKEWT'
%                   useMEX       - Boolean flag to use MEX implementations
%
% OUTPUTS:
%   NEGLOGL       - Negative log-likelihood value
%
% See also CCC_MVGARCH, GARCHLIKELIHOOD

% Check if correlation matrix is positive definite
[~, p] = chol(R);
if p > 0
    negLogL = 1e10;
    return;
end

% Get dimensions
[T, K] = size(std_residuals);

% Pre-compute constants for efficiency
logDetR = log(det(R));
invR = inv(R);

% Initialize log-likelihood
logL = 0;

% Calculate log-likelihood based on distribution
switch upper(options.distribution)
    case 'NORMAL'
        % Standard multivariate normal log-likelihood
        constant = -K/2 * log(2*pi);
        for t = 1:T
            zt = std_residuals(t, :)';
            quadForm = zt' * invR * zt;
            logL = logL + constant - 0.5 * logDetR - 0.5 * quadForm;
        end
        
    case 'T'
        % Multivariate t-distribution log-likelihood
        if ~isfield(options, 'nu') || isempty(options.nu)
            options.nu = 8; % Default degrees of freedom
        end
        nu = options.nu;
        
        % Constants for t-distribution
        constant = gammaln((nu + K)/2) - gammaln(nu/2) - K/2*log(pi*(nu-2));
        
        for t = 1:T
            zt = std_residuals(t, :)';
            quadForm = zt' * invR * zt;
            logL = logL + constant - 0.5 * logDetR - (nu + K)/2 * log(1 + quadForm/(nu-2));
        end
        
    case 'GED'
        % For GED, utilize univariate GED functions for simplicity
        % This is an approximation as a true multivariate GED is more complex
        if ~isfield(options, 'nu') || isempty(options.nu)
            options.nu = 1.5; % Default shape parameter
        end
        nu = options.nu;
        
        for t = 1:T
            % Transform to uncorrelated space
            zt = std_residuals(t, :)';
            transformed = invR^(1/2) * zt;
            
            % Compute likelihood with univariate GED for each component
            ll_t = 0;
            for k = 1:K
                ll_t = ll_t + gedloglik(transformed(k), nu, 0, 1);
            end
            
            % Add correlation term
            logL = logL + ll_t - 0.5 * logDetR;
        end
        
    case 'SKEWT'
        % For skewed t, use univariate approximation for simplicity
        if ~isfield(options, 'skewt_params') || isempty(options.skewt_params)
            options.skewt_params = [8, 0]; % Default [nu, lambda]
        end
        nu = options.skewt_params(1);
        lambda = options.skewt_params(2);
        
        for t = 1:T
            % Transform to uncorrelated space
            zt = std_residuals(t, :)';
            transformed = invR^(1/2) * zt;
            
            % Compute likelihood with univariate skewed t for each component
            ll_t = 0;
            for k = 1:K
                [~, ll_k] = skewtloglik(transformed(k), [nu, lambda, 0, 1]);
                ll_t = ll_t + sum(ll_k);
            end
            
            % Add correlation term
            logL = logL + ll_t - 0.5 * logDetR;
        end
        
    otherwise
        error('Unknown distribution type: %s', options.distribution);
end

% Use MEX implementation if available and requested
global MEX_AVAILABLE
if options.useMEX && MEX_AVAILABLE && exist('composite_likelihood', 'file') == 3 % MEX file exists
    % In practice, this would call the MEX function for faster computation
    % This is just a placeholder - actual MEX implementation would depend on
    % the specifics of the composite_likelihood function
end

% Return negative log-likelihood for minimization
negLogL = -logL;
end

%--------------------------------------------------------------------------
function R = compute_correlation_matrix(std_residuals)
% COMPUTE_CORRELATION_MATRIX Calculates constant correlation matrix from standardized residuals
%
% USAGE:
%   R = compute_correlation_matrix(STD_RESIDUALS)
%
% INPUTS:
%   STD_RESIDUALS - T by K matrix of standardized residuals
%
% OUTPUTS:
%   R             - K by K correlation matrix
%
% See also CCC_MVGARCH, CORR

% Compute correlation matrix
try
    % Use Statistics Toolbox's corr function if available
    R = corr(std_residuals);
catch
    % Manual correlation calculation if Statistics Toolbox is not available
    [T, K] = size(std_residuals);
    R = zeros(K, K);
    
    % Compute correlations manually
    for i = 1:K
        for j = 1:K
            if i == j
                R(i, j) = 1;
            else
                R(i, j) = (std_residuals(:, i)' * std_residuals(:, j)) / T;
                R(j, i) = R(i, j);
            end
        end
    end
end

% Ensure the correlation matrix is positive definite
[~, p] = chol(R);
if p > 0
    % If not positive definite, apply a correction
    % Method: shrink toward identity
    shrinkage = 0.01;
    while p > 0 && shrinkage < 1
        R_new = (1 - shrinkage) * R + shrinkage * eye(size(R));
        [~, p] = chol(R_new);
        shrinkage = shrinkage + 0.01;
    end
    R = R_new;
    warning('CCC_MVGARCH:CorrelationMatrix', 'Correlation matrix was not positive definite. Applied shrinkage correction.');
end

% Ensure diagonal is exactly 1
for i = 1:size(R, 1)
    R(i, i) = 1;
end
end

%--------------------------------------------------------------------------
function Forecast = ccc_mvgarch_forecast(model, horizon)
% CCC_MVGARCH_FORECAST Generates forecasts for CCC-MVGARCH model
%
% USAGE:
%   FORECAST = ccc_mvgarch_forecast(MODEL, HORIZON)
%
% INPUTS:
%   MODEL   - Structure with CCC-MVGARCH model parameters and results
%   HORIZON - Number of periods to forecast
%
% OUTPUTS:
%   FORECAST - Structure with forecasted values:
%              Ht - HORIZON by K by K array of conditional covariances
%              Dt - HORIZON by K matrix of conditional std. deviations
%
% See also CCC_MVGARCH, GARCH

% Get dimensions
K = size(model.R, 1);

% Initialize forecast arrays
Ht_forecast = zeros(horizon, K, K);
Dt_forecast = zeros(horizon, K);

% Get the constant correlation matrix
R = model.R;

% For each series, generate univariate variance forecasts
for k = 1:K
    % Extract univariate model parameters
    params = model.parameters{k};
    
    % Determine model type
    if isfield(model, 'model_types')
        model_type = model.model_types{k};
    else
        % Default to GARCH if not specified
        model_type = 'GARCH';
    end
    
    % Extract last observations for initialization
    last_h = model.Dt(end, k)^2;
    last_data = model.std_residuals(end, k) * model.Dt(end, k);
    
    % Generate univariate variance forecasts based on model type
    h_forecast = zeros(horizon, 1);
    
    switch upper(model_type)
        case 'GARCH'
            % Standard GARCH(1,1) forecast
            omega = params(1);
            alpha = params(2);
            beta = params(3);
            
            h_t = last_h;
            for t = 1:horizon
                if t == 1
                    h_t = omega + alpha * last_data^2 + beta * last_h;
                else
                    h_t = omega + (alpha + beta) * h_t;
                end
                h_forecast(t) = h_t;
            end
            
        case {'GJR', 'TARCH'}
            % GJR/TARCH with asymmetry
            omega = params(1);
            alpha = params(2);
            gamma = params(3);
            beta = params(4);
            
            h_t = last_h;
            for t = 1:horizon
                if t == 1
                    % For h_{T+1}, use last observed data with asymmetry if negative
                    asymmetry = (last_data < 0) * gamma * last_data^2;
                    h_t = omega + alpha * last_data^2 + asymmetry + beta * last_h;
                else
                    % For h_{T+h}, h > 1, use unconditional expectation
                    % E[e_t^2] = h_t and E[e_t^2 * I(e_t < 0)] ≈ 0.5 * h_t
                    h_t = omega + alpha * h_t + 0.5 * gamma * h_t + beta * h_t;
                end
                h_forecast(t) = h_t;
            end
            
        case 'EGARCH'
            % EGARCH with asymmetry
            omega = params(1);
            alpha = params(2);
            gamma = params(3);
            beta = params(4);
            
            log_h_t = log(last_h);
            for t = 1:horizon
                if t == 1
                    % For log(h_{T+1})
                    std_resid = last_data / sqrt(last_h);
                    abs_std_resid = abs(std_resid);
                    expected_abs = sqrt(2/pi); % E[|z|] for standard normal
                    
                    log_h_t = omega + alpha * (abs_std_resid - expected_abs) + ...
                              gamma * std_resid + beta * log_h_t;
                else
                    % For log(h_{T+h}), h > 1
                    % E[|z_t|] = sqrt(2/pi), E[z_t] = 0
                    log_h_t = omega + beta * log_h_t;
                end
                h_forecast(t) = exp(log_h_t);
            end
            
        case 'AGARCH'
            % Asymmetric GARCH
            omega = params(1);
            alpha = params(2);
            gamma = params(3); % asymmetry parameter
            beta = params(4);
            
            h_t = last_h;
            for t = 1:horizon
                if t == 1
                    % For h_{T+1}, use last observed data
                    h_t = omega + alpha * (last_data - gamma)^2 + beta * last_h;
                else
                    % For h_{T+h}, h > 1, use unconditional expectation
                    h_t = omega + alpha * (h_t + gamma^2) + beta * h_t;
                end
                h_forecast(t) = h_t;
            end
            
        case 'IGARCH'
            % Integrated GARCH
            omega = params(1);
            alpha = params(2:(end-1)); % IGARCH has constrained betas
            
            h_t = last_h;
            for t = 1:horizon
                if t == 1
                    % For h_{T+1}
                    h_t = omega + alpha(1) * last_data^2 + (1 - alpha(1)) * last_h;
                else
                    % For h_{T+h}, h > 1
                    h_t = omega + h_t;
                end
                h_forecast(t) = h_t;
            end
            
        otherwise
            % For other models, use simple persistence approach
            omega = params(1);
            persistence = 0.95; % Default persistence if model not recognized
            
            unconditional = omega / (1 - persistence);
            h_t = last_h;
            
            for t = 1:horizon
                h_t = omega + persistence * (h_t - omega) + ...
                      (1 - persistence) * (unconditional - omega);
                h_forecast(t) = h_t;
            end
    end
    
    % Store standard deviations
    Dt_forecast(:, k) = sqrt(h_forecast);
end

% Compute covariance matrices for each forecast horizon
for t = 1:horizon
    % Create diagonal matrix of forecasted standard deviations
    Dt = diag(Dt_forecast(t, :));
    
    % Calculate forecasted covariance: H_t = D_t * R * D_t
    Ht_forecast(t, :, :) = Dt * R * Dt;
end

% Return forecast structure
Forecast = struct();
Forecast.Ht = Ht_forecast;
Forecast.Dt = Dt_forecast;
end

%--------------------------------------------------------------------------
function diagnostics = validate_ccc_model(model)
% VALIDATE_CCC_MODEL Validates CCC-MVGARCH model and computes diagnostics
%
% USAGE:
%   DIAGNOSTICS = validate_ccc_model(MODEL)
%
% INPUTS:
%   MODEL      - Structure with CCC-MVGARCH model parameters and results
%
% OUTPUTS:
%   DIAGNOSTICS - Structure with validation results and diagnostics
%
% See also CCC_MVGARCH

% Initialize diagnostics structure
diagnostics = struct();

% Check stationarity of univariate GARCH models
K = size(model.R, 1);
stationarity = zeros(K, 1);

for k = 1:K
    % Extract univariate model parameters
    params = model.parameters{k};
    
    % Determine model type
    if isfield(model, 'model_types')
        model_type = model.model_types{k};
    else
        % Default to GARCH if not specified
        model_type = 'GARCH';
    end
    
    % Check stationarity based on model type
    switch upper(model_type)
        case 'GARCH'
            % For GARCH, check if alpha + beta < 1
            if length(params) < 3
                stationarity(k) = 1; % Assume stationary if params missing
            else
                alpha = params(2);
                beta = params(3);
                stationarity(k) = (alpha + beta < 1);
            end
            
        case {'GJR', 'TARCH'}
            % For GJR/TARCH, check if alpha + 0.5*gamma + beta < 1
            if length(params) < 4
                stationarity(k) = 1; % Assume stationary if params missing
            else
                alpha = params(2);
                gamma = params(3);
                beta = params(4);
                stationarity(k) = (alpha + 0.5*gamma + beta < 1);
            end
            
        case 'EGARCH'
            % For EGARCH, check if |beta| < 1
            if length(params) < 4
                stationarity(k) = 1; % Assume stationary if params missing
            else
                beta = params(4);
                stationarity(k) = (abs(beta) < 1);
            end
            
        case 'IGARCH'
            % IGARCH is non-stationary by design, but check if params valid
            stationarity(k) = 0; % Mark as non-stationary
            
        otherwise
            % For other models, assume stationary if missing info
            stationarity(k) = 1;
    end
end

diagnostics.stationarity = stationarity;
diagnostics.all_stationary = all(stationarity);

% Check if correlation matrix is positive definite
[~, p] = chol(model.R);
diagnostics.correlation_pd = (p == 0);

% Additional diagnostics could be added here
% These would typically include tests for remaining ARCH effects,
% tests for cross-correlation of squared standardized residuals, etc.

end