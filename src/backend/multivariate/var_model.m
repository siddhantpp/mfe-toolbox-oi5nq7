function [model] = var_model(y, p, options)
% VAR_MODEL Estimates a Vector Autoregression (VAR) model for multivariate time series analysis
%
% USAGE:
%   MODEL = var_model(Y, P)
%   MODEL = var_model(Y, P, OPTIONS)
%
% INPUTS:
%   Y       - T by K matrix of time series data where T is the number of observations
%             and K is the number of variables
%   P       - Non-negative integer or vector of integers representing the VAR lag order(s)
%             If P is a vector, the function selects the optimal lag using information criteria
%   OPTIONS - [OPTIONAL] Structure with estimation options:
%               OPTIONS.constant  - [OPTIONAL] Logical indicating whether to include
%                                   a constant term. Default is true.
%               OPTIONS.trend     - [OPTIONAL] Logical indicating whether to include
%                                   a linear trend. Default is false.
%               OPTIONS.criterion - [OPTIONAL] String indicating information criterion 
%                                   for model selection if P is a vector. Options are 
%                                   'aic' or 'sbic' (default).
%
% OUTPUTS:
%   MODEL   - Structure containing estimation results:
%               MODEL.coefficients    - Estimated VAR coefficients
%               MODEL.constant        - Estimated constant terms (if included)
%               MODEL.trend           - Estimated trend coefficients (if included)
%               MODEL.residuals       - Model residuals
%               MODEL.sigma           - Residual covariance matrix
%               MODEL.fitted          - Fitted values
%               MODEL.logL            - Log-likelihood
%               MODEL.aic             - Akaike Information Criterion
%               MODEL.sbic            - Schwarz Bayesian Information Criterion
%               MODEL.ljungbox        - Ljung-Box test on VAR residuals
%               MODEL.y               - Original dependent variable
%               MODEL.X               - Design matrix
%               MODEL.p               - VAR lag order
%               MODEL.k               - Number of variables
%               MODEL.T               - Number of observations
%               MODEL.nparams         - Number of parameters
%               MODEL.df              - Degrees of freedom
%               MODEL.options         - Estimation options used
%
% COMMENTS:
%   The VAR(p) model is specified as:
%   y_t = c + A_1*y_{t-1} + A_2*y_{t-2} + ... + A_p*y_{t-p} + u_t
%
%   Where:
%   - y_t is a K×1 vector of variables at time t
%   - c is a K×1 vector of constants
%   - A_i are K×K coefficient matrices for i=1,2,...,p
%   - u_t is a K×1 vector of innovations with E[u_t]=0 and E[u_t*u_t']=Σ
%
%   The function allows for model selection by providing a vector of lag orders P
%   and choosing the best model according to the specified information criterion.
%
% EXAMPLES:
%   % Basic VAR(2) model with 3 variables
%   model = var_model(data, 2);
%
%   % VAR model with automatic lag selection (up to 4)
%   model = var_model(data, 1:4, struct('criterion', 'aic'));
%
%   % VAR model with trend but no constant
%   model = var_model(data, 2, struct('constant', false, 'trend', true));
%
%   % Generate forecasts from the estimated model
%   forecasts = var_forecast(model, 10);
%
%   % Compute impulse responses
%   irf = var_irf(model, 20);
%
%   % Compute variance decompositions
%   fevd = var_fevd(model, 20);
%
% See also var_forecast, var_irf, var_fevd, datacheck, parametercheck
%
% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Set default options
defaultOptions = struct('constant', true, ...
                       'trend', false, ...
                       'criterion', 'sbic');

% Merge default options with user options
if nargin < 3 || isempty(options)
    options = defaultOptions;
else
    fields = fieldnames(defaultOptions);
    for i=1:length(fields)
        if ~isfield(options, fields{i})
            options.(fields{i}) = defaultOptions.(fields{i});
        end
    end
end

% Validate inputs
y = datacheck(y, 'y');

% Validate lag parameter
lagOptions = struct('isInteger', true, 'isNonNegative', true);
p = parametercheck(p, 'p', lagOptions);

% Get dimensions
[T, k] = size(y);

% If p is a vector, estimate models for each lag and select the best
if length(p) > 1
    bestCriterion = Inf;
    bestLag = p(1);
    bestModel = [];
    
    for i = 1:length(p)
        thisLag = p(i);
        
        % Skip if not enough observations for this lag
        if T <= thisLag*k
            continue;
        end
        
        % Estimate model with current lag
        lagOptions = options;
        tmpModel = var_model(y, thisLag, lagOptions);
        
        % Select criterion
        if strcmpi(options.criterion, 'aic')
            thisCriterion = tmpModel.aic;
        else  % default is 'sbic'
            thisCriterion = tmpModel.sbic;
        end
        
        % Update best model if this one is better
        if thisCriterion < bestCriterion
            bestCriterion = thisCriterion;
            bestLag = thisLag;
            bestModel = tmpModel;
        end
    end
    
    % Check if a valid model was found
    if isempty(bestModel)
        error('No valid model could be estimated with the provided lag orders.');
    end
    
    % Return the best model with lag selection info
    model = bestModel;
    model.lagSelection = struct('criterion', options.criterion, ...
                               'lags', p, ...
                               'bestLag', bestLag);
    return;
end

% Check if enough observations for estimation
if T <= p*k
    error('Not enough observations for VAR(%d) model with %d variables', p, k);
end

% Ensure data is in column format
y = columncheck(y, 'y');

% Set up the design matrix X
X = [];
startObs = p + 1;

% Add lagged variables
for i = 1:p
    X = [X, y((startObs-i):(T-i), :)];
end

% Add constant term if specified
if options.constant
    X = [X, ones(T-p, 1)];
end

% Add trend term if specified
if options.trend
    X = [X, ((1:T-p)')];
end

% Dependent variable
Y = y(startObs:T, :);

% Number of observations in estimation
Teff = T - p;

% Number of parameters per equation
nParam = p*k;
if options.constant
    nParam = nParam + 1;
end
if options.trend
    nParam = nParam + 1;
end

% Total number of parameters
totalParam = nParam * k;

% Estimate VAR coefficients using least squares (equation by equation)
% Use Moore-Penrose pseudoinverse for numerical stability
B = pinv(X)*Y;

% Compute residuals
residuals = Y - X*B;

% Estimate residual covariance matrix
sigma = (residuals'*residuals) / (Teff - nParam);

% Compute fitted values
fitted = X*B;

% Compute log-likelihood (assuming Gaussian errors)
logL = -0.5 * Teff * (k * log(2*pi) + log(det(sigma)) + k);

% Compute information criteria
ic = aicsbic(logL, totalParam, Teff);

% Perform Ljung-Box test on residuals (for each variable)
ljungBoxResults = cell(k, 1);
for i = 1:k
    ljungBoxResults{i} = ljungbox(residuals(:, i), min(10, floor(Teff/5)), p);
end

% Format coefficient matrix for easier interpretation
coeffMatrix = zeros(k, k*p + options.constant + options.trend);
coeffMatrix(:, 1:k*p) = reshape(B(1:k*p, :)', k, k*p);

% Extract constant and trend if included
offset = k*p;
constantCoeff = [];
trendCoeff = [];

if options.constant
    constantCoeff = B(offset+1, :)';
    offset = offset + 1;
end

if options.trend
    trendCoeff = B(offset+1, :)';
end

% Extract coefficient matrices for each lag
A = cell(p, 1);
for i = 1:p
    A{i} = reshape(B((i-1)*k+1:i*k, :)', k, k);
end

% Create output structure
model = struct();
model.coefficients = coeffMatrix; % Formatted coefficient matrix
model.A = A;                     % Coefficient matrices for each lag
model.B = B;                     % Raw coefficient matrix
if options.constant
    model.constant = constantCoeff;
end
if options.trend
    model.trend = trendCoeff;
end
model.residuals = residuals;
model.sigma = sigma;
model.fitted = fitted;
model.logL = logL;
model.aic = ic.aic;
model.sbic = ic.sbic;
model.ljungbox = ljungBoxResults;
model.y = y;
model.X = X;
model.p = p;
model.k = k;
model.T = T;
model.Teff = Teff;
model.nparams = totalParam;
model.df = Teff - nParam;
model.options = options;

end

function forecasts = var_forecast(model, h)
% VAR_FORECAST Generates forecasts from an estimated VAR model
%
% USAGE:
%   FORECASTS = var_forecast(MODEL, H)
%
% INPUTS:
%   MODEL     - Structure containing VAR model estimation results from var_model
%   H         - Positive integer indicating forecast horizon
%
% OUTPUTS:
%   FORECASTS - H by K matrix of forecasted values for T+1 to T+H
%
% COMMENTS:
%   This function generates out-of-sample forecasts for a VAR model estimated 
%   using var_model. The forecasts are generated recursively using the estimated
%   VAR coefficient matrices.
%
% EXAMPLES:
%   % Generate 10-step ahead forecasts
%   forecasts = var_forecast(model, 10);
%
%   % Plot original series and forecasts
%   T = size(model.y, 1);
%   figure;
%   for i=1:model.k
%       subplot(model.k, 1, i);
%       plot([model.y(:,i); forecasts(:,i)]);
%       hold on;
%       line([T, T], get(gca, 'YLim'), 'Color', 'r', 'LineStyle', '--');
%       title(['Variable ', num2str(i)]);
%   end
%
% See also var_model, var_irf, var_fevd

% Validate forecast horizon
hOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', hOptions);

% Extract model parameters
y = model.y;
p = model.p;
k = model.k;
T = model.T;
A = model.A;  % Coefficient matrices for each lag
options = model.options;

% Get constant and trend if included
if options.constant
    constant = model.constant;
else
    constant = zeros(k, 1);
end

if options.trend
    trend = model.trend;
else
    trend = zeros(k, 1);
end

% Last observed values to start forecast
lastObs = y(T-p+1:T, :);

% Initialize forecast array
forecasts = zeros(h, k);

% Generate forecasts recursively
for t = 1:h
    % Initialize forecast for time T+t
    forecastT = constant;
    
    % Add trend component if included
    if options.trend
        forecastT = forecastT + trend * (T + t);
    end
    
    % Add VAR dynamics (AR terms)
    for i = 1:p
        if t > i
            % Use previous forecasts for lags that exceed available data
            forecastT = forecastT + A{i} * forecasts(t-i, :)';
        else
            % Use actual data for initial lags
            idx = p - i + t;
            forecastT = forecastT + A{i} * lastObs(idx, :)';
        end
    end
    
    % Store forecast
    forecasts(t, :) = forecastT';
end

end

function irf = var_irf(model, h)
% VAR_IRF Computes impulse response functions for a VAR model
%
% USAGE:
%   IRF = var_irf(MODEL, H)
%
% INPUTS:
%   MODEL - Structure containing VAR model estimation results from var_model
%   H     - Positive integer indicating the horizon for impulse responses
%
% OUTPUTS:
%   IRF   - (H+1) by K by K array where IRF(t+1,i,j) is the response of
%           variable i to a one unit shock in variable j, t periods ago
%
% COMMENTS:
%   This function computes impulse response functions (IRFs) for a VAR model.
%   The IRFs show the dynamic responses of each variable in the system to a
%   one-unit shock in each variable. The function uses a Cholesky decomposition
%   to orthogonalize the innovations, which means that the ordering of variables
%   in the original model matters.
%
% EXAMPLES:
%   % Compute impulse responses for 20 periods
%   irf = var_irf(model, 20);
%
%   % Plot impulse responses
%   k = model.k;
%   figure;
%   for i=1:k
%       for j=1:k
%           subplot(k, k, (i-1)*k+j);
%           plot(0:20, squeeze(irf(:,i,j)));
%           title(['Response of ', num2str(i), ' to ', num2str(j)]);
%       end
%   end
%
% See also var_model, var_forecast, var_fevd

% Validate horizon
hOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', hOptions);

% Extract model parameters
p = model.p;
k = model.k;
A = model.A;  % Coefficient matrices for each lag
sigma = model.sigma;

% Initialize impulse responses
% IRF is an (h+1) x k x k array where:
% IRF(t+1,i,j) = response of variable i to a shock in variable j, t periods ago
irf = zeros(h+1, k, k);

% Cholesky decomposition to orthogonalize innovations
P = chol(sigma)';  % Lower triangular matrix P such that PP' = sigma

% Impact matrix (responses at horizon 0)
irf(1,:,:) = eye(k);

% Compute impulse responses recursively
Phi = cell(h+1, 1);
Phi{1} = eye(k);  % Phi_0 is identity matrix

for t = 1:h
    Phi_t = zeros(k, k);
    
    % Apply the recursion formula for Phi
    for j = 1:min(t, p)
        Phi_t = Phi_t + A{j} * Phi{t-j+1};
    end
    
    Phi{t+1} = Phi_t;
    
    % Compute orthogonalized impulse responses
    irf(t+1,:,:) = Phi_t * P;
end

end

function fevd = var_fevd(model, h)
% VAR_FEVD Computes forecast error variance decomposition for a VAR model
%
% USAGE:
%   FEVD = var_fevd(MODEL, H)
%
% INPUTS:
%   MODEL - Structure containing VAR model estimation results from var_model
%   H     - Positive integer indicating the forecast horizon for decomposition
%
% OUTPUTS:
%   FEVD  - (H+1) by K by K array where FEVD(t+1,i,j) is the proportion of the
%           forecast error variance of variable i at horizon t that is due to
%           shocks to variable j
%
% COMMENTS:
%   This function computes the forecast error variance decomposition (FEVD) for
%   a VAR model. The FEVD shows the proportion of the forecast error variance of
%   each variable that is due to shocks to each variable in the system at different
%   forecast horizons. The function relies on the impulse response functions and
%   uses the same Cholesky decomposition for identification.
%
% EXAMPLES:
%   % Compute variance decomposition for 20 periods
%   fevd = var_fevd(model, 20);
%
%   % Plot variance decomposition for the first variable
%   k = model.k;
%   figure;
%   area(0:20, squeeze(fevd(:,1,:)));
%   title('Variance Decomposition for Variable 1');
%   legend(arrayfun(@(x) ['Variable ' num2str(x)], 1:k, 'UniformOutput', false));
%
% See also var_model, var_forecast, var_irf

% Validate horizon
hOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', hOptions);

% Extract model parameters
k = model.k;

% Compute impulse responses
irf = var_irf(model, h);

% Initialize variance decomposition
fevd = zeros(h+1, k, k);

% Compute variance decomposition
for i = 1:k  % For each variable
    % Compute total forecast error variance at each horizon
    totalVariance = zeros(h+1, 1);
    
    for t = 1:h+1  % For each horizon (1 = impact)
        for l = 1:t  % Sum over all response periods up to t
            for j = 1:k  % Sum over all shocks
                totalVariance(t) = totalVariance(t) + irf(l,i,j)^2;
            end
        end
    end
    
    % Compute contribution of each shock to the forecast error variance
    for j = 1:k  % For each shock
        varianceDue = zeros(h+1, 1);
        
        for t = 1:h+1  % For each horizon
            for l = 1:t  % Sum over all response periods up to t
                varianceDue(t) = varianceDue(t) + irf(l,i,j)^2;
            end
            
            % Compute the proportion
            if totalVariance(t) > 0
                fevd(t,i,j) = varianceDue(t) / totalVariance(t);
            else
                fevd(t,i,j) = 0;
            end
        end
    end
end

end