function [model] = vecm_model(y, p, r, options)
% VECM_MODEL Estimates Vector Error Correction Model (VECM) for cointegrated multivariate time series.
%
% USAGE:
%   MODEL = vecm_model(Y, P, R)
%   MODEL = vecm_model(Y, P, R, OPTIONS)
%
% INPUTS:
%   Y       - T by K matrix of time series data where T is the number of observations
%             and K is the number of variables
%   P       - Non-negative integer representing the VAR lag order (VECM will use P-1 lags)
%   R       - Cointegration rank (number of cointegrating relationships, 0 <= R < K)
%   OPTIONS - [OPTIONAL] Structure with estimation options:
%               OPTIONS.det     - [OPTIONAL] Deterministic term specification:
%                                 0: No deterministic terms
%                                 1: Restricted constant (in cointegration)
%                                 2: Unrestricted constant
%                                 3: Restricted trend and unrestricted constant
%                                 4: Unrestricted trend
%                                 Default is 2 (unrestricted constant).
%               OPTIONS.test    - [OPTIONAL] Method for cointegration rank testing:
%                                 'trace': Trace test (default)
%                                 'maxeig': Maximum eigenvalue test
%               OPTIONS.alpha   - [OPTIONAL] Significance level for cointegration test
%                                 Default is 0.05 (5%)
%
% OUTPUTS:
%   MODEL   - Structure containing estimation results:
%               MODEL.alpha         - Adjustment coefficients (K x R)
%               MODEL.beta          - Cointegrating vectors (K x R)
%               MODEL.gamma         - Short-run dynamics coefficients
%               MODEL.Pi            - Long-run impact matrix (alpha * beta')
%               MODEL.mu            - Deterministic terms coefficients (if included)
%               MODEL.residuals     - Model residuals
%               MODEL.sigma         - Residual covariance matrix
%               MODEL.fitted        - Fitted values
%               MODEL.logL          - Log-likelihood
%               MODEL.aic           - Akaike Information Criterion
%               MODEL.sbic          - Schwarz Bayesian Information Criterion
%               MODEL.eigenvalues   - Eigenvalues from cointegration analysis
%               MODEL.eigenvectors  - Eigenvectors from cointegration analysis
%               MODEL.cointegration - Cointegration test results (if requested)
%               MODEL.y             - Original dependent variable
%               MODEL.dy            - First differences of y
%               MODEL.p             - VAR lag order (VECM has p-1 lags)
%               MODEL.r             - Cointegration rank
%               MODEL.k             - Number of variables
%               MODEL.T             - Number of observations
%               MODEL.Teff          - Effective sample size after lag adjustment
%               MODEL.nparams       - Number of estimated parameters
%               MODEL.det           - Deterministic term specification
%               MODEL.options       - Estimation options used
%
% COMMENTS:
%   The VECM(p-1) model is specified as:
%   Δy_t = αβ'y_{t-1} + Γ_1Δy_{t-1} + ... + Γ_{p-1}Δy_{t-p+1} + μ + ε_t
%
%   Where:
%   - y_t is a K×1 vector of variables at time t
%   - Δy_t is the first difference of y_t
%   - α is a K×R matrix of adjustment coefficients
%   - β is a K×R matrix of cointegrating vectors
%   - Γ_i are K×K matrices of short-run dynamics coefficients
%   - μ contains deterministic terms (constant, trend)
%   - ε_t is a K×1 vector of innovations with E[ε_t]=0 and E[ε_t*ε_t']=Σ
%
%   The function implements Johansen's procedure for cointegration analysis.
%
% EXAMPLES:
%   % VECM model with 2 lags (p=3 for equivalent VAR) and 1 cointegrating relationship
%   model = vecm_model(data, 3, 1);
%
%   % VECM model with restricted constant
%   model = vecm_model(data, 2, 1, struct('det', 1));
%
%   % Generate forecasts from the estimated model
%   forecasts = vecm_forecast(model, 10);
%
% See also johansen_test, vecm_forecast, vecm_irf, vecm_to_var

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Set default options
defaultOptions = struct('det', 2, ...
                       'test', 'trace', ...
                       'alpha', 0.05);

% Merge default options with user options
if nargin < 4 || isempty(options)
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

% Ensure y is in matrix form
if size(y, 2) == 1
    error('VECM requires multivariate time series data (at least 2 variables)');
end

% Get dimensions
[T, k] = size(y);

% Validate VAR lag order (p)
pOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
p = parametercheck(p, 'p', pOptions);

% Validate cointegration rank (r)
rOptions = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true, 'upperBound', k-1);
r = parametercheck(r, 'r', rOptions);

% Validate deterministic term specification
detOptions = struct('isscalar', true, 'isInteger', true, 'lowerBound', 0, 'upperBound', 4);
options.det = parametercheck(options.det, 'options.det', detOptions);

% Check if enough observations for estimation
if T <= p*k
    error('Not enough observations for VECM model with VAR(%d) lag structure and %d variables', p, k);
end

% Compute first differences
dy = diff(y);

% Prepare data for estimation
yLag = y(p:end-1, :);       % Lagged levels for error correction term
dyData = dy(p-1:end, :);    % Current differences (dependent variable)

% Build matrices of lagged differences
X = [];
if p > 1
    for i = 1:(p-1)
        X = [X, dy(p-i:end-i, :)];
    end
end

% Add deterministic terms based on specification
switch options.det
    case 0  % No deterministic terms
        % No additional terms
        
    case 1  % Restricted constant (in cointegration)
        % Add constant to yLag (will be part of cointegration)
        yLag = [yLag, ones(size(yLag, 1), 1)];
        
    case 2  % Unrestricted constant
        % Add constant to X
        X = [X, ones(size(dyData, 1), 1)];
        
    case 3  % Restricted trend and unrestricted constant
        % Add trend to yLag (part of cointegration) and constant to X
        trend = (1:size(yLag, 1))';
        yLag = [yLag, trend];
        X = [X, ones(size(dyData, 1), 1)];
        
    case 4  % Unrestricted trend
        % Add trend and constant to X
        trend = (1:size(dyData, 1))';
        X = [X, ones(size(dyData, 1), 1), trend];
end

% Effective sample size
Teff = size(dyData, 1);

% Adjust dimensions of lagged variables to match dependent variable
if size(yLag, 1) > size(dyData, 1)
    yLag = yLag(end-size(dyData, 1)+1:end, :);
end

% Perform reduced rank regression
if isempty(X)
    % No regressors other than error correction term
    M0 = dyData;
    M1 = yLag;
else
    % Regress dyData and yLag on X to get residuals
    M0 = dyData - X * (X \ dyData);
    M1 = yLag - X * (X \ yLag);
end

% Compute moment matrices
S00 = (M0' * M0) / Teff;
S11 = (M1' * M1) / Teff;
S01 = (M0' * M1) / Teff;
S10 = S01';

% Compute canonical correlations using SVD for numerical stability
try
    % Try Cholesky decomposition first (more efficient)
    S00_chol = chol(S00)';
    S11_chol = chol(S11)';
    S00_inv_chol = inv(S00_chol);
    S11_inv_chol = inv(S11_chol);
    C = S11_inv_chol' * S10 * S00_inv_chol * S00_inv_chol' * S01 * S11_inv_chol;
catch
    % If Cholesky fails, use SVD approach
    [U00, D00, V00] = svd(S00);
    [U11, D11, V11] = svd(S11);
    
    % Filter small eigenvalues for numerical stability
    tol = max(size(S00)) * eps(max(diag(D00)));
    idx00 = diag(D00) > tol;
    tol = max(size(S11)) * eps(max(diag(D11)));
    idx11 = diag(D11) > tol;
    
    D00_inv_sqrt = zeros(size(D00));
    D00_inv_sqrt(idx00, idx00) = diag(1./sqrt(diag(D00(idx00, idx00))));
    
    D11_inv_sqrt = zeros(size(D11));
    D11_inv_sqrt(idx11, idx11) = diag(1./sqrt(diag(D11(idx11, idx11))));
    
    S00_inv_sqrt = V00 * D00_inv_sqrt * U00';
    S11_inv_sqrt = V11 * D11_inv_sqrt * U11';
    
    C = S11_inv_sqrt * S10 * S00_inv_sqrt * S00_inv_sqrt' * S01 * S11_inv_sqrt;
end

% Get eigenvalues and eigenvectors
[eigenvectors, eigenvalues_matrix] = eig(C);
eigenvalues = diag(eigenvalues_matrix);

% Sort eigenvalues in descending order
[eigenvalues, idx] = sort(real(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, idx);

% Initialize model components
alpha = [];
beta = [];
Pi = [];
Gamma = [];
residuals = [];
fitted = [];
sigma = [];
mu = [];

if r > 0
    % Extract the first r eigenvectors
    V = eigenvectors(:, 1:r);
    lambda = eigenvalues(1:r);
    
    % Compute beta (cointegrating vectors)
    if exist('S11_inv_sqrt', 'var')
        beta = S11_inv_sqrt * V;
    else
        % Alternative computation using the canonical variates
        beta = (S11 \ S10) * (S00 \ S01) * eigenvectors(:, 1:r);
        % Normalize beta
        for i = 1:r
            beta(:, i) = beta(:, i) / norm(beta(:, i));
        end
    end
    
    % Normalize beta (first r rows form an identity matrix if possible)
    if options.det == 0 || options.det == 2 || options.det == 4
        % For specifications without restricted deterministic terms in cointegration
        beta_norm = [];
        if k >= r
            top_beta = beta(1:r, 1:r);
            if rcond(top_beta) > eps
                beta_norm = top_beta \ eye(r);
            end
        end
        
        if isempty(beta_norm) || rcond(beta_norm) < eps
            % If normalization fails, use a different approach
            for i = 1:r
                beta(:, i) = beta(:, i) / norm(beta(:, i));
            end
        else
            beta = beta * beta_norm;
        end
    else
        % For specifications with restricted deterministic terms in cointegration
        beta_norm = [];
        if k >= r
            top_beta = beta(1:k, 1:r);
            if rcond(top_beta' * top_beta) > eps
                % Find the maximum magnitude elements for normalization
                [~, max_idx] = max(abs(top_beta));
                norm_beta = zeros(r, r);
                for i = 1:r
                    norm_beta(i, i) = 1 / top_beta(max_idx(i), i);
                end
                beta = beta * norm_beta;
            end
        end
        
        if isempty(beta_norm)
            % If normalization fails, use a different approach
            for i = 1:r
                beta(:, i) = beta(:, i) / norm(beta(:, i));
            end
        end
    end
    
    % Compute alpha (adjustment coefficients)
    if ~isempty(X)
        % Compute Pi = alpha*beta'
        Pi = S01 * (S11 \ beta) * beta';
        
        % Extract alpha from Pi
        alpha = Pi * beta * inv(beta' * beta);
        
        % Estimate short-run dynamics (Gamma)
        Y_adjusted = dyData - yLag * Pi';
        Gamma = X \ Y_adjusted;
        
        % Compute residuals
        residuals = dyData - yLag * Pi' - X * Gamma;
    else
        % Simpler case without lagged differences or deterministic terms
        Pi = S01 * (S11 \ beta) * beta';
        alpha = Pi * beta * inv(beta' * beta);
        residuals = dyData - yLag * Pi';
    end
    
    % Compute fitted values
    fitted = dyData - residuals;
    
    % Estimate residual covariance matrix
    sigma = (residuals' * residuals) / (Teff - r * k - size(X, 2));
    
    % Extract deterministic terms
    if options.det >= 2 && ~isempty(X)
        if options.det == 2
            % Unrestricted constant only
            mu = Gamma(end, :)';
        elseif options.det == 3
            % Unrestricted constant with restricted trend
            mu = Gamma(end, :)';
        elseif options.det == 4
            % Unrestricted constant and trend
            mu = Gamma(end-1:end, :)';
        end
    end
    
    % Format Gamma matrices for output
    if p > 1
        gammaMatrices = cell(p-1, 1);
        for i = 1:(p-1)
            gammaMatrices{i} = Gamma((i-1)*k+1:i*k, :)';
        end
    else
        gammaMatrices = {};
    end
    
else
    % Case when r = 0 (no cointegration)
    % Estimate a VAR in differences
    if isempty(X)
        % No deterministic terms or lagged differences
        residuals = dyData;
        fitted = zeros(size(dyData));
    else
        % Estimate using OLS
        Gamma = X \ dyData;
        residuals = dyData - X * Gamma;
        fitted = X * Gamma;
        
        % Format Gamma matrices for output
        if p > 1
            gammaMatrices = cell(p-1, 1);
            for i = 1:(p-1)
                if i <= size(Gamma, 1) / k
                    gammaMatrices{i} = Gamma((i-1)*k+1:i*k, :)';
                else
                    gammaMatrices{i} = zeros(k, k);
                end
            end
        else
            gammaMatrices = {};
        end
        
        % Extract deterministic terms
        if options.det >= 2
            if options.det == 2
                % Unrestricted constant only
                mu = Gamma(end, :)';
            elseif options.det == 3
                % Unrestricted constant with restricted trend
                mu = Gamma(end, :)';
            elseif options.det == 4
                % Unrestricted constant and trend
                mu = Gamma(end-1:end, :)';
            end
        end
    end
    
    % Set empty cointegration parameters
    alpha = zeros(k, 0);
    beta = zeros(k, 0);
    Pi = zeros(k, k);
    
    % Estimate residual covariance matrix
    sigma = (residuals' * residuals) / (Teff - size(X, 2));
end

% Compute log-likelihood (assuming Gaussian errors)
logL = -0.5 * Teff * (k * log(2*pi) + log(max(det(sigma), eps)) + k);

% Compute number of parameters
nparams = r * (2*k - r);  % Parameters in alpha and beta
if p > 1
    nparams = nparams + (p-1) * k^2;  % Parameters in Gamma matrices
end

% Add deterministic term parameters
if options.det == 1  % Restricted constant
    nparams = nparams + r;  % Constant in cointegration
elseif options.det == 2  % Unrestricted constant
    nparams = nparams + k;  % Unrestricted constant
elseif options.det == 3  % Restricted trend, unrestricted constant
    nparams = nparams + r + k;  % Trend in cointegration, constant unrestricted
elseif options.det == 4  % Unrestricted trend
    nparams = nparams + 2*k;  % Unrestricted constant and trend
end

% Compute information criteria
ic = aicsbic(logL, nparams, Teff);

% Perform Ljung-Box test on residuals (for each variable)
ljungBoxResults = cell(k, 1);
for i = 1:k
    ljungBoxResults{i} = ljungbox(residuals(:, i), min(10, floor(Teff/5)), p-1);
end

% Create output structure
model = struct();
model.alpha = alpha;           % Adjustment coefficients
model.beta = beta;             % Cointegrating vectors
model.gamma = gammaMatrices;   % Short-run dynamics coefficients
model.Pi = Pi;                 % Long-run impact matrix (alpha * beta')
if ~isempty(mu)
    model.mu = mu;             % Deterministic terms
end
model.residuals = residuals;
model.sigma = sigma;
model.fitted = fitted;
model.logL = logL;
model.aic = ic.aic;
model.sbic = ic.sbic;
if r > 0
    model.eigenvalues = eigenvalues;
    model.eigenvectors = eigenvectors;
end
model.ljungbox = ljungBoxResults;
model.y = y;
model.dy = dy;
model.p = p;
model.r = r;
model.k = k;
model.T = T;
model.Teff = Teff;
model.nparams = nparams;
model.det = options.det;
model.options = options;

% Run cointegration test if requested
if isfield(options, 'performTest') && options.performTest
    cointegrationTest = johansen_test(y, p, options);
    model.cointegration = cointegrationTest;
end

end

function forecasts = vecm_forecast(model, h)
% VECM_FORECAST Generates forecasts from an estimated VECM model
%
% USAGE:
%   FORECASTS = vecm_forecast(MODEL, H)
%
% INPUTS:
%   MODEL     - Structure containing VECM model estimation results from vecm_model
%   H         - Positive integer indicating forecast horizon
%
% OUTPUTS:
%   FORECASTS - H by K matrix of forecasted values for T+1 to T+H
%
% COMMENTS:
%   This function generates out-of-sample forecasts for a VECM model. It first
%   converts the VECM to its equivalent VAR representation and then produces
%   forecasts recursively.
%
% EXAMPLES:
%   % Generate 10-step ahead forecasts
%   forecasts = vecm_forecast(model, 10);
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
% See also vecm_model, vecm_irf, johansen_test, vecm_to_var

% Validate forecast horizon
hOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', hOptions);

% Extract model parameters
y = model.y;
p = model.p;
k = model.k;
T = model.T;

% Convert VECM to VAR representation
var_model = vecm_to_var(model);
A = var_model.A;  % VAR coefficient matrices

% Last p observations to initialize forecast
lastObs = zeros(p, k);
for i = 1:p
    if T-p+i > 0
        lastObs(i, :) = y(T-p+i, :);
    else
        lastObs(i, :) = zeros(1, k);
    end
end

% Initialize forecast array
forecasts = zeros(h, k);

% Determine if we have deterministic terms
hasMu = isfield(model, 'mu');
hasTrend = model.det == 3 || model.det == 4;

% Get deterministic terms if present
mu = zeros(k, 1);
trend = zeros(k, 1);

if hasMu
    mu = model.mu;
    if size(mu, 2) > 1 && hasTrend
        trend = mu(:, 2);
        mu = mu(:, 1);
    end
end

% Generate forecasts recursively
for t = 1:h
    % Initialize forecast for time T+t
    forecast_t = mu;
    
    % Add trend component if included
    if hasTrend
        forecast_t = forecast_t + trend * (T + t);
    end
    
    % Add VAR dynamics
    for i = 1:p
        if t > i
            % Use previous forecasts
            forecast_t = forecast_t + A{i} * forecasts(t-i, :)';
        else
            % Use actual data
            forecast_t = forecast_t + A{i} * lastObs(p-i+t, :)';
        end
    end
    
    % Store forecast
    forecasts(t, :) = forecast_t';
end

end

function irf = vecm_irf(model, h)
% VECM_IRF Computes impulse response functions for a VECM model
%
% USAGE:
%   IRF = vecm_irf(MODEL, H)
%
% INPUTS:
%   MODEL - Structure containing VECM model estimation results from vecm_model
%   H     - Positive integer indicating the horizon for impulse responses
%
% OUTPUTS:
%   IRF   - (H+1) by K by K array where IRF(t+1,i,j) is the response of
%           variable i to a one unit shock in variable j, t periods ago
%
% COMMENTS:
%   This function computes impulse response functions (IRFs) for a VECM model.
%   The function first converts the VECM to its equivalent VAR representation 
%   and then computes the impulse responses. A Cholesky decomposition is used 
%   to orthogonalize the innovations.
%
% EXAMPLES:
%   % Compute impulse responses for 20 periods
%   irf = vecm_irf(model, 20);
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
% See also vecm_model, vecm_forecast, johansen_test, vecm_to_var

% Validate horizon
hOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', hOptions);

% Extract model parameters
p = model.p;
k = model.k;
sigma = model.sigma;

% Convert VECM to VAR
var_model = vecm_to_var(model);
A = var_model.A;  % VAR coefficient matrices

% Initialize impulse responses
% IRF is an (h+1) x k x k array where:
% IRF(t+1,i,j) = response of variable i to a shock in variable j, t periods ago
irf = zeros(h+1, k, k);

% Attempt Cholesky decomposition to orthogonalize innovations
try
    % Try regular Cholesky first
    P = chol(sigma)';  % Lower triangular matrix P such that PP' = sigma
catch
    % If Cholesky fails, use a regularized version
    % Add a small value to the diagonal to ensure positive definiteness
    regularized_sigma = sigma + eye(k) * max(1e-8, eps(max(diag(sigma))));
    try
        P = chol(regularized_sigma)';
    catch
        % If still fails, use SVD approach
        [U, S, ~] = svd(sigma);
        % Create a positive definite approximation
        S_sqrt = diag(sqrt(max(diag(S), eps)));
        P = U * S_sqrt;
    end
end

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

function results = johansen_test(y, p, options)
% JOHANSEN_TEST Performs Johansen's cointegration test
%
% USAGE:
%   RESULTS = johansen_test(Y, P)
%   RESULTS = johansen_test(Y, P, OPTIONS)
%
% INPUTS:
%   Y       - T by K matrix of time series data
%   P       - Non-negative integer representing the VAR lag order
%   OPTIONS - [OPTIONAL] Structure with test options:
%               OPTIONS.det   - Deterministic term specification:
%                              0: No deterministic terms
%                              1: Restricted constant (in cointegration)
%                              2: Unrestricted constant
%                              3: Restricted trend and unrestricted constant
%                              4: Unrestricted trend
%                              Default is 2 (unrestricted constant).
%               OPTIONS.test  - Test statistic type:
%                              'trace': Trace test (default)
%                              'maxeig': Maximum eigenvalue test
%               OPTIONS.alpha - Significance level for test
%                              Default is 0.05 (5%)
%
% OUTPUTS:
%   RESULTS - Structure containing test results:
%              RESULTS.r          - Number of cointegrating relations (0 to K-1)
%              RESULTS.trace      - Trace test statistics
%              RESULTS.maxeig     - Maximum eigenvalue test statistics
%              RESULTS.crit_trace - Critical values for trace test
%              RESULTS.crit_maxeig- Critical values for maximum eigenvalue test
%              RESULTS.pvals_trace- p-values for trace test
%              RESULTS.pvals_maxeig- p-values for maximum eigenvalue test
%              RESULTS.eigenvalues- Eigenvalues from cointegration analysis
%              RESULTS.eigenvectors- Eigenvectors from cointegration analysis
%              RESULTS.y          - Original data
%              RESULTS.p          - VAR lag order
%              RESULTS.k          - Number of variables
%              RESULTS.T          - Number of observations
%              RESULTS.det        - Deterministic term specification
%
% COMMENTS:
%   This function implements Johansen's procedure for cointegration testing.
%   The function calculates both trace and maximum eigenvalue test statistics
%   and determines the cointegration rank based on the specified test type.
%
%   Critical values are approximated based on the asymptotic distribution.
%   For more precise critical values, reference standard statistical tables.
%
% EXAMPLES:
%   % Basic Johansen test with unrestricted constant
%   results = johansen_test(data, 2);
%
%   % Johansen test with restricted constant
%   results = johansen_test(data, 2, struct('det', 1));
%
%   % Johansen test using maximum eigenvalue test at 1% significance
%   results = johansen_test(data, 2, struct('test', 'maxeig', 'alpha', 0.01));
%
% See also vecm_model, vecm_forecast, vecm_irf

% Set default options
defaultOptions = struct('det', 2, ...
                       'test', 'trace', ...
                       'alpha', 0.05);

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

% Ensure y is in matrix form
if size(y, 2) == 1
    error('Cointegration requires multivariate time series data (at least 2 variables)');
end

% Get dimensions
[T, k] = size(y);

% Validate VAR lag order (p)
pOptions = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
p = parametercheck(p, 'p', pOptions);

% Validate deterministic term specification
detOptions = struct('isscalar', true, 'isInteger', true, 'lowerBound', 0, 'upperBound', 4);
options.det = parametercheck(options.det, 'options.det', detOptions);

% Validate test type
if ~isfield(options, 'test')
    options.test = 'trace';
elseif ~ismember(options.test, {'trace', 'maxeig'})
    error('Test type must be either ''trace'' or ''maxeig''');
end

% Validate significance level
if ~isfield(options, 'alpha')
    options.alpha = 0.05;
elseif options.alpha <= 0 || options.alpha >= 1
    error('Significance level must be between 0 and 1');
end

% Check if enough observations for estimation
if T <= p*k
    error('Not enough observations for cointegration test with VAR(%d) lag structure and %d variables', p, k);
end

% Compute first differences
dy = diff(y);

% Prepare data for estimation
yLag = y(p:end-1, :);       % Lagged levels for error correction term
dyData = dy(p-1:end, :);    % Current differences (dependent variable)

% Build matrices of lagged differences
X = [];
if p > 1
    for i = 1:(p-1)
        X = [X, dy(p-i:end-i, :)];
    end
end

% Add deterministic terms based on specification
switch options.det
    case 0  % No deterministic terms
        % No additional terms
        
    case 1  % Restricted constant (in cointegration)
        % Add constant to yLag (will be part of cointegration)
        yLag = [yLag, ones(size(yLag, 1), 1)];
        
    case 2  % Unrestricted constant
        % Add constant to X
        X = [X, ones(size(dyData, 1), 1)];
        
    case 3  % Restricted trend and unrestricted constant
        % Add trend to yLag (part of cointegration) and constant to X
        trend = (1:size(yLag, 1))';
        yLag = [yLag, trend];
        X = [X, ones(size(dyData, 1), 1)];
        
    case 4  % Unrestricted trend
        % Add trend and constant to X
        trend = (1:size(dyData, 1))';
        X = [X, ones(size(dyData, 1), 1), trend];
end

% Effective sample size
Teff = size(dyData, 1);

% Adjust dimensions of lagged variables to match dependent variable
if size(yLag, 1) > size(dyData, 1)
    yLag = yLag(end-size(dyData, 1)+1:end, :);
end

% Perform reduced rank regression
if isempty(X)
    % No regressors other than error correction term
    M0 = dyData;
    M1 = yLag;
else
    % Regress dyData and yLag on X to get residuals
    M0 = dyData - X * (X \ dyData);
    M1 = yLag - X * (X \ yLag);
end

% Compute moment matrices
S00 = (M0' * M0) / Teff;
S11 = (M1' * M1) / Teff;
S01 = (M0' * M1) / Teff;
S10 = S01';

% Compute canonical correlations using SVD for numerical stability
try
    % Try SVD approach
    [~, S00_sqrt, V00] = svd(S00);
    [~, S11_sqrt, V11] = svd(S11);
    
    % Filter small eigenvalues for numerical stability
    tol00 = max(size(S00)) * eps(max(diag(S00_sqrt)));
    idx00 = diag(S00_sqrt) > tol00;
    tol11 = max(size(S11)) * eps(max(diag(S11_sqrt)));
    idx11 = diag(S11_sqrt) > tol11;
    
    S00_inv_sqrt = V00 * diag(1./sqrt(diag(S00_sqrt))) * V00';
    S11_inv_sqrt = V11 * diag(1./sqrt(diag(S11_sqrt))) * V11';
    
    C = S11_inv_sqrt * S10 * S00_inv_sqrt * S00_inv_sqrt' * S01 * S11_inv_sqrt;
catch
    % If SVD fails, use a more direct approach
    % This is numerically less stable but may work in some cases
    C = inv(S11) * S10 * inv(S00) * S01;
end

% Get eigenvalues and eigenvectors
[eigenvectors, eigenvalues_matrix] = eig(C);
eigenvalues = diag(eigenvalues_matrix);

% Sort eigenvalues in descending order
[eigenvalues, idx] = sort(real(eigenvalues), 'descend');
eigenvectors = eigenvectors(:, idx);

% Compute test statistics
trace_stats = zeros(k+1, 1);
maxeig_stats = zeros(k+1, 1);

for i = 0:k
    if i == k
        trace_stats(i+1) = 0;
        maxeig_stats(i+1) = 0;
    else
        trace_stats(i+1) = -Teff * sum(log(1 - eigenvalues(i+1:k)));
        maxeig_stats(i+1) = -Teff * log(1 - eigenvalues(i+1));
    end
end

% Critical values for Johansen's test (approximate)
% These critical values are simplified and should be used with caution
% For more precise values, reference statistical tables
% Critical values depend on:
% 1. Test type (trace or maximum eigenvalue)
% 2. Deterministic term specification (det = 0,1,2,3,4)
% 3. Number of variables (k)
% 4. Significance level (alpha)

% Define critical values for the most common cases
% These are approximations based on Johansen's tables
% For a complete implementation, a full table lookup should be used

% Critical values for k=2, alpha=0.05
crit_trace_k2_005 = [
    [20.26, 9.16, 0];      % det=0 (no constant)
    [15.50, 3.84, 0];      % det=1 (restricted constant)
    [15.50, 3.84, 0];      % det=2 (unrestricted constant)
    [25.87, 12.52, 0];     % det=3 (restricted trend)
    [18.17, 3.74, 0]       % det=4 (unrestricted trend)
];

crit_maxeig_k2_005 = [
    [15.89, 9.16, 0];      % det=0 (no constant)
    [14.26, 3.84, 0];      % det=1 (restricted constant)
    [14.26, 3.84, 0];      % det=2 (unrestricted constant)
    [19.39, 12.52, 0];     % det=3 (restricted trend)
    [16.87, 3.74, 0]       % det=4 (unrestricted trend)
];

% Critical values for k=3, alpha=0.05
crit_trace_k3_005 = [
    [35.19, 20.26, 9.16, 0];      % det=0 (no constant)
    [29.80, 15.50, 3.84, 0];      % det=1 (restricted constant)
    [29.80, 15.50, 3.84, 0];      % det=2 (unrestricted constant)
    [42.92, 25.87, 12.52, 0];     % det=3 (restricted trend)
    [31.88, 18.17, 3.74, 0]       % det=4 (unrestricted trend)
];

crit_maxeig_k3_005 = [
    [22.29, 15.89, 9.16, 0];      % det=0 (no constant)
    [21.13, 14.26, 3.84, 0];      % det=1 (restricted constant)
    [21.13, 14.26, 3.84, 0];      % det=2 (unrestricted constant)
    [25.82, 19.39, 12.52, 0];     % det=3 (restricted trend)
    [23.97, 16.87, 3.74, 0]       % det=4 (unrestricted trend)
];

% Critical values for k=4, alpha=0.05
crit_trace_k4_005 = [
    [54.07, 35.19, 20.26, 9.16, 0];    % det=0 (no constant)
    [47.86, 29.80, 15.50, 3.84, 0];    % det=1 (restricted constant)
    [47.86, 29.80, 15.50, 3.84, 0];    % det=2 (unrestricted constant)
    [63.87, 42.92, 25.87, 12.52, 0];   % det=3 (restricted trend)
    [49.65, 31.88, 18.17, 3.74, 0]     % det=4 (unrestricted trend)
];

crit_maxeig_k4_005 = [
    [28.58, 22.29, 15.89, 9.16, 0];    % det=0 (no constant)
    [27.58, 21.13, 14.26, 3.84, 0];    % det=1 (restricted constant)
    [27.58, 21.13, 14.26, 3.84, 0];    % det=2 (unrestricted constant)
    [32.15, 25.82, 19.39, 12.52, 0];   % det=3 (restricted trend)
    [30.34, 23.97, 16.87, 3.74, 0]     % det=4 (unrestricted trend)
];

% Select appropriate critical values based on k and deterministic specification
if k == 2
    crit_trace = crit_trace_k2_005(options.det+1, 1:k+1)';
    crit_maxeig = crit_maxeig_k2_005(options.det+1, 1:k+1)';
elseif k == 3
    crit_trace = crit_trace_k3_005(options.det+1, 1:k+1)';
    crit_maxeig = crit_maxeig_k3_005(options.det+1, 1:k+1)';
elseif k == 4
    crit_trace = crit_trace_k4_005(options.det+1, 1:k+1)';
    crit_maxeig = crit_maxeig_k4_005(options.det+1, 1:k+1)';
else
    % For k > 4, use approximation (this is a simplification)
    warning('Using approximated critical values for k > 4');
    % Extrapolate critical values based on k=4 case
    crit_trace = zeros(k+1, 1);
    crit_maxeig = zeros(k+1, 1);
    
    for i = 1:(k+1)
        if i <= 5
            crit_trace(i) = crit_trace_k4_005(options.det+1, i);
            crit_maxeig(i) = crit_maxeig_k4_005(options.det+1, i);
        else
            % For higher dimensions, use approximation
            crit_trace(i) = 0;
            crit_maxeig(i) = 0;
        end
    end
end

% Compute approximate p-values (simplified)
% This is a crude approximation; more accurate p-values require specific tables
pvals_trace = zeros(k+1, 1);
pvals_maxeig = zeros(k+1, 1);

for i = 1:(k+1)
    if trace_stats(i) > crit_trace(i)
        pvals_trace(i) = 0.01;  % Strongly reject
    elseif trace_stats(i) > 0.9 * crit_trace(i)
        pvals_trace(i) = 0.05;  % Reject at 5%
    elseif trace_stats(i) > 0.8 * crit_trace(i)
        pvals_trace(i) = 0.10;  % Reject at 10%
    else
        pvals_trace(i) = 0.20;  % Do not reject
    end
    
    if maxeig_stats(i) > crit_maxeig(i)
        pvals_maxeig(i) = 0.01;
    elseif maxeig_stats(i) > 0.9 * crit_maxeig(i)
        pvals_maxeig(i) = 0.05;
    elseif maxeig_stats(i) > 0.8 * crit_maxeig(i)
        pvals_maxeig(i) = 0.10;
    else
        pvals_maxeig(i) = 0.20;
    end
end

% Determine cointegration rank based on test type
r = 0;
if strcmpi(options.test, 'trace')
    % Trace test: find first r where we cannot reject H0: rank<=r
    for i = 0:k
        if pvals_trace(i+1) > options.alpha || trace_stats(i+1) < crit_trace(i+1)
            r = i;
            break;
        end
    end
else
    % Max eigenvalue test: find first r where we cannot reject H0: rank=r vs rank=r+1
    for i = 0:k
        if pvals_maxeig(i+1) > options.alpha || maxeig_stats(i+1) < crit_maxeig(i+1)
            r = i;
            break;
        end
    end
end

% Create output structure
results = struct();
results.r = r;
results.trace = trace_stats;
results.maxeig = maxeig_stats;
results.crit_trace = crit_trace;
results.crit_maxeig = crit_maxeig;
results.pvals_trace = pvals_trace;
results.pvals_maxeig = pvals_maxeig;
results.eigenvalues = eigenvalues;
results.eigenvectors = eigenvectors;
results.y = y;
results.p = p;
results.k = k;
results.T = T;
results.det = options.det;
results.alpha = options.alpha;
results.test = options.test;

end

function var_model = vecm_to_var(vecm_model)
% VECM_TO_VAR Converts a VECM representation to its equivalent VAR representation
%
% USAGE:
%   VAR_MODEL = vecm_to_var(VECM_MODEL)
%
% INPUTS:
%   VECM_MODEL - Structure containing VECM model estimation results from vecm_model
%
% OUTPUTS:
%   VAR_MODEL  - Structure containing the equivalent VAR representation with fields:
%                 VAR_MODEL.A      - Cell array of VAR coefficient matrices
%                 VAR_MODEL.k      - Number of variables
%                 VAR_MODEL.p      - VAR lag order
%                 VAR_MODEL.Pi     - Long-run impact matrix (from VECM)
%                 VAR_MODEL.alpha  - Adjustment coefficients (from VECM)
%                 VAR_MODEL.beta   - Cointegrating vectors (from VECM)
%
% COMMENTS:
%   The function converts a VECM(p-1) representation to its equivalent VAR(p) form.
%   The VECM model is specified as:
%   Δy_t = αβ'y_{t-1} + Γ_1Δy_{t-1} + ... + Γ_{p-1}Δy_{t-p+1} + μ + ε_t
%
%   The equivalent VAR(p) model is:
%   y_t = A_1 y_{t-1} + A_2 y_{t-2} + ... + A_p y_{t-p} + μ + ε_t
%
%   Where:
%   A_1 = I + αβ' + Γ_1
%   A_i = Γ_i - Γ_{i-1} for i = 2,...,p-1
%   A_p = -Γ_{p-1}
%
% EXAMPLES:
%   % Convert VECM to VAR
%   var_representation = vecm_to_var(vecm_model);
%
%   % Access VAR coefficient matrices
%   A1 = var_representation.A{1};
%   A2 = var_representation.A{2};
%
% See also vecm_model, vecm_forecast, vecm_irf

% Extract VECM parameters
k = vecm_model.k;
p = vecm_model.p;
alpha = vecm_model.alpha;
beta = vecm_model.beta;
Pi = vecm_model.Pi;
gamma = vecm_model.gamma;

% Initialize cell array for VAR coefficient matrices
A = cell(p, 1);

% Compute A_1 (first lag coefficient)
if p > 1 && ~isempty(gamma) && length(gamma) >= 1
    A{1} = eye(k) + Pi + gamma{1};
else
    A{1} = eye(k) + Pi;
end

% Compute A_i for i = 2, ..., p-1
for i = 2:(p-1)
    if ~isempty(gamma) && length(gamma) >= i
        A{i} = gamma{i} - gamma{i-1};
    elseif ~isempty(gamma) && length(gamma) >= i-1
        A{i} = -gamma{i-1};
    else
        A{i} = zeros(k);
    end
end

% Compute A_p (last lag coefficient)
if p > 1 && ~isempty(gamma) && length(gamma) >= p-1
    A{p} = -gamma{p-1};
else
    A{p} = zeros(k);
end

% Create output structure
var_model = struct();
var_model.A = A;
var_model.k = k;
var_model.p = p;
var_model.Pi = Pi;
var_model.alpha = alpha;
var_model.beta = beta;

% Include deterministic terms if present
if isfield(vecm_model, 'mu')
    var_model.mu = vecm_model.mu;
end

% Include other relevant VECM parameters
var_model.sigma = vecm_model.sigma;
var_model.det = vecm_model.det;

end