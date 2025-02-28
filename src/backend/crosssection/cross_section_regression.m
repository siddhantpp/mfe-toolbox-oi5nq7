function results = cross_section_regression(y, X, options)
% CROSS_SECTION_REGRESSION Comprehensive cross-sectional regression analysis
%
% USAGE:
%   RESULTS = cross_section_regression(Y, X)
%   RESULTS = cross_section_regression(Y, X, OPTIONS)
%
% INPUTS:
%   Y       - T by 1 vector of dependent variable
%   X       - T by K matrix of regressors (independent variables)
%   OPTIONS - [OPTIONAL] Structure with options:
%               OPTIONS.method         - Estimation method: 'ols' (default), 'wls', 'robust'
%               OPTIONS.weights        - Weights for WLS (required for 'wls' method)
%               OPTIONS.add_constant   - Add constant term (default = true)
%               OPTIONS.robust_options - Options for robust regression:
%                                        .weight_fn: 'huber' (default), 'bisquare', 'andrews'
%                                        .tuning: Tuning parameter for weight function
%                                        .max_iter: Maximum iterations (default = 50)
%                                        .tol: Convergence tolerance (default = 1e-6)
%               OPTIONS.se_type        - Standard error type: 'standard' (default), 
%                                        'robust', 'newey-west', 'bootstrap'
%               OPTIONS.nw_lags        - Lags for Newey-West SEs (default = floor(T^(1/3)))
%               OPTIONS.boot_options   - Bootstrap options (see bootstrap_confidence_intervals)
%               OPTIONS.alpha          - Significance level for tests (default = 0.05)
%               OPTIONS.filter_options - Options for filter_cross_section preprocessing
%               OPTIONS.filter_data    - Apply filter_cross_section before regression (default = false)
%
% OUTPUTS:
%   RESULTS - Structure with fields:
%     .beta         - K by 1 vector of coefficient estimates
%     .se           - K by 1 vector of standard errors
%     .tstat        - K by 1 vector of t-statistics
%     .pval         - K by 1 vector of p-values
%     .ci           - K by 2 matrix of confidence intervals [lower upper]
%     .residuals    - T by 1 vector of residuals
%     .fitted       - T by 1 vector of fitted values
%     .vcv          - K by K variance-covariance matrix
%     .r2           - R-squared
%     .r2_adj       - Adjusted R-squared
%     .f_stat       - F-statistic
%     .f_pval       - P-value for F-statistic
%     .aic          - Akaike Information Criterion
%     .bic          - Bayesian Information Criterion
%     .diagnostics  - Structure with diagnostic test results:
%                     .white - White's test for heteroskedasticity
%                     .lm    - LM test for autocorrelation
%                     .jb    - Jarque-Bera test for normality
%                     .influence - Influence measures and outlier detection
%     .options_used - Structure with options used in estimation
%     .bootstrap    - [If requested] Bootstrap results
%
% COMMENTS:
%   This function implements comprehensive cross-sectional regression analysis
%   with multiple estimation methods, diagnostics, and inference options.
%   
%   The function supports ordinary least squares (OLS), weighted least squares (WLS),
%   and robust regression methods with outlier-resistant properties.
%   
%   For inference, various standard error calculations are available including
%   heteroskedasticity-robust, Newey-West and bootstrap methods.
%   
%   The function automatically performs several diagnostic tests on the residuals
%   to check for heteroskedasticity, autocorrelation, and normality.
%
% EXAMPLES:
%   % Basic OLS regression
%   results = cross_section_regression(y, X);
%
%   % WLS regression with supplied weights
%   options.method = 'wls';
%   options.weights = 1./var_estimates;
%   results = cross_section_regression(y, X, options);
%
%   % Robust regression with Newey-West standard errors
%   options.method = 'robust';
%   options.se_type = 'newey-west';
%   options.nw_lags = 3;
%   results = cross_section_regression(y, X, options);
%
%   % OLS with bootstrap standard errors and confidence intervals
%   options.se_type = 'bootstrap';
%   options.boot_options.replications = 1000;
%   options.boot_options.bootstrap_type = 'block';
%   options.boot_options.block_size = 10;
%   results = cross_section_regression(y, X, options);
%
% See also datacheck, columncheck, parametercheck, nwse, white_test, lmtest1,
%          jarque_bera, bootstrap_confidence_intervals, filter_cross_section

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate inputs
y = datacheck(y, 'y');
y = columncheck(y, 'y');
X = datacheck(X, 'X');

% Get dimensions
[T, K] = size(X);
if length(y) ~= T
    error('Y and X must have the same number of observations.');
end

% Step 2: Set default options or validate provided options
if nargin < 3 || isempty(options)
    options = struct();
end

% Set default method
if ~isfield(options, 'method') || isempty(options.method)
    options.method = 'ols';
else
    options.method = lower(options.method);
    if ~ismember(options.method, {'ols', 'wls', 'robust'})
        error('OPTIONS.method must be one of: ''ols'', ''wls'', or ''robust''.');
    end
end

% Check for required options based on method
if strcmpi(options.method, 'wls') && (~isfield(options, 'weights') || isempty(options.weights))
    error('OPTIONS.weights must be provided when using WLS method.');
end

% Set default for add_constant
if ~isfield(options, 'add_constant') || isempty(options.add_constant)
    options.add_constant = true;
end

% Set default for se_type
if ~isfield(options, 'se_type') || isempty(options.se_type)
    options.se_type = 'standard';
else
    options.se_type = lower(options.se_type);
    if ~ismember(options.se_type, {'standard', 'robust', 'newey-west', 'bootstrap'})
        error('OPTIONS.se_type must be one of: ''standard'', ''robust'', ''newey-west'', or ''bootstrap''.');
    end
end

% Set default for nw_lags
if ~isfield(options, 'nw_lags') || isempty(options.nw_lags)
    options.nw_lags = floor(T^(1/3));
end

% Set default for alpha
if ~isfield(options, 'alpha') || isempty(options.alpha)
    options.alpha = 0.05;
else
    % Validate alpha
    alpha_options = struct('isscalar', true, 'lowerBound', 0, 'upperBound', 1);
    options.alpha = parametercheck(options.alpha, 'alpha', alpha_options);
end

% Set default for filter_data
if ~isfield(options, 'filter_data') || isempty(options.filter_data)
    options.filter_data = false;
end

% Step 3: Apply data preprocessing if requested
if options.filter_data
    if ~isfield(options, 'filter_options')
        options.filter_options = struct();
    end
    
    % Combine y and X for filtering
    data_matrix = [y X];
    filter_result = filter_cross_section(data_matrix, options.filter_options);
    
    % Extract filtered data
    filtered_data = filter_result.data;
    y = filtered_data(:, 1);
    X = filtered_data(:, 2:end);
    
    % Update dimensions
    [T, K] = size(X);
end

% Step 4: Add constant if requested
if options.add_constant
    X = [ones(T, 1) X];
    K = K + 1;
end

% Step 5: Perform regression based on selected method
switch options.method
    case 'ols'
        estimation_results = ols_estimation(y, X);
        
    case 'wls'
        % Validate weights
        weights = datacheck(options.weights, 'weights');
        weights = columncheck(weights, 'weights');
        if length(weights) ~= T
            error('Length of weights must match the number of observations.');
        end
        
        estimation_results = wls_estimation(y, X, weights);
        
    case 'robust'
        % Set default robust options
        if ~isfield(options, 'robust_options') || isempty(options.robust_options)
            options.robust_options = struct();
        end
        
        % Default weight function
        if ~isfield(options.robust_options, 'weight_fn') || isempty(options.robust_options.weight_fn)
            options.robust_options.weight_fn = 'huber';
        else
            options.robust_options.weight_fn = lower(options.robust_options.weight_fn);
            if ~ismember(options.robust_options.weight_fn, {'huber', 'bisquare', 'andrews'})
                error('Robust weight function must be one of: ''huber'', ''bisquare'', or ''andrews''.');
            end
        end
        
        % Default tuning parameter
        if ~isfield(options.robust_options, 'tuning') || isempty(options.robust_options.tuning)
            % Default tuning parameters for different weight functions
            switch options.robust_options.weight_fn
                case 'huber'
                    options.robust_options.tuning = 1.345;
                case 'bisquare'
                    options.robust_options.tuning = 4.685;
                case 'andrews'
                    options.robust_options.tuning = 1.339;
            end
        end
        
        % Default max iterations
        if ~isfield(options.robust_options, 'max_iter') || isempty(options.robust_options.max_iter)
            options.robust_options.max_iter = 50;
        end
        
        % Default tolerance
        if ~isfield(options.robust_options, 'tol') || isempty(options.robust_options.tol)
            options.robust_options.tol = 1e-6;
        end
        
        estimation_results = robust_estimation(y, X, options.robust_options);
end

% Step 6: Extract common results
beta = estimation_results.beta;
residuals = estimation_results.residuals;
fitted = estimation_results.fitted;

% Step 7: Compute standard errors based on selected method
se_results = compute_standard_errors(X, residuals, options);
standard_errors = se_results.se;
vcv = se_results.vcv;

% Step 8: Compute t-statistics and p-values
t_stats = beta ./ standard_errors;
dof = T - K;  % Degrees of freedom
p_values = 2 * (1 - tcdf(abs(t_stats), dof));

% Step 9: Compute confidence intervals
ci_alpha = options.alpha / 2;
t_critical = tinv(1 - ci_alpha, dof);
ci_lower = beta - t_critical * standard_errors;
ci_upper = beta + t_critical * standard_errors;
confidence_intervals = [ci_lower ci_upper];

% Step 10: Compute bootstrap confidence intervals if requested
if strcmpi(options.se_type, 'bootstrap')
    if ~isfield(options, 'boot_options') || isempty(options.boot_options)
        options.boot_options = struct();
    end
    
    % Set confidence level
    if ~isfield(options.boot_options, 'conf_level') || isempty(options.boot_options.conf_level)
        options.boot_options.conf_level = 1 - options.alpha;
    end
    
    bootstrap_results = bootstrap_coefficients(y, X, beta, options.boot_options);
    confidence_intervals = [bootstrap_results.ci_lower bootstrap_results.ci_upper];
    
    % Update standard errors with bootstrap standard errors
    standard_errors = bootstrap_results.se;
    
    % Update t-statistics and p-values with bootstrap standard errors
    t_stats = beta ./ standard_errors;
    p_values = 2 * (1 - tcdf(abs(t_stats), dof));
end

% Step 11: Compute goodness-of-fit statistics
goodness_results = compute_goodness_of_fit(y, X, residuals, beta);

% Step 12: Compute diagnostic statistics
diagnostic_results = compute_diagnostics(y, X, residuals, beta);

% Step 13: Compile results
results = struct();

% Basic regression results
results.beta = beta;
results.se = standard_errors;
results.tstat = t_stats;
results.pval = p_values;
results.ci = confidence_intervals;
results.vcv = vcv;
results.residuals = residuals;
results.fitted = fitted;

% Goodness-of-fit measures
results.r2 = goodness_results.r2;
results.r2_adj = goodness_results.r2_adj;
results.f_stat = goodness_results.f_stat;
results.f_pval = goodness_results.f_pval;
results.aic = goodness_results.aic;
results.bic = goodness_results.bic;

% Diagnostic test results
results.diagnostics = diagnostic_results;

% Options used
results.options_used = options;

% Additional information
results.nobs = T;
results.df_model = K - options.add_constant;
results.df_residual = T - K;

% Include bootstrap results if applicable
if strcmpi(options.se_type, 'bootstrap')
    results.bootstrap = bootstrap_results;
end

end

function results = ols_estimation(y, X)
% OLS_ESTIMATION Performs ordinary least squares estimation for linear regression models
%
% USAGE:
%   RESULTS = ols_estimation(Y, X)
%
% INPUTS:
%   Y - T by 1 vector of dependent variable
%   X - T by K matrix of regressors
%
% OUTPUTS:
%   RESULTS - Structure with OLS estimation results containing coefficients,
%             variance-covariance matrix, and fitted values

% Get dimensions
[T, K] = size(X);

% Check for perfect multicollinearity
XX = X' * X;
rcond_XX = rcond(XX);
if rcond_XX < eps*1000
    warning('X''X matrix is nearly singular with condition number %g.', 1/rcond_XX);
end

% Compute beta: β = (X'X)^(-1)X'y
Xy = X' * y;
beta = inv(XX) * Xy;

% Compute fitted values and residuals
fitted = X * beta;
residuals = y - fitted;

% Compute error variance estimate: σ² = e'e/(n-k)
sigma2 = (residuals' * residuals) / (T - K);

% Compute variance-covariance matrix: V = σ² * (X'X)^(-1)
vcv = sigma2 * inv(XX);

% Return results
results = struct(...
    'beta', beta, ...
    'fitted', fitted, ...
    'residuals', residuals, ...
    'sigma2', sigma2, ...
    'vcv', vcv, ...
    'XX_inv', inv(XX));
end

function results = wls_estimation(y, X, weights)
% WLS_ESTIMATION Performs weighted least squares estimation for heteroskedastic regression models
%
% USAGE:
%   RESULTS = wls_estimation(Y, X, WEIGHTS)
%
% INPUTS:
%   Y       - T by 1 vector of dependent variable
%   X       - T by K matrix of regressors
%   WEIGHTS - T by 1 vector of weights
%
% OUTPUTS:
%   RESULTS - Structure with WLS estimation results containing coefficients,
%             variance-covariance matrix, and fitted values

% Validate weights
weights = parametercheck(weights, 'weights');
if length(weights) ~= size(X, 1)
    error('Length of weights must match the number of observations.');
end

% Get dimensions
[T, K] = size(X);

% Create diagonal weight matrix W
W = diag(weights);

% Compute beta: β = (X'WX)^(-1)X'Wy
XW = X' * W;
XWX = XW * X;
XWy = XW * y;

% Check for perfect multicollinearity in weighted data
rcond_XWX = rcond(XWX);
if rcond_XWX < eps*1000
    warning('X''WX matrix is nearly singular with condition number %g.', 1/rcond_XWX);
end

beta = inv(XWX) * XWy;

% Compute fitted values and residuals
fitted = X * beta;
residuals = y - fitted;

% Compute error variance estimate
weighted_sse = residuals' * W * residuals;
sigma2 = weighted_sse / (T - K);

% Compute variance-covariance matrix: V = (X'WX)^(-1)
vcv = inv(XWX);

% Return results
results = struct(...
    'beta', beta, ...
    'fitted', fitted, ...
    'residuals', residuals, ...
    'sigma2', sigma2, ...
    'vcv', vcv, ...
    'weights', weights);
end

function results = robust_estimation(y, X, options)
% ROBUST_ESTIMATION Performs robust regression estimation that is resistant
% to outliers and influential observations
%
% USAGE:
%   RESULTS = robust_estimation(Y, X, OPTIONS)
%
% INPUTS:
%   Y       - T by 1 vector of dependent variable
%   X       - T by K matrix of regressors
%   OPTIONS - Structure with options for robust regression:
%             .weight_fn: Weight function 'huber', 'bisquare', 'andrews'
%             .tuning: Tuning parameter for weight function
%             .max_iter: Maximum iterations
%             .tol: Convergence tolerance
%
% OUTPUTS:
%   RESULTS - Structure with robust estimation results containing coefficients,
%             variance-covariance matrix, and fitted values

% Get dimensions
[T, K] = size(X);

% Step 1: Start with initial OLS estimates
ols_results = ols_estimation(y, X);
beta = ols_results.beta;
residuals = ols_results.residuals;

% Compute initial scale using median absolute deviation (MAD)
sigma = median(abs(residuals - median(residuals))) / 0.6745;

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Default weight function
if ~isfield(options, 'weight_fn') || isempty(options.weight_fn)
    options.weight_fn = 'huber';
end

% Default tuning parameter
if ~isfield(options, 'tuning') || isempty(options.tuning)
    % Default tuning parameters for different weight functions
    switch options.weight_fn
        case 'huber'
            options.tuning = 1.345;
        case 'bisquare'
            options.tuning = 4.685;
        case 'andrews'
            options.tuning = 1.339;
    end
end

% Default max iterations
if ~isfield(options, 'max_iter') || isempty(options.max_iter)
    options.max_iter = 50;
end

% Default tolerance
if ~isfield(options, 'tol') || isempty(options.tol)
    options.tol = 1e-6;
end

% Create function for computing weights based on selected weight function
switch options.weight_fn
    case 'huber'
        % Huber weights: w(e) = min(1, k/|e|)
        weight_fn = @(e, s, k) min(1, k ./ abs(e/s));
        
    case 'bisquare'
        % Tukey's bisquare weights: w(e) = (1-(e/k)^2)^2 if |e|<k, 0 otherwise
        weight_fn = @(e, s, k) (abs(e/s) < k) .* (1 - (e/(k*s)).^2).^2;
        
    case 'andrews'
        % Andrews' sine weights: w(e) = sin(e/k)/(e/k) if |e|<k*pi, 0 otherwise
        weight_fn = @(e, s, k) ((abs(e/s) < k*pi) .* sin(e/(k*s)) ./ (e/(k*s)));
        % Handle the limiting case as e approaches 0
        weight_fn = @(e, s, k) deal_with_zeros(e, s, k);
        
    otherwise
        error('Unsupported weight function: %s', options.weight_fn);
end

% Helper function to handle zeros in Andrews weights
function w = deal_with_zeros(e, s, k)
    e_scaled = e/(k*s);
    w = zeros(size(e));
    zero_idx = abs(e_scaled) < 1e-10;
    nonzero_idx = ~zero_idx & (abs(e_scaled) < pi);
    w(zero_idx) = 1;
    w(nonzero_idx) = sin(e_scaled(nonzero_idx)) ./ e_scaled(nonzero_idx);
end

% Initialize convergence tracking
beta_old = beta;
iter = 0;
converged = false;

% Iteratively reweighted least squares (IRLS) algorithm
while ~converged && iter < options.max_iter
    iter = iter + 1;
    
    % Compute residuals with current beta
    residuals = y - X * beta;
    
    % Update scale estimate (robust standard deviation)
    sigma = median(abs(residuals - median(residuals))) / 0.6745;
    
    % Compute weights using the selected weight function
    weights = weight_fn(residuals, sigma, options.tuning);
    
    % Adjust weights to handle potential zero weights
    weights = max(weights, 1e-6);
    
    % Weighted least squares step
    wls_results = wls_estimation(y, X, weights);
    beta = wls_results.beta;
    
    % Check for convergence
    beta_diff = norm(beta - beta_old) / (norm(beta_old) + eps);
    converged = beta_diff < options.tol;
    beta_old = beta;
end

% Compute final results
fitted = X * beta;
residuals = y - fitted;

% Compute final weights for variance estimation
weights = weight_fn(residuals, sigma, options.tuning);
weights = max(weights, 1e-6);

% Create diagonal weight matrix W
W = diag(weights);

% Compute weighted error variance and covariance matrix
XW = X' * W;
XWX = XW * X;
weighted_sse = residuals' * W * residuals;
sigma2 = weighted_sse / (T - K);
vcv = sigma2 * inv(XWX);

% Return results
results = struct(...
    'beta', beta, ...
    'fitted', fitted, ...
    'residuals', residuals, ...
    'sigma2', sigma2, ...
    'vcv', vcv, ...
    'weights', weights, ...
    'iterations', iter, ...
    'converged', converged, ...
    'scale', sigma);
end

function results = compute_standard_errors(X, residuals, options)
% COMPUTE_STANDARD_ERRORS Calculates various types of standard errors for
% regression coefficients
%
% USAGE:
%   RESULTS = compute_standard_errors(X, RESIDUALS, OPTIONS)
%
% INPUTS:
%   X          - T by K matrix of regressors
%   RESIDUALS  - T by 1 vector of regression residuals
%   OPTIONS    - Structure with standard error options:
%                .se_type: Type of standard errors
%                .nw_lags: Lags for Newey-West
%                .boot_options: Options for bootstrap (if applicable)
%
% OUTPUTS:
%   RESULTS    - Structure with standard errors and variance-covariance matrix

% Get dimensions
[T, K] = size(X);

% Extract options
se_type = options.se_type;

% Standard OLS variance-covariance matrix
XX_inv = inv(X' * X);
e2 = residuals.^2;
sigma2 = sum(e2) / (T - K);
V = sigma2 * XX_inv;

% Calculate selected standard error type
switch se_type
    case 'standard'
        % Standard homoskedastic standard errors
        vcv = V;
        
    case 'robust'
        % White heteroskedasticity-robust standard errors
        % HC0 version: V = (X'X)^(-1) * X' * diag(e^2) * X * (X'X)^(-1)
        X_e2_X = X' * diag(e2) * X;
        vcv = XX_inv * X_e2_X * XX_inv;
        
    case 'newey-west'
        % Newey-West HAC standard errors
        nw_lags = options.nw_lags;
        
        % Call nwse function to compute Newey-West standard errors
        se_nw = nwse(X, residuals, nw_lags);
        
        % Reconstruct variance-covariance matrix
        vcv = diag(se_nw.^2);
        
    case 'bootstrap'
        % Bootstrap standard errors
        % For bootstrap SE, we use the standard VCV temporarily
        % The actual bootstrap SEs will be computed later in bootstrap_coefficients
        vcv = V;
        
    otherwise
        error('Unsupported standard error type: %s', se_type);
end

% Extract standard errors from diagonal of vcv
se = sqrt(diag(vcv));

% Return results
results = struct(...
    'se', se, ...
    'vcv', vcv, ...
    'type', se_type);
end

function results = compute_diagnostics(y, X, residuals, beta)
% COMPUTE_DIAGNOSTICS Calculates regression diagnostic statistics for model validation
%
% USAGE:
%   RESULTS = compute_diagnostics(Y, X, RESIDUALS, BETA)
%
% INPUTS:
%   Y         - T by 1 vector of dependent variable
%   X         - T by K matrix of regressors
%   RESIDUALS - T by 1 vector of regression residuals
%   BETA      - K by 1 vector of coefficient estimates
%
% OUTPUTS:
%   RESULTS   - Structure with diagnostic statistics including heteroskedasticity,
%               autocorrelation, and normality tests

% Get dimensions
[T, K] = size(X);

% White's test for heteroskedasticity
white_results = white_test(residuals, X);

% LM test for autocorrelation (use up to min(10, T/5) lags)
lags = min(10, floor(T/5));
lm_results = lmtest1(residuals, lags);

% Jarque-Bera test for normality
jb_results = jarque_bera(residuals);

% Calculate leverage (diagonal elements of hat matrix)
XX_inv = inv(X' * X);
H = X * XX_inv * X';
leverage = diag(H);

% Calculate standardized residuals
sigma2 = sum(residuals.^2) / (T - K);
std_residuals = residuals ./ sqrt(sigma2 * (1 - leverage));

% Calculate Cook's distance
cook_d = (std_residuals.^2 .* leverage) ./ (K * (1 - leverage));

% Calculate DFFITS
dffits = std_residuals .* sqrt(leverage ./ (1 - leverage));

% Identify potential outliers and influential observations
outlier_threshold = 2;  % Standardized residuals > 2 in absolute value
influence_threshold = 4/T;  % Cook's distance > 4/T
leverage_threshold = 2*K/T;  % Leverage > 2k/T

potential_outliers = abs(std_residuals) > outlier_threshold;
influential_obs = cook_d > influence_threshold;
high_leverage = leverage > leverage_threshold;

% Return results
influence_measures = struct(...
    'leverage', leverage, ...
    'std_residuals', std_residuals, ...
    'cook_d', cook_d, ...
    'dffits', dffits, ...
    'potential_outliers', potential_outliers, ...
    'influential_obs', influential_obs, ...
    'high_leverage', high_leverage);

results = struct(...
    'white_test', white_results, ...
    'lm_test', lm_results, ...
    'jb_test', jb_results, ...
    'influence', influence_measures);
end

function results = compute_goodness_of_fit(y, X, residuals, beta)
% COMPUTE_GOODNESS_OF_FIT Calculates goodness-of-fit measures for 
% regression model evaluation
%
% USAGE:
%   RESULTS = compute_goodness_of_fit(Y, X, RESIDUALS, BETA)
%
% INPUTS:
%   Y         - T by 1 vector of dependent variable
%   X         - T by K matrix of regressors
%   RESIDUALS - T by 1 vector of regression residuals
%   BETA      - K by 1 vector of coefficient estimates
%
% OUTPUTS:
%   RESULTS   - Structure with goodness-of-fit statistics including R²,
%               adjusted R², F-statistic, AIC, and BIC

% Get dimensions
[T, K] = size(X);

% Calculate sum of squared errors (SSE)
SSE = residuals' * residuals;

% Calculate total sum of squares (SST)
y_mean = mean(y);
SST = sum((y - y_mean).^2);

% Calculate sum of squares due to regression (SSR)
SSR = SST - SSE;

% Calculate coefficient of determination (R^2)
r2 = 1 - SSE/SST;

% Calculate adjusted R^2
r2_adj = 1 - (SSE/(T-K)) / (SST/(T-1));

% Calculate F-statistic
if K > 1  % Only if there are multiple regressors
    F = (SSR/(K-1)) / (SSE/(T-K));
else
    F = (SSR/K) / (SSE/(T-K));
end

% Calculate p-value for F-statistic
if K > 1
    f_pval = 1 - fcdf(F, K-1, T-K);
else
    f_pval = 1 - fcdf(F, K, T-K);
end

% Calculate log-likelihood (assuming normal errors)
sigma2 = SSE / T;
log_lik = -T/2 * (1 + log(2*pi) + log(sigma2));

% Calculate information criteria
aic = -2 * log_lik + 2 * K;  % Akaike Information Criterion
bic = -2 * log_lik + K * log(T);  % Bayesian Information Criterion

% Return results
results = struct(...
    'r2', r2, ...
    'r2_adj', r2_adj, ...
    'f_stat', F, ...
    'f_pval', f_pval, ...
    'log_lik', log_lik, ...
    'aic', aic, ...
    'bic', bic, ...
    'sse', SSE, ...
    'sst', SST, ...
    'ssr', SSR, ...
    'sigma2', sigma2);
end

function results = bootstrap_coefficients(y, X, beta, options)
% BOOTSTRAP_COEFFICIENTS Generates bootstrap distribution and confidence intervals
% for regression coefficients
%
% USAGE:
%   RESULTS = bootstrap_coefficients(Y, X, BETA, OPTIONS)
%
% INPUTS:
%   Y       - T by 1 vector of dependent variable
%   X       - T by K matrix of regressors
%   BETA    - K by 1 vector of coefficient estimates
%   OPTIONS - Structure with bootstrap options (see bootstrap_confidence_intervals)
%
% OUTPUTS:
%   RESULTS - Structure with bootstrap results including confidence intervals,
%             standard errors, and full bootstrap distributions

% Set default options if not provided
if nargin < 4 || isempty(options)
    options = struct();
end

% Default bootstrap type
if ~isfield(options, 'bootstrap_type') || isempty(options.bootstrap_type)
    options.bootstrap_type = 'block';
end

% Combine y and X for bootstrap sampling
data = [y X];

% Define statistic function for bootstrap
stat_fn = @(data) bs_regression(data);

% Call bootstrap_confidence_intervals function
bs_results = bootstrap_confidence_intervals(data, stat_fn, options);

% Extract relevant results
boot_stats = bs_results.bootstrap_statistics;
ci_lower = bs_results.lower;
ci_upper = bs_results.upper;

% Calculate bootstrap standard errors (standard deviation of bootstrap distribution)
bootstrap_se = std(boot_stats)';  % Ensure column vector

% Return results
results = struct(...
    'ci_lower', ci_lower, ...
    'ci_upper', ci_upper, ...
    'se', bootstrap_se, ...
    'bootstrap_distribution', boot_stats, ...
    'options_used', options);

    % Nested helper function for bootstrap regression
    function beta = bs_regression(data)
        % Extract y and X from combined data
        y_bs = data(:, 1);
        X_bs = data(:, 2:end);
        
        % Perform regression and return coefficients
        beta = (X_bs' * X_bs) \ (X_bs' * y_bs);
    end
end