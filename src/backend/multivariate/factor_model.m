function [model] = factor_model(data, k, options)
% FACTOR_MODEL Estimates a factor model for multivariate time series data.
%
% USAGE:
%   MODEL = factor_model(DATA, K)
%   MODEL = factor_model(DATA, K, OPTIONS)
%
% INPUTS:
%   DATA    - T by N matrix of time series data where T is the number of observations
%             and N is the number of variables
%   K       - Positive integer indicating the number of factors to extract
%   OPTIONS - [OPTIONAL] Structure with estimation options:
%               OPTIONS.method       - [OPTIONAL] Method for factor extraction:
%                                     'principal' (default) or 'ml' (maximum likelihood)
%               OPTIONS.standardize  - [OPTIONAL] Logical indicating whether to 
%                                     standardize the data. Default is true.
%               OPTIONS.robust       - [OPTIONAL] Logical indicating whether to use
%                                     robust estimation. Default is false.
%               OPTIONS.rotate       - [OPTIONAL] Factor rotation method:
%                                     'none' (default), 'varimax', 'quartimax',
%                                     'promax', or 'oblimin'
%               OPTIONS.scores       - [OPTIONAL] Method for computing factor scores:
%                                     'regression' (default), 'bartlett', or 'anderson'
%               OPTIONS.bootstrap    - [OPTIONAL] Logical indicating whether to perform 
%                                     bootstrap inference. Default is false.
%               OPTIONS.boot_options - [OPTIONAL] Structure with bootstrap options
%                                     (see bootstrap_confidence_intervals)
%
% OUTPUTS:
%   MODEL   - Structure containing factor model estimation results:
%               MODEL.loadings        - N by K matrix of factor loadings
%               MODEL.factors         - T by K matrix of factor scores
%               MODEL.communalities   - N by 1 vector of communalities
%               MODEL.uniquenesses    - N by 1 vector of uniquenesses (specific variances)
%               MODEL.eigenvalues     - K by 1 vector of eigenvalues (for principal components)
%               MODEL.var_explained   - K by 1 vector with proportion of variance explained
%               MODEL.var_cumulative  - K by 1 vector with cumulative proportion
%               MODEL.method          - Method used for extraction
%               MODEL.rotation        - Rotation method applied
%               MODEL.rotation_matrix - Rotation matrix (if rotation was applied)
%               MODEL.factor_corr     - Factor correlation matrix (for oblique rotations)
%               MODEL.data_cov        - Covariance/correlation matrix of the data
%               MODEL.residual_cov    - Residual covariance/correlation matrix
%               MODEL.goodness_of_fit - Goodness-of-fit statistics
%               MODEL.bootstrap       - Bootstrap results (if bootstrap=true)
%
% COMMENTS:
%   Factor analysis is a dimension reduction technique that represents observed
%   variables as linear combinations of a smaller number of unobserved factors.
%   The model can be expressed as: X = Λ*F + ε, where:
%   - X is the N-dimensional vector of observed variables
%   - Λ is the N×K matrix of factor loadings
%   - F is the K-dimensional vector of common factors
%   - ε is the N-dimensional vector of specific factors
%
%   This function implements both principal component and maximum likelihood
%   factor extraction methods, with options for factor rotation and different
%   approaches to computing factor scores.
%
%   For financial applications, factor models are useful for:
%   - Identifying common risk factors in asset returns
%   - Reducing dimensionality in large datasets
%   - Analyzing the covariance structure of financial variables
%   - Building parsimonious models for forecasting
%
% EXAMPLES:
%   % Basic usage with 3 factors using principal components
%   model = factor_model(returns, 3);
%
%   % Maximum likelihood estimation with varimax rotation
%   options = struct('method', 'ml', 'rotate', 'varimax');
%   model = factor_model(returns, 3, options);
%
%   % Factor model with bootstrap inference
%   options = struct('bootstrap', true, 'boot_options', struct('replications', 1000));
%   model = factor_model(returns, 3, options);
%
% See also dynamic_factor_model, var_model, bootstrap_confidence_intervals

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Defining default options
defaultOptions = struct('method', 'principal', ...
                       'standardize', true, ...
                       'robust', false, ...
                       'rotate', 'none', ...
                       'scores', 'regression', ...
                       'bootstrap', false, ...
                       'boot_options', []);

% Step 1: Validate inputs
data = datacheck(data, 'data');

% Validate number of factors
k_options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
k = parametercheck(k, 'k', k_options);

% Check dimensions
[T, N] = size(data);
if k >= N
    error('Number of factors (k) must be less than the number of variables (N).');
end

% Merge options with defaults
if nargin < 3 || isempty(options)
    options = defaultOptions;
else
    fields = fieldnames(defaultOptions);
    for i = 1:length(fields)
        if ~isfield(options, fields{i})
            options.(fields{i}) = defaultOptions.(fields{i});
        end
    end
end

% Step 2: Preprocess data
% Center the data (subtract mean)
data_mean = mean(data);
data_centered = data - ones(T, 1) * data_mean;

% Standardize if requested
if options.standardize
    data_std = std(data_centered);
    X = data_centered ./ (ones(T, 1) * data_std);
else
    X = data_centered;
end

% Step 3: Compute covariance/correlation matrix
if options.robust
    % Use robust estimator of covariance/correlation
    R = corr(X, 'type', 'Spearman');
else
    % Use standard covariance/correlation
    if options.standardize
        R = corr(X);
    else
        R = cov(X);
    end
end

% Check if covariance/correlation matrix is positive definite
if ~matrixdiagnostics.isPositiveDefinite(R)
    warning(['Covariance/correlation matrix is not positive definite. ', ...
             'Adding small diagonal adjustment for numerical stability.']);
    % Add small adjustment to diagonal for numerical stability
    R = R + eye(N) * eps(max(diag(R))) * 1000;
end

% Step 4: Extract factors using specified method
switch lower(options.method)
    case 'principal'
        % Principal component factor extraction
        factors_results = extract_factors_principal(R, k);
    case 'ml'
        % Maximum likelihood factor extraction
        ml_options = struct('max_iterations', 1000, 'tolerance', 1e-6);
        factors_results = extract_factors_ml(R, k, ml_options);
    otherwise
        error('Unsupported factor extraction method: %s', options.method);
end

% Extract results
loadings = factors_results.loadings;
communalities = factors_results.communalities;
uniquenesses = factors_results.uniquenesses;

% Step 5: Apply rotation if requested
if ~strcmpi(options.rotate, 'none')
    rotation_results = rotate_factors(loadings, options.rotate, []);
    loadings = rotation_results.loadings;
    rotation_matrix = rotation_results.rotation_matrix;
    
    % Update communalities based on rotated loadings
    communalities = sum(loadings.^2, 2);
    uniquenesses = 1 - communalities;
    
    % Store factor correlations for oblique rotations
    if isfield(rotation_results, 'factor_corr')
        factor_corr = rotation_results.factor_corr;
    else
        factor_corr = eye(k);
    end
else
    rotation_matrix = eye(k);
    factor_corr = eye(k);
end

% Step 6: Compute factor scores
scores = compute_factor_scores(X, loadings, uniquenesses, options.scores);

% Step 7: Compute model diagnostics
% Reproduce the correlation/covariance matrix based on the factor model
reproduced_R = loadings * loadings' + diag(uniquenesses);
residual_R = R - reproduced_R;

% Calculate goodness-of-fit measures
% Root Mean Square Residual (RMSR)
off_diagonal = ~eye(N);
rmsr = sqrt(mean(residual_R(off_diagonal).^2));

% Calculate chi-square test statistic for maximum likelihood
if strcmpi(options.method, 'ml')
    chi_sq = (T - 1 - (2*N + 5)/6 - (2*k)/3) * (log(det(R)) - log(det(reproduced_R)) + trace(reproduced_R * inv(R)) - N);
    chi_df = 0.5 * ((N - k)^2 - N - k);
    chi_pval = 1 - chi2cdf(chi_sq, chi_df);
else
    % For principal components, calculate a simpler fit measure
    chi_sq = T * sum(residual_R(off_diagonal).^2);
    chi_df = (N * (N - 1) / 2) - (N * k - k * (k - 1) / 2);
    chi_pval = 1 - chi2cdf(chi_sq, chi_df);
end

% Calculate AIC and BIC
aic = chi_sq - 2 * chi_df;
bic = chi_sq - chi_df * log(T);

% Compute goodness-of-fit metrics
goodness_of_fit = struct(...
    'rmsr', rmsr, ...
    'chi_square', chi_sq, ...
    'df', chi_df, ...
    'p_value', chi_pval, ...
    'aic', aic, ...
    'bic', bic, ...
    'residual_correlations', residual_R);

% Step 8: Perform bootstrap analysis if requested
if options.bootstrap
    bootstrap_results = factor_model_bootstrap(data, struct(...
        'loadings', loadings, ...
        'uniquenesses', uniquenesses, ...
        'k', k, ...
        'method', options.method, ...
        'standardize', options.standardize, ...
        'rotate', options.rotate), ...
        options.boot_options);
else
    bootstrap_results = [];
end

% Step 9: Construct the output model structure
model = struct(...
    'loadings', loadings, ...
    'factors', scores, ...
    'communalities', communalities, ...
    'uniquenesses', uniquenesses, ...
    'eigenvalues', factors_results.eigenvalues, ...
    'var_explained', factors_results.var_explained, ...
    'var_cumulative', factors_results.var_cumulative, ...
    'method', options.method, ...
    'rotation', options.rotate, ...
    'rotation_matrix', rotation_matrix, ...
    'factor_corr', factor_corr, ...
    'data_cov', R, ...
    'residual_cov', residual_R, ...
    'goodness_of_fit', goodness_of_fit, ...
    'T', T, ...
    'N', N, ...
    'k', k, ...
    'standardized', options.standardize, ...
    'robust', options.robust, ...
    'data_mean', data_mean);

% Add bootstrap results if available
if ~isempty(bootstrap_results)
    model.bootstrap = bootstrap_results;
end

end

function results = extract_factors_principal(R, k)
% EXTRACT_FACTORS_PRINCIPAL Extracts factors using principal component method
%
% USAGE:
%   RESULTS = extract_factors_principal(R, K)
%
% INPUTS:
%   R - N by N correlation or covariance matrix
%   K - Number of factors to extract
%
% OUTPUTS:
%   RESULTS - Structure with fields:
%               RESULTS.loadings       - N by K matrix of factor loadings
%               RESULTS.eigenvalues    - K by 1 vector of eigenvalues
%               RESULTS.communalities  - N by 1 vector of communalities
%               RESULTS.uniquenesses   - N by 1 vector of uniquenesses
%               RESULTS.var_explained  - K by 1 proportion of variance explained
%               RESULTS.var_cumulative - K by 1 cumulative proportion

% Get the dimension of the covariance/correlation matrix
N = size(R, 1);

% Compute eigenvalues and eigenvectors
[V, D] = eig(R);
eigenvalues = diag(D);

% Sort eigenvalues and eigenvectors in descending order
[eigenvalues, idx] = sort(eigenvalues, 'descend');
V = V(:, idx);

% Select the first K eigenvalues and eigenvectors
eigenvalues = eigenvalues(1:k);
V = V(:, 1:k);

% Calculate factor loadings as eigenvectors scaled by sqrt(eigenvalues)
loadings = V .* sqrt(eigenvalues)';

% Calculate communalities (proportion of variance explained by factors)
communalities = sum(loadings.^2, 2);

% Calculate uniquenesses (specific variances)
uniquenesses = 1 - communalities;

% Calculate proportion of variance explained by each factor
total_var = trace(R);
var_explained = eigenvalues / total_var;

% Calculate cumulative proportion of variance explained
var_cumulative = cumsum(var_explained);

% Return results
results = struct(...
    'loadings', loadings, ...
    'eigenvalues', eigenvalues, ...
    'communalities', communalities, ...
    'uniquenesses', uniquenesses, ...
    'var_explained', var_explained, ...
    'var_cumulative', var_cumulative);

end

function results = extract_factors_ml(R, k, options)
% EXTRACT_FACTORS_ML Extracts factors using maximum likelihood estimation
%
% USAGE:
%   RESULTS = extract_factors_ml(R, K, OPTIONS)
%
% INPUTS:
%   R       - N by N correlation or covariance matrix
%   K       - Number of factors to extract
%   OPTIONS - [OPTIONAL] Structure with options:
%              OPTIONS.max_iterations - Maximum number of iterations (default: 1000)
%              OPTIONS.tolerance      - Convergence tolerance (default: 1e-6)
%
% OUTPUTS:
%   RESULTS - Structure with fields:
%              RESULTS.loadings        - N by K matrix of factor loadings
%              RESULTS.uniquenesses    - N by 1 vector of uniquenesses
%              RESULTS.communalities   - N by 1 vector of communalities
%              RESULTS.iterations      - Number of iterations to convergence
%              RESULTS.converged       - Logical indicating convergence
%              RESULTS.log_likelihood  - Log-likelihood of the model
%              RESULTS.eigenvalues     - Eigenvalues of factors
%              RESULTS.var_explained   - Proportion of variance explained by each factor
%              RESULTS.var_cumulative  - Cumulative proportion of variance explained

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

if ~isfield(options, 'max_iterations')
    options.max_iterations = 1000;
end

if ~isfield(options, 'tolerance')
    options.tolerance = 1e-6;
end

% Get dimensions
N = size(R, 1);

% Initialize uniquenesses with small positive values
% A common initialization is to set uniquenesses to diagonal elements of
% the inverse of the correlation/covariance matrix
psi = ones(N, 1) * 0.1;

% Initialize log-likelihood to negative infinity for comparison
old_loglik = -Inf;
converged = false;

% Start iterative procedure
for iter = 1:options.max_iterations
    % Step 1: Compute the adjusted correlation matrix R* = R - diag(psi)
    R_star = R - diag(psi);
    
    % Step 2: Compute eigenvalues and eigenvectors of R*
    [V, D] = eig(R_star);
    eigenvalues = diag(D);
    
    % Sort in descending order
    [eigenvalues, idx] = sort(eigenvalues, 'descend');
    V = V(:, idx);
    
    % Step 3: Select largest k eigenvalues and corresponding eigenvectors
    eigenvalues_k = eigenvalues(1:k);
    V_k = V(:, 1:k);
    
    % Check for negative eigenvalues (can happen in early iterations)
    if any(eigenvalues_k <= 0)
        warning('Negative eigenvalues detected. Adjusting for numerical stability.');
        eigenvalues_k = max(eigenvalues_k, eps);
    end
    
    % Step 4: Compute factor loadings
    loadings = V_k .* sqrt(eigenvalues_k)';
    
    % Step 5: Compute implied correlation matrix
    implied_R = loadings * loadings' + diag(psi);
    
    % Step 6: Update uniquenesses
    communalities = sum(loadings.^2, 2);
    new_psi = diag(R) - communalities;
    
    % Ensure uniquenesses are positive
    new_psi = max(new_psi, eps);
    
    % Step 7: Compute log-likelihood
    log_det_R = log(det(implied_R));
    log_det_psi = sum(log(new_psi));
    
    % The log-likelihood (up to a constant)
    new_loglik = -0.5 * (log_det_R + trace(R * inv(implied_R)) - log_det_psi - N);
    
    % Step 8: Check for convergence
    if abs(new_loglik - old_loglik) < options.tolerance
        converged = true;
        break;
    end
    
    % Update for next iteration
    psi = new_psi;
    old_loglik = new_loglik;
end

% Calculate final communalities and uniquenesses
communalities = sum(loadings.^2, 2);
uniquenesses = diag(R) - communalities;

% Calculate proportion of variance explained
total_var = trace(R);
var_explained = eigenvalues_k / total_var;
var_cumulative = cumsum(var_explained);

% Prepare results
results = struct(...
    'loadings', loadings, ...
    'uniquenesses', uniquenesses, ...
    'communalities', communalities, ...
    'iterations', iter, ...
    'converged', converged, ...
    'log_likelihood', new_loglik, ...
    'eigenvalues', eigenvalues_k, ...
    'var_explained', var_explained, ...
    'var_cumulative', var_cumulative);

% Warning if not converged
if ~converged
    warning('Maximum likelihood estimation did not converge within %d iterations.', options.max_iterations);
end

end

function results = rotate_factors(loadings, method, options)
% ROTATE_FACTORS Applies rotation to factor loadings to improve interpretability
%
% USAGE:
%   RESULTS = rotate_factors(LOADINGS, METHOD)
%   RESULTS = rotate_factors(LOADINGS, METHOD, OPTIONS)
%
% INPUTS:
%   LOADINGS - N by K matrix of factor loadings
%   METHOD   - String specifying rotation method:
%              'varimax', 'quartimax', 'promax', 'oblimin', or 'none'
%   OPTIONS  - [OPTIONAL] Structure with rotation options:
%              For 'promax': OPTIONS.kappa - Kappa parameter (default: 4)
%              For 'oblimin': OPTIONS.gamma - Gamma parameter (default: 0)
%
% OUTPUTS:
%   RESULTS  - Structure with fields:
%               RESULTS.loadings - N by K matrix of rotated loadings
%               RESULTS.rotation_matrix - K by K rotation matrix
%               RESULTS.factor_corr - K by K factor correlation matrix
%                                     (only for oblique rotations)

% Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Get dimensions of the loadings matrix
[N, K] = size(loadings);

% If no rotation or 'none' is specified, return original loadings
if strcmpi(method, 'none')
    results = struct(...
        'loadings', loadings, ...
        'rotation_matrix', eye(K));
    return;
end

% Apply selected rotation method
switch lower(method)
    case 'varimax'
        % Varimax rotation (orthogonal)
        % Kaiser normalization (normalize rows by their communalities)
        h = sqrt(sum(loadings.^2, 2));
        A = loadings ./ h;
        
        % Initialize rotation matrix
        T = eye(K);
        
        % Iteration until convergence
        max_iter = 100;
        tol = 1e-8;
        d = 1;
        iter = 0;
        
        while d > tol && iter < max_iter
            iter = iter + 1;
            
            % For each pair of factors
            for i = 1:(K-1)
                for j = (i+1):K
                    % Extract the relevant columns
                    a = A(:, i);
                    b = A(:, j);
                    
                    % Calculate rotation angle
                    u = a.^2 - b.^2;
                    v = 2 * a .* b;
                    num = 2 * sum(u .* v);
                    den = sum(u.^2 - v.^2);
                    tau = atan2(num, den) / 4;
                    
                    % Create rotation matrix for this pair
                    Tij = eye(K);
                    Tij(i, i) = cos(tau);
                    Tij(i, j) = -sin(tau);
                    Tij(j, i) = sin(tau);
                    Tij(j, j) = cos(tau);
                    
                    % Update A and T
                    A = A * Tij;
                    T = T * Tij;
                end
            end
            
            % Calculate criterion
            B = A.^4;
            B_sum = sum(B, 1);
            g1 = sum(B_sum.^2);
            g2 = sum(sum(A.^2).^2);
            new_d = g1 - g2 / N;
            
            % Check convergence
            if iter > 1
                d = abs(new_d - old_d);
            end
            old_d = new_d;
        end
        
        % Reverse normalization
        A = A .* h;
        
        % Return results
        results = struct(...
            'loadings', A, ...
            'rotation_matrix', T);
        
    case 'quartimax'
        % Quartimax rotation (orthogonal)
        % Kaiser normalization
        h = sqrt(sum(loadings.^2, 2));
        A = loadings ./ h;
        
        % Initialize rotation matrix
        T = eye(K);
        
        % Iteration until convergence
        max_iter = 100;
        tol = 1e-8;
        d = 1;
        iter = 0;
        
        while d > tol && iter < max_iter
            iter = iter + 1;
            
            % For each pair of factors
            for i = 1:(K-1)
                for j = (i+1):K
                    % Extract the relevant columns
                    a = A(:, i);
                    b = A(:, j);
                    
                    % Calculate rotation angle for quartimax
                    num = sum(a.^3 .* b - a .* b.^3);
                    den = sum(a.^4 - 6 * a.^2 .* b.^2 + b.^4) / 4;
                    tau = atan2(num, den) / 4;
                    
                    % Create rotation matrix for this pair
                    Tij = eye(K);
                    Tij(i, i) = cos(tau);
                    Tij(i, j) = -sin(tau);
                    Tij(j, i) = sin(tau);
                    Tij(j, j) = cos(tau);
                    
                    % Update A and T
                    A = A * Tij;
                    T = T * Tij;
                end
            end
            
            % Calculate criterion
            old_d = d;
            d = sum(sum(A.^4));
            
            % Check convergence
            if iter > 1
                if abs(d - old_d) < tol
                    break;
                end
            end
        end
        
        % Reverse normalization
        A = A .* h;
        
        % Return results
        results = struct(...
            'loadings', A, ...
            'rotation_matrix', T);
        
    case 'promax'
        % Promax rotation (oblique)
        % First apply varimax rotation
        varimax_results = rotate_factors(loadings, 'varimax', []);
        rotated_loadings = varimax_results.loadings;
        
        % Get kappa parameter (power for promax, typically between 2 and 4)
        if isfield(options, 'kappa')
            kappa = options.kappa;
        else
            kappa = 4;  % Default kappa value
        end
        
        % Create target matrix: element-wise power of the varimax-rotated loadings
        P = rotated_loadings.^2 .* sign(rotated_loadings);
        P = sign(rotated_loadings) .* abs(rotated_loadings).^kappa;
        
        % Fit each column of P to the corresponding column of rotated_loadings
        X = rotated_loadings' * rotated_loadings;
        B = inv(X) * rotated_loadings' * P;
        
        % Normalize columns of B to have unit sum of squares
        for j = 1:K
            B(:, j) = B(:, j) / sqrt(sum(B(:, j).^2));
        end
        
        % Compute the oblique transformation matrix
        T = varimax_results.rotation_matrix * B;
        
        % Compute factor loadings
        L = loadings * T;
        
        % Compute factor correlation matrix
        phi = inv(T' * T);
        
        % Return results
        results = struct(...
            'loadings', L, ...
            'rotation_matrix', T, ...
            'factor_corr', phi);
        
    case 'oblimin'
        % Oblimin rotation (oblique)
        % First apply varimax rotation as initial rotation
        varimax_results = rotate_factors(loadings, 'varimax', []);
        L = varimax_results.loadings;
        T = varimax_results.rotation_matrix;
        
        % Get gamma parameter for oblimin
        if isfield(options, 'gamma')
            gamma = options.gamma;
        else
            gamma = 0;  % Default is direct quartimin
        end
        
        % Kaiser normalization (normalize rows by their communalities)
        h = sqrt(sum(L.^2, 2));
        A = L ./ h;
        
        % Initialize oblique transformation matrix
        Q = eye(K);
        
        % Iteration until convergence
        max_iter = 100;
        tol = 1e-8;
        d = 1;
        iter = 0;
        
        while d > tol && iter < max_iter
            iter = iter + 1;
            
            % For each pair of factors
            for i = 1:(K-1)
                for j = (i+1):K
                    % Calculate rotation parameters for oblimin
                    p = A(:, i);
                    q = A(:, j);
                    S = A' * A;
                    
                    % Oblimin specific calculations
                    w1 = p.^2 - gamma * sum(p.^2) / N;
                    w2 = q.^2 - gamma * sum(q.^2) / N;
                    
                    num = 2 * sum(p .* q .* w2) - 2 * sum(p .* q .* w1);
                    den = sum(p.^2 .* w2) - sum(q.^2 .* w1);
                    
                    if abs(den) > 1e-10
                        tan4tau = num / den;
                        tau = atan(tan4tau) / 4;
                    else
                        tau = pi/8 * sign(num);  % 45 degrees if denominator is close to zero
                    end
                    
                    % Create rotation matrix for this pair
                    Qij = eye(K);
                    c = cos(tau);
                    s = sin(tau);
                    Qij(i, i) = c;
                    Qij(i, j) = s;
                    Qij(j, i) = -s;
                    Qij(j, j) = c;
                    
                    % Update A and Q
                    A = A * Qij;
                    Q = Q * Qij;
                end
            end
            
            % Calculate criterion for oblimin
            S = A' * A;
            B = A.^2;
            C = sum(B, 1)';
            E = N * gamma;
            
            G = zeros(K, K);
            for i = 1:K
                for j = 1:K
                    if i ~= j
                        G(i, j) = S(i, j) * (C(i) - E) * (C(j) - E);
                    end
                end
            end
            
            new_d = sum(G(:).^2);
            
            % Check convergence
            if iter > 1
                d = abs(new_d - old_d);
            end
            old_d = new_d;
        end
        
        % Reverse normalization
        A = A .* h;
        
        % Construct final transformation matrix
        Tfinal = T * Q;
        
        % Compute factor correlation matrix
        phi = inv(Q' * Q);
        
        % Return results
        results = struct(...
            'loadings', A, ...
            'rotation_matrix', Tfinal, ...
            'factor_corr', phi);
        
    otherwise
        error('Unsupported rotation method: %s', method);
end

end

function scores = compute_factor_scores(data, loadings, uniquenesses, method)
% COMPUTE_FACTOR_SCORES Calculates factor scores for observations based on the factor model
%
% USAGE:
%   SCORES = compute_factor_scores(DATA, LOADINGS, UNIQUENESSES, METHOD)
%
% INPUTS:
%   DATA         - T by N matrix of observed variables
%   LOADINGS     - N by K matrix of factor loadings
%   UNIQUENESSES - N by 1 vector of uniquenesses (specific variances)
%   METHOD       - String indicating the score computation method:
%                 'regression' (default), 'bartlett', or 'anderson'
%
% OUTPUTS:
%   SCORES       - T by K matrix of factor scores
%
% COMMENTS:
%   This function implements three common methods for computing factor scores:
%   1. Regression method: Computes scores using multiple regression
%   2. Bartlett method: Computes maximum likelihood estimates of the factor scores
%   3. Anderson-Rubin method: Produces uncorrelated scores with unit variance

% Get dimensions
[T, N] = size(data);
[N2, K] = size(loadings);

% Check for dimension consistency
if N ~= N2
    error('Number of variables in DATA and LOADINGS must match.');
end

if length(uniquenesses) ~= N
    error('Length of UNIQUENESSES must match number of variables.');
end

% Ensure uniquenesses are positive
uniquenesses = max(uniquenesses, eps);

% Compute factor scores based on selected method
switch lower(method)
    case 'regression'
        % Regression method (Thompson's scores)
        % F = X * Lambda * (Lambda' * Lambda)^(-1)
        
        % Compute regression coefficients
        Lambda = loadings;
        coef = Lambda * inv(Lambda' * Lambda);
        
        % Compute scores
        scores = data * coef;
        
    case 'bartlett'
        % Bartlett's method (weighted least squares)
        % F = (Lambda' * Psi^(-1) * Lambda)^(-1) * Lambda' * Psi^(-1) * X
        
        % Create diagonal matrix of uniquenesses
        Psi_inv = diag(1 ./ uniquenesses);
        
        % Compute Bartlett's weights
        Lambda = loadings;
        W = inv(Lambda' * Psi_inv * Lambda) * Lambda' * Psi_inv;
        
        % Compute scores
        scores = data * W';
        
    case 'anderson'
        % Anderson-Rubin method (standardized orthogonal scores)
        % F = X * Psi^(-1/2) * Lambda * (Lambda' * Psi^(-1) * Lambda)^(-1/2)
        
        % Create diagonal matrix of uniquenesses
        Psi_inv = diag(1 ./ uniquenesses);
        Psi_inv_sqrt = diag(1 ./ sqrt(uniquenesses));
        
        % Compute Anderson-Rubin weights
        Lambda = loadings;
        M = inv(sqrtm(Lambda' * Psi_inv * Lambda));
        W = Psi_inv_sqrt * Lambda * M;
        
        % Compute scores
        scores = data * W;
        
        % Standardize to ensure unit variance
        scores = scores ./ std(scores);
        
    otherwise
        error('Unsupported factor score method: %s', method);
end

end

function model = dynamic_factor_model(data, k, p, options)
% DYNAMIC_FACTOR_MODEL Estimates a dynamic factor model combining factor analysis with VAR dynamics
%
% USAGE:
%   MODEL = dynamic_factor_model(DATA, K, P)
%   MODEL = dynamic_factor_model(DATA, K, P, OPTIONS)
%
% INPUTS:
%   DATA    - T by N matrix of time series data 
%   K       - Number of factors to extract
%   P       - VAR lag order for modeling factor dynamics
%   OPTIONS - [OPTIONAL] Structure with estimation options:
%               OPTIONS.factor_options - Options for factor_model function
%               OPTIONS.var_options    - Options for var_model function
%
% OUTPUTS:
%   MODEL   - Structure with fields:
%               MODEL.factor_model - Factor model estimation results
%               MODEL.var_model    - VAR model for factor dynamics
%               MODEL.loadings     - Factor loadings
%               MODEL.factors      - Extracted factors
%               MODEL.residuals    - Model residuals
%               MODEL.fitted       - Fitted values
%               MODEL.sigma        - Innovation covariance matrix
%               MODEL.k            - Number of factors
%               MODEL.p            - VAR lag order
%               MODEL.impulse      - Impulse response analysis
%               MODEL.fevd         - Forecast error variance decomposition
%
% COMMENTS:
%   Dynamic factor models combine factor analysis with time series dynamics.
%   The model can be represented as:
%   1. Static factor model: X_t = Lambda * f_t + e_t
%   2. Dynamic factor process: f_t = A_1*f_{t-1} + ... + A_p*f_{t-p} + u_t
%
%   This formulation allows for modeling both cross-sectional and temporal 
%   dependencies in multivariate time series data, which is particularly
%   useful for macroeconomic and financial data analysis.

% Step 1: Validate inputs
data = datacheck(data, 'data');

% Validate number of factors
k_options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
k = parametercheck(k, 'k', k_options);

% Validate lag order
p_options = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
p = parametercheck(p, 'p', p_options);

% Set default options
if nargin < 4 || isempty(options)
    options = struct();
end

if ~isfield(options, 'factor_options')
    options.factor_options = struct();
end

if ~isfield(options, 'var_options')
    options.var_options = struct();
end

% Step 2: Extract static factors
factor_model_results = factor_model(data, k, options.factor_options);
factors = factor_model_results.factors;
loadings = factor_model_results.loadings;

% Step 3: Model dynamic evolution of factors using VAR
% Skip if p=0 (static model only)
if p > 0
    var_model_results = var_model(factors, p, options.var_options);
else
    % Create a minimal structure for consistency
    var_model_results = struct(...
        'coefficients', [], ...
        'constant', [], ...
        'residuals', zeros(size(factors)), ...
        'sigma', eye(k), ...
        'fitted', factors, ...
        'p', 0);
end

% Step 4: Compute fitted values and residuals for the full model
if p > 0
    % For the dynamic model, fitted values are based on the VAR predictions
    fitted_factors = var_model_results.fitted;
    factor_residuals = var_model_results.residuals;
    
    % The start of the sample has missing fitted values due to lags
    % Fill these with the actual factors for convenience
    if size(fitted_factors, 1) < size(factors, 1)
        fitted_factors = [factors(1:p, :); fitted_factors];
        factor_residuals = [zeros(p, k); factor_residuals];
    end
else
    % For the static model, fitted values are just the factors
    fitted_factors = factors;
    factor_residuals = zeros(size(factors));
end

% Compute fitted values for the original variables
fitted_values = fitted_factors * loadings';

% Compute residuals for the original variables
residuals = data - fitted_values;

% Step 5: Perform impulse response analysis for the dynamic model
if p > 0
    % Calculate impulse responses for 20 periods ahead
    impulse = var_irf(var_model_results, 20);
    
    % Calculate forecast error variance decomposition
    fevd = var_fevd(var_model_results, 20);
else
    impulse = [];
    fevd = [];
end

% Step 6: Create the output structure
model = struct(...
    'factor_model', factor_model_results, ...
    'var_model', var_model_results, ...
    'loadings', loadings, ...
    'factors', factors, ...
    'fitted_factors', fitted_factors, ...
    'factor_residuals', factor_residuals, ...
    'fitted', fitted_values, ...
    'residuals', residuals, ...
    'k', k, ...
    'p', p, ...
    'impulse', impulse, ...
    'fevd', fevd);

end

function forecasts = factor_model_forecast(model, h)
% FACTOR_MODEL_FORECAST Generates forecasts from a factor model
%
% USAGE:
%   FORECASTS = factor_model_forecast(MODEL, H)
%
% INPUTS:
%   MODEL      - Factor model structure returned by factor_model or dynamic_factor_model
%   H          - Forecast horizon (number of periods ahead)
%
% OUTPUTS:
%   FORECASTS  - T+1:T+H by N matrix of forecasts for the original variables
%
% COMMENTS:
%   This function generates forecasts from both static and dynamic factor models.
%   For static models, forecasts are based on the latest factor values.
%   For dynamic models, factor forecasts are generated using the VAR dynamics,
%   then mapped back to the original variables using the factor loadings.

% Step 1: Validate inputs
h_options = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
h = parametercheck(h, 'h', h_options);

% Determine if we're dealing with a static or dynamic model
is_dynamic = isfield(model, 'var_model') && isfield(model.var_model, 'p') && model.var_model.p > 0;

% Step 2: Generate factor forecasts
if is_dynamic
    % For a dynamic model, use VAR forecasting
    factor_forecasts = var_forecast(model.var_model, h);
else
    % For a static model, just repeat the last factor values
    factors = model.factors;
    factor_forecasts = repmat(factors(end, :), h, 1);
end

% Step 3: Map factor forecasts to original variables using factor loadings
loadings = model.loadings;
variable_forecasts = factor_forecasts * loadings';

% Step 4: If data was mean-centered during estimation, add back the mean
if isfield(model, 'data_mean') || (isfield(model.factor_model, 'data_mean'))
    if isfield(model, 'data_mean')
        data_mean = model.data_mean;
    else
        data_mean = model.factor_model.data_mean;
    end
    
    variable_forecasts = variable_forecasts + ones(h, 1) * data_mean;
end

% Return forecasts
forecasts = variable_forecasts;

end

function results = factor_model_bootstrap(data, model, options)
% FACTOR_MODEL_BOOTSTRAP Performs bootstrap inference for factor model parameters
%
% USAGE:
%   RESULTS = factor_model_bootstrap(DATA, MODEL, OPTIONS)
%
% INPUTS:
%   DATA    - T by N matrix of original data
%   MODEL   - Structure containing factor model parameters (loadings, uniquenesses, etc.)
%             as returned by factor_model
%   OPTIONS - [OPTIONAL] Structure with bootstrap options:
%             OPTIONS.replications - Number of bootstrap samples (default: 1000)
%             OPTIONS.conf_level   - Confidence level (default: 0.95)
%             OPTIONS.method       - Bootstrap method: 'block' or 'stationary'
%             OPTIONS.block_size   - Block size for block bootstrap (if method='block')
%             OPTIONS.p            - Probability parameter for stationary bootstrap
%
% OUTPUTS:
%   RESULTS - Structure with bootstrap results:
%             RESULTS.loadings_se     - Standard errors for factor loadings
%             RESULTS.loadings_ci     - Confidence intervals for loadings
%             RESULTS.uniquenesses_se - Standard errors for uniquenesses
%             RESULTS.uniquenesses_ci - Confidence intervals for uniquenesses
%             RESULTS.bootstrap_stats - Full bootstrap distribution statistics
%
% COMMENTS:
%   This function performs bootstrap inference for factor model parameters,
%   accounting for the time series nature of the data. It uses either 
%   block or stationary bootstrap to preserve the temporal dependence structure.
%   
%   The analysis provides standard errors and confidence intervals for 
%   factor loadings and uniquenesses, which are essential for assessing 
%   the statistical significance of factor model results.

% Step 1: Set default options if not provided
if nargin < 3 || isempty(options)
    options = struct();
end

% Set bootstrap defaults
if ~isfield(options, 'replications')
    options.replications = 1000;
end

if ~isfield(options, 'conf_level')
    options.conf_level = 0.95;
end

if ~isfield(options, 'method')
    options.method = 'block';
end

% Extract model parameters
loadings = model.loadings;
uniquenesses = model.uniquenesses;
k = model.k;
method = model.method;
standardize = model.standardize;
rotate = model.rotate;

% Step 2: Define statistic function for bootstrap
stat_fn = @(x) compute_bootstrap_stat(x, k, method, standardize, rotate);

% Step 3: Call bootstrap_confidence_intervals
bootstrap_results = bootstrap_confidence_intervals(data, stat_fn, options);

% Step 4: Process bootstrap results
bootstrap_stats = bootstrap_results.bootstrap_statistics;

% Get dimensions
[N, K] = size(loadings);
num_boot = size(bootstrap_stats, 1);

% Extract loadings and uniquenesses from bootstrap distribution
boot_loadings = zeros(num_boot, N, K);
boot_uniquenesses = zeros(num_boot, N);

for b = 1:num_boot
    % Bootstrap statistics are stored as a vector, need to reshape
    boot_params = bootstrap_stats(b, :);
    
    % Extract loadings (first N*K elements)
    boot_loadings(b, :, :) = reshape(boot_params(1:(N*K)), N, K);
    
    % Extract uniquenesses (next N elements)
    boot_uniquenesses(b, :) = boot_params((N*K+1):(N*K+N));
end

% Compute bootstrap standard errors
loadings_se = zeros(N, K);
uniquenesses_se = zeros(N, 1);

for i = 1:N
    for j = 1:K
        loadings_se(i, j) = std(squeeze(boot_loadings(:, i, j)));
    end
    uniquenesses_se(i) = std(boot_uniquenesses(:, i));
end

% Compute bootstrap confidence intervals
alpha = 1 - options.conf_level;
lower_pct = alpha / 2 * 100;
upper_pct = (1 - alpha / 2) * 100;

loadings_ci = zeros(N, K, 2);
uniquenesses_ci = zeros(N, 2);

for i = 1:N
    for j = 1:K
        loadings_ci(i, j, 1) = prctile(squeeze(boot_loadings(:, i, j)), lower_pct);
        loadings_ci(i, j, 2) = prctile(squeeze(boot_loadings(:, i, j)), upper_pct);
    end
    
    uniquenesses_ci(i, 1) = prctile(boot_uniquenesses(:, i), lower_pct);
    uniquenesses_ci(i, 2) = prctile(boot_uniquenesses(:, i), upper_pct);
end

% Prepare results structure
results = struct(...
    'loadings_se', loadings_se, ...
    'loadings_ci', loadings_ci, ...
    'uniquenesses_se', uniquenesses_se, ...
    'uniquenesses_ci', uniquenesses_ci, ...
    'bootstrap_stats', bootstrap_stats, ...
    'options', options);

end

function stat = compute_bootstrap_stat(x, k, method, standardize, rotate)
% Helper function for bootstrap inference that re-estimates the factor model
% on a bootstrap sample and returns the parameters as a vector

% Re-estimate the factor model on bootstrap sample
options = struct('method', method, 'standardize', standardize, 'rotate', rotate);
model = factor_model(x, k, options);

% Extract parameters of interest
loadings = model.loadings;
uniquenesses = model.uniquenesses;

% Convert parameters to a vector for bootstrap_confidence_intervals
stat = [loadings(:); uniquenesses];

end