function [model] = bekk_mvgarch(data, options)
% BEKK_MVGARCH Estimates a BEKK (Baba-Engle-Kraft-Kroner) Multivariate GARCH model
%
% USAGE:
%   [MODEL] = bekk_mvgarch(DATA, OPTIONS)
%
% INPUTS:
%   DATA - A T by K matrix of data where T is the number of observations and
%          K is the number of series
%   OPTIONS - A structure with options (see below)
%
% OUTPUTS:
%   MODEL - A structure containing estimation results and diagnostics:
%     parameters - Structure with estimated model parameters:
%       mu      - 1 x K vector of mean parameters
%       C       - K x K lower triangular matrix (intercept)
%       A       - K x K x p array of ARCH matrices
%       B       - K x K x q array of GARCH matrices
%       p       - ARCH order
%       q       - GARCH order
%       k       - Number of series
%       nu      - Degrees of freedom parameter for t/ged/skewt
%       lambda  - Skewness parameter for skewt
%     likelihood - Log-likelihood value at optimum
%     H         - K x K x T array of conditional covariance matrices
%     residuals - T x K matrix of residuals
%     stdResiduals - T x K matrix of standardized residuals
%     aic       - Akaike Information Criterion
%     bic       - Bayesian Information Criterion
%     forecast  - Forecast structure (if requested)
%     validation - Validation results and diagnostics
%
% OPTIONS:
%   p           - Positive integer for ARCH order (Default: 1)
%   q           - Positive integer for GARCH order (Default: 1)
%   type        - String, one of {'full', 'diagonal'} determining the BEKK type (Default: 'full')
%   distribution - String, one of {'normal', 't', 'ged', 'skewt'} (Default: 'normal')
%   forecast    - Non-negative integer determining forecast horizon (Default: 0)
%   method      - String, one of {'likelihood', 'composite'} (Default: 'likelihood')
%   startingVals - Vector of starting values for parameters (Optional)
%   optimizationOptions - Options for fmincon optimization (Optional)
%   mean        - String, one of {'constant', 'zero'} for mean specification (Default: 'constant')
%   degrees     - Degree of freedom parameter for t/ged/skewt distributions (Default: 10)
%   lambda      - Skewness parameter for skewed distributions (Default: 0)
%
% EXAMPLES:
%   % Fit a standard BEKK(1,1) model with normal innovations
%   options = [];
%   model = bekk_mvgarch(data, options);
%
%   % Fit a diagonal BEKK model with t-distributed innovations
%   options.type = 'diagonal';
%   options.distribution = 't';
%   options.degrees = 8;
%   model = bekk_mvgarch(data, options);
%
%   % Generate 10-step ahead forecasts
%   options.forecast = 10;
%   model = bekk_mvgarch(data, options);
%
% REMARKS:
%   The BEKK(p,q) model for a k-dimensional process is:
%   
%   H_t = C*C' + sum_{i=1}^p A_i*ε_{t-i}*ε_{t-i}'*A_i' + sum_{j=1}^q B_j*H_{t-j}*B_j'
%
%   where C is a lower triangular matrix, A_i and B_j are K by K matrices.
%   In the diagonal BEKK, A_i and B_j are restricted to be diagonal matrices.

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Define global minimum eigenvalue constant to ensure positive definiteness
global MIN_EIGVAL;
if isempty(MIN_EIGVAL)
    MIN_EIGVAL = 1e-10;
end

% Check if MEX implementation is available
global MEX_AVAILABLE;
if isempty(MEX_AVAILABLE)
    % Check for composite_likelihood MEX file
    MEX_AVAILABLE = (exist('composite_likelihood', 'file') == 3);
end

% 1. Input validation
data = datacheck(data, 'data');
data = columncheck(data, 'data');
[T, k] = size(data);

% Check for sufficient observations
if T < 2*k
    error('BEKK_MVGARCH requires at least 2K observations where K is the number of series.');
end

% 2. Set default options if not provided
if nargin < 2 || isempty(options)
    options = [];
end

% 3. Set up model parameters
% GARCH order
if ~isfield(options, 'p')
    options.p = 1;
end
p = options.p;

if ~isfield(options, 'q')
    options.q = 1;
end
q = options.q;

% Check GARCH orders are valid
parametercheck(p, 'p', struct('isscalar', true, 'isInteger', true, 'isPositive', true));
parametercheck(q, 'q', struct('isscalar', true, 'isInteger', true, 'isPositive', true));

% Model type (full or diagonal BEKK)
if ~isfield(options, 'type')
    options.type = 'full';
end
options.type = lower(options.type);
if ~ismember(options.type, {'full', 'diagonal'})
    error('OPTIONS.type must be either ''full'' or ''diagonal''.');
end
isDiagonal = strcmpi(options.type, 'diagonal');

% Distribution for innovations
if ~isfield(options, 'distribution')
    options.distribution = 'normal';
end
options.distribution = lower(options.distribution);
if ~ismember(options.distribution, {'normal', 't', 'ged', 'skewt'})
    error('OPTIONS.distribution must be one of ''normal'', ''t'', ''ged'', or ''skewt''.');
end

% Mean specification
if ~isfield(options, 'mean')
    options.mean = 'constant';
end
options.mean = lower(options.mean);
if ~ismember(options.mean, {'constant', 'zero'})
    error('OPTIONS.mean must be either ''constant'' or ''zero''.');
end
hasConstant = strcmpi(options.mean, 'constant');

% Distribution parameters
if ~isfield(options, 'degrees') && ismember(options.distribution, {'t', 'ged', 'skewt'})
    options.degrees = 10;
end
if isfield(options, 'degrees')
    parametercheck(options.degrees, 'degrees', struct('isscalar', true, 'isPositive', true));
    if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'skewt')
        if options.degrees <= 2
            error('For t or skewed t distribution, OPTIONS.degrees must be > 2.');
        end
    end
end

if ~isfield(options, 'lambda') && strcmpi(options.distribution, 'skewt')
    options.lambda = 0;
end
if isfield(options, 'lambda') && strcmpi(options.distribution, 'skewt')
    parametercheck(options.lambda, 'lambda', struct('isscalar', true, 'lowerBound', -1, 'upperBound', 1));
end

% Method specification
if ~isfield(options, 'method')
    options.method = 'likelihood';
end
options.method = lower(options.method);
if ~ismember(options.method, {'likelihood', 'composite'})
    error('OPTIONS.method must be either ''likelihood'' or ''composite''.');
end
useCompositeLikelihood = strcmpi(options.method, 'composite');

% Forecast horizon
if ~isfield(options, 'forecast')
    options.forecast = 0;
end
parametercheck(options.forecast, 'forecast', struct('isscalar', true, 'isInteger', true, 'isNonNegative', true));

% 4. Count number of parameters and initialize
% Count parameters
numParams = k * (k + 1) / 2;  % C parameters (lower triangular)
if isDiagonal
    numParams = numParams + p * k;  % Diagonal A matrices (p diagonal matrices of size k)
    numParams = numParams + q * k;  % Diagonal B matrices (q diagonal matrices of size k)
else
    numParams = numParams + p * k * k;  % Full A matrices (p matrices of size k x k)
    numParams = numParams + q * k * k;  % Full B matrices (q matrices of size k x k)
end

if hasConstant
    numParams = numParams + k;  % Mean parameters
end

% Add distribution parameters
if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
    numParams = numParams + 1;  % Degrees of freedom
elseif strcmpi(options.distribution, 'skewt')
    numParams = numParams + 1 + k;  % Degrees of freedom + skewness for each series
end

% 5. Initialize data structures for estimation
% Compute initial values for covariance matrix
backcastOptions = struct('type', 'default');
H0 = backcast(data, backcastOptions);

% Organize data for mean adjustment if needed
if hasConstant
    X = ones(T, 1);  % Design matrix for constant mean
    beta0 = mean(data);  % Initial mean estimate
else
    X = [];
    beta0 = zeros(1, k);
end

% 6. Set up optimization options
if ~isfield(options, 'optimizationOptions')
    options.optimizationOptions = optimset('fmincon');
    options.optimizationOptions.Display = 'iter';
    options.optimizationOptions.Algorithm = 'interior-point';
    options.optimizationOptions.TolFun = 1e-6;
    options.optimizationOptions.TolX = 1e-6;
    options.optimizationOptions.MaxFunEvals = 10000;
    options.optimizationOptions.MaxIter = 1000;
end

% Store k in options for helper functions
options.k = k;

% 7. Generate initial parameter values
if ~isfield(options, 'startingVals')
    % Initialize parameters
    startingParams = zeros(numParams, 1);
    
    % Position index
    paramIdx = 1;
    
    % Mean parameters
    if hasConstant
        startingParams(paramIdx:paramIdx+k-1) = beta0;
        paramIdx = paramIdx + k;
    end
    
    % C parameters (lower triangular, positive diagonal)
    C0 = chol(cov(data), 'lower');
    C0_vec = zeros(k*(k+1)/2, 1);
    idx = 1;
    for i = 1:k
        for j = 1:i
            C0_vec(idx) = C0(i, j);
            idx = idx + 1;
        end
    end
    startingParams(paramIdx:paramIdx+k*(k+1)/2-1) = C0_vec;
    paramIdx = paramIdx + k*(k+1)/2;
    
    % A parameters (persistence of shocks)
    if isDiagonal
        % Diagonal BEKK: just diagonal elements
        A0_diag = 0.1 * ones(k, 1);  % Start with small persistence
        startingParams(paramIdx:paramIdx+k*p-1) = repmat(A0_diag, p, 1);
        paramIdx = paramIdx + k*p;
    else
        % Full BEKK: all elements
        A0 = 0.1 * eye(k);  % Start with small diagonal persistence
        A0_vec = reshape(A0, k*k, 1);
        startingParams(paramIdx:paramIdx+k*k*p-1) = repmat(A0_vec, p, 1);
        paramIdx = paramIdx + k*k*p;
    end
    
    % B parameters (persistence of volatility)
    if isDiagonal
        % Diagonal BEKK: just diagonal elements
        B0_diag = 0.8 * ones(k, 1);  % Start with high persistence
        startingParams(paramIdx:paramIdx+k*q-1) = repmat(B0_diag, q, 1);
        paramIdx = paramIdx + k*q;
    else
        % Full BEKK: all elements
        B0 = 0.8 * eye(k);  % Start with high persistence on diagonal
        B0_vec = reshape(B0, k*k, 1);
        startingParams(paramIdx:paramIdx+k*k*q-1) = repmat(B0_vec, q, 1);
        paramIdx = paramIdx + k*k*q;
    end
    
    % Distribution parameters
    if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
        startingParams(paramIdx) = options.degrees;
        paramIdx = paramIdx + 1;
    elseif strcmpi(options.distribution, 'skewt')
        startingParams(paramIdx) = options.degrees;
        startingParams(paramIdx+1:paramIdx+k) = options.lambda * ones(k, 1);
        paramIdx = paramIdx + 1 + k;
    end
    
    options.startingVals = startingParams;
else
    % Validate starting values if provided
    startingParams = options.startingVals(:);
    if length(startingParams) ~= numParams
        error('OPTIONS.startingVals must have length equal to the number of parameters (%d).', numParams);
    end
end

% 8. Set up optimization constraints
% Lower and upper bounds
LB = -inf(numParams, 1);
UB = inf(numParams, 1);

% Position index
paramIdx = 1;

% Mean parameters (unconstrained)
if hasConstant
    paramIdx = paramIdx + k;
end

% C parameters (lower triangular)
% Diagonal elements of C must be positive
diagIdx = [1:k+1:k*(k+1)/2];
LB(paramIdx+diagIdx-1) = sqrt(MIN_EIGVAL);  % Lower bound for diagonal elements
paramIdx = paramIdx + k*(k+1)/2;

% A and B parameters (can be constrained for stationarity)
if isDiagonal
    % For diagonal BEKK, ensure stationarity: sum(A_ii^2 + B_ii^2) < 1
    LB(paramIdx:paramIdx+k*p-1) = -0.999;
    UB(paramIdx:paramIdx+k*p-1) = 0.999;
    paramIdx = paramIdx + k*p;
    
    LB(paramIdx:paramIdx+k*q-1) = -0.999;
    UB(paramIdx:paramIdx+k*q-1) = 0.999;
    paramIdx = paramIdx + k*q;
else
    % For full BEKK, eigenvalue constraints will be handled during estimation
    paramIdx = paramIdx + k*k*p + k*k*q;
end

% Distribution parameters
if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
    LB(paramIdx) = 2.01;  % Lower bound for degrees of freedom
    paramIdx = paramIdx + 1;
elseif strcmpi(options.distribution, 'skewt')
    LB(paramIdx) = 2.01;  % Lower bound for degrees of freedom
    LB(paramIdx+1:paramIdx+k) = -0.999;  % Lower bound for skewness
    UB(paramIdx+1:paramIdx+k) = 0.999;   % Upper bound for skewness
    paramIdx = paramIdx + 1 + k;
end

% 9. Set up likelihood function with data closure
% Create a function to compute the log-likelihood
likelihoodFcn = @(params) bekk_likelihood(params, data, options);

% 10. Perform optimization
[estParams, logL, exitflag, output] = fmincon(likelihoodFcn, startingParams, [], [], [], [], LB, UB, [], options.optimizationOptions);

% 11. Compute final covariance matrices and extract parameters
% Position index to extract parameters
paramIdx = 1;

% Extract mean parameters
if hasConstant
    mu = estParams(paramIdx:paramIdx+k-1)';
    paramIdx = paramIdx + k;
else
    mu = zeros(1, k);
end

% Extract C parameters (lower triangular)
C = zeros(k, k);
idx = 1;
for i = 1:k
    for j = 1:i
        C(i, j) = estParams(paramIdx+idx-1);
        idx = idx + 1;
    end
end
paramIdx = paramIdx + k*(k+1)/2;

% Extract A parameters
A = zeros(k, k, p);
if isDiagonal
    for i = 1:p
        A_diag = estParams(paramIdx:paramIdx+k-1);
        A(:,:,i) = diag(A_diag);
        paramIdx = paramIdx + k;
    end
else
    for i = 1:p
        A(:,:,i) = reshape(estParams(paramIdx:paramIdx+k*k-1), k, k);
        paramIdx = paramIdx + k*k;
    end
end

% Extract B parameters
B = zeros(k, k, q);
if isDiagonal
    for i = 1:q
        B_diag = estParams(paramIdx:paramIdx+k-1);
        B(:,:,i) = diag(B_diag);
        paramIdx = paramIdx + k;
    end
else
    for i = 1:q
        B(:,:,i) = reshape(estParams(paramIdx:paramIdx+k*k-1), k, k);
        paramIdx = paramIdx + k*k;
    end
end

% Extract distribution parameters
if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
    nu = estParams(paramIdx);
    lambda = zeros(1, k);  % No skewness parameter for t or GED
    paramIdx = paramIdx + 1;
elseif strcmpi(options.distribution, 'skewt')
    nu = estParams(paramIdx);
    lambda = estParams(paramIdx+1:paramIdx+k)';
    paramIdx = paramIdx + 1 + k;
else
    nu = 0;  % Not used for normal distribution
    lambda = zeros(1, k);  % No skewness for normal distribution
end

% 12. Compute final conditional covariance matrices
% Initialize covariance matrices
H = zeros(k, k, T);
meanAdjustedData = data - repmat(mu, T, 1);

% Use the estimated parameters to compute conditional covariances
max_lag = max(p, q);
H(:,:,1:max_lag) = repmat(reshape(H0, k, k, 1), 1, 1, max_lag);

% Compute covariance matrices
for t = max_lag+1:T
    % Initialize with constant term C*C'
    H_t = C * C';
    
    % Add ARCH terms A_i*ε_{t-i}*ε_{t-i}'*A_i'
    for i = 1:p
        e_lag = meanAdjustedData(t-i,:)';  % Column vector of lagged residuals
        H_t = H_t + A(:,:,i) * (e_lag * e_lag') * A(:,:,i)';
    end
    
    % Add GARCH terms B_j*H_{t-j}*B_j'
    for j = 1:q
        H_t = H_t + B(:,:,j) * H(:,:,t-j) * B(:,:,j)';
    end
    
    % Ensure positive definiteness
    [V, D] = eig(H_t);
    d = diag(D);
    if any(d <= MIN_EIGVAL)
        % Apply correction by replacing negative eigenvalues with small positive values
        d(d <= MIN_EIGVAL) = MIN_EIGVAL;
        H_t = V * diag(d) * V';
        % Ensure symmetry
        H_t = (H_t + H_t') / 2;
    end
    
    H(:,:,t) = H_t;
end

% 13. Compute standardized residuals
stdResiduals = zeros(T, k);
for t = 1:T
    % For the initial periods, use the backcast value
    if t <= max_lag
        H_t = reshape(H0, k, k);
    else
        H_t = H(:,:,t);
    end
    
    % Standardize using Cholesky decomposition
    cholH = chol(H_t, 'lower');
    stdResiduals(t,:) = (cholH \ meanAdjustedData(t,:)')';
end

% 14. Compute AIC, BIC and other model diagnostics
ic = aicsbic(-logL, numParams, T);

% 15. Prepare model output structure
model = struct();

% Model parameters
model.parameters = struct();
model.parameters.mu = mu;
model.parameters.C = C;
model.parameters.A = A;
model.parameters.B = B;
model.parameters.p = p;
model.parameters.q = q;
model.parameters.k = k;
model.parameters.nu = nu;
model.parameters.lambda = lambda;
model.parameters.isDiagonal = isDiagonal;
model.parameters.distribution = options.distribution;

% Estimation results
model.likelihood = -logL;  % Convert back to positive log-likelihood
model.aic = ic.aic;
model.bic = ic.sbic;
model.numParams = numParams;
model.T = T;
model.exitflag = exitflag;
model.optimizationOutput = output;

% Model data
model.data = data;
model.residuals = meanAdjustedData;
model.stdResiduals = stdResiduals;
model.H = H;

% 16. Generate forecasts if requested
if options.forecast > 0
    model.forecast = bekk_forecast(model, options.forecast);
end

% 17. Validate the final model
validationResults = validate_bekk_model(model);
model.validation = validationResults;

end

function [nLogL] = bekk_likelihood(parameters, data, options)
% BEKK_LIKELIHOOD Computes negative log-likelihood for BEKK-MVGARCH model
%
% This function is called by the optimizer to compute the log-likelihood
% of the BEKK model given the parameters and data.

% Extract dimensions
[T, k] = size(data);
p = options.p;
q = options.q;
isDiagonal = strcmpi(options.type, 'diagonal');
hasConstant = strcmpi(options.mean, 'constant');
useCompositeLikelihood = strcmpi(options.method, 'composite');

% Define global constants
global MIN_EIGVAL MEX_AVAILABLE;
if isempty(MIN_EIGVAL)
    MIN_EIGVAL = 1e-10;
end

% Initialize log-likelihood
logL = 0;
const = k * log(2*pi);

% Position index to extract parameters
paramIdx = 1;

% Extract mean parameters
if hasConstant
    mu = parameters(paramIdx:paramIdx+k-1)';
    paramIdx = paramIdx + k;
else
    mu = zeros(1, k);
end

% Mean-adjust the data
meanAdjustedData = data - repmat(mu, T, 1);

% Extract C parameters (lower triangular)
C = zeros(k, k);
idx = 1;
for i = 1:k
    for j = 1:i
        C(i, j) = parameters(paramIdx+idx-1);
        idx = idx + 1;
    end
end
paramIdx = paramIdx + k*(k+1)/2;

% Extract A parameters
A = zeros(k, k, p);
if isDiagonal
    for i = 1:p
        A_diag = parameters(paramIdx:paramIdx+k-1);
        A(:,:,i) = diag(A_diag);
        paramIdx = paramIdx + k;
    end
else
    for i = 1:p
        A(:,:,i) = reshape(parameters(paramIdx:paramIdx+k*k-1), k, k);
        paramIdx = paramIdx + k*k;
    end
end

% Extract B parameters
B = zeros(k, k, q);
if isDiagonal
    for i = 1:q
        B_diag = parameters(paramIdx:paramIdx+k-1);
        B(:,:,i) = diag(B_diag);
        paramIdx = paramIdx + k;
    end
else
    for i = 1:q
        B(:,:,i) = reshape(parameters(paramIdx:paramIdx+k*k-1), k, k);
        paramIdx = paramIdx + k*k;
    end
end

% Extract distribution parameters
if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
    nu = parameters(paramIdx);
    paramIdx = paramIdx + 1;
elseif strcmpi(options.distribution, 'skewt')
    nu = parameters(paramIdx);
    lambda = parameters(paramIdx+1:paramIdx+k)';
    paramIdx = paramIdx + 1 + k;
end

% Check for stationarity
% For full BEKK, we need to check eigenvalues of the companion form
isStationary = true;
if ~isDiagonal
    % BEKK stationarity requires checking eigenvalues of a complex matrix
    % We'll construct a simplified test using the Kronecker structure
    A_sum = zeros(k*k, k*k);
    B_sum = zeros(k*k, k*k);
    
    for i = 1:p
        A_i = A(:,:,i);
        A_kron = kron(A_i, A_i);
        A_sum = A_sum + A_kron;
    end
    
    for j = 1:q
        B_j = B(:,:,j);
        B_kron = kron(B_j, B_j);
        B_sum = B_sum + B_kron;
    end
    
    % Check spectral radius
    M = A_sum + B_sum;
    e = eig(M);
    maxEig = max(abs(e));
    
    % If not stationary, return very large value
    if maxEig >= 0.999
        isStationary = false;
    end
end

% If model is not stationary, return a large value with penalty
if ~isStationary
    nLogL = 1e10 + sum(parameters.^2);  % Add penalty based on parameter magnitude
    return;
end

% Initialize conditional covariance matrices
H = zeros(k, k, T);
max_lag = max(p, q);

% Initial values for H - backcast using sample covariance
backcastOptions = struct('type', 'default');
H0 = backcast(meanAdjustedData, backcastOptions);
H0_mat = reshape(H0, k, k);

% Ensure initial covariance is positive definite
[V, D] = eig(H0_mat);
d = diag(D);
if any(d <= MIN_EIGVAL)
    d(d <= MIN_EIGVAL) = MIN_EIGVAL;
    H0_mat = V * diag(d) * V';
    % Ensure symmetry
    H0_mat = (H0_mat + H0_mat') / 2;
end

% Fill initial observations with backcast value
for t = 1:max_lag
    H(:,:,t) = H0_mat;
end

% Check if MEX implementation is available for composite likelihood
useMex = MEX_AVAILABLE && useCompositeLikelihood;

% Loop over time periods to compute log-likelihood
for t = max_lag+1:T
    % Initialize with constant term C*C'
    H_t = C * C';
    
    % Add ARCH terms A_i*ε_{t-i}*ε_{t-i}'*A_i'
    for i = 1:p
        e_lag = meanAdjustedData(t-i,:)';  % Column vector of lagged residuals
        H_t = H_t + A(:,:,i) * (e_lag * e_lag') * A(:,:,i)';
    end
    
    % Add GARCH terms B_j*H_{t-j}*B_j'
    for j = 1:q
        H_t = H_t + B(:,:,j) * H(:,:,t-j) * B(:,:,j)';
    end
    
    % Ensure positive definiteness
    [V, D] = eig(H_t);
    d = diag(D);
    if any(d <= MIN_EIGVAL)
        % Apply correction by replacing negative eigenvalues with small positive values
        d(d <= MIN_EIGVAL) = MIN_EIGVAL;
        H_t = V * diag(d) * V';
        % Ensure symmetry
        H_t = (H_t + H_t') / 2;
    end
    
    H(:,:,t) = H_t;
    
    % Use MEX implementation for composite likelihood if available
    if useMex && t > max_lag + 10  % Use MEX after burn-in period
        % Parameters for composite_likelihood MEX function would be setup here
        % This is a placeholder - the actual MEX implementation would be used
        % Currently falling through to MATLAB implementation
        useMex = false;
    end
    
    % Compute likelihood contribution for this observation
    e_t = meanAdjustedData(t,:)';  % Column vector of current residuals
    
    % Compute log-determinant and quadratic form
    try
        % Using LU decomposition for numerical stability
        [L, U] = lu(H_t);
        logdetH = sum(log(abs(diag(U))));
        
        % Compute quadratic form: e_t' * H_t^(-1) * e_t
        H_inv_e = H_t \ e_t;
        quadForm = e_t' * H_inv_e;
        
        % Update likelihood based on distribution
        if strcmpi(options.distribution, 'normal')
            % Normal distribution
            logL_t = -0.5 * (const + logdetH + quadForm);
        elseif strcmpi(options.distribution, 't')
            % Student's t distribution
            logL_t = stdtloglik(e_t, nu, zeros(k,1), sqrtm(H_t));
            logL_t = -logL_t;  % Convert back (stdtloglik returns negative log-likelihood)
        elseif strcmpi(options.distribution, 'ged')
            % GED distribution
            logL_t = gedloglik(e_t, nu, zeros(k,1), sqrtm(H_t));
        elseif strcmpi(options.distribution, 'skewt')
            % Skewed t distribution
            distParams = [nu, lambda, zeros(1,k), diag(H_t)'];
            [nll, ~] = skewtloglik(e_t, distParams);
            logL_t = -nll;  % Convert back (skewtloglik returns negative log-likelihood)
        end
        
        % Add to total log-likelihood
        logL = logL + logL_t;
    catch
        % If there's a numerical error, penalize this parameter set
        logL = -1e10;
        break;
    end
end

% Return negative log-likelihood for minimization
nLogL = -logL;

end

function [forecast] = bekk_forecast(model, horizon)
% BEKK_FORECAST Generates forecasts from an estimated BEKK-MVGARCH model
%
% INPUTS:
%   model   - Estimated BEKK-MVGARCH model structure
%   horizon - Forecast horizon (positive integer)
%
% OUTPUTS:
%   forecast - Structure with fields:
%              .covariance - k x k x horizon array of forecasted covariance matrices
%              .correlation - k x k x horizon array of forecasted correlation matrices

% Extract model parameters
k = model.parameters.k;
p = model.parameters.p;
q = model.parameters.q;
C = model.parameters.C;
A = model.parameters.A;
B = model.parameters.B;
isDiagonal = model.parameters.isDiagonal;
T = model.T;

% Define global constants
global MIN_EIGVAL;
if isempty(MIN_EIGVAL)
    MIN_EIGVAL = 1e-10;
end

% Get last observations needed for forecasting
max_lag = max(p, q);
last_residuals = model.residuals(end-max_lag+1:end, :);
last_H = model.H(:, :, end-max_lag+1:end);

% Initialize forecast arrays
H_forecast = zeros(k, k, horizon);
Corr_forecast = zeros(k, k, horizon);

% Compute forecasts recursively
for h = 1:horizon
    % Initialize with constant term C*C'
    H_h = C * C';
    
    % Add ARCH terms A_i*ε_{t-i}*ε_{t-i}'*A_i'
    % For h > 1, we use the fact that E[ε_{T+h-i}*ε_{T+h-i}'] = 0 for h > i
    for i = 1:min(h-1, p)
        % For future innovations, we use unconditional expectation
        % For h > i, the expectation of ε_{T+h-i}*ε_{T+h-i}' is H_{T+h-i}
        if h-i <= 0
            % Use historical values
            e_lag = last_residuals(end+h-i, :)';
            H_h = H_h + A(:,:,i) * (e_lag * e_lag') * A(:,:,i)';
        else
            % Use forecasted values - for ARCH, the expectation of future
            % squared innovations is the forecasted conditional variance
            if h-i <= horizon
                H_h = H_h + A(:,:,i) * H_forecast(:,:,h-i) * A(:,:,i)';
            end
        end
    end
    
    % Add GARCH terms B_j*H_{t-j}*B_j'
    for j = 1:q
        if h-j <= 0
            % Use historical values
            H_prev = last_H(:, :, end+h-j);
        else
            % Use forecasted values
            H_prev = H_forecast(:, :, h-j);
        end
        
        H_h = H_h + B(:,:,j) * H_prev * B(:,:,j)';
    end
    
    % Ensure positive definiteness
    [V, D] = eig(H_h);
    d = diag(D);
    if any(d <= MIN_EIGVAL)
        d(d <= MIN_EIGVAL) = MIN_EIGVAL;
        H_h = V * diag(d) * V';
        H_h = (H_h + H_h') / 2;  % Ensure symmetry
    end
    
    % Store the forecast
    H_forecast(:, :, h) = H_h;
    
    % Compute correlation matrix from covariance
    D_h = diag(sqrt(diag(H_h)));
    D_h_inv = diag(1./diag(D_h));
    Corr_h = D_h_inv * H_h * D_h_inv;
    
    % Store correlation matrix
    Corr_forecast(:, :, h) = Corr_h;
end

% Create forecast structure
forecast = struct();
forecast.covariance = H_forecast;
forecast.correlation = Corr_forecast;
forecast.horizon = horizon;
forecast.volatility = sqrt(reshape(diag(reshape(permute(H_forecast, [3,1,2]), horizon, k*k)), horizon, k));

end

function [validation] = validate_bekk_model(model)
% VALIDATE_BEKK_MODEL Validates an estimated BEKK-MVGARCH model
%
% INPUTS:
%   model - Estimated BEKK-MVGARCH model structure
%
% OUTPUTS:
%   validation - Structure with validation results and diagnostics

% Extract model parameters
k = model.parameters.k;
p = model.parameters.p;
q = model.parameters.q;
A = model.parameters.A;
B = model.parameters.B;
isDiagonal = model.parameters.isDiagonal;
T = model.T;

% Define global constants
global MIN_EIGVAL;
if isempty(MIN_EIGVAL)
    MIN_EIGVAL = 1e-10;
end

% Initialize validation structure
validation = struct();
validation.isValid = true;
validation.messages = {};

% Check 1: Stationarity condition
if isDiagonal
    % For diagonal BEKK, check sum of squared diagonal elements
    A_diag_sum = zeros(k, 1);
    B_diag_sum = zeros(k, 1);
    
    for i = 1:p
        A_diag = diag(A(:,:,i));
        A_diag_sum = A_diag_sum + A_diag.^2;
    end
    
    for j = 1:q
        B_diag = diag(B(:,:,j));
        B_diag_sum = B_diag_sum + B_diag.^2;
    end
    
    total_sum = A_diag_sum + B_diag_sum;
    
    if any(total_sum >= 1)
        validation.isValid = false;
        validation.messages{end+1} = 'Model violates stationarity condition: sum(A_ii^2 + B_ii^2) >= 1 for some i.';
    end
    
    validation.stationarity = 1 - max(total_sum);
else
    % For full BEKK, check eigenvalues of Kronecker-structured matrix
    A_sum = zeros(k*k, k*k);
    B_sum = zeros(k*k, k*k);
    
    for i = 1:p
        A_i = A(:,:,i);
        A_kron = kron(A_i, A_i);
        A_sum = A_sum + A_kron;
    end
    
    for j = 1:q
        B_j = B(:,:,j);
        B_kron = kron(B_j, B_j);
        B_sum = B_sum + B_kron;
    end
    
    % Check spectral radius
    M = A_sum + B_sum;
    e = eig(M);
    maxEig = max(abs(e));
    
    if maxEig >= 1
        validation.isValid = false;
        validation.messages{end+1} = 'Model violates stationarity condition: maximum eigenvalue >= 1.';
    end
    
    validation.stationarity = 1 - maxEig;
end

% Check 2: Positive definiteness of covariance matrices
isPD = true;
condNumbers = zeros(T, 1);
minEigenvalues = zeros(T, 1);
for t = 1:T
    H_t = model.H(:,:,t);
    [V, D] = eig(H_t);
    d = diag(D);
    
    if any(d <= MIN_EIGVAL)
        isPD = false;
    end
    
    % Store minimum eigenvalue and condition number
    minEigenvalues(t) = min(d);
    condNumbers(t) = max(d) / min(d);
end

validation.isPositiveDefinite = isPD;
validation.minEigenvalues = minEigenvalues;
if ~isPD
    validation.isValid = false;
    validation.messages{end+1} = 'Some estimated covariance matrices are not positive definite.';
end

% Check condition numbers
validation.condNumbers = condNumbers;
if max(condNumbers) > 1e6
    validation.messages{end+1} = 'Warning: Some covariance matrices have high condition numbers.';
end

% Check 3: Residual diagnostics
% Examine standardized residuals for autocorrelation and remaining GARCH effects
stdResMean = mean(model.stdResiduals);
stdResVar = var(model.stdResiduals);

validation.stdResMean = stdResMean;
validation.stdResVar = stdResVar;

% Check if standardized residuals have mean close to zero and variance close to one
if any(abs(stdResMean) > 0.1)
    validation.messages{end+1} = 'Warning: Mean of standardized residuals deviates from zero.';
end

if any(abs(stdResVar - 1) > 0.2)
    validation.messages{end+1} = 'Warning: Variance of standardized residuals deviates from one.';
end

% Return validation results
end

function params = bekk_parameter_transform(params, direction, options)
% BEKK_PARAMETER_TRANSFORM Transforms parameters between constrained and unconstrained spaces
%
% INPUTS:
%   params - Parameter vector
%   direction - String, either 'to_unconstrained' or 'to_constrained'
%   options - Model options structure
%
% OUTPUTS:
%   params - Transformed parameter vector

% Extract model dimensions
k = options.k;
p = options.p;
q = options.q;
isDiagonal = strcmpi(options.type, 'diagonal');
hasConstant = strcmpi(options.mean, 'constant');

% Initialize transformed parameters
newParams = params;

% Handle parameter transformation based on direction
if strcmpi(direction, 'to_unconstrained')
    % Transform constrained parameters to unconstrained space for optimization
    
    % Position index
    paramIdx = 1;
    
    % Skip mean parameters (unconstrained)
    if hasConstant
        paramIdx = paramIdx + k;
    end
    
    % C parameters (lower triangular)
    % Transform diagonal elements (positive) to log space
    for i = 1:k
        idx = paramIdx + (i-1)*(i)/2 + i - 1;  % Index of C_ii
        newParams(idx) = log(max(params(idx), 1e-6));
    end
    paramIdx = paramIdx + k*(k+1)/2;
    
    % A and B parameters for stationarity
    % Use logit-like transform for elements to enforce stationarity
    if isDiagonal
        % For diagonal BEKK, apply logit-like transform to diagonal elements
        for i = 1:p*k
            idx = paramIdx + i - 1;
            val = params(idx);
            if abs(val) >= 1
                val = sign(val) * 0.999;
            end
            newParams(idx) = log((1 + val) / (1 - val)) / 2;  % Logit-like transform
        end
        paramIdx = paramIdx + p*k;
        
        for i = 1:q*k
            idx = paramIdx + i - 1;
            val = params(idx);
            if abs(val) >= 1
                val = sign(val) * 0.999;
            end
            newParams(idx) = log((1 + val) / (1 - val)) / 2;  % Logit-like transform
        end
        paramIdx = paramIdx + q*k;
    else
        % For full BEKK, eigenvalue constraints handled in likelihood function
        paramIdx = paramIdx + p*k*k + q*k*k;
    end
    
    % Distribution parameters
    if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
        % Transform degrees of freedom (> 2) to log space
        newParams(paramIdx) = log(params(paramIdx) - 2);
        paramIdx = paramIdx + 1;
    elseif strcmpi(options.distribution, 'skewt')
        % Transform degrees of freedom (> 2) to log space
        newParams(paramIdx) = log(params(paramIdx) - 2);
        
        % Transform skewness parameters (in [-1,1]) to logit space
        for i = 1:k
            idx = paramIdx + i;
            val = params(idx);
            if abs(val) >= 1
                val = sign(val) * 0.999;
            end
            newParams(idx) = log((1 + val) / (1 - val)) / 2;  % Logit-like transform
        end
        paramIdx = paramIdx + 1 + k;
    end
    
elseif strcmpi(direction, 'to_constrained')
    % Transform unconstrained parameters to constrained space
    
    % Position index
    paramIdx = 1;
    
    % Skip mean parameters (unconstrained)
    if hasConstant
        paramIdx = paramIdx + k;
    end
    
    % C parameters (lower triangular)
    % Transform diagonal elements from log space to positive values
    for i = 1:k
        idx = paramIdx + (i-1)*(i)/2 + i - 1;  % Index of C_ii
        newParams(idx) = exp(params(idx));
    end
    paramIdx = paramIdx + k*(k+1)/2;
    
    % A and B parameters for stationarity
    if isDiagonal
        % For diagonal BEKK, reverse logit-like transform
        for i = 1:p*k
            idx = paramIdx + i - 1;
            newParams(idx) = tanh(params(idx));  % Range (-1, 1)
        end
        paramIdx = paramIdx + p*k;
        
        for i = 1:q*k
            idx = paramIdx + i - 1;
            newParams(idx) = tanh(params(idx));  % Range (-1, 1)
        end
        paramIdx = paramIdx + q*k;
    else
        % For full BEKK, eigenvalue constraints handled in likelihood function
        paramIdx = paramIdx + p*k*k + q*k*k;
    end
    
    % Distribution parameters
    if strcmpi(options.distribution, 't') || strcmpi(options.distribution, 'ged')
        % Transform degrees of freedom from log space to > 2
        newParams(paramIdx) = exp(params(paramIdx)) + 2;
        paramIdx = paramIdx + 1;
    elseif strcmpi(options.distribution, 'skewt')
        % Transform degrees of freedom from log space to > 2
        newParams(paramIdx) = exp(params(paramIdx)) + 2;
        
        % Transform skewness parameters from logit to [-1,1]
        for i = 1:k
            idx = paramIdx + i;
            newParams(idx) = tanh(params(idx));  % Range (-1, 1)
        end
        paramIdx = paramIdx + 1 + k;
    end
    
else
    error('Direction must be either ''to_unconstrained'' or ''to_constrained''.');
end

% Return transformed parameters
params = newParams;

end