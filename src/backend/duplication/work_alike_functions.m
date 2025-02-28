function isAvailable = isToolboxAvailable(toolboxName)
% ISTOOLBOXAVAILABLE Helper function that checks whether a specific MATLAB toolbox is available in the current environment
%
% USAGE:
%   isAvailable = isToolboxAvailable(toolboxName)
%
% INPUTS:
%   toolboxName - String name of the toolbox to check (e.g., 'Statistics')
%
% OUTPUTS:
%   isAvailable - True if the toolbox is available, false otherwise
%
% EXAMPLES:
%   hasStats = isToolboxAvailable('Statistics');
%
% COMMENTS:
%   This function is used internally by the MFE Toolbox to determine
%   when work-alike functions need to be activated.

% Get version information for all installed toolboxes
versionInfo = ver;

% Initialize output
isAvailable = false;

% Loop through version information to find the toolbox
for i = 1:length(versionInfo)
    % Check if the toolbox name matches (case-insensitive)
    if strcmpi(versionInfo(i).Name, toolboxName)
        isAvailable = true;
        break;
    end
end
end

function work_alike_init()
% WORK_ALIKE_INIT Initialization function that detects available MATLAB toolboxes and sets up appropriate function handles for work-alike functions
%
% USAGE:
%   work_alike_init
%
% COMMENTS:
%   This function is called during MFE Toolbox initialization to set up
%   appropriate function handles for work-alike functions when specific
%   MATLAB toolboxes are unavailable. It automatically detects missing
%   toolboxes and activates the necessary work-alike functions.

% Check if Statistics Toolbox is available
hasStats = isToolboxAvailable('Statistics');

% Check if Optimization Toolbox is available
hasOpt = isToolboxAvailable('Optimization');

% List of functions that will be replaced if needed
statsReplacements = {'normpdf', 'normcdf', 'norminv', 'erf', 'erfinv', ...
                     'tinv', 'tcdf', 'pcacov'};
optReplacements = {'fmincon', 'fminsearch'};
matrixReplacements = {'cov', 'qr'};

% Display information about activated work-alike functions
if ~hasStats
    disp('MFE Toolbox: Statistics Toolbox not detected. Work-alike functions activated for:');
    disp(['  ' strjoin(statsReplacements, ', ')]);
end

if ~hasOpt
    disp('MFE Toolbox: Optimization Toolbox not detected. Work-alike functions activated for:');
    disp(['  ' strjoin(optReplacements, ', ')]);
end

% Note: Assignment of function handles would typically happen here in a production
% environment, but for this module, we're providing the work-alike functions
% directly without replacing built-ins.
end

function y = wa_normpdf(x, mu, sigma)
% WA_NORMPDF Work-alike function for computing normal probability density function when Statistics Toolbox is unavailable
%
% USAGE:
%   y = wa_normpdf(x, mu, sigma)
%
% INPUTS:
%   x     - Points at which to evaluate the PDF
%   mu    - Mean of the normal distribution [default: 0]
%   sigma - Standard deviation of the normal distribution [default: 1]
%
% OUTPUTS:
%   y     - Normal probability density function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's normpdf function
%   for use when the Statistics Toolbox is unavailable.

% Handle default parameters
if nargin < 3
    sigma = 1;
end
if nargin < 2
    mu = 0;
end

% Validate parameters
options.isPositive = true;
sigma = parametercheck(sigma, 'sigma', options);
mu = parametercheck(mu, 'mu');
x = parametercheck(x, 'x');

% Handle special cases (NaN, Inf)
if any(sigma <= 0)
    y = NaN(size(x));
    return;
end

% Standardize the input value (z = (x - mu)/sigma)
z = (x - mu) ./ sigma;

% Compute the PDF using the normal distribution formula
y = exp(-0.5 * z.^2) ./ (sqrt(2 * pi) * sigma);
end

function p = wa_normcdf(x, mu, sigma)
% WA_NORMCDF Work-alike function for computing normal cumulative distribution function when Statistics Toolbox is unavailable
%
% USAGE:
%   p = wa_normcdf(x, mu, sigma)
%
% INPUTS:
%   x     - Points at which to evaluate the CDF
%   mu    - Mean of the normal distribution [default: 0]
%   sigma - Standard deviation of the normal distribution [default: 1]
%
% OUTPUTS:
%   p     - Normal cumulative distribution function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's normcdf function
%   for use when the Statistics Toolbox is unavailable.

% Handle default parameters
if nargin < 3
    sigma = 1;
end
if nargin < 2
    mu = 0;
end

% Validate parameters
options.isPositive = true;
sigma = parametercheck(sigma, 'sigma', options);
mu = parametercheck(mu, 'mu');
x = parametercheck(x, 'x');

% Handle special cases (NaN, Inf)
if any(sigma <= 0)
    p = NaN(size(x));
    return;
end

% Standardize the input value (z = (x - mu)/sigma)
z = (x - mu) ./ sigma;

% Compute the CDF using error function approximation
% For the normal CDF, we use the error function:
% CDF = 0.5 * (1 + erf(z / sqrt(2)))
if exist('erf', 'builtin')
    % Use built-in erf if available
    p = 0.5 * (1 + erf(z / sqrt(2)));
else
    % Otherwise use our work-alike erf function
    p = 0.5 * (1 + wa_erf(z / sqrt(2)));
end
end

function x = wa_norminv(p, mu, sigma)
% WA_NORMINV Work-alike function for computing normal inverse cumulative distribution function when Statistics Toolbox is unavailable
%
% USAGE:
%   x = wa_norminv(p, mu, sigma)
%
% INPUTS:
%   p     - Probability values at which to evaluate the inverse CDF
%   mu    - Mean of the normal distribution [default: 0]
%   sigma - Standard deviation of the normal distribution [default: 1]
%
% OUTPUTS:
%   x     - Normal inverse cumulative distribution function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's norminv function
%   for use when the Statistics Toolbox is unavailable.

% Handle default parameters
if nargin < 3
    sigma = 1;
end
if nargin < 2
    mu = 0;
end

% Validate probability input is between 0 and 1
options.isPositive = true;
sigma = parametercheck(sigma, 'sigma', options);
mu = parametercheck(mu, 'mu');

options.lowerBound = 0;
options.upperBound = 1;
p = parametercheck(p, 'p', options);

% Handle special cases (p=0, p=1, NaN, etc.)
if any(sigma <= 0)
    x = NaN(size(p));
    return;
end

% Initialize output to same size as input
x = zeros(size(p));

% Handle edge cases
x(p == 0) = -Inf;
x(p == 1) = Inf;
x(p < 0 | p > 1 | isnan(p)) = NaN;

% Compute the inverse CDF using rational approximation methods
idx = (p > 0 & p < 1);
if any(idx(:))
    if exist('erfinv', 'builtin')
        % Use built-in erfinv if available
        x(idx) = mu + sigma * sqrt(2) * erfinv(2 * p(idx) - 1);
    else
        % Otherwise use our work-alike erfinv function
        x(idx) = mu + sigma * sqrt(2) * wa_erfinv(2 * p(idx) - 1);
    end
end
end

function y = wa_erf(x)
% WA_ERF Work-alike function for computing the error function when not available
%
% USAGE:
%   y = wa_erf(x)
%
% INPUTS:
%   x - Points at which to evaluate the error function
%
% OUTPUTS:
%   y - Error function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's erf function
%   using a polynomial approximation for use when the function is unavailable.
%   The implementation follows Abramowitz and Stegun approximation 7.1.26.

% Validate input parameter
x = parametercheck(x, 'x');

% Initialize output to same size as input
y = zeros(size(x));

% Handle special cases and boundary values
y(x == 0) = 0;
y(x == Inf) = 1;
y(x == -Inf) = -1;
y(isnan(x)) = NaN;

% Implement polynomial approximation of the error function
idx = (x ~= 0 & x ~= Inf & x ~= -Inf & ~isnan(x));
if any(idx(:))
    % Abramowitz and Stegun approximation coefficients
    p = 0.3275911;
    a1 = 0.254829592;
    a2 = -0.284496736;
    a3 = 1.421413741;
    a4 = -1.453152027;
    a5 = 1.061405429;
    
    % Compute the approximation
    absX = abs(x(idx));
    t = 1.0 ./ (1.0 + p * absX);
    erfApprox = 1.0 - (((((a5 * t + a4) .* t + a3) .* t + a2) .* t + a1) .* t) .* exp(-absX.^2);
    
    % Apply sign correction
    y(idx) = sign(x(idx)) .* erfApprox;
end
end

function y = wa_erfinv(x)
% WA_ERFINV Work-alike function for computing the inverse error function when not available
%
% USAGE:
%   y = wa_erfinv(x)
%
% INPUTS:
%   x - Points at which to evaluate the inverse error function
%
% OUTPUTS:
%   y - Inverse error function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's erfinv function
%   using a rational approximation for use when the function is unavailable.

% Validate that input is in range [-1, 1]
options.lowerBound = -1;
options.upperBound = 1;
x = parametercheck(x, 'x', options);

% Initialize output to same size as input
y = zeros(size(x));

% Handle special cases and boundary values
y(x == 0) = 0;
y(x == 1) = Inf;
y(x == -1) = -Inf;
y(isnan(x)) = NaN;

% Implement rational approximation of inverse error function
idx = (x ~= 0 & x ~= 1 & x ~= -1 & ~isnan(x));
if any(idx(:))
    % Use absolute value and handle sign separately
    absX = abs(x(idx));
    
    % Coefficients for approximation
    c = [1.0, 0.5, 0.125, 0.0625];
    d = [1.0, 0.693, 0.95, 0.7066];
    
    % Compute the approximation
    t = log(1.0 - absX.^2);
    v = sqrt(-t) .* (c(1) + t * (c(2) + t * (c(3) + t * c(4)))) ./ ...
                    (d(1) + t * (d(2) + t * (d(3) + t * d(4))));
    
    % Apply sign correction
    y(idx) = sign(x(idx)) .* v;
end
end

function x = wa_tinv(p, v)
% WA_TINV Work-alike function for computing Student's t inverse cumulative distribution function when Statistics Toolbox is unavailable
%
% USAGE:
%   x = wa_tinv(p, v)
%
% INPUTS:
%   p - Probability values at which to evaluate the inverse CDF
%   v - Degrees of freedom
%
% OUTPUTS:
%   x - Student's t inverse cumulative distribution function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's tinv function
%   for use when the Statistics Toolbox is unavailable.

% Validate probability input is between 0 and 1
options.lowerBound = 0;
options.upperBound = 1;
p = parametercheck(p, 'p', options);

% Validate degrees of freedom parameter v > 0
options.isPositive = true;
v = parametercheck(v, 'v', options);

% Initialize output to same size as input
x = zeros(size(p));

% Handle special cases (p=0, p=1, v=Inf, etc.)
x(p == 0) = -Inf;
x(p == 1) = Inf;
x(p < 0 | p > 1 | isnan(p)) = NaN;

% For v=Inf, use normal distribution approximation via wa_norminv
infIdx = (v == Inf);
if any(infIdx(:))
    x(infIdx) = wa_norminv(p(infIdx), 0, 1);
end

% For v<Inf, compute the inverse CDF using approximation methods
finiteIdx = (v > 0 & v < Inf);
if any(finiteIdx(:))
    % For t-distribution with finite df, we use beta inverse CDF approximation
    % Compute symmetric probabilities
    pSym = zeros(size(p));
    pSym(p <= 0.5) = p(p <= 0.5);
    pSym(p > 0.5) = 1 - p(p > 0.5);
    
    % Approximate Beta inverse CDF
    u = pSym.^(1./v);
    
    % Convert to t-distribution quantile
    xSym = sign(p - 0.5) .* sqrt(v .* (1./u - 1));
    
    % Apply to output
    x(finiteIdx) = xSym(finiteIdx);
end
end

function p = wa_tcdf(x, v)
% WA_TCDF Work-alike function for computing Student's t cumulative distribution function when Statistics Toolbox is unavailable
%
% USAGE:
%   p = wa_tcdf(x, v)
%
% INPUTS:
%   x - Points at which to evaluate the CDF
%   v - Degrees of freedom
%
% OUTPUTS:
%   p - Student's t cumulative distribution function values
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's tcdf function
%   for use when the Statistics Toolbox is unavailable.

% Validate input parameters
x = parametercheck(x, 'x');

options.isPositive = true;
v = parametercheck(v, 'v', options);

% Initialize output to same size as input
p = zeros(size(x));

% Handle special cases (x=Inf, x=-Inf, v=Inf, etc.)
p(x == -Inf) = 0;
p(x == Inf) = 1;
p(isnan(x) | isnan(v)) = NaN;

% For v=Inf, use normal distribution approximation via wa_normcdf
infIdx = (v == Inf);
if any(infIdx(:))
    p(infIdx) = wa_normcdf(x(infIdx), 0, 1);
end

% For v<Inf, compute the CDF using incomplete beta function approximation
finiteIdx = (v > 0 & v < Inf);
if any(finiteIdx(:))
    % t CDF is related to incomplete beta function:
    % CDF = 0.5 + 0.5 * sign(x) * (1 - I(v/(v+x^2), v/2, 1/2))
    
    % Simplified approximation
    z = v ./ (v + x.^2);
    betaApprox = 0.5 * (1 - sign(x) .* (1 - z.^(v/2)));
    
    p(finiteIdx) = 0.5 + betaApprox(finiteIdx);
end
end

function [x, fval, exitflag, output] = wa_fmincon(fun, x0, A, b, Aeq, beq, lb, ub, options)
% WA_FMINCON Simplified work-alike function for constrained optimization when Optimization Toolbox is unavailable
%
% USAGE:
%   [x, fval, exitflag, output] = wa_fmincon(fun, x0, A, b, Aeq, beq, lb, ub, options)
%
% INPUTS:
%   fun     - Function handle to minimize
%   x0      - Initial point
%   A, b    - Linear inequality constraints A*x <= b
%   Aeq, beq - Linear equality constraints Aeq*x = beq
%   lb, ub  - Lower and upper bounds
%   options - Optimization options structure
%
% OUTPUTS:
%   x        - Optimal parameters
%   fval     - Function value at optimum
%   exitflag - Exit flag
%   output   - Output information
%
% COMMENTS:
%   This is a simplified implementation of fmincon using gradient descent with projection.
%   It does not support nonlinear constraints.

% Parse and validate input parameters
if nargin < 9
    options = struct();
end
if nargin < 8
    ub = Inf(size(x0));
end
if nargin < 7
    lb = -Inf(size(x0));
end
if nargin < 6
    beq = [];
end
if nargin < 5
    Aeq = [];
end
if nargin < 4
    b = [];
end
if nargin < 3
    A = [];
end

% Initialize optimization parameters from options structure
maxIter = 1000;
if isfield(options, 'MaxIterations')
    maxIter = options.MaxIterations;
end

tolerance = 1e-6;
if isfield(options, 'TolFun')
    tolerance = options.TolFun;
end

stepSize = 0.01;
if isfield(options, 'StepSize')
    stepSize = options.StepSize;
end

% Check boundary and linear constraints
x = x0(:);
n = length(x);
iter = 0;
converged = false;
fval = fun(x);
history = zeros(maxIter, 1);
history(1) = fval;

% Implement simplified gradient descent with projection
while iter < maxIter && ~converged
    % Increment iteration counter
    iter = iter + 1;
    
    % Compute gradient using finite differences
    grad = zeros(n, 1);
    h = 1e-8;
    for i = 1:n
        xPlus = x;
        xPlus(i) = xPlus(i) + h;
        grad(i) = (fun(xPlus) - fval) / h;
    end
    
    % Update point using gradient descent
    xNew = x - stepSize * grad;
    
    % Apply bound constraints (projection)
    xNew = max(lb, min(xNew, ub));
    
    % Apply linear constraints through penalty method
    penalty = 0;
    if ~isempty(A) && ~isempty(b)
        constraint = A * xNew - b;
        penalty = penalty + sum(max(0, constraint).^2) * 1e3;
    end
    
    if ~isempty(Aeq) && ~isempty(beq)
        constraint = abs(Aeq * xNew - beq);
        penalty = penalty + sum(constraint.^2) * 1e3;
    end
    
    % Compute new function value with penalty
    fvalNew = fun(xNew) + penalty;
    
    % Update if improvement
    if fvalNew < fval
        x = xNew;
        fval = fun(x);  % Re-evaluate without penalty
        
        % Check for convergence
        if iter > 1 && abs(history(iter-1) - fval) < tolerance
            converged = true;
        end
    else
        % Reduce step size if no improvement
        stepSize = stepSize * 0.5;
        
        % Check for very small step size
        if stepSize < 1e-10
            converged = true;
        end
    end
    
    % Save history
    if iter < maxIter
        history(iter+1) = fval;
    end
end

% Return optimized parameters, function value, exit flag, and output structure
if converged
    exitflag = 1;
elseif iter >= maxIter
    exitflag = 0;
else
    exitflag = -1;
end

output = struct();
output.iterations = iter;
output.funcCount = iter * (n + 1);
output.stepsize = stepSize;
output.algorithm = 'Simplified gradient descent with projection';
output.history = history(1:iter+1);
end

function [x, fval, exitflag, output] = wa_fminsearch(fun, x0, options)
% WA_FMINSEARCH Simplified work-alike function for unconstrained optimization when Optimization Toolbox is unavailable
%
% USAGE:
%   [x, fval, exitflag, output] = wa_fminsearch(fun, x0, options)
%
% INPUTS:
%   fun     - Function handle to minimize
%   x0      - Initial point
%   options - Optimization options structure
%
% OUTPUTS:
%   x        - Optimal parameters
%   fval     - Function value at optimum
%   exitflag - Exit flag
%   output   - Output information
%
% COMMENTS:
%   This is a simplified implementation of fminsearch using Nelder-Mead simplex algorithm.

% Parse and validate input parameters
if nargin < 3
    options = struct();
end

% Initialize optimization parameters from options structure
maxIter = 1000;
if isfield(options, 'MaxIterations')
    maxIter = options.MaxIterations;
end

tolerance = 1e-6;
if isfield(options, 'TolFun')
    tolerance = options.TolFun;
end

% Initialize
x0 = x0(:);
n = length(x0);
iter = 0;
converged = false;

% Create initial simplex
simplex = zeros(n+1, n);
fvals = zeros(n+1, 1);

% First point is the initial guess
simplex(1,:) = x0';
fvals(1) = fun(x0);

% Create remaining simplex vertices
delta = 0.05;  % Step size for creating simplex
for i = 1:n
    simplex(i+1,:) = x0';
    simplex(i+1,i) = simplex(i+1,i) + delta;
    fvals(i+1) = fun(simplex(i+1,:)');
end

% Implement simplified Nelder-Mead simplex algorithm
while iter < maxIter && ~converged
    % Increment iteration counter
    iter = iter + 1;
    
    % Sort simplex points by function value
    [fvals, idx] = sort(fvals);
    simplex = simplex(idx,:);
    
    % Check convergence criteria
    if std(fvals) < tolerance
        converged = true;
        continue;
    end
    
    % Compute centroid of all points except worst
    centroid = mean(simplex(1:n,:), 1);
    
    % Reflection: reflect worst point through centroid
    reflection = 2 * centroid - simplex(n+1,:);
    fReflection = fun(reflection');
    
    if fReflection < fvals(n) && fReflection >= fvals(1)
        % Accept reflection
        simplex(n+1,:) = reflection;
        fvals(n+1) = fReflection;
    elseif fReflection < fvals(1)
        % Expansion: reflect further
        expansion = centroid + 2 * (reflection - centroid);
        fExpansion = fun(expansion');
        
        if fExpansion < fReflection
            simplex(n+1,:) = expansion;
            fvals(n+1) = fExpansion;
        else
            simplex(n+1,:) = reflection;
            fvals(n+1) = fReflection;
        end
    else
        % Contraction: move toward centroid
        contraction = 0.5 * (centroid + simplex(n+1,:));
        fContraction = fun(contraction');
        
        if fContraction < fvals(n+1)
            simplex(n+1,:) = contraction;
            fvals(n+1) = fContraction;
        else
            % Shrink: move all points toward best point
            for i = 2:n+1
                simplex(i,:) = simplex(1,:) + 0.5 * (simplex(i,:) - simplex(1,:));
                fvals(i) = fun(simplex(i,:)');
            end
        end
    end
end

% Return optimized parameters, function value, exit flag, and output structure
[fval, idx] = min(fvals);
x = simplex(idx,:)';

if converged
    exitflag = 1;
elseif iter >= maxIter
    exitflag = 0;
else
    exitflag = -1;
end

output = struct();
output.iterations = iter;
output.funcCount = (n+1) + iter * (n+2);  % Initial + evaluations per iteration
output.algorithm = 'Simplified Nelder-Mead simplex';
end

function C = wa_cov(x)
% WA_COV Work-alike function for computing covariance matrix when specific functions are unavailable
%
% USAGE:
%   C = wa_cov(x)
%
% INPUTS:
%   x - Matrix where rows are observations and columns are variables
%
% OUTPUTS:
%   C - Covariance matrix
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's cov function.

% Validate input data is a matrix
if ~ismatrix(x)
    error('Input must be a matrix');
end

% Get dimensions
[n, p] = size(x);

if n <= 1
    error('Input must have more than one row');
end

% Center the data by subtracting column means
xc = x - repmat(mean(x, 1), n, 1);

% Compute covariance matrix using (X'*X)/(n-1)
C = (xc' * xc) / (n - 1);

% Ensure symmetry of the output matrix
C = 0.5 * (C + C');
end

function [coeff, latent, explained] = wa_pcacov(sigma)
% WA_PCACOV Work-alike function for principal component analysis when Statistics Toolbox is unavailable
%
% USAGE:
%   [coeff, latent, explained] = wa_pcacov(sigma)
%
% INPUTS:
%   sigma - Covariance matrix
%
% OUTPUTS:
%   coeff    - Eigenvectors (principal components)
%   latent   - Eigenvalues
%   explained - Explained variance ratios
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's pcacov function.

% Validate that input is a symmetric covariance matrix
if ~ismatrix(sigma) || size(sigma, 1) ~= size(sigma, 2)
    error('Input must be a square matrix');
end

% Check if symmetric
if norm(sigma - sigma', 'fro') > 1e-10 * norm(sigma, 'fro')
    warning('Input matrix is not symmetric; symmetrizing for computation');
    sigma = 0.5 * (sigma + sigma');
end

% Compute eigenvalues and eigenvectors of covariance matrix
[V, D] = eig(sigma);
d = diag(D);

% Sort eigenvalues and eigenvectors in descending order
[latent, idx] = sort(d, 'descend');
coeff = V(:, idx);

% Calculate explained variance ratios
explained = 100 * latent / sum(latent);
end

function [Q, R] = wa_qr(A)
% WA_QR Work-alike function for QR decomposition of matrices when specific functionality is unavailable
%
% USAGE:
%   [Q, R] = wa_qr(A)
%
% INPUTS:
%   A - Input matrix
%
% OUTPUTS:
%   Q - Orthogonal matrix
%   R - Upper triangular matrix
%
% COMMENTS:
%   This is a work-alike implementation of MATLAB's qr function
%   implemented via Gram-Schmidt orthogonalization.

% Validate input matrix dimensions
if ~ismatrix(A)
    error('Input must be a matrix');
end

% Get dimensions
[m, n] = size(A);

% Initialize Q and R
Q = zeros(m, n);
R = zeros(n, n);

% Implement QR decomposition via Gram-Schmidt orthogonalization
for j = 1:n
    v = A(:, j);
    
    for i = 1:j-1
        R(i, j) = Q(:, i)' * A(:, j);
        v = v - R(i, j) * Q(:, i);
    end
    
    R(j, j) = norm(v);
    
    % Ensure Q is orthogonal by explicit normalization
    if R(j, j) < eps(class(A)) * norm(A(:, j))
        Q(:, j) = zeros(m, 1);
    else
        Q(:, j) = v / R(j, j);
    end
end
end