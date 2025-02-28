function se = nwse(X, residuals, lag)
% NWSE Computes Newey-West heteroskedasticity and autocorrelation consistent (HAC) standard errors
%
% USAGE:
%   SE = nwse(X, residuals, lag)
%
% INPUTS:
%   X         - T by K matrix of regressors
%   residuals - T by 1 vector of regression residuals
%   lag       - Non-negative integer indicating the number of lags to use in HAC estimation
%
% OUTPUTS:
%   SE        - K by 1 vector of Newey-West HAC standard errors
%
% COMMENTS:
%   Implements Newey-West (1987) HAC standard errors with Bartlett kernel for
%   heteroskedasticity and autocorrelation consistent covariance estimation.
%   The implementation uses the formula:
%
%   V = (X'X)^(-1) * S * (X'X)^(-1)
%
%   where S is a weighted sum of autocovariance matrices:
%
%   S = S0 + sum_{l=1}^L [w(l,L) * (Sl + Sl')]
%
%   with S0 = X'*diag(e^2)*X, Sl = X'_{t-l}*e_t*e_{t-l}*X_{t-l}, and w(l,L) = 1 - l/(L+1)
%   is the Bartlett kernel weight.
%
% EXAMPLES:
%   % Linear regression model
%   y = X * beta + e;
%   beta_hat = (X'*X)\(X'*y);
%   residuals = y - X * beta_hat;
%   robust_se = nwse(X, residuals, 4);
%
% REFERENCES:
%   Newey, W.K., and K.D. West (1987). "A Simple, Positive Semi-definite,
%   Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
%   Econometrica, 55(3), 703-708.
%
% See also OLS, DATACHECK, COLUMNCHECK, PARAMETERCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate inputs
X = datacheck(X, 'X');
residuals = columncheck(residuals, 'residuals');

% Validate lag parameter
if nargin < 3 || isempty(lag)
    error('LAG must be provided and cannot be empty');
end

if ~isscalar(lag)
    error('LAG must be a scalar value');
end

% Use parametercheck for comprehensive lag validation
options = struct('isscalar', true, 'isInteger', true, 'isNonNegative', true);
lag = parametercheck(lag, 'lag', options);

% Step 2: Check if dimensions are compatible
[T, k] = size(X);
[r, c] = size(residuals);
if r ~= T
    error('X and residuals must have the same number of observations');
end

% Step 3: Compute the "bread" of the sandwich estimator
XX = X' * X;
XXinv = inv(XX);

% Step 4: Initialize the meat of the sandwich (weighted sum of autocovariance matrices)
S = zeros(k, k);

% Step 5: Calculate the base S0 matrix (heteroskedasticity component)
e2 = residuals.^2;
S0 = X' * diag(e2) * X;  % This explicitly follows the formula
S = S + S0;

% Step 6: Add autocorrelation components
for l = 1:lag
    Sl = zeros(k, k);
    for t = (l+1):T
        Sl = Sl + residuals(t) * residuals(t-l) * (X(t,:)' * X(t-l,:));
    end
    % Apply Bartlett kernel weight (declining with lag)
    weight = 1 - l/(lag+1);
    % Add both Sl and its transpose (for symmetry)
    S = S + weight * (Sl + Sl');
end

% Step 7: Apply small sample adjustment
S = S * (T / (T - k));

% Step 8: Compute the full sandwich estimator
V = XXinv * S * XXinv;

% Step 9: Extract the standard errors from the diagonal
se = sqrt(diag(V));

end