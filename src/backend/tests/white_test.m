function results = white_test(residuals, X)
% WHITE_TEST Performs White's test for heteroskedasticity in regression residuals.
%
% USAGE:
%   RESULTS = white_test(residuals, X)
%
% INPUTS:
%   residuals - Residuals from a regression model (T x 1 vector)
%   X         - Regressors matrix from the regression model (T x K matrix)
%
% OUTPUTS:
%   RESULTS   - Structure containing:
%               results.stat    - White test statistic
%               results.pval    - P-value of the test
%               results.df      - Degrees of freedom
%               results.crit    - Critical values at [10%, 5%, 1%] significance levels
%               results.rej     - Rejection indicators at [10%, 5%, 1%] significance levels
%
% COMMENTS:
%   White's test examines the null hypothesis that the variance of the regression
%   errors is constant (homoskedasticity). The test involves regressing the squared
%   residuals on the original regressors, their squares, and their cross-products.
%   
%   The test statistic (T*R²) follows a chi-square distribution under the null,
%   with degrees of freedom equal to the number of regressors in the auxiliary
%   regression (excluding the constant).
%
%   A rejection of the null hypothesis indicates the presence of heteroskedasticity,
%   which has implications for the validity of standard errors and statistical inference
%   in regression analysis.
%
% EXAMPLES:
%   % Assuming a regression model: y = X*b + e
%   b = regress(y, X);                % Compute regression coefficients
%   residuals = y - X*b;              % Obtain residuals
%   results = white_test(residuals, X); % Perform White's test
%   disp(['White test p-value: ', num2str(results.pval)]);
%
% See also datacheck, columncheck, parametercheck, chi2cdf

% Copyright: MFE Toolbox
% Revision: 4.0    Date: 2009/10/28

% Input validation
residuals = datacheck(residuals, 'residuals');
residuals = columncheck(residuals, 'residuals');
X = datacheck(X, 'X');

% Check for consistent dimensions
[T, K] = size(X);
if length(residuals) ~= T
    error('Number of observations in residuals and X must be the same');
end

% Create auxiliary regressors matrix
% 1. Start with constant term (column of ones)
X_aux = ones(T, 1);

% 2. Add original regressors
X_aux = [X_aux, X];

% 3. Add squares of each regressor
for i = 1:K
    X_aux = [X_aux, X(:, i).^2];
end

% 4. Add cross-products
for i = 1:K
    for j = (i+1):K
        X_aux = [X_aux, X(:, i) .* X(:, j)];
    end
end

% Get final dimensions of auxiliary regression matrix
[~, num_aux_regressors] = size(X_aux);

% Square the residuals (dependent variable for auxiliary regression)
squared_residuals = residuals.^2;

% Compute White test statistic
stat = compute_white_statistic(squared_residuals, X_aux);

% Degrees of freedom = number of regressors in auxiliary regression (excluding constant)
df = num_aux_regressors - 1;

% Compute p-value
pval = 1 - chi2cdf(stat, df);

% Critical values for significance levels [10%, 5%, 1%]
% These values correspond to the 90th, 95th, and 99th percentiles of the chi-square distribution
significance_levels = [0.10, 0.05, 0.01];
crit = zeros(1, 3);

% Compute approximate critical values based on properties of chi-square distribution
for i = 1:length(significance_levels)
    if significance_levels(i) == 0.10
        crit(i) = df * (1 + 2/(9*df) - 1.645/(3*sqrt(df)))^3;
    elseif significance_levels(i) == 0.05
        crit(i) = df * (1 + 2/(9*df) - 1.96/(3*sqrt(df)))^3;
    elseif significance_levels(i) == 0.01
        crit(i) = df * (1 + 2/(9*df) - 2.576/(3*sqrt(df)))^3;
    end
end

% Rejection indicators
rej = zeros(1, 3);
for i = 1:3
    rej(i) = (stat > crit(i));
end

% Prepare results structure
results = struct('stat', stat, ...
                'pval', pval, ...
                'df', df, ...
                'crit', crit, ...
                'rej', rej);
end

function stat = compute_white_statistic(squared_residuals, X_aux)
% COMPUTE_WHITE_STATISTIC Computes White's test statistic from residuals and regressors.
%
% INPUTS:
%   squared_residuals - Squared residuals from original regression (T x 1)
%   X_aux             - Auxiliary regressors matrix (T x M) including constant
%
% OUTPUTS:
%   stat              - White test statistic (T*R²)

% Sample size
T = size(squared_residuals, 1);

% Perform auxiliary regression to get R²
% Compute OLS estimates: b = (X'X)^(-1)X'y
b_aux = (X_aux'*X_aux)\(X_aux'*squared_residuals);

% Compute fitted values: y_hat = X*b
fitted = X_aux * b_aux;

% Compute residuals: e = y - y_hat
aux_residuals = squared_residuals - fitted;

% Compute total sum of squares (TSS)
mean_squared_residuals = mean(squared_residuals);
TSS = sum((squared_residuals - mean_squared_residuals).^2);

% Compute residual sum of squares (RSS)
RSS = sum(aux_residuals.^2);

% Compute R²
R2 = 1 - (RSS/TSS);

% Compute White test statistic
stat = T * R2;
end