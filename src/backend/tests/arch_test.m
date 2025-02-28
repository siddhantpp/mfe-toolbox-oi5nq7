function results = arch_test(data, lags)
% ARCH_TEST Performs the ARCH (AutoRegressive Conditional Heteroskedasticity) test
%
% USAGE:
%   RESULTS = arch_test(DATA)
%   RESULTS = arch_test(DATA, LAGS)
%
% INPUTS:
%   DATA        - Vector of data to be tested for ARCH effects (typically residuals 
%                 from a time series model)
%   LAGS        - [Optional] Number of lags to include in the test (positive integer).
%                 Default is min(20, T/4) where T is the length of DATA.
%
% OUTPUTS:
%   RESULTS     - Structure with the following fields:
%       .statistic   - ARCH LM test statistic
%       .pval        - P-value of the test statistic
%       .critical    - Critical values at 10%, 5%, and 1% significance
%       .lags        - Number of lags used in the test
%       .H0rejected  - Structure with rejection decisions at 10%, 5%, and 1% levels
%
% COMMENTS:
%   Tests the null hypothesis that there are no ARCH effects against the
%   alternative that the conditional variance can be modeled as an ARCH(lags)
%   process. The test is conducted by regressing squared residuals on lagged
%   squared residuals and computing the TR² statistic, which follows a
%   chi-square distribution with LAGS degrees of freedom under the null hypothesis.
%
%   Rejection of the null hypothesis suggests the presence of time-varying
%   volatility that should be modeled with appropriate ARCH/GARCH models.
%
% EXAMPLES:
%   % Test residuals from a time series model for ARCH effects
%   results = arch_test(residuals);
%
%   % Test with 5 lags
%   results = arch_test(returns, 5);
%
% See also DATACHECK, COLUMNCHECK, PARAMETERCHECK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate input data
data = datacheck(data, 'data');

% Step 2: Ensure data is a column vector
data = columncheck(data, 'data');

% Step 3: Determine the number of lags if not provided
if nargin < 2 || isempty(lags)
    T = length(data);
    lags = min(20, floor(T/4));
end

% Step 4: Validate lags parameter
options.isscalar = true;
options.isInteger = true;
options.isPositive = true;
options.upperBound = length(data) - 10; % Ensure enough observations after lag adjustment
lags = parametercheck(lags, 'lags', options);

% Step 5: Square the data (residuals) to test for volatility clustering
squared_residuals = data.^2;

% Step 6: Compute the ARCH test statistic
stat = compute_arch_statistic(squared_residuals, lags);

% Step 7: Calculate p-value using chi-square distribution
pval = 1 - chi2cdf(stat, lags);

% Step 8: Determine critical values for different significance levels
critical.ten = chi2inv(0.90, lags);
critical.five = chi2inv(0.95, lags);
critical.one = chi2inv(0.99, lags);

% Step 9: Determine rejection decisions at different significance levels
H0rejected.ten = (stat > critical.ten);
H0rejected.five = (stat > critical.five);
H0rejected.one = (stat > critical.one);

% Step 10: Construct and return results structure
results.statistic = stat;
results.pval = pval;
results.critical = critical;
results.lags = lags;
results.H0rejected = H0rejected;
end

function stat = compute_arch_statistic(squared_residuals, lags)
% COMPUTE_ARCH_STATISTIC Internal function to compute the ARCH LM test statistic
%
% INPUTS:
%   squared_residuals - Vector of squared residuals
%   lags              - Number of lags to include in the test
%
% OUTPUTS:
%   stat              - ARCH LM test statistic (TR²)

% Step 1: Compute sample size after accounting for lags
T = length(squared_residuals);
effective_T = T - lags;

% Step 2: Set up regression matrices for LM test
% Dependent variable: squared_residuals(lags+1:T)
y = squared_residuals(lags+1:T);

% Independent variables: constant and lagged squared residuals
X = zeros(effective_T, lags+1);
X(:,1) = 1; % Constant term

% Fill the X matrix with lagged squared residuals
for i = 1:lags
    X(:,i+1) = squared_residuals(lags+1-i:T-i);
end

% Step 3: Compute auxiliary regression 
% Use \ operator for better numerical stability
beta = X \ y;

% Compute fitted values and residuals
fitted = X * beta;
residuals = y - fitted;
SSR = residuals'*residuals;
SST = (y - mean(y))'*(y - mean(y));

% Step 4: Compute R-squared
Rsquared = 1 - SSR/SST;

% Step 5: Compute LM statistic (T*R^2)
stat = effective_T * Rsquared;
end