function results = jarque_bera(data)
% JARQUE_BERA Performs the Jarque-Bera test for normality of a data series.
%
% USAGE:
%   RESULTS = jarque_bera(DATA)
%
% INPUTS:
%   DATA     - A T by 1 vector of data to be tested for normality
%
% OUTPUTS:
%   RESULTS  - A structure with the following fields:
%              statistic - The Jarque-Bera test statistic
%              pval      - The p-value of the test
%              crit_val  - Critical values at [0.1 0.05 0.01]
%              reject    - The decision to reject (1) or not (0) at 
%                          significance levels [0.1 0.05 0.01]
%
% COMMENTS:
%   The Jarque-Bera test examines whether the skewness and kurtosis of 
%   a data series matches that of a normal distribution. Under the null 
%   hypothesis of normality, the test statistic is chi-square distributed 
%   with 2 degrees of freedom.
%
%   The test statistic is calculated as:
%   JB = (T/6) * (S^2 + (K-3)^2/4)
%   where:
%     - T is the sample size
%     - S is the sample skewness
%     - K is the sample kurtosis
%     - The value 3 is the kurtosis of a normal distribution
%
%   Reject the null hypothesis of normality when the JB statistic exceeds
%   the critical value, or equivalently, when the p-value is less than the
%   significance level.
%
% EXAMPLES:
%   returns = randn(1000,1);
%   jb_test = jarque_bera(returns);
%   disp(['JB statistic: ' num2str(jb_test.statistic)]);
%   disp(['p-value: ' num2str(jb_test.pval)]);
%
% See also datacheck, columncheck, chi2cdf

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Validate input using datacheck from utility
data = datacheck(data, 'data');

% Ensure data is a column vector using columncheck from utility
data = columncheck(data, 'data');

% Calculate sample size
T = length(data);

% Standardize data (subtract mean and divide by standard deviation)
mu = mean(data);
sigma = std(data);
standardized_data = (data - mu) / sigma;

% Compute sample skewness: S = (1/T) * sum((data)^3)
S = sum(standardized_data.^3) / T;

% Compute sample kurtosis: K = (1/T) * sum((data)^4)
K = sum(standardized_data.^4) / T;

% Calculate the Jarque-Bera test statistic: JB = T/6 * (S^2 + (K-3)^2/4)
JB = (T/6) * (S^2 + ((K-3)^2)/4);

% Calculate p-value using chi-square distribution with 2 degrees of freedom
% MATLAB Statistics Toolbox function: chi2cdf (compatible with MATLAB R2009b)
pval = 1 - chi2cdf(JB, 2);

% Critical values at significance levels [0.1 0.05 0.01]
% These are chi-square critical values with 2 degrees of freedom
crit_val = [4.605, 5.991, 9.210];

% Determine whether to reject the null hypothesis at each significance level
reject = JB > crit_val;

% Create results structure
results = struct(...
    'statistic', JB, ...      % Test statistic
    'pval', pval, ...         % p-value
    'crit_val', crit_val, ... % Critical values at [0.1 0.05 0.01]
    'reject', reject ...      % Reject decisions at [0.1 0.05 0.01]
);

end