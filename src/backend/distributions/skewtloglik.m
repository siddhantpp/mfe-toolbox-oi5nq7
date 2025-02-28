function [nlogL, logL] = skewtloglik(data, parameters)
% SKEWTLOGLIK Computes the log-likelihood of Hansen's skewed t-distribution
%
% USAGE:
%   [NLOGL, LOGL] = skewtloglik(DATA, PARAMETERS)
%
% INPUTS:
%   DATA       - Data vector (T by 1)
%   PARAMETERS - Vector of distribution parameters:
%                [nu, lambda, mu, sigma]
%                nu     - Degrees of freedom, nu > 2
%                lambda - Skewness parameter, -1 <= lambda <= 1
%                mu     - Location parameter
%                sigma  - Scale parameter, sigma > 0
%
% OUTPUTS:
%   NLOGL - Negative log-likelihood value (scalar)
%   LOGL  - Optional vector of log-likelihoods for each observation (T by 1)
%
% COMMENTS:
%   Used for maximum likelihood estimation of Hansen's skewed t-distribution parameters.
%   Returns the negative log-likelihood for compatibility with optimization routines
%   that minimize rather than maximize.
%
%   For reference, see Hansen's 1994 paper:
%   "Autoregressive Conditional Density Estimation" International Economic Review
%
% REFERENCES:
%   Hansen, B. E. (1994). "Autoregressive Conditional Density Estimation".
%   International Economic Review, 35(3), 705-730.
%
% See also SKEWTPDF, STDTLOGLIK

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 1    Date: 28-Oct-2009

% Handle the case of empty input
if isempty(data)
    nlogL = [];
    if nargout > 1
        logL = [];
    end
    return;
end

% Validate input data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Check parameter vector length
if numel(parameters) ~= 4
    error('PARAMETERS must be a 4-element vector containing [nu, lambda, mu, sigma]');
end

% Extract parameters
nu = parameters(1);
lambda = parameters(2);
mu = parameters(3);
sigma = parameters(4);

% Validate degrees of freedom (nu > 2)
options.lowerBound = 2;
nu = parametercheck(nu, 'nu', options);

% Validate skewness parameter (lambda in [-1,1])
options = struct();
options.lowerBound = -1;
options.upperBound = 1;
lambda = parametercheck(lambda, 'lambda', options);

% No specific constraints on location parameter mu
mu = parametercheck(mu, 'mu');

% Validate scale parameter (sigma > 0)
options = struct();
options.isPositive = true;
sigma = parametercheck(sigma, 'sigma', options);

% Standardize the data
standardized_data = (data - mu) / sigma;

% Compute PDF values
pdf_values = skewtpdf(standardized_data, nu, lambda);

% Compute log-likelihood values
% Add adjustment for the scale transformation: -log(sigma)
logL = log(pdf_values) - log(sigma);

% For numerical stability, replace any -Inf with a large negative number
logL(logL == -Inf) = -1e6;

% Compute negative log-likelihood (sum of negative individual contributions)
nlogL = -sum(logL);

% If no second output requested, don't return logL
if nargout <= 1
    clear logL;
end

end