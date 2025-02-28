function parameters = gedfit(data, options)
% GEDFIT Estimates parameters of the Generalized Error Distribution (GED)
%
% USAGE:
%   [PARAMETERS] = gedfit(DATA)
%   [PARAMETERS] = gedfit(DATA, OPTIONS)
%
% INPUTS:
%   DATA    - A vector of data to be fitted
%   OPTIONS - [OPTIONAL] A structure with optional parameters:
%             options.startingvals - A 3x1 vector of starting values for [nu, mu, sigma]
%             options.display - Display option for fmincon {'off','iter','final'}
%             options.algorithm - Algorithm for fmincon optimization
%             options.MaxIter - Maximum iterations for fmincon
%             options.MaxFunEvals - Maximum function evaluations
%             options.TolFun - Function tolerance
%             options.TolX - Parameter tolerance
%
% OUTPUTS:
%   PARAMETERS - A structure with the following fields:
%                parameters.nu - Shape parameter estimate
%                parameters.mu - Location parameter estimate (mean)
%                parameters.sigma - Scale parameter estimate (std-like)
%                parameters.loglik - Log-likelihood at optimum
%                parameters.vcv - Variance-covariance matrix of parameters
%                parameters.stderrors - Standard errors of parameters
%                parameters.exitflag - Optimization exit flag
%                parameters.output - Detailed optimization output
%
% COMMENTS:
%   The GED has the PDF:
%   f(x) = [nu/(2*sigma*gamma(1/nu))] * exp(-(1/2)*|[(x-mu)/sigma]|^nu)
%
%   where:
%   - nu > 0 is the shape parameter (when nu=2, the distribution is normal)
%   - mu is the location parameter (mean when nu=2)
%   - sigma > 0 is the scale parameter (standard deviation when nu=2)
%
%   This function estimates these parameters using maximum likelihood
%   estimation with robust initialization and bound constraints.
%
% EXAMPLES:
%   % Estimate GED parameters for financial returns data
%   parameters = gedfit(returns);
%
%   % Estimate with custom optimization settings
%   options.display = 'iter';
%   options.MaxIter = 1000;
%   parameters = gedfit(returns, options);
%
% See also GEDLOGLIK, GEDPDF, GEDCDF, GEDINV, GEDRND, GEDSTAT

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Step 1: Validate the input data
data = datacheck(data, 'data');
data = columncheck(data, 'data');

% Step 2: Set default optimization options
if nargin < 2
    options = [];
end

% Default optimization options
MaxFunEvals = 400;
MaxIter = 200;
Display = 'off';
Algorithm = 'interior-point'; % MATLAB Optimization Toolbox v5.0+
TolFun = 1e-8;
TolX = 1e-8;

% Override defaults with user options
if ~isempty(options)
    if isfield(options, 'MaxFunEvals')
        MaxFunEvals = parametercheck(options.MaxFunEvals, 'options.MaxFunEvals');
    end
    if isfield(options, 'MaxIter')
        MaxIter = parametercheck(options.MaxIter, 'options.MaxIter');
    end
    if isfield(options, 'Display')
        validDisplayOptions = {'off', 'iter', 'final'};
        if ~ismember(options.Display, validDisplayOptions)
            error('options.Display must be one of: ''off'', ''iter'', or ''final''');
        end
        Display = options.Display;
    end
    if isfield(options, 'Algorithm')
        Algorithm = options.Algorithm;
    end
    if isfield(options, 'TolFun')
        TolFun = parametercheck(options.TolFun, 'options.TolFun');
    end
    if isfield(options, 'TolX')
        TolX = parametercheck(options.TolX, 'options.TolX');
    end
end

% Step 3: Calculate initial parameter estimates
muStart = mean(data);
sigmaStart = std(data);
nuStart = 2; % Starting with normal distribution

% Override with user-provided starting values
if ~isempty(options) && isfield(options, 'startingvals')
    startingvals = parametercheck(options.startingvals, 'options.startingvals');
    if length(startingvals) ~= 3
        error('options.startingvals must be a 3-element vector [nu, mu, sigma]');
    end
    nuStart = startingvals(1);
    muStart = startingvals(2);
    sigmaStart = startingvals(3);
end

% Initial parameter vector
startingvals = [nuStart, muStart, sigmaStart];

% Step 4: Define parameter bounds
% nu > 0, mu unconstrained, sigma > 0
lowerBounds = [1e-4, -inf, 1e-6];
upperBounds = [inf, inf, inf];

% Configure optimization options
optOptions = optimset('fmincon');
optOptions = optimset(optOptions, 'MaxFunEvals', MaxFunEvals, ...
                       'MaxIter', MaxIter, ...
                       'Display', Display, ...
                       'Algorithm', Algorithm, ...
                       'TolFun', TolFun, ...
                       'TolX', TolX);

% Step 5: Define objective function (negative log-likelihood)
objFun = @(params) -gedloglik(data, params(1), params(2), params(3));

% Step 6: Execute constrained optimization
[parameters_vec, fval, exitflag, output] = fmincon(objFun, startingvals, ...
    [], [], [], [], lowerBounds, upperBounds, [], optOptions);

% Step 7: Extract parameter estimates
nu = parameters_vec(1);
mu = parameters_vec(2);
sigma = parameters_vec(3);

% Step 8: Check convergence
if exitflag <= 0
    warning('MFE:Optimization', ['Optimization failed to converge. ', ...
        'Exit flag = %d. Try different starting values or adjust options.'], exitflag);
end

% Step 9: Compute standard errors using finite difference approximation of Hessian
vcv = [];
stderrors = [];
try
    % Compute numerical approximation of Hessian
    h = 1e-4 * max(abs(parameters_vec), 1);
    hess = zeros(3, 3);
    
    % Compute diagonal elements of Hessian
    for i = 1:3
        x_plus = parameters_vec;
        x_plus(i) = x_plus(i) + h(i);
        f_plus = objFun(x_plus);
        
        x_minus = parameters_vec;
        x_minus(i) = x_minus(i) - h(i);
        f_minus = objFun(x_minus);
        
        hess(i,i) = (f_plus - 2*fval + f_minus) / (h(i)^2);
    end
    
    % Compute off-diagonal elements
    for i = 1:2
        for j = (i+1):3
            x_pp = parameters_vec;
            x_pp(i) = x_pp(i) + h(i);
            x_pp(j) = x_pp(j) + h(j);
            f_pp = objFun(x_pp);
            
            x_mm = parameters_vec;
            x_mm(i) = x_mm(i) - h(i);
            x_mm(j) = x_mm(j) - h(j);
            f_mm = objFun(x_mm);
            
            x_pm = parameters_vec;
            x_pm(i) = x_pm(i) + h(i);
            x_pm(j) = x_pm(j) - h(j);
            f_pm = objFun(x_pm);
            
            x_mp = parameters_vec;
            x_mp(i) = x_mp(i) - h(i);
            x_mp(j) = x_mp(j) + h(j);
            f_mp = objFun(x_mp);
            
            hess(i,j) = (f_pp + f_mm - f_pm - f_mp) / (4*h(i)*h(j));
            hess(j,i) = hess(i,j);  % Hessian symmetry
        end
    end
    
    % Invert Hessian to get covariance matrix
    vcv = inv(hess);
    stderrors = sqrt(diag(vcv));
catch
    warning('MFE:StdError', 'Standard error calculation failed. Setting to NaN.');
    vcv = nan(3, 3);
    stderrors = nan(3, 1);
end

% Step 10: Compute final log-likelihood
loglik = gedloglik(data, nu, mu, sigma);

% Step 11: Return results
parameters = struct();
parameters.nu = nu;
parameters.mu = mu;
parameters.sigma = sigma;
parameters.loglik = loglik;
parameters.vcv = vcv;
parameters.stderrors = stderrors;
parameters.exitflag = exitflag;
parameters.output = output;
end