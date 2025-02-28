function ht = garchcore(parameters, data, options)
% GARCHCORE Core computation engine for GARCH family variance models
%
% USAGE:
%   [HT] = garchcore(PARAMETERS, DATA, OPTIONS)
%
% INPUTS:
%   PARAMETERS - Vector of parameters governing the variance process
%                Format depends on model type (see OPTIONS.model)
%   DATA       - Zero mean residuals, T by 1 column vector
%   OPTIONS    - Structure with model specification fields:
%                model - String specifying the model type: 'GARCH' (default),
%                        'EGARCH', 'GJR', 'TARCH', 'AGARCH', 'NAGARCH', 'IGARCH'
%                p     - Positive integer for the GARCH order (default = 1)
%                q     - Positive integer for the ARCH order (default = 1)
%                useMEX - Boolean, whether to use MEX implementations if available
%                         (default = true)
%                backcast - Scalar value for initial variance or structure for backcast options
%                          (see backcast.m for details)
%
% OUTPUTS:
%   HT - Conditional variances, T by 1 vector
%
% COMMENTS:
%   This is the core computational engine for estimating conditional variances across
%   various GARCH model specifications. It provides a unified interface for different
%   GARCH-family models, with specialized handling for each model variant.
%
%   The variance (h_t) evolution equations for each model are:
%
%   GARCH(p,q):    h_t = omega + sum(alpha(i)*e_{t-i}^2) + sum(beta(j)*h_{t-j})
%
%   EGARCH(p,q):   log(h_t) = omega + sum(alpha(i)*(|e_{t-i}/sqrt(h_{t-i})| - E[|e_t/sqrt(h_t)|]) 
%                             + sum(gamma(i)*e_{t-i}/sqrt(h_{t-i})) + sum(beta(j)*log(h_{t-j}))
%
%   GJR/TARCH(p,q): h_t = omega + sum(alpha(i)*e_{t-i}^2) 
%                           + sum(gamma(i)*e_{t-i}^2*I[e_{t-i}<0]) + sum(beta(j)*h_{t-j})
%
%   AGARCH(p,q):   h_t = omega + sum(alpha(i)*(e_{t-i} - gamma)^2) + sum(beta(j)*h_{t-j})
%
%   NAGARCH(p,q):  h_t = omega + sum(alpha(i)*h_{t-i}*(e_{t-i}/sqrt(h_{t-i}) - gamma)^2)
%                           + sum(beta(j)*h_{t-j})
%
%   IGARCH(p,q):   h_t = omega + sum(alpha(i)*e_{t-i}^2) + sum(beta(j)*h_{t-j})
%                   with constraint: sum(alpha) + sum(beta) = 1
%
% EXAMPLES:
%   % Standard GARCH(1,1) model
%   ht = garchcore([0.01; 0.1; 0.8], residuals, struct('model','GARCH','p',1,'q',1));
%
%   % GJR-GARCH(1,1) model with asymmetry
%   ht = garchcore([0.01; 0.05; 0.10; 0.8], residuals, struct('model','GJR','p',1,'q',1));
%
%   % EGARCH(1,1) with MEX optimization disabled
%   options = struct('model','EGARCH','p',1,'q',1,'useMEX',false);
%   ht = garchcore([0.01; 0.1; 0.05; 0.8], residuals, options);
%
% See also GARCH, EGARCH, GJR, TARCH, AGARCH, IGARCH, NAGARCH, BACKCAST

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Minimum variance level for numerical stability
MIN_VARIANCE = 1e-12;

%% Input validation
% Check parameters
parametercheck(parameters, 'parameters');

% Check data
data = datacheck(data, 'data');

% Ensure data is a column vector
data = columncheck(data, 'data');

%% Process options
if nargin < 3 || isempty(options)
    options = struct();
end

% Set default model type if not specified
if ~isfield(options, 'model') || isempty(options.model)
    options.model = 'GARCH';
else
    % Convert model type to uppercase
    options.model = upper(options.model);
end

% Set default GARCH order (p) if not specified
if ~isfield(options, 'p') || isempty(options.p)
    options.p = 1;
else
    % Validate p (must be positive integer)
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.p, 'options.p', opts);
end

% Set default ARCH order (q) if not specified
if ~isfield(options, 'q') || isempty(options.q)
    options.q = 1;
else
    % Validate q (must be positive integer)
    opts = struct('isscalar', true, 'isInteger', true, 'isPositive', true);
    parametercheck(options.q, 'options.q', opts);
end

% Set default for MEX usage if not specified
if ~isfield(options, 'useMEX') || isempty(options.useMEX)
    options.useMEX = true;
end

%% Determine model type and perform computation
modelType = options.model;
p = options.p;
q = options.q;
T = length(data);

% Initialize variance series
ht = zeros(T, 1);

%% Calculate backcast value if not provided
if ~isfield(options, 'backcast') || isempty(options.backcast)
    % Use default backcast method
    backcastValue = backcast(data);
elseif isstruct(options.backcast)
    % Use backcast with provided options
    backcastValue = backcast(data, options.backcast);
elseif isscalar(options.backcast) && options.backcast > 0
    % Use provided scalar value directly
    backcastValue = options.backcast;
else
    error('OPTIONS.backcast must be a positive scalar or a structure of backcast options.');
end

%% Compute conditional variance based on model type
% Check if MEX implementation should be used
useMex = use_mex_implementation(modelType, options);

switch modelType
    case 'GARCH'
        % Standard GARCH model
        ht = compute_garch_variance(data, parameters, backcastValue, p, q);
        
    case {'GJR', 'TARCH'}
        % Threshold ARCH / GJR-GARCH model
        if useMex && exist('tarch_core', 'file') == 3  % 3 = MEX file exists
            % Use MEX implementation if available
            ht = tarch_core(data, parameters, backcastValue, p, q, T);
        else
            % Use MATLAB implementation
            ht = compute_tarch_variance(data, parameters, backcastValue, p, q);
        end
        
    case 'EGARCH'
        % Exponential GARCH model
        if useMex && exist('egarch_core', 'file') == 3  % 3 = MEX file exists
            % Use MEX implementation if available
            ht = egarch_core(data, parameters, backcastValue, p, q, T);
        else
            % Use MATLAB implementation
            ht = compute_egarch_variance(data, parameters, backcastValue, p, q);
        end
        
    case 'AGARCH'
        % Asymmetric GARCH model
        if useMex && exist('agarch_core', 'file') == 3  % 3 = MEX file exists
            % Use MEX implementation if available
            ht = agarch_core(data, parameters, backcastValue, p, q, T);
        else
            % Use MATLAB implementation
            ht = compute_agarch_variance(data, parameters, backcastValue, p, q);
        end
        
    case 'IGARCH'
        % Integrated GARCH model
        if useMex && exist('igarch_core', 'file') == 3  % 3 = MEX file exists
            % Use MEX implementation if available
            ht = igarch_core(data, parameters, backcastValue, p, q, T);
        else
            % Use MATLAB implementation
            ht = compute_igarch_variance(data, parameters, backcastValue, p, q);
        end
        
    case 'NAGARCH'
        % Nonlinear Asymmetric GARCH model
        ht = compute_nagarch_variance(data, parameters, backcastValue, p, q);
        
    otherwise
        error('Unknown model type: %s. Supported models are GARCH, EGARCH, GJR, TARCH, AGARCH, IGARCH, and NAGARCH.', modelType);
end

%% Ensure numerical stability by applying minimum variance threshold
ht = max(ht, MIN_VARIANCE);

end

%% Helper functions for computing variances for each model type

function ht = compute_garch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for standard GARCH(p,q) model

% Get data length
T = length(data);

% Extract parameters
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

beta = parameters(offset + 1:offset + p);

% Pre-allocate variance vector
ht = zeros(T, 1);

% Initialize pre-sample variances with backcast
maxPQ = max(p, q);
ePadded = zeros(maxPQ, 1);
hPadded = ones(maxPQ, 1) * backcast;

% Squared returns for ARCH component
e2 = data.^2;

% Main GARCH recursion
for t = 1:T
    % ARCH component (weighted sum of past squared residuals)
    archComponent = 0;
    for i = 1:q
        if t > i
            archComponent = archComponent + alpha(i) * e2(t-i);
        else
            archComponent = archComponent + alpha(i) * ePadded(i-t+1);
        end
    end
    
    % GARCH component (weighted sum of past conditional variances)
    garchComponent = 0;
    for j = 1:p
        if t > j
            garchComponent = garchComponent + beta(j) * ht(t-j);
        else
            garchComponent = garchComponent + beta(j) * hPadded(j-t+1);
        end
    end
    
    % Compute conditional variance
    ht(t) = omega + archComponent + garchComponent;
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function ht = compute_egarch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for EGARCH model with asymmetric effects

% Get data length
T = length(data);

% Extract parameters
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

gamma = parameters(offset + 1:offset + q);
offset = offset + q;

beta = parameters(offset + 1:offset + p);

% Pre-allocate variance vector and log-variance
ht = zeros(T, 1);
loght = zeros(T, 1);

% Initialize pre-sample values
maxPQ = max(p, q);
stdResidPadded = zeros(maxPQ, 1);
absStdResidPadded = zeros(maxPQ, 1);
loghPadded = ones(maxPQ, 1) * log(backcast);

% Expected value of |z_t| for standard normal = sqrt(2/pi)
abseExpected = sqrt(2/pi);

% Main EGARCH recursion (on log variance)
for t = 1:T
    % Initialize log variance with omega
    logVariance = omega;
    
    % ARCH components (standardized innovations)
    for i = 1:q
        if t > i
            stdResid = data(t-i) / sqrt(ht(t-i));
            absStdResid = abs(stdResid);
            
            % Add size effect (absolute standardized residual minus expected value)
            logVariance = logVariance + alpha(i) * (absStdResid - abseExpected);
            
            % Add sign effect (standardized residual captures asymmetry)
            logVariance = logVariance + gamma(i) * stdResid;
        else
            % Use pre-sample values
            logVariance = logVariance + alpha(i) * (absStdResidPadded(i-t+1) - abseExpected);
            logVariance = logVariance + gamma(i) * stdResidPadded(i-t+1);
        end
    end
    
    % GARCH components (past log-variances)
    for j = 1:p
        if t > j
            logVariance = logVariance + beta(j) * loght(t-j);
        else
            logVariance = logVariance + beta(j) * loghPadded(j-t+1);
        end
    end
    
    % Store log variance
    loght(t) = logVariance;
    
    % Convert to variance by exponentiating
    ht(t) = exp(logVariance);
    
    % Update standardized residuals for next iteration if needed
    if t <= maxPQ && q > 0
        stdResid = data(t) / sqrt(ht(t));
        stdResidPadded(maxPQ-t+1) = stdResid;
        absStdResidPadded(maxPQ-t+1) = abs(stdResid);
    end
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function ht = compute_tarch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for Threshold ARCH (TARCH/GJR) model

% Get data length
T = length(data);

% Extract parameters
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

gamma = parameters(offset + 1:offset + q);
offset = offset + q;

beta = parameters(offset + 1:offset + p);

% Pre-allocate variance vector
ht = zeros(T, 1);

% Initialize pre-sample variances with backcast
maxPQ = max(p, q);
ePadded = zeros(maxPQ, 1);
hPadded = ones(maxPQ, 1) * backcast;

% Squared returns for ARCH component
e2 = data.^2;

% Generate negative shock indicators I[e_t < 0]
negativeShock = (data < 0);

% Main TARCH recursion
for t = 1:T
    % Initialize with constant term
    ht(t) = omega;
    
    % Add ARCH components (past squared residuals with threshold effect)
    for i = 1:q
        if t > i
            % Basic ARCH effect
            archEffect = alpha(i) * e2(t-i);
            
            % Add threshold effect for negative returns
            if negativeShock(t-i)
                archEffect = archEffect + gamma(i) * e2(t-i);
            end
            
            ht(t) = ht(t) + archEffect;
        else
            % Use backcast values for pre-sample period
            ht(t) = ht(t) + alpha(i) * ePadded(i-t+1);
            % No threshold effect for backcast values as we don't have sign information
        end
    end
    
    % Add GARCH components (past conditional variances)
    for j = 1:p
        if t > j
            ht(t) = ht(t) + beta(j) * ht(t-j);
        else
            ht(t) = ht(t) + beta(j) * hPadded(j-t+1);
        end
    end
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function ht = compute_agarch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for Asymmetric GARCH (AGARCH) model

% Get data length
T = length(data);

% Extract parameters
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

gamma = parameters(offset + 1); % Asymmetry parameter
offset = offset + 1;

beta = parameters(offset + 1:offset + p);

% Pre-allocate variance vector
ht = zeros(T, 1);

% Initialize pre-sample variances with backcast
maxPQ = max(p, q);
ePadded = zeros(maxPQ, 1);
hPadded = ones(maxPQ, 1) * backcast;

% Main AGARCH recursion
for t = 1:T
    % Initialize with constant term
    ht(t) = omega;
    
    % Add ARCH component with asymmetry adjustment
    for i = 1:q
        if t > i
            % Asymmetric news impact: (e_{t-i} - gamma)^2
            asymNews = (data(t-i) - gamma)^2;
            ht(t) = ht(t) + alpha(i) * asymNews;
        else
            % Use backcast for pre-sample period (without asymmetry)
            ht(t) = ht(t) + alpha(i) * ePadded(i-t+1);
        end
    end
    
    % Add GARCH component
    for j = 1:p
        if t > j
            ht(t) = ht(t) + beta(j) * ht(t-j);
        else
            ht(t) = ht(t) + beta(j) * hPadded(j-t+1);
        end
    end
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function ht = compute_igarch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for Integrated GARCH (IGARCH) model

% Get data length
T = length(data);

% Extract parameters (note: in IGARCH, betas are derived from alphas due to integration constraint)
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

beta = parameters(offset + 1:offset + p);
% Ensure sum(alpha) + sum(beta) = 1 for IGARCH constraint
% This should be enforced during parameter estimation

% Pre-allocate variance vector
ht = zeros(T, 1);

% Initialize pre-sample variances with backcast
maxPQ = max(p, q);
ePadded = zeros(maxPQ, 1);
hPadded = ones(maxPQ, 1) * backcast;

% Squared returns for ARCH component
e2 = data.^2;

% Main IGARCH recursion
for t = 1:T
    % Initialize with constant term
    ht(t) = omega;
    
    % Add ARCH component (past squared residuals)
    for i = 1:q
        if t > i
            ht(t) = ht(t) + alpha(i) * e2(t-i);
        else
            ht(t) = ht(t) + alpha(i) * ePadded(i-t+1);
        end
    end
    
    % Add GARCH component (past conditional variances)
    for j = 1:p
        if t > j
            ht(t) = ht(t) + beta(j) * ht(t-j);
        else
            ht(t) = ht(t) + beta(j) * hPadded(j-t+1);
        end
    end
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function ht = compute_nagarch_variance(data, parameters, backcast, p, q)
% Computes conditional variances for Nonlinear Asymmetric GARCH (NAGARCH) model

% Get data length
T = length(data);

% Extract parameters
offset = 0;
omega = parameters(offset + 1);
offset = offset + 1;

alpha = parameters(offset + 1:offset + q);
offset = offset + q;

gamma = parameters(offset + 1); % Asymmetry parameter
offset = offset + 1;

beta = parameters(offset + 1:offset + p);

% Pre-allocate variance vector
ht = zeros(T, 1);

% Initialize pre-sample variances with backcast
maxPQ = max(p, q);
ePadded = zeros(maxPQ, 1);
hPadded = ones(maxPQ, 1) * backcast;

% Main NAGARCH recursion
for t = 1:T
    % Initialize with constant term
    ht(t) = omega;
    
    % Add GARCH component first (to use in ARCH component)
    for j = 1:p
        if t > j
            ht(t) = ht(t) + beta(j) * ht(t-j);
        else
            ht(t) = ht(t) + beta(j) * hPadded(j-t+1);
        end
    end
    
    % Add ARCH component with nonlinear asymmetry adjustment
    for i = 1:q
        if t > i
            % Standardized residual with asymmetry term
            stdResid = data(t-i) / sqrt(ht(t-i));
            nonlinAsym = (stdResid - gamma)^2;
            ht(t) = ht(t) + alpha(i) * ht(t-i) * nonlinAsym;
        else
            % Use backcast for pre-sample period (without asymmetry)
            ht(t) = ht(t) + alpha(i) * ePadded(i-t+1);
        end
    end
end

% Ensure minimum variance
ht = max(ht, MIN_VARIANCE);

end

function useMex = use_mex_implementation(modelType, options)
% Determines if MEX implementation should be used for the specified model

% Check if MEX use is explicitly disabled
if ~options.useMEX
    useMex = false;
    return;
end

% Check if model type has MEX implementation
switch upper(modelType)
    case {'GJR', 'TARCH'}
        useMex = (exist('tarch_core', 'file') == 3); % 3 = MEX file exists
    case 'EGARCH'
        useMex = (exist('egarch_core', 'file') == 3);
    case 'AGARCH'
        useMex = (exist('agarch_core', 'file') == 3);
    case 'IGARCH'
        useMex = (exist('igarch_core', 'file') == 3);
    otherwise
        % No MEX implementation for other models
        useMex = false;
end

end