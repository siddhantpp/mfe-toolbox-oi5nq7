function x = skewtinv(p, nu, lambda)
% SKEWTINV Computes the inverse cumulative distribution function (quantile function) 
% of Hansen's skewed t-distribution.
%
% USAGE:
%   X = skewtinv(P, NU, LAMBDA)
%
% INPUTS:
%   P       - Probability values at which to evaluate the inverse CDF, must be in [0,1]
%   NU      - Degrees of freedom parameter, must be > 2
%   LAMBDA  - Skewness parameter, must be in range [-1, 1]
%
% OUTPUTS:
%   X       - Quantile values corresponding to each probability in P
%
% COMMENTS:
%   Hansen's skewed t-distribution extends the Student's t-distribution by
%   incorporating a skewness parameter. This function computes quantiles
%   (inverse CDF values) which are useful for Value-at-Risk calculations and
%   risk assessment in financial applications.
%
%   The implementation follows Hansen (1994) and requires NU > 2 for a
%   well-defined variance. It uses numerical root-finding to invert the CDF.
%
% REFERENCES:
%   Hansen, B. E. (1994). Autoregressive conditional density estimation.
%   International Economic Review, 35(3), 705-730.
%
% See also: skewtcdf, skewtpdf, skewtrnd, stdtinv

% Copyright: Kevin Sheppard
% kevin.sheppard@economics.ox.ac.uk
% Revision: 4.0    Date: 2009/10/28

% Handle special case of empty input
if isempty(p)
    x = p;
    return;
end

% Validate parameters
options.lowerBound = 2;  % NU must be > 2
nu = parametercheck(nu, 'nu', options);

options.lowerBound = -1;
options.upperBound = 1;
lambda = parametercheck(lambda, 'lambda', options);

% Ensure p is in column format
p = columncheck(p, 'p');

% Validate input probabilities
p = datacheck(p, 'p');

% Additional validation for probabilities
if any(p(:) < 0) || any(p(:) > 1)
    error('p must contain values between 0 and 1 inclusive');
end

% Initialize output array
x = zeros(size(p));

% Handle boundary cases
x(p == 0) = -Inf;
x(p == 1) = Inf;

% Calculate constants for the skewed t-distribution
c = gamma((nu+1)/2) / (gamma(nu/2) * sqrt(pi*(nu-2)));
a = 4 * lambda * c * (nu-2) / nu;
b = sqrt(1 + 3*lambda^2 - a^2);

% Process remaining cases
idx = (p > 0) & (p < 1);
if any(idx)
    % Process based on scalar/vector inputs
    if isscalar(nu) && isscalar(lambda)
        % Handle special case: p = 0.5 and lambda = 0 (symmetric case)
        symmetric_midpoint = idx & (abs(lambda) < 1e-8) & (abs(p - 0.5) < 1e-8);
        if any(symmetric_midpoint)
            x(symmetric_midpoint) = 0;
        end
        
        % Calculate threshold probability
        threshold_p = (1-lambda)^2/2;
        
        % Process regular cases requiring numerical inversion
        regular = idx & ~symmetric_midpoint;
        if any(regular)
            for i = 1:sum(regular)
                curr_idx = find(regular, i, 'first');
                curr_idx = curr_idx(end);
                
                % Very close to threshold point
                if abs(p(curr_idx) - threshold_p) < 1e-8
                    x(curr_idx) = -a/b;
                    continue;
                end
                
                % Define objective function for finding the root
                f = @(z) skewtcdf(z, nu, lambda) - p(curr_idx);
                
                % Determine appropriate starting points based on probability region
                if abs(lambda) < 1e-8  % Effectively symmetric case
                    if p(curr_idx) < 0.5
                        x0 = -3 - sqrt(nu/(nu-2));
                    else
                        x0 = 3 + sqrt(nu/(nu-2));
                    end
                else
                    if p(curr_idx) < threshold_p
                        % Left tail region
                        scale = sqrt(nu/(nu-2)) * (1 + abs(lambda));
                        x0 = -scale * (1 + min(20, (threshold_p/max(p(curr_idx), 1e-6))^(1/3)));
                    elseif p(curr_idx) > 0.5
                        % Right tail region
                        scale = sqrt(nu/(nu-2)) * (1 + abs(lambda));
                        x0 = scale * (1 + min(20, ((1-threshold_p)/max(1-p(curr_idx), 1e-6))^(1/3)));
                    else
                        % Central region
                        x0 = -a/b + sign(p(curr_idx) - threshold_p);
                    end
                end
                
                % Use root-finding to invert the CDF
                try
                    [x_val, ~, exitflag] = fzero(f, x0);
                    
                    % If fzero fails to converge, try with different starting points
                    if exitflag ~= 1
                        % Try alternative starting points
                        if p(curr_idx) < 0.5
                            x0 = -5 * sqrt(nu/(nu-2));
                        else
                            x0 = 5 * sqrt(nu/(nu-2));
                        end
                        [x_val, ~, exitflag] = fzero(f, x0);
                        
                        % If still fails, try a bracket approach
                        if exitflag ~= 1
                            if p(curr_idx) < 0.5
                                bracket = [-100, 0];
                            else
                                bracket = [0, 100];
                            end
                            [x_val, ~, ~] = fzero(f, bracket);
                        end
                    end
                    
                    x(curr_idx) = x_val;
                catch
                    % If fzero fails completely, use a more robust but slower approach
                    % Define objective function for minimization (squared difference)
                    fmin = @(z) (skewtcdf(z, nu, lambda) - p(curr_idx))^2;
                    
                    % Determine search bounds based on probability
                    if p(curr_idx) < 0.1
                        bounds = [-100, -0.1];
                    elseif p(curr_idx) > 0.9
                        bounds = [0.1, 100];
                    else
                        bounds = [-20, 20];
                    end
                    
                    % Use bounded minimization
                    x_val = fminbnd(fmin, bounds(1), bounds(2));
                    x(curr_idx) = x_val;
                end
            end
        end
    else
        % Handle case where parameters are non-scalar
        % Process each probability individually
        for i = 1:sum(idx)
            curr_idx = find(idx, i, 'first');
            curr_idx = curr_idx(end);
            
            % Get current parameter values
            if isscalar(nu)
                nu_i = nu;
            else
                nu_i = nu(curr_idx);
            end
            
            if isscalar(lambda)
                lambda_i = lambda;
            else
                lambda_i = lambda(curr_idx);
            end
            
            % Calculate constants for this specific set of parameters
            c_i = gamma((nu_i+1)/2) / (gamma(nu_i/2) * sqrt(pi*(nu_i-2)));
            a_i = 4 * lambda_i * c_i * (nu_i-2) / nu_i;
            b_i = sqrt(1 + 3*lambda_i^2 - a_i^2);
            threshold_p_i = (1-lambda_i)^2/2;
            
            % Special case: symmetric at midpoint
            if abs(lambda_i) < 1e-8 && abs(p(curr_idx) - 0.5) < 1e-8
                x(curr_idx) = 0;
                continue;
            end
            
            % Very close to threshold point
            if abs(p(curr_idx) - threshold_p_i) < 1e-8
                x(curr_idx) = -a_i/b_i;
                continue;
            end
            
            % Define objective function
            f = @(z) skewtcdf(z, nu_i, lambda_i) - p(curr_idx);
            
            % Determine appropriate starting point
            if abs(lambda_i) < 1e-8  % Effectively symmetric case
                if p(curr_idx) < 0.5
                    x0 = -3 - sqrt(nu_i/(nu_i-2));
                else
                    x0 = 3 + sqrt(nu_i/(nu_i-2));
                end
            else
                if p(curr_idx) < threshold_p_i
                    % Left tail region
                    scale = sqrt(nu_i/(nu_i-2)) * (1 + abs(lambda_i));
                    x0 = -scale * (1 + min(20, (threshold_p_i/max(p(curr_idx), 1e-6))^(1/3)));
                elseif p(curr_idx) > 0.5
                    % Right tail region
                    scale = sqrt(nu_i/(nu_i-2)) * (1 + abs(lambda_i));
                    x0 = scale * (1 + min(20, ((1-threshold_p_i)/max(1-p(curr_idx), 1e-6))^(1/3)));
                else
                    % Central region
                    x0 = -a_i/b_i + sign(p(curr_idx) - threshold_p_i);
                end
            end
            
            % Use root-finding
            try
                [x_val, ~, exitflag] = fzero(f, x0);
                
                % If fzero fails to converge, try with different starting points
                if exitflag ~= 1
                    if p(curr_idx) < 0.5
                        x0 = -5 * sqrt(nu_i/(nu_i-2));
                    else
                        x0 = 5 * sqrt(nu_i/(nu_i-2));
                    end
                    [x_val, ~, exitflag] = fzero(f, x0);
                    
                    % If still fails, try a bracket approach
                    if exitflag ~= 1
                        if p(curr_idx) < 0.5
                            bracket = [-100, 0];
                        else
                            bracket = [0, 100];
                        end
                        [x_val, ~, ~] = fzero(f, bracket);
                    end
                end
                
                x(curr_idx) = x_val;
            catch
                % If fzero fails completely, use a more robust approach
                % Define objective function for minimization
                fmin = @(z) (skewtcdf(z, nu_i, lambda_i) - p(curr_idx))^2;
                
                % Determine search bounds
                if p(curr_idx) < 0.1
                    bounds = [-100, -0.1];
                elseif p(curr_idx) > 0.9
                    bounds = [0.1, 100];
                else
                    bounds = [-20, 20];
                end
                
                % Use bounded minimization
                x_val = fminbnd(fmin, bounds(1), bounds(2));
                x(curr_idx) = x_val;
            end
        end
    end
end

end