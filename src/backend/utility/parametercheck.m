function parameter = parametercheck(parameter, parameterName, options)
% PARAMETERCHECK Validates numerical parameters for econometric and statistical functions
%
% This function checks if a parameter meets the following requirements:
%   1. Not empty
%   2. Numeric
%   3. Scalar (optional)
%   4. Free of NaN values
%   5. Free of infinite values
%   6. Within specified bounds (optional)
%   7. Contains only integer values (optional)
%   8. Contains only non-negative values (optional)
%   9. Contains only positive values (optional)
%
% USAGE:
%   parameter = parametercheck(parameter, parameterName)
%   parameter = parametercheck(parameter, parameterName, options)
%
% INPUTS:
%   parameter     - Parameter to validate
%   parameterName - Name of parameter for error message references
%   options       - Structure containing validation options:
%                   options.isscalar      - If true, parameter must be scalar [default: false]
%                   options.lowerBound    - If specified, parameter must be >= lowerBound
%                   options.upperBound    - If specified, parameter must be <= upperBound
%                   options.isInteger     - If true, parameter must contain only integer values [default: false]
%                   options.isNonNegative - If true, parameter must contain only non-negative values [default: false]
%                   options.isPositive    - If true, parameter must contain only positive values [default: false]
%
% OUTPUTS:
%   parameter     - Validated parameter (unchanged if validation passes)
%
% COMMENTS:
%   This is a core utility function for the MFE Toolbox that ensures computational
%   stability by validating parameters before they're used in statistical calculations.

% Check if parameter is empty
if isempty(parameter)
    error('The parameter ''%s'' cannot be empty.', parameterName);
end

% Verify parameter is numeric
if ~isnumeric(parameter)
    error('The parameter ''%s'' must be numeric.', parameterName);
end

% Check if parameter should be scalar
if nargin > 2 && isfield(options, 'isscalar') && options.isscalar
    if ~isscalar(parameter)
        error('The parameter ''%s'' must be a scalar.', parameterName);
    end
end

% Check for NaN values
if any(isnan(parameter(:)))
    error('The parameter ''%s'' cannot contain NaN values.', parameterName);
end

% Check for infinite values
if ~all(isfinite(parameter(:)))
    error('The parameter ''%s'' cannot contain infinite values.', parameterName);
end

% Check lower bound if specified
if nargin > 2 && isfield(options, 'lowerBound')
    if any(parameter(:) < options.lowerBound)
        error('The parameter ''%s'' must be greater than or equal to %g.', ...
            parameterName, options.lowerBound);
    end
end

% Check upper bound if specified
if nargin > 2 && isfield(options, 'upperBound')
    if any(parameter(:) > options.upperBound)
        error('The parameter ''%s'' must be less than or equal to %g.', ...
            parameterName, options.upperBound);
    end
end

% Check if parameter should contain only integer values
if nargin > 2 && isfield(options, 'isInteger') && options.isInteger
    if any(parameter(:) ~= floor(parameter(:)))
        error('The parameter ''%s'' must contain only integer values.', parameterName);
    end
end

% Check if parameter should contain only non-negative values
if nargin > 2 && isfield(options, 'isNonNegative') && options.isNonNegative
    if any(parameter(:) < 0)
        error('The parameter ''%s'' must contain only non-negative values.', parameterName);
    end
end

% Check if parameter should contain only positive values
if nargin > 2 && isfield(options, 'isPositive') && options.isPositive
    if any(parameter(:) <= 0)
        error('The parameter ''%s'' must contain only positive values.', parameterName);
    end
end

% Return the validated parameter
end